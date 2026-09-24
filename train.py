"""
train.py - Training Engine for PyTorch CORAL Ordinal Binary Classification.

Features:
- FP16 Mixed Precision training using torch.amp for high throughput and 6GB VRAM safety.
- GroupKFold cross-validation grouped on review text to prevent data leakage.
- Differential learning rates (backbone vs. CORAL classification head).
- Gradient clipping and learning rate warmup scheduler.
- Checkpoint saving based on validation Mean Absolute Error (MAE).
- CLI interface for full pipeline execution.
"""

import argparse
import os
import sys
import time
from typing import Dict, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from dataset import DrugReviewDataset, create_group_kfold_splits, get_dataloaders
from model import CoralLoss, CoralTransformer, logits_to_ratings


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[object],
    loss_fn: nn.Module,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    max_grad_norm: float = 1.0,
    use_amp: bool = True,
) -> Tuple[float, float]:
    """
    Executes one training epoch with FP16 Automatic Mixed Precision (AMP).

    Args:
        model: CoralTransformer model instance.
        dataloader: Training DataLoader.
        optimizer: PyTorch optimizer (AdamW).
        scheduler: Learning rate scheduler.
        loss_fn: CoralLoss instance.
        scaler: GradScaler for AMP.
        device: Target execution device (cuda / cpu / mps).
        max_grad_norm: Maximum gradient norm for clipping.
        use_amp: Whether to enable FP16 mixed precision.

    Returns:
        Tuple of (average_epoch_loss, average_epoch_mae).
    """
    model.train()
    total_loss = 0.0
    total_abs_error = 0.0
    total_samples = 0

    device_type = device.type if device.type in ["cuda", "cpu", "mps"] else "cuda"

    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        true_ratings = batch["rating"].to(device, non_blocking=True)

        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # FP16 Mixed Precision forward pass
        with torch.amp.autocast(device_type=device_type, enabled=use_amp):
            logits = model(input_ids, attention_mask, token_type_ids)
            loss = loss_fn(logits, labels)

        # Scale loss and backward pass
        scaler.scale(loss).backward()

        # Unscale gradients for clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        # Step optimizer and update scaler
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        # Track metrics
        batch_size = input_ids.size(0)
        total_loss += loss.item() * batch_size

        with torch.no_grad():
            pred_ratings = logits_to_ratings(logits, soft=False)
            abs_err = torch.sum(torch.abs(pred_ratings - true_ratings)).item()
            total_abs_error += abs_err
            total_samples += batch_size

    avg_loss = total_loss / max(total_samples, 1)
    avg_mae = total_abs_error / max(total_samples, 1)

    return avg_loss, avg_mae


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    use_amp: bool = True,
) -> Tuple[float, float, float]:
    """
    Evaluates model performance on validation/test dataloader.

    Returns:
        Tuple of (average_loss, MAE, RMSE).
    """
    model.eval()
    total_loss = 0.0
    total_abs_error = 0.0
    total_sq_error = 0.0
    total_samples = 0

    device_type = device.type if device.type in ["cuda", "cpu", "mps"] else "cuda"

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        true_ratings = batch["rating"].to(device, non_blocking=True)

        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device_type, enabled=use_amp):
            logits = model(input_ids, attention_mask, token_type_ids)
            loss = loss_fn(logits, labels)

        batch_size = input_ids.size(0)
        total_loss += loss.item() * batch_size

        pred_ratings = logits_to_ratings(logits, soft=False)
        diff = pred_ratings - true_ratings
        total_abs_error += torch.sum(torch.abs(diff)).item()
        total_sq_error += torch.sum(diff ** 2).item()
        total_samples += batch_size

    avg_loss = total_loss / max(total_samples, 1)
    mae = total_abs_error / max(total_samples, 1)
    rmse = np.sqrt(total_sq_error / max(total_samples, 1))

    return avg_loss, mae, rmse


def get_optimizer_and_scheduler(
    model: CoralTransformer,
    num_training_steps: int,
    backbone_lr: float = 2e-5,
    head_lr: float = 1e-3,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
) -> Tuple[torch.optim.Optimizer, object]:
    """
    Constructs AdamW optimizer with differential learning rates and linear warmup scheduler.
    """
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.backbone.named_parameters()
                if not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": weight_decay,
            "lr": backbone_lr,
        },
        {
            "params": [
                p for n, p in model.backbone.named_parameters()
                if any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": 0.0,
            "lr": backbone_lr,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if not n.startswith("backbone.") and p.requires_grad
            ],
            "weight_decay": weight_decay,
            "lr": head_lr,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

    num_warmup_steps = int(num_training_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    return optimizer, scheduler


def train_coral_pipeline(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
    model_name: str = "distilbert-base-uncased",
    batch_size: int = 16,
    max_length: int = 256,
    epochs: int = 3,
    backbone_lr: float = 2e-5,
    head_lr: float = 1e-3,
    output_dir: str = "./checkpoints",
    use_amp: bool = True,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Executes complete training loop for CORAL classification model.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Initializing training on device: {device} (AMP Enabled: {use_amp and device.type == 'cuda'})")

    train_loader, val_loader = get_dataloaders(
        train_df=train_df,
        val_df=val_df if val_df is not None else None,
        tokenizer_name=model_name,
        batch_size=batch_size,
        max_length=max_length,
    )

    model = CoralTransformer(model_name=model_name, num_classes=10)
    model.to(device)

    loss_fn = CoralLoss(reduction="mean")
    total_steps = len(train_loader) * epochs

    optimizer, scheduler = get_optimizer_and_scheduler(
        model=model,
        num_training_steps=total_steps,
        backbone_lr=backbone_lr,
        head_lr=head_lr,
    )

    device_type = device.type if device.type in ["cuda", "cpu", "mps"] else "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))

    best_val_mae = float("inf")
    best_checkpoint_path = os.path.join(output_dir, "best_coral_model.pt")

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        train_loss, train_mae = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            scaler=scaler,
            device=device,
            use_amp=(use_amp and device.type == "cuda"),
        )
        elapsed = time.time() - start_time

        log_str = f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Train MAE: {train_mae:.4f} | Time: {elapsed:.1f}s"

        if val_loader is not None:
            val_loss, val_mae, val_rmse = evaluate_epoch(
                model=model,
                dataloader=val_loader,
                loss_fn=loss_fn,
                device=device,
                use_amp=(use_amp and device.type == "cuda"),
            )
            log_str += f" | Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.4f} | Val RMSE: {val_rmse:.4f}"

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_mae": val_mae,
                        "model_name": model_name,
                    },
                    best_checkpoint_path,
                )
                log_str += " 🌟 [Saved Best Checkpoint]"

        print(log_str)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if val_loader is None:
        # Save final model if no val split
        torch.save({"model_state_dict": model.state_dict(), "model_name": model_name}, best_checkpoint_path)

    print(f"✅ Training completed. Best Checkpoint saved to {best_checkpoint_path}")
    return {"best_val_mae": best_val_mae if val_loader is not None else train_mae}


def main():
    parser = argparse.ArgumentParser(description="Train PyTorch CORAL Ordinal Classification Model on UCI Drug Reviews")
    parser.add_argument("--train_path", type=str, default="data/processed/cleaned_train.csv", help="Path to training CSV file")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased", help="Transformer checkpoint")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (16 default for 6GB VRAM)")
    parser.add_argument("--max_length", type=int, default=256, help="Max token sequence length (256)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--backbone_lr", type=float, default=2e-5, help="Backbone learning rate")
    parser.add_argument("--head_lr", type=float, default=1e-3, help="CORAL classification head learning rate")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--use_group_kfold", action="store_true", help="Run GroupKFold cross-validation")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of GroupKFold splits")

    args = parser.parse_args()

    print(f"📖 Reading dataset from {args.train_path}...")
    df = pd.read_csv(args.train_path)

    if args.use_group_kfold:
        print(f"🔀 Running {args.n_splits}-fold GroupKFold Cross-Validation grouped on 'review'...")
        splits = create_group_kfold_splits(df, n_splits=args.n_splits, group_col="review", rating_col="rating")
        fold_maes = []
        for fold, (train_idx, val_idx) in enumerate(splits, 1):
            print(f"\n--- Fold {fold}/{args.n_splits} ---")
            train_fold = df.iloc[train_idx]
            val_fold = df.iloc[val_idx]
            fold_dir = os.path.join(args.output_dir, f"fold_{fold}")
            metrics = train_coral_pipeline(
                train_df=train_fold,
                val_df=val_fold,
                model_name=args.model_name,
                batch_size=args.batch_size,
                max_length=args.max_length,
                epochs=args.epochs,
                backbone_lr=args.backbone_lr,
                head_lr=args.head_lr,
                output_dir=fold_dir,
            )
            fold_maes.append(metrics["best_val_mae"])
        print(f"\n🏆 Average CV Val MAE across {args.n_splits} folds: {np.mean(fold_maes):.4f}")
    else:
        train_coral_pipeline(
            train_df=df,
            model_name=args.model_name,
            batch_size=args.batch_size,
            max_length=args.max_length,
            epochs=args.epochs,
            backbone_lr=args.backbone_lr,
            head_lr=args.head_lr,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
