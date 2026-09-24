"""
evaluate.py - Evaluation Engine for PyTorch CORAL Ordinal Binary Classification Model.

Computes comprehensive clinical NLP and ordinal metrics:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Exact Match Accuracy & Off-by-1 Accuracy
- Kendall's Tau (tau) & Spearman Rank Correlation (rho)
- Confusion Matrix & Multiclass Classification Report
"""

import argparse
import os
import sys
import time
from typing import Dict, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import classification_report, confusion_matrix
import torch

from dataset import DrugReviewDataset
from model import CoralTransformer, logits_to_ratings
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate_model_pipeline(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
) -> Dict[str, object]:
    """
    Runs FP16 mixed-precision evaluation on test dataset and computes ordinal metrics.

    Args:
        model: Loaded CoralTransformer model.
        test_loader: DataLoader containing test data.
        device: Device to run evaluation on.
        use_amp: Whether to use FP16 mixed precision during inference.

    Returns:
        Dict containing scalar metrics, raw predictions, confusion matrix, and report text.
    """
    model.eval()
    all_true_ratings = []
    all_pred_discrete = []
    all_pred_soft = []

    device_type = device.type if device.type in ["cuda", "cpu", "mps"] else "cuda"

    start_time = time.time()
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        true_ratings = batch["rating"].numpy()

        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device_type, enabled=(use_amp and device.type == "cuda")):
            logits = model(input_ids, attention_mask, token_type_ids)

        discrete_preds = logits_to_ratings(logits, soft=False).cpu().numpy()
        soft_preds = logits_to_ratings(logits, soft=True).cpu().numpy()

        all_true_ratings.extend(true_ratings)
        all_pred_discrete.extend(discrete_preds)
        all_pred_soft.extend(soft_preds)

    inference_time = time.time() - start_time

    y_true = np.array(all_true_ratings, dtype=int)
    y_pred = np.array(all_pred_discrete, dtype=int)
    y_pred_soft = np.array(all_pred_soft, dtype=float)

    # Compute Metrics
    abs_errors = np.abs(y_true - y_pred)
    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae_soft = float(np.mean(np.abs(y_true - y_pred_soft)))
    rmse_soft = float(np.sqrt(np.mean((y_true - y_pred_soft) ** 2)))

    exact_acc = float(np.mean(y_true == y_pred) * 100.0)
    off_by_1_acc = float(np.mean(abs_errors <= 1) * 100.0)

    tau, _ = kendalltau(y_true, y_pred)
    rho, _ = spearmanr(y_true, y_pred)

    conf_mat = confusion_matrix(y_true, y_pred, labels=list(range(1, 11)))
    report = classification_report(y_true, y_pred, labels=list(range(1, 11)), digits=4, zero_division=0)

    results = {
        "mae": mae,
        "rmse": rmse,
        "mae_soft": mae_soft,
        "rmse_soft": rmse_soft,
        "exact_accuracy": exact_acc,
        "off_by_1_accuracy": off_by_1_acc,
        "kendall_tau": float(tau),
        "spearman_rho": float(rho),
        "inference_time_sec": inference_time,
        "total_samples": len(y_true),
        "confusion_matrix": conf_mat,
        "classification_report": report,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_pred_soft": y_pred_soft,
    }

    return results


def print_evaluation_summary(metrics: Dict[str, object]):
    """Prints formatted evaluation report."""
    print("=" * 65)
    print("      CORAL ORDINAL CLASSIFICATION EVALUATION REPORT")
    print("=" * 65)
    print(f" Total Evaluated Samples:     {metrics['total_samples']}")
    print(f" Inference Time:              {metrics['inference_time_sec']:.2f}s ({metrics['total_samples'] / metrics['inference_time_sec']:.1f} samples/sec)")
    print("-" * 65)
    print(f" Discrete MAE:                {metrics['mae']:.4f}")
    print(f" Discrete RMSE:               {metrics['rmse']:.4f}")
    print(f" Continuous Soft MAE:         {metrics['mae_soft']:.4f}")
    print(f" Continuous Soft RMSE:        {metrics['rmse_soft']:.4f}")
    print("-" * 65)
    print(f" Exact Match Accuracy:        {metrics['exact_accuracy']:.2f}%")
    print(f" Off-by-1 Accuracy (<= 1):    {metrics['off_by_1_accuracy']:.2f}%")
    print("-" * 65)
    print(f" Kendall's Tau (rank correlation):   {metrics['kendall_tau']:.4f}")
    print(f" Spearman's Rho (rank correlation):  {metrics['spearman_rho']:.4f}")
    print("=" * 65)
    print("\nMulticlass Classification Report:\n")
    print(metrics["classification_report"])


def main():
    parser = argparse.ArgumentParser(description="Evaluate PyTorch CORAL Ordinal Classification Model")
    parser.add_argument("--test_path", type=str, default="data/processed/cleaned_test_deduplicated.csv", help="Path to test CSV file")
    parser.add_argument("--checkpoint_path", type=str, default="./checkpoints/best_coral_model.pt", help="Path to model checkpoint")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased", help="Backbone model architecture")
    parser.add_argument("--batch_size", type=int, default=32, help="Inference batch size")
    parser.add_argument("--max_length", type=int, default=256, help="Token max length (256)")
    parser.add_argument("--text_col", type=str, default="review", help="Review text column name")
    parser.add_argument("--rating_col", type=str, default="rating", help="Rating column name")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🔍 Loading model from {args.checkpoint_path} onto {device}...")

    model = CoralTransformer(model_name=args.model_name, num_classes=10)

    if os.path.exists(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print("✅ Checkpoint loaded successfully.")
    else:
        print(f"⚠️ Warning: Checkpoint file '{args.checkpoint_path}' not found. Initializing with raw weights for dry-run evaluation.")

    model.to(device)

    print(f"📖 Reading test dataset from {args.test_path}...")
    test_df = pd.read_csv(args.test_path)

    test_dataset = DrugReviewDataset(
        texts=test_df[args.text_col].tolist(),
        ratings=test_df[args.rating_col].tolist(),
        tokenizer_name=args.model_name,
        max_length=args.max_length,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    metrics = evaluate_model_pipeline(
        model=model,
        test_loader=test_loader,
        device=device,
        use_amp=True,
    )

    print_evaluation_summary(metrics)


if __name__ == "__main__":
    main()
