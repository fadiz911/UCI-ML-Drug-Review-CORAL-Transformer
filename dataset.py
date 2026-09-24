"""
dataset.py - Dataset module for PyTorch CORAL Ordinal Binary Classification.

Handles 9-dimensional ordinal binary target encoding, tokenization, PyTorch Dataset
wrapping, GroupKFold splitting on review text to prevent data leakage, and DataLoader creation.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from sklearn.model_selection import GroupKFold


def encode_ordinal_targets(ratings: Union[torch.Tensor, np.ndarray, List[int], int]) -> torch.Tensor:
    """
    Converts 1-to-10 integer ratings into 9-dimensional binary ordinal targets.
    Formula: y_k = I(rating > k) for k in {1, 2, ..., 9}.

    Examples:
        Rating 1  -> [0, 0, 0, 0, 0, 0, 0, 0, 0]
        Rating 2  -> [1, 0, 0, 0, 0, 0, 0, 0, 0]
        Rating 5  -> [1, 1, 1, 1, 0, 0, 0, 0, 0]
        Rating 10 -> [1, 1, 1, 1, 1, 1, 1, 1, 1]

    Args:
        ratings: Scaled or unscaled integer ratings in [1, 10].

    Returns:
        torch.Tensor: FloatTensor of shape (N, 9) or (9,) containing binary targets.
    """
    if isinstance(ratings, int):
        ratings_tensor = torch.tensor([ratings], dtype=torch.long)
        single_val = True
    elif isinstance(ratings, list):
        ratings_tensor = torch.tensor(ratings, dtype=torch.long)
        single_val = False
    elif isinstance(ratings, np.ndarray):
        ratings_tensor = torch.from_numpy(ratings).long()
        single_val = False
    elif isinstance(ratings, torch.Tensor):
        ratings_tensor = ratings.long()
        single_val = (ratings_tensor.dim() == 0)
        if single_val:
            ratings_tensor = ratings_tensor.unsqueeze(0)
    else:
        raise TypeError(f"Unsupported ratings type: {type(ratings)}")

    ratings_tensor = ratings_tensor.view(-1, 1)  # Shape: (N, 1)
    thresholds = torch.arange(1, 10, dtype=torch.long)  # Shape: (9,)

    # Vectorized comparison y_k = (rating > k)
    binary_targets = (ratings_tensor > thresholds).float()  # Shape: (N, 9)

    if single_val:
        return binary_targets.squeeze(0)  # Shape: (9,)
    return binary_targets


class DrugReviewDataset(Dataset):
    """
    PyTorch Dataset for Clinical NLP Ordinal Binary Classification on Drug Reviews.
    Strictly enforces max_length=256 for 6GB VRAM GPU memory optimization.
    """

    def __init__(
        self,
        texts: List[str],
        ratings: Optional[List[int]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        tokenizer_name: str = "distilbert-base-uncased",
        max_length: int = 256,
    ):
        """
        Args:
            texts: List of review text strings.
            ratings: Optional list of integer ratings (1 to 10).
            tokenizer: Pre-initialized HuggingFace tokenizer.
            tokenizer_name: Name of HF tokenizer to load if tokenizer is None.
            max_length: Maximum token sequence length (strictly 256).
        """
        self.texts = [str(t) if pd.notna(t) else "" for t in texts]
        self.ratings = ratings
        self.max_length = max_length

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        if self.ratings is not None:
            if len(self.texts) != len(self.ratings):
                raise ValueError("Length of texts and ratings must match.")
            self.encoded_targets = encode_ordinal_targets(self.ratings)
        else:
            self.encoded_targets = None

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),        # Shape: (256,)
            "attention_mask": encoding["attention_mask"].squeeze(0),# Shape: (256,)
        }

        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)

        if self.encoded_targets is not None:
            item["labels"] = self.encoded_targets[idx]            # Shape: (9,)
            item["rating"] = torch.tensor(self.ratings[idx], dtype=torch.long) # Scalar rating

        return item


def create_group_kfold_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    group_col: str = "review",
    rating_col: str = "rating",
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generates GroupKFold split indices grouped on review text to prevent data leakage.

    Args:
        df: Input DataFrame containing review text and ratings.
        n_splits: Number of cross-validation folds.
        group_col: Column name to group by (e.g. 'review').
        rating_col: Column name containing target ratings.

    Returns:
        List of (train_indices, val_indices) numpy array tuples.
    """
    gkf = GroupKFold(n_splits=n_splits)
    groups = df[group_col].astype(str).values
    X = df.index.values
    y = df[rating_col].values

    splits = []
    for train_idx, val_idx in gkf.split(X, y, groups=groups):
        splits.append((train_idx, val_idx))

    return splits


def get_dataloaders(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
    tokenizer_name: str = "distilbert-base-uncased",
    batch_size: int = 16,
    max_length: int = 256,
    num_workers: int = 0,
    text_col: str = "review",
    rating_col: str = "rating",
) -> Union[DataLoader, Tuple[DataLoader, DataLoader]]:
    """
    DataLoader factory for training and validation sets.

    Args:
        train_df: Training DataFrame.
        val_df: Optional validation DataFrame.
        tokenizer_name: Pretrained transformer tokenizer name.
        batch_size: Batch size (default 16 for 6GB VRAM safety).
        max_length: Token length cutoff (default 256).
        num_workers: PyTorch DataLoader workers.
        text_col: Column containing review text.
        rating_col: Column containing rating targets.

    Returns:
        DataLoader or Tuple of (train_loader, val_loader).
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    train_dataset = DrugReviewDataset(
        texts=train_df[text_col].tolist(),
        ratings=train_df[rating_col].tolist() if rating_col in train_df else None,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    if val_df is None:
        return train_loader, None

    val_dataset = DrugReviewDataset(
        texts=val_df[text_col].tolist(),
        ratings=val_df[rating_col].tolist() if rating_col in val_df else None,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader
