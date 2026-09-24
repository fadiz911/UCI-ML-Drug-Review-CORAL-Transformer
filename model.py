"""
model.py - Neural Network Architecture & Loss Function for CORAL Ordinal Binary Classification.

Implements the Consistent Rank Logits (CORAL) framework wrapping a HuggingFace Transformer backbone.
Features rank-consistent weight sharing, ordinal bias vector, BCE-with-logits loss, and vectorized rating decoding.
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class CoralTransformer(nn.Module):
    """
    CORAL (Consistent Rank Logits) Transformer Model for Ordinal Binary Classification.

    Uses a shared weight linear projection W and 9 rank-specific bias parameters b_k.
    Logit for rank k is defined as: z_k = W^T h(x) + b_k
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_classes: int = 10,
        dropout_prob: float = 0.2,
    ):
        """
        Args:
            model_name: HuggingFace model checkpoint (e.g. 'distilbert-base-uncased' or clinical BERT).
            num_classes: Total ordinal rating classes (10 for ratings 1-10).
            dropout_prob: Dropout probability before the classification head.
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_tasks = num_classes - 1  # 9 binary tasks for 10 ranks

        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)

        hidden_size = getattr(self.config, "hidden_size", getattr(self.config, "dim", 768))

        self.dropout = nn.Dropout(dropout_prob)
        # Shared weight projection (1D output, no bias term)
        self.shared_linear = nn.Linear(hidden_size, 1, bias=False)

        # 9 rank-specific bias terms
        self.ordinal_biases = nn.Parameter(torch.zeros(self.num_tasks, dtype=torch.float32))

        import math
        # Initialize linear weight and biases
        nn.init.kaiming_uniform_(self.shared_linear.weight, a=math.sqrt(5))
        nn.init.zeros_(self.ordinal_biases)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass computing 9 rank logits per input sequence.

        Args:
            input_ids: Tensor of shape (B, sequence_length)
            attention_mask: Tensor of shape (B, sequence_length)
            token_type_ids: Optional Tensor of shape (B, sequence_length)

        Returns:
            torch.Tensor: Logits tensor of shape (B, 9)
        """
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None and "token_type_ids" in self.backbone.forward.__code__.co_varnames:
            kwargs["token_type_ids"] = token_type_ids

        outputs = self.backbone(**kwargs)

        # Extract sequence representation ([CLS] token or pooler output)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            # Fallback to [CLS] token representation (first token)
            pooled_output = outputs.last_hidden_state[:, 0, :]

        pooled_output = self.dropout(pooled_output)

        # Single 1D feature scalar per batch item: g(x) = W^T h(x), shape (B, 1)
        base_logits = self.shared_linear(pooled_output)

        # Add ordinal bias for each binary task: z_k = g(x) + b_k, shape (B, 9)
        logits = base_logits + self.ordinal_biases

        return logits


class CoralLoss(nn.Module):
    """
    Loss function for CORAL Ordinal Binary Classification.
    Computes Binary Cross Entropy with Logits across all K-1 = 9 task dimensions.
    """

    def __init__(self, reduction: str = "mean"):
        """
        Args:
            reduction: 'mean' or 'sum' reduction for BCE loss over batch and task dimensions.
        """
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Predicted logits tensor of shape (B, 9)
            targets: Binary ordinal ground truth targets of shape (B, 9)

        Returns:
            torch.Tensor: Scalar loss tensor.
        """
        return self.bce_loss(logits, targets)


def logits_to_ratings(logits: torch.Tensor, soft: bool = False) -> torch.Tensor:
    """
    Decodes predicted CORAL rank logits into rating predictions (1 to 10).

    Discrete decoding formula:
        r_hat = 1 + sum_{k=1}^9 I(z_k > 0)

    Soft continuous decoding formula:
        r_hat_soft = 1 + sum_{k=1}^9 sigmoid(z_k)

    Args:
        logits: Logits tensor of shape (B, 9) or (9,)
        soft: If True, returns continuous float ratings; if False, integer ratings in [1, 10].

    Returns:
        torch.Tensor: Predicted ratings tensor of shape (B,) or scalar.
    """
    if soft:
        probs = torch.sigmoid(logits)
        ratings = 1.0 + torch.sum(probs, dim=-1)
    else:
        binary_preds = (logits > 0.0).long()
        ratings = 1 + torch.sum(binary_preds, dim=-1)

    return ratings
