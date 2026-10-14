"""Hadamard Product Operations for PARROT Fusion"""

import torch
import torch.nn as nn


class HadamardFusion(nn.Module):
    """
    Hadamard Product-based fusion of two feature streams.

    Performs element-wise multiplication to fuse complementary information
    from Mamba and Attention-based models.
    """

    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale

    def forward(self, mamba_features: torch.Tensor, attention_features: torch.Tensor) -> torch.Tensor:
        """
        Fuse features using Hadamard product (element-wise multiplication).

        Args:
            mamba_features: Features from Mamba model [batch, feature_dim]
            attention_features: Features from Attention model [batch, feature_dim]

        Returns:
            Fused features [batch, feature_dim]
        """
        # Element-wise multiplication (Hadamard product)
        fused = mamba_features * attention_features * self.scale

        return fused


def weighted_hadamard_fusion(
    features_list: list,
    weights: torch.Tensor
) -> torch.Tensor:
    """
    Weighted Hadamard product fusion of multiple features.

    Args:
        features_list: List of feature tensors
        weights: Fusion weights [num_features]

    Returns:
        Fused features
    """
    # Initialize with first feature
    fused = features_list[0] * weights[0]

    # Hadamard product with remaining features
    for i in range(1, len(features_list)):
        fused = fused * (features_list[i] * weights[i])

    return fused
