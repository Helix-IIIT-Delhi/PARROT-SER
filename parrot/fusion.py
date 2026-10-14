"""PARROT Fusion Framework for Speech Emotion Recognition"""

import torch
import torch.nn as nn
from typing import Dict, Tuple
from .transport import optimal_transport_distance
from .hadamard import HadamardFusion


class PARROTFusion(nn.Module):
    """
    PARROT: Parallel-branch fusion of Mamba and Attention-based PTMs
    using Optimal Transport and Hadamard product.

    Args:
        mamba_model: Mamba-based pre-trained model
        attention_model: Attention-based pre-trained model
        num_emotions: Number of emotion classes
        feature_dim: Feature dimension for fusion
        learnable_weights: Whether to learn fusion weights
    """

    def __init__(
        self,
        mamba_model: nn.Module,
        attention_model: nn.Module,
        num_emotions: int = 7,
        feature_dim: int = 512,
        learnable_weights: bool = True,
    ):
        super().__init__()
        self.mamba_model = mamba_model
        self.attention_model = attention_model
        self.num_emotions = num_emotions
        self.feature_dim = feature_dim

        # Feature projections
        self.mamba_proj = nn.Linear(
            mamba_model.output_dim if hasattr(mamba_model, 'output_dim') else 512,
            feature_dim
        )
        self.attention_proj = nn.Linear(
            attention_model.output_dim if hasattr(attention_model, 'output_dim') else 768,
            feature_dim
        )

        # Hadamard fusion
        self.hadamard_fusion = HadamardFusion(scale=1.0)

        # Learnable fusion weights
        if learnable_weights:
            self.fusion_weight_mamba = nn.Parameter(torch.ones(1) * 0.5)
            self.fusion_weight_attention = nn.Parameter(torch.ones(1) * 0.5)
        else:
            self.register_buffer("fusion_weight_mamba", torch.ones(1) * 0.5)
            self.register_buffer("fusion_weight_attention", torch.ones(1) * 0.5)

        # Emotion classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_emotions),
        )

    def forward(self, audio_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with PARROT parallel-branch fusion.

        Args:
            audio_features: Input audio features

        Returns:
            Dict with:
                - emotion_logits: Emotion classification logits
                - emotion_probs: Emotion probability distribution
                - fused_features: Fused representation
                - transport_distance: Optimal Transport distance
        """
        # Extract features from both models (parallel branches)
        mamba_out = self.mamba_model(audio_features)
        attention_out = self.attention_model(audio_features)

        # Project to common space
        mamba_feat = self.mamba_proj(mamba_out)
        attention_feat = self.attention_proj(attention_out)

        # Compute Optimal Transport distance for adaptive weighting
        ot_distance = optimal_transport_distance(mamba_feat, attention_feat, method="wasserstein")

        # Adaptive weights based on OT distance
        # Higher divergence → higher weight
        ot_weight = torch.sigmoid(ot_distance.unsqueeze(-1))

        # Normalize weights
        w_mamba = torch.sigmoid(self.fusion_weight_mamba) * (1 + ot_weight)
        w_attention = torch.sigmoid(self.fusion_weight_attention) * (1 + (1 - ot_weight))

        # Hadamard product fusion
        fused_mamba = w_mamba * mamba_feat
        fused_attention = w_attention * attention_feat
        fused_features = self.hadamard_fusion(fused_mamba, fused_attention)

        # Emotion classification
        emotion_logits = self.classifier(fused_features)
        emotion_probs = torch.softmax(emotion_logits, dim=1)

        return {
            "emotion_logits": emotion_logits,
            "emotion_probs": emotion_probs,
            "fused_features": fused_features,
            "transport_distance": ot_distance,
            "fusion_weights_mamba": w_mamba,
            "fusion_weights_attention": w_attention,
        }

    def predict(self, audio_features: torch.Tensor) -> int:
        """
        Predict emotion class for input audio.

        Args:
            audio_features: Input audio features

        Returns:
            Predicted emotion class ID
        """
        with torch.no_grad():
            output = self.forward(audio_features)
            emotion_id = torch.argmax(output["emotion_logits"], dim=1)

        return emotion_id.item() if emotion_id.numel() == 1 else emotion_id
