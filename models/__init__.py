"""
Spatial Linking Models for HOI Detection with Qwen3-VL.

This module provides:
- SpatialLinkingModule: Cross-attention mechanism linking <|box_end|> tokens to image patches
- SpatialLinkingInteractionModel: Extended Qwen3-VL with spatial linking for 3-region representation
"""

from .spatial_linking import SpatialLinkingModule
from .spatial_model import (
    SpatialLinkingInteractionModel,
    compute_interaction_box,
    compute_union_box,
)

__all__ = [
    "SpatialLinkingModule",
    "SpatialLinkingInteractionModel",
    "compute_interaction_box",
    "compute_union_box",
]
