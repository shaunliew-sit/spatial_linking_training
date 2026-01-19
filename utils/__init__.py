"""
Utility functions for Spatial Linking HOI Training.

This module provides:
- Evaluation metrics (AR, BERTScore)
- Logging utilities for WandB
- Attention visualization for interpretability
"""

from .metrics import (
    calculate_iou,
    evaluate_grounding,
    evaluate_referring,
    compute_grounding_metrics,
)
from .logging_utils import setup_wandb, log_metrics

# Visualization utilities (optional dependencies)
try:
    from .visualization import (
        visualize_spatial_attention,
        visualize_multi_region_attention,
        visualize_attention_heads,
        save_attention_batch,
        log_attention_to_wandb,
        create_attention_heatmap,
    )
    _HAS_VISUALIZATION = True
except ImportError:
    _HAS_VISUALIZATION = False

__all__ = [
    "calculate_iou",
    "evaluate_grounding",
    "evaluate_referring",
    "compute_grounding_metrics",
    "setup_wandb",
    "log_metrics",
]

if _HAS_VISUALIZATION:
    __all__.extend([
        "visualize_spatial_attention",
        "visualize_multi_region_attention",
        "visualize_attention_heads",
        "save_attention_batch",
        "log_attention_to_wandb",
        "create_attention_heatmap",
    ])
