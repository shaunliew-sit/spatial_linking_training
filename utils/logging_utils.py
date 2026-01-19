"""
Logging utilities for HOI Spatial Linking Training.

Provides WandB integration and logging helpers.
"""

import os
import logging
from typing import Dict, Optional, Any
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
) -> None:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for logging
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    handlers = [logging.StreamHandler()]
    
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers,
    )


def setup_wandb(
    project: str = "spatial-linking-hoi",
    run_name: Optional[str] = None,
    config: Optional[Dict] = None,
    tags: Optional[list] = None,
    notes: Optional[str] = None,
    resume: bool = False,
) -> Optional[Any]:
    """
    Initialize Weights & Biases for experiment tracking.
    
    Args:
        project: WandB project name
        run_name: Name for this run (auto-generated if None)
        config: Configuration dict to log
        tags: List of tags for the run
        notes: Notes about the run
        resume: Whether to resume a previous run
        
    Returns:
        WandB run object or None if WandB not available
    """
    try:
        import wandb
    except ImportError:
        logger.warning("WandB not installed. Install with: pip install wandb")
        return None
    
    if run_name is None:
        run_name = f"spatial_linking_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        run = wandb.init(
            project=project,
            name=run_name,
            config=config,
            tags=tags,
            notes=notes,
            resume="allow" if resume else False,
        )
        logger.info(f"WandB initialized: {run.url}")
        return run
    except Exception as e:
        logger.warning(f"Failed to initialize WandB: {e}")
        return None


def log_metrics(
    metrics: Dict[str, float],
    step: Optional[int] = None,
    prefix: str = "",
) -> None:
    """
    Log metrics to WandB and console.
    
    Args:
        metrics: Dict of metric names to values
        step: Training step (optional)
        prefix: Prefix for metric names
    """
    try:
        import wandb
        
        if wandb.run is not None:
            log_dict = {}
            for key, value in metrics.items():
                full_key = f"{prefix}/{key}" if prefix else key
                log_dict[full_key] = value
            
            if step is not None:
                wandb.log(log_dict, step=step)
            else:
                wandb.log(log_dict)
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Failed to log to WandB: {e}")
    
    # Also log to console
    logger.info(f"Metrics (step={step}): {metrics}")


def log_model_info(model, config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Log model architecture information.
    
    Args:
        model: PyTorch model
        config: Training configuration
        
    Returns:
        Dict with model info
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    info = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_ratio": trainable_params / total_params if total_params > 0 else 0,
        "trainable_params_millions": trainable_params / 1e6,
    }
    
    # Log spatial linking params if available
    if hasattr(model, 'spatial_linking'):
        sl_params = sum(p.numel() for p in model.spatial_linking.parameters())
        info["spatial_linking_params"] = sl_params
        info["spatial_linking_params_millions"] = sl_params / 1e6
    
    logger.info(f"Model Info:")
    logger.info(f"  Total params: {total_params:,}")
    logger.info(f"  Trainable params: {trainable_params:,} ({info['trainable_ratio']*100:.2f}%)")
    
    try:
        import wandb
        if wandb.run is not None:
            wandb.log({"model_info": info})
    except:
        pass
    
    return info


def log_training_progress(
    epoch: int,
    step: int,
    loss: float,
    learning_rate: float,
    **kwargs
) -> None:
    """
    Log training progress.
    
    Args:
        epoch: Current epoch
        step: Current step
        loss: Training loss
        learning_rate: Current learning rate
        **kwargs: Additional metrics
    """
    metrics = {
        "epoch": epoch,
        "step": step,
        "loss": loss,
        "learning_rate": learning_rate,
        **kwargs
    }
    
    log_metrics(metrics, step=step, prefix="train")


def log_evaluation_results(
    results: Dict[str, float],
    epoch: Optional[int] = None,
    split: str = "val",
) -> None:
    """
    Log evaluation results.
    
    Args:
        results: Dict of evaluation metrics
        epoch: Current epoch (optional)
        split: Dataset split (train, val, test)
    """
    step = epoch if epoch is not None else None
    log_metrics(results, step=step, prefix=f"eval/{split}")


def finish_wandb() -> None:
    """Finish WandB run."""
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
            logger.info("WandB run finished")
    except:
        pass


class TrainingLogger:
    """
    Unified training logger with WandB integration.
    
    Usage:
        logger = TrainingLogger(project="my-project", config=config)
        logger.log_step(step=100, loss=0.5, lr=1e-4)
        logger.log_eval(metrics={"accuracy": 0.95})
        logger.finish()
    """
    
    def __init__(
        self,
        project: str = "spatial-linking-hoi",
        run_name: Optional[str] = None,
        config: Optional[Dict] = None,
        log_dir: Optional[str] = None,
    ):
        self.project = project
        self.run_name = run_name
        self.config = config or {}
        self.log_dir = Path(log_dir) if log_dir else None
        
        # Setup file logging if log_dir provided
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            log_file = self.log_dir / "training.log"
            setup_logging(log_file=str(log_file))
        
        # Initialize WandB
        self.wandb_run = setup_wandb(
            project=project,
            run_name=run_name,
            config=config,
        )
        
        self.step = 0
        self.epoch = 0
    
    def log_step(
        self,
        loss: float,
        learning_rate: float,
        step: Optional[int] = None,
        **kwargs
    ) -> None:
        """Log a training step."""
        if step is not None:
            self.step = step
        else:
            self.step += 1
        
        metrics = {
            "loss": loss,
            "learning_rate": learning_rate,
            **kwargs
        }
        
        log_metrics(metrics, step=self.step, prefix="train")
    
    def log_epoch(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Log epoch-level metrics."""
        self.epoch = epoch
        log_metrics(metrics, step=self.step, prefix=f"epoch_{epoch}")
    
    def log_eval(
        self,
        metrics: Dict[str, float],
        split: str = "val",
    ) -> None:
        """Log evaluation metrics."""
        log_evaluation_results(metrics, epoch=self.epoch, split=split)
    
    def log_model(self, model) -> None:
        """Log model information."""
        log_model_info(model, self.config)
    
    def finish(self) -> None:
        """Finish logging."""
        finish_wandb()
