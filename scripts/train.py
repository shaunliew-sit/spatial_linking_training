#!/usr/bin/env python3
"""
Main Training Script for Spatial Linking HOI Model.

Uses TRL SFTTrainer with PEFT (LoRA) for efficient fine-tuning of
Qwen3-VL-8B-Instruct with spatial linking module.

Model: Qwen/Qwen3-VL-8B-Instruct
Architecture: SpatialLinkingInteractionModel (extends Qwen3VLForConditionalGeneration)

Features:
- TRL SFTTrainer for supervised fine-tuning
- PEFT/LoRA for parameter-efficient training
- WandB integration for experiment tracking
- Support for both spatial linking and CoF tool-calling datasets
- Attention visualization for spatial linking interpretability
- Multi-GPU training support via accelerate/DeepSpeed

Usage:
    # Single GPU
    python scripts/train.py --config configs/sft_lora_config.yaml
    
    # Multi-GPU with accelerate (recommended for H200/H100 clusters)
    accelerate launch --num_processes=8 scripts/train.py --config configs/sft_lora_config.yaml
    
    # Multi-GPU with torchrun
    torchrun --nproc_per_node=8 scripts/train.py --config configs/sft_lora_config.yaml
    
    # With DeepSpeed ZeRO-2 (for very large models)
    accelerate launch --config_file configs/accelerate_config.yaml scripts/train.py --config configs/sft_lora_config.yaml
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional
import yaml

import torch
from transformers import AutoProcessor, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.spatial_model import SpatialLinkingInteractionModel
from data.collator import HOISpatialCollator
from data.dataset import load_combined_dataset, create_train_val_split
from utils.logging_utils import setup_logging, setup_wandb, log_model_info, finish_wandb

def setup_file_logging(output_dir: str, run_id: str = None):
    """Setup logging to both console and file with unique run ID.
    
    Captures DEBUG level to file and INFO to console.
    Also sets up exception hook to log uncaught exceptions.
    """
    import datetime
    import sys
    import traceback
    
    # Generate unique run ID if not provided
    if run_id is None:
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create run-specific directory
    log_dir = Path(output_dir) / f"run_{run_id}"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "training.log"
    
    # Create detailed formatter for file (includes more info)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Simpler formatter for console
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler - capture DEBUG and above (more detailed)
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    # Console handler - INFO and above
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Also configure transformers and other library loggers
    for lib_name in ['transformers', 'datasets', 'peft', 'trl', 'accelerate']:
        lib_logger = logging.getLogger(lib_name)
        lib_logger.setLevel(logging.DEBUG)
        lib_logger.addHandler(file_handler)
    
    # Setup exception hook to log uncaught exceptions
    def exception_hook(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            # Don't log keyboard interrupt
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        error_msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        root_logger.error(f"Uncaught exception:\n{error_msg}")
        # Also call the default hook
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
    
    sys.excepthook = exception_hook
    
    # Log startup info
    root_logger.info(f"=" * 80)
    root_logger.info(f"Training started - Run ID: {run_id}")
    root_logger.info(f"Log file: {log_file}")
    root_logger.info(f"=" * 80)
    
    return log_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def merge_config_with_args(config: Dict, args: argparse.Namespace) -> Dict:
    """Merge command line arguments into config."""
    args_dict = vars(args)
    for key, value in args_dict.items():
        if value is not None and key != 'config':
            config[key] = value
    return config


# =============================================================================
# MODEL SETUP
# =============================================================================

def setup_model(config: Dict):
    """
    Setup model with spatial linking and LoRA.
    
    Returns:
        Tuple of (model, processor)
    """
    model_name = config.get("model_name_or_path", "Qwen/Qwen3-VL-8B-Instruct")
    
    logger.info(f"Loading model: {model_name}")
    
    # Load processor
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    
    # Ensure pad token is set
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    # Check if running with accelerate/distributed
    import os
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1
    
    if is_distributed:
        logger.info(f"Distributed training: world_size={world_size}, local_rank={local_rank}")
        # For distributed training, don't use device_map="auto"
        # Let accelerate/trainer handle device placement
        device_map = {"": local_rank} if local_rank >= 0 else None
    else:
        logger.info("Single GPU training with device_map='auto'")
        device_map = "auto"
    
    # Load model with spatial linking
    # Note: Using "sdpa" (Scaled Dot Product Attention) instead of "flash_attention_2"
    # because flash-attn doesn't have official CUDA 13.0 wheels yet.
    # SDPA is built into PyTorch 2.x and offers comparable performance.
    model = SpatialLinkingInteractionModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=device_map,
        attn_implementation="sdpa",  # Use PyTorch's native SDPA (flash-attn requires CUDA 12.x)
    )
    
    # Set loss_type to avoid warning (set it regardless of whether it exists)
    model.config.loss_type = "ForCausalLMLoss"
    
    # Set box token IDs from tokenizer
    model.set_box_token_ids(processor.tokenizer)
    
    # Freeze components as specified
    freeze_vision = config.get("freeze_vision_tower", True)
    freeze_llm = config.get("freeze_llm", False)  # Don't freeze if using LoRA
    
    if freeze_vision:
        logger.info("Freezing vision tower")
        if hasattr(model, 'visual'):
            for param in model.visual.parameters():
                param.requires_grad = False
    
    # Setup LoRA if configured
    if config.get("finetuning_type") == "lora":
        logger.info("Setting up LoRA")
        
        lora_config = LoraConfig(
            r=config.get("lora_rank", 64),
            lora_alpha=config.get("lora_alpha", 128),
            target_modules=get_target_modules(config),
            lora_dropout=config.get("lora_dropout", 0.05),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # Ensure spatial linking is trainable
    if config.get("train_spatial_linking", True):
        logger.info("Enabling spatial linking training")
        for param in model.spatial_linking.parameters():
            param.requires_grad = True
    
    return model, processor


def get_target_modules(config: Dict) -> list:
    """Get LoRA target modules from config."""
    target = config.get("lora_target", "all")
    
    if target == "all":
        # Common target modules for Qwen models
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    elif isinstance(target, str):
        return target.split(",")
    else:
        return target


# =============================================================================
# DATASET SETUP
# =============================================================================

def setup_dataset(config: Dict, processor):
    """
    Setup training and validation datasets.
    
    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    # Support both separate and combined dataset configurations
    spatial_sft_file = config.get("spatial_sft_file")
    cof_sft_file = config.get("cof_sft_file")
    
    # Check which files exist
    data_files = []
    if spatial_sft_file and Path(spatial_sft_file).exists():
        data_files.append(spatial_sft_file)
        logger.info(f"Found spatial SFT data: {spatial_sft_file}")
    if cof_sft_file and Path(cof_sft_file).exists():
        data_files.append(cof_sft_file)
        logger.info(f"Found CoF SFT data: {cof_sft_file}")
    
    if not data_files:
        raise FileNotFoundError(
            f"No data files found. Check config paths:\n"
            f"  spatial_sft_file: {spatial_sft_file}\n"
            f"  cof_sft_file: {cof_sft_file}"
        )
    
    # Load dataset
    dataset = load_dataset("json", data_files=data_files, split="train")
    logger.info(f"Loaded {len(dataset)} samples")
    
    # Create train/val split
    val_ratio = config.get("val_ratio", 0.05)
    if val_ratio > 0:
        split = dataset.train_test_split(test_size=val_ratio, seed=42)
        train_dataset = split["train"]
        eval_dataset = split["test"]
        logger.info(f"Train: {len(train_dataset)}, Val: {len(eval_dataset)}")
    else:
        train_dataset = dataset
        eval_dataset = None
    
    return train_dataset, eval_dataset


# =============================================================================
# TRAINING
# =============================================================================

from transformers import TrainerCallback

class FileLoggingCallback(TrainerCallback):
    """Callback to log training progress to file."""
    
    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.start_time = None
    
    def on_train_begin(self, args, state, control, **kwargs):
        import time
        self.start_time = time.time()
        self._log(f"Training started - Total steps: {state.max_steps}")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            import time
            elapsed = time.time() - self.start_time if self.start_time else 0
            elapsed_str = f"{elapsed/3600:.1f}h" if elapsed > 3600 else f"{elapsed/60:.1f}m"
            
            step = state.global_step
            total = state.max_steps
            progress = 100 * step / total if total > 0 else 0
            
            # Estimate remaining time
            if step > 0:
                time_per_step = elapsed / step
                remaining = (total - step) * time_per_step
                eta_str = f"{remaining/3600:.1f}h" if remaining > 3600 else f"{remaining/60:.1f}m"
            else:
                eta_str = "calculating..."
            
            # Handle both training logs (loss) and evaluation logs (eval_loss)
            loss = logs.get('loss', logs.get('eval_loss', None))
            lr = logs.get('learning_rate', None)
            
            # Format loss safely (could be None or a number)
            loss_str = f"{loss:.4f}" if isinstance(loss, (int, float)) else "N/A"
            lr_str = f"{lr:.2e}" if isinstance(lr, (int, float)) else "N/A"
            
            # Check if this is an evaluation log
            if 'eval_loss' in logs:
                eval_loss = logs.get('eval_loss', 0)
                eval_runtime = logs.get('eval_runtime', 0)
                self._log(f"Evaluation | Step {step}/{total} | Eval Loss: {eval_loss:.4f} | Runtime: {eval_runtime:.1f}s")
            else:
                self._log(f"Step {step}/{total} ({progress:.1f}%) | Loss: {loss_str} | LR: {lr_str} | Elapsed: {elapsed_str} | ETA: {eta_str}")
    
    def on_train_end(self, args, state, control, **kwargs):
        import time
        total_time = time.time() - self.start_time if self.start_time else 0
        self._log(f"Training completed in {total_time/3600:.2f} hours")
    
    def _log(self, message: str):
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, 'a') as f:
            f.write(f"{timestamp} - {message}\n")
        print(f"{timestamp} - {message}")


def find_latest_checkpoint(run_dir: Path) -> Optional[str]:
    """Find the latest checkpoint in a run directory."""
    if not run_dir.exists():
        return None
    
    checkpoints = list(run_dir.glob("checkpoint-*"))
    if not checkpoints:
        return None
    
    # Sort by checkpoint number
    def get_step(p):
        try:
            return int(p.name.split("-")[1])
        except (IndexError, ValueError):
            return 0
    
    checkpoints.sort(key=get_step, reverse=True)
    return str(checkpoints[0])


def train(config: Dict, resume_from_checkpoint: str = None, resume_run_id: str = None):
    """Main training function with resume capability."""
    import datetime
    
    base_output_dir = config.get("output_dir", "./outputs/spatial_linking_sft")
    
    # Handle resume logic
    if resume_run_id:
        # Resume from a specific run ID - find latest checkpoint
        run_dir = Path(base_output_dir) / f"run_{resume_run_id}"
        resume_from_checkpoint = find_latest_checkpoint(run_dir)
        if resume_from_checkpoint:
            logger.info(f"Resuming from run {resume_run_id}, checkpoint: {resume_from_checkpoint}")
            run_id = resume_run_id
        else:
            logger.warning(f"No checkpoint found in {run_dir}, starting fresh run")
            run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            resume_from_checkpoint = None
    elif resume_from_checkpoint:
        # Resume from specific checkpoint path
        checkpoint_path = Path(resume_from_checkpoint)
        if checkpoint_path.exists():
            # Extract run_id from checkpoint path
            run_dir = checkpoint_path.parent
            if run_dir.name.startswith("run_"):
                run_id = run_dir.name.replace("run_", "")
            else:
                run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")
        else:
            logger.warning(f"Checkpoint not found: {resume_from_checkpoint}, starting fresh run")
            run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            resume_from_checkpoint = None
    else:
        # Fresh run
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Setup output directory and file logging with run ID
    log_file = setup_file_logging(base_output_dir, run_id)
    
    # Update output_dir to include run_id for model checkpoints
    output_dir = str(Path(base_output_dir) / f"run_{run_id}")
    
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Log file: {log_file}")
    if resume_from_checkpoint:
        logger.info(f"Resuming from: {resume_from_checkpoint}")
    
    # Setup logging
    setup_logging()
    
    # Setup WandB with unique run name
    if config.get("report_to") == "wandb":
        wandb_run_name = f"{config.get('run_name', 'spatial_linking')}_{run_id}"
        setup_wandb(
            project=config.get("wandb_project", "spatial-linking-hoi"),
            run_name=wandb_run_name,
            config=config,
        )
    
    # Setup model and processor
    model, processor = setup_model(config)
    
    # Log model info
    log_model_info(model, config)
    
    # Setup dataset
    train_dataset, eval_dataset = setup_dataset(config, processor)
    
    # Setup data collator
    collator = HOISpatialCollator(
        processor=processor,
        max_length=config.get("cutoff_len", 2048),
        image_base_dir=config.get("image_base_dir", "/dataset"),
    )
    
    # Setup training arguments
    # Note: output_dir was already set with run_id at line 383
    # Don't overwrite it here
    
    training_args = SFTConfig(
        output_dir=output_dir,  # Uses the run_id-prefixed path set earlier
        
        # Training hyperparameters
        learning_rate=float(config.get("learning_rate", 2e-5)),
        num_train_epochs=config.get("num_train_epochs", 3),
        per_device_train_batch_size=config.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=config.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 8),
        
        # Optimizer
        optim=config.get("optim", "adamw_torch"),
        weight_decay=config.get("weight_decay", 0.01),
        
        # Learning rate schedule
        warmup_ratio=config.get("warmup_ratio", 0.1),
        lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
        
        # Precision
        bf16=config.get("bf16", True),
        fp16=config.get("fp16", False),
        
        # Logging
        logging_steps=config.get("logging_steps", 10),
        logging_first_step=True,
        
        # Saving - save checkpoints for resume capability
        save_strategy=config.get("save_strategy", "steps"),
        save_steps=config.get("save_steps", 50),  # Save every 50 steps for resume
        save_total_limit=config.get("save_total_limit", 3),
        
        # Evaluation
        eval_strategy=config.get("eval_strategy", "epoch") if eval_dataset else "no",
        
        # Important for VLM training
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        
        # Reporting
        report_to=config.get("report_to", "wandb"),
        run_name=config.get("run_name"),
        
        # Gradient checkpointing for memory efficiency
        gradient_checkpointing=config.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        
        # Max length (TRL 0.24+ uses max_length)
        max_length=config.get("cutoff_len", 2048),
    )
    
    # Create file logging callback
    file_callback = FileLoggingCallback(log_file)
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=processor,
        callbacks=[file_callback],
    )
    
    # Train (with optional resume)
    logger.info("Starting training...")
    logger.info(f"Monitor progress: tail -f {log_file}")
    
    if resume_from_checkpoint:
        logger.info(f"Resuming training from checkpoint: {resume_from_checkpoint}")
        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        train_result = trainer.train()
    
    # Save final model
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model()
    
    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # Finish WandB
    finish_wandb()
    
    logger.info("Training complete!")
    
    return metrics


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Spatial Linking HOI Model")
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/sft_lora_config.yaml",
        help="Path to config YAML file"
    )
    
    # Resume training option
    parser.add_argument(
        "--resume_from_checkpoint", 
        type=str, 
        default=None,
        help="Path to checkpoint directory to resume from (e.g., outputs/spatial_linking_sft_dgx/run_20260119_093045/checkpoint-100)"
    )
    parser.add_argument(
        "--resume_run_id",
        type=str,
        default=None,
        help="Run ID to resume (will auto-find latest checkpoint in that run directory)"
    )
    
    # Override options
    parser.add_argument("--model_name_or_path", type=str, help="Model name or path")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, help="Number of epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, help="Gradient accumulation")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--run_name", type=str, help="WandB run name")
    parser.add_argument("--lora_rank", type=int, help="LoRA rank")
    parser.add_argument("--spatial_sft_file", type=str, help="Spatial SFT data file")
    parser.add_argument("--cof_sft_file", type=str, help="CoF SFT data file")
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        config = load_config(str(config_path))
        logger.info(f"Loaded config from {config_path}")
    else:
        logger.warning(f"Config not found: {config_path}, using defaults")
        config = {}
    
    # Merge with command line args
    config = merge_config_with_args(config, args)
    
    # Set default model if not specified
    if "model_name_or_path" not in config:
        config["model_name_or_path"] = "Qwen/Qwen3-VL-8B-Instruct"
    
    # Train (with optional resume)
    train(
        config,
        resume_from_checkpoint=args.resume_from_checkpoint,
        resume_run_id=args.resume_run_id,
    )


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        # Ensure the error is logged before exiting
        logger.error(f"Training failed with error: {e}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        raise
