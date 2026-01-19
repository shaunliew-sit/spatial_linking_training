# Spatial Linking HOI Training Pipeline

A modular training pipeline for Qwen3-VL with Spatial Linking and Interaction Token for Human-Object Interaction (HOI) detection.

## Overview

This project implements:
- **Spatial Linking Module**: Cross-attention mechanism that links `<|box_end|>` tokens to image patches, preserving spatial position information
- **Interaction Token**: Three-region representation using person, object, and union (interaction) boxes
- **TRL SFTTrainer**: Supervised fine-tuning with LoRA for parameter-efficient training
- **Tool-Calling CoF**: Chain-of-Focus style tool-augmented reasoning
- **Multi-GPU Training**: Support for distributed training with accelerate/DeepSpeed

## Project Structure

```
spatial_linking_training/
├── models/
│   ├── __init__.py
│   ├── spatial_linking.py      # SpatialLinkingModule
│   └── spatial_model.py        # SpatialLinkingInteractionModel (Qwen3VL)
├── data/
│   ├── __init__.py
│   ├── collator.py             # HOISpatialCollator
│   └── dataset.py              # Dataset utilities
├── scripts/
│   ├── add_refer_boxes.py           # Add refer_boxes to existing datasets
│   ├── create_spatial_sft_data.py   # Generate spatial linking SFT data
│   ├── create_cof_tool_sft_data.py  # Generate CoF tool-calling data
│   ├── train.py                      # Main training script
│   └── evaluate.py                   # Evaluation script
├── configs/
│   ├── sft_lora_config.yaml              # Base training config
│   ├── sft_lora_config_dgx.yaml          # DGX Spark (128GB) config
│   ├── sft_lora_config_dgx_safe.yaml     # DGX safe (lower batch) config
│   ├── sft_lora_config_multi_gpu.yaml    # Multi-GPU (H200/H100) config
│   ├── accelerate_config.yaml            # Accelerate launcher config
│   └── deepspeed_config.json             # DeepSpeed ZeRO-2 config
├── utils/
│   ├── __init__.py
│   ├── metrics.py              # Evaluation metrics (AR, BERTScore)
│   ├── logging_utils.py        # WandB logging
│   └── visualization.py        # Attention heatmap visualization
└── docs/
    ├── README.md               # This file
    └── ARCHITECTURE.md         # Detailed architecture explanation
```

## Quick Start

### 0. Install Dependencies

```bash
cd spatial_linking_training
pip install -r requirements.txt
```

### 1. Prepare Dataset (Add refer_boxes)

If using the existing `hoi_cof_sft` dataset, add `refer_boxes` field:

```bash
python scripts/add_refer_boxes.py \
    --input /workspace/dataset/hoi_cof_sft/hoi_cof_sft_data.json \
    --output /workspace/dataset/hoi_cof_sft/hoi_cof_sft_data_with_boxes.json
```

Or generate new SFT datasets:

```bash
# Generate spatial linking SFT data (~5K samples)
python scripts/create_spatial_sft_data.py \
    --input-dir /workspace/dataset/benchmarks_simplified \
    --output-file /workspace/dataset/spatial_sft_data.json \
    --num-samples 5000

# Generate CoF tool-calling SFT data (~5K samples)
python scripts/create_cof_tool_sft_data.py \
    --input-dir /workspace/dataset/benchmarks_simplified \
    --output-file /workspace/dataset/cof_tool_sft_data.json \
    --num-samples 5000 \
    --tool-ratio 0.35
```

---

## Training Commands

### Single GPU (DGX Spark 128GB)

```bash
# Standard training
python scripts/train.py --config configs/sft_lora_config_dgx.yaml

# Safe mode (lower memory usage)
python scripts/train.py --config configs/sft_lora_config_dgx_safe.yaml
```

### Multi-GPU with Accelerate (Recommended for H200/H100)

```bash
# 8 GPUs with default settings
accelerate launch --num_processes=8 scripts/train.py \
    --config configs/sft_lora_config_multi_gpu.yaml

# With custom accelerate config
accelerate launch --config_file configs/accelerate_config.yaml \
    scripts/train.py --config configs/sft_lora_config_multi_gpu.yaml

# 4 GPUs
accelerate launch --num_processes=4 scripts/train.py \
    --config configs/sft_lora_config_multi_gpu.yaml
```

### Multi-GPU with torchrun

```bash
# 8 GPUs
torchrun --nproc_per_node=8 scripts/train.py \
    --config configs/sft_lora_config_multi_gpu.yaml

# 4 GPUs
torchrun --nproc_per_node=4 scripts/train.py \
    --config configs/sft_lora_config_multi_gpu.yaml
```

### Multi-GPU with DeepSpeed ZeRO-2

Add `deepspeed: configs/deepspeed_config.json` to your config YAML, then:

```bash
accelerate launch --num_processes=8 scripts/train.py \
    --config configs/sft_lora_config_multi_gpu.yaml
```

### Resume Training from Checkpoint

```bash
# Resume using run ID (finds latest checkpoint automatically)
python scripts/train.py \
    --config configs/sft_lora_config_dgx.yaml \
    --resume_run_id 20260119_092419

# Resume from specific checkpoint path
python scripts/train.py \
    --config configs/sft_lora_config_dgx.yaml \
    --resume_from_checkpoint outputs/spatial_linking_sft_dgx/run_20260119_092419/checkpoint-150
```

### Override Config via Command Line

```bash
python scripts/train.py \
    --config configs/sft_lora_config_dgx.yaml \
    --learning_rate 1e-5 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 8
```

---

## Evaluation Commands

### Referring Task (boxes → action description)

```bash
python scripts/evaluate.py \
    --model_path outputs/spatial_linking_sft_dgx/run_XXXX \
    --test_file /workspace/dataset/benchmarks_simplified/hico_action_referring_test_simplified.json \
    --task_type referring \
    --image_base_dir /workspace/dataset/hoi_cof_sft/images
```

### Grounding Task (action query → boxes)

```bash
python scripts/evaluate.py \
    --model_path outputs/spatial_linking_sft_dgx/run_XXXX \
    --test_file /workspace/dataset/benchmarks_simplified/hico_ground_test_simplified.json \
    --task_type grounding \
    --image_base_dir /workspace/dataset/hoi_cof_sft/images
```

### With Attention Visualization

```bash
python scripts/evaluate.py \
    --model_path outputs/spatial_linking_sft_dgx/run_XXXX \
    --test_file /workspace/dataset/benchmarks_simplified/hico_action_referring_test_simplified.json \
    --task_type referring \
    --image_base_dir /workspace/dataset/hoi_cof_sft/images \
    --visualize_attention \
    --viz_output_dir ./attention_viz \
    --max_samples 20
```

### Save Detailed Per-Sample Results

```bash
python scripts/evaluate.py \
    --model_path outputs/spatial_linking_sft_dgx/run_XXXX \
    --test_file /workspace/dataset/benchmarks_simplified/hico_action_referring_test_simplified.json \
    --task_type referring \
    --image_base_dir /workspace/dataset/hoi_cof_sft/images \
    --save_detailed \
    --output_file ./eval_results.json
```

### Full Evaluation with All Options

```bash
python scripts/evaluate.py \
    --model_path outputs/spatial_linking_sft_dgx/run_XXXX \
    --test_file /workspace/dataset/benchmarks_simplified/hico_action_referring_test_simplified.json \
    --task_type referring \
    --image_base_dir /workspace/dataset/hoi_cof_sft/images \
    --visualize_attention \
    --viz_output_dir ./attention_viz \
    --save_detailed \
    --output_file ./eval_results.json \
    --max_samples 100
```

---

## Monitoring Training

### WandB Dashboard

Training logs to WandB automatically. View at: https://wandb.ai/YOUR_USERNAME/spatial-linking-hoi

### Local Log Files

Each run creates a log file:
```
outputs/spatial_linking_sft_dgx/run_YYYYMMDD_HHMMSS/training.log
```

Monitor in real-time:
```bash
tail -f outputs/spatial_linking_sft_dgx/run_*/training.log
```

### GPU Usage

```bash
watch -n 1 nvidia-smi
```

---

## Components

### Spatial Linking Module

The spatial linking module enhances `<|box_end|>` tokens with region-specific features via cross-attention:

```python
from models import SpatialLinkingModule

module = SpatialLinkingModule(
    hidden_dim=3584,  # Qwen3-VL-8B hidden size
    num_heads=8,
    dropout=0.1,
    use_refine_mlp=True
)

# Forward pass
enhanced_embeds = module(
    inputs_embeds=inputs_embeds,
    input_ids=input_ids,
    image_mask=image_mask,
    refer_boxes=refer_boxes,
    image_grid_thw=image_grid_thw,
    box_end_token_id=151649,
    output_attentions=False  # Set True for visualization
)

# With attention output for visualization
enhanced_embeds, attention_info = module(
    ...,
    output_attentions=True
)
```

Key features:
- Cross-attention: Query from `<|box_end|>`, Key/Value from image patches in bbox
- **Residual addition**: Preserves original token semantics
- Trainable params: ~10M

### Interaction Token (Union Box)

Computes the union of person and object boxes to capture interaction context:

```python
from models import compute_interaction_box

person_box = [100, 100, 300, 400]  # [x1, y1, x2, y2]
object_box = [250, 200, 450, 350]

interaction_box = compute_interaction_box(person_box, object_box)
# Result: [100, 100, 450, 400] - union of both boxes
```

### Combined Model

```python
from models import SpatialLinkingInteractionModel

model = SpatialLinkingInteractionModel.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct",
    torch_dtype=torch.bfloat16
)

# Freeze base model, train only spatial linking
model.freeze_base_model(freeze_vision=True, freeze_llm=True)

# Forward with refer_boxes
outputs = model(
    input_ids=input_ids,
    pixel_values=pixel_values,
    image_grid_thw=image_grid_thw,
    refer_boxes=[[person_box, object_box]],  # Interaction box auto-computed
)

# With attention visualization
outputs = model(
    ...,
    output_spatial_attentions=True
)
attention_info = model.get_spatial_attention_weights()
```

### Data Collator

Custom collator for multimodal training:

```python
from data import HOISpatialCollator

collator = HOISpatialCollator(
    processor=processor,
    max_length=2048,
    image_base_dir="/dataset"
)

# Collate batch
batch = collator(examples)
# Returns: input_ids, attention_mask, labels, pixel_values, refer_boxes
```

### Evaluation Metrics

Ported from `hoi-benchmarks`:

```python
from utils.metrics import evaluate_grounding, evaluate_referring

# Grounding: AR metrics at IoU thresholds
grounding_metrics = evaluate_grounding(predictions, ground_truths)
# Returns: AR, AR@0.5, AR@0.75, ARs, ARm, ARl

# Referring: BERTScore
referring_metrics = evaluate_referring(predictions, references)
# Returns: precision, recall, f1
```

## Configuration

See `configs/sft_lora_config.yaml` for all options:

```yaml
# Model
model_name_or_path: Qwen/Qwen3-VL-8B-Instruct

# LoRA
lora_rank: 64
lora_alpha: 128
lora_target: all

# Training
learning_rate: 2.0e-5
num_train_epochs: 3
per_device_train_batch_size: 2
gradient_accumulation_steps: 16

# Freezing
freeze_vision_tower: true
train_spatial_linking: true

# WandB
report_to: wandb
```

## Dataset Format

### Spatial SFT Format

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "<image> What action..."},
    {"role": "assistant", "content": "riding bicycle"}
  ],
  "images": ["path/to/image.jpg"],
  "refer_boxes": [[person_box], [object_box], [interaction_box]],
  "task_type": "referring"
}
```

### CoF Tool-Calling Format

```json
{
  "messages": [
    {"role": "system", "content": "You are helpful...\n<tools>...</tools>"},
    {"role": "user", "content": "<image> What action..."},
    {"role": "assistant", "content": "<think>...</think>\n<tool_call>...</tool_call>"},
    {"role": "user", "content": "<image> (zoomed)"},
    {"role": "assistant", "content": "<think>...</think>\n<answer>riding bicycle</answer>"}
  ],
  "images": ["original.jpg", "cropped.jpg"],
  "uses_tool": true
}
```

## Training Strategy

### Quality over Quantity

Following research on SFT data quality:
- **Spatial SFT**: ~5K high-quality samples
- **CoF Tool SFT**: ~5K samples (35% with tools, 65% direct)

### Combined Training

Both datasets are combined with weighted sampling to prevent catastrophic forgetting:
- Trains spatial linking module on both tasks simultaneously
- LoRA adapters learn unified representations

### Freezing Strategy

- **Vision encoder**: Frozen (preserves visual representations)
- **LLM**: Trainable via LoRA (parameter-efficient)
- **Spatial linking**: Always trainable (~10M params)

## Attention Visualization

Visualize how spatial linking attends to image patches:

```python
from utils.visualization import (
    visualize_spatial_attention,
    visualize_multi_region_attention,
    log_attention_to_wandb
)

# Single region visualization
fig = visualize_spatial_attention(
    image="path/to/image.jpg",
    attention_info=attention_info[0][0],  # First sample, first box
    title="Person Box Attention"
)

# Multi-region comparison (person, object, interaction)
fig = visualize_multi_region_attention(
    image="path/to/image.jpg",
    attention_info_list=attention_info[0],  # All boxes for first sample
    output_path="attention_viz.png"
)

# Log to WandB during training
log_attention_to_wandb(
    image="path/to/image.jpg",
    attention_info_list=attention_info[0],
    step=global_step
)
```

## Dependencies

```txt
transformers>=4.45.0
trl>=0.10.0
peft>=0.10.0
datasets
wandb
bert_score
torch>=2.0.0
Pillow
qwen-vl-utils
matplotlib  # For visualization
scipy       # For heatmap interpolation
accelerate  # For multi-GPU training
```

---

## Quick Reference (Copy-Paste Commands)

### Complete Workflow

```bash
# 1. Install dependencies
cd spatial_linking_training
pip install -r requirements.txt

# 2. Prepare dataset (add refer_boxes)
python scripts/add_refer_boxes.py \
    --input /workspace/dataset/hoi_cof_sft/hoi_cof_sft_data.json \
    --output /workspace/dataset/hoi_cof_sft/hoi_cof_sft_data_with_boxes.json

# 3a. Train (Single GPU - DGX)
python scripts/train.py --config configs/sft_lora_config_dgx.yaml

# 3b. Train (Multi-GPU - 8x H200/H100)
accelerate launch --num_processes=8 scripts/train.py --config configs/sft_lora_config_multi_gpu.yaml

# 4. Evaluate referring task
python scripts/evaluate.py \
    --model_path outputs/spatial_linking_sft_dgx/run_XXXX \
    --test_file /workspace/dataset/benchmarks_simplified/hico_action_referring_test_simplified.json \
    --task_type referring \
    --image_base_dir /workspace/dataset/hoi_cof_sft/images \
    --visualize_attention \
    --viz_output_dir ./attention_viz

# 5. Evaluate grounding task
python scripts/evaluate.py \
    --model_path outputs/spatial_linking_sft_dgx/run_XXXX \
    --test_file /workspace/dataset/benchmarks_simplified/hico_ground_test_simplified.json \
    --task_type grounding \
    --image_base_dir /workspace/dataset/hoi_cof_sft/images
```

### Config Files Quick Reference

| Config | Use Case | Batch Size | GPUs |
|--------|----------|------------|------|
| `sft_lora_config_dgx.yaml` | DGX Spark 128GB | 16 | 1 |
| `sft_lora_config_dgx_safe.yaml` | DGX (safe mode) | 8 | 1 |
| `sft_lora_config_multi_gpu.yaml` | H200/H100 cluster | 4/GPU | 8 |

---

## Citation

If you use this code, please cite:
- Qwen3-VL: https://github.com/QwenLM/Qwen3-VL
- TRL: https://github.com/huggingface/trl
- HICO-DET: https://www.cs.rochester.edu/~cxu22/research/hico/
- SWIG-HOI: https://arxiv.org/abs/2007.12766
