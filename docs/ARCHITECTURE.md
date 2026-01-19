# Architecture Explanation

## Overview

This implementation extends **Qwen3-VL-8B-Instruct** with a **Spatial Linking Module** to enhance region-based reasoning for Human-Object Interaction (HOI) detection. The architecture introduces a learnable mechanism that preserves spatial position information while maintaining token semantics.

## Core Components

### 1. Spatial Linking Module (`SpatialLinkingModule`)

**Purpose**: Links `<|box_end|>` tokens to original image patch embeddings within bounding boxes via cross-attention.

**Key Innovation**: 
- **Residual Addition** (not replacement) - preserves original token semantics
- **Cross-Attention** - Query from `<|box_end|>`, Key/Value from image patches in bbox
- **Spatial Preservation** - Patches retain their position embeddings (2D + MRoPE)

**Architecture**:
```
<|box_end|> token embedding (Query)
    ↓
Cross-Attention Layer (8 heads, 3584 dim)
    ↓
Key/Value: Image patches within bbox
    ↓
Linked features
    ↓
ADD to original <|box_end|> embedding (residual)
    ↓
Layer Norm
    ↓
Refine MLP (3584 → 7168 → 3584)
    ↓
Enhanced <|box_end|> token
```

**Trainable Parameters**: ~10M (only cross-attention + MLP)

### 2. Interaction Token (Union Box)

**Purpose**: Captures spatial context of human-object interactions.

**Three-Region Representation**:
1. **Person Box**: Bounding box of the person
2. **Object Box**: Bounding box of the object
3. **Interaction Box**: Union of person + object boxes

**Why Union Box?**
- Captures the **connecting space** between entities
- Provides **holistic spatial context** for interaction
- Helps distinguish similar interactions (e.g., "adjusting necktie" vs "wearing necktie")

**Computation**:
```python
interaction_box = [
    min(person_x1, object_x1),  # x1
    min(person_y1, object_y1),  # y1
    max(person_x2, object_x2),  # x2
    max(person_y2, object_y2)   # y2
]
```

### 3. Combined Model (`SpatialLinkingInteractionModel`)

**Base Model**: Extends `Qwen3VLForConditionalGeneration` (Qwen3-VL-8B-Instruct)

> **Note**: The implementation uses `Qwen3VLForConditionalGeneration` and `Qwen3VLConfig` 
> from `transformers`. This properly handles Qwen3-VL specific features like:
> - Nested configuration (`config.text_config.hidden_size`)
> - DeepStack visual features returned as tuples from `get_image_features()`
> - Interleaved-MRoPE position encoding

**Forward Pass Flow**:
```
1. Input Processing
   ├── Text tokens → Text embeddings
   └── Images → Vision encoder → Image embeddings

2. Embedding Merging
   └── Merge image embeddings into text sequence at <image> positions

3. Spatial Linking (if refer_boxes provided)
   ├── For each <|box_end|> token:
   │   ├── Find corresponding bbox (person/object/interaction)
   │   ├── Get image patches within bbox
   │   ├── Cross-attention: <|box_end|> → patches
   │   └── ADD result to <|box_end|> embedding
   └── Apply to all 3 boxes (person, object, interaction)

4. LLM Forward
   └── Process enhanced embeddings through language model
```

**Freezing Strategy**:
- ✅ **Vision Encoder**: Frozen (preserves visual representations)
- ✅ **LLM**: Trainable via LoRA (parameter-efficient, ~64M params)
- ✅ **Spatial Linking**: Always trainable (~10M params)

## Data Flow

### Training Data Format

**Spatial SFT Sample**:
```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "<image> What action is the person performing..."},
    {"role": "assistant", "content": "riding bicycle"}
  ],
  "images": ["path/to/image.jpg"],
  "refer_boxes": [
    [100, 100, 300, 400],  // person box
    [250, 200, 450, 350],  // object box
    [100, 100, 450, 400]   // interaction box (union)
  ]
}
```

**CoF Tool-Calling Sample**:
```json
{
  "messages": [
    {"role": "system", "content": "You are helpful...\n<tools>zoom_in</tools>"},
    {"role": "user", "content": "<image> What action..."},
    {"role": "assistant", "content": "<think>...</think>\n<tool_call>zoom_in</tool_call>"},
    {"role": "user", "content": "<image> (zoomed)"},
    {"role": "assistant", "content": "<think>...</think>\n<answer>riding bicycle</answer>"}
  ],
  "images": ["original.jpg", "cropped.jpg"]
}
```

### Training Process

1. **Data Collation** (`HOISpatialCollator`):
   - Loads images from paths
   - Applies chat template to messages
   - Tokenizes with Qwen3-VL processor
   - Pads `refer_boxes` for batching

2. **Forward Pass**:
   - Model processes batch with `refer_boxes`
   - Spatial linking enhances `<|box_end|>` tokens
   - LLM generates predictions

3. **Loss Computation**:
   - Standard causal language modeling loss
   - Only assistant tokens contribute to loss

## Key Design Decisions

### Why Residual Addition?

**Problem**: Replacing `<|box_end|>` embeddings destroys learned semantics.

**Solution**: Add cross-attention output to original embedding:
```
enhanced = original_embedding + cross_attn_output
```

This preserves:
- ✅ Token semantics (learned from pretraining)
- ✅ Spatial information (from patches)
- ✅ Model stability (residual connections)

### Why Three-Region Representation?

**Problem**: Two boxes (person + object) lack explicit interaction context.

**Solution**: Add interaction box (union) that captures:
- Relative positioning between entities
- Connecting space where interaction occurs
- Environmental context

### Why Quality over Quantity?

**Research Finding**: Small, high-quality SFT datasets outperform large noisy ones.

**Implementation**:
- ~5K spatial linking samples (filtered for clarity)
- ~5K CoF tool samples (35% with tools, 65% direct)
- Combined training prevents catastrophic forgetting

## Training Configuration

**LoRA Settings**:
- Rank: 64
- Alpha: 128
- Target: All attention + MLP layers
- Dropout: 0.05

**Training Hyperparameters**:
- Learning Rate: 2e-5
- Batch Size: 2 per device × 16 accumulation = 32 effective
- Epochs: 3
- Warmup: 10%
- Scheduler: Cosine decay

**Memory Optimization**:
- Gradient checkpointing enabled
- bfloat16 precision
- Vision encoder frozen

## Evaluation Metrics

**Grounding Task** (box prediction):
- AR: Average Recall @ IoU 0.5:0.95
- AR@0.5, AR@0.75: Recall at specific thresholds
- ARs, ARm, ARl: Size-based metrics (small/medium/large objects)

**Referring Task** (action prediction):
- BERTScore: Semantic similarity (P, R, F1)
- Model: roberta-large or deberta-v2-xxlarge-mnli

## Advantages of This Architecture

1. **Parameter Efficient**: Only ~10M trainable params (spatial linking) + LoRA
2. **Semantic Preservation**: Residual addition maintains token meaning
3. **Spatial Awareness**: Cross-attention to patches preserves position info
4. **Flexible**: Works with both grounding and referring tasks
5. **Extensible**: Easy to add more tools or regions

## Attention Visualization

The implementation includes comprehensive attention visualization for interpretability:

**Visualization Functions** (`utils/visualization.py`):
- `visualize_spatial_attention()`: Single region attention heatmap
- `visualize_multi_region_attention()`: Compare person/object/interaction attention
- `visualize_attention_heads()`: Per-head attention patterns
- `log_attention_to_wandb()`: Automatic WandB logging

**Usage**:
```python
from spatial_linking_training.utils import visualize_spatial_attention

# During inference with output_spatial_attentions=True
outputs = model(
    ...,
    output_spatial_attentions=True
)
attention_info = model.get_spatial_attention_weights()

# Visualize
fig = visualize_spatial_attention(
    image="path/to/image.jpg",
    attention_info=attention_info[0][0],  # First sample, first box
)
```

**What You Can See**:
- Which image patches each `<|box_end|>` token attends to
- How attention differs across person/object/interaction boxes
- Per-head attention specialization

## Limitations & Future Work

**Current Limitations**:
- Single tool (zoom_in) in CoF implementation
- Fixed 3-region representation

**Potential Improvements**:
- Multi-tool support (zoom_out, detect_objects)
- Dynamic region selection
- RL fine-tuning with verl-tool (next stage)
