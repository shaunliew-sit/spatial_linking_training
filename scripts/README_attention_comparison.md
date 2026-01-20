# Attention Heatmap Comparison

This script compares spatial attention heatmaps between the base Qwen3-VL model and your fine-tuned model.

## Usage

### Referring Task (with spatial attention)

```bash
python scripts/compare_attention.py \
    --base_model Qwen/Qwen3-VL-8B-Instruct \
    --finetuned_model outputs/spatial_linking_sft_multi_gpu/run_20260120_034050 \
    --image_path /workspace/dataset/hico_20160224_det/images/test2015/HICO_test2015_00000001.jpg \
    --output_dir ./attention_comparison \
    --task_type referring \
    --person_box 320 306 359 349 \
    --object_box 148 345 376 414 \
    --device cuda:0
```

### Grounding Task

Note: Grounding task may not have spatial attention available since it doesn't use `refer_boxes` initially.

```bash
python scripts/compare_attention.py \
    --base_model Qwen/Qwen3-VL-8B-Instruct \
    --finetuned_model outputs/spatial_linking_sft_multi_gpu/run_20260120_034050 \
    --image_path /workspace/dataset/hico_20160224_det/images/test2015/HICO_test2015_00000001.jpg \
    --output_dir ./attention_comparison \
    --task_type grounding \
    --action "sitting on" \
    --object_category bench \
    --device cuda:0
```

## Output

The script generates a side-by-side comparison visualization showing:
- **Top row**: Base model attention heatmaps for person, object, and interaction boxes
- **Bottom row**: Fine-tuned model attention heatmaps for the same boxes

Saved to: `{output_dir}/attention_comparison_{task_type}.png`

## Understanding the Visualization

- **Hot colormap**: Red/yellow = high attention, dark = low attention
- **Bounding boxes**: 
  - Green = Person box
  - Cyan = Object box  
  - Yellow = Interaction box (union of person + object)
- **Attention patterns**: Shows which image regions the model focuses on when processing each box

## Notes

- The base model may not have trained spatial_linking weights, so attention patterns may be random/untrained
- The fine-tuned model should show more focused attention on relevant regions
- For referring task, spatial attention is available via `refer_boxes`
- For grounding task, spatial attention may not be available (model doesn't receive boxes as input)
