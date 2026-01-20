#!/bin/bash
# Example script to compare attention between base and fine-tuned models

# Referring task example
python scripts/compare_attention.py \
    --base_model Qwen/Qwen3-VL-8B-Instruct \
    --finetuned_model outputs/spatial_linking_sft_multi_gpu/run_20260120_034050 \
    --image_path /workspace/dataset/hico_20160224_det/images/test2015/HICO_test2015_00000001.jpg \
    --output_dir ./attention_comparison \
    --task_type referring \
    --person_box 320 306 359 349 \
    --object_box 148 345 376 414 \
    --device cuda:0

# Grounding task example (note: may not have spatial attention)
python scripts/compare_attention.py \
    --base_model Qwen/Qwen3-VL-8B-Instruct \
    --finetuned_model outputs/spatial_linking_sft_multi_gpu/run_20260120_034050 \
    --image_path /workspace/dataset/hico_20160224_det/images/test2015/HICO_test2015_00000001.jpg \
    --output_dir ./attention_comparison \
    --task_type grounding \
    --action "sitting on" \
    --object_category bench \
    --device cuda:0
