#!/bin/bash
# Full pipeline for one model. Replace MODEL and paths as needed.

MODEL="llava_1_5_7b"          # or: internvl_2_5_8b, qwen2_5_vl_7b
OUTPUT="outputs/${MODEL}"
CONFIG="configs/model_configs.yaml"
DEVICE="cuda"

# Step 1: Generate descriptions + GPT-4o labeling
python scripts/generate_and_label.py \
    --model $MODEL \
    --config $CONFIG \
    --output-dir $OUTPUT \
    --device $DEVICE \
    --resume

# Step 2: Extract ADS + CGC features
python scripts/extract_features.py \
    --model $MODEL \
    --config $CONFIG \
    --output-dir $OUTPUT \
    --device $DEVICE \
    --resume

# Step 3: Train classifiers + evaluate
python scripts/train_and_eval.py \
    --model $MODEL \
    --config $CONFIG \
    --output-dir $OUTPUT
