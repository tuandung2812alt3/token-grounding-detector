#!/bin/bash
# POPE evaluation for one model.

MODEL="llava_1_5_7b"          # or: internvl_2_5_8b, qwen2_5_vl_7b
OUTPUT="outputs/pope/${MODEL}"
CONFIG="configs/model_configs.yaml"
DEVICE="cuda"

python scripts/pope_pipeline.py \
    --model $MODEL \
    --config $CONFIG \
    --pope-dir data/pope \
    --coco-image-dir data/coco/val2014 \
    --output-dir $OUTPUT \
    --device $DEVICE \
    --resume
