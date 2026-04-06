# Token Grounding Detector

Official implementation of **"Beyond the Global Scores: Fine-Grained Token Grounding as a Robust Detector of LVLM Hallucinations"** (CVPR 2026).

## Overview

We propose two patch-level structural metrics for detecting hallucinated tokens in Large Vision-Language Models:

- **Attention Dispersion Score (ADS)**: Measures how spatially focused a token's attention is over image patches. Hallucinated tokens produce diffuse, scattered attention; faithful tokens show compact focus.
- **Cross-modal Grounding Consistency (CGC)**: Measures semantic alignment between a token's hidden representation and the most relevant image patches. Hallucinated tokens show weak alignment with all regions.


## Installation

```bash
conda create -n tgd python=3.10 -y
conda activate tgd
pip install -r requirements.txt
```


## Data Preparation

### MS-COCO 2014

Download the COCO 2014 validation set and annotations:

Expected structure:
```
data/coco/
├── val2014/
│   ├── COCO_val2014_000000000042.jpg
│   └── ...
└── annotations/
    ├── instances_val2014.json
    └── captions_val2014.json
```


## Usage

The pipeline has three steps. Run all three with `bash run.sh`, or individually:

### Step 1: Generate Descriptions + Label with GPT-4o

Generate image captions with the LVLM, then use GPT-4o to identify hallucinated object tokens:

```bash
export OPENAI_API_KEY="sk-..."

python scripts/generate_and_label.py \
    --model llava_1_5_7b \
    --output-dir outputs/llava_1_5_7b \
    --device cuda \
    --resume
```

This produces `generations.json`, `labeling.json`, and `image_splits.json` in the output directory.


### Step 2: Extract ADS + CGC Features

Run prefix forward passes to compute per-layer ADS and CGC features for every labeled token:

```bash
python scripts/extract_features.py \
    --model llava_1_5_7b \
    --output-dir outputs/llava_1_5_7b \
    --device cuda \
    --resume
```

This produces `features.pkl` containing per-token feature vectors.

### Step 3: Train Classifiers + Evaluate

Train XGB, RF, and MLP classifiers on the extracted features:

```bash
python scripts/train_and_eval.py \
    --model llava_1_5_7b \
    --output-dir outputs/llava_1_5_7b
```

Results are saved to `outputs` folder.


## Project Structure

```
token-grounding-detector/
├── configs/model_configs.yaml   # Model and training configurations
├── models/                      # LVLM wrappers (LLaVA, InternVL, Qwen)
├── features/
│   ├── ads.py                   # Attention Dispersion Score
│   ├── cgc.py                   # Cross-modal Grounding Consistency
│   ├── extractor.py             # MS-COCO feature extraction
├── detection/
│   ├── train.py                 # Classifier training (XGB, RF, MLP)
│   └── evaluate.py              # Evaluation utilities
├── labeling/
│   ├── gpt4_labeler.py          # GPT-4o hallucination labeling
│   └── token_finder.py          # Token position finder
├── data/
│   ├── coco_loader.py           # MS-COCO data loader
│   └── pope_loader.py           # POPE data loader
├── scripts/
│   ├── generate_and_label.py    # Step 1: Generation + labeling
│   ├── extract_features.py      # Step 2: Feature extraction
│   ├── train_and_eval.py        # Step 3: Training + evaluation
├── run.sh                       # Full MS-COCO pipeline
```

## Citation

```bibtex
@inproceedings{nguyen2025tokengrounding,
  title={Beyond the Global Scores: Fine-Grained Token Grounding as a Robust Detector of LVLM Hallucinations},
  author={Nguyen, Tuan Dung and Ho, Minh Khoi and Chen, Qi and Xie, Yutong and Nguyen, Cam-Tu and Nguyen, Minh Khoi and Nguyen, Dang Huy Pham and van den Hengel, Anton and Verjans, Johan W. and Nguyen, Phi Le and Phan, Vu Minh Hieu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```

## Acknowledgement

This research is funded by Hanoi University of Science and Technology (HUST) under grant number T2024-TĐ-002.
Parts of our codebase build upon [SVAR](https://github.com/ZhangqiJiang07/middle_layers_indicating_hallucinations). We thank the authors for their contributions.

## Contact

For questions or issues, please contact [tuandung2812alt@gmail.com](mailto:tuandung2812alt@gmail.com).
