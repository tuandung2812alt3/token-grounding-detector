# Token Grounding Detector

Official implementation of **"Beyond the Global Scores: Fine-Grained Token Grounding as a Robust Detector of LVLM Hallucinations"** (CVPR 2026).

## Overview

We propose two patch-level structural metrics for detecting hallucinated tokens in Large Vision-Language Models:

- **Attention Dispersion Score (ADS)**: Measures how spatially focused a token's attention is over image patches. Hallucinated tokens produce diffuse, scattered attention; faithful tokens show compact focus.
- **Cross-modal Grounding Consistency (CGC)**: Measures semantic alignment between a token's hidden representation and the most relevant image patches. Hallucinated tokens show weak alignment with all regions.

These per-layer features are concatenated into a 2L-dimensional vector and fed to a lightweight classifier (XGB/RF/MLP) for token-level hallucination detection.

## Results

### MS-COCO Image Captioning (Table 2)

| Model | PR | RC | F1 | AUC |
|---|---|---|---|---|
| LLaVA-1.5-7B | 0.82 | 0.81 | 0.82 | 0.88 |
| Qwen2.5-VL-7B | 0.86 | 0.83 | 0.84 | 0.94 |
| InternVL2.5-8B | 0.88 | 0.91 | 0.90 | 0.94 |

### POPE (Table 3)

| Method | F1 | AUC |
|---|---|---|
| Ours | 0.41 | 0.75 |

## Installation

```bash
conda create -n tgd python=3.10 -y
conda activate tgd
pip install -r requirements.txt
```

For GPU support with PyTorch, follow the [official instructions](https://pytorch.org/get-started/locally/).

## Data Preparation

### MS-COCO 2014

Download the COCO 2014 validation set and annotations:

```bash
mkdir -p data/coco && cd data/coco
wget http://images.cocodataset.org/zips/val2014.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip val2014.zip
unzip annotations_trainval2014.zip
cd ../..
```

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

### POPE

Download the POPE benchmark files:

```bash
mkdir -p data/pope && cd data/pope
wget https://github.com/AoiDragon/POPE/raw/main/output/coco/coco_pope_random.json
wget https://github.com/AoiDragon/POPE/raw/main/output/coco/coco_pope_popular.json
wget https://github.com/AoiDragon/POPE/raw/main/output/coco/coco_pope_adversarial.json
cd ../..
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

**Supported models:** `llava_1_5_7b`, `internvl_2_5_8b`, `qwen2_5_vl_7b`

### Step 2: Extract ADS + CGC Features

Run prefix forward passes to compute per-layer ADS and CGC features for every labeled token:

```bash
python scripts/extract_features.py \
    --model llava_1_5_7b \
    --output-dir outputs/llava_1_5_7b \
    --device cuda \
    --resume
```

This produces `features.pkl` containing per-token feature vectors of dimension 2L (L layers × 2 metrics).

### Step 3: Train Classifiers + Evaluate

Train XGB, RF, and MLP classifiers on the extracted features:

```bash
python scripts/train_and_eval.py \
    --model llava_1_5_7b \
    --output-dir outputs/llava_1_5_7b
```

Results are saved to `outputs/llava_1_5_7b/results/`.

### POPE Evaluation

Run the full POPE pipeline (extraction + training + evaluation):

```bash
bash run_pope.sh
```

Or directly:

```bash
python scripts/pope_pipeline.py \
    --model llava_1_5_7b \
    --pope-dir data/pope \
    --coco-image-dir data/coco/val2014 \
    --output-dir outputs/pope/llava_1_5_7b \
    --device cuda
```

## Project Structure

```
token-grounding-detector/
├── configs/model_configs.yaml   # Model and training configurations
├── models/                      # LVLM wrappers (LLaVA, InternVL, Qwen)
├── features/
│   ├── ads.py                   # Attention Dispersion Score
│   ├── cgc.py                   # Cross-modal Grounding Consistency
│   ├── extractor.py             # MS-COCO feature extraction
│   └── pope_extractor.py        # POPE feature extraction
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
│   └── pope_pipeline.py         # POPE end-to-end pipeline
├── run.sh                       # Full MS-COCO pipeline
└── run_pope.sh                  # Full POPE pipeline
```

## Citation

```bibtex
@inproceedings{nguyen2025tokengrounding,
  title={Beyond the Global Scores: Fine-Grained Token Grounding as a Robust Detector of LVLM Hallucinations},
  author={Nguyen, Tuan Dung and Ho, Minh Khoi and Chen, Qi and Xie, Yutong and Nguyen, Cam-Tu and Nguyen, Minh Khoi and Nguyen, Dang Huy Pham and van den Hengel, Anton and Verjans, Johan W. and Nguyen, Phi Le and Phan, Vu Minh Hieu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}
```

## Acknowledgement

This research is funded by Hanoi University of Science and Technology (HUST) under grant number T2024-TĐ-002.

Parts of our codebase build upon [SVAR](https://github.com/ZhangqiJiang07/middle_layers_indicating_hallucinations). We thank the authors for releasing their code.

## Contact

For questions or issues, please contact [tuandung2812alt@gmail.com](mailto:tuandung2812alt@gmail.com).