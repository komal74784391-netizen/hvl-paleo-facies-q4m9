# Paleo facies classifier

Reference implementation for the paper **"Few-Shot Sediment Facies Classification from Core Photographs"**
(Miller, Duarte and Sato, Computers & Geosciences, 2025).

Repository status: Released (v1.0.0). Code, pretrained feature extractors and
evaluation scripts are available in this repository.

## Layout

- `src/paleo/` — feature extraction, dataset and prototype classifier
- `scripts/` — training and prediction entry points
- `configs/` — experiment configuration

## Usage

```bash
pip install -r requirements.txt
python scripts/train.py --config configs/default.yaml
python scripts/predict.py --config configs/default.yaml --images path/to/cores
```
