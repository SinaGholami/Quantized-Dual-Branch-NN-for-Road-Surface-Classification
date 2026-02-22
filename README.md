# Quantized Dual-Branch NN for Road Surface Classification
A lightweight, edge-friendly **dual-branch quantized neural network (QNN)** for **road surface classification** using **smartphone inertial signals** (time-series) plus **engineered statistical features**.

This repository provides a reproducible pipeline for:
- data preparation
- feature engineering
- model training (quantized)
- evaluation (cross-validation)
- reporting metrics and plots

> **Target use-case:** real-time inference on low-power devices (edge AI).

## Highlights
- **Dual-branch architecture:** raw time-series branch + statistical-features branch
- **Quantization-aware design** for small model size and fast inference
- **Reproducible evaluation** with cross-validation
- **Clear metrics**: macro-F1, class-wise precision/recall, confusion matrix

## Dataset
This project uses the **AsphaltPavementType** dataset from the UCR/UEA time-series archive (or equivalent source).
The dataset is **not included** in the repository.

### Download & Prepare
1. Download the dataset from: (https://www.sciencedirect.com/science/article/pii/S0952197618301349)
2. Place files in:
   - `data/raw/`
3. Run preprocessing:
```bash
python src/data/make_dataset.py --input data/raw --output data/processed
````
If you prefer automatic download, enable:
```bash
python src/data/make_dataset.py --download --output data/processed
```

## Method

### Dual-Branch Model

* **Branch A (raw signals):** processes time-series windows (e.g., CNN/TCN style)
* **Branch B (features):** processes engineered statistics (mean, std, skewness, kurtosis, zero-crossing rate, energy, etc.)
* **Fusion:** concatenation + MLP classifier

## Results
Evaluation setup:

* **<<5>>-fold cross-validation**
* Metrics: **macro-F1**, accuracy, class-wise precision/recall

|     Metric | Value            |
| ---------: | :--------------- |
|   Macro F1 | <<0.939>>  |
|   Accuracy | <<0.940>>        |
| Model size | <<354 KB>> |

> For full details, see `results/metrics.json`.

## Installation

### Pip
```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

## Quickstart

### Train
```bash
python src/train.py --config configs/default.yaml
```

### Evaluate
```bash
python src/evaluate.py --config configs/default.yaml
```

Outputs:

* metrics: `results/metrics.json`
* plots: `results/figures/`

## Reproducibility
* Random seeds are fixed in `src/utils.py`.
* Cross-validation split strategy is controlled in `configs/default.yaml`.

To reproduce the main results:
```bash
python src/train.py --config configs/default.yaml --cv <<K>>
python src/evaluate.py --config configs/default.yaml
```

## Citation
If you use this code in academic work, please cite:

```bibtex
@article{<<(https://doi.org/10.1007/978-3-032-17174-0_25)>>,
  title   = {<<A Dual Branch Quantized Neural Network for Road Surface Classification Using Smartphone Sensors>>},
  author  = {<<Sina Gholami Fashkhami and Others>>},
  year    = {<<2026>>},
}
```

## License
MIT
