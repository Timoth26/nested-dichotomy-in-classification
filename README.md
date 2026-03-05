# Nested Dichotomy in Image Classification

This repository contains a series of Jupyter notebook experiments focused on **multi-class image classification** with a **nested dichotomy** strategy and a `DecisionTreeClassifier` base learner.

The project investigates how model performance changes depending on:
- feature extraction method,
- data balancing strategy,
- hyperparameter selection method,
- implementation details of nested dichotomy.

## Project Goals

The notebooks answer four research questions:

1. **`exp1.ipynb`** – *How different feature extraction methods affect the effectiveness of classification?*
2. **`exp2.ipynb`** – *How data balancing methods affect model quality?*
3. **`exp3.ipynb`** – *How parameter selection affects model quality?*
4. **`exp4.ipynb`** – *How nested dichotomy implementation affects model quality?*

## Methods Used

- **Classifier**: `DecisionTreeClassifier` (scikit-learn)
- **Multi-class strategy**: Nested Dichotomy
- **Cross-validation**: K-Fold (typically 5-fold)
- **Feature extraction**:
	- `VGG16`
	- `InceptionV3`
	- `MobileNetV2`
- **Balancing methods** (selected experiments):
	- `SMOTE`
	- `RandomOverSampler`
	- `RandomUnderSampler`
	- `TomekLinks`
	- `SMOTETomek`
- **Hyperparameter optimization** (selected experiments):
	- `GridSearchCV`
	- `RandomizedSearchCV`
	- `BayesSearchCV`
- **Evaluation metrics**:
	- Accuracy
	- Precision (weighted)
	- Recall (weighted)
	- F1-score (weighted)
	- Per-class accuracy

## Repository Structure

```text
.
├── exp1.ipynb
├── exp2.ipynb
├── exp3.ipynb
├── exp4.ipynb
├── requirements.txt
└── README.md
```

## Dataset Layout

The notebooks expect image data in a local directory structure similar to:

```text
data/
├── train/
│   ├── class_1/
│   ├── class_2/
│   └── ...
├── test/
│   ├── class_1/
│   ├── class_2/
│   └── ...
├── validation/
│   ├── class_1/
│   ├── class_2/
│   └── ...
└── augmented/
    ├── class_1/
    ├── class_2/
    └── ...
```

Each class folder should contain `.jpg` images (some notebooks also include `.jpeg` files from `augmented/`).

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure the dataset is placed in the expected `data/` subfolders.

## Running Experiments

Open the notebooks in JupyterLab/VS Code and run cells sequentially:

- `exp1.ipynb` – feature extraction comparison
- `exp2.ipynb` – class balancing comparison
- `exp3.ipynb` – hyperparameter search comparison
- `exp4.ipynb` – nested dichotomy implementation variants

The notebooks generate printed metric reports and visualizations (e.g., bar plots, radar charts, confusion-related outputs).

## Reproducibility Notes

- K-Fold splitting is configured with a fixed random seed (`random_state=42`) in the notebooks.
- Some model/training operations may still include non-deterministic behavior depending on library/hardware configuration.
- Results can vary if the dataset composition or augmentation content differs.

## Requirements

All Python dependencies are listed in `requirements.txt`.
Key libraries include:
- TensorFlow / Keras
- scikit-learn
- imbalanced-learn
- scikit-optimize
- NumPy, pandas, SciPy, matplotlib