# Dermatology Image Classification with Noisy Labels

## Problem Statement
This project addresses a real-world AI/ML problem where training data contains noisy labels, while validation data is clean and expert-verified. The goal is to build a robust image classification model that generalizes well despite label noise.

## Dataset
- Image Size: 28x28 grayscale
- Number of Classes: 7
- Training Data: Noisy labels
- Validation Data: Clean (gold standard)

## Approach
- Performed Exploratory Data Analysis (EDA) to understand class distribution
- Used a CNN-based architecture for image classification
- Applied **label smoothing** to reduce overfitting to incorrect labels
- Used **early stopping** based on validation loss
- Model selection was done using clean validation data

## Files
- `noisy_label_dermatology_classification.ipynb` – Full training pipeline
- `inference.py` – Live inference function for new test files
- `best_model.keras` – Best performing saved model
- `requirements.txt` – Dependencies

## Live Inference
```python
from inference import evaluate_on_new_data
evaluate_on_new_data("test_file.npz")
