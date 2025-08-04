# InternIntelligence_ML_Tuning_DeepLearning

This repository contains two machine‑learning exercises:

1. **Task 1 – Hyperparameter Tuning**: uses the Breast Cancer Wisconsin dataset to compare a Random Forest and SVC. Run with:
   ```bash
   python task1_hyperparameter_tuning.py
   ```
   Prediction files are written to `predictions/randomforest_predictions.csv` and `predictions/svc_predictions.csv`.
2. **Task 2 – Deep‑learning Model**: trains a small CNN on the scikit‑learn Digits dataset. Execute with:
   ```bash
   python task2_digits_cnn.py
   ```
   Test‑set predictions are saved to `predictions/digits_predictions.csv` for external verification.

Install dependencies with:
```bash
pip install -r requirements.txt
```

The reports `task1_hyperparameter_tuning.md` and `task2_deep_learning.md` summarize the methods and results.
