# InternIntelligence_ML_Tuning_DeepLearning

This repository contains two machine‑learning exercises:

1. **Task 1 – Hyperparameter Tuning**: uses the Breast Cancer Wisconsin dataset to compare a Random Forest and SVC. Run with:
   ```bash
   python task1_hyperparameter_tuning.py
   ```
   This script writes prediction files (`randomforest_predictions.csv` and `svc_predictions.csv`) containing true and predicted labels for the test set.
2. **Task 2 – Deep‑learning Model**: trains a small CNN on the scikit‑learn Digits dataset. Execute with:
   ```bash
   python task2_digits_cnn.py
   ```
   The script saves test‑set predictions to `task2_digits_predictions.csv` for external verification.

Install dependencies with:
```bash
pip install -r requirements.txt
```

The reports `task1_hyperparameter_tuning.md` and `task2_deep_learning.md` summarize the methods and results.
