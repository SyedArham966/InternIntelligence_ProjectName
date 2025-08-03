# Task 1 – Hyperparameter Tuning on the Breast Cancer Dataset

This experiment uses scikit-learn's Breast Cancer Wisconsin diagnostic dataset. Features were scaled using `StandardScaler` and the data was split into 80% training and 20% test sets.

## Models and Search Space

Two classifiers were tuned using `GridSearchCV` with 5-fold cross validation and F1 score for refitting.

### RandomForestClassifier
- `n_estimators`: [50, 100]
- `max_depth`: [None, 5, 10]
- `min_samples_split`: [2, 5]

### Support Vector Classifier
- `C`: [0.1, 1, 10]
- `kernel`: ['linear', 'rbf']
- `gamma`: ['scale', 'auto']

## Results

| Model | Best Parameters | Accuracy | Precision | Recall | F1 |
|-------|-----------------|----------|-----------|--------|----|
| RandomForest | `{'clf__max_depth': None, 'clf__min_samples_split': 5, 'clf__n_estimators': 50}` | 0.947 | 0.958 | 0.958 | 0.958 |
| SVC | `{'clf__C': 0.1, 'clf__gamma': 'scale', 'clf__kernel': 'linear'}` | 0.982 | 0.986 | 0.986 | 0.986 |

The Support Vector Classifier with a linear kernel achieved the best performance on the hold‑out test set, surpassing the Random Forest in all evaluated metrics.

Prediction outputs for the test set are saved in `randomforest_predictions.csv` and `svc_predictions.csv` for external accuracy verification.
