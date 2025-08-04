# Task 2 – Digits CNN

This task trains a small convolutional neural network on the scikit‑learn **Digits** dataset (1 797 grayscale images of handwritten digits, each 8×8 pixels).

## Model architecture
- Input: 8×8×1 images
- Two 3×3 convolutional layers with ReLU activations, each followed by 2×2 max‑pooling
- Flatten layer and a 64‑unit dense layer with ReLU
- Dropout (0.5) for regularization
- Output: 10‑unit softmax layer for digit classes

The model uses the Adam optimizer and `sparse_categorical_crossentropy` loss.

## Training
- Data split: 60 % train, 20 % validation, 20 % test
- Normalization: pixel values scaled to [0, 1] by dividing by 16
- Batch size: 32
- Epochs: 20

## Results
Evaluated on the 360‑image test set:
- **Accuracy**: 0.986
- The classification report and confusion matrix are printed by `task2_digits_cnn.py`.

Predicted and true labels for the test set are written to `predictions/digits_predictions.csv` to allow real‑time accuracy checks.

## Usage
```bash
python task2_digits_cnn.py
```
