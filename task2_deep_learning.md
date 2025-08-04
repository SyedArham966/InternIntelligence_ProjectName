# Task 2 – CIFAR-10 CNN

This task trains a convolutional neural network on the **CIFAR-10** dataset (60 000 colour images of 10 object classes, each 32×32 pixels).

## Model architecture
- Input: 32×32×3 images
- Two 3×3 convolutional layers with ReLU activations, each followed by 2×2 max‑pooling
- Flatten layer and a 128‑unit dense layer with ReLU
- Dropout (0.5) for regularization
- Output: 10‑unit softmax layer for the CIFAR‑10 classes

The model uses the Adam optimizer and `sparse_categorical_crossentropy` loss.

## Training
- Training/validation/test split: 80 %/20 % from the 50 000 training images, plus the 10 000‑image test set
- Normalization: pixel values scaled to the [0, 1] range
- Batch size: 64
- Epochs: 5

## Results
Evaluated on the 10 000‑image test set:
- **Accuracy**: *(see console output; typically ~0.7 after 5 epochs)*
- The classification report and confusion matrix are printed by `task2_cifar10_cnn.py`.

Predicted and true labels for the test set are written to `predictions/cifar10_predictions.csv` for external accuracy checks.

## Usage
```bash
python task2_cifar10_cnn.py
```