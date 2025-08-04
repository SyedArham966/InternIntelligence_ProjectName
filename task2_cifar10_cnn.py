import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    InputLayer,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
)
from tensorflow.keras.optimizers import Adam


def load_data(val_size: float = 0.2):
    """Load the CIFAR-10 dataset and split into train/validation/test sets."""
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    # create validation split from training data
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_size,
        random_state=42,
        stratify=y_train,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_model():
    """Define a simple CNN for 32x32 RGB images."""
    model = Sequential(
        [
            InputLayer(input_shape=(32, 32, 3)),
            Conv2D(32, (3, 3), activation="relu", padding="same"),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation="relu", padding="same"),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.5),
            Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs: int = 5,
    batch_size: int = 64,
):
    """Train the CNN."""
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
    )
    return history


def evaluate_model(
    model,
    X_test,
    y_test,
    pred_path: str = os.path.join("predictions", "cifar10_predictions.csv"),
):
    """Evaluate the trained model on the test set, print metrics, and save predictions."""
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    os.makedirs(os.path.dirname(pred_path), exist_ok=True)
    pd.DataFrame({"actual": y_test.flatten(), "predicted": y_pred}).to_csv(
        pred_path, index=False
    )
    print(f"Saved predictions to {pred_path}")
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print(f"Test accuracy: {accuracy:.4f}")
    print("Classification report:\n", report)
    print("Confusion matrix:\n", cm)


def main():
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    model = build_model()
    epochs = int(os.getenv("EPOCHS", 5))
    train_model(model, X_train, y_train, X_val, y_val, epochs=epochs)
    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()
