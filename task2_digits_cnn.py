import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam


def load_data(test_size: float = 0.2, val_size: float = 0.25):
    """Load the digits dataset and split into train/validation/test sets."""
    digits = load_digits()
    X = digits.images / 16.0  # normalize pixel values
    X = np.expand_dims(X, -1)  # shape (n_samples, 8, 8, 1)
    y = digits.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=42, stratify=y_train
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_model():
    """Define a simple convolutional neural network for 8x8 digit images."""
    model = Sequential(
        [
            InputLayer(input_shape=(8, 8, 1)),
            Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(64, activation="relu"),
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


def train_model(model, X_train, y_train, X_val, y_val, epochs: int = 20, batch_size: int = 32):
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


def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model on the test set and print metrics."""
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    y_pred = np.argmax(model.predict(X_test), axis=1)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print(f"Test accuracy: {accuracy:.4f}")
    print("Classification report:\n", report)
    print("Confusion matrix:\n", cm)


def main():
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    model = build_model()
    train_model(model, X_train, y_train, X_val, y_val)
    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()
