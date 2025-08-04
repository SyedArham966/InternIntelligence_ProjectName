import pathlib
import sys

import numpy as np
import tensorflow as tf

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import task2_digits_cnn as t


def test_task2_accuracy_above_threshold():
    np.random.seed(42)
    tf.random.set_seed(42)
    X_train, X_val, X_test, y_train, y_val, y_test = t.load_data()
    model = t.build_model()
    t.train_model(model, X_train, y_train, X_val, y_val, epochs=20)
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    assert acc > 0.97
