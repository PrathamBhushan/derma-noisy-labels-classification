import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

def evaluate_on_new_data(npz_path, model_path="best_model.keras"):
    """
    Loads a new test .npz file and prints accuracy.
    """

    data = np.load(npz_path)

    X_test = data["X_test"]
    y_test = data["y_test"]

    # Preprocessing
    X_test = X_test / 255.0
    X_test = X_test[..., np.newaxis]
    y_test_cat = to_categorical(y_test, 7)

    # Load model
    model = tf.keras.models.load_model(model_path)

    loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"Test Accuracy: {acc:.4f}")

    return acc

