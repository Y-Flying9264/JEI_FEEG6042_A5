"""
train.py

Train three models on MNIST using a local npz file:
- Sequential dense network
- Sequential CNN
- Parallel dense network with two branches

Weights and training logs are saved under ./weights and ./logs.
"""

import os
import numpy as np
import tensorflow as tf

from models import (
    build_dense_model,
    build_cnn_model,
    build_parallel_dense_model,
)


# Config
RANDOM_SEED = 6042      # Module Code: FEEG6042
EPOCHS = 20
BATCH_SIZE = 128
VAL_SPLIT = 0.1

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "mnist.npz")
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
logs_dir_name = "logs"
LOGS_DIR = os.path.join(BASE_DIR, logs_dir_name)


def load_mnist_local():
    """Load MNIST from a local npz file and prepare inputs for CNN and dense models."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"MNIST npz not found at: {DATA_PATH}")

    data = np.load(DATA_PATH)
    x_train = data["x_train"]
    y_train = data["y_train"]
    x_test = data["x_test"]
    y_test = data["y_test"]

    print("Original shapes:")
    print("  x_train:", x_train.shape, x_train.dtype)
    print("  y_train:", y_train.shape, y_train.dtype)
    print("  x_test :", x_test.shape, x_test.dtype)
    print("  y_test :", y_test.shape, y_test.dtype)

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train_cnn = np.expand_dims(x_train, -1)
    x_test_cnn = np.expand_dims(x_test, -1)

    x_train_flat = x_train.reshape(-1, 28 * 28)
    x_test_flat = x_test.reshape(-1, 28 * 28)

    print("\nPrepared shapes:")
    print("  x_train_cnn :", x_train_cnn.shape)
    print("  x_train_flat:", x_train_flat.shape)

    return (
        x_train_cnn,
        x_train_flat,
        y_train,
        x_test_cnn,
        x_test_flat,
        y_test,
    )


def main():
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    (
        x_train_cnn,
        x_train_flat,
        y_train,
        x_test_cnn,
        x_test_flat,
        y_test,
    ) = load_mnist_local()

    fit_kwargs = dict(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VAL_SPLIT,
        verbose=2,
    )

    summary_lines = []

    # Dense
    print("\n===== Training dense model =====")
    dense_model = build_dense_model()
    dense_params = dense_model.count_params()
    print(f"Dense model parameters: {dense_params}")

    hist_dense = dense_model.fit(
        x_train_flat,
        y_train,
        **fit_kwargs,
    )

    dense_weights_path = os.path.join(WEIGHTS_DIR, "dense_mnist.weights.h5")
    dense_model.save_weights(dense_weights_path)

    np.savez(
        os.path.join(LOGS_DIR, "hist_dense.npz"),
        **hist_dense.history,
    )

    dense_test = dense_model.evaluate(x_test_flat, y_test, verbose=0)
    print(f"Dense model test accuracy: {dense_test[1]:.4f}")

    summary_lines.append("=== Dense model ===")
    summary_lines.append(f"#Params: {dense_params}")
    summary_lines.append(
        f"Final train loss: {hist_dense.history['loss'][-1]:.4f}, "
        f"train acc: {hist_dense.history['accuracy'][-1]:.4f}"
    )
    summary_lines.append(
        f"Final val loss:   {hist_dense.history['val_loss'][-1]:.4f}, "
        f"val acc: {hist_dense.history['val_accuracy'][-1]:.4f}"
    )
    summary_lines.append(
        f"Test loss: {dense_test[0]:.4f}, test acc: {dense_test[1]:.4f}"
    )
    summary_lines.append("")

    # CNN
    print("\n===== Training CNN model =====")
    cnn_model = build_cnn_model()
    cnn_params = cnn_model.count_params()
    print(f"CNN model parameters: {cnn_params}")

    hist_cnn = cnn_model.fit(
        x_train_cnn,
        y_train,
        **fit_kwargs,
    )

    cnn_weights_path = os.path.join(WEIGHTS_DIR, "cnn_mnist.weights.h5")
    cnn_model.save_weights(cnn_weights_path)

    np.savez(
        os.path.join(LOGS_DIR, "hist_cnn.npz"),
        **hist_cnn.history,
    )

    cnn_test = cnn_model.evaluate(x_test_cnn, y_test, verbose=0)
    print(f"CNN model test accuracy: {cnn_test[1]:.4f}")

    summary_lines.append("=== CNN model ===")
    summary_lines.append(f"#Params: {cnn_params}")
    summary_lines.append(
        f"Final train loss: {hist_cnn.history['loss'][-1]:.4f}, "
        f"train acc: {hist_cnn.history['accuracy'][-1]:.4f}"
    )
    summary_lines.append(
        f"Final val loss:   {hist_cnn.history['val_loss'][-1]:.4f}, "
        f"val acc: {hist_cnn.history['val_accuracy'][-1]:.4f}"
    )
    summary_lines.append(
        f"Test loss: {cnn_test[0]:.4f}, test acc: {cnn_test[1]:.4f}"
    )
    summary_lines.append("")

    # Parallel Dense
    print("\n===== Training parallel dense model =====")
    parallel_model = build_parallel_dense_model()
    parallel_params = parallel_model.count_params()
    print(f"Parallel dense model parameters: {parallel_params}")

    hist_parallel = parallel_model.fit(
        x_train_flat,
        y_train,
        **fit_kwargs,
    )

    parallel_weights_path = os.path.join(
        WEIGHTS_DIR, "parallel_mnist.weights.h5"
    )
    parallel_model.save_weights(parallel_weights_path)

    np.savez(
        os.path.join(LOGS_DIR, "hist_parallel.npz"),
        **hist_parallel.history,
    )

    parallel_test = parallel_model.evaluate(x_test_flat, y_test, verbose=0)
    print(f"Parallel model test accuracy: {parallel_test[1]:.4f}")

    summary_lines.append("=== Parallel dense model ===")
    summary_lines.append(f"#Params: {parallel_params}")
    summary_lines.append(
        f"Final train loss: {hist_parallel.history['loss'][-1]:.4f}, "
        f"train acc: {hist_parallel.history['accuracy'][-1]:.4f}"
    )
    summary_lines.append(
        f"Final val loss:   {hist_parallel.history['val_loss'][-1]:.4f}, "
        f"val acc: {hist_parallel.history['val_accuracy'][-1]:.4f}"
    )
    summary_lines.append(
        f"Test loss: {parallel_test[0]:.4f}, "
        f"test acc: {parallel_test[1]:.4f}"
    )
    summary_lines.append("")

    # Training Summary
    summary_path = os.path.join(LOGS_DIR, "train_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"RANDOM_SEED = {RANDOM_SEED}\n")
        f.write(f"EPOCHS      = {EPOCHS}\n")
        f.write(f"BATCH_SIZE  = {BATCH_SIZE}\n")
        f.write(f"VAL_SPLIT   = {VAL_SPLIT}\n\n")
        for line in summary_lines:
            f.write(line + "\n")

    print("\n=== Training complete ===")
    print(f"Weights directory: {WEIGHTS_DIR}")
    print(f"Logs directory:    {LOGS_DIR}")
    print(f"Training summary:  {summary_path}")


if __name__ == "__main__":
    main()
