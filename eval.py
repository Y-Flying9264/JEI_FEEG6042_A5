"""
eval.py

Load trained weights for the three MNIST models and:
- Evaluate test performance
- Plot validation accuracy and training loss curves
- Compute and plot confusion matrices

Results are written to ./results as text files and PNG figures.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from models import (
    build_dense_model,
    build_cnn_model,
    build_parallel_dense_model,
)


# Config
RANDOM_SEED = 6042
BATCH_SIZE = 128

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "mnist.npz")
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def load_mnist_local():
    """Load MNIST from a local npz file and prepare inputs for CNN and dense models."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"MNIST npz not found at: {DATA_PATH}")

    data = np.load(DATA_PATH)
    x_train = data["x_train"]
    y_train = data["y_train"]
    x_test = data["x_test"]
    y_test = data["y_test"]

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train_cnn = np.expand_dims(x_train, -1)
    x_test_cnn = np.expand_dims(x_test, -1)

    x_train_flat = x_train.reshape(-1, 28 * 28)
    x_test_flat = x_test.reshape(-1, 28 * 28)

    return (
        x_train_cnn,
        x_train_flat,
        y_train,
        x_test_cnn,
        x_test_flat,
        y_test,
    )


def compute_confusion_matrix(y_true, y_pred, num_classes=10):
    """Simple confusion matrix implementation using NumPy only."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10

def plot_confusion_matrix(cm, path, title):
    """Plot and save a confusion matrix heatmap with labels."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")

    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    ticks = np.arange(cm.shape[0])
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(ticks)
    ax.set_yticklabels(ticks)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    # Add numbers in each cell
    max_val = cm.max() if cm.max() > 0 else 1
    thresh = max_val / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            colour = "white" if value > thresh else "black"
            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                color=colour,
                fontsize=7,
            )

    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main():
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    (
        x_train_cnn,
        x_train_flat,
        y_train,
        x_test_cnn,
        x_test_flat,
        y_test,
    ) = load_mnist_local()


    # Rebuild models and load weights
    dense_model = build_dense_model()
    cnn_model = build_cnn_model()
    parallel_model = build_parallel_dense_model()

    dense_weights_path = os.path.join(WEIGHTS_DIR, "dense_mnist.weights.h5")
    cnn_weights_path = os.path.join(WEIGHTS_DIR, "cnn_mnist.weights.h5")
    parallel_weights_path = os.path.join(
        WEIGHTS_DIR, "parallel_mnist.weights.h5"
    )

    if not all(
        os.path.exists(p)
        for p in (dense_weights_path, cnn_weights_path, parallel_weights_path)
    ):
        raise FileNotFoundError(
            "Weight files not found. Please run train.py before eval.py."
        )

    dense_model.load_weights(dense_weights_path)
    cnn_model.load_weights(cnn_weights_path)
    parallel_model.load_weights(parallel_weights_path)


    # Test evaluation
    dense_test = dense_model.evaluate(x_test_flat, y_test, verbose=0)
    cnn_test = cnn_model.evaluate(x_test_cnn, y_test, verbose=0)
    parallel_test = parallel_model.evaluate(x_test_flat, y_test, verbose=0)

    print("\n===== Test results on MNIST =====")
    print(f"Dense    - loss: {dense_test[0]:.4f}, acc: {dense_test[1]:.4f}")
    print(f"CNN      - loss: {cnn_test[0]:.4f}, acc: {cnn_test[1]:.4f}")
    print(
        f"Parallel - loss: {parallel_test[0]:.4f}, "
        f"acc: {parallel_test[1]:.4f}"
    )

    results_txt = os.path.join(RESULTS_DIR, "test_results.txt")
    with open(results_txt, "w", encoding="utf-8") as f:
        f.write(f"RANDOM_SEED = {RANDOM_SEED}\n\n")
        f.write("===== Test results on MNIST =====\n")
        f.write(
            f"Dense    - loss: {dense_test[0]:.4f}, "
            f"acc: {dense_test[1]:.4f}\n"
        )
        f.write(
            f"CNN      - loss: {cnn_test[0]:.4f}, "
            f"acc: {cnn_test[1]:.4f}\n"
        )
        f.write(
            f"Parallel - loss: {parallel_test[0]:.4f}, "
            f"acc: {parallel_test[1]:.4f}\n"
        )

    print(f"\nTest results written to: {results_txt}")


    # Load training logs and plot curves
    hist_dense = np.load(os.path.join(LOGS_DIR, "hist_dense.npz"))
    hist_cnn = np.load(os.path.join(LOGS_DIR, "hist_cnn.npz"))
    hist_parallel = np.load(os.path.join(LOGS_DIR, "hist_parallel.npz"))

    def to_dict(hist_npz):
        return {k: hist_npz[k] for k in hist_npz.files}

    h_dense = to_dict(hist_dense)
    h_cnn = to_dict(hist_cnn)
    h_parallel = to_dict(hist_parallel)

    epochs = range(1, len(h_dense["accuracy"]) + 1)

    # Validation accuracy comparison
    fig_acc, ax_acc = plt.subplots(figsize=(7, 5))
    ax_acc.plot(
        epochs,
        h_dense["val_accuracy"],
        label="Dense",
        linewidth=2.0,
        marker="o",
        markersize=3,
    )
    ax_acc.plot(
        epochs,
        h_cnn["val_accuracy"],
        label="CNN",
        linewidth=2.0,
        marker="s",
        markersize=3,
    )
    ax_acc.plot(
        epochs,
        h_parallel["val_accuracy"],
        label="Parallel dense",
        linewidth=2.0,
        marker="^",
        markersize=3,
    )
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Validation accuracy")
    ax_acc.set_title("Validation accuracy comparison")
    ax_acc.grid(True, linestyle="--", alpha=0.4)
    ax_acc.legend()
    fig_acc.tight_layout()
    acc_fig_path = os.path.join(RESULTS_DIR, "val_accuracy_comparison.png")
    fig_acc.savefig(acc_fig_path, dpi=300)
    plt.close(fig_acc)

    # Training loss comparison
    fig_loss, ax_loss = plt.subplots(figsize=(7, 5))
    ax_loss.plot(
        epochs,
        h_dense["loss"],
        label="Dense",
        linewidth=2.0,
        marker="o",
        markersize=3,
    )
    ax_loss.plot(
        epochs,
        h_cnn["loss"],
        label="CNN",
        linewidth=2.0,
        marker="s",
        markersize=3,
    )
    ax_loss.plot(
        epochs,
        h_parallel["loss"],
        label="Parallel dense",
        linewidth=2.0,
        marker="^",
        markersize=3,
    )
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Training loss")
    ax_loss.set_title("Training loss comparison")
    ax_loss.grid(True, linestyle="--", alpha=0.4)
    ax_loss.legend()
    fig_loss.tight_layout()
    loss_fig_path = os.path.join(
        RESULTS_DIR, "training_loss_comparison.png"
    )
    fig_loss.savefig(loss_fig_path, dpi=300)
    plt.close(fig_loss)

    print("Validation accuracy plot:", acc_fig_path)
    print("Training loss plot:      ", loss_fig_path)


    # Confusion matrices
    print("\nComputing confusion matrices...")

    y_pred_dense = np.argmax(
        dense_model.predict(x_test_flat, batch_size=BATCH_SIZE, verbose=0),
        axis=1,
    )
    y_pred_cnn = np.argmax(
        cnn_model.predict(x_test_cnn, batch_size=BATCH_SIZE, verbose=0),
        axis=1,
    )
    y_pred_parallel = np.argmax(
        parallel_model.predict(x_test_flat, batch_size=BATCH_SIZE, verbose=0),
        axis=1,
    )

    cm_dense = compute_confusion_matrix(y_test, y_pred_dense, num_classes=10)
    cm_cnn = compute_confusion_matrix(y_test, y_pred_cnn, num_classes=10)
    cm_parallel = compute_confusion_matrix(
        y_test, y_pred_parallel, num_classes=10
    )

    # Save numeric matrices
    np.savetxt(
        os.path.join(RESULTS_DIR, "confusion_dense.txt"),
        cm_dense,
        fmt="%d",
    )
    np.savetxt(
        os.path.join(RESULTS_DIR, "confusion_cnn.txt"),
        cm_cnn,
        fmt="%d",
    )
    np.savetxt(
        os.path.join(RESULTS_DIR, "confusion_parallel.txt"),
        cm_parallel,
        fmt="%d",
    )

    # Save heatmaps
    plot_confusion_matrix(
        cm_dense,
        os.path.join(RESULTS_DIR, "confusion_dense.png"),
        "Confusion matrix - Dense model",
    )
    plot_confusion_matrix(
        cm_cnn,
        os.path.join(RESULTS_DIR, "confusion_cnn.png"),
        "Confusion matrix - CNN model",
    )
    plot_confusion_matrix(
        cm_parallel,
        os.path.join(RESULTS_DIR, "confusion_parallel.png"),
        "Confusion matrix - Parallel dense model",
    )

    print("Confusion matrices written to:")
    print("  ", os.path.join(RESULTS_DIR, "confusion_dense.txt"))
    print("  ", os.path.join(RESULTS_DIR, "confusion_cnn.txt"))
    print("  ", os.path.join(RESULTS_DIR, "confusion_parallel.txt"))


if __name__ == "__main__":
    main()
