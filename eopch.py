import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "logs")

def analyse_history(name):
    path = os.path.join(LOGS_DIR, f"hist_{name}.npz")
    hist = np.load(path)
    val_loss = hist["val_loss"]
    val_acc = hist["val_accuracy"]
    train_loss = hist["loss"]
    train_acc = hist["accuracy"]

    # The epoch (starting from 1) with the minimum validation loss
    best_idx = int(np.argmin(val_loss))
    best_epoch = best_idx + 1

    print(f"=== {name} model ===")
    print(f"  best epoch (val_loss): {best_epoch}")
    print(f"  train_loss: {train_loss[best_idx]:.4f}, "
          f"train_acc: {train_acc[best_idx]:.4f}")
    print(f"  val_loss:   {val_loss[best_idx]:.4f}, "
          f"val_acc: {val_acc[best_idx]:.4f}")
    print(f"  gap (train_acc - val_acc): "
          f"{train_acc[best_idx] - val_acc[best_idx]:.4f}")
    print()

if __name__ == "__main__":
    analyse_history("dense")
    analyse_history("cnn")
    analyse_history("parallel")
