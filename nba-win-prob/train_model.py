import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve

# ── Settings ──────────────────────────────────────────────────────────────────
DATA_PATH   = "data/features.parquet"
MODEL_DIR   = "models"
MODEL_PATH  = os.path.join(MODEL_DIR, "best_model.pt")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.npy")

FEATURE_COLS = ["score_diff", "secs_left", "period", "home_foul_diff", "momentum"]
LABEL_COL    = "home_win"

BATCH_SIZE    = 256
EPOCHS        = 1000
LEARNING_RATE = 1e-3
VAL_SPLIT     = 0.2
RANDOM_SEED   = 42
PATIENCE      = 20


# ── 1. Dataset class ──────────────────────────────────────────────────────────
class WinProbDataset(Dataset):
    """Wraps numpy arrays into a PyTorch Dataset for use with DataLoader."""

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


# ── 2. Model architecture ─────────────────────────────────────────────────────
class WinProbModel(nn.Module):
    """
    MLP with 3 linear layers, BatchNorm, GELU, and Dropout.

    Input (5 features)
      → Linear(5→64) → BatchNorm → GELU → Dropout(0.2)
      → Linear(64→32) → BatchNorm → GELU
      → Linear(32→1) → Sigmoid
    """

    def __init__(self, input_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# ── 3. Load and prepare data ──────────────────────────────────────────────────
def load_data():
    """Loads features.parquet, splits into train/val, and fits a scaler."""
    print("[LOADING] Reading features.parquet...")
    df = pd.read_parquet(DATA_PATH).dropna(subset=FEATURE_COLS + [LABEL_COL])

    print(f"  Total rows   : {len(df):,}")
    print(f"  Home win rate: {df[LABEL_COL].mean():.1%}")

    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[LABEL_COL].values.astype(np.float32)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VAL_SPLIT, random_state=RANDOM_SEED
    )

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)

    np.save(SCALER_PATH, np.array([scaler.mean_, scaler.scale_]))
    print(f"  Scaler saved to {SCALER_PATH}")
    print(f"  Train rows   : {len(X_train):,}")
    print(f"  Val rows     : {len(X_val):,}")

    return X_train, X_val, y_train, y_val, scaler


# ── 4. Logistic regression baseline ──────────────────────────────────────────
def run_baseline(X_train, X_val, y_train, y_val):
    """Fits logistic regression as a performance floor for the neural net."""
    print("\n[BASELINE] Logistic Regression...")
    lr    = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)

    preds = lr.predict(X_val)
    probs = lr.predict_proba(X_val)[:, 1]
    acc   = accuracy_score(y_val, preds)
    loss  = log_loss(y_val, probs)

    print(f"  Accuracy : {acc:.4f}  ({acc:.1%})")
    print(f"  Log-loss : {loss:.4f}")
    print(f"  (Neural net needs to beat both of these)")
    return acc, loss


# ── 5. Training loop ──────────────────────────────────────────────────────────
def train(model, train_dl, val_dl, optimizer, criterion):
    """
    Trains for up to EPOCHS epochs; stops early if val loss doesn't improve
    for PATIENCE consecutive epochs. Saves the training curve graph on exit
    regardless of whether the loop finishes, early-stops, or is interrupted.
    """
    train_losses      = []
    val_losses        = []
    best_val_loss     = float("inf")
    epochs_no_improve = 0

    print(f"\n[TRAINING] {EPOCHS} epochs, batch size {BATCH_SIZE}, "
          f"patience {PATIENCE}...")
    print(f"  {'Epoch':>5}  {'Train Loss':>10}  {'Val Loss':>10}  {'Saved':>6}")
    print(f"  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*6}")

    try:
        for epoch in range(1, EPOCHS + 1):

            # ── Training phase ────────────────────────────────────────────────
            model.train()
            batch_losses = []
            for X_batch, y_batch in train_dl:
                optimizer.zero_grad()
                preds = model(X_batch)
                loss  = criterion(preds, y_batch)
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())

            train_loss = np.mean(batch_losses)
            train_losses.append(train_loss)

            # ── Validation phase ──────────────────────────────────────────────
            model.eval()
            val_batch_losses = []
            with torch.no_grad():
                for X_batch, y_batch in val_dl:
                    preds = model(X_batch)
                    loss  = criterion(preds, y_batch)
                    val_batch_losses.append(loss.item())

            val_loss = np.mean(val_batch_losses)
            val_losses.append(val_loss)

            # ── Save best model ───────────────────────────────────────────────
            saved = ""
            if val_loss < best_val_loss:
                best_val_loss     = val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), MODEL_PATH)
                saved = "saved"
            else:
                epochs_no_improve += 1

            print(
                f"  {epoch:>5}  {train_loss:>10.4f}  "
                f"{val_loss:>10.4f}  {saved:>6}"
            )

            # ── Early stopping ────────────────────────────────────────────────
            if epochs_no_improve >= PATIENCE:
                print(
                    f"\n  [EARLY STOP] Val loss didn't improve for "
                    f"{PATIENCE} epochs. Stopping at epoch {epoch}."
                )
                break

    except KeyboardInterrupt:
        print(
            "\n  [INTERRUPTED] Ctrl+C caught — "
            "saving graphs from completed epochs..."
        )

    print(f"\n  Best val loss: {best_val_loss:.4f}")
    print(f"  Model saved to {MODEL_PATH}")

    if train_losses:
        plot_training_curves(train_losses, val_losses)

    return train_losses, val_losses


# ── 6. Evaluation ─────────────────────────────────────────────────────────────
def evaluate(model, X_val, y_val):
    """
    Measures accuracy and log-loss, then saves calibration and distribution
    charts to models/evaluation.png.
    """
    model.eval()
    X_tensor = torch.tensor(X_val, dtype=torch.float32)
    with torch.no_grad():
        probs = model(X_tensor).numpy().flatten()

    preds = (probs >= 0.5).astype(int)
    acc   = accuracy_score(y_val, preds)
    loss  = log_loss(y_val, probs)

    print(f"\n[EVALUATION] Neural Network on validation set:")
    print(f"  Accuracy : {acc:.4f}  ({acc:.1%})")
    print(f"  Log-loss : {loss:.4f}")

    # ── Calibration curve ─────────────────────────────────────────────────────
    _, axes = plt.subplots(1, 2, figsize=(12, 4))

    frac_pos, mean_pred = calibration_curve(y_val, probs, n_bins=10)
    axes[0].plot(mean_pred, frac_pos, "s-", label="Model", color="#185FA5")
    axes[0].plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    axes[0].set_xlabel("Predicted probability")
    axes[0].set_ylabel("Actual win rate")
    axes[0].set_title("Calibration curve")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # ── Prediction distribution ───────────────────────────────────────────────
    axes[1].hist(probs, bins=50, color="#185FA5", edgecolor="white", alpha=0.8)
    axes[1].set_xlabel("Predicted win probability")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Distribution of predictions")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("models/evaluation.png", dpi=150)
    plt.show()
    print("  Evaluation chart saved to models/evaluation.png")

    return acc, loss


# ── 7. Training curve plot ────────────────────────────────────────────────────
def plot_training_curves(train_losses, val_losses):
    """Saves a train-vs-val loss chart to models/training_curves.png."""
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train loss", color="#185FA5")
    plt.plot(val_losses,   label="Val loss",   color="#993C1D")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.title("Training curves")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("models/training_curves.png", dpi=150)
    plt.show()
    print("  Training curves saved to models/training_curves.png")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    """Orchestrates data loading, baseline, training, and evaluation."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # 1. Load data
    X_train, X_val, y_train, y_val, _ = load_data()

    # 2. Baseline
    baseline_acc, baseline_loss = run_baseline(X_train, X_val, y_train, y_val)

    # 3. Build DataLoaders
    train_ds = WinProbDataset(X_train, y_train)
    val_ds   = WinProbDataset(X_val,   y_val)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    # 4. Build model, optimizer, loss function
    model     = WinProbModel(input_dim=len(FEATURE_COLS))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()

    # 5. Train (graphs saved inside train() on any exit)
    _, _ = train(model, train_dl, val_dl, optimizer, criterion)

    # 6. Load best saved weights and evaluate
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    nn_acc, nn_loss = evaluate(model, X_val, y_val)

    # 7. Final comparison
    print(f"\n[SUMMARY]")
    print(f"  {'':20s}  {'Accuracy':>10}  {'Log-loss':>10}")
    print(f"  {'-'*20}  {'-'*10}  {'-'*10}")
    print(f"  {'Logistic Regression':20s}  {baseline_acc:>10.4f}  "
          f"{baseline_loss:>10.4f}")
    print(f"  {'Neural Network':20s}  {nn_acc:>10.4f}  {nn_loss:>10.4f}")
    improvement = nn_acc - baseline_acc
    print(
        f"\n  Neural net "
        f"{'outperforms' if improvement > 0 else 'underperforms'} "
        f"baseline by {abs(improvement):.2%}"
    )

    if improvement <= 0:
        print("  Tip: try adding more features in build_features.py")
    else:
        print("  Model is ready — run server/app.py next!")


if __name__ == "__main__":
    main()
