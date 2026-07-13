import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, log_loss, brier_score_loss,
    confusion_matrix, ConfusionMatrixDisplay,
)

# ── Settings ──────────────────────────────────────────────────────────────────
DATA_PATH   = "data/matchup_features.parquet"
MODEL_DIR   = "models"
MODEL_PATH  = os.path.join(MODEL_DIR, "best_model.pt")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.npy")

FEATURE_COLS = [
    "win_pct_home",        "ppg_home",
    "opp_ppg_home",        "plus_minus_avg_home",   "last10_win_pct_home",
    "win_pct_away",        "ppg_away",
    "opp_ppg_away",        "plus_minus_avg_away",   "last10_win_pct_away",
    "home_split_win_pct",  "away_split_win_pct",    "net_rtg_diff",
    "home_team_away_ppg",  "away_team_home_ppg",
    "off_efficiency_home", "off_efficiency_away",
    "tov_rate_home",       "tov_rate_away",
]
LABEL_COL   = "home_win"
N_FEATURES  = len(FEATURE_COLS)

BATCH_SIZE    = 256
EPOCHS        = 500
LEARNING_RATE = 1e-4
VAL_SPLIT     = 0.2
RANDOM_SEED      = 42
LABEL_SMOOTHING  = 0.1   # smooths 0→0.05, 1→0.95 to prevent overconfidence
PATIENCE         = 20    # stop after this many epochs with no val-loss improvement

# Injury adjustments are NOT applied during training — they happen at inference
# time in injury_adjust.py. The model learns from clean historical team stats.


# ── 1. Dataset ────────────────────────────────────────────────────────────────
class MatchupDataset(Dataset):
    """Wraps scaled feature arrays and labels into a PyTorch Dataset."""

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


# ── 2. Model: three-hidden-layer MLP with BatchNorm ──────────────────────────
class WinProbModel(nn.Module):
    """Three hidden layers with BatchNorm, ReLU, and Dropout; outputs a win prob."""

    def __init__(self, input_dim=N_FEATURES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


# ── 3. Load and prepare data ──────────────────────────────────────────────────
def load_data():
    """Loads matchup features, scales them, and splits into train/val sets."""
    print("[LOADING] Reading matchup_features.parquet...")
    df = pd.read_parquet(DATA_PATH).dropna(subset=FEATURE_COLS + [LABEL_COL])

    print(f"  Total matchups : {len(df):,}")
    print(f"  Home win rate  : {df[LABEL_COL].mean():.1%}")

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
    print(f"  Train rows     : {len(X_train):,}")
    print(f"  Val rows       : {len(X_val):,}")

    return X_train, X_val, y_train, y_val


# ── 4. Training loop ──────────────────────────────────────────────────────────
def train(model, train_dl, val_dl, optimizer, criterion, scheduler):
    """
    Trains for EPOCHS epochs with ReduceLROnPlateau. Saves the best val-loss
    checkpoint to MODEL_PATH and plots training curves on any exit path.
    """
    train_losses      = []
    val_losses        = []
    best_val_loss     = float("inf")
    epochs_no_improve = 0

    print(f"\n[TRAINING] {EPOCHS} epochs, batch {BATCH_SIZE}, "
          f"lr {LEARNING_RATE}, patience {PATIENCE}...")
    print(f"  {'Epoch':>5}  {'Train Loss':>10}  {'Val Loss':>10}  "
          f"{'LR':>8}  {'Saved':>6}")
    print(f"  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*6}")

    try:
        for epoch in range(1, EPOCHS + 1):

            # ── Train ─────────────────────────────────────────────────────────
            model.train()
            batch_losses = []
            for X_batch, y_batch in train_dl:
                optimizer.zero_grad()
                preds    = model(X_batch)
                y_smooth = y_batch * (1 - LABEL_SMOOTHING) + LABEL_SMOOTHING / 2
                loss     = criterion(preds, y_smooth)
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())

            train_loss = np.mean(batch_losses)
            train_losses.append(train_loss)

            # ── Validate ──────────────────────────────────────────────────────
            model.eval()
            val_batch_losses = []
            with torch.no_grad():
                for X_batch, y_batch in val_dl:
                    preds = model(X_batch)
                    loss  = criterion(preds, y_batch)
                    val_batch_losses.append(loss.item())

            val_loss = np.mean(val_batch_losses)
            val_losses.append(val_loss)

            # ── LR scheduler step ─────────────────────────────────────────────
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            # ── Checkpoint ────────────────────────────────────────────────────
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
                f"{val_loss:>10.4f}  {current_lr:>8.2e}  {saved:>6}"
            )

            if epochs_no_improve >= PATIENCE:
                print(
                    f"\n  [EARLY STOP] No improvement for {PATIENCE} epochs. "
                    f"Stopping at epoch {epoch}."
                )
                break

    except KeyboardInterrupt:
        print("\n  [INTERRUPTED] Saving graphs from completed epochs...")

    print(f"\n  Best val loss : {best_val_loss:.4f}")
    print(f"  Model saved to {MODEL_PATH}")

    if train_losses:
        plot_training_curves(train_losses, val_losses)

    return train_losses, val_losses


# ── 5. Evaluation ─────────────────────────────────────────────────────────────
def evaluate(model, X_val, y_val):
    """
    Reports accuracy, log-loss, and Brier score. Saves a three-panel chart
    (calibration curve, prediction distribution, confusion matrix) and prints
    the learned feature coefficients for interpretability.
    """
    model.eval()
    X_tensor = torch.tensor(X_val, dtype=torch.float32)
    with torch.no_grad():
        probs = model(X_tensor).squeeze(1).numpy()

    preds = (probs >= 0.5).astype(int)
    acc   = accuracy_score(y_val,  preds)
    loss  = log_loss(y_val,        probs)
    brier = brier_score_loss(y_val, probs)

    print(f"\n[EVALUATION] Validation set:")
    print(f"  Accuracy    : {acc:.4f}  ({acc:.1%})")
    print(f"  Log-loss    : {loss:.4f}")
    print(f"  Brier score : {brier:.4f}  (lower = better, 0.25 = random)")

    # ── Layer weight norms (rough interpretability for MLP) ───────────────────
    print("\n  Layer weight L2 norms:")
    for name, param in model.named_parameters():
        if "weight" in name:
            print(f"    {name:<35} {param.data.norm():.4f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    _, axes = plt.subplots(1, 3, figsize=(18, 4))

    frac_pos, mean_pred = calibration_curve(y_val, probs, n_bins=10)
    axes[0].plot(mean_pred, frac_pos, "s-", label="Model", color="#185FA5")
    axes[0].plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    axes[0].set_xlabel("Predicted probability")
    axes[0].set_ylabel("Actual win rate")
    axes[0].set_title("Calibration curve")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].hist(probs, bins=40, color="#185FA5", edgecolor="white", alpha=0.8)
    axes[1].set_xlabel("Predicted home win probability")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Distribution of predictions")
    axes[1].grid(alpha=0.3)

    cm   = confusion_matrix(y_val, preds)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Away Win", "Home Win"]
    )
    disp.plot(ax=axes[2], colorbar=False, cmap="Blues")
    axes[2].set_title("Confusion matrix")

    plt.tight_layout()
    plt.savefig("models/evaluation.png", dpi=150)
    plt.show()
    print("\n  Evaluation chart saved to models/evaluation.png")

    return acc, loss, brier


# ── 6. Training curve plot ────────────────────────────────────────────────────
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
    """Orchestrates data loading, training, and evaluation."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # 1. Load data
    X_train, X_val, y_train, y_val = load_data()

    # 2. DataLoaders
    train_dl = DataLoader(
        MatchupDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True
    )
    val_dl = DataLoader(
        MatchupDataset(X_val, y_val), batch_size=BATCH_SIZE
    )

    # 3. Model, optimizer, loss, scheduler
    model     = WinProbModel()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4
    )
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    # 4. Train
    _, _ = train(model, train_dl, val_dl, optimizer, criterion, scheduler)

    # 5. Load best weights and evaluate
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    acc, loss, brier = evaluate(model, X_val, y_val)

    print(f"\n[SUMMARY]")
    print(f"  Accuracy    : {acc:.1%}")
    print(f"  Log-loss    : {loss:.4f}")
    print(f"  Brier score : {brier:.4f}")

    if acc >= 0.60:
        print("\n  Model is ready — run server/app.py next!")
    else:
        print("\n  Tip: try adding more features in build_features.py")


if __name__ == "__main__":
    main()
