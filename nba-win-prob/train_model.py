import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, log_loss, confusion_matrix,
    ConfusionMatrixDisplay, brier_score_loss,
)
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve

# ── Settings ──────────────────────────────────────────────────────────────────
DATA_PATH   = "data/sequences.npz"
MODEL_DIR   = "models"
MODEL_PATH  = os.path.join(MODEL_DIR, "best_model.pt")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.npy")

N_FEATURES    = 5
HIDDEN_SIZE   = 128
NUM_LAYERS    = 2
DROPOUT       = 0.3

BATCH_SIZE    = 128
EPOCHS        = 30
LEARNING_RATE = 1e-3
VAL_SPLIT     = 0.2
RANDOM_SEED   = 42
PATIENCE      = 20
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── 1. Dataset ────────────────────────────────────────────────────────────────
class WinProbDataset(Dataset):
    """Wraps padded game sequences, labels, and true lengths for the LSTM."""

    def __init__(self, X, y, lengths):
        self.X       = torch.tensor(X,       dtype=torch.float32)
        self.y       = torch.tensor(y,       dtype=torch.float32)
        self.lengths = torch.tensor(lengths, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i], self.lengths[i]


# ── 2. Model ──────────────────────────────────────────────────────────────────
class WinProbModel(nn.Module):
    """
    Two-layer LSTM followed by a single linear classifier.

    Input  : (batch, seq_len, 5) padded sequences
    LSTM   : hidden_size=128, num_layers=2, dropout=0.3
    Output : (batch, 1) win probability via Sigmoid

    The LSTM reads each game play-by-play in order. Only the hidden state at
    the last real play (not padding) is passed to the classifier, so the model
    learns from the full temporal context of each game.
    """

    def __init__(
        self,
        input_size  = N_FEATURES,
        hidden_size = HIDDEN_SIZE,
        num_layers  = NUM_LAYERS,
        dropout     = DROPOUT,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, lengths):
        # Pack so the LSTM skips padding positions entirely
        packed        = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        # hidden: (num_layers, batch, hidden_size) — take the top layer
        last_hidden   = hidden[-1]
        return self.classifier(last_hidden)


# ── 3. Load and prepare data ──────────────────────────────────────────────────
def load_data():
    """
    Loads sequences.npz, splits by game into train/val, and fits a scaler
    on the non-padded training timesteps only.
    """
    print("[LOADING] Reading sequences.npz...")
    data    = np.load(DATA_PATH)
    X       = data["X"]          # (n_games, max_seq_len, n_features)
    y       = data["y"]          # (n_games,)
    lengths = data["lengths"]    # (n_games,)

    n_games = len(y)
    print(f"  Games        : {n_games:,}")
    print(f"  X shape      : {X.shape}")
    print(f"  Home win rate: {y.mean():.1%}")

    # Split by game index so no game leaks across train/val
    idx = np.arange(n_games)
    train_idx, val_idx = train_test_split(
        idx, test_size=VAL_SPLIT, random_state=RANDOM_SEED
    )

    X_train, X_val         = X[train_idx],       X[val_idx]
    y_train, y_val         = y[train_idx],        y[val_idx]
    lengths_train, lengths_val = lengths[train_idx], lengths[val_idx]

    # Fit scaler on non-padded training timesteps only
    train_mask = np.zeros(
        (len(X_train), X_train.shape[1]), dtype=bool
    )
    for i, l in enumerate(lengths_train):
        train_mask[i, :l] = True

    scaler  = StandardScaler()
    scaler.fit(X_train[train_mask])

    # Apply scaling to all timesteps (padding is ignored by pack_padded_sequence)
    X_train = scaler.transform(
        X_train.reshape(-1, N_FEATURES)
    ).reshape(X_train.shape)
    X_val = scaler.transform(
        X_val.reshape(-1, N_FEATURES)
    ).reshape(X_val.shape)

    np.save(SCALER_PATH, np.array([scaler.mean_, scaler.scale_]))
    print(f"  Scaler saved to {SCALER_PATH}")
    print(f"  Train games  : {len(X_train):,}")
    print(f"  Val games    : {len(X_val):,}")

    return X_train, X_val, y_train, y_val, lengths_train, lengths_val


# ── 4. Logistic regression baseline ──────────────────────────────────────────
def run_baseline(X_train, X_val, y_train, y_val, lengths_train, lengths_val):
    """
    Fits logistic regression on mean-pooled features per game as a baseline.
    Mean-pooling collapses each game's sequence into one feature vector.
    """
    print("\n[BASELINE] Logistic Regression (mean-pooled features)...")

    def mean_pool(X, lengths):
        pooled = np.zeros((len(X), N_FEATURES), dtype=np.float32)
        for i, l in enumerate(lengths):
            pooled[i] = X[i, :l].mean(axis=0)
        return pooled

    X_train_pooled = mean_pool(X_train, lengths_train)
    X_val_pooled   = mean_pool(X_val,   lengths_val)

    lr    = LogisticRegression(max_iter=1000)
    lr.fit(X_train_pooled, y_train)

    preds = lr.predict(X_val_pooled)
    probs = lr.predict_proba(X_val_pooled)[:, 1]
    acc   = accuracy_score(y_val, preds)
    loss  = log_loss(y_val, probs)

    print(f"  Accuracy : {acc:.4f}  ({acc:.1%})")
    print(f"  Log-loss : {loss:.4f}")
    print(f"  (LSTM needs to beat both of these)")
    return acc, loss


# ── 5. Training loop ──────────────────────────────────────────────────────────
def train(model, train_dl, val_dl, optimizer, criterion):
    """
    Trains for up to EPOCHS epochs with early stopping after PATIENCE epochs
    of no val-loss improvement. Saves the training curve on any exit.
    """
    train_losses      = []
    val_losses        = []
    best_val_loss     = float("inf")
    epochs_no_improve = 0

    print(f"\n[TRAINING] {EPOCHS} epochs, batch {BATCH_SIZE}, "
          f"patience {PATIENCE}...")
    print(f"  {'Epoch':>5}  {'Train Loss':>10}  {'Val Loss':>10}  {'Saved':>6}")
    print(f"  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*6}")

    try:
        for epoch in range(1, EPOCHS + 1):

            # ── Training phase ────────────────────────────────────────────────
            model.train()
            batch_losses = []
            for X_batch, y_batch, len_batch in train_dl:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                optimizer.zero_grad()
                preds = model(X_batch, len_batch).squeeze(1)
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
                for X_batch, y_batch, len_batch in val_dl:
                    X_batch = X_batch.to(DEVICE)
                    y_batch = y_batch.to(DEVICE)
                    preds = model(X_batch, len_batch).squeeze(1)
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

            if epochs_no_improve >= PATIENCE:
                print(
                    f"\n  [EARLY STOP] No improvement for {PATIENCE} epochs. "
                    f"Stopping at epoch {epoch}."
                )
                break

    except KeyboardInterrupt:
        print(
            "\n  [INTERRUPTED] Ctrl+C caught — "
            "saving graphs from completed epochs..."
        )

    print(f"\n  Best val loss : {best_val_loss:.4f}")
    print(f"  Model saved to {MODEL_PATH}")

    if train_losses:
        plot_training_curves(train_losses, val_losses)

    return train_losses, val_losses


# ── 6. Evaluation ─────────────────────────────────────────────────────────────
def evaluate(model, X_val, y_val, lengths_val):
    """
    Reports accuracy, log-loss, and Brier score, then saves a three-panel
    evaluation chart (calibration curve, prediction distribution, confusion
    matrix) to models/evaluation.png.
    """
    model.eval()
    X_tensor   = torch.tensor(X_val,       dtype=torch.float32)
    len_tensor = torch.tensor(lengths_val, dtype=torch.long)

    with torch.no_grad():
        probs = model(X_tensor.to(DEVICE), len_tensor).squeeze(1).cpu().numpy()

    preds = (probs >= 0.5).astype(int)
    acc   = accuracy_score(y_val,  preds)
    loss  = log_loss(y_val,        probs)
    brier = brier_score_loss(y_val, probs)

    print(f"\n[EVALUATION] LSTM on validation set:")
    print(f"  Accuracy    : {acc:.4f}  ({acc:.1%})")
    print(f"  Log-loss    : {loss:.4f}")
    print(f"  Brier score : {brier:.4f}  (lower = better, 0.25 = random)")

    # ── Calibration curve ─────────────────────────────────────────────────────
    _, axes = plt.subplots(1, 3, figsize=(18, 4))

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

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm   = confusion_matrix(y_val, preds)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Away Win", "Home Win"]
    )
    disp.plot(ax=axes[2], colorbar=False, cmap="Blues")
    axes[2].set_title("Confusion matrix")

    plt.tight_layout()
    plt.savefig("models/evaluation.png", dpi=150)
    plt.show()
    print("  Evaluation chart saved to models/evaluation.png")

    return acc, loss, brier


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
    """Orchestrates data loading, baseline, LSTM training, and evaluation."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    print(f"[DEVICE] Using: {DEVICE}")

    # 1. Load data
    X_train, X_val, y_train, y_val, lengths_train, lengths_val = load_data()

    # 2. Baseline
    baseline_acc, baseline_loss = run_baseline(
        X_train, X_val, y_train, y_val, lengths_train, lengths_val
    )

    # 3. DataLoaders
    train_ds = WinProbDataset(X_train, y_train, lengths_train)
    val_ds   = WinProbDataset(X_val,   y_val,   lengths_val)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    # 4. Model, optimizer, loss
    model     = WinProbModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()

    # 5. Train (graphs saved inside train() on any exit)
    _, _ = train(model, train_dl, val_dl, optimizer, criterion)

    # 6. Load best weights and evaluate
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    nn_acc, nn_loss, nn_brier = evaluate(model, X_val, y_val, lengths_val)

    # 7. Summary
    print(f"\n[SUMMARY]")
    print(f"  {'':20s}  {'Accuracy':>10}  {'Log-loss':>10}  {'Brier':>8}")
    print(f"  {'-'*20}  {'-'*10}  {'-'*10}  {'-'*8}")
    print(f"  {'Logistic Regression':20s}  {baseline_acc:>10.4f}  "
          f"{baseline_loss:>10.4f}  {'N/A':>8}")
    print(f"  {'LSTM':20s}  {nn_acc:>10.4f}  "
          f"{nn_loss:>10.4f}  {nn_brier:>8.4f}")

    improvement = nn_acc - baseline_acc
    print(
        f"\n  LSTM {'outperforms' if improvement > 0 else 'underperforms'} "
        f"baseline by {abs(improvement):.2%}"
    )

    if improvement <= 0:
        print("  Tip: try increasing HIDDEN_SIZE or NUM_LAYERS")
    else:
        print("  Model is ready — update server/app.py with the LSTM architecture!")


if __name__ == "__main__":
    main()
