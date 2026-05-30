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
 
BATCH_SIZE   = 256
EPOCHS       = 30
LEARNING_RATE = 1e-3
VAL_SPLIT    = 0.2
RANDOM_SEED  = 42
 
 
# ── 1. Dataset class ──────────────────────────────────────────────────────────
class WinProbDataset(Dataset):
    """
    Wraps numpy arrays into a PyTorch Dataset.
    PyTorch's DataLoader needs this to batch and shuffle your data.
    """
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # shape: (N, 1)
 
    def __len__(self):
        return len(self.X)
 
    def __getitem__(self, i):
        return self.X[i], self.y[i]
 
 
# ── 2. Model architecture ─────────────────────────────────────────────────────
class WinProbModel(nn.Module):
    """
    A multilayer perceptron (MLP) with 3 linear layers.
 
    Input (5 features)
      → Linear(5→64) → ReLU → Dropout(0.2)
      → Linear(64→32) → ReLU
      → Linear(32→1) → Sigmoid   ← outputs a probability between 0 and 1
    """
    def __init__(self, input_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
 
    def forward(self, x):
        return self.net(x)
 
 
# ── 3. Load and prepare data ──────────────────────────────────────────────────
def load_data():
    print("[LOADING] Reading features.parquet...")
    df = pd.read_parquet(DATA_PATH).dropna(subset=FEATURE_COLS + [LABEL_COL])
 
    print(f"  Total rows   : {len(df):,}")
    print(f"  Home win rate: {df[LABEL_COL].mean():.1%}")
 
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[LABEL_COL].values.astype(np.float32)
 
    # Split into train and validation sets before scaling
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VAL_SPLIT, random_state=RANDOM_SEED
    )
 
    # Scale features to mean=0, std=1
    # This helps the neural net train faster and more stably
    # We fit ONLY on training data, then apply the same scale to validation
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
 
    # Save scaler parameters so we can apply the same scaling at inference time
    np.save(SCALER_PATH, np.array([scaler.mean_, scaler.scale_]))
    print(f"  Scaler saved to {SCALER_PATH}")
 
    print(f"  Train rows   : {len(X_train):,}")
    print(f"  Val rows     : {len(X_val):,}")
 
    return X_train, X_val, y_train, y_val, scaler
 
 
# ── 4. Logistic regression baseline ──────────────────────────────────────────
def run_baseline(X_train, X_val, y_train, y_val):
    """
    Fits a simple logistic regression first.
    If our neural net can't beat this, we need better features — not a bigger model.
    Target: accuracy above ~72%, log-loss below ~0.55.
    """
    print("\n[BASELINE] Logistic Regression...")
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
 
    preds      = lr.predict(X_val)
    probs      = lr.predict_proba(X_val)[:, 1]
    acc        = accuracy_score(y_val, preds)
    loss       = log_loss(y_val, probs)
 
    print(f"  Accuracy : {acc:.4f}  ({acc:.1%})")
    print(f"  Log-loss : {loss:.4f}")
    print(f"  (Neural net needs to beat both of these)")
    return acc, loss
 
 
# ── 5. Training loop ──────────────────────────────────────────────────────────
def train(model, train_dl, val_dl, optimizer, criterion):
    """
    Trains the model for EPOCHS epochs.
    Each epoch:
      1. Forward pass  — run features through the network, get predictions
      2. Compute loss  — compare predictions to true labels with BCELoss
      3. Backward pass — compute gradients (how to adjust each weight)
      4. Step          — nudge all weights in the direction that reduces loss
    After training, evaluate on validation set WITHOUT updating weights.
    Save the model whenever validation loss improves.
    """
    train_losses = []
    val_losses   = []
    best_val_loss = float("inf")
 
    print(f"\n[TRAINING] {EPOCHS} epochs, batch size {BATCH_SIZE}...")
    print(f"  {'Epoch':>5}  {'Train Loss':>10}  {'Val Loss':>10}  {'Saved':>6}")
    print(f"  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*6}")
 
    for epoch in range(1, EPOCHS + 1):
 
        # ── Training phase ────────────────────────────────────────────────────
        model.train()
        batch_losses = []
        for X_batch, y_batch in train_dl:
            optimizer.zero_grad()                  # clear old gradients
            preds = model(X_batch)                 # forward pass
            loss  = criterion(preds, y_batch)      # compute loss
            loss.backward()                        # backward pass
            optimizer.step()                       # update weights
            batch_losses.append(loss.item())
 
        train_loss = np.mean(batch_losses)
        train_losses.append(train_loss)
 
        # ── Validation phase ──────────────────────────────────────────────────
        model.eval()
        val_batch_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_dl:
                preds = model(X_batch)
                loss  = criterion(preds, y_batch)
                val_batch_losses.append(loss.item())
 
        val_loss = np.mean(val_batch_losses)
        val_losses.append(val_loss)
 
        # ── Save best model ───────────────────────────────────────────────────
        saved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            saved = "saved"
 
        print(f"  {epoch:>5}  {train_loss:>10.4f}  {val_loss:>10.4f}  {saved:>6}")
 
    print(f"\n  Best val loss: {best_val_loss:.4f}")
    print(f"  Model saved to {MODEL_PATH}")
    return train_losses, val_losses
 
 
# ── 6. Evaluation ─────────────────────────────────────────────────────────────
def evaluate(model, X_val, y_val):
    """
    Measures accuracy and log-loss on the validation set.
    Then plots two charts:
      1. Calibration curve — does 70% predicted mean 70% actual win rate?
      2. Prediction distribution — are predictions spread out or clumped near 0.5?
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
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
 
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
    """
    Plots train loss vs val loss per epoch.
    If val loss stops going down while train loss keeps falling = overfitting.
    Both curves going down together = healthy training.
    """
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
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
 
    # 1. Load data
    X_train, X_val, y_train, y_val, scaler = load_data()
 
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
 
    # 5. Train
    train_losses, val_losses = train(model, train_dl, val_dl, optimizer, criterion)
 
    # 6. Plot training curves
    plot_training_curves(train_losses, val_losses)
 
    # 7. Load best saved weights and evaluate
    model.load_state_dict(torch.load(MODEL_PATH))
    nn_acc, nn_loss = evaluate(model, X_val, y_val)
 
    # 8. Final comparison
    print(f"\n[SUMMARY]")
    print(f"  {'':20s}  {'Accuracy':>10}  {'Log-loss':>10}")
    print(f"  {'-'*20}  {'-'*10}  {'-'*10}")
    print(f"  {'Logistic Regression':20s}  {baseline_acc:>10.4f}  {baseline_loss:>10.4f}")
    print(f"  {'Neural Network':20s}  {nn_acc:>10.4f}  {nn_loss:>10.4f}")
    improvement = nn_acc - baseline_acc
    print(f"\n  Neural net {'outperforms' if improvement > 0 else 'underperforms'} baseline by {abs(improvement):.2%}")
 
    if improvement <= 0:
        print("  Tip: try adding more features in build_features.py")
    else:
        print("  Model is ready — run server/app.py next!")
 
 
if __name__ == "__main__":
    main()
 
