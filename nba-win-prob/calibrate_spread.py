import os
import numpy as np
import pandas as pd
import torch
from scipy.stats import norm

from train_model import WinProbModel, FEATURE_COLS, MODEL_PATH, SCALER_PATH

# ── Settings ──────────────────────────────────────────────────────────────────
DATA_PATH   = "data/matchup_features.parquet"
SIGMA_PATH  = "models/spread_sigma.npy"
PROB_EPS    = 1e-4   # clip probabilities away from 0/1 before norm.ppf (avoids ±inf)


# ── Fit the win-prob → point-spread conversion ────────────────────────────────
def fit_sigma(win_probs, home_margins):
    """
    Fits spread = sigma * norm.ppf(win_prob) by closed-form least squares
    (no intercept — a 50% win prob must always imply a pick'em spread of 0).
    Returns the fitted sigma (points of standard deviation in game margin).
    """
    clipped = np.clip(win_probs, PROB_EPS, 1 - PROB_EPS)
    z = norm.ppf(clipped)
    sigma = np.sum(home_margins * z) / np.sum(z * z)
    return float(sigma)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    """
    Runs the trained model over every historical matchup, then fits the
    win-prob-to-spread conversion against actual final score margins.
    """
    print("[LOADING] matchup_features.parquet...")
    df = pd.read_parquet(DATA_PATH).dropna(
        subset=FEATURE_COLS + ["home_win", "home_margin"]
    )
    print(f"  Matchups with known margin: {len(df):,}")

    scaler_data  = np.load(SCALER_PATH)
    scaler_mean  = scaler_data[0]
    scaler_scale = scaler_data[1]

    X = df[FEATURE_COLS].values.astype(np.float32)
    X = (X - scaler_mean) / scaler_scale

    model = WinProbModel()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()

    with torch.no_grad():
        probs = model(torch.tensor(X, dtype=torch.float32)).squeeze(1).numpy()

    margins = df["home_margin"].values.astype(np.float64)
    sigma   = fit_sigma(probs, margins)

    predicted = sigma * norm.ppf(np.clip(probs, PROB_EPS, 1 - PROB_EPS))
    mae       = np.mean(np.abs(predicted - margins))
    corr      = np.corrcoef(predicted, margins)[0, 1]

    os.makedirs("models", exist_ok=True)
    np.save(SIGMA_PATH, np.array([sigma]))

    print(f"\n[CALIBRATION]")
    print(f"  Fitted sigma        : {sigma:.2f} points")
    print(f"  Mean absolute error : {mae:.2f} points")
    print(f"  Correlation w/ actual margin : {corr:.3f}")
    print(f"  Saved to {SIGMA_PATH}")


if __name__ == "__main__":
    main()
