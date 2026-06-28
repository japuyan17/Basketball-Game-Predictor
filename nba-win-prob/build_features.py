import os
import re
import numpy as np
import pandas as pd

# ── Settings ──────────────────────────────────────────────────────────────────
DATA_DIR = "data"
SEASONS  = [
    "2018-19", "2019-20", "2020-21",
    "2021-22", "2022-23", "2023-24",
]
OUT_PATH = os.path.join(DATA_DIR, "sequences.npz")

FEATURE_COLS = [
    "score_diff", "secs_left", "period", "home_foul_diff", "momentum"
]


# ── Helper: parse V3 clock string → total seconds remaining in game ───────────
def parse_clock(clock_str, period):
    """Converts a V3 ISO clock string into total seconds left in the game."""
    try:
        match = re.match(r"PT0*(\d+)M0*([\d.]+)S", str(clock_str))
        if not match:
            return None
        mins = int(match.group(1))
        secs = float(match.group(2))
        secs_left_in_period = mins * 60 + secs
        if period <= 4:
            return max(0, 4 - period) * 720 + secs_left_in_period
        else:
            return secs_left_in_period
    except Exception:
        return None


# ── Core: build ordered feature sequence for one game ─────────────────────────
def build_game_sequence(df):
    """
    Takes raw play-by-play for one game, returns (sequence, label) where
    sequence is shape (n_plays, 5) and label is 1 if home won, 0 otherwise.
    Returns None if the game is invalid (tie or empty after cleaning).
    """
    df = df.copy().sort_values("actionNumber").reset_index(drop=True)

    # Score differential
    home_score   = pd.to_numeric(df["scoreHome"], errors="coerce").ffill().fillna(0)
    away_score   = pd.to_numeric(df["scoreAway"], errors="coerce").ffill().fillna(0)
    df["score_diff"] = (home_score - away_score).astype(float)

    # Time remaining
    df["secs_left"] = df.apply(
        lambda r: parse_clock(r["clock"], r["period"]), axis=1
    ).ffill().fillna(0).astype(float)

    # Period
    df["period"] = df["period"].fillna(1).astype(int)

    # Foul differential
    is_foul        = df["description"].str.contains("FOUL", case=False, na=False)
    home_fouls     = (is_foul & (df["location"] == "h")).cumsum()
    away_fouls     = (is_foul & (df["location"] == "v")).cumsum()
    df["home_foul_diff"] = (home_fouls - away_fouls).astype(float)

    # Momentum: score-diff change over the last 5 plays
    df["momentum"] = df["score_diff"].diff(5).fillna(0).astype(float)

    # Label
    final_diff = df["score_diff"].iloc[-1]
    if final_diff == 0:
        return None
    label = int(final_diff > 0)

    seq = df[FEATURE_COLS].dropna().values.astype(np.float32)
    if len(seq) == 0:
        return None

    return seq, label


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    """Builds per-game sequences and saves them as a padded numpy archive."""
    all_sequences = []
    all_labels    = []
    total_games   = 0
    skipped       = 0

    for season in SEASONS:
        path = os.path.join(DATA_DIR, f"pbp_{season}.parquet")
        if not os.path.exists(path):
            print(f"[SKIP] {path} not found — run fetch_data.py first")
            continue

        print(f"\n[LOADING] {season}...")
        raw      = pd.read_parquet(path)
        game_ids = raw["GAME_ID"].unique()
        print(f"  {len(game_ids)} games found")

        season_count = 0
        for gid in game_ids:
            game_df = raw[raw["GAME_ID"] == gid].copy()
            try:
                result = build_game_sequence(game_df)
                if result is not None:
                    seq, label = result
                    all_sequences.append(seq)
                    all_labels.append(label)
                    total_games  += 1
                    season_count += 1
                else:
                    skipped += 1
            except Exception as e:
                print(f"  [ERROR] {gid}: {e}")
                skipped += 1

        print(f"  Built sequences for {season_count} games")

    if not all_sequences:
        print("\nNo data found. Make sure fetch_data.py has been run first.")
        return

    # ── Pad all sequences to the same length with zeros ───────────────────────
    max_len    = max(len(s) for s in all_sequences)
    n_games    = len(all_sequences)
    n_features = len(FEATURE_COLS)

    X       = np.zeros((n_games, max_len, n_features), dtype=np.float32)
    lengths = np.zeros(n_games, dtype=np.int32)
    y       = np.array(all_labels, dtype=np.float32)

    for i, seq in enumerate(all_sequences):
        seq_len         = len(seq)
        X[i, :seq_len]  = seq
        lengths[i]      = seq_len

    print(f"\n[STATS]")
    print(f"  Total games   : {total_games:,}")
    print(f"  Skipped games : {skipped:,}")
    print(f"  X shape       : {X.shape}  (games × timesteps × features)")
    print(f"  Max seq length: {max_len}")
    print(f"  Avg seq length: {lengths.mean():.0f}")
    print(f"  Home win rate : {y.mean():.1%}")

    np.savez(OUT_PATH, X=X, y=y, lengths=lengths)
    print(f"\n[DONE] Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
