import os
import re
import pandas as pd
import numpy as np

# ── Settings ─────────────────────────────────────────────────────────────────
DATA_DIR    = "data"
SEASONS     = ["2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24"]
OUT_PATH    = os.path.join(DATA_DIR, "features.parquet")
 
# ── Helper: parse V3 clock string → total seconds remaining in game ───────────
def parse_clock(clock_str, period):
    """
    Converts a V3 clock like "PT08M42.00S" (8:42 left in the period) into total
    seconds left in the game. Regulation periods are 720s; in overtime only the
    seconds left in the current OT period are used. Returns None if unparseable.
    """
    try:
        match = re.match(r"PT0*(\d+)M0*([\d.]+)S", str(clock_str))
        if not match:
            return None
        mins = int(match.group(1))
        secs = float(match.group(2))
        secs_left_in_period = mins * 60 + secs
        if period <= 4:
            periods_left = max(0, 4 - period)
            return periods_left * 720 + secs_left_in_period
        else:
            # overtime: clock already reflects time left in the OT period
            return secs_left_in_period
    except:
        return None


# ── Core: build features for one game ────────────────────────────────────────
def build_features(df):
    """
    Takes raw play-by-play for one game.
    Returns a clean DataFrame with one row per event and these columns:
      score_diff      — home score minus away score at this moment
      secs_left       — total seconds remaining in the game
      period          — current period (1–4, 5+ for OT)
      home_foul_diff  — home team fouls minus away team fouls (running total)
      momentum        — change in score diff over the last 5 events
      home_win        — 1 if home team won, 0 if they lost (label)
      GAME_ID         — kept for reference
    """
    df = df.copy().reset_index(drop=True)
 
    # ── Score differential (V3: separate home/away score columns) ─────────────
    home_score = pd.to_numeric(df["scoreHome"], errors="coerce").ffill().fillna(0)
    away_score = pd.to_numeric(df["scoreAway"], errors="coerce").ffill().fillna(0)
    df["score_diff"] = (home_score - away_score).astype(float)

    # ── Time remaining ────────────────────────────────────────────────────────
    df["secs_left"] = df.apply(
        lambda r: parse_clock(r["clock"], r["period"]), axis=1
    )
    df["secs_left"] = df["secs_left"].ffill().fillna(0).astype(float)

    # ── Period ────────────────────────────────────────────────────────────────
    df["period"] = df["period"].fillna(1).astype(int)

    # ── Foul differential (V3: one description column + h/v location) ─────────
    is_foul = df["description"].str.contains("FOUL", case=False, na=False)
    home_fouls = (is_foul & (df["location"] == "h")).cumsum()
    away_fouls = (is_foul & (df["location"] == "v")).cumsum()
    df["home_foul_diff"] = (home_fouls - away_fouls).astype(float)
 
    # ── Momentum (score diff change over last 5 events) ───────────────────────
    df["momentum"] = df["score_diff"].diff(5).fillna(0).astype(float)
 
    # ── Label: did the home team win? ─────────────────────────────────────────
    final_score_diff = df["score_diff"].iloc[-1]
    if final_score_diff == 0:
        return None                        # discard ties (essentially never happens)
    df["home_win"] = int(final_score_diff > 0)
 
    # ── Drop rows with any nulls in our feature columns ───────────────────────
    feature_cols = ["score_diff", "secs_left", "period", "home_foul_diff", "momentum", "home_win", "GAME_ID"]
    df = df[feature_cols].dropna()
 
    return df
 
 
# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    all_frames = []
    total_games = 0
    skipped = 0
 
    for season in SEASONS:
        path = os.path.join(DATA_DIR, f"pbp_{season}.parquet")
 
        if not os.path.exists(path):
            print(f"[SKIP] {path} not found — run fetch_data.py first")
            continue
 
        print(f"\n[LOADING] {season}...")
        raw = pd.read_parquet(path)
        game_ids = raw["GAME_ID"].unique()
        print(f"  {len(game_ids)} games found")
 
        season_frames = []
        for gid in game_ids:
            game_df = raw[raw["GAME_ID"] == gid].copy()
            try:
                features = build_features(game_df)
                if features is not None and len(features) > 0:
                    season_frames.append(features)
                    total_games += 1
                else:
                    skipped += 1
            except Exception as e:
                print(f"  [ERROR] {gid}: {e}")
                skipped += 1
 
        print(f"  Built features for {len(season_frames)} games")
        all_frames.extend(season_frames)
 
    if not all_frames:
        print("\nNo data found. Make sure fetch_data.py has been run first.")
        return
 
    # ── Combine and save ──────────────────────────────────────────────────────
    print("\n[COMBINING] all seasons...")
    final_df = pd.concat(all_frames, ignore_index=True)
 
    print(f"[STATS]")
    print(f"  Total rows     : {len(final_df):,}")
    print(f"  Total games    : {total_games:,}")
    print(f"  Skipped games  : {skipped:,}")
    print(f"  Home win rate  : {final_df['home_win'].mean():.1%}")
    print(f"  Feature columns: {list(final_df.columns)}")
 
    final_df.to_parquet(OUT_PATH, index=False)
    print(f"\n[DONE] Saved to {OUT_PATH}")
 
 
if __name__ == "__main__":
    main()
