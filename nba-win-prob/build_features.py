import os
import numpy as np
import pandas as pd

# ── Settings ──────────────────────────────────────────────────────────────────
DATA_DIR       = "data"
SEASONS        = [
    "2018-19", "2019-20", "2020-21", "2021-22",
    "2022-23", "2023-24", "2024-25", "2025-26",
]
OUT_FEATURES   = os.path.join(DATA_DIR, "matchup_features.parquet")
OUT_TEAM_STATS = os.path.join(DATA_DIR, "team_stats_latest.parquet")
MIN_GAMES      = 5   # drop matchups where either team has fewer games of history


# ── Compute rolling per-team stats for one season dataframe ───────────────────
def add_rolling_stats(df):
    """
    Adds rolling season stats to each team-game row using only games played
    BEFORE the current game (shift(1) prevents data leakage).
    New columns: win_pct, ppg, opp_ppg, plus_minus_avg, last10_win_pct,
    home_game_win_pct, away_game_win_pct, games_played.
    """
    df = df.sort_values(["TEAM_ID", "GAME_DATE"]).copy()
    df["win"]     = (df["WL"] == "W").astype(float)
    df["is_home"] = df["MATCHUP"].str.contains(r"vs\.", regex=True).astype(float)
    df["is_away"] = 1.0 - df["is_home"]

    grp = df.groupby("TEAM_ID")

    # Cumulative games played before this game
    df["games_played"] = grp.cumcount()  # 0-indexed = games before this one

    # Overall win %
    df["cum_wins"] = grp["win"].transform(
        lambda x: x.cumsum().shift(1)
    ).fillna(0)
    df["win_pct"]  = df["cum_wins"] / df["games_played"].replace(0, np.nan)

    # Points per game
    df["cum_pts"] = grp["PTS"].transform(
        lambda x: x.cumsum().shift(1)
    ).fillna(0)
    df["ppg"] = df["cum_pts"] / df["games_played"].replace(0, np.nan)

    # Net rating (avg plus/minus per game)
    df["cum_pm"] = grp["PLUS_MINUS"].transform(
        lambda x: x.cumsum().shift(1)
    ).fillna(0)
    df["plus_minus_avg"] = df["cum_pm"] / df["games_played"].replace(0, np.nan)

    # Opponent PPG derived from PPG and net rating (avoids a second join)
    df["opp_ppg"] = df["ppg"] - df["plus_minus_avg"]

    # Last-10 win %
    df["last10_win_pct"] = (
        grp["win"].transform(lambda x: x.shift(1).rolling(10, min_periods=1).sum())
        / grp["win"].transform(
            lambda x: x.shift(1).rolling(10, min_periods=1).count()
        ).replace(0, np.nan)
    )

    # Home-game win % (win% in games played at home only)
    df["home_win_contrib"] = df["win"] * df["is_home"]
    df["cum_home_wins"]    = grp["home_win_contrib"].transform(
        lambda x: x.cumsum().shift(1)
    ).fillna(0)
    df["cum_home_games"]   = grp["is_home"].transform(
        lambda x: x.cumsum().shift(1)
    ).fillna(0)
    df["home_game_win_pct"] = (
        df["cum_home_wins"] / df["cum_home_games"].replace(0, np.nan)
    )

    # Away-game win % (win% in games played on the road only)
    df["away_win_contrib"] = df["win"] * df["is_away"]
    df["cum_away_wins"]    = grp["away_win_contrib"].transform(
        lambda x: x.cumsum().shift(1)
    ).fillna(0)
    df["cum_away_games"]   = grp["is_away"].transform(
        lambda x: x.cumsum().shift(1)
    ).fillna(0)
    df["away_game_win_pct"] = (
        df["cum_away_wins"] / df["cum_away_games"].replace(0, np.nan)
    )

    # PPG in away games only (how a team scores on the road)
    df["pts_away_contrib"] = df["PTS"] * df["is_away"]
    df["cum_pts_away"]     = grp["pts_away_contrib"].transform(
        lambda x: x.cumsum().shift(1)
    ).fillna(0)
    df["away_ppg"] = df["cum_pts_away"] / df["cum_away_games"].replace(0, np.nan)

    # PPG in home games only (how a team scores at home)
    df["pts_home_contrib"] = df["PTS"] * df["is_home"]
    df["cum_pts_home"]     = grp["pts_home_contrib"].transform(
        lambda x: x.cumsum().shift(1)
    ).fillna(0)
    df["home_ppg"] = df["cum_pts_home"] / df["cum_home_games"].replace(0, np.nan)

    # Pace-adjusted offensive efficiency (points per 100 possessions)
    df["poss"]     = df["FGA"] - df["OREB"] + df["TOV"] + 0.44 * df["FTA"]
    df["cum_poss"] = grp["poss"].transform(
        lambda x: x.cumsum().shift(1)
    ).fillna(0)
    df["off_efficiency"] = (
        df["cum_pts"] / df["cum_poss"].replace(0, np.nan)
    ) * 100

    # Turnover rate (turnovers per 100 possessions)
    df["cum_tov"] = grp["TOV"].transform(
        lambda x: x.cumsum().shift(1)
    ).fillna(0)
    df["tov_rate"] = (
        df["cum_tov"] / df["cum_poss"].replace(0, np.nan)
    ) * 100

    return df


# ── Build matchup rows by joining home and away team stats per game ────────────
def build_matchups(df):
    """
    Splits team-game rows into home and away, then joins on GAME_ID to create
    one row per game with both teams' rolling stats side by side.
    """
    home = df[df["MATCHUP"].str.contains(r"vs\.", regex=True)].copy()
    away = df[df["MATCHUP"].str.contains(r"@",    regex=True)].copy()

    # NOTE: "PLUS_MINUS" here is the raw per-game final score margin (not the
    # rolling plus_minus_avg). Only the home side is kept, renamed to
    # "home_margin" below. It is NEVER added to FEATURE_COLS — it exists
    # solely for spread calibration in calibrate_spread.py and must not be
    # fed into the model (that would be a leakage bug: it's the game result).
    stat_cols = [
        "GAME_ID", "GAME_DATE", "TEAM_NAME", "TEAM_ABBREVIATION",
        "win", "win_pct", "ppg", "opp_ppg", "plus_minus_avg",
        "last10_win_pct", "home_game_win_pct", "away_game_win_pct",
        "away_ppg", "home_ppg",
        "off_efficiency", "tov_rate", "games_played", "PLUS_MINUS",
    ]

    matchups = home[stat_cols].merge(
        away[stat_cols],
        on=["GAME_ID", "GAME_DATE"],
        suffixes=("_home", "_away"),
    )

    matchups = matchups.rename(columns={
        "TEAM_NAME_home":         "home_team",
        "TEAM_NAME_away":         "away_team",
        "TEAM_ABBREVIATION_home": "home_abbr",
        "TEAM_ABBREVIATION_away": "away_abbr",
        "win_home":               "home_win",
        "home_game_win_pct_home": "home_split_win_pct",
        "away_game_win_pct_away": "away_split_win_pct",
        # how many points each team scores in their non-native venue
        "away_ppg_home":          "home_team_away_ppg",
        "home_ppg_away":          "away_team_home_ppg",
        # actual final score margin (home - away) — calibration only, not a feature
        "PLUS_MINUS_home":        "home_margin",
    })

    matchups = matchups.drop(columns=["PLUS_MINUS_away"])

    # Net rating differential: positive = home team is better
    matchups["net_rtg_diff"] = (
        matchups["plus_minus_avg_home"] - matchups["plus_minus_avg_away"]
    )

    # Drop rows where either team lacks enough history
    matchups = matchups[
        (matchups["games_played_home"] >= MIN_GAMES) &
        (matchups["games_played_away"] >= MIN_GAMES)
    ]

    return matchups.drop(columns=["win_away"])


# ── Save latest stats per team for use by the server at inference time ─────────
def save_latest_team_stats(df):
    """
    Keeps only the most recent row per team (highest games_played) so the
    server can look up any team's current stats by name.
    """
    latest = (
        df.sort_values("games_played", ascending=False)
          .groupby("TEAM_ID")
          .first()
          .reset_index()
    )
    # ppg, plus_minus_avg, opp_ppg, off_efficiency are read by injury_adjust.py
    # at inference time — do not remove or rename them (see Lessons.md #12).
    keep = [
        "TEAM_ID", "TEAM_NAME", "TEAM_ABBREVIATION",
        "win_pct", "ppg", "opp_ppg", "plus_minus_avg",
        "last10_win_pct", "home_game_win_pct", "away_game_win_pct",
        "away_ppg", "home_ppg",
        "off_efficiency", "tov_rate", "games_played",
    ]
    latest[keep].to_parquet(OUT_TEAM_STATS, index=False)
    print(f"  Latest team stats saved to {OUT_TEAM_STATS}  ({len(latest)} teams)")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    """Loads team game logs, builds matchup features, and saves outputs."""
    all_team_rows = []
    all_matchups  = []

    for season in SEASONS:
        path = os.path.join(DATA_DIR, f"team_games_{season}.parquet")
        if not os.path.exists(path):
            print(f"[SKIP] {path} not found — run fetch_data.py first")
            continue

        print(f"\n[LOADING] {season}...")
        raw = pd.read_parquet(path)
        raw["GAME_DATE"] = pd.to_datetime(raw["GAME_DATE"])

        enriched  = add_rolling_stats(raw)
        matchups  = build_matchups(enriched)

        all_team_rows.append(enriched)
        all_matchups.append(matchups)

        print(
            f"  {matchups['GAME_ID'].nunique()} matchups built  "
            f"({len(matchups)} rows after MIN_GAMES={MIN_GAMES} filter)"
        )

    if not all_matchups:
        print("\nNo data found. Run fetch_data.py first.")
        return

    # ── Combine and save matchup features ─────────────────────────────────────
    final = pd.concat(all_matchups, ignore_index=True)

    feature_cols = [
        "win_pct_home",        "ppg_home",
        "opp_ppg_home",        "plus_minus_avg_home",   "last10_win_pct_home",
        "win_pct_away",        "ppg_away",
        "opp_ppg_away",        "plus_minus_avg_away",   "last10_win_pct_away",
        "home_split_win_pct",  "away_split_win_pct",    "net_rtg_diff",
        "home_team_away_ppg",  "away_team_home_ppg",
        "off_efficiency_home", "off_efficiency_away",
        "tov_rate_home",       "tov_rate_away",
    ]

    final = final.dropna(subset=feature_cols + ["home_win"])
    final.to_parquet(OUT_FEATURES, index=False)

    print(f"\n[STATS]")
    print(f"  Total matchups : {len(final):,}")
    print(f"  Home win rate  : {final['home_win'].mean():.1%}")
    print(f"  Features       : {feature_cols}")
    print(f"\n[DONE] Matchup features saved to {OUT_FEATURES}")

    # ── Save latest team stats for server inference ────────────────────────────
    # Use the most recent season only for current stats
    latest_season_rows = all_team_rows[-1]
    save_latest_team_stats(latest_season_rows)


if __name__ == "__main__":
    main()
