import os
import time
import pandas as pd
from nba_api.stats.endpoints import LeagueGameFinder

# ── Settings ──────────────────────────────────────────────────────────────────
SEASONS      = [
    "2018-19", "2019-20", "2020-21", "2021-22",
    "2022-23", "2023-24", "2024-25", "2025-26",
]
DATA_DIR     = "data"
SLEEP_SECONDS = 0.7


# ── Fetch one season of team-level game stats ─────────────────────────────────
def fetch_team_games(season):
    """Pulls every regular-season game row for all teams in a given season."""
    time.sleep(SLEEP_SECONDS)
    finder = LeagueGameFinder(
        season_nullable=season,
        league_id_nullable="00",
        season_type_nullable="Regular Season",
    )
    df = finder.get_data_frames()[0]
    df["SEASON"] = season
    return df


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    """Downloads team game logs for all seasons and saves each as a parquet."""
    os.makedirs(DATA_DIR, exist_ok=True)

    for season in SEASONS:
        out_path = os.path.join(DATA_DIR, f"team_games_{season}.parquet")

        if os.path.exists(out_path):
            print(f"[SKIP] {season} already saved")
            continue

        print(f"\n[FETCHING] {season}...")
        try:
            df = fetch_team_games(season)
            df.to_parquet(out_path, index=False)
            print(
                f"[DONE] {season} — {len(df)} team-game rows "
                f"({df['GAME_ID'].nunique()} games) saved to {out_path}"
            )
        except Exception as e:
            print(f"[ERROR] {season}: {e}")

    print("\nAll seasons complete. Run build_features.py next.")


if __name__ == "__main__":
    main()
