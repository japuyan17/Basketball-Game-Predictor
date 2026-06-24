import time
import os
import pandas as pd
from nba_api.stats.endpoints import LeagueGameFinder, PlayByPlayV3
 
# ── Settings ────────────────────────────────────────────────────────────────
SEASONS = ["2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24"]
DATA_DIR = "data"
SLEEP_SECONDS = 0.7   # pause between API calls so NBA.com doesn't block you
 
# ── Helpers ──────────────────────────────────────────────────────────────────
 
def fetch_game_ids(season):
    print(f"  Fetching game IDs for {season}...")
    time.sleep(SLEEP_SECONDS)
    finder = LeagueGameFinder(
        season_nullable=season,
        league_id_nullable="00",          # 00 = NBA
        season_type_nullable="Regular Season"
    )
    games = finder.get_data_frames()[0]
    ids = games["GAME_ID"].unique().tolist()
    print(f"  Found {len(ids)} games")
    return ids


def fetch_pbp(game_id):
    time.sleep(SLEEP_SECONDS)
    pbp = PlayByPlayV3(game_id=game_id)
    return pbp.get_data_frames()[0]
 
 
# ── Main loop ────────────────────────────────────────────────────────────────
 
def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    for season in SEASONS:
        out_path = os.path.join(DATA_DIR, f"pbp_{season}.parquet")
 
        # Skip seasons we already downloaded
        if os.path.exists(out_path):
            print(f"[SKIP] {season} already saved")
            continue

        print(f"\n[START] Season {season}")
        game_ids = fetch_game_ids(season)
 
        frames = []
        total = len(game_ids)
 
        for i, gid in enumerate(game_ids, 1):
            try:
                df = fetch_pbp(gid)
                df["GAME_ID"] = gid
                frames.append(df)
 
                if i % 50 == 0:
                    print(f"  {i}/{total} games fetched...")
 
            except Exception as e:
                print(f"  [ERROR] game {gid}: {e} — skipping")

        if frames:
            season_df = pd.concat(frames, ignore_index=True)
            season_df.to_parquet(out_path, index=False)
            print(f"[DONE] {season} — {len(frames)} games saved to {out_path}")
        else:
            print(f"[WARN] {season} — no data fetched")

    print("\nAll seasons complete. Check your data/ folder.")


if __name__ == "__main__":
    main()
 
