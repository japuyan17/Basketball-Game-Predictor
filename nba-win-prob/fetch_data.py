import time
import os
import pandas as pd
from nba_api.stats.endpoints import LeagueGameFinder, PlayByPlayV2

# ── Settings ────────────────────────────────────────────────────────────────
SEASONS = ["2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24"]
DATA_DIR = "data"
SLEEP_SECONDS = 3
MAX_RETRIES = 5

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://www.nba.com/",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Origin": "https://www.nba.com",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
}

# ── Helpers ──────────────────────────────────────────────────────────────────

def fetch_with_retry(fn, label):
    """Call fn(), retrying up to MAX_RETRIES times with exponential backoff."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn()
        except Exception as e:
            wait = 10 * (2 ** (attempt - 1))  # 10s, 20s, 40s, 80s, 160s
            print(f"  [RETRY {attempt}/{MAX_RETRIES}] {label}: {e} — waiting {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Failed after {MAX_RETRIES} retries: {label}")


def fetch_game_ids(season):
    """Return every regular-season game ID for one season."""
    print(f"  Fetching game IDs for {season}...")
    time.sleep(SLEEP_SECONDS)

    def _call():
        finder = LeagueGameFinder(
            season_nullable=season,
            league_id_nullable="00",
            season_type_nullable="Regular Season",
            headers=HEADERS,
            timeout=120
        )
        return finder.get_data_frames()[0]

    games = fetch_with_retry(_call, f"LeagueGameFinder {season}")
    ids = games["GAME_ID"].unique().tolist()
    print(f"  Found {len(ids)} games")
    return ids


def fetch_pbp(game_id):
    """Return the play-by-play DataFrame for one game."""
    time.sleep(SLEEP_SECONDS)

    def _call():
        pbp = PlayByPlayV2(game_id=game_id, headers=HEADERS, timeout=120)
        return pbp.get_data_frames()[0]

    return fetch_with_retry(_call, f"PlayByPlayV2 {game_id}")


# ── Main loop ────────────────────────────────────────────────────────────────

def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    for season in SEASONS:
        out_path = os.path.join(DATA_DIR, f"pbp_{season}.parquet")

        if os.path.exists(out_path):
            print(f"[SKIP] {season} already saved — delete {out_path} to re-fetch")
            continue

        print(f"\n[START] Season {season}")

        try:
            game_ids = fetch_game_ids(season)
        except RuntimeError as e:
            print(f"[FATAL] Could not fetch game IDs for {season}: {e}")
            continue

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
