import time
import pandas as pd
from nba_api.stats.endpoints import LeagueDashPlayerStats

CURRENT_SEASON = "2024-25"
PPG_FLOOR      = 70.0   # minimum adjusted ppg to prevent nonsensical values


# ── Fetch season-wide player stats from nba_api ───────────────────────────────
def fetch_player_stats(season=CURRENT_SEASON):
    """Fetches per-player season totals and returns a cleaned DataFrame."""
    time.sleep(0.6)   # rate limit
    raw = LeagueDashPlayerStats(
        season=season,
        per_mode_simple="Totals",
        league_id_nullable="00",
    ).get_data_frames()[0]

    keep = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "GP", "MIN", "PTS",
            "PLUS_MINUS"]
    df = raw[keep].copy()
    df = df[df["GP"] > 0].reset_index(drop=True)
    return df


# ── Case-insensitive substring match on player name ───────────────────────────
def fuzzy_find_player(name, player_stats_df):
    """Returns the best-matching player row for the given name, or None."""
    name_lower = name.strip().lower()
    for _, row in player_stats_df.iterrows():
        if name_lower in row["PLAYER_NAME"].lower():
            return row
    return None


# ── Adjust a team stats row for one or more missing players ───────────────────
def apply_injury_adjustments(team_row, missing_names, player_stats_df):
    """
    Subtracts missing players' per-game PPG and plus/minus from team_row,
    then scales off_efficiency proportionally. Returns (adjusted_row, log).
    """
    row       = team_row.copy()
    orig_ppg  = float(row["ppg"])
    log       = []    # list of human-readable adjustment strings

    for name in missing_names:
        player = fuzzy_find_player(name, player_stats_df)

        if player is None:
            log.append(f"  [INJURY] Player not found: '{name}' — skipped")
            continue

        ppg_loss = float(player["PTS"]) / float(player["GP"])
        pm_loss  = float(player["PLUS_MINUS"]) / float(player["GP"])

        row["ppg"]             = max(float(row["ppg"])             - ppg_loss, PPG_FLOOR)
        row["plus_minus_avg"]  = float(row["plus_minus_avg"])      - pm_loss
        row["opp_ppg"]         = row["ppg"] - row["plus_minus_avg"]

        log.append(
            f"  [INJURY] {player['PLAYER_NAME']} ({player['TEAM_ABBREVIATION']})"
            f"  ppg -{ppg_loss:.1f}  pm -{pm_loss:+.1f}"
        )

    # Scale off_efficiency proportionally to the PPG drop
    if orig_ppg > 0:
        row["off_efficiency"] = float(row["off_efficiency"]) * (
            float(row["ppg"]) / orig_ppg
        )

    return row, log
