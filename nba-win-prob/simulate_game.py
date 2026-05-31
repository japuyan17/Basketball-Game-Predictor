import time
import random
import pandas as pd
import socketio
 
# ── Settings ──────────────────────────────────────────────────────────────────
SERVER_URL   = "http://localhost:5000"
DATA_PATH    = "data/features.parquet"
DELAY        = 0.4   # seconds between each play-by-play event
 
# Team name pairs for display — randomly picked if not specified
TEAM_PAIRS = [
    ("Lakers",    "Celtics"),
    ("Warriors",  "Bulls"),
    ("Heat",      "Bucks"),
    ("Nets",      "Suns"),
    ("Nuggets",   "Clippers"),
]
 
 
# ── Helpers ───────────────────────────────────────────────────────────────────
def seconds_to_clock(secs_left):
    """Convert total seconds remaining to a readable clock string like '4:23'."""
    secs_left = max(0, int(secs_left))
    if secs_left == 0:
        return "0:00"
    period_secs = secs_left % 720 if secs_left <= 2880 else secs_left % 300
    mins = period_secs // 60
    secs = period_secs % 60
    return f"{mins}:{secs:02d}"
 
 
def secs_to_period(secs_left):
    """Infer the period number from total seconds remaining."""
    if secs_left > 2160: return 1
    if secs_left > 1440: return 2
    if secs_left > 720:  return 3
    if secs_left > 0:    return 4
    return 5  # overtime
 
 
def pick_game(df):
    """
    Pick an interesting game to simulate — one with a lead change in the
    4th quarter so the probability chart is dramatic.
    """
    game_ids = df["GAME_ID"].unique()
 
    for gid in random.sample(list(game_ids), min(50, len(game_ids))):
        game = df[df["GAME_ID"] == gid].reset_index(drop=True)
        q4   = game[game["period"] == 4]
        if len(q4) < 20:
            continue
        # Look for a lead change in Q4
        diffs = q4["score_diff"].values
        if (diffs.min() < -3) and (diffs.max() > 3):
            print(f"[GAME] Found dramatic game: {gid}")
            return game
 
    # Fallback: just pick a random game
    gid = random.choice(list(game_ids))
    print(f"[GAME] Using game: {gid}")
    return df[df["GAME_ID"] == gid].reset_index(drop=True)
 
 
# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Load features
    print("[LOAD] Reading features.parquet...")
    df = pd.read_parquet(DATA_PATH)
    print(f"  {df['GAME_ID'].nunique():,} games available")
 
    # Pick a game
    game = pick_game(df)
    home_team, away_team = random.choice(TEAM_PAIRS)
    total_rows = len(game)
    print(f"  Simulating {total_rows} events as {home_team} vs {away_team}\n")
 
    # Connect to server
    client = socketio.Client()
 
    @client.on("connect")
    def on_connect():
        print("[WS] Connected to server")
 
    @client.on("prediction")
    def on_prediction(data):
        prob  = data["home_win_prob"]
        score = f"{data['home_score']} - {data['away_score']}"
        clock = data["game_clock"]
        period = data["period"]
        bar_len = 30
        filled = int(prob * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"  Q{period} {clock:>5}  |  {home_team} {score} {away_team}  |  [{bar}] {prob:.0%}")
 
    @client.on("error")
    def on_error(data):
        print(f"[ERROR] {data['message']}")
 
    client.connect(SERVER_URL)
 
    # ── Replay the game event by event ────────────────────────────────────────
    print(f"{'Period':>8}  {'Clock':>6}  {'Score':^15}  {'Home Win Prob':^34}")
    print("-" * 75)
 
    # Reconstruct running home/away scores from score_diff
    # We don't have raw scores in features.parquet so we approximate
    base_score = 50  # starting point for display purposes
 
    for i, row in game.iterrows():
        score_diff  = float(row["score_diff"])
        secs_left   = float(row["secs_left"])
        period      = int(row["period"])
        foul_diff   = float(row["home_foul_diff"])
        momentum    = float(row["momentum"])
        game_clock  = seconds_to_clock(secs_left)
 
        # Approximate display scores from differential
        home_score = int(base_score + score_diff / 2)
        away_score = int(base_score - score_diff / 2)
 
        client.emit("game_state", {
            "features":   [score_diff, secs_left, period, foul_diff, momentum],
            "home_team":  home_team,
            "away_team":  away_team,
            "home_score": home_score,
            "away_score": away_score,
            "game_clock": game_clock,
            "period":     period,
        })
 
        time.sleep(DELAY)
 
    print("\n[DONE] Game simulation complete.")
    client.disconnect()
 
 
if __name__ == "__main__":
    main()
