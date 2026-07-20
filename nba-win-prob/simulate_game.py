import requests

# ── Settings ──────────────────────────────────────────────────────────────────
SERVER_URL = "http://localhost:5000"

STAT_ROWS = [
    ("Win %",         "win_pct",           ".1%"),
    ("PPG",           "ppg",               ".1f"),
    ("Opp PPG",       "opp_ppg",           ".1f"),
    ("Net Rating",    "plus_minus_avg",    "+.1f"),
    ("Last 10 Win %", "last10_win_pct",    ".1%"),
    ("Home Win %",    "home_game_win_pct", ".1%"),
    ("Away Win %",    "away_game_win_pct", ".1%"),
    ("Off Efficiency","off_efficiency",    ".1f"),
    ("TOV Rate",      "tov_rate",          ".1f"),
]


# ── Confidence tier based on win probability margin ───────────────────────────
def confidence_label(prob):
    """Returns a descriptive tier for a given win probability."""
    margin = abs(prob - 0.5)
    if margin >= 0.20:
        return "Strong favorite"
    elif margin >= 0.10:
        return "Lean"
    else:
        return "Toss-up"


# ── Side-by-side stats table for both teams ───────────────────────────────────
def print_stats_comparison(away_stats, home_stats, away_name, home_name):
    """Prints a formatted side-by-side stat breakdown for both teams."""
    col = 20
    print(f"\n  {'Stat':<16}  {away_name:^{col}}  {home_name:^{col}}")
    print(f"  {'─' * 16}  {'─' * col}  {'─' * col}")
    for label, key, fmt in STAT_ROWS:
        a_val = away_stats.get(key)
        h_val = home_stats.get(key)
        a_str = format(a_val, fmt) if a_val is not None else "N/A"
        h_str = format(h_val, fmt) if h_val is not None else "N/A"
        print(f"  {label:<16}  {a_str:^{col}}  {h_str:^{col}}")


# ── Main prediction display ───────────────────────────────────────────────────
def print_prediction(result, away_stats, home_stats):
    """Prints win probability bar, predicted winner, confidence, and stat table."""
    home      = result["home_team"]
    away      = result["away_team"]
    home_prob = result["home_win_prob"]
    away_prob = result["away_win_prob"]

    bar_len   = 40
    home_fill = int(home_prob * bar_len)
    bar       = "█" * home_fill + "░" * (bar_len - home_fill)

    print("\n" + "─" * 58)
    print(f"  {away:^26} @  {home:^26}")
    print("─" * 58)
    print(f"  [{bar}]")
    print(f"  {away:<26}    {home:>26}")
    print(f"  {f'{away_prob:.1%}':<26}    {f'{home_prob:.1%}':>25}")
    print("─" * 58)

    winner = home if home_prob > 0.5 else away
    conf   = max(home_prob, away_prob)
    tier   = confidence_label(home_prob)
    print(f"  Predicted winner : {winner}  ({conf:.1%} — {tier})")

    margin = result.get("home_margin")
    if margin is not None:
        favored, by = (home, margin) if margin >= 0 else (away, -margin)
        print(f"  Predicted spread : {favored} by {by:.1f}")

    adjustments = result.get("adjustments", [])
    if adjustments:
        print("\n  Injuries applied (auto-detected + manual):")
        for line in adjustments:
            print(line)

    if away_stats and home_stats:
        print_stats_comparison(away_stats, home_stats, away, home)

    print()


# ── Fetch teams with stats from server ────────────────────────────────────────
def get_teams():
    """Fetches all teams from the server and displays them in 3 columns."""
    resp  = requests.get(f"{SERVER_URL}/teams", timeout=5)
    teams = resp.json().get("teams", [])

    col_width = 32
    n_cols    = 3
    print("\nAvailable teams:")
    for i in range(0, len(teams), n_cols):
        row  = teams[i : i + n_cols]
        line = "".join(
            f"  {t['TEAM_NAME']} ({t['TEAM_ABBREVIATION']})".ljust(col_width + 2)
            for t in row
        )
        print(line)

    return teams


# ── Fuzzy team lookup against the fetched list ────────────────────────────────
def find_team_stats(name, teams):
    """Finds a team by name or abbreviation match; returns its stats dict."""
    name_lower = name.strip().lower()
    for t in teams:
        if (
            name_lower in t["TEAM_NAME"].lower()
            or name_lower == t["TEAM_ABBREVIATION"].lower()
        ):
            return t
    return None


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    """Prompts the user for two teams and prints the win probability prediction."""
    print("\n── NBA Pre-Game Predictor ──")

    try:
        resp = requests.get(f"{SERVER_URL}/health", timeout=5)
        resp.raise_for_status()
    except Exception:
        print(
            f"[ERROR] Cannot connect to server at {SERVER_URL}.\n"
            "Make sure server/app.py is running first."
        )
        return

    teams = get_teams()

    print()
    away_input = input("Enter away team name or abbreviation: ").strip()
    home_input = input("Enter home team name or abbreviation: ").strip()

    print(
        "\nOut players are auto-detected from the live injury report. "
        "Add any extras below (or press Enter to skip)."
    )
    away_out_raw = input(
        f"Additional {away_input} players out? (comma-separated): "
    ).strip()
    home_out_raw = input(
        f"Additional {home_input} players out? (comma-separated): "
    ).strip()

    away_injuries = [p.strip() for p in away_out_raw.split(",") if p.strip()]
    home_injuries = [p.strip() for p in home_out_raw.split(",") if p.strip()]

    resp = requests.post(
        f"{SERVER_URL}/predict",
        json={
            "home_team":     home_input,
            "away_team":     away_input,
            "home_injuries": home_injuries or None,
            "away_injuries": away_injuries or None,
        },
        timeout=5,
    )
    result = resp.json()

    if "error" in result:
        print(f"\n[ERROR] {result['error']}")
        return

    away_stats = find_team_stats(away_input, teams)
    home_stats = find_team_stats(home_input, teams)

    print_prediction(result, away_stats, home_stats)


if __name__ == "__main__":
    main()
