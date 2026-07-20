import os
import requests
from dotenv import load_dotenv

load_dotenv()

# ── Settings ──────────────────────────────────────────────────────────────────
ODDS_API_KEY = os.environ.get("ODDS_API_KEY")
ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
REQUEST_TIMEOUT = 10

# PrizePicks is a DFS pick'em platform, not a licensed sportsbook — it has no
# point-spread market and isn't listed as a bookmaker by The Odds API, so it
# is intentionally excluded here.
BOOKMAKERS = ["draftkings", "fanduel"]


# ── Fetch current NBA point spreads from The Odds API ─────────────────────────
def fetch_vegas_spreads(bookmakers=BOOKMAKERS):
    """
    Fetches upcoming NBA games with point spreads from the given sportsbooks
    via The Odds API. Returns a list of dicts:
    {game_id, commence_time, home_team, away_team, spreads, home_spread_avg,
    num_books}. "spreads" maps bookmaker key -> home-team spread (negative =
    home favored, standard Vegas sign convention).
    """
    if not ODDS_API_KEY:
        raise RuntimeError(
            "ODDS_API_KEY is not set. Sign up for a free key at "
            "https://the-odds-api.com and add it to a .env file:\n"
            "  ODDS_API_KEY=your_key_here"
        )

    response = requests.get(
        ODDS_API_URL,
        params={
            "apiKey":     ODDS_API_KEY,
            "bookmakers": ",".join(bookmakers),
            "markets":    "spreads",
            "oddsFormat": "american",
        },
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()

    remaining = response.headers.get("x-requests-remaining")
    if remaining is not None:
        print(f"[VEGAS] The Odds API requests remaining this month: {remaining}")

    games = []
    for event in response.json():
        home_team = event["home_team"]
        away_team = event["away_team"]

        spreads = {}
        for bookmaker in event.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market["key"] != "spreads":
                    continue
                for outcome in market["outcomes"]:
                    if outcome["name"] == home_team:
                        spreads[bookmaker["key"]] = outcome["point"]

        if not spreads:
            continue

        games.append({
            "game_id":         event["id"],
            "commence_time":   event["commence_time"],
            "home_team":       home_team,
            "away_team":       away_team,
            "spreads":         spreads,
            "home_spread_avg": round(sum(spreads.values()) / len(spreads), 1),
            "num_books":       len(spreads),
        })

    return games


# ── Fuzzy team-name lookup against fetched Vegas games ─────────────────────────
def find_vegas_spread(home_name, away_name, games):
    """
    Finds the Vegas spread for a given matchup by case-insensitive substring
    match on team names. Returns the matching game dict, or None if no
    upcoming game with both teams is listed.
    """
    home_lower = home_name.strip().lower()
    away_lower = away_name.strip().lower()

    for game in games:
        if (
            home_lower in game["home_team"].lower()
            and away_lower in game["away_team"].lower()
        ):
            return game
    return None


# ── Standalone test run ────────────────────────────────────────────────────────
def main():
    """Fetches and prints all currently available NBA spreads, per book."""
    games = fetch_vegas_spreads()
    print(f"\n[VEGAS] {len(games)} games with spreads available:\n")
    for g in games:
        per_book = "  ".join(
            f"{book}: {point:+.1f}" for book, point in g["spreads"].items()
        )
        print(f"  {g['away_team']:<28} @ {g['home_team']:<28}")
        print(f"    {per_book}   (avg {g['home_spread_avg']:+.1f})")


if __name__ == "__main__":
    main()
