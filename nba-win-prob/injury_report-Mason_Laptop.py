import time

import requests

# ── Settings ──────────────────────────────────────────────────────────────────
ESPN_INJURY_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
)
OUT_STATUS      = "Out"    # only fully-out players are auto-deducted
DEFAULT_TTL     = 1800     # seconds a cached report stays fresh (30 minutes)
RETRY_BACKOFF   = 60       # seconds to wait before retrying after a failed fetch
REQUEST_TIMEOUT = 10       # seconds before an ESPN request is abandoned


# ── Fetch the raw league-wide injury report from ESPN ─────────────────────────
def fetch_injury_report(timeout=REQUEST_TIMEOUT):
    """Fetches ESPN's NBA injury JSON; returns the parsed dict or None on error."""
    try:
        response = requests.get(ESPN_INJURY_URL, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except Exception as error:
        print(f"[INJURY] Fetch failed: {error}")
        return None


# ── Extract out-players grouped by full team name ─────────────────────────────
def parse_out_players(report):
    """Returns {team_name_lower: [player_name, ...]} for players listed as Out."""
    out_by_team = {}
    for team in report.get("injuries", []):
        team_name = team.get("displayName", "").strip()
        if not team_name:
            continue
        names = []
        for entry in team.get("injuries", []):
            if entry.get("status") != OUT_STATUS:
                continue
            athlete = entry.get("athlete") or {}
            name    = athlete.get("displayName")
            if name:
                names.append(name)
        out_by_team[team_name.lower()] = names
    return out_by_team


# ══════════════════════════════════════════════════════════════════════════════
# InjuryReportCache — caches the parsed report and refreshes it on a TTL
# ══════════════════════════════════════════════════════════════════════════════
class InjuryReportCache:
    """Caches ESPN out-player data and refreshes it once the TTL expires."""

    # Store the TTL and initialize an empty, never-fetched cache.
    def __init__(self, ttl_seconds=DEFAULT_TTL):
        self.ttl_seconds   = ttl_seconds
        self._out_by_team  = {}
        self._fetched_at   = 0.0   # last successful fetch (epoch seconds)
        self._last_attempt = 0.0   # last fetch attempt, success or failure

    # Decide whether a refresh is due without hammering ESPN after a failure.
    def _should_refresh(self):
        now = time.time()
        if now - self._last_attempt < RETRY_BACKOFF:
            return False
        return (now - self._fetched_at) >= self.ttl_seconds

    # Fetch and parse a fresh report, keeping old data if the fetch fails.
    def refresh(self):
        """Forces a fetch; returns True on success, False if ESPN was unreachable."""
        self._last_attempt = time.time()
        report = fetch_injury_report()
        if report is None:
            return False
        self._out_by_team = parse_out_players(report)
        self._fetched_at  = time.time()
        return True

    # Look up a team's out-players, refreshing first if the cache is stale.
    def get_out_players(self, team_name):
        """Returns the list of out-player names for the given full team name."""
        if self._should_refresh():
            self.refresh()

        key = (team_name or "").strip().lower()
        if not key:
            return []
        if key in self._out_by_team:
            return list(self._out_by_team[key])

        # Fall back to a loose match (e.g. "Lakers" vs "Los Angeles Lakers")
        for espn_name, names in self._out_by_team.items():
            if key in espn_name or espn_name in key:
                return list(names)
        return []
