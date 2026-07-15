import re

# ── Limits for incoming frontend payloads ─────────────────────────────────────
MAX_NAME_LENGTH    = 40
MAX_INJURY_PLAYERS = 5

# Whitelist: letters, digits, spaces, periods, hyphens, apostrophes only.
# Anything else (brackets, quotes, semicolons, etc.) looks like code — reject.
_NAME_PATTERN = re.compile(r"^[A-Za-z0-9 .'\-]+$")


# ── Sanitize a single team or player name ─────────────────────────────────────
def sanitize_name(value):
    """Returns the cleaned name string, or None if it fails validation."""
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not cleaned or len(cleaned) > MAX_NAME_LENGTH:
        return None
    if not _NAME_PATTERN.match(cleaned):
        return None
    return cleaned


# ── Validate an optional list of injured player names ─────────────────────────
def parse_injury_list(raw, label):
    """Returns (list_of_names, error) — empty list is valid, bad input is not."""
    if raw is None:
        return [], None
    if not isinstance(raw, list):
        return None, f"'{label}' must be a list of player names"
    if len(raw) > MAX_INJURY_PLAYERS:
        return None, f"'{label}' cannot exceed {MAX_INJURY_PLAYERS} players"

    names = []
    for item in raw:
        cleaned = sanitize_name(item)
        if cleaned is None:
            return None, f"Invalid player name in '{label}': {item!r}"
        names.append(cleaned)
    return names, None


# ── Validate the full /predict request body from the frontend ─────────────────
def parse_predict_request(data):
    """Returns (payload_dict, error) — payload is None when validation fails."""
    if not isinstance(data, dict):
        return None, "Request body must be a JSON object"

    home_team = sanitize_name(data.get("home_team"))
    away_team = sanitize_name(data.get("away_team"))

    if home_team is None or away_team is None:
        return None, "Body must include valid 'home_team' and 'away_team'"

    home_injuries, error = parse_injury_list(
        data.get("home_injuries"), "home_injuries"
    )
    if error:
        return None, error

    away_injuries, error = parse_injury_list(
        data.get("away_injuries"), "away_injuries"
    )
    if error:
        return None, error

    payload = {
        "home_team":     home_team,
        "away_team":     away_team,
        "home_injuries": home_injuries or None,
        "away_injuries": away_injuries or None,
    }
    return payload, None


# ── Shape the backend prediction result for the frontend ──────────────────────
def build_prediction_response(result):
    """Merges frontend-friendly camelCase keys into the backend result dict."""
    if "error" in result:
        return result

    return {
        **result,
        "homeTeam":    result["home_team"],
        "awayTeam":    result["away_team"],
        "homeProb":    result["home_win_prob"],
        "awayProb":    result["away_win_prob"],
        "adjustments": [line.strip() for line in result.get("adjustments", [])],
    }
