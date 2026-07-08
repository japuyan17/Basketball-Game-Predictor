import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit

# ── Settings ──────────────────────────────────────────────────────────────────
_DIR            = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH      = os.path.join(_DIR, "..", "..", "models", "best_model.pt")
SCALER_PATH     = os.path.join(_DIR, "..", "..", "models", "scaler.npy")
TEAM_STATS_PATH = os.path.join(_DIR, "..", "..", "data",   "team_stats_latest.parquet")
PORT            = 5000

FEATURE_COLS = [
    "win_pct_home",        "ppg_home",
    "opp_ppg_home",        "plus_minus_avg_home",   "last10_win_pct_home",
    "win_pct_away",        "ppg_away",
    "opp_ppg_away",        "plus_minus_avg_away",   "last10_win_pct_away",
    "home_split_win_pct",  "away_split_win_pct",    "net_rtg_diff",
    "home_team_away_ppg",  "away_team_home_ppg",
    "off_efficiency_home", "off_efficiency_away",
    "tov_rate_home",       "tov_rate_away",
]
N_FEATURES = len(FEATURE_COLS)


# ── Model: must match train_model.py exactly ──────────────────────────────────
class WinProbModel(nn.Module):
    """Three hidden layers with BatchNorm, ReLU, and Dropout; outputs a win prob."""

    def __init__(self, input_dim=N_FEATURES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


# ── Load model, scaler, and team stats at startup ─────────────────────────────
def load_assets():
    """Loads the PyTorch model, numpy scaler, and team stats table."""
    model = WinProbModel()
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    )
    model.eval()

    scaler_data   = np.load(SCALER_PATH)
    scaler_mean   = scaler_data[0]
    scaler_scale  = scaler_data[1]

    team_stats = pd.read_parquet(TEAM_STATS_PATH)

    print(f"[MODEL]  Loaded from {MODEL_PATH}")
    print(f"[SCALER] Loaded from {SCALER_PATH}")
    print(f"[TEAMS]  {len(team_stats)} teams loaded")

    return model, scaler_mean, scaler_scale, team_stats


# ── Fuzzy team name lookup ────────────────────────────────────────────────────
def find_team(name, team_stats):
    """
    Finds a team by matching the input against full team name or abbreviation
    (case-insensitive). Returns the matching row as a Series or None.
    """
    name_lower = name.strip().lower()
    for _, row in team_stats.iterrows():
        if (
            name_lower in row["TEAM_NAME"].lower()
            or name_lower == row["TEAM_ABBREVIATION"].lower()
        ):
            return row
    return None


# ── Build and scale the feature vector ───────────────────────────────────────
def build_feature_vector(home_row, away_row, scaler_mean, scaler_scale):
    """Assembles and scales the 19-feature matchup vector (order must match FEATURE_COLS)."""
    net_rtg_diff = home_row["plus_minus_avg"] - away_row["plus_minus_avg"]
    raw = np.array([
        home_row["win_pct"],           home_row["ppg"],
        home_row["opp_ppg"],           home_row["plus_minus_avg"],  home_row["last10_win_pct"],
        away_row["win_pct"],           away_row["ppg"],
        away_row["opp_ppg"],           away_row["plus_minus_avg"],  away_row["last10_win_pct"],
        home_row["home_game_win_pct"], away_row["away_game_win_pct"],
        net_rtg_diff,
        home_row["away_ppg"],          away_row["home_ppg"],
        home_row["off_efficiency"],    away_row["off_efficiency"],
        home_row["tov_rate"],          away_row["tov_rate"],
    ], dtype=np.float32)
    return (raw - scaler_mean) / scaler_scale


# ── Core prediction logic ─────────────────────────────────────────────────────
def predict_matchup(home_name, away_name, model, scaler_mean, scaler_scale,
                    team_stats):
    """
    Looks up both teams, builds the scaled feature vector, runs inference,
    and returns a result dict with win probabilities.
    """
    home_row = find_team(home_name, team_stats)
    away_row = find_team(away_name, team_stats)

    if home_row is None:
        return {"error": f"Team not found: '{home_name}'"}
    if away_row is None:
        return {"error": f"Team not found: '{away_name}'"}

    scaled = build_feature_vector(
        home_row, away_row, scaler_mean, scaler_scale
    )
    tensor = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        prob = model(tensor).item()

    return {
        "home_team":     home_row["TEAM_NAME"],
        "away_team":     away_row["TEAM_NAME"],
        "home_win_prob": round(prob, 4),
        "away_win_prob": round(1 - prob, 4),
    }


# ── Flask + SocketIO ──────────────────────────────────────────────────────────
app      = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

model, scaler_mean, scaler_scale, team_stats = load_assets()


# ── POST /predict ─────────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def http_predict():
    """
    REST endpoint for pre-game predictions.
    Body: { "home_team": "Lakers", "away_team": "Celtics" }
    Returns home and away win probabilities.
    """
    data = request.get_json()
    if not data or "home_team" not in data or "away_team" not in data:
        return jsonify(
            {"error": "Body must include 'home_team' and 'away_team'"}
        ), 400

    result = predict_matchup(
        data["home_team"], data["away_team"],
        model, scaler_mean, scaler_scale, team_stats,
    )

    if "error" in result:
        return jsonify(result), 404

    return jsonify(result)


# ── GET /health ───────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    """Returns server status and number of teams available."""
    return jsonify({
        "status":       "ok",
        "model":        "mlp_pytorch_3layer",
        "features":     N_FEATURES,
        "teams_loaded": len(team_stats),
    })


# ── GET /teams ────────────────────────────────────────────────────────────────
@app.route("/teams", methods=["GET"])
def list_teams():
    """Returns all available teams with their key stats for display."""
    keep_cols = [
        "TEAM_NAME", "TEAM_ABBREVIATION",
        "win_pct", "ppg", "opp_ppg", "plus_minus_avg",
        "last10_win_pct", "home_game_win_pct", "away_game_win_pct",
        "off_efficiency", "tov_rate",
    ]
    available = [c for c in keep_cols if c in team_stats.columns]
    teams = (
        team_stats[available]
        .sort_values("TEAM_NAME")
        .to_dict(orient="records")
    )
    return jsonify({"teams": teams})


# ── WebSocket: predict_game ───────────────────────────────────────────────────
@socketio.on("predict_game")
def handle_predict_game(data):
    """
    Listens for a 'predict_game' event and emits a 'prediction' event back.
    Expected payload: { "home_team": "Lakers", "away_team": "Celtics" }
    """
    result = predict_matchup(
        data.get("home_team", ""),
        data.get("away_team", ""),
        model, scaler_mean, scaler_scale, team_stats,
    )

    if "error" in result:
        emit("error", {"message": result["error"]})
    else:
        emit("prediction", result)


# ── WebSocket: connect / disconnect ──────────────────────────────────────────
@socketio.on("connect")
def handle_connect():
    """Acknowledges a new WebSocket connection."""
    print("[WS] Client connected")
    emit("connected", {"message": "Connected to NBA Game Predictor"})


@socketio.on("disconnect")
def handle_disconnect():
    """Logs when a client disconnects."""
    print("[WS] Client disconnected")


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n[SERVER] Starting on http://localhost:{PORT}")
    print(f"[SERVER] Health : http://localhost:{PORT}/health")
    print(f"[SERVER] Teams  : http://localhost:{PORT}/teams")
    print(f"[SERVER] Press Ctrl+C to stop\n")
    socketio.run(app, host="0.0.0.0", port=PORT, debug=False)
