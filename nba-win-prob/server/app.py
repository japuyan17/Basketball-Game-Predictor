import os
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
 
# ── Settings ──────────────────────────────────────────────────────────────────
MODEL_PATH  = "../models/best_model.pt"
SCALER_PATH = "../models/scaler.npy"
FEATURE_COLS = ["score_diff", "secs_left", "period", "home_foul_diff", "momentum"]
PORT = 5000
 
 
# ── Model architecture (must match train_model.py exactly) ───────────────────
class WinProbModel(nn.Module):
    """
    Same architecture as in train_model.py.
    We redefine it here so the server has no dependency on the training file.
    """
    def __init__(self, input_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
 
    def forward(self, x):
        return self.net(x)
 
 
# ── Load model and scaler at startup ─────────────────────────────────────────
def load_model():
    """
    Loads the saved model weights from best_model.pt.
    model.eval() switches off Dropout so predictions are deterministic.
    We call this once when the server starts, not on every request.
    """
    model = WinProbModel(input_dim=len(FEATURE_COLS))
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()
    print(f"[MODEL] Loaded from {MODEL_PATH}")
    return model
 
 
def load_scaler():
    """
    Loads the mean and scale saved during training.
    Every feature vector that comes in must be scaled the same way
    the training data was — otherwise predictions will be garbage.
    """
    data = np.load(SCALER_PATH)
    mean  = data[0]
    scale = data[1]
    print(f"[SCALER] Loaded from {SCALER_PATH}")
    return mean, scale
 
 
def scale_features(raw_features, mean, scale):
    """
    Applies the same StandardScaler transform used during training:
      scaled = (value - mean) / std
    Input:  list of 5 raw feature values
    Output: numpy array of 5 scaled values
    """
    arr = np.array(raw_features, dtype=np.float32)
    return (arr - mean) / scale
 
 
def predict(features_raw, model, scaler_mean, scaler_scale):
    """
    Full prediction pipeline:
      1. Scale the raw features
      2. Convert to a PyTorch tensor with shape (1, 5)
      3. Run through the model
      4. Return a single float between 0 and 1
    """
    scaled  = scale_features(features_raw, scaler_mean, scaler_scale)
    tensor  = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)  # (1, 5)
    with torch.no_grad():
        prob = model(tensor).item()
    return round(prob, 4)
 
 
# ── Flask + SocketIO setup ────────────────────────────────────────────────────
app      = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")
 
model        = load_model()
scaler_mean, scaler_scale = load_scaler()
 
 
# ── HTTP endpoint: POST /predict ──────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def http_predict():
    """
    REST endpoint for one-off predictions.
    Send a POST request with JSON body:
      { "features": [score_diff, secs_left, period, home_foul_diff, momentum] }
    Returns:
      { "home_win_prob": 0.7431, "away_win_prob": 0.2569 }
 
    Test it from terminal:
      curl -X POST http://localhost:5000/predict \
           -H "Content-Type: application/json" \
           -d '{"features": [5, 300, 4, 1, 2]}'
    """
    data = request.get_json()
 
    if not data or "features" not in data:
        return jsonify({"error": "Missing 'features' key in request body"}), 400
 
    features = data["features"]
    if len(features) != len(FEATURE_COLS):
        return jsonify({
            "error": f"Expected {len(FEATURE_COLS)} features, got {len(features)}",
            "expected_order": FEATURE_COLS
        }), 400
 
    try:
        prob = predict(features, model, scaler_mean, scaler_scale)
        return jsonify({
            "home_win_prob": prob,
            "away_win_prob": round(1 - prob, 4),
            "features_received": dict(zip(FEATURE_COLS, features))
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
 
 
# ── HTTP endpoint: GET /health ────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    """
    Quick check that the server is running.
    Open http://localhost:5000/health in your browser to confirm.
    """
    return jsonify({"status": "ok", "model_loaded": True})
 
 
# ── WebSocket event: game_state ───────────────────────────────────────────────
@socketio.on("game_state")
def handle_game_state(data):
    """
    Listens for a 'game_state' event from any connected client (the dashboard
    or simulate_game.py). Runs inference and emits a 'prediction' event back
    to the same client.
 
    Expected data format:
    {
        "features":   [score_diff, secs_left, period, home_foul_diff, momentum],
        "home_team":  "Lakers",      (optional, passed through for display)
        "away_team":  "Celtics",     (optional)
        "home_score": 87,            (optional)
        "away_score": 82,            (optional)
        "game_clock": "4:23",        (optional)
        "period":     4              (optional)
    }
    """
    features = data.get("features")
 
    if not features or len(features) != len(FEATURE_COLS):
        emit("error", {"message": f"Expected {len(FEATURE_COLS)} features in 'features' key"})
        return
 
    try:
        prob = predict(features, model, scaler_mean, scaler_scale)
 
        emit("prediction", {
            "home_win_prob":  prob,
            "away_win_prob":  round(1 - prob, 4),
            "home_team":      data.get("home_team",  "Home"),
            "away_team":      data.get("away_team",  "Away"),
            "home_score":     data.get("home_score", 0),
            "away_score":     data.get("away_score", 0),
            "game_clock":     data.get("game_clock", ""),
            "period":         data.get("period",     1),
        })
 
    except Exception as e:
        emit("error", {"message": str(e)})
 
 
# ── WebSocket event: connect / disconnect ─────────────────────────────────────
@socketio.on("connect")
def handle_connect():
    print(f"[WS] Client connected")
    emit("connected", {"message": "Connected to NBA Win Probability server"})
 
 
@socketio.on("disconnect")
def handle_disconnect():
    print(f"[WS] Client disconnected")
 
 
# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n[SERVER] Starting on http://localhost:{PORT}")
    print(f"[SERVER] Health check: http://localhost:{PORT}/health")
    print(f"[SERVER] Press Ctrl+C to stop\n")
    socketio.run(app, host="0.0.0.0", port=PORT, debug=False)
 
