# Basketball Game Predictor

Pre-game NBA win probability model. Enter two teams and get a predicted winner with
confidence tier and a side-by-side stat breakdown.

---

## Model

**Architecture:** 3-hidden-layer MLP with BatchNorm, ReLU, and Dropout.

```
Input (19 features)
  → Linear(19→128) + BatchNorm + ReLU + Dropout(0.3)
  → Linear(128→64) + BatchNorm + ReLU + Dropout(0.2)
  → Linear(64→32)  + BatchNorm + ReLU + Dropout(0.1)
  → Linear(32→1) + Sigmoid
Output: P(home team wins) ∈ [0, 1]
```

19 pre-game features — see `features.md` for the full table.

---

## Server Endpoints

Base URL: `http://localhost:5000`

| Method | Route | Purpose |
|--------|-------|---------|
| GET | `/health` | Server status + model info |
| GET | `/teams` | All teams with current stats |
| POST | `/predict` | Win probability prediction |

### POST /predict

**Minimal:**
```json
{ "home_team": "Celtics", "away_team": "Lakers" }
```

**With injuries:**
```json
{
  "home_team": "Celtics",
  "away_team": "Lakers",
  "home_injuries": ["Jayson Tatum"],
  "away_injuries": ["LeBron James", "Anthony Davis"]
}
```

**Response:**
```json
{
  "home_team": "Boston Celtics",
  "away_team": "Los Angeles Lakers",
  "home_win_prob": 0.6812,
  "away_win_prob": 0.3188,
  "adjustments": [
    "  [INJURY] LeBron James (LAL)  ppg -27.0  pm -+6.1"
  ]
}
```

---

## How it Works

We take the 19 features (see `features.md`) and train a 3-layer neural network (MLP) that learns what makes a team a winner — offensive rating, defensive rating, turnover rate, home/away splits, and more. The model outputs a probability between 0 and 1, which becomes the confidence score for the predicted winner.

---

## Injury Adjustments

When players are listed as out, the server deducts their per-game scoring and plus/minus
from the team's feature vector before running inference. Player stats are fetched from
`nba_api` automatically at server startup.

Player names use case-insensitive substring matching — "LeBron" matches "LeBron James".
Unrecognized names are logged and skipped.

---

## Dependencies

```bash
pip install -r nba-win-prob/requirements.txt
```

Key packages: `torch`, `nba_api`, `pandas`, `scikit-learn`, `flask`, `flask-socketio`,
`pyarrow`

---

## Key Files

| File | Purpose |
|------|---------|
| `nba-win-prob/fetch_data.py` | Downloads team game logs from nba_api |
| `nba-win-prob/build_features.py` | Computes rolling stats, builds matchup parquet |
| `nba-win-prob/train_model.py` | Trains and evaluates the MLP |
| `nba-win-prob/injury_adjust.py` | Injury adjustment logic (player stat deductions) |
| `nba-win-prob/server/app.py` | Flask REST + WebSocket prediction server |
| `nba-win-prob/simulate_game.py` | CLI predictor client |
| `features.md` | Full feature table + injury adjustment docs |
| `Lessons.md` | Running log of bugs fixed and rules to follow |
