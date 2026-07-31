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
  "home_margin": 5.8,
  "adjustments": [
    "  [INJURY] LeBron James (LAL)  ppg -27.0  pm -+6.1"
  ]
}
```
`home_margin` is the model's predicted point spread (positive = home favored, negative =
away favored). It's `null` until `calibrate_spread.py` has been run once.

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

## Predicted Spread

`calibrate_spread.py` fits a conversion from win probability to a point spread using real
historical game margins (no retraining of the win-prob model required). Run it once after
training, and again any time you retrain:

```bash
python nba-win-prob/build_features.py
python nba-win-prob/train_model.py
python nba-win-prob/calibrate_spread.py
```

See `features.md` for the formula and leakage-safety notes.

---

## Vegas Odds (DraftKings / FanDuel)

`vegas_odds.py` fetches live spreads from The Odds API for comparison against the model's
own prediction. Requires a free API key:

1. Sign up at https://the-odds-api.com (500 requests/month free)
2. `cp .env.example .env` and set `ODDS_API_KEY=your_key_here`
3. `pip install -r nba-win-prob/requirements.txt`
4. Test it: `python nba-win-prob/vegas_odds.py`

PrizePicks is not available through this data source (no traditional point-spread market).

---

## Dependencies

```bash
pip install -r nba-win-prob/requirements.txt
```

Key packages: `torch`, `nba_api`, `pandas`, `scikit-learn`, `flask`, `flask-socketio`,
`pyarrow`, `scipy`, `python-dotenv`, `requests`

---

## Key Files

| File | Purpose |
|------|---------|
| `nba-win-prob/fetch_data.py` | Downloads team game logs from nba_api |
| `nba-win-prob/build_features.py` | Computes rolling stats, builds matchup parquet |
| `nba-win-prob/train_model.py` | Trains and evaluates the MLP |
| `nba-win-prob/injury_adjust.py` | Injury adjustment logic (player stat deductions) |
| `nba-win-prob/calibrate_spread.py` | Fits win-prob → point-spread conversion |
| `nba-win-prob/vegas_odds.py` | Fetches DraftKings/FanDuel spreads for comparison |
| `nba-win-prob/server/app.py` | Flask REST + WebSocket prediction server |
| `nba-win-prob/server/Translator.py` | Validates/shapes all frontend ⇄ backend traffic |
| `nba-win-prob/simulate_game.py` | CLI predictor client |
| `features.md` | Full feature table, injury layer, spread, and odds docs |
| `Lessons.md` | Running log of bugs fixed and rules to follow |
