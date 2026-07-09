# Model Features

**Seasons:** 2018-19 · 2019-20 · 2020-21 · 2021-22 · 2022-23 · 2023-24 · 2024-25 · 2025-26

All 19 features fed into the MLP for each matchup. Every stat is a rolling
cumulative average computed from games played **before** the current game
(`shift(1)` — no data leakage).

| # | Feature | Team | Description |
|---|---------|------|-------------|
| 1 | `win_pct_home` | Home | Overall season win percentage |
| 2 | `ppg_home` | Home | Points scored per game |
| 3 | `opp_ppg_home` | Home | Points allowed per game |
| 4 | `plus_minus_avg_home` | Home | Average point differential per game |
| 5 | `last10_win_pct_home` | Home | Win % over the last 10 games |
| 6 | `win_pct_away` | Away | Overall season win percentage |
| 7 | `ppg_away` | Away | Points scored per game |
| 8 | `opp_ppg_away` | Away | Points allowed per game |
| 9 | `plus_minus_avg_away` | Away | Average point differential per game |
| 10 | `last10_win_pct_away` | Away | Win % over the last 10 games |
| 11 | `home_split_win_pct` | Home | Win % in home games specifically |
| 12 | `away_split_win_pct` | Away | Win % in away games specifically |
| 13 | `net_rtg_diff` | Both | `plus_minus_avg_home − plus_minus_avg_away` |
| 14 | `home_team_away_ppg` | Home | PPG when this team plays on the road |
| 15 | `away_team_home_ppg` | Away | PPG when this team plays at home |
| 16 | `off_efficiency_home` | Home | Offensive rating — points per 100 possessions |
| 17 | `off_efficiency_away` | Away | Offensive rating — points per 100 possessions |
| 18 | `tov_rate_home` | Home | Turnover rate — turnovers per 100 possessions |
| 19 | `tov_rate_away` | Away | Turnover rate — turnovers per 100 possessions |

## Possession Formula

Possessions are estimated using the standard NBA approximation:

```
poss = FGA − OREB + TOV + 0.44 × FTA
```

This is used as the denominator for `off_efficiency` and `tov_rate`.

## Where each feature is defined

| File | Role |
|------|------|
| `nba-win-prob/build_features.py` | Computes all rolling stats, joins matchups, saves parquet |
| `nba-win-prob/train_model.py` | `FEATURE_COLS` list used during training |
| `nba-win-prob/server/app.py` | `FEATURE_COLS` + `build_feature_vector()` used at inference |

> **Rule:** Any new feature must be added to all three files in the same change.
> See `Lessons.md` entry 9 for details.

---

## Injury Adjustment Layer

Handled by `nba-win-prob/injury_adjust.py` — applied at inference time only. Does **not**
require retraining.

When one or more players are listed as out, the server adjusts the affected team's feature
vector before it hits the model:

| Stat adjusted | How |
|---------------|-----|
| `ppg` | Subtract injured player's `PTS / GP` (floored at 70.0) |
| `plus_minus_avg` | Subtract injured player's `PLUS_MINUS / GP` |
| `opp_ppg` | Recalculated as `ppg − plus_minus_avg` (derived, not raw) |
| `off_efficiency` | Scaled proportionally: `off_efficiency × (new_ppg / orig_ppg)` |

Player stats are fetched from `nba_api.LeagueDashPlayerStats` (current season) at server
startup. Multiple injured players stack — each deduction is applied in sequence.

**POST /predict injury payload:**
```json
{
  "home_team": "Celtics",
  "away_team": "Lakers",
  "home_injuries": ["Jayson Tatum"],
  "away_injuries": ["LeBron James", "Anthony Davis"]
}
```

Player names are matched via case-insensitive substring search. Unrecognized names are
logged and skipped without crashing.
