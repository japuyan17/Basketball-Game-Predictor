# Lessons Learned

Running log of issues fixed and what to remember so they don't recur.
**Always read this file before writing or changing code in this project.**

Format: each entry = the problem, the fix, and the rule to follow going forward.

---

## 1. Keep the two `WinProbModel` definitions in sync

- **Problem:** `train_model.py` was updated to `BatchNorm1d + GELU`, but the
  copy of `WinProbModel` in `server/app.py` still used `ReLU` only. Loading
  the trained `best_model.pt` into the mismatched server architecture would
  fail (missing/unexpected keys) or silently produce garbage predictions.
- **Fix:** Updated the server's `WinProbModel.net` to match training exactly.
- **Rule:** `server/app.py` intentionally redefines `WinProbModel` (no import
  from training). **Any architecture change in `train_model.py` must be
  mirrored in `server/app.py` in the same change.**

## 2. Use `weights_only=True` with `torch.load`

- **Problem:** `torch.load(path)` without `weights_only=True` triggers a
  deprecation warning in PyTorch ≥2.x and is unsafe (can execute arbitrary
  pickled code); the default flips in future versions.
- **Fix:** `torch.load(MODEL_PATH, map_location="cpu", weights_only=True)` in
  both `server/app.py` and `train_model.py`.
- **Rule:** Always load model state dicts with `weights_only=True`.

## 3. Mark intentionally unused variables with `_`

- **Problem:** `fig, axes = plt.subplots(...)` and
  `..., scaler = load_data()` left `fig`/`scaler` unused (linter hints).
- **Fix:** Replaced the unused names with `_` (`_, axes = ...`).
- **Rule:** Use `_` for values you must unpack but don't use.

## 4. Win-prob bar: don't size segments by independently-rounded percents

- **Problem (frontend, WinProbBar.jsx):** Each segment width is set from a
  separately `Math.round`-ed percentage. With extreme model values (e.g.
  99.6% / 0.4%) the rounded widths plus the CSS `min-width` can sum to >100%
  and overflow the bar.
- **Status:** Acceptable for v1 (mock data is exact 74/26). Tighten when
  real `/predict` data is wired in (iteration 2).
- **Rule:** When two values should fill 100%, derive the second width as
  `100 - first` (or `flex`), don't round both independently.

## 5. Use PlayByPlayV3, not V2 (V2 returns empty JSON)

- **Problem:** `fetch_data.py` used `PlayByPlayV2`, which the NBA API has
  deprecated — it now returns empty JSON, so every game raised a
  `'resultSet'` KeyError (GitHub issue swar/nba_api#591).
- **Fix:** Switched to `PlayByPlayV3` in `fetch_data.py` and migrated
  `build_features.py` to V3's schema.
- **V2 → V3 column map (important for feature code):**
  - `SCORE` ("away - home" string) → `scoreHome` + `scoreAway` (two cols)
  - `PCTIMESTRING` ("8:42") → `clock` (ISO 8601, e.g. "PT08M42.00S")
  - `PERIOD` → `period`; `GAME_ID` → `gameId`
  - `HOMEDESCRIPTION` / `VISITORDESCRIPTION` → single `description` +
    `location` ("h" / "v")
- **Rule:** Use V3 endpoints. Parse the V3 clock with
  `re.match(r"PT0*(\d+)M0*([\d.]+)S", clock)`. Detect home/away events via
  `location == "h"` / `"v"`, not separate description columns.

## 6. Re-run `build_features.py` whenever feature columns change

- **Problem:** `train_model.py` raised `KeyError: ['home_split_win_pct',
  'away_split_win_pct', 'net_rtg_diff']` because the saved
  `data/matchup_features.parquet` was generated before those columns were
  added to `build_features.py`.
- **Fix:** Re-ran `build_features.py` to regenerate the parquet with the new
  columns, then re-ran `train_model.py`.
- **Rule:** Any time `FEATURE_COLS` grows in `build_features.py`, you must
  regenerate `data/matchup_features.parquet` and `data/team_stats_latest.parquet`
  before training. The old parquet will be missing columns and crash `dropna`.

## 7. Upgrade model architecture: single-layer → MLP with BatchNorm

- **What changed:** `WinProbModel` was a single `nn.Linear(N, 1) + Sigmoid`
  (logistic regression). Upgraded to two hidden layers with BatchNorm:
  `Linear(N→64) → BatchNorm1d(64) → ReLU → Dropout(0.3) →
  Linear(64→32) → BatchNorm1d(32) → ReLU → Dropout(0.2) →
  Linear(32→1) → Sigmoid`.
- **Why BatchNorm here:** Normalizes activations between layers, stabilizes
  training on tabular data, and typically gives a 1–2% accuracy lift.
- **Rule:** `server/app.py` must always mirror `train_model.py`'s architecture
  exactly (see Lesson 1). After any architecture change, delete the old
  `best_model.pt` and retrain from scratch.

## 8. Add `ReduceLROnPlateau` alongside early stopping

- **What changed:** Added `torch.optim.lr_scheduler.ReduceLROnPlateau(
  optimizer, mode="min", factor=0.5, patience=10)`. The scheduler halves the
  LR after 10 epochs of no val-loss improvement; early stopping still fires at
  PATIENCE=20.
- **Why:** Gives the optimizer a chance to fine-tune at a lower LR before
  giving up entirely. Without it, the model exits at the same LR it started
  with, leaving potential improvement on the table.
- **Rule:** Call `scheduler.step(val_loss)` every epoch after the validation
  pass, before checking early stopping. Print `current_lr` each epoch so LR
  drops are visible in the log.

## 9. Feature column count must stay in sync across all three files

- **Problem:** `FEATURE_COLS` exists in three places:
  `build_features.py` (feature list + parquet output),
  `train_model.py` (training), `server/app.py` (inference).
  Adding a feature to one without updating the others causes shape mismatches
  at load time or wrong inference results.
- **Rule:** When adding features, update all three files in the same change:
  1. `build_features.py` — compute the column + add to `feature_cols` + add to
     `save_latest_team_stats` keep list
  2. `train_model.py` — add to `FEATURE_COLS`
  3. `server/app.py` — add to `FEATURE_COLS` + add to `build_feature_vector`
     raw array in the same index position

## 10. Home/away split stats reuse already-computed denominators

- **Pattern:** When computing split stats (e.g., PPG in away games only),
  reuse `cum_away_games` / `cum_home_games` that were already computed for
  the win% splits — don't recompute them.
- **Formula:**
  ```python
  df["pts_away_contrib"] = df["PTS"] * df["is_away"]
  df["cum_pts_away"]     = grp["pts_away_contrib"].transform(
      lambda x: x.cumsum().shift(1)
  ).fillna(0)
  df["away_ppg"] = df["cum_pts_away"] / df["cum_away_games"].replace(0, np.nan)
  ```
- **Rule:** All split stats use `.cumsum().shift(1)` to prevent leakage.
  Replace 0-game denominators with `np.nan` so early-season rows produce NaN
  (dropped later by `dropna`) rather than division-by-zero.

## 6. `to_parquet` needs a Parquet engine installed

- **Problem:** `season_df.to_parquet(...)` crashed with `ImportError` from
  pandas `get_engine` — the script fetched data fine but couldn't save it
  because no Parquet engine was installed in the Python 3.14 env.
- **Fix:** `python -m pip install pyarrow` (cp314 wheel = `pyarrow 24.0.0`).
- **Rule:** `pandas.to_parquet`/`read_parquet` require `pyarrow` (or
  `fastparquet`). It's in `requirements.txt`, but confirm it's actually
  installed in the interpreter being run (`python -c "import pyarrow"`),
  especially on brand-new Python versions where wheels may lag.
