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

## 6. `to_parquet` needs a Parquet engine installed

- **Problem:** `season_df.to_parquet(...)` crashed with `ImportError` from
  pandas `get_engine` — the script fetched data fine but couldn't save it
  because no Parquet engine was installed in the Python 3.14 env.
- **Fix:** `python -m pip install pyarrow` (cp314 wheel = `pyarrow 24.0.0`).
- **Rule:** `pandas.to_parquet`/`read_parquet` require `pyarrow` (or
  `fastparquet`). It's in `requirements.txt`, but confirm it's actually
  installed in the interpreter being run (`python -c "import pyarrow"`),
  especially on brand-new Python versions where wheels may lag.
