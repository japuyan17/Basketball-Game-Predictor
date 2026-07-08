# Frontend Roadmap — NBA Win Probability Dashboard

The frontend is built in phased iterations. Each one stays runnable on
localhost so the layout can be reviewed before any backend wiring.

**Stack:** React + Vite (JavaScript/JSX). Run with `npm run dev`.

---

## Backend API Contract (reference)

The frontend targets the existing Flask + SocketIO server (`../server/app.py`):

- `GET  /health` → `{ "status": "ok", "model_loaded": true }`
- `POST /predict` — body
  `{ "features": [score_diff, secs_left, period, home_foul_diff, momentum] }`
  → `{ "home_win_prob": 0.74, "away_win_prob": 0.26, "features_received": {...} }`
- WebSocket: emit `game_state` → receive `prediction`
  (`home_win_prob`, `away_win_prob`, `home_team`, `away_team`, `home_score`,
  `away_score`, `game_clock`, `period`)

---

## ✅ Iteration 1 — Layout Only (current)

Static, styled dashboard with **mock data** and **no network calls**.

- `Scoreboard` — away/home teams, scores, period, game clock
- `WinProbBar` — home vs away split bar (blue `#185FA5` / red `#993C1D`,
  matching the training charts)
- `ControlPanel` — `Connect to Live`, `Simulate Game`, `Reset` buttons
  (placeholders that only `console.log`)
- Mock data lives in `src/mock/mockGame.js`, shaped like the WebSocket
  `prediction` payload so later iterations need no component changes.

**Run / verify:**
```
cd nba-win-prob/frontend
npm install
npm run dev          # http://localhost:5173
```
Confirm: scoreboard shows mock game, win-prob bar splits ~74/26 with the
right colors, all three buttons render and log on click, no console errors.

---

## ⬜ Iteration 2 — Manual Prediction (`POST /predict`)

- Add a feature-input form (the 5 model features) or a
  "Predict from scoreboard" button.
- Add `src/api/client.js` with `predict(features)` (uses `fetch`,
  returns `{ homeProb, awayProb }`).
- The `/api` Vite proxy (already configured in `vite.config.js`) forwards
  to `http://localhost:5000` so calls are same-origin — **no backend CORS
  change needed.**
- Predict updates `WinProbBar` from the real model response.

---

## ⬜ Iteration 3 — Live WebSocket Dashboard

- Add `socket.io-client`; manage the connection in `src/api/socket.js`.
- `Connect to Live` connects and listens for `prediction` events,
  updating `Scoreboard` + `WinProbBar` in real time.
- `Simulate Game` drives a game-state feed emitting `game_state`.
- `Reset` clears the dashboard back to an idle state.

---

## ⬜ Iteration 4 — Stats & Comparison (deferred)

- `Team / Player Stats` view (search + table), NBA API backed.
- `Player Comparison` charts. Out of scope until the core dashboard is done.

---

## File Map

```
frontend/
├── PLAN.md                  ← this roadmap
├── package.json
├── vite.config.js           ← dev server + /api proxy (for iteration 2)
├── index.html
├── .gitignore
└── src/
    ├── main.jsx             ← React entry point
    ├── App.jsx              ← page layout + state
    ├── index.css            ← dark theme
    ├── mock/
    │   └── mockGame.js      ← static sample state (v1)
    └── components/
        ├── Scoreboard.jsx
        ├── WinProbBar.jsx
        └── ControlPanel.jsx
```
