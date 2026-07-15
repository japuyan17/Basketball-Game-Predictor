// All backend communication for the frontend goes through this client.
// Requests hit the Vite /api proxy (vite.config.js) → Flask on port 5000,
// where Translator.py validates every payload before it reaches the model.
const API_BASE = "/api";

// Fetches all teams with their current stats from GET /teams.
export async function fetchTeams() {
  const response = await fetch(`${API_BASE}/teams`);
  if (!response.ok) {
    throw new Error(`Failed to load teams (HTTP ${response.status})`);
  }
  const data = await response.json();
  return data.teams;
}

// Requests a win probability from POST /predict, with optional injuries.
export async function predictGame({
  homeTeam,
  awayTeam,
  homeInjuries,
  awayInjuries,
}) {
  const response = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      home_team: homeTeam,
      away_team: awayTeam,
      home_injuries: homeInjuries.length ? homeInjuries : null,
      away_injuries: awayInjuries.length ? awayInjuries : null,
    }),
  });

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || `Prediction failed (HTTP ${response.status})`);
  }
  return data;
}

// Splits a comma-separated input string into a clean list of player names.
export function parseInjuryInput(raw) {
  return raw
    .split(",")
    .map((name) => name.trim())
    .filter(Boolean);
}
