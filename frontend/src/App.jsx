import { useEffect, useState } from "react";
import Scoreboard from "./components/Scoreboard.jsx";
import WinProbBar from "./components/WinProbBar.jsx";
import ControlPanel from "./components/ControlPanel.jsx";
import { fetchTeams, predictGame } from "./api/client.js";

// Maps a win probability margin to the same confidence tiers the CLI uses.
function confidenceLabel(homeProb) {
  const margin = Math.abs(homeProb - 0.5);
  if (margin >= 0.2) return "Strong favorite";
  if (margin >= 0.1) return "Lean";
  return "Toss-up";
}

// Top-level page: pre-game predictor wired to the Flask backend via /api.
export default function App() {
  const [teams, setTeams] = useState([]);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Load the team list once on mount so the dropdowns can populate.
  useEffect(() => {
    fetchTeams()
      .then(setTeams)
      .catch(() =>
        setError("Cannot reach the prediction server — is app.py running?")
      );
  }, []);

  // Calls the backend for a prediction and stores the result.
  const handlePredict = async (matchup) => {
    setLoading(true);
    setError(null);
    try {
      const result = await predictGame(matchup);
      setPrediction(result);
    } catch (err) {
      setError(err.message);
      setPrediction(null);
    } finally {
      setLoading(false);
    }
  };

  // Clears the current prediction and any error state.
  const handleReset = () => {
    setPrediction(null);
    setError(null);
  };

  const winner =
    prediction &&
    (prediction.homeProb >= 0.5 ? prediction.homeTeam : prediction.awayTeam);
  const winnerProb =
    prediction && Math.max(prediction.homeProb, prediction.awayProb);

  return (
    <div className="app">
      <header className="app-header">
        <h1>NBA Win Probability</h1>
        <p className="subtitle">Pre-game matchup predictor</p>
      </header>

      <main className="dashboard">
        <ControlPanel
          teams={teams}
          onPredict={handlePredict}
          onReset={handleReset}
          loading={loading}
        />

        {error && <div className="error-banner card">{error}</div>}

        {prediction && (
          <>
            <Scoreboard
              homeTeam={prediction.homeTeam}
              awayTeam={prediction.awayTeam}
              homeScore="—"
              awayScore="—"
              period={null}
              clock="Pre-game"
            />

            <WinProbBar
              homeTeam={prediction.homeTeam}
              awayTeam={prediction.awayTeam}
              homeProb={prediction.homeProb}
              awayProb={prediction.awayProb}
            />

            <section className="verdict card">
              <p className="verdict-line">
                Predicted winner: <strong>{winner}</strong>{" "}
                ({Math.round(winnerProb * 100)}% —{" "}
                {confidenceLabel(prediction.homeProb)})
              </p>
              {prediction.homeMargin != null && (
                <p className="verdict-line verdict-line--spread">
                  Predicted spread:{" "}
                  <strong>
                    {prediction.homeMargin >= 0 ? prediction.homeTeam : prediction.awayTeam}{" "}
                    by {Math.abs(prediction.homeMargin).toFixed(1)}
                  </strong>
                </p>
              )}
              {prediction.adjustments.length > 0 && (
                <ul className="adjustments">
                  {prediction.adjustments.map((line) => (
                    <li key={line}>{line}</li>
                  ))}
                </ul>
              )}
            </section>
          </>
        )}
      </main>

      <footer className="app-footer">
        <span className="badge">v2 · live backend</span>
      </footer>
    </div>
  );
}
