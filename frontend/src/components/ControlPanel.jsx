import { useState } from "react";
import { parseInjuryInput } from "../api/client.js";

// Matchup form: pick away/home teams, optionally list injured players,
// then fire a prediction request (handled by App via onPredict).
export default function ControlPanel({ teams, onPredict, onReset, loading }) {
  const [awayTeam, setAwayTeam] = useState("");
  const [homeTeam, setHomeTeam] = useState("");
  const [awayInjuryText, setAwayInjuryText] = useState("");
  const [homeInjuryText, setHomeInjuryText] = useState("");

  const canPredict =
    awayTeam && homeTeam && awayTeam !== homeTeam && !loading;

  // Submits the current form selections up to App for the API call.
  const handleSubmit = (event) => {
    event.preventDefault();
    if (!canPredict) return;
    onPredict({
      homeTeam,
      awayTeam,
      homeInjuries: parseInjuryInput(homeInjuryText),
      awayInjuries: parseInjuryInput(awayInjuryText),
    });
  };

  // Clears all form fields and tells App to drop the current prediction.
  const handleReset = () => {
    setAwayTeam("");
    setHomeTeam("");
    setAwayInjuryText("");
    setHomeInjuryText("");
    onReset();
  };

  return (
    <form className="controls card" onSubmit={handleSubmit}>
      <div className="control-row">
        <label className="control-field">
          <span className="control-label">Away Team</span>
          <select
            className="control-input"
            value={awayTeam}
            onChange={(e) => setAwayTeam(e.target.value)}
          >
            <option value="">Select away team…</option>
            {teams.map((team) => (
              <option key={team.TEAM_ABBREVIATION} value={team.TEAM_NAME}>
                {team.TEAM_NAME}
              </option>
            ))}
          </select>
        </label>

        <label className="control-field">
          <span className="control-label">Home Team</span>
          <select
            className="control-input"
            value={homeTeam}
            onChange={(e) => setHomeTeam(e.target.value)}
          >
            <option value="">Select home team…</option>
            {teams.map((team) => (
              <option key={team.TEAM_ABBREVIATION} value={team.TEAM_NAME}>
                {team.TEAM_NAME}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div className="control-row">
        <label className="control-field">
          <span className="control-label">Away Players Out (optional)</span>
          <input
            className="control-input"
            type="text"
            placeholder="e.g. Kevin Durant, Fred VanVleet"
            value={awayInjuryText}
            onChange={(e) => setAwayInjuryText(e.target.value)}
          />
        </label>

        <label className="control-field">
          <span className="control-label">Home Players Out (optional)</span>
          <input
            className="control-input"
            type="text"
            placeholder="e.g. Jayson Tatum"
            value={homeInjuryText}
            onChange={(e) => setHomeInjuryText(e.target.value)}
          />
        </label>
      </div>

      <div className="control-row control-row--buttons">
        <button className="btn btn--primary" type="submit"
          disabled={!canPredict}>
          {loading ? "Predicting…" : "Predict Game"}
        </button>
        <button className="btn btn--ghost" type="button"
          onClick={handleReset}>
          Reset
        </button>
      </div>
    </form>
  );
}
