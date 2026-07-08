// Renders a horizontal split bar showing each team's win probability.
// Home uses brand blue (#185FA5), away uses brand orange-red (#993C1D) —
// the same colors as the matplotlib charts in train_model.py.
export default function WinProbBar({ homeTeam, awayTeam, homeProb, awayProb }) {
  // Convert 0-1 probabilities into whole-percent values for display + width.
  const homePercent = Math.round(homeProb * 100);
  const awayPercent = Math.round(awayProb * 100);

  return (
    <section className="winprob card">
      <h2 className="winprob-title">Win Probability</h2>

      <div className="winprob-bar" role="img"
        aria-label={`${homeTeam} ${homePercent}%, ${awayTeam} ${awayPercent}%`}>
        <div className="winprob-segment winprob-segment--home"
          style={{ width: `${homePercent}%` }}>
          {homePercent}%
        </div>
        <div className="winprob-segment winprob-segment--away"
          style={{ width: `${awayPercent}%` }}>
          {awayPercent}%
        </div>
      </div>

      <div className="winprob-legend">
        <span className="legend-item">
          <span className="swatch swatch--home" /> {homeTeam} (Home)
        </span>
        <span className="legend-item">
          <span className="swatch swatch--away" /> {awayTeam} (Away)
        </span>
      </div>
    </section>
  );
}
