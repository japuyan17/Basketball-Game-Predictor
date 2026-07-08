// Renders the game scoreboard: away team on the left, home team on the
// right, with the current period and game clock shown in the center.
export default function Scoreboard({
  homeTeam,
  awayTeam,
  homeScore,
  awayScore,
  period,
  clock,
}) {
  // Format the period as "Q1"-"Q4" for regulation, "OT"/"2OT" for overtime.
  const periodLabel = formatPeriod(period);

  return (
    <section className="scoreboard card">
      <div className="team team--away">
        <span className="team-label">AWAY</span>
        <span className="team-name">{awayTeam}</span>
        <span className="team-score">{awayScore}</span>
      </div>

      <div className="game-clock">
        <span className="period">{periodLabel}</span>
        <span className="clock">{clock}</span>
      </div>

      <div className="team team--home">
        <span className="team-label">HOME</span>
        <span className="team-name">{homeTeam}</span>
        <span className="team-score">{homeScore}</span>
      </div>
    </section>
  );
}

// Converts a numeric period into a readable label (Q1-Q4, OT, 2OT, ...).
function formatPeriod(period) {
  if (period <= 4) return `Q${period}`;
  const overtimeNumber = period - 4;
  return overtimeNumber === 1 ? "OT" : `${overtimeNumber}OT`;
}
