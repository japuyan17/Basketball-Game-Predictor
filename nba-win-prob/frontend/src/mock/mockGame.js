// Static sample game state used for the v1 layout (no backend yet).
// Matches the shape the WebSocket "prediction" event will send later,
// so wiring up real data in iteration 3 needs no component changes.
export const mockGame = {
  homeTeam: "Lakers",
  awayTeam: "Celtics",
  homeScore: 87,
  awayScore: 82,
  period: 4,
  clock: "4:23",
  homeProb: 0.74,
  awayProb: 0.26,
};
