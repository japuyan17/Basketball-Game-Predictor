import Scoreboard from "./components/Scoreboard.jsx";
import WinProbBar from "./components/WinProbBar.jsx";
import ControlPanel from "./components/ControlPanel.jsx";
import { mockGame } from "./mock/mockGame.js";

// Top-level page: lays out the live win-probability dashboard.
// v1 reads from static mock data; later iterations replace this with
// live state from POST /predict and the WebSocket feed.
export default function App() {
  const game = mockGame;

  return (
    <div className="app">
      <header className="app-header">
        <h1>NBA Win Probability</h1>
        <p className="subtitle">Live game dashboard</p>
      </header>

      <main className="dashboard">
        <Scoreboard
          homeTeam={game.homeTeam}
          awayTeam={game.awayTeam}
          homeScore={game.homeScore}
          awayScore={game.awayScore}
          period={game.period}
          clock={game.clock}
        />

        <WinProbBar
          homeTeam={game.homeTeam}
          awayTeam={game.awayTeam}
          homeProb={game.homeProb}
          awayProb={game.awayProb}
        />

        <ControlPanel />
      </main>

      <footer className="app-footer">
        <span className="badge">v1 · layout preview (mock data)</span>
      </footer>
    </div>
  );
}
