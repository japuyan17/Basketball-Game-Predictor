// Renders the dashboard control buttons. In v1 these are placeholders
// that only log to the console; iterations 2-3 wire them to the backend
// (Connect to Live -> WebSocket, Simulate Game -> game feed, Reset).
export default function ControlPanel() {
  // Placeholder click handler so buttons are interactive without a backend.
  const handleClick = (action) => {
    console.log(`[v1 placeholder] "${action}" clicked — not wired up yet`);
  };

  return (
    <section className="controls card">
      <button className="btn btn--primary"
        onClick={() => handleClick("Connect to Live")}>
        Connect to Live
      </button>
      <button className="btn"
        onClick={() => handleClick("Simulate Game")}>
        Simulate Game
      </button>
      <button className="btn btn--ghost"
        onClick={() => handleClick("Reset")}>
        Reset
      </button>
    </section>
  );
}
