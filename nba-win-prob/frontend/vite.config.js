import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Vite config for the NBA Win Probability dashboard.
// The /api proxy is unused in v1 (no backend calls yet) but is set up now
// so iteration 2 can call the Flask server same-origin without CORS changes.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:5000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
