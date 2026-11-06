import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development (npm run dev) the page is served by Vite on :5173 with hot reload,
// and every /api request is forwarded to the Flask API (LEON_API, default http://127.0.0.1:8000).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // changeOrigin rewrites the Host header to the API's own address, which the API requires.
      "/api": { target: process.env.LEON_API ?? "http://127.0.0.1:8000", changeOrigin: true },
    },
  },
});
