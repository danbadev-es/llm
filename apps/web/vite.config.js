import path from "node:path";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// In development (npm run dev) the page is served by Vite on :5173 with hot reload,
// and every /api and /v1 request is forwarded to the Flask API (LEON_API, default http://127.0.0.1:8000).
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: { "@": path.resolve(import.meta.dirname, "src") }, // "@/components/ui/button", as shadcn writes it
  },
  server: {
    port: 5173,
    proxy: {
      // changeOrigin rewrites the Host header to the API's own address, which the API requires.
      "/api": { target: process.env.LEON_API ?? "http://127.0.0.1:8000", changeOrigin: true },
      "/v1": { target: process.env.LEON_API ?? "http://127.0.0.1:8000", changeOrigin: true },
    },
  },
});
