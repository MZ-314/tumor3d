import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig({
  plugins: [react()],
  root: "chat",
  publicDir: "../public",
  resolve: {
    alias: {
      "@shared": path.resolve(__dirname, "../shared/schemas/typescript"),
      "@viewer": path.resolve(__dirname, "viewer/src"),
    },
  },
  server: {
    port: 5173,
    proxy: {
      "/static": "http://127.0.0.1:8000",
      "/reconstruct": "http://127.0.0.1:8000",
      "/health": "http://127.0.0.1:8000",
      "/meshes": "http://127.0.0.1:8000",
    },
  },
  build: {
    outDir: "../dist/chat",
    emptyOutDir: true,
  },
});
