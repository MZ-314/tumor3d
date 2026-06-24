import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, __dirname, "");
  const proxyTarget =
    env.VITE_API_PROXY_TARGET || env.VITE_API_BASE || "http://127.0.0.1:8000";

  const apiRoutes = ["/static", "/reconstruct", "/health", "/meshes", "/chats"];

  return {
    plugins: [react()],
    root: "chat",
    envDir: path.resolve(__dirname),
    publicDir: "../public",
    resolve: {
      alias: {
        "@shared": path.resolve(__dirname, "../shared/schemas/typescript"),
        "@viewer": path.resolve(__dirname, "viewer/src"),
      },
    },
    server: {
      port: 5173,
      proxy: Object.fromEntries(
        apiRoutes.map((route) => [route, { target: proxyTarget, changeOrigin: true }]),
      ),
    },
    build: {
      outDir: "../dist/chat",
      emptyOutDir: true,
    },
  };
});
