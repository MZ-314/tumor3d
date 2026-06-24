import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { API_BASE } from "@viewer/api";
import { App } from "./App";
import "./styles.css";

if (import.meta.env.DEV && !API_BASE) {
  console.warn(
    "[tumor3d] VITE_API_BASE is not set in frontend/.env — requests go to localhost:8000. " +
      "Set your RunPod URL (e.g. VITE_API_BASE=https://YOUR-POD-8000.proxy.runpod.net) and restart npm run dev.",
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
