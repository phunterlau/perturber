import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./styles.css";

try {
  const response = await fetch("/probe-config.json");
  if (response.ok) {
    const config: { token: string } = await response.json();
    window.__PROBE_TOKEN__ = config.token;
  }
} catch {
  // Development may inject the token separately; API errors remain visible in-app.
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
