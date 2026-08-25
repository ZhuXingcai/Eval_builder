import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import AgentShell from "./AgentShell";
import "./styles.css";

const root = document.getElementById("root");
if (!root) throw new Error("Console root is missing.");

createRoot(root).render(
  <StrictMode>
    <AgentShell />
  </StrictMode>,
);
