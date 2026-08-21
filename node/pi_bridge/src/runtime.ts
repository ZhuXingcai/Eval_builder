import {
  createAgentSession,
  DefaultResourceLoader,
  ModelRuntime,
  SessionManager,
  type AgentSession,
} from "@earendil-works/pi-coding-agent";
import path from "node:path";
import { isPathInside, createPathGuardExtension } from "./extensions/path_guard.js";
import type { ResumeCommand, StartCommand } from "./protocol.js";

export type RuntimeStartCommand = StartCommand | ResumeCommand;

export type ActiveSession = {
  requestId: string;
  session: AgentSession;
  cancelled: boolean;
  failureReason?: string;
  dispose: () => void;
};

function buildSystemPolicy(command: RuntimeStartCommand): string {
  return [
    "You build initial environment attachments for agent evaluations.",
    "Operate only inside the current workspace.",
    "Do not create the final deliverable requested by the task.",
    "Do not include answer keys, scoring hints, or precomputed conclusions.",
    "Use existing tools and validate every file you create.",
    command.systemPolicy ?? "",
  ]
    .filter(Boolean)
    .join("\n");
}

export async function startSession(command: RuntimeStartCommand): Promise<ActiveSession> {
  const agentDir = path.join(command.workspace, ".pi-home");
  const modelRuntime = await ModelRuntime.create({
    authPath: path.join(agentDir, "auth.json"),
    modelsPath: path.join(agentDir, "models.json"),
  });
  const model =
    command.provider && command.model
      ? modelRuntime.getModel(command.provider, command.model)
      : undefined;
  if (command.provider && command.model && !model) {
    throw new Error(`Pi model not found: ${command.provider}/${command.model}`);
  }

  const loader = new DefaultResourceLoader({
    cwd: command.workspace,
    agentDir,
    systemPromptOverride: (_base) => buildSystemPolicy(command),
    noExtensions: true,
    noSkills: true,
    noPromptTemplates: true,
    noThemes: true,
    extensionFactories: [
      createPathGuardExtension(command.workspace, command.excludeTools ?? []),
    ],
  });
  await loader.reload();
  const sessionDir = path.join(command.workspace, ".envmock-sessions");
  let sessionManager: SessionManager;
  if (command.type === "resume") {
    if (!isPathInside(command.workspace, command.sessionFile)) {
      throw new Error(`Pi session file escapes workspace: ${command.sessionFile}`);
    }
    sessionManager = SessionManager.open(command.sessionFile, sessionDir, command.workspace);
  } else {
    sessionManager = SessionManager.create(command.workspace, sessionDir);
  }
  const result = await createAgentSession({
    cwd: command.workspace,
    ...(model ? { model } : {}),
    modelRuntime,
    tools: command.tools ?? ["read", "write", "edit", "bash", "grep", "find", "ls"],
    ...(command.excludeTools ? { excludeTools: command.excludeTools } : {}),
    thinkingLevel: command.thinkingLevel ?? "high",
    sessionManager,
    resourceLoader: loader,
  });
  return {
    requestId: command.requestId,
    session: result.session,
    cancelled: false,
    dispose: () => result.session.dispose(),
  };
}
