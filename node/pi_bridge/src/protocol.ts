export type RuntimeCommandBase = {
  requestId: string;
  workspace: string;
  prompt: string;
  systemPolicy?: string;
  provider?: string;
  model?: string;
  thinkingLevel?: "off" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max";
  tools?: Array<"read" | "write" | "edit" | "bash" | "grep" | "find" | "ls">;
  excludeTools?: string[];
  maxTurns?: number;
  maxToolEvents?: number;
};

export type StartCommand = RuntimeCommandBase & {
  type: "start";
};

export type ResumeCommand = RuntimeCommandBase & {
  type: "resume";
  sessionFile: string;
};

export type CancelCommand = {
  type: "cancel";
  requestId: string;
};

export type ProbeCommand = {
  type: "probe";
  requestId: string;
};

export type ShutdownCommand = {
  type: "shutdown";
  requestId: string;
};

export type BridgeCommand =
  | StartCommand
  | ResumeCommand
  | CancelCommand
  | ProbeCommand
  | ShutdownCommand;

export type BridgeEvent = {
  type: string;
  requestId: string;
  timestamp: string;
  payload?: Record<string, unknown>;
};

export function emit(event: BridgeEvent): void {
  process.stdout.write(`${JSON.stringify(event)}\n`);
}

export function now(): string {
  return new Date().toISOString();
}
