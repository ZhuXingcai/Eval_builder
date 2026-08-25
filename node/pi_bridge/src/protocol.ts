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

export type ProbeResultPayload = {
  available: boolean;
  protocolAvailable: boolean;
  bridgeVersion: string;
  piVersion: string;
  supportsCancel: boolean;
  supportsResume: boolean;
  supportsEventStreaming: boolean;
  supportsToolProgress: boolean;
  credentialStatus: "READY" | "NOT_REQUIRED" | "UNKNOWN" | "MISSING" | "INVALID";
  sandboxEnforcement: "FULL" | "PARTIAL" | "NONE" | "NOT_APPLICABLE";
};

export const PI_BRIDGE_PROBE_RESULT: Readonly<ProbeResultPayload> = Object.freeze({
  available: true,
  protocolAvailable: true,
  bridgeVersion: "0.2.0",
  piVersion: "0.80.10",
  supportsCancel: true,
  supportsResume: true,
  supportsEventStreaming: true,
  supportsToolProgress: true,
  credentialStatus: "UNKNOWN",
  sandboxEnforcement: "NONE",
});

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
