import readline from "node:readline";
import { emit, now, type BridgeCommand } from "./protocol.js";
import { startSession, type ActiveSession, type RuntimeStartCommand } from "./runtime.js";

const active = new Map<string, ActiveSession>();

function serializeEvent(event: unknown): Record<string, unknown> {
  if (typeof event === "object" && event !== null) {
    return JSON.parse(JSON.stringify(event)) as Record<string, unknown>;
  }
  return { value: event };
}

function usagePayload(session: ActiveSession): Record<string, unknown> {
  const usage = {
    inputTokens: 0,
    outputTokens: 0,
    cacheReadTokens: 0,
    cacheWriteTokens: 0,
    costUsd: 0,
    turns: 0,
  };
  for (const raw of session.session.messages as unknown[]) {
    if (typeof raw !== "object" || raw === null) continue;
    const message = raw as Record<string, unknown>;
    if (message.role !== "assistant") continue;
    usage.turns += 1;
    const rawUsage = message.usage;
    if (typeof rawUsage !== "object" || rawUsage === null) continue;
    const values = rawUsage as Record<string, unknown>;
    usage.inputTokens += Number(values.input ?? 0);
    usage.outputTokens += Number(values.output ?? 0);
    usage.cacheReadTokens += Number(values.cacheRead ?? 0);
    usage.cacheWriteTokens += Number(values.cacheWrite ?? 0);
    const cost = values.cost;
    if (typeof cost === "object" && cost !== null) {
      usage.costUsd += Number((cost as Record<string, unknown>).total ?? 0);
    }
  }
  return usage;
}

function sessionFailure(session: ActiveSession): string | undefined {
  for (const raw of [...(session.session.messages as unknown[])].reverse()) {
    if (typeof raw !== "object" || raw === null) continue;
    const message = raw as Record<string, unknown>;
    if (message.role !== "assistant" || message.stopReason !== "error") continue;
    return String(message.errorMessage ?? "Pi model request failed");
  }
  return undefined;
}

async function run(command: RuntimeStartCommand): Promise<void> {
  if (active.has(command.requestId)) {
    throw new Error(`Pi request is already active: ${command.requestId}`);
  }
  emit({ type: "runtime_started", requestId: command.requestId, timestamp: now() });
  const current = await startSession(command);
  active.set(command.requestId, current);
  let toolCalls = 0;
  let turns = 0;
  const unsubscribe = current.session.subscribe((event) => {
    if (event.type === "tool_execution_start") {
      toolCalls += 1;
      if (toolCalls > (command.maxToolEvents ?? 300)) {
        current.failureReason = `Pi exceeded ${command.maxToolEvents ?? 300} tool calls`;
        void current.session.abort();
      }
    }
    if (event.type === "turn_start") {
      turns += 1;
      if (turns > (command.maxTurns ?? 50)) {
        current.failureReason = `Pi exceeded ${command.maxTurns ?? 50} turns`;
        void current.session.abort();
      }
    }
    emit({
      type: event.type,
      requestId: command.requestId,
      timestamp: now(),
      payload: serializeEvent(event),
    });
  });
  try {
    await current.session.prompt(command.prompt);
    const modelFailure = sessionFailure(current);
    if (!current.failureReason && modelFailure) {
      current.failureReason = modelFailure;
    }
    const sessionPayload = {
      sessionId: current.session.sessionId,
      sessionFile: current.session.sessionFile,
    };
    emit({
      type: "usage",
      requestId: command.requestId,
      timestamp: now(),
      payload: {
        ...usagePayload(current),
        toolCalls,
        turns,
        ...sessionPayload,
      },
    });
    if (current.cancelled || current.failureReason) {
      emit({
        type: "runtime_failed",
        requestId: command.requestId,
        timestamp: now(),
        payload: {
          error: current.cancelled ? "Pi runtime cancelled" : current.failureReason,
          errorCode: current.cancelled
            ? "cancelled"
            : current.failureReason?.includes("401")
              ? "auth_unavailable"
              : "process_error",
          ...sessionPayload,
        },
      });
    } else {
      emit({
        type: "runtime_finished",
        requestId: command.requestId,
        timestamp: now(),
        payload: sessionPayload,
      });
    }
  } finally {
    unsubscribe();
    current.dispose();
    active.delete(command.requestId);
  }
}

async function handle(command: BridgeCommand): Promise<void> {
  if (command.type === "probe") {
    emit({
      type: "probe_result",
      requestId: command.requestId,
      timestamp: now(),
      payload: {
        available: true,
        bridgeVersion: "0.2.0",
        piVersion: "0.80.10",
        supportsCancel: true,
        supportsResume: true,
      },
    });
    return;
  }

  if (command.type === "cancel") {
    const current = active.get(command.requestId);
    if (current) {
      current.cancelled = true;
      await current.session.abort();
    }
    emit({ type: "cancelled", requestId: command.requestId, timestamp: now() });
    return;
  }

  if (command.type === "shutdown") {
    for (const current of active.values()) {
      current.cancelled = true;
      await current.session.abort();
      current.dispose();
    }
    active.clear();
    emit({ type: "shutdown", requestId: command.requestId, timestamp: now() });
    process.exitCode = 0;
    return;
  }

  await run(command);
}

const input = readline.createInterface({
  input: process.stdin,
  crlfDelay: Infinity,
});

input.on("line", (line) => {
  if (!line.trim()) return;
  void (async () => {
    let command: BridgeCommand;
    try {
      command = JSON.parse(line) as BridgeCommand;
      await handle(command);
    } catch (error) {
      const requestId =
        typeof command! === "object" && command! !== null && "requestId" in command!
          ? command!.requestId
          : "unknown";
      emit({
        type: "runtime_failed",
        requestId,
        timestamp: now(),
        payload: {
          error: error instanceof Error ? error.message : String(error),
          errorCode: "process_error",
        },
      });
    }
  })();
});
