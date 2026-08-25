import assert from "node:assert/strict";
import test from "node:test";
import {
  PI_BRIDGE_PROBE_RESULT,
  type BridgeCommand,
  type ResumeCommand,
} from "./protocol.js";

test("resume command is part of the bridge protocol", () => {
  const command: ResumeCommand = {
    type: "resume",
    requestId: "run:artifact",
    workspace: "/tmp/workspace",
    prompt: "Continue.",
    sessionFile: "/tmp/workspace/.envmock-sessions/session.jsonl",
  };
  const protocolValue: BridgeCommand = command;
  assert.equal(protocolValue.type, "resume");
  assert.equal(protocolValue.sessionFile, command.sessionFile);
});

test("probe distinguishes protocol availability from credential readiness", () => {
  assert.equal(PI_BRIDGE_PROBE_RESULT.available, true);
  assert.equal(PI_BRIDGE_PROBE_RESULT.protocolAvailable, true);
  assert.equal(PI_BRIDGE_PROBE_RESULT.credentialStatus, "UNKNOWN");
  assert.equal(PI_BRIDGE_PROBE_RESULT.sandboxEnforcement, "NONE");
  assert.equal(PI_BRIDGE_PROBE_RESULT.supportsEventStreaming, true);
  assert.equal(PI_BRIDGE_PROBE_RESULT.supportsToolProgress, true);
});
