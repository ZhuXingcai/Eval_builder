import assert from "node:assert/strict";
import test from "node:test";
import type { BridgeCommand, ResumeCommand } from "./protocol.js";

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
