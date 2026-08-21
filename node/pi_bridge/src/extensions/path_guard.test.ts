import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { isPathInside, validateToolCall } from "./path_guard.js";

test("path guard rejects traversal and symlink escape", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "envmock-root-"));
  const outside = fs.mkdtempSync(path.join(os.tmpdir(), "envmock-outside-"));
  fs.symlinkSync(outside, path.join(root, "link"), "dir");
  assert.equal(isPathInside(root, "inside.txt"), true);
  assert.equal(isPathInside(root, "../outside.txt"), false);
  assert.equal(isPathInside(root, "link/leak.txt"), false);
});

test("path guard blocks dangerous bash and denied tools", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "envmock-root-"));
  assert.match(validateToolCall(root, "bash", { command: "sudo id" }) ?? "", /blocked/);
  assert.match(
    validateToolCall(root, "write", { path: "../outside.txt" }) ?? "",
    /escapes workspace/,
  );
  assert.match(
    validateToolCall(root, "bash", { command: "printf ok" }, ["bash"]) ?? "",
    /denied/,
  );
});
