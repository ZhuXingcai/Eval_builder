import type { InlineExtension } from "@earendil-works/pi-coding-agent";
import fs from "node:fs";
import path from "node:path";

const blockedCommandPatterns = [
  /(^|\s)sudo(\s|$)/,
  /rm\s+-[^\n]*r[^\n]*\s+\/(?:\s|$)/,
  /git\s+push(?:\s|$)/,
  /curl\b[^\n]*\|\s*(?:sh|bash)\b/,
  /wget\b[^\n]*\|\s*(?:sh|bash)\b/,
];

function nearestExistingParent(candidate: string): string {
  let current = candidate;
  while (!fs.existsSync(current)) {
    const parent = path.dirname(current);
    if (parent === current) return current;
    current = parent;
  }
  return current;
}

export function isPathInside(root: string, candidate: string): boolean {
  const resolvedRoot = fs.realpathSync(path.resolve(root));
  const lexicalCandidate = path.isAbsolute(candidate)
    ? path.resolve(candidate)
    : path.resolve(resolvedRoot, candidate);
  const lexicalRelative = path.relative(resolvedRoot, lexicalCandidate);
  if (lexicalRelative.startsWith("..") || path.isAbsolute(lexicalRelative)) {
    return false;
  }
  const existingParent = nearestExistingParent(lexicalCandidate);
  const realParent = fs.realpathSync(existingParent);
  const realRelative = path.relative(resolvedRoot, realParent);
  return !realRelative.startsWith("..") && !path.isAbsolute(realRelative);
}

export function validateToolCall(
  root: string,
  toolName: string,
  input: Record<string, unknown>,
  excludeTools: string[] = [],
): string | undefined {
  if (excludeTools.map((value) => value.toLowerCase()).includes(toolName.toLowerCase())) {
    return `Tool denied by runtime policy: ${toolName}`;
  }
  if (["read", "write", "edit", "grep", "find", "ls"].includes(toolName)) {
    const candidate = input.path;
    if (typeof candidate === "string" && !isPathInside(root, candidate)) {
      return `${toolName} path escapes workspace: ${candidate}`;
    }
  }
  if (toolName === "bash") {
    const command = String(input.command ?? "");
    for (const pattern of blockedCommandPatterns) {
      if (pattern.test(command)) {
        return `Bash command blocked by local-process policy: ${pattern.source}`;
      }
    }
    const absolutePaths =
      command.match(/(?<![\w.-])\/(?:Users|home|etc|var|tmp|opt|private)\/[^\s;&|<>`$()]+/g) ??
      [];
    for (const candidate of absolutePaths) {
      if (!isPathInside(root, candidate)) {
        return `Bash command references path outside workspace: ${candidate}`;
      }
    }
  }
  return undefined;
}

export function createPathGuardExtension(
  root: string,
  excludeTools: string[] = [],
): InlineExtension {
  return {
    name: "envmock-path-guard",
    factory: (pi) => {
      pi.on("tool_call", (event) => {
        const reason = validateToolCall(
          root,
          event.toolName,
          event.input as Record<string, unknown>,
          excludeTools,
        );
        return reason ? { block: true, reason } : undefined;
      });
    },
  };
}
