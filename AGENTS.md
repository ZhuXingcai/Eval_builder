# Agent Instructions

## Mandatory Session Bootstrap

This repository is designed to survive context loss. Before planning or editing:

1. Read `PROJECT_STATE.md`.
2. Read `.specify/memory/constitution.md`.
3. Read the active spec named by `PROJECT_STATE.md`; currently `specs/002-eval-dataset-factory/spec.md`
   and `plan.md`.
4. Run `bd prime`.
5. Run `uv run envmock context bootstrap`.
6. Run `python3 ./.trellis/scripts/get_context.py` when `.trellis/` is present.
7. Claim exactly one ready Beads issue before changing code.

Before handing off:

1. Run the claimed issue's validation commands.
2. Update or close the Beads issue with concrete evidence.
3. Run:
   ```bash
   uv run envmock context checkpoint --task-id <id> \
     --validation "<result>" \
     --next-command "<exact command>"
   ```
   Add repeatable `--failure "<blocker>"` options for unresolved failures.
4. Report changed files, validation, remaining blockers, and the next Beads issue.

Chat transcripts, Claude/Pi sessions, and compaction summaries are not project truth.

## Product Invariants

- Generate input-state attachments, never requested final deliverables or answer hints.
- Evidence priority: observable source facts, explicit prompt requirements, then inferred dependencies.
- Use deterministic providers before open-ended agent runtimes.
- Critical capability gaps return `BLOCKED_CAPABILITY`; never silently downgrade.
- Keep structural validation and semantic review separate.
- Three standard review rounds cover solvability, realism, and leakage/executability.
- Runtime code is vendor-neutral. Vendor SDK imports stay in vendor-specific modules.
- Existing LH fusion packages and `cc-mock-env` are read-only benchmark sources.
- Local process mode is trusted/monitored only and must not use bypass-permission modes.

## Build And Test

```bash
uv sync --extra dev
uv run ruff check .
uv run ruff format --check .
uv run mypy src scripts
uv run pytest
npm run check --prefix node/pi_bridge
```

Live tests are opt-in through pytest markers and may require credentials.

This project uses **bd** (beads) for issue tracking. Run `bd prime` for full workflow context.

## Trellis Project Specs

This repository is initialized with Trellis for project-local coding guidance and session memory.
Trellis does not replace Beads or Spec Kit unless the user explicitly asks for a task-tracking
migration.

- Read `.trellis/spec/backend/index.md` before implementation work.
- Use `.trellis/spec/**` as coding convention input for Trellis implement/check flows.
- Use `python3 ./.trellis/scripts/get_context.py` to inspect Trellis task/spec state.
- Keep `session_auto_commit: false` in `.trellis/config.yaml`; do not let Trellis auto-commit.
- Do not edit `.trellis/.template-hashes.json`, `.trellis/.runtime/`, or `~/.trellis/channels/**`
  directly.

> **Architecture in one line:** Issues live in a local Dolt database
> (`.beads/dolt/`); cross-machine sync uses `bd dolt push/pull` (a
> git-compatible protocol), stored under `refs/dolt/data` on your git
> remote — separate from `refs/heads/*` where your code lives.
> `.beads/issues.jsonl` is a passive export, not the wire protocol.
>
> See [SYNC_CONCEPTS.md](https://github.com/gastownhall/beads/blob/main/docs/SYNC_CONCEPTS.md)
> for the one-screen overview and anti-patterns (don't treat JSONL as the
> source of truth; don't `bd import` during normal operation; don't
> reach for third-party Dolt hosting before trying the default).

## Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work atomically
bd close <id>         # Complete work
bd dolt push          # Push beads data to remote
```

## Non-Interactive Shell Commands

**ALWAYS use non-interactive flags** with file operations to avoid hanging on confirmation prompts.

Shell commands like `cp`, `mv`, and `rm` may be aliased to include `-i` (interactive) mode on some systems, causing the agent to hang indefinitely waiting for y/n input.

**Use these forms instead:**
```bash
# Force overwrite without prompting
cp -f source dest           # NOT: cp source dest
mv -f source dest           # NOT: mv source dest
rm -f file                  # NOT: rm file

# For recursive operations
rm -rf directory            # NOT: rm -r directory
cp -rf source dest          # NOT: cp -r source dest
```

**Other commands that may prompt:**
- `scp` - use `-o BatchMode=yes` for non-interactive
- `ssh` - use `-o BatchMode=yes` to fail instead of prompting
- `apt-get` - use `-y` flag
- `brew` - use `HOMEBREW_NO_AUTO_UPDATE=1` env var

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:6cd5cc61 -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See https://github.com/gastownhall/beads/blob/main/docs/SYNC_CONCEPTS.md for details and anti-patterns.

## Agent Context Profiles

The managed Beads block is task-tracking guidance, not permission to override repository, user, or orchestrator instructions.

- **Conservative (default)**: Use `bd` for task tracking. Do not run git commits, git pushes, or Dolt remote sync unless explicitly asked. At handoff, report changed files, validation, and suggested next commands.
- **Minimal**: Keep tool instruction files as pointers to `bd prime`; use the same conservative git policy unless active instructions say otherwise.
- **Team-maintainer**: Only when the repository explicitly opts in, agents may close beads, run quality gates, commit, and push as part of session close. A current "do not commit" or "do not push" instruction still wins.

## Session Completion

This protocol applies when ending a Beads implementation workflow. It is subordinate to explicit user, repository, and orchestrator instructions.

1. **File issues for remaining work** - Create beads for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **Handle git/sync by active profile**:
   ```bash
   # Conservative/minimal/default: report status and proposed commands; wait for approval.
   git status

   # Team-maintainer opt-in only, unless current instructions forbid it:
   git pull --rebase
   git push
   git status
   ```
5. **Hand off** - Summarize changes, validation, issue status, and any blocked sync/commit/push step

**Critical rules:**
- Explicit user or orchestrator instructions override this Beads block.
- Do not commit or push without clear authority from the active profile or the current user request.
- If a required sync or push is blocked, stop and report the exact command and error.
<!-- END BEADS INTEGRATION -->

<!-- BEGIN BEADS CODEX SETUP: generated by bd setup codex -->
## Beads Issue Tracker

Use Beads (`bd`) for durable task tracking in repositories that include it. Use the `beads` skill at `.agents/skills/beads/SKILL.md` (project install) or `~/.agents/skills/beads/SKILL.md` (global install) for Beads workflow guidance, then use the `bd` CLI for issue operations.

### Quick Reference

```bash
bd ready                # Find available work
bd show <id>            # View issue details
bd update <id> --claim  # Claim work
bd close <id>           # Complete work
bd prime                # Refresh Beads context
```

### Rules

- Use `bd` for all task tracking; do not create markdown TODO lists.
- Run `bd prime` when Beads context is missing or stale. Codex 0.129.0+ can load Beads context automatically through native hooks; use `/hooks` to inspect or toggle them.
- Keep persistent project memory in Beads via `bd remember`; do not create ad hoc memory files.

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See https://github.com/gastownhall/beads/blob/main/docs/SYNC_CONCEPTS.md for details and anti-patterns.
<!-- END BEADS CODEX SETUP -->
