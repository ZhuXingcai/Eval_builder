# ADR 0006: Limit Local Process Mode To Trusted Inputs

- Status: Accepted
- Date: 2026-07-18

## Context

The initial machine has no container runtime. Claude and Pi explicitly state that a working-directory setting is
not a security sandbox.

## Decision

The vertical slice uses local processes only for trusted, monitored LH inputs. It applies isolated homes, path
guards, tool allow/deny rules, timeouts, process-group cancellation, and no bypass-permission mode. Untrusted or
unattended operation remains disabled until ContainerWorkspace exists.

## Consequences

- The initial implementation can proceed without pretending to have a sandbox.
- Security status is visible in `envmock doctor`.
- Containerization is a release gate for unattended external inputs.
