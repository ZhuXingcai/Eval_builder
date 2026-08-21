# Eval Factory First Labels And Release Profiles v1

- Status: Approved
- Date: 2026-07-20
- Beads: `env_mock_agent-ujc.1.11`
- Spec: `specs/002-eval-dataset-factory/spec.md`
- Contract manifest:
  `d550d84262d3f727394c65e52ec0bf6e41fdaf532bfde0851c72e6d1495d59a9`

## Decision

Approve these first LabelSpecs:

1. `search_tool_usage/v1`: deterministic structured label.
2. `powershell_error_signature/v1`: deterministic paired error-signature label.
3. `contextual_recovery_after_tool_error/v1`: deterministic prerequisite followed by a bounded semantic
   residual with explicit abstention.

Approve LH v1 as the mandatory first release profile for `CANARY` and `INTERNAL_REVIEW`. Production is
disabled until R8 issues a current ProductionReadinessAttestation.

Defer Generic v1 release. Generic is not enabled for any registry channel and requires a new approved
release-profile decision.

## Label Boundaries

### Search Tool Usage

The positive fact is a normalized executed `ToolCallRecord` whose `tool_family=search`. Tool
definitions, prompt text, tool mentions, malformed-request heuristics, and fetch-only calls are not
positive evidence.

A negative decision requires complete `tool_events` capability over the decision window. Missing,
partial, or heuristic evidence yields `ABSTAIN`, not `NO_MATCH`.

### PowerShell Error Signature

The positive fact is a versioned normalized signature on one `ToolCallRecord` that:

- identifies PowerShell;
- binds call and error result by explicit call identity;
- has paired-error status;
- retains the normalized error signature and evidence spans.

Generic shell errors, unpaired results, text mentions, and malformed-request heuristics are excluded.
A negative decision requires complete call/result pairing capability.

### Contextual Recovery After Tool Error

The deterministic prerequisite proves an executed error followed by a later task-directed action in the
bounded window. Traces that fail this prerequisite do not invoke a semantic model.

The semantic residual receives only:

- the authorized error event;
- the immediately relevant user-intent span;
- bounded subsequent actions.

It decides whether the later action materially adapts strategy while continuing toward the same intent.
It abstains when intent continuity, action meaning, relevant spans, or truncation state are incomplete.
An identical retry alone is not recovery.

## Calibration Boundary

R0 canaries are development data. They may be used to create and debug gold records but cannot support
production precision, recall, or F1 claims.

R3-06 performs functional calibration on adjudicated canary annotations. R8 performs statistical
evaluation on frozen independent sets:

- at least 50 positive and 50 negative traces for each structured label;
- at least 100 frozen independent traces for the semantic label;
- trace-ID isolation between development and independent evaluation.

Until R0-09 human review completes, canary manifest traits and candidates remain routing evidence, not
gold label truth.

## Release Boundaries

### LH v1

- Required first profile.
- Enabled only for `CANARY` and `INTERNAL_REVIEW` in R0-R7.
- Uses an isolated non-production registry.
- Requires current package hash, exact ProvenanceManifest equality, QualityReport, human reviews, and
  ReleaseDecision.
- Production requires R8 attestation and a new production ReleaseDecision.

### Generic v1

- `DEFERRED_APPROVAL_REQUIRED`.
- Enabled channels: none.
- Cannot be selected by fallback.
- Requires a new approved decision covering package shape, evaluator compatibility, registry,
  provenance inventory, migration, and rollback.

## Rejected Alternatives

### Structured Labels Only

Rejected because the factory must exercise a bounded semantic residual path and abstention before R3.

### Semantic Labels Only

Rejected because search usage and paired PowerShell errors are deterministic facts; model calls would
increase cost and reduce reproducibility.

### Count Search And Fetch As One Label

Rejected. Search execution is the approved label. Fetch may be a prerequisite or separate future label,
but does not silently satisfy search usage.

### Treat Any PowerShell Text As An Error

Rejected. The label requires executed call/result identity and error status.

### Enable Generic Alongside LH

Rejected for the first slice. It expands release and evaluator surface without an approved Generic
dataset contract.

## Durable Artifacts

- `specs/002-eval-dataset-factory/labels/v1/search-tool-usage.json`
- `specs/002-eval-dataset-factory/labels/v1/powershell-error-signature.json`
- `specs/002-eval-dataset-factory/labels/v1/contextual-recovery-after-tool-error.json`
- `specs/002-eval-dataset-factory/release-profiles/v1/decision.json`

Changes create a new version and a CompatibilityDeclaration. Existing files are not overwritten.
