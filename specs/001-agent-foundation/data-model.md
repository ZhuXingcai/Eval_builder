# Data Model: Agent Foundation

## Core Entities

- `TaskSpec`: normalized task intent, dependencies, outputs, and source references.
- `DependencySpec`: one required input with evidence, criticality, strategy, and quality floor.
- `ForbiddenOutputSpec`: an output or conclusion that must not appear in the input package.
- `SourceEvidence`: retrieved source metadata, hash, relevant excerpt, and usage policy.
- `ArtifactPlan`: provider/runtime choice, content contract, render contract, and validators.
- `RuntimeRequest` / `RuntimeEvent`: vendor-neutral execution protocol.
- `ValidationFinding`: deterministic or semantic quality finding.
- `Problem`: stable cross-round issue with severity and lifecycle.
- `RunRecord`: persisted workflow identity, status, timestamps, and output paths.

## State Rules

- All identifiers are stable strings.
- Timestamps are timezone-aware UTC values.
- File paths stored in schemas are portable POSIX-style relative paths unless explicitly external.
- Critical dependencies cannot end in a degraded success state.
- P0 problems block export.
- Problems are appended or transitioned; previously closed records are not overwritten.
