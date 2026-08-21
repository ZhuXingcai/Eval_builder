# Versioning And Release Lines

Eval Builder uses explicit product lines rather than treating every milestone
as one continuously widening product.

## EDF Foundation

The foundation contains the original trace-driven evaluation dataset factory:

- Trace registration, parsing, recovery, normalization, and storage;
- task authoring and attachment reconstruction;
- rubric, grading, quality, safety, duplicate, leakage, and lineage checks;
- approval, release projection, readiness, and production-blocking contracts.

This line remains the owner of domain behavior used by the Harness.

## V1 Vertical Edition

V1 is the focused product for generic Agent evaluation data.

```text
Stage 0  Harness contracts
Stage 1  Persistent session and requirement Agent loop
Stage 2  Capability runtime and static generic-agent-trace Pack
Stage 3  Collaborative Team runtime
Stage 4  Default Factory Graph and immutable candidate delivery
Stage 5  Conversation-first Agent Shell
Stage 6  User delivery and evidence freeze
```

Current status:

| Stage | State |
|---|---|
| 0 | complete |
| 1 | mechanism complete; real semantic pending |
| 2 | complete |
| 3 | complete |
| 4 | mechanism complete; real semantic pending |
| 5 | planned |
| 6 | planned |

V1 keeps one statically registered `generic-agent-trace` Pack. It does not
include runtime plugin installation or a generic graph editor.

## V2 Platform Edition

V2 extends the same Harness with:

- installable and versioned Eval Packs;
- registry-driven Blueprint compilation;
- namespaced schema and capability compatibility;
- Pack certification, upgrade, rollback, and replay;
- a second materially different first-party Pack;
- cross-domain workspaces and delivery adapters.

V2 implementation must not start until Stage 6 has frozen the V1 static
composition and real semantic evidence.

## Public Git Tags

Public snapshots use descriptive pre-stable tags:

```text
v1-stage4-mechanism
  Stages 0-4 implementation, recovery, full regression, and current
  PlanReview/Graph Console.

v1-stage5-6-planning
  The same implementation plus reviewed Stage 5/6 product and technical
  plans. This tag does not claim those stages are implemented.
```

No `v1.0.0` tag should be created until:

1. Stage 5 implementation and Browser E2E pass;
2. Stage 6 deterministic delivery package passes;
3. the frozen real semantic benchmark passes;
4. the evidence freeze reports `V1_STABLE`;
5. production-release status remains accurately represented.

## Compatibility Rules

- Pack `1.0.0`, `1.1.0`, and `1.2.0` resolve by exact version.
- Replays never silently upgrade Pack, Blueprint, schema, provider, graph, or
  delivery authority.
- Additive V2 contracts do not mutate frozen V1 artifacts in place.
- Fixture and real semantic evidence are separate versioned authorities.
- Production release is not implied by a package, tag, test pass, or
  non-production publication.
