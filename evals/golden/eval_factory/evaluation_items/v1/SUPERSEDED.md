# Superseded R0-09 Review Workflow

The immutable v1 `gold-worklist.json` and `review-queue.json` are retained as historical R0 artifacts.
Their bytes and hashes must not change.

They are not scheduled by the active Eval Dataset Factory workflow. ADR 0011 and Beads decision
`env_mock_agent-ujc.1.13` replaced mandatory independent submissions, adjudication, reviewer
identities, and quorums with optional user-directed approval checkpoints.

The active R0-09 reference fixtures and supersession binding are recorded in:

```text
evals/golden/eval_factory/evaluation_items/v2/manifest.json
```

No pending v1 case requires assignment or completion.
