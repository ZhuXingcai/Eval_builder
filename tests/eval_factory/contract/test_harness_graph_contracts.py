from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.harness.graph_models import (
    GraphCheckpointOutcomeV1,
    GraphCheckpointPhaseV1,
    HarnessGraphCheckpointV1,
    HarnessGraphExecutionBindingV1,
)

HASH = "a" * 64


def _ref(
    object_type: str,
    suffix: str = "example",
    *,
    version: str = "v1",
) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://{suffix}/{version}",
        object_version=version,
        object_sha256=HASH,
    )


def _audit(
    *,
    created_by: str = "graph-contract-test",
) -> ContractAudit:
    return ContractAudit(
        created_at=datetime(2026, 8, 19, tzinfo=UTC),
        created_by=created_by,
        governing_versions=(
            VersionBinding(
                component="eval-harness-stage4",
                version="v1",
                sha256=HASH,
            ),
        ),
    )


def _binding() -> HarnessGraphExecutionBindingV1:
    return _create_binding()


def _create_binding(
    **overrides: object,
) -> HarnessGraphExecutionBindingV1:
    values: dict[str, object] = {
        "binding_id": "graph-binding.stage4",
        "session_ref": _ref("harness-session"),
        "requirement_ref": _ref(
            "evaluation-requirement-spec",
            version="v2",
        ),
        "requirement_policy_ref": _ref("harness-requirement-policy"),
        "factory_run_ref": _ref("factory-run", version="v2"),
        "factory_request_ref": _ref(
            "factory-dataset-run-request",
            version="v2",
        ),
        "factory_policy_ref": _ref("factory-run-policy", version="v2"),
        "expected_factory_run_version": 2,
        "pack_manifest_ref": _ref("eval-pack-manifest"),
        "composition_ref": _ref("harness-composition"),
        "blueprint_ref": _ref("evaluation-blueprint"),
        "team_ref": _ref("agent-team"),
        "roster_ref": _ref("team-roster"),
        "task_graph_ref": _ref("team-task-graph"),
        "execution_authority_ref": _ref("execution-authority"),
        "team_checkpoint_ref": _ref("team-checkpoint"),
        "expected_team_version": 1,
        "expected_task_graph_revision": 1,
        "expected_authority_version": 1,
        "blackboard_head_refs": (
            _ref("artifact-head", "b"),
            _ref("artifact-head", "a"),
        ),
        "thread_id": "graph-thread-stage4",
        "max_transitions": 256,
        "audit": _audit(),
    }
    values.update(overrides)
    return HarnessGraphExecutionBindingV1.create(**values)  # type: ignore[arg-type]


def test_graph_binding_is_strict_ref_only_and_audit_independent() -> None:
    binding = _binding()
    changed_audit = _create_binding(
        audit=_audit(created_by="another-actor"),
    )

    assert binding.to_ref() == changed_audit.to_ref()
    assert binding.blackboard_head_refs == (
        _ref("artifact-head", "a"),
        _ref("artifact-head", "b"),
    )
    assert "payload" not in type(binding).model_fields
    schema = HarnessGraphExecutionBindingV1.model_json_schema()
    assert schema["additionalProperties"] is False

    unknown = binding.model_dump(mode="python")
    unknown["raw_trace"] = "forbidden"
    with pytest.raises(ValidationError):
        HarnessGraphExecutionBindingV1.model_validate(unknown)


def test_graph_binding_rejects_wrong_authority_types() -> None:
    with pytest.raises(
        ValidationError,
        match="blueprint_ref",
    ):
        _create_binding(
            blueprint_ref=_ref("dataset-build-plan", version="v2"),
        )


def test_graph_checkpoint_enforces_pre_post_authority() -> None:
    binding = _binding()
    pre = HarnessGraphCheckpointV1.create(
        checkpoint_id="graph-checkpoint.stage4.1.pre",
        binding_ref=binding.to_ref(),
        phase=GraphCheckpointPhaseV1.PRE_TRANSITION,
        node="dispatch_ready_work",
        transition_number=1,
        predecessor_checkpoint_ref=None,
        factory_run_ref=binding.factory_run_ref,
        team_ref=binding.team_ref,
        team_checkpoint_ref=binding.team_checkpoint_ref,
        blackboard_head_refs=binding.blackboard_head_refs,
        planner_assessment_ref=None,
        outcome=GraphCheckpointOutcomeV1.READY,
        reason_codes=(),
        audit=_audit(),
    )
    post = HarnessGraphCheckpointV1.create(
        checkpoint_id="graph-checkpoint.stage4.1.post",
        binding_ref=binding.to_ref(),
        phase=GraphCheckpointPhaseV1.POST_TRANSITION,
        node=pre.node,
        transition_number=pre.transition_number,
        predecessor_checkpoint_ref=pre.to_ref(),
        factory_run_ref=binding.factory_run_ref,
        team_ref=binding.team_ref,
        team_checkpoint_ref=binding.team_checkpoint_ref,
        blackboard_head_refs=binding.blackboard_head_refs,
        planner_assessment_ref=_ref(
            "planner-assessment",
            version="v2",
        ),
        outcome=GraphCheckpointOutcomeV1.COMMITTED,
        reason_codes=(),
        audit=_audit(),
    )

    assert post.predecessor_checkpoint_ref == pre.to_ref()
    assert post.audit.input_refs == tuple(
        sorted(
            post.audit.input_refs,
            key=lambda value: (
                value.object_type,
                value.object_id,
                value.object_version,
                value.object_sha256,
            ),
        )
    )

    with pytest.raises(ValidationError, match="READY"):
        HarnessGraphCheckpointV1.create(
            checkpoint_id=pre.checkpoint_id,
            binding_ref=pre.binding_ref,
            phase=pre.phase,
            node=pre.node,
            transition_number=pre.transition_number,
            predecessor_checkpoint_ref=pre.predecessor_checkpoint_ref,
            factory_run_ref=pre.factory_run_ref,
            team_ref=pre.team_ref,
            team_checkpoint_ref=pre.team_checkpoint_ref,
            blackboard_head_refs=pre.blackboard_head_refs,
            planner_assessment_ref=pre.planner_assessment_ref,
            outcome=GraphCheckpointOutcomeV1.COMMITTED,
            reason_codes=pre.reason_codes,
            audit=_audit(),
        )


def test_graph_checkpoint_requires_closed_block_reason() -> None:
    binding = _binding()
    with pytest.raises(ValidationError, match="reason codes"):
        HarnessGraphCheckpointV1.create(
            checkpoint_id="graph-checkpoint.stage4.blocked",
            binding_ref=binding.to_ref(),
            phase=GraphCheckpointPhaseV1.RECONCILED,
            node="planner_assessment",
            transition_number=2,
            predecessor_checkpoint_ref=_ref("graph-checkpoint"),
            factory_run_ref=binding.factory_run_ref,
            team_ref=binding.team_ref,
            team_checkpoint_ref=binding.team_checkpoint_ref,
            blackboard_head_refs=(),
            planner_assessment_ref=None,
            outcome=GraphCheckpointOutcomeV1.VERIFICATION_REQUIRED,
            reason_codes=(),
            audit=_audit(),
        )
