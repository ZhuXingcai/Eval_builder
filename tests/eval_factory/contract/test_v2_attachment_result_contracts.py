from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from eval_factory.contracts import (
    ArtifactBuildResultOutcomeV2,
    ArtifactBuildResultProjectionGapV2,
    ArtifactBuildResultV2,
    ArtifactRouteEntryOutcomeV2,
    ArtifactRoutePlanEntryV2,
    ArtifactRoutingReasonV2,
    AttachmentReconstructionOutcomeV2,
    AttachmentReconstructionResultV2,
    artifact_build_result_v2_carried_sha256,
    artifact_build_result_v2_ref,
    attachment_reconstruction_result_v2_carried_sha256,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.contracts.task import AttachmentCriticality

HASH = "a" * 64
NOW = datetime(2026, 7, 28, tzinfo=UTC)


def _ref(
    object_type: str,
    suffix: str,
    *,
    version: str = "v2",
    digest: str = HASH,
) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://{suffix}",
        object_version=version,
        object_sha256=digest,
    )


def _audit(*refs: ObjectRef, at: datetime = NOW) -> ContractAudit:
    return ContractAudit(
        created_at=at,
        created_by="artifact-result-contract-test",
        governing_versions=(
            VersionBinding(
                component="artifact-results",
                version="r5-07",
            ),
        ),
        input_refs=tuple(sorted(refs, key=_ref_key)),
    )


def _ref_key(ref: ObjectRef) -> tuple[str, str, str, str]:
    return (
        ref.object_type,
        ref.object_id,
        ref.object_version,
        ref.object_sha256,
    )


def _blocked_entry(
    *,
    criticality: AttachmentCriticality = AttachmentCriticality.REQUIRED,
) -> ArtifactRoutePlanEntryV2:
    return ArtifactRoutePlanEntryV2(
        artifact_evidence_target_ref=_ref(
            "artifact-evidence-target",
            "a",
        ),
        attachment_dependency_id="attachment-dependency://a",
        artifact_id="artifact://a",
        criticality=criticality,
        outcome=ArtifactRouteEntryOutcomeV2.BLOCKED_CAPABILITY,
        facade_route_request_ref=None,
        facade_route_decision=None,
        build_spec=None,
        reasons=frozenset({ArtifactRoutingReasonV2.ROUTE_UNAVAILABLE}),
    )


def _blocked_result(
    *,
    at: datetime = NOW,
) -> ArtifactBuildResultV2:
    plan_ref = _ref("artifact-routing-plan", "blocked")
    value = ArtifactBuildResultV2(
        artifact_build_result_v2_id="artifact-build-result://pending",
        artifact_routing_plan_ref=plan_ref,
        route_entry=_blocked_entry(),
        artifact_execution_plan_ref=None,
        execution_receipt=None,
        outcome=ArtifactBuildResultOutcomeV2.BLOCKED_CAPABILITY,
        retryable=False,
        frozen_result=None,
        frozen_projection_gap=ArtifactBuildResultProjectionGapV2.BUILD_SPEC_UNAVAILABLE,
        policy_version="artifact-results/r5-07-v1",
        artifact_build_result_v2_sha256=HASH,
        audit=_audit(plan_ref, at=at),
    )
    digest = artifact_build_result_v2_carried_sha256(value)
    return value.model_copy(
        update={
            "artifact_build_result_v2_id": f"artifact-build-result://sha256/{digest}",
            "artifact_build_result_v2_sha256": digest,
        }
    )


def _not_required_result() -> AttachmentReconstructionResultV2:
    request_ref = _ref(
        "artifact-routing-request",
        "none",
        version="r5-05",
    )
    compilation_ref = _ref(
        "artifact-routing-compilation-result",
        "none",
        version="r5-05",
    )
    value = AttachmentReconstructionResultV2(
        attachment_reconstruction_result_v2_id=("attachment-reconstruction-result://pending"),
        routing_request_ref=request_ref,
        routing_compilation_result_ref=compilation_ref,
        attachment_planning_context_ref=_ref(
            "attachment-planning-context",
            "current",
        ),
        producer_task_view_ref=_ref("producer-task-view", "current"),
        evidence_matrix_ref=None,
        artifact_routing_plan_ref=None,
        artifact_execution_plan_ref=None,
        artifact_execution_batch_ref=None,
        artifact_results=(),
        artifact_result_refs=(),
        outcome=AttachmentReconstructionOutcomeV2.NOT_REQUIRED,
        candidate_output_refs=(),
        accepted_artifact_refs=(),
        failed_artifact_ids=(),
        resumable_artifact_ids=(),
        dependency_blocked_artifact_ids=(),
        required_incomplete_artifact_ids=(),
        optional_incomplete_artifact_ids=(),
        frozen_result=None,
        frozen_projection_gap="PRE_VALIDATION",
        environment_spec_ref=None,
        provenance_manifest_ref=None,
        quality_report_ref=None,
        package_sha256=None,
        input_state_only=None,
        policy_version="artifact-results/r5-07-v1",
        attachment_reconstruction_result_v2_sha256=HASH,
        audit=_audit(request_ref, compilation_ref),
    )
    digest = attachment_reconstruction_result_v2_carried_sha256(value)
    return value.model_copy(
        update={
            "attachment_reconstruction_result_v2_id": (f"attachment-reconstruction-result://sha256/{digest}"),
            "attachment_reconstruction_result_v2_sha256": digest,
        }
    )


def test_blocked_artifact_result_requires_exact_projection_gap() -> None:
    result = _blocked_result()

    assert result.outcome is ArtifactBuildResultOutcomeV2.BLOCKED_CAPABILITY
    assert result.frozen_result is None
    assert result.frozen_projection_gap is (ArtifactBuildResultProjectionGapV2.BUILD_SPEC_UNAVAILABLE)
    assert result.execution_receipt is None
    assert artifact_build_result_v2_ref(result).object_sha256 == (result.artifact_build_result_v2_sha256)

    values = result.model_dump(mode="python")
    values["frozen_projection_gap"] = ArtifactBuildResultProjectionGapV2.DEPENDENCY_NOT_ATTEMPTED
    with pytest.raises(ValidationError, match="projection gap"):
        ArtifactBuildResultV2.model_validate(values)


def test_result_contracts_reject_unknown_or_premature_package_truth() -> None:
    blocked = _blocked_result()
    blocked_values = blocked.model_dump(mode="python")
    blocked_values["output_path"] = "/private/staging/a.txt"
    with pytest.raises(ValidationError, match="extra"):
        ArtifactBuildResultV2.model_validate(blocked_values)

    aggregate = _not_required_result()
    aggregate_values = aggregate.model_dump(mode="python")
    aggregate_values["package_sha256"] = HASH
    with pytest.raises(ValidationError, match="package"):
        AttachmentReconstructionResultV2.model_validate(aggregate_values)


def test_not_required_aggregate_is_pre_validation_and_content_free() -> None:
    result = _not_required_result()

    assert result.outcome is AttachmentReconstructionOutcomeV2.NOT_REQUIRED
    assert result.artifact_results == ()
    assert result.accepted_artifact_refs == ()
    assert result.environment_spec_ref is None
    assert result.provenance_manifest_ref is None
    assert result.quality_report_ref is None
    assert result.package_sha256 is None
    assert result.input_state_only is None

    serialized = result.model_dump(mode="json")
    for denied in (
        "output_path",
        "runtime_transcript",
        "world_ledger_values",
        "private_reference",
    ):
        assert denied not in serialized


def test_aggregate_rejects_accepted_or_misclassified_artifacts() -> None:
    aggregate = _not_required_result()
    values = aggregate.model_dump(mode="python")
    values["accepted_artifact_refs"] = (_ref("attachment-output", "unvalidated"),)
    with pytest.raises(ValidationError, match="accepted"):
        AttachmentReconstructionResultV2.model_validate(values)

    blocked = _blocked_result()
    blocked_ref = artifact_build_result_v2_ref(blocked)
    blocked_values = aggregate.model_dump(mode="python")
    blocked_values.update(
        artifact_routing_plan_ref=blocked.artifact_routing_plan_ref,
        artifact_results=(blocked,),
        artifact_result_refs=(blocked_ref,),
        outcome=AttachmentReconstructionOutcomeV2.BLOCKED,
        failed_artifact_ids=("artifact://a",),
        required_incomplete_artifact_ids=("artifact://a",),
    )
    with pytest.raises(ValidationError, match="evidence matrix"):
        AttachmentReconstructionResultV2.model_validate(blocked_values)

    blocked_values.update(
        evidence_matrix_ref=_ref("artifact-evidence-matrix", "current"),
        failed_artifact_ids=(),
    )
    with pytest.raises(ValidationError, match="failed"):
        AttachmentReconstructionResultV2.model_validate(blocked_values)


def test_result_carried_hashes_ignore_audit_actor_and_time() -> None:
    first = _blocked_result(at=datetime(2026, 7, 27, tzinfo=UTC))
    second = _blocked_result(at=datetime(2026, 7, 29, tzinfo=UTC)).model_copy(
        update={"audit": _blocked_result().audit.model_copy(update={"created_by": "another-result-compiler"})}
    )

    assert artifact_build_result_v2_carried_sha256(first) == (artifact_build_result_v2_carried_sha256(second))
    assert first.canonical_sha256() != second.canonical_sha256()
