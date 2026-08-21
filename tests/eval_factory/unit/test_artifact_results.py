from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError
from test_artifact_execution import (
    HASH,
    _audit,
    _ControlledExecutionFacade,
    _definition,
    _execution_plan,
    _ref,
    _routing_source,
)

from env_mock_agent.facade import (
    AttachmentExecutionRequestV2,
    AttachmentExecutionResultV2,
    attachment_execution_result_carried_sha256,
)
from eval_factory.attachment_planning import (
    ArtifactExecutionPlanningOutcome,
    ArtifactExecutionPlanningResult,
    ArtifactGroupExecutor,
    ArtifactResultCompiler,
    ArtifactResultPolicyError,
    ArtifactRoutingCompilationResult,
    ArtifactRoutingRequest,
    artifact_routing_compilation_result_carried_sha256,
    artifact_routing_request_carried_sha256,
    artifact_routing_request_ref,
)
from eval_factory.contracts import (
    ArtifactBuildResultOutcomeV2,
    ArtifactBuildResultProjectionGapV2,
    ArtifactRouteEntryOutcomeV2,
    ArtifactRoutePlanEntryV2,
    ArtifactRoutingAggregateOutcomeV2,
    ArtifactRoutingPlanV2,
    ArtifactRoutingReasonV2,
    AttachmentReconstructionOutcomeV2,
    AttachmentReconstructionResultV2,
    artifact_build_result_v2_ref,
    artifact_execution_batch_carried_sha256,
    artifact_execution_plan_ref,
    artifact_execution_receipt_ref,
    artifact_routing_plan_carried_sha256,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.contracts.task import AttachmentCriticality

NOW = datetime(2026, 7, 28, tzinfo=UTC)
ROOT = Path(__file__).resolve().parents[3]
GOLD_PATH = (
    ROOT / "evals/golden/eval_factory/attachment_results" / "r5-07-partial-failure-minimal-retry-v1.json"
)


def _result_audit(*refs: ObjectRef) -> ContractAudit:
    return ContractAudit(
        created_at=NOW,
        created_by="artifact-result-test",
        governing_versions=(
            VersionBinding(
                component="artifact-results",
                version="r5-07",
            ),
        ),
        input_refs=refs,
    )


def _planning_result(plan) -> ArtifactExecutionPlanningResult:
    return ArtifactExecutionPlanningResult(
        outcome=ArtifactExecutionPlanningOutcome.PLANNED,
        world_ledger_snapshot=plan.world_ledger_snapshot,
        execution_plan=plan,
        audit=_audit(),
    )


async def _compiled_execution(
    tmp_path: Path,
    *,
    artifacts: tuple[tuple[str, str], ...],
    definitions,
    facade: _ControlledExecutionFacade,
):
    routing_request, routing_result = _routing_source(artifacts)
    execution_plan = await _execution_plan(
        tmp_path,
        artifacts,
        definitions,
    )
    batch = await ArtifactGroupExecutor().run(
        execution_plan,
        facade=facade,
        audit=_audit(),
    )
    result = ArtifactResultCompiler().compile(
        routing_request=routing_request,
        routing_result=routing_result,
        execution_result=_planning_result(execution_plan),
        execution_batch=batch,
        audit=_result_audit(),
    )
    return routing_request, routing_result, execution_plan, batch, result


def _not_required_sources() -> tuple[
    ArtifactRoutingRequest,
    ArtifactRoutingCompilationResult,
    ArtifactExecutionPlanningResult,
]:
    request = ArtifactRoutingRequest(
        request_id="artifact-routing-request://pending",
        attachment_planning_context_ref=_ref(
            "attachment-planning-context",
            "current",
        ),
        producer_task_view_ref=_ref("producer-task-view", "current"),
        artifact_evidence_matrix_ref=None,
        artifact_routing_policy_ref=_ref(
            "artifact-routing-policy",
            "current",
        ),
        build_contracts=(),
        facade_requests=(),
        missing_contract_target_refs=(),
        blocked_mode_target_refs=(),
        policy_version="artifact-routing/r5-05-v1",
        request_sha256=HASH,
        audit=_audit(),
    )
    request_digest = artifact_routing_request_carried_sha256(request)
    request = request.model_copy(
        update={
            "request_id": f"artifact-routing-request://sha256/{request_digest}",
            "request_sha256": request_digest,
        }
    )
    result = ArtifactRoutingCompilationResult(
        result_id="artifact-routing-compilation-result://pending",
        request_ref=artifact_routing_request_ref(request),
        execution_result_ref=_ref(
            "artifact-routing-execution-result",
            "none",
            version="r5-05",
        ),
        outcome=ArtifactRoutingAggregateOutcomeV2.NOT_REQUIRED,
        routing_plan=None,
        policy_version="artifact-routing/r5-05-v1",
        result_sha256=HASH,
        audit=_audit(),
    )
    result_digest = artifact_routing_compilation_result_carried_sha256(result)
    result = result.model_copy(
        update={
            "result_id": f"artifact-routing-compilation-result://sha256/{result_digest}",
            "result_sha256": result_digest,
        }
    )
    execution = ArtifactExecutionPlanningResult(
        outcome=ArtifactExecutionPlanningOutcome.NOT_REQUIRED,
        world_ledger_snapshot=None,
        execution_plan=None,
        audit=_audit(),
    )
    return request, result, execution


def _blocked_sources(
    *,
    criticality: AttachmentCriticality,
) -> tuple[
    ArtifactRoutingRequest,
    ArtifactRoutingCompilationResult,
    ArtifactExecutionPlanningResult,
]:
    request, result = _routing_source((("artifact://a", "inputs/a.txt"),))
    assert result.routing_plan is not None
    routed = result.routing_plan.entries[0]
    entry_values = routed.model_dump(mode="python")
    entry_values.update(
        criticality=criticality,
        outcome=ArtifactRouteEntryOutcomeV2.BLOCKED_CAPABILITY,
        facade_route_request_ref=None,
        facade_route_decision=None,
        build_spec=None,
        reasons=frozenset({ArtifactRoutingReasonV2.ROUTE_UNAVAILABLE}),
    )
    blocked = ArtifactRoutePlanEntryV2.model_validate(entry_values)
    aggregate = (
        ArtifactRoutingAggregateOutcomeV2.PARTIALLY_ROUTED
        if criticality is AttachmentCriticality.OPTIONAL
        else ArtifactRoutingAggregateOutcomeV2.BLOCKED_CAPABILITY
    )
    plan_values = result.routing_plan.model_dump(mode="python")
    plan_values.update(
        entries=(blocked,),
        aggregate_outcome=aggregate,
        routed_artifact_ids=(),
        blocked_required_artifact_ids=(
            () if criticality is AttachmentCriticality.OPTIONAL else ("artifact://a",)
        ),
        blocked_optional_artifact_ids=(
            ("artifact://a",) if criticality is AttachmentCriticality.OPTIONAL else ()
        ),
        artifact_routing_plan_sha256=HASH,
    )
    plan = ArtifactRoutingPlanV2.model_validate(plan_values)
    plan_digest = artifact_routing_plan_carried_sha256(plan)
    plan = plan.model_copy(
        update={
            "artifact_routing_plan_id": f"artifact-routing-plan://sha256/{plan_digest}",
            "artifact_routing_plan_sha256": plan_digest,
        }
    )
    result_values = result.model_dump(mode="python")
    result_values.update(
        outcome=aggregate,
        routing_plan=plan,
        result_sha256=HASH,
    )
    result = ArtifactRoutingCompilationResult.model_validate(result_values)
    result_digest = artifact_routing_compilation_result_carried_sha256(result)
    result = result.model_copy(
        update={
            "result_id": f"artifact-routing-compilation-result://sha256/{result_digest}",
            "result_sha256": result_digest,
        }
    )
    execution = ArtifactExecutionPlanningResult(
        outcome=ArtifactExecutionPlanningOutcome.NO_ROUTED_ARTIFACTS,
        world_ledger_snapshot=None,
        execution_plan=None,
        audit=_audit(),
    )
    return request, result, execution


class _LongWorkerExecutionFacade(_ControlledExecutionFacade):
    @staticmethod
    def _result(
        request: AttachmentExecutionRequestV2,
        *,
        status,
    ) -> AttachmentExecutionResultV2:
        result = _ControlledExecutionFacade._result(
            request,
            status=status,
        )
        if result.worker_version is None:
            return result
        result = result.model_copy(
            update={
                "worker_version": "w" * 129,
                "execution_result_sha256": HASH,
            }
        )
        digest = attachment_execution_result_carried_sha256(result)
        return result.model_copy(
            update={
                "execution_result_id": f"attachment-execution-result://sha256/{digest}",
                "execution_result_sha256": digest,
            }
        )


class _ObservedFailureExecutionFacade(_ControlledExecutionFacade):
    @staticmethod
    def _result(
        request: AttachmentExecutionRequestV2,
        *,
        status,
    ) -> AttachmentExecutionResultV2:
        result = _ControlledExecutionFacade._result(
            request,
            status=status,
        )
        if result.failure_code is None:
            return result
        result = result.model_copy(
            update={
                "worker_version": "test.FailedProvider",
                "execution_result_sha256": HASH,
            }
        )
        digest = attachment_execution_result_carried_sha256(result)
        return result.model_copy(
            update={
                "execution_result_id": (f"attachment-execution-result://sha256/{digest}"),
                "execution_result_sha256": digest,
            }
        )


@pytest.mark.asyncio
async def test_compile_successes_as_validation_candidates(
    tmp_path: Path,
) -> None:
    artifacts = (
        ("artifact://a", "inputs/a.txt"),
        ("artifact://b", "inputs/b.txt"),
    )
    request, routing, plan, batch, result = await _compiled_execution(
        tmp_path,
        artifacts=artifacts,
        definitions=(
            _definition("artifact://a"),
            _definition("artifact://b"),
        ),
        facade=_ControlledExecutionFacade(expected_parallelism=2),
    )

    assert result.outcome is AttachmentReconstructionOutcomeV2.PENDING_VALIDATION
    assert len(result.candidate_output_refs) == 2
    assert result.accepted_artifact_refs == ()
    assert result.failed_artifact_ids == ()
    assert result.resumable_artifact_ids == ()
    assert result.dependency_blocked_artifact_ids == ()
    assert result.environment_spec_ref is None
    assert result.provenance_manifest_ref is None
    assert result.quality_report_ref is None
    assert result.package_sha256 is None
    assert result.input_state_only is None
    assert all(item.frozen_result is not None for item in result.artifact_results)
    assert all(
        item.frozen_result.validation_result_refs == () and item.frozen_result.finding_refs == ()
        for item in result.artifact_results
        if item.frozen_result is not None
    )

    ArtifactResultCompiler().validate_current(
        result,
        routing_request=request,
        routing_result=routing,
        execution_result=_planning_result(plan),
        execution_batch=batch,
    )
    values = result.model_dump(mode="python")
    values["artifact_execution_plan_ref"] = _ref(
        "artifact-execution-plan",
        "other",
    )
    with pytest.raises(ValidationError, match="aggregate execution plan"):
        AttachmentReconstructionResultV2.model_validate(values)


@pytest.mark.asyncio
async def test_partial_failure_uses_direct_retry_root_and_preserves_success_on_resume(
    tmp_path: Path,
) -> None:
    artifacts = (
        ("artifact://a", "inputs/a.txt"),
        ("artifact://b", "inputs/b.txt"),
        ("artifact://c", "inputs/c.txt"),
    )
    definitions = (
        _definition("artifact://a"),
        _definition("artifact://b", dependencies=("artifact://a",)),
        _definition("artifact://c"),
    )
    request, routing, plan, first_batch, first_result = await _compiled_execution(
        tmp_path,
        artifacts=artifacts,
        definitions=definitions,
        facade=_ControlledExecutionFacade(
            fail_once=frozenset({"artifact://a"}),
            expected_parallelism=2,
        ),
    )

    assert first_result.outcome is AttachmentReconstructionOutcomeV2.PARTIAL_FAILURE
    assert first_result.failed_artifact_ids == ("artifact://a",)
    assert first_result.resumable_artifact_ids == ("artifact://a",)
    assert first_result.dependency_blocked_artifact_ids == ("artifact://b",)
    assert first_result.required_incomplete_artifact_ids == (
        "artifact://a",
        "artifact://b",
    )
    by_id = {item.route_entry.artifact_id: item for item in first_result.artifact_results}
    assert by_id["artifact://a"].frozen_projection_gap is (
        ArtifactBuildResultProjectionGapV2.WORKER_VERSION_UNAVAILABLE
    )
    assert by_id["artifact://b"].frozen_projection_gap is (
        ArtifactBuildResultProjectionGapV2.DEPENDENCY_NOT_ATTEMPTED
    )
    first_c_ref = artifact_build_result_v2_ref(by_id["artifact://c"])

    facade = _ControlledExecutionFacade()
    second_batch = await ArtifactGroupExecutor().run(
        plan,
        facade=facade,
        audit=_audit(),
        prior_batch=first_batch,
    )
    second_result = ArtifactResultCompiler().compile(
        routing_request=request,
        routing_result=routing,
        execution_result=_planning_result(plan),
        execution_batch=second_batch,
        audit=_result_audit(),
    )

    assert second_result.outcome is AttachmentReconstructionOutcomeV2.PENDING_VALIDATION
    second_by_id = {item.route_entry.artifact_id: item for item in second_result.artifact_results}
    assert artifact_build_result_v2_ref(second_by_id["artifact://c"]) == first_c_ref
    assert facade.calls == Counter(
        {
            "artifact://a": 1,
            "artifact://b": 1,
        }
    )
    with pytest.raises(ArtifactResultPolicyError, match="stale"):
        ArtifactResultCompiler().validate_current(
            first_result,
            routing_request=request,
            routing_result=routing,
            execution_result=_planning_result(plan),
            execution_batch=second_batch,
        )


@pytest.mark.asyncio
async def test_unrepresentable_worker_version_has_closed_projection_gap(
    tmp_path: Path,
) -> None:
    _, _, _, _, result = await _compiled_execution(
        tmp_path,
        artifacts=(("artifact://a", "inputs/a.txt"),),
        definitions=(_definition("artifact://a"),),
        facade=_LongWorkerExecutionFacade(),
    )

    artifact = result.artifact_results[0]
    assert artifact.outcome is ArtifactBuildResultOutcomeV2.SUCCEEDED
    assert artifact.frozen_result is None
    assert artifact.frozen_projection_gap is (
        ArtifactBuildResultProjectionGapV2.WORKER_VERSION_UNREPRESENTABLE
    )
    assert result.candidate_output_refs


@pytest.mark.asyncio
async def test_observed_retryable_failure_projects_safe_frozen_failure(
    tmp_path: Path,
) -> None:
    _, _, _, _, result = await _compiled_execution(
        tmp_path,
        artifacts=(("artifact://a", "inputs/a.txt"),),
        definitions=(_definition("artifact://a"),),
        facade=_ObservedFailureExecutionFacade(fail_once=frozenset({"artifact://a"})),
    )

    assert result.outcome is AttachmentReconstructionOutcomeV2.RETRYABLE_FAILURE
    assert result.failed_artifact_ids == ("artifact://a",)
    assert result.resumable_artifact_ids == ("artifact://a",)
    artifact = result.artifact_results[0]
    assert artifact.frozen_projection_gap is None
    assert artifact.frozen_result is not None
    assert artifact.frozen_result.status.value == "RETRYABLE_FAILURE"
    assert artifact.frozen_result.failure is not None
    assert artifact.frozen_result.failure.failure_class.value == ("ENVIRONMENT_FAILURE")
    assert artifact.frozen_result.failure.code == "RUNTIME_EXECUTION_FAILED"
    assert artifact.frozen_result.failure.retryable is True
    assert artifact.route_entry.build_spec is not None
    assert artifact.frozen_result.build_spec_sha256 == (
        artifact.route_entry.build_spec.artifact_build_spec_v2_sha256
    )


@pytest.mark.parametrize(
    "criticality",
    [
        AttachmentCriticality.REQUIRED,
        AttachmentCriticality.OPTIONAL,
    ],
)
def test_compile_all_routing_blocked_without_fabricated_build_result(
    criticality: AttachmentCriticality,
) -> None:
    request, routing, execution = _blocked_sources(criticality=criticality)

    result = ArtifactResultCompiler().compile(
        routing_request=request,
        routing_result=routing,
        execution_result=execution,
        execution_batch=None,
        audit=_result_audit(),
    )

    assert result.outcome is AttachmentReconstructionOutcomeV2.BLOCKED
    assert result.failed_artifact_ids == ("artifact://a",)
    assert result.resumable_artifact_ids == ()
    assert result.dependency_blocked_artifact_ids == ()
    assert result.candidate_output_refs == ()
    artifact = result.artifact_results[0]
    assert artifact.execution_receipt is None
    assert artifact.frozen_result is None
    assert artifact.frozen_projection_gap is (ArtifactBuildResultProjectionGapV2.BUILD_SPEC_UNAVAILABLE)
    if criticality is AttachmentCriticality.OPTIONAL:
        assert result.optional_incomplete_artifact_ids == ("artifact://a",)
        assert result.required_incomplete_artifact_ids == ()
    else:
        assert result.required_incomplete_artifact_ids == ("artifact://a",)
        assert result.optional_incomplete_artifact_ids == ()


def test_compile_not_required_without_artifact_or_package_claims() -> None:
    request, routing, execution = _not_required_sources()

    result = ArtifactResultCompiler().compile(
        routing_request=request,
        routing_result=routing,
        execution_result=execution,
        execution_batch=None,
        audit=_result_audit(),
    )

    assert result.outcome is AttachmentReconstructionOutcomeV2.NOT_REQUIRED
    assert result.artifact_results == ()
    assert result.artifact_result_refs == ()
    assert result.candidate_output_refs == ()
    assert result.frozen_result is None
    assert result.frozen_projection_gap == "PRE_VALIDATION"


def test_validate_current_rejects_tampered_result_audit() -> None:
    request, routing, execution = _not_required_sources()
    compiler = ArtifactResultCompiler()
    result = compiler.compile(
        routing_request=request,
        routing_result=routing,
        execution_result=execution,
        execution_batch=None,
        audit=_result_audit(),
    )
    tampered = result.model_copy(update={"audit": result.audit.model_copy(update={"input_refs": ()})})

    with pytest.raises(ArtifactResultPolicyError, match="audit lineage"):
        compiler.validate_current(
            tampered,
            routing_request=request,
            routing_result=routing,
            execution_result=execution,
            execution_batch=None,
        )


@pytest.mark.asyncio
async def test_compile_rejects_cross_plan_execution_batch(
    tmp_path: Path,
) -> None:
    request, routing, plan, batch, _ = await _compiled_execution(
        tmp_path,
        artifacts=(("artifact://a", "inputs/a.txt"),),
        definitions=(_definition("artifact://a"),),
        facade=_ControlledExecutionFacade(),
    )
    other_plan_ref = _ref(
        "artifact-execution-plan",
        "other",
    )
    batch_audit_refs = (
        other_plan_ref,
        *(artifact_execution_receipt_ref(item) for item in batch.receipts),
    )
    batch = batch.model_copy(
        update={
            "artifact_execution_plan_ref": other_plan_ref,
            "artifact_execution_batch_sha256": HASH,
            "audit": batch.audit.model_copy(
                update={
                    "input_refs": tuple(
                        sorted(
                            batch_audit_refs,
                            key=lambda ref: (
                                ref.object_type,
                                ref.object_id,
                                ref.object_version,
                                ref.object_sha256,
                            ),
                        )
                    )
                }
            ),
        }
    )
    digest = artifact_execution_batch_carried_sha256(batch)
    batch = batch.model_copy(update={"artifact_execution_batch_sha256": digest})

    with pytest.raises(ArtifactResultPolicyError, match="plan"):
        ArtifactResultCompiler().compile(
            routing_request=request,
            routing_result=routing,
            execution_result=_planning_result(plan),
            execution_batch=batch,
            audit=_result_audit(artifact_execution_plan_ref(plan)),
        )


@pytest.mark.asyncio
async def test_artifact_result_gold_is_executable(tmp_path: Path) -> None:
    payload = json.loads(GOLD_PATH.read_text(encoding="utf-8"))

    for case in payload["cases"]:
        result = await _gold_case_result(
            case["scenario"],
            tmp_path / case["case_id"],
        )
        assert result.outcome.value == case["expected_outcome"]
        assert len(result.candidate_output_refs) == case["expected_candidate_count"]
        assert list(result.failed_artifact_ids) == case["expected_failed_artifact_ids"]
        assert list(result.resumable_artifact_ids) == case["expected_resumable_artifact_ids"]
        assert (
            list(result.dependency_blocked_artifact_ids) == case["expected_dependency_blocked_artifact_ids"]
        )
        assert (
            list(result.required_incomplete_artifact_ids) == case["expected_required_incomplete_artifact_ids"]
        )
        assert (
            list(result.optional_incomplete_artifact_ids) == case["expected_optional_incomplete_artifact_ids"]
        )


async def _gold_case_result(
    scenario: str,
    tmp_path: Path,
) -> AttachmentReconstructionResultV2:
    if scenario == "NOT_REQUIRED":
        request, routing, execution = _not_required_sources()
        return ArtifactResultCompiler().compile(
            routing_request=request,
            routing_result=routing,
            execution_result=execution,
            execution_batch=None,
            audit=_result_audit(),
        )
    if scenario in {
        "ROUTING_BLOCKED_REQUIRED",
        "ROUTING_BLOCKED_OPTIONAL",
    }:
        criticality = (
            AttachmentCriticality.REQUIRED
            if scenario == "ROUTING_BLOCKED_REQUIRED"
            else AttachmentCriticality.OPTIONAL
        )
        request, routing, execution = _blocked_sources(criticality=criticality)
        return ArtifactResultCompiler().compile(
            routing_request=request,
            routing_result=routing,
            execution_result=execution,
            execution_batch=None,
            audit=_result_audit(),
        )
    if scenario == "ALL_SUCCESS":
        artifacts = (
            ("artifact://a", "inputs/a.txt"),
            ("artifact://b", "inputs/b.txt"),
        )
        return (
            await _compiled_execution(
                tmp_path,
                artifacts=artifacts,
                definitions=(
                    _definition("artifact://a"),
                    _definition("artifact://b"),
                ),
                facade=_ControlledExecutionFacade(expected_parallelism=2),
            )
        )[-1]
    if scenario == "PARTIAL_RETRY_DEPENDENCY":
        return (
            await _compiled_execution(
                tmp_path,
                artifacts=(
                    ("artifact://a", "inputs/a.txt"),
                    ("artifact://b", "inputs/b.txt"),
                    ("artifact://c", "inputs/c.txt"),
                ),
                definitions=(
                    _definition("artifact://a"),
                    _definition(
                        "artifact://b",
                        dependencies=("artifact://a",),
                    ),
                    _definition("artifact://c"),
                ),
                facade=_ControlledExecutionFacade(
                    fail_once=frozenset({"artifact://a"}),
                    expected_parallelism=2,
                ),
            )
        )[-1]
    raise AssertionError(f"unknown gold scenario: {scenario}")
