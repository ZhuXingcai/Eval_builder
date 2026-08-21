from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from eval_factory.contracts.agent_system_v2 import FactoryRunStatusV2
from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.contracts.dataset_runtime_v2 import (
    CandidateDatasetDeliveryManifestV2,
    CandidateDatasetOutcomeV2,
    CompiledDatasetDeliveryPlanV2,
    DatasetDeliveryPlanV2,
    FactoryDatasetAggregateResultV2,
    FactoryDatasetNextActionV2,
    FactoryDatasetPlanningAuthorityV2,
    FactoryDatasetRunRequestV2,
    FactoryDatasetRunViewV2,
    FactoryItemRunBindingV2,
    FactoryItemStageHeadV2,
    FactoryItemStageOutcomeV2,
    FactoryItemStageV2,
)

HASH = "a" * 64


def _ref(
    object_type: str,
    suffix: str = "example",
    *,
    version: str = "v2",
) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://{suffix}/{version}",
        object_version=version,
        object_sha256=HASH,
    )


def _audit(
    actor: str = "dataset-runtime-contract-test",
    *,
    day: int = 7,
) -> ContractAudit:
    return ContractAudit(
        created_at=datetime(2026, 8, day, tzinfo=UTC),
        created_by=actor,
        governing_versions=(
            VersionBinding(
                component="factory-dataset-runtime",
                version="v1",
                sha256=HASH,
            ),
        ),
    )


def _request(*, audit: ContractAudit | None = None) -> FactoryDatasetRunRequestV2:
    return FactoryDatasetRunRequestV2.create(
        dataset_run_id="factory-run://dataset/example",
        requirement_spec_ref=_ref("evaluation-requirement-spec"),
        manifest_ref=_ref("trace-manifest"),
        source_authorization_ref=_ref("trace-source-authorization"),
        factory_policy_ref=_ref("factory-run-policy"),
        pipeline_policy_refs=(
            _ref("batch-quality-policy"),
            _ref("release-projection-policy"),
        ),
        gateway_registry_refs=(
            _ref("agent-registry"),
            _ref("model-catalog"),
            _ref("prompt-registry"),
        ),
        output_target_ref=_ref("candidate-output-target"),
        idempotency_key="create-dataset-example",
        max_transitions=512,
        audit=audit or _audit(),
    )


def _binding(
    *,
    version: int = 1,
    predecessor: ObjectRef | None = None,
    suffix: str = "one",
    audit: ContractAudit | None = None,
) -> FactoryItemRunBindingV2:
    return FactoryItemRunBindingV2.create(
        dataset_run_ref=_ref("factory-run", "dataset"),
        item_run_ref=_ref("factory-run", f"item-{suffix}"),
        item_id=f"item://dataset/{suffix}",
        binding_version=version,
        predecessor_binding_ref=predecessor,
        trace_candidate_decision_ref=_ref(
            "trace-candidate-decision",
            suffix,
        ),
        extracted_prompt_ref=_ref("extracted-user-prompt", suffix),
        inferred_intent_ref=_ref("inferred-user-intent", suffix),
        rewrite_candidate_ref=_ref("task-rewrite-candidate", suffix),
        audit=audit or _audit(),
    )


def _stage_head(
    binding: FactoryItemRunBindingV2,
    *,
    stage: FactoryItemStageV2 = FactoryItemStageV2.ATTACHMENT,
    result_type: str = "attachment-subgraph-result",
    version: int = 1,
    predecessor: ObjectRef | None = None,
    outcome: FactoryItemStageOutcomeV2 = (FactoryItemStageOutcomeV2.SUCCEEDED),
    reasons: tuple[str, ...] = (),
) -> FactoryItemStageHeadV2:
    return FactoryItemStageHeadV2.create(
        item_binding_ref=binding.to_ref(),
        item_run_ref=binding.item_run_ref,
        stage=stage,
        stage_version=version,
        predecessor_head_ref=predecessor,
        dependency_result_refs=(_ref("r4-task-contract-set", "dependency"),),
        result_ref=_ref(result_type),
        outcome=outcome,
        reason_codes=reasons,
        audit=_audit(),
    )


def test_dataset_run_request_is_strict_safe_and_audit_independent() -> None:
    first = _request()
    second = _request(audit=_audit("another-actor", day=8))

    assert first.object_sha256 == second.object_sha256
    assert first.object_id == second.object_id
    assert first.pipeline_policy_refs == tuple(
        sorted(first.pipeline_policy_refs, key=lambda value: value.object_type)
    )
    assert first.to_ref().object_type == "factory-dataset-run-request"

    payload = first.model_dump(mode="python")
    payload["raw_root"] = "/private/raw"
    with pytest.raises(ValidationError):
        FactoryDatasetRunRequestV2.model_validate(payload)

    payload.pop("raw_root")
    payload["object_sha256"] = "b" * 64
    with pytest.raises(ValidationError, match="identity"):
        FactoryDatasetRunRequestV2.model_validate(payload)

    with pytest.raises(ValidationError, match="sensitive"):
        FactoryDatasetRunRequestV2.create(
            dataset_run_id="factory-run://dataset/unsafe",
            requirement_spec_ref=_ref("evaluation-requirement-spec"),
            manifest_ref=_ref("trace-manifest"),
            source_authorization_ref=_ref("credential"),
            factory_policy_ref=_ref("factory-run-policy"),
            pipeline_policy_refs=(_ref("batch-quality-policy"),),
            gateway_registry_refs=(_ref("agent-registry"),),
            output_target_ref=_ref("candidate-output-target"),
            idempotency_key="unsafe",
            max_transitions=32,
            audit=_audit(),
        )


def test_item_binding_requires_contiguous_versions_and_distinct_runs() -> None:
    first = _binding()
    successor = _binding(
        version=2,
        predecessor=first.to_ref(),
    )

    assert successor.binding_version == 2
    assert successor.predecessor_binding_ref == first.to_ref()
    assert successor.to_ref().object_type == "factory-item-run-binding"

    with pytest.raises(ValidationError, match="version one"):
        _binding(version=1, predecessor=first.to_ref())
    with pytest.raises(ValidationError, match="predecessor"):
        _binding(version=2, predecessor=None)

    with pytest.raises(ValidationError, match="distinct"):
        FactoryItemRunBindingV2.create(
            dataset_run_ref=first.dataset_run_ref,
            item_run_ref=first.dataset_run_ref,
            item_id=first.item_id,
            binding_version=1,
            predecessor_binding_ref=None,
            trace_candidate_decision_ref=(first.trace_candidate_decision_ref),
            extracted_prompt_ref=first.extracted_prompt_ref,
            inferred_intent_ref=first.inferred_intent_ref,
            rewrite_candidate_ref=first.rewrite_candidate_ref,
            audit=_audit(),
        )


def test_planning_authority_binds_route_and_contiguous_version() -> None:
    first = FactoryDatasetPlanningAuthorityV2.create(
        dataset_run_ref=_ref("factory-run", "dataset"),
        planning_version=1,
        predecessor_authority_ref=None,
        plan_ref=_ref("dataset-build-plan"),
        route_decision_ref=_ref("model-route-decision"),
        invocation_result_ref=_ref("gateway-invocation-result"),
        audit=_audit(),
    )
    successor = FactoryDatasetPlanningAuthorityV2.create(
        dataset_run_ref=_ref("factory-run", "dataset-v2"),
        planning_version=2,
        predecessor_authority_ref=first.to_ref(),
        plan_ref=_ref("dataset-build-plan", "v2"),
        route_decision_ref=_ref(
            "model-route-decision",
            "v2",
        ),
        invocation_result_ref=_ref(
            "gateway-invocation-result",
            "v2",
        ),
        audit=_audit(),
    )
    assert successor.predecessor_authority_ref == first.to_ref()

    with pytest.raises(ValidationError, match="predecessor"):
        FactoryDatasetPlanningAuthorityV2.create(
            dataset_run_ref=_ref("factory-run", "dataset"),
            planning_version=2,
            predecessor_authority_ref=None,
            plan_ref=_ref("dataset-build-plan"),
            route_decision_ref=_ref("model-route-decision"),
            invocation_result_ref=_ref("gateway-invocation-result"),
            audit=_audit(),
        )


def test_item_stage_head_binds_stage_result_and_successor_chain() -> None:
    binding = _binding()
    first = _stage_head(binding)
    successor = _stage_head(
        binding,
        version=2,
        predecessor=first.to_ref(),
    )

    assert successor.predecessor_head_ref == first.to_ref()
    assert successor.to_ref().object_type == "factory-item-stage-head"

    with pytest.raises(ValidationError, match="requires criteria-rubric-result"):
        _stage_head(
            binding,
            stage=FactoryItemStageV2.CRITERIA_RUBRIC,
            result_type="attachment-subgraph-result",
        )
    with pytest.raises(ValidationError, match="successful"):
        _stage_head(
            binding,
            outcome=FactoryItemStageOutcomeV2.SUCCEEDED,
            reasons=("UNEXPECTED",),
        )
    with pytest.raises(ValidationError, match="non-success"):
        _stage_head(
            binding,
            outcome=FactoryItemStageOutcomeV2.BLOCKED_POLICY,
            reasons=(),
        )


def test_aggregate_result_requires_exact_item_partition_and_batch_truth() -> None:
    one = _binding(suffix="one")
    two = _binding(suffix="two")
    three = _binding(suffix="three")
    aggregate = FactoryDatasetAggregateResultV2.create(
        dataset_run_ref=_ref("factory-run", "dataset"),
        core_vertical_result_ref=_ref("core-vertical-result"),
        item_binding_refs=(
            one.to_ref(),
            two.to_ref(),
            three.to_ref(),
        ),
        candidate_binding_refs=(one.to_ref(), two.to_ref()),
        rejected_binding_refs=(three.to_ref(),),
        blocked_binding_refs=(),
        batch_quality_ref=_ref("batch-quality-report"),
        outcome=CandidateDatasetOutcomeV2.PARTIAL,
        reason_codes=("REJECTED_ITEMS_PRESENT",),
        audit=_audit(),
    )
    assert aggregate.item_count == 3
    assert aggregate.candidate_count == 2

    with pytest.raises(ValidationError, match="exact partition"):
        FactoryDatasetAggregateResultV2.create(
            dataset_run_ref=aggregate.dataset_run_ref,
            core_vertical_result_ref=aggregate.core_vertical_result_ref,
            item_binding_refs=aggregate.item_binding_refs,
            candidate_binding_refs=(one.to_ref(),),
            rejected_binding_refs=(two.to_ref(),),
            blocked_binding_refs=(),
            batch_quality_ref=aggregate.batch_quality_ref,
            outcome=CandidateDatasetOutcomeV2.PARTIAL,
            reason_codes=("MISSING_ITEM",),
            audit=_audit(),
        )

    with pytest.raises(ValidationError, match="batch quality"):
        FactoryDatasetAggregateResultV2.create(
            dataset_run_ref=aggregate.dataset_run_ref,
            core_vertical_result_ref=aggregate.core_vertical_result_ref,
            item_binding_refs=(one.to_ref(),),
            candidate_binding_refs=(one.to_ref(),),
            rejected_binding_refs=(),
            blocked_binding_refs=(),
            batch_quality_ref=None,
            outcome=CandidateDatasetOutcomeV2.COMPLETE,
            reason_codes=(),
            audit=_audit(),
        )


def test_delivery_manifest_binds_evaluation_items_and_never_production() -> None:
    manifest = CandidateDatasetDeliveryManifestV2.create(
        dataset_run_ref=_ref("factory-run", "dataset"),
        aggregate_result_ref=_ref("factory-dataset-aggregate-result"),
        batch_quality_ref=_ref("batch-quality-report"),
        candidate_item_refs=(
            _ref("evaluation-item", "one"),
            _ref("evaluation-item", "two"),
        ),
        candidate_projection_refs=(
            _ref("release-projection-result", "one"),
            _ref("release-projection-result", "two"),
        ),
        candidate_stage_head_refs=(
            _ref("factory-item-stage-head", "one"),
            _ref("factory-item-stage-head", "two"),
        ),
        rejected_binding_refs=(_ref("factory-item-run-binding", "three"),),
        blocked_binding_refs=(),
        inventory_ref=_ref("candidate-dataset-inventory"),
        provenance_manifest_refs=(
            _ref("provenance-manifest", "one"),
            _ref("provenance-manifest", "two"),
        ),
        audit=_audit(),
    )
    assert manifest.item_count == 2
    assert manifest.to_ref().object_type == ("candidate-dataset-delivery-manifest")
    rendered = manifest.model_dump_json().casefold()
    assert "production-readiness-attestation" not in rendered
    assert "production-registry" not in rendered

    with pytest.raises(ValidationError, match="exact candidate coverage"):
        CandidateDatasetDeliveryManifestV2.create(
            dataset_run_ref=manifest.dataset_run_ref,
            aggregate_result_ref=manifest.aggregate_result_ref,
            batch_quality_ref=manifest.batch_quality_ref,
            candidate_item_refs=manifest.candidate_item_refs,
            candidate_projection_refs=manifest.candidate_projection_refs[:1],
            candidate_stage_head_refs=(manifest.candidate_stage_head_refs),
            rejected_binding_refs=manifest.rejected_binding_refs,
            blocked_binding_refs=manifest.blocked_binding_refs,
            inventory_ref=manifest.inventory_ref,
            provenance_manifest_refs=manifest.provenance_manifest_refs,
            audit=_audit(),
        )


def test_delivery_plan_is_strict_reviewable_and_never_production() -> None:
    plan = DatasetDeliveryPlanV2.create(
        plan_id="dataset-delivery-plan://example",
        run_ref=_ref("factory-run", "dataset"),
        plan_version=1,
        predecessor_plan_ref=None,
        aggregate_result_ref=_ref("factory-dataset-aggregate-result"),
        candidate_item_refs=(
            _ref("evaluation-item", "two"),
            _ref("evaluation-item", "one"),
        ),
        candidate_projection_refs=(
            _ref("release-projection-result", "two"),
            _ref("release-projection-result", "one"),
        ),
        candidate_stage_head_refs=(
            _ref("factory-item-stage-head", "two"),
            _ref("factory-item-stage-head", "one"),
        ),
        rejected_binding_refs=(_ref("factory-item-run-binding", "rejected"),),
        blocked_binding_refs=(_ref("factory-item-run-binding", "blocked"),),
        output_target_ref=_ref("candidate-output-target"),
        max_files=1_000,
        max_total_bytes=10_000_000,
        audit=_audit(),
    )
    replay = DatasetDeliveryPlanV2.create(
        plan_id=plan.plan_id,
        run_ref=plan.run_ref,
        plan_version=plan.plan_version,
        predecessor_plan_ref=plan.predecessor_plan_ref,
        aggregate_result_ref=plan.aggregate_result_ref,
        candidate_item_refs=tuple(reversed(plan.candidate_item_refs)),
        candidate_projection_refs=tuple(reversed(plan.candidate_projection_refs)),
        candidate_stage_head_refs=tuple(reversed(plan.candidate_stage_head_refs)),
        rejected_binding_refs=plan.rejected_binding_refs,
        blocked_binding_refs=plan.blocked_binding_refs,
        output_target_ref=plan.output_target_ref,
        max_files=plan.max_files,
        max_total_bytes=plan.max_total_bytes,
        audit=_audit("delivery-replay", day=8),
    )
    compiled = CompiledDatasetDeliveryPlanV2.create(
        compiled_plan_id=(f"compiled-dataset-delivery-plan://{plan.object_sha256}"),
        source_plan_ref=plan.to_ref(),
        policy_ref=_ref("factory-run-policy"),
        aggregate_result_ref=plan.aggregate_result_ref,
        output_target_ref=plan.output_target_ref,
        candidate_count=len(plan.candidate_item_refs),
        audit=_audit(),
    )

    assert replay.to_ref() == plan.to_ref()
    assert plan.production_release_allowed is False
    assert compiled.production_release_allowed is False
    assert compiled.source_plan_ref == plan.to_ref()

    unknown = plan.model_dump(mode="python")
    unknown["production_registry"] = "forbidden"
    with pytest.raises(ValidationError):
        DatasetDeliveryPlanV2.model_validate(unknown)

    production = plan.model_dump(mode="python")
    production["production_release_allowed"] = True
    with pytest.raises(ValidationError):
        DatasetDeliveryPlanV2.model_validate(production)

    with pytest.raises(ValidationError, match="inventories differ"):
        DatasetDeliveryPlanV2.create(
            plan_id=plan.plan_id,
            run_ref=plan.run_ref,
            plan_version=plan.plan_version,
            predecessor_plan_ref=plan.predecessor_plan_ref,
            aggregate_result_ref=plan.aggregate_result_ref,
            candidate_item_refs=plan.candidate_item_refs,
            candidate_projection_refs=plan.candidate_projection_refs[:1],
            candidate_stage_head_refs=plan.candidate_stage_head_refs,
            rejected_binding_refs=plan.rejected_binding_refs,
            blocked_binding_refs=plan.blocked_binding_refs,
            output_target_ref=plan.output_target_ref,
            max_files=plan.max_files,
            max_total_bytes=plan.max_total_bytes,
            audit=_audit(),
        )

    with pytest.raises(ValidationError, match="overlap"):
        DatasetDeliveryPlanV2.create(
            plan_id=plan.plan_id,
            run_ref=plan.run_ref,
            plan_version=plan.plan_version,
            predecessor_plan_ref=plan.predecessor_plan_ref,
            aggregate_result_ref=plan.aggregate_result_ref,
            candidate_item_refs=plan.candidate_item_refs,
            candidate_projection_refs=plan.candidate_projection_refs,
            candidate_stage_head_refs=plan.candidate_stage_head_refs,
            rejected_binding_refs=plan.rejected_binding_refs,
            blocked_binding_refs=plan.rejected_binding_refs,
            output_target_ref=plan.output_target_ref,
            max_files=plan.max_files,
            max_total_bytes=plan.max_total_bytes,
            audit=_audit(),
        )

    with pytest.raises(ValidationError, match="source_plan_ref"):
        CompiledDatasetDeliveryPlanV2.create(
            compiled_plan_id=compiled.compiled_plan_id,
            source_plan_ref=_ref("dataset-build-plan"),
            policy_ref=compiled.policy_ref,
            aggregate_result_ref=compiled.aggregate_result_ref,
            output_target_ref=compiled.output_target_ref,
            candidate_count=compiled.candidate_count,
            audit=_audit(),
        )


def test_runtime_view_has_closed_status_counts_and_next_action() -> None:
    request = _request()
    bindings = (
        _binding(suffix="one").to_ref(),
        _binding(suffix="two").to_ref(),
    )
    view = FactoryDatasetRunViewV2.create(
        request_ref=request.to_ref(),
        dataset_run_ref=_ref("factory-run", "dataset"),
        status=FactoryRunStatusV2.WAITING_REVIEW,
        pending_review_refs=(_ref("plan-review-request"),),
        item_binding_refs=bindings,
        candidate_count=1,
        rejected_count=0,
        blocked_count=0,
        incomplete_count=1,
        aggregate_result_ref=None,
        delivery_manifest_ref=None,
        next_action=FactoryDatasetNextActionV2.REVIEW_PLAN,
        audit=_audit(),
    )
    assert view.next_action is FactoryDatasetNextActionV2.REVIEW_PLAN

    with pytest.raises(ValidationError, match="pending review"):
        FactoryDatasetRunViewV2.create(
            request_ref=request.to_ref(),
            dataset_run_ref=view.dataset_run_ref,
            status=FactoryRunStatusV2.WAITING_REVIEW,
            pending_review_refs=(),
            item_binding_refs=bindings,
            candidate_count=0,
            rejected_count=0,
            blocked_count=0,
            incomplete_count=2,
            aggregate_result_ref=None,
            delivery_manifest_ref=None,
            next_action=FactoryDatasetNextActionV2.REVIEW_PLAN,
            audit=_audit(),
        )
