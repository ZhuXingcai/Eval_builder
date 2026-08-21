from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from test_factory_control_store import _plan as _global_plan
from test_factory_dataset_store import (
    _audit,
    _binding,
    _create_run,
    _ref,
)
from test_release_projection import _policy
from test_release_projection_persistence import (
    _prepared_release_job,
)

from eval_factory.agent_system.candidate_output import (
    AuthorizedCandidateExport,
    CandidateDatasetOutputAssembler,
)
from eval_factory.agent_system.candidate_projection_material import (
    FactoryCandidateProjectionMaterialStore,
)
from eval_factory.agent_system.candidate_projection_runtime import (
    FactoryCandidateProjectionRuntime,
)
from eval_factory.agent_system.delivery_runtime import (
    FactoryDeliveryRuntime,
    FactoryDeliveryRuntimeConfig,
    FactoryDeliveryWaitingView,
)
from eval_factory.agent_system.plan_review import PlanReviewService
from eval_factory.agent_system.private_store import (
    FactoryPrivateObjectStore,
)
from eval_factory.agent_system.release_runtime import (
    FactoryDatasetReleaseContext,
    FactoryDatasetReleaseRuntime,
)
from eval_factory.agent_system.release_source_builder import (
    FactoryReleaseSourceBuilder,
    FactoryReleaseSourceBuilderError,
    FactoryReleaseSourceItem,
)
from eval_factory.agent_system.store import FactoryControlStore
from eval_factory.contracts.approval import (
    ApprovalCheckpoint,
    ApprovalMode,
    FinalReviewScope,
)
from eval_factory.contracts.batch_quality_v2 import (
    batch_quality_report_v2_ref,
)
from eval_factory.contracts.dataset_runtime_v2 import (
    CandidateDatasetOutcomeV2,
    FactoryDatasetAggregateResultV2,
    FactoryItemStageHeadV2,
    FactoryItemStageOutcomeV2,
    FactoryItemStageV2,
)
from eval_factory.contracts.quality_v2 import (
    item_quality_compilation_result_ref,
)
from eval_factory.contracts.task_v2 import (
    r4_task_contract_set_ref,
)
from eval_factory.orchestration.job_store import JobStore


def _stage(
    *,
    binding,
    stage: FactoryItemStageV2,
    result_ref,
    dependency_refs,
) -> FactoryItemStageHeadV2:
    return FactoryItemStageHeadV2.create(
        item_binding_ref=binding.to_ref(),
        item_run_ref=binding.item_run_ref,
        stage=stage,
        stage_version=1,
        predecessor_head_ref=None,
        dependency_result_refs=dependency_refs,
        result_ref=result_ref,
        outcome=FactoryItemStageOutcomeV2.SUCCEEDED,
        reason_codes=(),
        audit=_audit(),
    )


async def _authority(tmp_path: Path):
    job_store, sources, *_ = await _prepared_release_job(
        tmp_path / "release-job",
    )
    store = FactoryControlStore(tmp_path / "factory.sqlite3")
    dataset_ref = _create_run(
        store,
        "factory-run://dataset/release-source",
    )
    parent = store.get_run("factory-run://dataset/release-source")
    global_plan, global_compiled = _global_plan(
        run_ref=parent.to_ref(),
        policy_ref=parent.policy_ref,
    )
    store.commit_plan(
        run_id=parent.run_id,
        expected_run_version=parent.run_version,
        plan=global_plan,
        compiled_plan=global_compiled,
        audit=_audit(),
        idempotency_key="commit-release-source-global-plan",
    )
    bindings = []
    items = []
    for source in sources:
        child = _create_run(
            store,
            f"factory-run://dataset/{source.item.item_id}",
        )
        binding = store.commit_item_binding(
            _binding(
                dataset_ref,
                child,
                item_id=source.item.item_id,
            ),
            idempotency_key=(f"bind-release-source-{source.item.item_id}"),
        )
        task_head = _stage(
            binding=binding,
            stage=FactoryItemStageV2.TASK_AUTHORING,
            result_ref=r4_task_contract_set_ref(
                source.task_contract_set,
            ),
            dependency_refs=(binding.rewrite_candidate_ref,),
        )
        store.commit_item_stage_head(
            task_head,
            idempotency_key=(f"release-source-task-{source.item.item_id}"),
        )
        quality_head = _stage(
            binding=binding,
            stage=FactoryItemStageV2.ITEM_QUALITY,
            result_ref=item_quality_compilation_result_ref(
                source.item_quality,
            ),
            dependency_refs=(
                _ref(
                    "attachment-subgraph-result",
                    source.item.item_id,
                ),
            ),
        )
        store.commit_item_stage_head(
            quality_head,
            idempotency_key=(f"release-source-quality-{source.item.item_id}"),
        )
        criteria_result_ref = _ref(
            "criteria-rubric-result",
            source.item.item_id,
        )
        criteria_material_ref = _ref(
            "criteria-authority-material",
            source.item.item_id,
        )
        criteria_head = _stage(
            binding=binding,
            stage=FactoryItemStageV2.CRITERIA_RUBRIC,
            result_ref=criteria_result_ref,
            dependency_refs=(quality_head.result_ref,),
        )
        store.commit_item_stage_head(
            criteria_head,
            idempotency_key=(f"release-source-criteria-{source.item.item_id}"),
            material_ref=criteria_material_ref,
        )
        grading_result_ref = _ref(
            "grading-design-result",
            source.item.item_id,
        )
        grading_head = _stage(
            binding=binding,
            stage=FactoryItemStageV2.GRADING_DESIGN,
            result_ref=grading_result_ref,
            dependency_refs=(criteria_result_ref,),
        )
        store.commit_item_stage_head(
            grading_head,
            idempotency_key=(f"release-source-grading-{source.item.item_id}"),
        )
        bindings.append(binding)
        items.append(
            FactoryReleaseSourceItem(
                binding=binding,
                source_trace_ref=source.source_trace_refs[0],
                label_decisions=source.label_decisions,
                task_draft=source.task_draft,
                task_prompt_safety_gate=(source.task_prompt_safety_gate),
                base_task_contract_set=(source.task_contract_set),
                task_contract_set=source.task_contract_set,
                attachment_result=source.attachment_result,
                attachment_item_quality_ref=(item_quality_compilation_result_ref(source.item_quality)),
                item_quality=source.item_quality,
                criteria_result_ref=criteria_result_ref,
                criteria_material_ref=criteria_material_ref,
                grading_result_ref=grading_result_ref,
            )
        )
    batch_ref = batch_quality_report_v2_ref(
        sources[0].batch_quality,
    )
    aggregate = FactoryDatasetAggregateResultV2.create(
        dataset_run_ref=dataset_ref,
        core_vertical_result_ref=_ref(
            "core-vertical-result",
            "release-source",
        ),
        item_binding_refs=tuple(binding.to_ref() for binding in bindings),
        candidate_binding_refs=tuple(binding.to_ref() for binding in bindings),
        rejected_binding_refs=(),
        blocked_binding_refs=(),
        batch_quality_ref=batch_ref,
        outcome=CandidateDatasetOutcomeV2.COMPLETE,
        reason_codes=(),
        audit=_audit(),
    )
    aggregate = store.commit_dataset_aggregate(
        aggregate,
        material_ref=_ref(
            "batch-quality-material",
            "release-source",
        ),
        idempotency_key="commit-release-source-aggregate",
    )
    return (
        store,
        job_store,
        sources,
        tuple(items),
        aggregate,
    )


@pytest.mark.asyncio
async def test_release_source_builder_loads_complete_current_authority(
    tmp_path: Path,
) -> None:
    (
        store,
        job_store,
        sources,
        items,
        aggregate,
    ) = await _authority(tmp_path)
    builder = FactoryReleaseSourceBuilder(
        store=store,
        job_store=job_store,
    )

    first = builder.build(
        dataset_run_id="factory-run://dataset/release-source",
        aggregate=aggregate,
        items=items,
        batch_quality=sources[0].batch_quality,
        approval_policy=sources[0].approval_policy,
    )
    replay = builder.build(
        dataset_run_id="factory-run://dataset/release-source",
        aggregate=aggregate,
        items=items,
        batch_quality=sources[0].batch_quality,
        approval_policy=sources[0].approval_policy,
    )
    canonical_checkpoints = builder.build(
        dataset_run_id="factory-run://dataset/release-source",
        aggregate=aggregate,
        items=items,
        batch_quality=sources[0].batch_quality,
        approval_policy=sources[0].approval_policy,
        not_required_checkpoints=(
            ApprovalCheckpoint.FINAL_DATASET_REVIEW,
            ApprovalCheckpoint.ENVIRONMENT_STRATEGY,
        ),
    )

    assert replay == first
    assert tuple(value.source for value in first) == sources
    assert tuple(value.binding.to_ref() for value in first) == aggregate.candidate_binding_refs
    assert canonical_checkpoints[0].source.not_required_checkpoints == (
        ApprovalCheckpoint.ENVIRONMENT_STRATEGY,
        ApprovalCheckpoint.FINAL_DATASET_REVIEW,
    )


@pytest.mark.asyncio
async def test_release_source_builder_rejects_stale_quality_and_policy(
    tmp_path: Path,
) -> None:
    (
        store,
        job_store,
        sources,
        items,
        aggregate,
    ) = await _authority(tmp_path)
    builder = FactoryReleaseSourceBuilder(
        store=store,
        job_store=job_store,
    )
    changed_item = replace(
        items[0],
        item_quality=items[1].item_quality,
    )

    with pytest.raises(
        FactoryReleaseSourceBuilderError,
        match="quality",
    ):
        builder.build(
            dataset_run_id="factory-run://dataset/release-source",
            aggregate=aggregate,
            items=(changed_item, items[1]),
            batch_quality=sources[0].batch_quality,
            approval_policy=sources[0].approval_policy,
        )

    final_policy_source = (
        await _prepared_release_job(
            tmp_path / "different-policy",
            mode=ApprovalMode.FINAL_ONLY,
        )
    )[1][0]
    assert final_policy_source.approval_policy.final_review_scope is FinalReviewScope.SELECTED_ITEMS
    with pytest.raises(
        FactoryReleaseSourceBuilderError,
        match="approval policy",
    ):
        builder.build(
            dataset_run_id="factory-run://dataset/release-source",
            aggregate=aggregate,
            items=items,
            batch_quality=sources[0].batch_quality,
            approval_policy=(final_policy_source.approval_policy),
        )


@pytest.mark.asyncio
async def test_release_source_builder_rejects_cross_job_store(
    tmp_path: Path,
) -> None:
    (
        store,
        _job_store,
        sources,
        items,
        aggregate,
    ) = await _authority(tmp_path)
    other_job_store = JobStore(tmp_path / "empty-job-store.sqlite3")
    builder = FactoryReleaseSourceBuilder(
        store=store,
        job_store=other_job_store,
    )

    with pytest.raises(
        FactoryReleaseSourceBuilderError,
        match="JobStore",
    ):
        builder.build(
            dataset_run_id="factory-run://dataset/release-source",
            aggregate=aggregate,
            items=items,
            batch_quality=sources[0].batch_quality,
            approval_policy=sources[0].approval_policy,
        )


@pytest.mark.asyncio
async def test_release_source_builder_rejects_aggregate_partition_and_trace_drift(
    tmp_path: Path,
) -> None:
    (
        store,
        job_store,
        sources,
        items,
        aggregate,
    ) = await _authority(tmp_path)
    builder = FactoryReleaseSourceBuilder(
        store=store,
        job_store=job_store,
    )

    with pytest.raises(
        FactoryReleaseSourceBuilderError,
        match="aggregate",
    ):
        builder.build(
            dataset_run_id="factory-run://dataset/release-source",
            aggregate=aggregate.model_copy(
                update={
                    "reason_codes": ("DRIFT",),
                }
            ),
            items=items,
            batch_quality=sources[0].batch_quality,
            approval_policy=sources[0].approval_policy,
        )

    with pytest.raises(
        FactoryReleaseSourceBuilderError,
        match="partition",
    ):
        builder.build(
            dataset_run_id="factory-run://dataset/release-source",
            aggregate=aggregate,
            items=items[:1],
            batch_quality=sources[0].batch_quality,
            approval_policy=sources[0].approval_policy,
        )

    changed_trace = replace(
        items[0],
        source_trace_ref=items[1].source_trace_ref,
    )
    with pytest.raises(
        FactoryReleaseSourceBuilderError,
        match="trace",
    ):
        builder.build(
            dataset_run_id="factory-run://dataset/release-source",
            aggregate=aggregate,
            items=(changed_trace, items[1]),
            batch_quality=sources[0].batch_quality,
            approval_policy=sources[0].approval_policy,
        )


@pytest.mark.asyncio
async def test_release_context_reaches_final_delivery_review_without_auto_approval(
    tmp_path: Path,
) -> None:
    (
        store,
        job_store,
        sources,
        items,
        aggregate,
    ) = await _authority(tmp_path)
    private_store = FactoryPrivateObjectStore(tmp_path / "private")
    candidates = FactoryCandidateProjectionRuntime(
        store=store,
        job_store=job_store,
        materials=FactoryCandidateProjectionMaterialStore(private_store),
    )
    reviews = PlanReviewService(store)
    delivery = FactoryDeliveryRuntime(
        store=store,
        plan_reviews=reviews,
        assembler=CandidateDatasetOutputAssembler(
            store=store,
            root=tmp_path / "candidate-output",
        ),
        config=FactoryDeliveryRuntimeConfig(
            output_target_ref=_ref(
                "candidate-output-target",
                "release-source",
            ),
            max_files=1_000,
            max_total_bytes=10_000_000,
        ),
        requested_by="user://release-source-owner",
    )
    context = FactoryDatasetReleaseContext(
        source_builder=FactoryReleaseSourceBuilder(
            store=store,
            job_store=job_store,
        ),
        runtime=FactoryDatasetReleaseRuntime(
            candidates=candidates,
            delivery=delivery,
        ),
        approval_policy=sources[0].approval_policy,
        release_policy=_policy(),
        exports=tuple(
            AuthorizedCandidateExport(
                item_id=source.item.item_id,
                files=(),
            )
            for source in sources
        ),
    )
    parent_policy = store.get_policy(
        store.get_run("factory-run://dataset/release-source").policy_ref.object_id
    )

    first = context.advance(
        dataset_run_id="factory-run://dataset/release-source",
        aggregate=aggregate,
        items=items,
        batch_quality=sources[0].batch_quality,
        factory_policy=parent_policy,
        audit=_audit(),
    )
    replay = context.advance(
        dataset_run_id="factory-run://dataset/release-source",
        aggregate=aggregate,
        items=items,
        batch_quality=sources[0].batch_quality,
        factory_policy=parent_policy,
        audit=_audit(),
    )

    assert replay == first
    assert len(first.candidates) == 2
    assert isinstance(
        first.delivery,
        FactoryDeliveryWaitingView,
    )
    review = reviews.show(first.delivery.review_ref.object_id)
    assert review.request.plan_kind.value == "FINAL_DELIVERY"
    assert (
        tuple(value.projection.evaluation_item_ref for value in first.candidates)
        == review.plan.candidate_item_refs
    )
    assert not (tmp_path / "candidate-output" / "bundles").exists()
