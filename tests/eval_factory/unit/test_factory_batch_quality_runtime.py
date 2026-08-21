from __future__ import annotations

from pathlib import Path

import pytest
from test_batch_quality_report import (
    _cross_item_policy,
    _duplicate_policy,
    _FingerprintFacade,
    _graph,
    _policy,
    _ScanFacade,
    _source,
)
from test_factory_dataset_store import (
    _audit,
    _binding,
    _create_run,
    _ref,
)

from eval_factory.agent_system.batch_quality_material import (
    FactoryBatchQualityMaterialError,
    FactoryBatchQualityMaterialStore,
    FactoryBatchTerminalPartitionResult,
)
from eval_factory.agent_system.batch_quality_runtime import (
    FactoryBatchQualityContext,
    FactoryBatchQualityRuntime,
)
from eval_factory.agent_system.private_store import (
    FactoryPrivateObjectStore,
)
from eval_factory.agent_system.store import FactoryControlStore
from eval_factory.contracts.dataset_runtime_v2 import (
    CandidateDatasetOutcomeV2,
    FactoryItemStageHeadV2,
    FactoryItemStageOutcomeV2,
    FactoryItemStageV2,
)
from eval_factory.contracts.task_v2 import (
    PromptLeakageCategoryV2,
)


def _prepare_bindings(
    store: FactoryControlStore,
    *,
    dataset_run_ref,
    sources,
) -> None:
    for source in sources:
        child = _create_run(
            store,
            f"factory-run://dataset/{source.item_id}",
        )
        binding = _binding(
            dataset_run_ref,
            child,
            item_id=source.item_id,
        )
        store.commit_item_binding(
            binding,
            idempotency_key=(f"bind-batch-quality-{source.item_id}"),
        )
        grading = FactoryItemStageHeadV2.create(
            item_binding_ref=binding.to_ref(),
            item_run_ref=binding.item_run_ref,
            stage=FactoryItemStageV2.GRADING_DESIGN,
            stage_version=1,
            predecessor_head_ref=None,
            dependency_result_refs=(
                _ref(
                    "criteria-rubric-result",
                    source.item_id,
                ),
            ),
            result_ref=_ref(
                "grading-design-result",
                source.item_id,
            ),
            outcome=FactoryItemStageOutcomeV2.SUCCEEDED,
            reason_codes=(),
            audit=_audit(),
        )
        store.commit_item_stage_head(
            grading,
            idempotency_key=(f"commit-batch-grading-{source.item_id}"),
        )


@pytest.mark.asyncio
async def test_batch_quality_runtime_compiles_parent_authority_and_replays(
    tmp_path: Path,
) -> None:
    graph, item_ids = _graph(2)
    sources = tuple(
        _source(
            item_id=item_id,
            source_trace_ref=source_ref,
            suffix=f"factory-batch-{index}",
            prompt=(
                "Inspect the alpha workspace and summarize the visible architecture."
                if index == 0
                else "Analyze the beta workspace and report the visible dependency structure."
            ),
            category=(
                PromptLeakageCategoryV2.FINAL_ANSWER
                if index == 0
                else PromptLeakageCategoryV2.PRIVATE_REFERENCE
            ),
            restricted_text=(
                "alpha answer one two three four five six seven eight"
                if index == 0
                else "beta private reference nine ten eleven twelve thirteen fourteen"
            ),
        )
        for index, (item_id, source_ref) in enumerate(
            zip(
                item_ids,
                graph.source_trace_refs,
                strict=True,
            )
        )
    )
    store = FactoryControlStore(tmp_path / "factory.sqlite3")
    dataset_ref = _create_run(
        store,
        "factory-run://dataset/batch-quality",
    )
    _prepare_bindings(
        store,
        dataset_run_ref=dataset_ref,
        sources=sources,
    )
    duplicate_facade = _FingerprintFacade()
    cross_facade = _ScanFacade()
    private_store = FactoryPrivateObjectStore(tmp_path / "private")
    materials = FactoryBatchQualityMaterialStore(private_store)
    runtime = FactoryBatchQualityRuntime(
        store=store,
        materials=materials,
    )
    context = FactoryBatchQualityContext(
        resolved_job_work_graph=graph,
        duplicate_policy=_duplicate_policy(),
        duplicate_facade=duplicate_facade,
        cross_item_policy=_cross_item_policy(),
        cross_item_facade=cross_facade,
        batch_policy=_policy(),
    )
    core_ref = _ref(
        "core-vertical-result",
        "factory-batch",
    )

    first = await runtime.advance(
        dataset_run_id="factory-run://dataset/batch-quality",
        core_vertical_result_ref=core_ref,
        sources=sources,
        context=context,
        audit=_audit(),
    )
    replay = await runtime.advance(
        dataset_run_id="factory-run://dataset/batch-quality",
        core_vertical_result_ref=core_ref,
        sources=sources,
        context=context,
        audit=_audit(),
    )

    assert replay == first
    assert first.result.report.approvable is True
    assert first.aggregate.outcome is CandidateDatasetOutcomeV2.COMPLETE
    assert len(first.aggregate.candidate_binding_refs) == 2
    assert first.aggregate.rejected_binding_refs == ()
    assert first.aggregate.blocked_binding_refs == ()
    assert duplicate_facade.calls == []
    assert cross_facade.calls == []
    assert store.get_dataset_aggregate("factory-run://dataset/batch-quality") == first.aggregate
    assert (
        store.get_dataset_aggregate_material_ref("factory-run://dataset/batch-quality") == first.material_ref
    )
    assert materials.get(first.material_ref) == first.result
    with pytest.raises(
        FactoryBatchQualityMaterialError,
        match="reference type",
    ):
        materials.get(first.material_ref.model_copy(update={"object_type": "wrong-material"}))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("partition", "outcome"),
    (
        (
            "rejected",
            CandidateDatasetOutcomeV2.NO_ELIGIBLE_ITEMS,
        ),
        (
            "blocked",
            CandidateDatasetOutcomeV2.BLOCKED,
        ),
    ),
)
async def test_batch_quality_runtime_persists_zero_candidate_terminal_partition(
    tmp_path: Path,
    partition: str,
    outcome: CandidateDatasetOutcomeV2,
) -> None:
    graph, item_ids = _graph(2)
    store = FactoryControlStore(tmp_path / "factory.sqlite3")
    dataset_ref = _create_run(
        store,
        "factory-run://dataset/zero-candidate",
    )
    bindings = []
    for item_id in item_ids:
        child = _create_run(
            store,
            f"factory-run://dataset/{item_id}",
        )
        binding = _binding(
            dataset_ref,
            child,
            item_id=item_id,
        )
        bindings.append(
            store.commit_item_binding(
                binding,
                idempotency_key=(f"bind-zero-candidate-{item_id}"),
            )
        )
    binding_refs = tuple(binding.to_ref() for binding in bindings)
    duplicate_facade = _FingerprintFacade()
    cross_facade = _ScanFacade()
    materials = FactoryBatchQualityMaterialStore(FactoryPrivateObjectStore(tmp_path / "private"))
    runtime = FactoryBatchQualityRuntime(
        store=store,
        materials=materials,
    )
    context = FactoryBatchQualityContext(
        resolved_job_work_graph=graph,
        duplicate_policy=_duplicate_policy(),
        duplicate_facade=duplicate_facade,
        cross_item_policy=_cross_item_policy(),
        cross_item_facade=cross_facade,
        batch_policy=_policy(),
    )
    kwargs = {
        "rejected_binding_refs": (binding_refs if partition == "rejected" else ()),
        "blocked_binding_refs": (binding_refs if partition == "blocked" else ()),
    }

    first = await runtime.advance(
        dataset_run_id=("factory-run://dataset/zero-candidate"),
        core_vertical_result_ref=_ref(
            "core-vertical-result",
            "zero-candidate",
        ),
        sources=(),
        context=context,
        audit=_audit(),
        **kwargs,
    )
    replay = await runtime.advance(
        dataset_run_id=("factory-run://dataset/zero-candidate"),
        core_vertical_result_ref=_ref(
            "core-vertical-result",
            "zero-candidate",
        ),
        sources=(),
        context=context,
        audit=_audit(),
        **kwargs,
    )

    assert replay == first
    assert first.aggregate.outcome is outcome
    assert first.aggregate.candidate_binding_refs == ()
    assert first.aggregate.batch_quality_ref is None
    assert isinstance(
        first.result,
        FactoryBatchTerminalPartitionResult,
    )
    assert duplicate_facade.calls == []
    assert cross_facade.calls == []


@pytest.mark.asyncio
async def test_batch_quality_runtime_blocks_single_candidate_without_cross_item_work(
    tmp_path: Path,
) -> None:
    graph, item_ids = _graph(1)
    source = _source(
        item_id=item_ids[0],
        source_trace_ref=graph.source_trace_refs[0],
        suffix="factory-single-candidate",
        prompt="Inspect the only candidate workspace.",
        category=PromptLeakageCategoryV2.FINAL_ANSWER,
        restricted_text="private single candidate answer material",
    )
    store = FactoryControlStore(tmp_path / "factory.sqlite3")
    dataset_ref = _create_run(
        store,
        "factory-run://dataset/single-candidate",
    )
    _prepare_bindings(
        store,
        dataset_run_ref=dataset_ref,
        sources=(source,),
    )
    duplicate_facade = _FingerprintFacade()
    cross_facade = _ScanFacade()
    materials = FactoryBatchQualityMaterialStore(
        FactoryPrivateObjectStore(tmp_path / "private"),
    )
    runtime = FactoryBatchQualityRuntime(
        store=store,
        materials=materials,
    )
    context = FactoryBatchQualityContext(
        resolved_job_work_graph=graph,
        duplicate_policy=_duplicate_policy(),
        duplicate_facade=duplicate_facade,
        cross_item_policy=_cross_item_policy(),
        cross_item_facade=cross_facade,
        batch_policy=_policy(),
    )
    core_ref = _ref(
        "core-vertical-result",
        "single-candidate",
    )

    first = await runtime.advance(
        dataset_run_id="factory-run://dataset/single-candidate",
        core_vertical_result_ref=core_ref,
        sources=(source,),
        context=context,
        audit=_audit(),
    )
    replay = await runtime.advance(
        dataset_run_id="factory-run://dataset/single-candidate",
        core_vertical_result_ref=core_ref,
        sources=(source,),
        context=context,
        audit=_audit(),
    )

    assert replay == first
    assert first.aggregate.outcome is CandidateDatasetOutcomeV2.BLOCKED
    assert first.aggregate.candidate_binding_refs == ()
    assert len(first.aggregate.blocked_binding_refs) == 1
    assert first.aggregate.reason_codes == ("INSUFFICIENT_BATCH_CANDIDATES",)
    assert first.aggregate.batch_quality_ref is None
    assert isinstance(
        first.result,
        FactoryBatchTerminalPartitionResult,
    )
    assert duplicate_facade.calls == []
    assert cross_facade.calls == []
