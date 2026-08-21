from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import ValidationError
from test_factory_control_store import _plan as _global_plan
from test_factory_dataset_store import (
    _audit as factory_audit,
)
from test_factory_dataset_store import (
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
    CandidateDatasetInventoryV1,
    CandidateDatasetOutputAssembler,
    CandidateOutputConflictError,
    CandidateOutputError,
    CandidateOutputFileV1,
    CandidateOutputIntegrityError,
    CandidateOutputWrite,
    _inventory,
    _require_reviewed_limits,
)
from eval_factory.agent_system.candidate_projection_material import (
    FactoryCandidateProjectionMaterialError,
    FactoryCandidateProjectionMaterialStore,
)
from eval_factory.agent_system.candidate_projection_runtime import (
    FactoryCandidateProjectionInput,
    FactoryCandidateProjectionRuntime,
)
from eval_factory.agent_system.delivery_runtime import (
    FactoryDeliveryRuntime,
    FactoryDeliveryRuntimeConfig,
    FactoryDeliveryWaitingView,
)
from eval_factory.agent_system.plan_review import (
    PlanReviewDecisionSubmissionV1,
    PlanReviewResumeSubmissionV1,
    PlanReviewService,
)
from eval_factory.agent_system.private_store import (
    FactoryPrivateObjectStore,
)
from eval_factory.agent_system.release_runtime import (
    FactoryDatasetReleaseRuntime,
)
from eval_factory.agent_system.store import (
    FactoryControlNotFoundError,
    FactoryControlStore,
)
from eval_factory.contracts.agent_system_v2 import (
    FactoryRunStatusV2,
    PlanDecisionKindV2,
    PlanKindV2,
)
from eval_factory.contracts.batch_quality_v2 import (
    batch_quality_report_v2_ref,
)
from eval_factory.contracts.core import ObjectRef
from eval_factory.contracts.dataset_runtime_v2 import (
    CandidateDatasetDeliveryManifestV2,
    CandidateDatasetOutcomeV2,
    CompiledDatasetDeliveryPlanV2,
    DatasetDeliveryPlanV2,
    FactoryDatasetAggregateResultV2,
    FactoryItemStageHeadV2,
    FactoryItemStageOutcomeV2,
    FactoryItemStageV2,
)
from eval_factory.contracts.quality_v2 import (
    item_quality_compilation_result_ref,
)
from eval_factory.contracts.release_v2 import ReleaseStateV2
from eval_factory.contracts.task_v2 import (
    r4_task_contract_set_ref,
)


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
        audit=factory_audit(),
    )


@pytest.mark.asyncio
async def test_candidate_projection_persists_jobstore_and_item_heads(
    tmp_path: Path,
) -> None:
    job_store, sources, *_ = await _prepared_release_job(tmp_path / "release-job")
    store = FactoryControlStore(tmp_path / "factory.sqlite3")
    dataset_ref = _create_run(
        store,
        "factory-run://dataset/release-candidate",
    )
    created_parent = store.get_run("factory-run://dataset/release-candidate")
    global_plan, global_compiled = _global_plan(
        run_ref=created_parent.to_ref(),
        policy_ref=created_parent.policy_ref,
    )
    store.commit_plan(
        run_id=created_parent.run_id,
        expected_run_version=created_parent.run_version,
        plan=global_plan,
        compiled_plan=global_compiled,
        audit=factory_audit(),
        idempotency_key="commit-release-global-plan",
    )
    bindings = []
    input_authority: dict[
        str,
        tuple[ObjectRef, ObjectRef, ObjectRef, ObjectRef, ObjectRef],
    ] = {}
    for source in sources:
        child = _create_run(
            store,
            f"factory-run://dataset/{source.item.item_id}",
        )
        binding = _binding(
            dataset_ref,
            child,
            item_id=source.item.item_id,
        )
        binding = store.commit_item_binding(
            binding,
            idempotency_key=(f"bind-release-{source.item.item_id}"),
        )
        task_head = _stage(
            binding=binding,
            stage=FactoryItemStageV2.TASK_AUTHORING,
            result_ref=r4_task_contract_set_ref(source.task_contract_set),
            dependency_refs=(binding.rewrite_candidate_ref,),
        )
        store.commit_item_stage_head(
            task_head,
            idempotency_key=(f"release-task-head-{source.item.item_id}"),
        )
        quality_head = _stage(
            binding=binding,
            stage=FactoryItemStageV2.ITEM_QUALITY,
            result_ref=item_quality_compilation_result_ref(source.item_quality),
            dependency_refs=(
                _ref(
                    "attachment-subgraph-result",
                    source.item.item_id,
                ),
            ),
        )
        store.commit_item_stage_head(
            quality_head,
            idempotency_key=(f"release-quality-head-{source.item.item_id}"),
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
            idempotency_key=(f"release-criteria-head-{source.item.item_id}"),
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
            idempotency_key=(f"release-grading-head-{source.item.item_id}"),
        )
        bindings.append(binding)
        input_authority[binding.item_id] = (
            task_head.result_ref,
            quality_head.result_ref,
            criteria_result_ref,
            criteria_material_ref,
            grading_result_ref,
        )
    rejected_child = _create_run(
        store,
        "factory-run://dataset/item-rejected",
    )
    rejected_binding = _binding(
        dataset_ref,
        rejected_child,
        item_id="item://factory-dataset/rejected",
    )
    rejected_binding = store.commit_item_binding(
        rejected_binding,
        idempotency_key="bind-release-item-rejected",
    )
    batch_ref = batch_quality_report_v2_ref(sources[0].batch_quality)
    aggregate = FactoryDatasetAggregateResultV2.create(
        dataset_run_ref=dataset_ref,
        core_vertical_result_ref=_ref(
            "core-vertical-result",
            "release-candidate",
        ),
        item_binding_refs=(
            *(value.to_ref() for value in bindings),
            rejected_binding.to_ref(),
        ),
        candidate_binding_refs=tuple(value.to_ref() for value in bindings),
        rejected_binding_refs=(rejected_binding.to_ref(),),
        blocked_binding_refs=(),
        batch_quality_ref=batch_ref,
        outcome=CandidateDatasetOutcomeV2.PARTIAL,
        reason_codes=("PARTIAL_CANDIDATE_SET",),
        audit=factory_audit(),
    )
    aggregate = store.commit_dataset_aggregate(
        aggregate,
        material_ref=_ref(
            "batch-quality-material",
            "release-candidate",
        ),
        idempotency_key="commit-release-aggregate",
    )
    private_store = FactoryPrivateObjectStore(tmp_path / "private")
    materials = FactoryCandidateProjectionMaterialStore(private_store)
    runtime = FactoryCandidateProjectionRuntime(
        store=store,
        job_store=job_store,
        materials=materials,
    )
    inputs = tuple(
        FactoryCandidateProjectionInput(
            binding=binding,
            source=source,
            task_authoring_result_ref=(input_authority[binding.item_id][0]),
            attachment_item_quality_ref=(input_authority[binding.item_id][1]),
            criteria_result_ref=(input_authority[binding.item_id][2]),
            criteria_material_ref=(input_authority[binding.item_id][3]),
            grading_result_ref=(input_authority[binding.item_id][4]),
        )
        for binding, source in zip(
            bindings,
            sources,
            strict=True,
        )
    )

    first = runtime.advance(
        dataset_run_id=("factory-run://dataset/release-candidate"),
        aggregate=aggregate,
        inputs=inputs,
        policy=_policy(),
        audit=factory_audit(),
    )
    replay = runtime.advance(
        dataset_run_id=("factory-run://dataset/release-candidate"),
        aggregate=aggregate,
        inputs=inputs,
        policy=_policy(),
        audit=factory_audit(),
    )

    assert replay == first
    assert len(first) == 2
    assert aggregate.outcome is CandidateDatasetOutcomeV2.PARTIAL
    assert aggregate.rejected_binding_refs == (rejected_binding.to_ref(),)
    assert all(value.projection.item_projection.release_state is ReleaseStateV2.CANDIDATE for value in first)
    for value in first:
        head = store.get_item_stage_head(
            value.item_id,
            FactoryItemStageV2.RELEASE_CANDIDATE,
        )
        assert head.to_ref() == value.stage_head_ref
        assert store.get_item_stage_material_ref(head.to_ref()) == value.material_ref
        restored = materials.get(value.material_ref)
        assert restored.projection == value.projection
    with pytest.raises(
        FactoryControlNotFoundError,
        match="not found",
    ):
        store.get_item_stage_head(
            rejected_binding.item_id,
            FactoryItemStageV2.RELEASE_CANDIDATE,
        )
    with pytest.raises(
        FactoryCandidateProjectionMaterialError,
        match="reference type",
    ):
        materials.get(first[0].material_ref.model_copy(update={"object_type": "wrong-material"}))

    candidate_results = tuple(materials.get(value.material_ref) for value in first)
    reviews = PlanReviewService(store)
    pending_faults: set[str] = set()

    def inject_output_fault(point: str) -> None:
        if point in pending_faults:
            pending_faults.remove(point)
            raise RuntimeError(f"injected candidate output fault: {point}")

    assembler = CandidateDatasetOutputAssembler(
        store=store,
        root=tmp_path / "candidate-output",
        fault_injector=inject_output_fault,
    )
    delivery = FactoryDeliveryRuntime(
        store=store,
        plan_reviews=reviews,
        assembler=assembler,
        config=FactoryDeliveryRuntimeConfig(
            output_target_ref=_ref(
                "candidate-output-target",
                "release-candidate",
            ),
            max_files=1_000,
            max_total_bytes=10_000_000,
        ),
        requested_by="user://release-candidate-owner",
    )
    release_runtime = FactoryDatasetReleaseRuntime(
        candidates=runtime,
        delivery=delivery,
    )
    exports = tuple(
        AuthorizedCandidateExport(
            item_id=value.source.item.item_id,
            files=(),
        )
        for value in candidate_results
    )
    parent_policy = store.get_policy(
        store.get_run("factory-run://dataset/release-candidate").policy_ref.object_id
    )
    release_waiting = release_runtime.advance(
        dataset_run_id=("factory-run://dataset/release-candidate"),
        aggregate=aggregate,
        inputs=inputs,
        release_policy=_policy(),
        exports=exports,
        factory_policy=parent_policy,
        audit=factory_audit(),
    )
    release_waiting_replay = release_runtime.advance(
        dataset_run_id=("factory-run://dataset/release-candidate"),
        aggregate=aggregate,
        inputs=inputs,
        release_policy=_policy(),
        exports=exports,
        factory_policy=parent_policy,
        audit=factory_audit(),
    )
    waiting = release_waiting.delivery
    waiting_replay = release_waiting_replay.delivery
    assert isinstance(waiting, FactoryDeliveryWaitingView)
    assert release_waiting.candidates == first
    assert waiting_replay == waiting

    reviews.decide(
        waiting.review_ref.object_id,
        PlanReviewDecisionSubmissionV1(
            expected_plan_version=1,
            decision=PlanDecisionKindV2.APPROVE,
            decided_by="user://release-candidate-owner",
            reason_code="DELIVERY_PLAN_APPROVED",
            idempotency_key="approve-release-delivery",
        ),
        audit=factory_audit(),
    )
    reviews.resume(
        waiting.review_ref.object_id,
        PlanReviewResumeSubmissionV1(
            expected_plan_version=1,
            resumed_by="user://release-candidate-owner",
            idempotency_key="resume-release-delivery",
        ),
        audit=factory_audit(),
    )
    prepared_after_resume = release_runtime.prepare_review(
        dataset_run_id=("factory-run://dataset/release-candidate"),
        aggregate=aggregate,
        inputs=inputs,
        release_policy=_policy(),
        exports=exports,
        factory_policy=parent_policy,
        audit=factory_audit(),
    )
    assert isinstance(
        prepared_after_resume.delivery,
        FactoryDeliveryWaitingView,
    )
    assert (
        store.get_run(
            "factory-run://dataset/release-candidate",
        ).status
        is not FactoryRunStatusV2.COMPLETED
    )
    for fault_point in (
        "before_rename",
        "after_rename",
        "after_bundle",
        "after_inventory",
        "after_manifest",
        "after_completion",
        "after_output_head",
        "after_run",
        "after_run_head",
        "after_outbox",
        "after_idempotency",
    ):
        pending_faults.add(fault_point)
        with pytest.raises(
            RuntimeError,
            match=f"injected candidate output fault: {fault_point}",
        ):
            delivery.advance(
                dataset_run_id=("factory-run://dataset/release-candidate"),
                aggregate=aggregate,
                candidates=candidate_results,
                exports=exports,
                policy=parent_policy,
                audit=factory_audit(),
            )
        assert not pending_faults
        assert (
            store.get_run("factory-run://dataset/release-candidate").status
            is not FactoryRunStatusV2.COMPLETED
        )
        with pytest.raises(
            CandidateOutputIntegrityError,
            match="not found",
        ):
            assembler.get("factory-run://dataset/release-candidate")
        connection = store._connect()
        try:
            for table in (
                "candidate_output_inventories",
                "candidate_delivery_manifests",
                "candidate_run_completions",
                "candidate_output_current_heads",
            ):
                count = connection.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()["count"]
                assert count == 0
        finally:
            connection.close()
        if fault_point == "before_rename":
            assert not tuple(assembler._staging.iterdir())
        else:
            assert len(tuple(assembler._cas.glob("*/*"))) == 1

    release_output = release_runtime.advance(
        dataset_run_id=("factory-run://dataset/release-candidate"),
        aggregate=aggregate,
        inputs=inputs,
        release_policy=_policy(),
        exports=exports,
        factory_policy=parent_policy,
        audit=factory_audit(),
    )
    release_output_replay = release_runtime.advance(
        dataset_run_id=("factory-run://dataset/release-candidate"),
        aggregate=aggregate,
        inputs=inputs,
        release_policy=_policy(),
        exports=exports,
        factory_policy=parent_policy,
        audit=factory_audit(),
    )
    output = release_output.delivery
    output_replay = release_output_replay.delivery

    assert isinstance(output, CandidateOutputWrite)
    assert isinstance(output_replay, CandidateOutputWrite)
    assert output.reused is True
    assert output_replay.inventory == output.inventory
    assert output_replay.bundle_path == output.bundle_path
    assert output_replay.reused is True
    assert output.bundle_path.is_dir()
    assert output.manifest.candidate_item_refs == tuple(
        sorted(
            (value.projection.evaluation_item_ref for value in candidate_results),
            key=lambda value: (
                value.object_type,
                value.object_id,
                value.object_version,
                value.object_sha256,
            ),
        )
    )
    assert output.manifest.rejected_binding_refs == (rejected_binding.to_ref(),)
    rendered_bundle = b"\n".join(
        path.read_bytes()
        for path in sorted(output.bundle_path.rglob("*"))
        if path.is_file() and not path.is_symlink()
    ).lower()
    for forbidden_marker in (
        b"completed_deliverable",
        b"credential",
        b"final_answer",
        b"grader_rule",
        b"hidden_condition",
        b"model-response-content",
        b"private_reference",
        b"raw_trace",
        b"runtime-transcript",
    ):
        assert forbidden_marker not in rendered_bundle
    completed = store.get_run("factory-run://dataset/release-candidate")
    assert completed.status is FactoryRunStatusV2.COMPLETED
    assert completed.completion_ref == output.completion.to_ref()
    assert completed.delivery_manifest_ref == output.manifest.to_ref()
    delivery_material = store.get_domain_plan(
        completed.run_id,
        PlanKindV2.FINAL_DELIVERY,
    )
    delivery_plan = DatasetDeliveryPlanV2.model_validate_json(delivery_material.plan_record_json)
    compiled_delivery_plan = CompiledDatasetDeliveryPlanV2.model_validate_json(
        delivery_material.compiled_plan_record_json
    )
    parent_compiled_plan = store.get_plan(completed.run_id)[1]
    with pytest.raises(
        CandidateOutputConflictError,
        match="replay",
    ):
        assembler.assemble(
            dataset_run_id=completed.run_id,
            aggregate=aggregate,
            plan=delivery_plan,
            compiled_plan=compiled_delivery_plan,
            parent_compiled_plan_ref=_ref(
                "compiled-dataset-build-plan",
                "changed",
            ),
            candidates=candidate_results,
            exports=exports,
            audit=factory_audit(),
            idempotency_key=(f"assemble-factory-candidate-output-{delivery_plan.object_sha256}"),
        )
    with pytest.raises(
        CandidateOutputConflictError,
        match="already exists",
    ):
        assembler.assemble(
            dataset_run_id=completed.run_id,
            aggregate=aggregate,
            plan=delivery_plan,
            compiled_plan=compiled_delivery_plan,
            parent_compiled_plan_ref=parent_compiled_plan.to_ref(),
            candidates=candidate_results,
            exports=exports,
            audit=factory_audit(),
            idempotency_key="second-candidate-output-authority",
        )
    with pytest.raises(
        CandidateOutputConflictError,
        match="duplicate items",
    ):
        assembler._payloads(
            aggregate=aggregate,
            candidates=candidate_results,
            exports=(exports[0], exports[0]),
        )
    with pytest.raises(
        CandidateOutputConflictError,
        match="inventory is not exact",
    ):
        assembler._payloads(
            aggregate=aggregate,
            candidates=candidate_results,
            exports=(
                AuthorizedCandidateExport(
                    item_id=exports[0].item_id,
                    files=(("unexpected.txt", b"unexpected"),),
                ),
                exports[1],
            ),
        )
    with pytest.raises(
        CandidateOutputConflictError,
        match="unknown items",
    ):
        assembler._payloads(
            aggregate=aggregate,
            candidates=candidate_results,
            exports=(
                *exports,
                AuthorizedCandidateExport(
                    item_id="item://unknown",
                    files=(),
                ),
            ),
        )
    payload_path = output.bundle_path / "candidate-items.jsonl"
    original_payload = payload_path.read_bytes()
    payload_path.write_bytes(b"drifted")
    with pytest.raises(
        CandidateOutputIntegrityError,
        match="inventory drifted",
    ):
        assembler.get(completed.run_id)
    payload_path.write_bytes(original_payload)
    manifest_path = output.bundle_path / "dataset-manifest.json"
    manifest_payload = manifest_path.read_bytes()
    manifest_path.unlink()
    with pytest.raises(
        CandidateOutputIntegrityError,
        match="manifest is missing",
    ):
        assembler.get(completed.run_id)
    manifest_path.write_bytes(manifest_payload)
    assert assembler.get(completed.run_id).inventory == (output.inventory)
    symlink_path = output.bundle_path / "unsafe-link"
    symlink_path.symlink_to("candidate-items.jsonl")
    with pytest.raises(
        CandidateOutputIntegrityError,
        match="symlink",
    ):
        assembler.get(completed.run_id)
    symlink_path.unlink()
    special_path = output.bundle_path / "unsafe-special"
    os.mkfifo(special_path)
    try:
        with pytest.raises(
            CandidateOutputIntegrityError,
            match="special file",
        ):
            assembler.get(completed.run_id)
    finally:
        special_path.unlink()
    connection = store._connect()
    try:
        connection.execute(
            "DELETE FROM candidate_output_current_heads WHERE run_id = ?",
            (completed.run_id,),
        )
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(
        CandidateOutputIntegrityError,
        match="not found",
    ):
        assembler.get(completed.run_id)
    assert assembler.rebuild_current_heads() == (completed.run_id,)
    assert assembler.get(completed.run_id).inventory == output.inventory


def test_candidate_output_after_rename_reuses_orphan_bundle(
    tmp_path: Path,
) -> None:
    payloads = {"README.md": b"# Candidate Dataset\n"}
    inventory = _inventory(
        dataset_run_id="factory-run://dataset/after-rename",
        aggregate_result_ref=_ref(
            "factory-dataset-aggregate-result",
            "after-rename",
        ),
        delivery_plan_ref=_ref(
            "dataset-delivery-plan",
            "after-rename",
        ),
        payloads=payloads,
    )
    manifest = CandidateDatasetDeliveryManifestV2.create(
        dataset_run_ref=_ref("factory-run", "after-rename"),
        aggregate_result_ref=inventory.aggregate_result_ref,
        batch_quality_ref=_ref("batch-quality-report", "after-rename"),
        candidate_item_refs=(_ref("evaluation-item", "after-rename"),),
        candidate_projection_refs=(_ref("release-projection-result", "after-rename"),),
        candidate_stage_head_refs=(_ref("factory-item-stage-head", "after-rename"),),
        rejected_binding_refs=(),
        blocked_binding_refs=(),
        inventory_ref=inventory.to_ref(),
        provenance_manifest_refs=(_ref("provenance-manifest", "after-rename"),),
        audit=factory_audit(),
    )
    payloads["dataset-manifest.json"] = manifest.canonical_json() + b"\n"
    root = tmp_path / "candidate-output"
    store = FactoryControlStore(tmp_path / "factory.sqlite3")

    def crash_after_rename(point: str) -> None:
        if point == "after_rename":
            raise RuntimeError("injected candidate output fault: after_rename")

    crashing = CandidateDatasetOutputAssembler(
        store=store,
        root=root,
        fault_injector=crash_after_rename,
    )
    with pytest.raises(
        RuntimeError,
        match="injected candidate output fault: after_rename",
    ):
        crashing._write_bundle(payloads, inventory, manifest)

    orphan = crashing._bundle_path(inventory.bundle_sha256)
    orphan_inode = orphan.stat().st_ino
    assert orphan.is_dir()
    assert not tuple(crashing._staging.iterdir())

    restarted = CandidateDatasetOutputAssembler(store=store, root=root)
    bundle_path, reused = restarted._write_bundle(
        payloads,
        inventory,
        manifest,
    )

    assert reused is True
    assert bundle_path == orphan
    assert bundle_path.stat().st_ino == orphan_inode
    assert len(tuple(restarted._cas.glob("*/*"))) == 1


@pytest.mark.parametrize(
    "relative_path",
    (
        "/absolute",
        "../escape",
        ".",
        "a/./b",
        "a//b",
        r"a\b",
    ),
)
def test_candidate_output_file_rejects_unsafe_path(
    relative_path: str,
) -> None:
    with pytest.raises(ValidationError, match="safe and relative"):
        CandidateOutputFileV1(
            relative_path=relative_path,
            size_bytes=0,
            content_sha256="a" * 64,
        )


def test_candidate_output_inventory_rejects_drift() -> None:
    value = CandidateOutputFileV1(
        relative_path="a.json",
        size_bytes=1,
        content_sha256="a" * 64,
    )
    base = {
        "inventory_id": (f"candidate-dataset-inventory://sha256/{'a' * 64}"),
        "dataset_run_id": "factory-run://inventory",
        "aggregate_result_ref": _ref("factory-dataset-aggregate-result"),
        "delivery_plan_ref": _ref("dataset-delivery-plan"),
        "bundle_sha256": "a" * 64,
        "inventory_sha256": "a" * 64,
    }
    with pytest.raises(ValidationError, match="sorted and unique"):
        CandidateDatasetInventoryV1(
            **base,
            files=(value, value),
        )
    with pytest.raises(ValidationError, match="bundle identity"):
        CandidateDatasetInventoryV1(
            **base,
            files=(value,),
        )


def test_candidate_output_rejects_reviewed_limits() -> None:
    plan = DatasetDeliveryPlanV2.create(
        plan_id="dataset-delivery-plan://limits",
        run_ref=_ref("factory-run", "limits"),
        plan_version=1,
        predecessor_plan_ref=None,
        aggregate_result_ref=_ref(
            "factory-dataset-aggregate-result",
            "limits",
        ),
        candidate_item_refs=(_ref("evaluation-item", "limits"),),
        candidate_projection_refs=(_ref("release-projection-result", "limits"),),
        candidate_stage_head_refs=(_ref("factory-item-stage-head", "limits"),),
        rejected_binding_refs=(),
        blocked_binding_refs=(),
        output_target_ref=_ref("candidate-output-target", "limits"),
        max_files=1,
        max_total_bytes=1,
        audit=factory_audit(),
    )

    with pytest.raises(
        CandidateOutputConflictError,
        match="reviewed limits",
    ):
        _require_reviewed_limits(
            {
                "one": b"",
                "two": b"",
            },
            plan,
        )
    with pytest.raises(
        CandidateOutputConflictError,
        match="reviewed limits",
    ):
        _require_reviewed_limits(
            {"one": b"too large"},
            plan,
        )


def test_candidate_output_rejects_non_directory_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "not-a-directory"
    root.write_text("x", encoding="utf-8")
    store = FactoryControlStore(tmp_path / "empty.sqlite3")
    with pytest.raises(
        CandidateOutputError,
        match="real directory",
    ):
        CandidateDatasetOutputAssembler(
            store=store,
            root=root,
        )


def test_candidate_output_rejects_symlink_root(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target"
    target.mkdir()
    root = tmp_path / "symlink-root"
    root.symlink_to(target, target_is_directory=True)
    store = FactoryControlStore(tmp_path / "empty.sqlite3")
    with pytest.raises(
        CandidateOutputError,
        match="real directory",
    ):
        CandidateDatasetOutputAssembler(
            store=store,
            root=root,
        )
