from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from eval_factory.agent_system.store import (
    FactoryControlConflictError,
    FactoryControlHeadRebuild,
    FactoryControlInjectedCrash,
    FactoryControlIntegrityError,
    FactoryControlNotFoundError,
    FactoryControlStore,
    FactoryControlStoreFaultPoint,
    StaticFactoryControlStoreFaultInjector,
)
from eval_factory.contracts.agent_system_v2 import (
    EvaluationRequirementSpecV2,
    FactoryRunPolicyV2,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.contracts.dataset_runtime_v2 import (
    CandidateDatasetOutcomeV2,
    FactoryDatasetAggregateResultV2,
    FactoryItemRunBindingV2,
    FactoryItemStageHeadV2,
    FactoryItemStageOutcomeV2,
    FactoryItemStageV2,
)

HASH = "a" * 64


def _ref(object_type: str, suffix: str = "example") -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://{suffix}/v2",
        object_version="v2",
        object_sha256=HASH,
    )


def _audit(actor: str = "factory-dataset-store-test") -> ContractAudit:
    return ContractAudit(
        created_at=datetime(2026, 8, 7, tzinfo=UTC),
        created_by=actor,
        governing_versions=(
            VersionBinding(
                component="factory-dataset-runtime",
                version="v1",
                sha256=HASH,
            ),
        ),
    )


def _policy() -> FactoryRunPolicyV2:
    return FactoryRunPolicyV2.create(
        policy_id="factory-run-policy://dataset-runtime",
        allowed_task_kinds=(
            "attachment-generation",
            "criteria-rubric",
            "grading-design",
            "planning",
            "task-authoring",
            "task-rewrite",
            "trace-extraction",
        ),
        max_transitions=2_048,
        max_plan_revisions=8,
        max_agent_attempts=3,
        max_model_requests=10_000,
        max_model_tokens=20_000_000,
        max_cost_micro_usd=250_000_000,
        audit=_audit(),
    )


def _requirement(run_id: str) -> EvaluationRequirementSpecV2:
    suffix = run_id.rsplit("/", 1)[-1]
    return EvaluationRequirementSpecV2.create(
        requirement_spec_id=f"evaluation-requirement-spec://{suffix}",
        run_id=run_id,
        source_ref=_ref("evaluation-requirement-source", suffix),
        goals=("Build safe candidate evaluation items.",),
        constraints=("Do not expose restricted material.",),
        assumptions=("Provider boundaries use deterministic fakes.",),
        open_questions=(),
        requirement_version=1,
        audit=_audit(),
    )


def _create_run(
    store: FactoryControlStore,
    run_id: str,
) -> ObjectRef:
    return store.create_run(
        policy=_policy(),
        requirement=_requirement(run_id),
        idempotency_key=f"create:{run_id}",
    ).to_ref()


def _binding(
    dataset_run_ref: ObjectRef,
    item_run_ref: ObjectRef,
    *,
    item_id: str = "item://dataset/one",
    version: int = 1,
    predecessor: ObjectRef | None = None,
) -> FactoryItemRunBindingV2:
    suffix = item_id.rsplit("/", 1)[-1]
    return FactoryItemRunBindingV2.create(
        dataset_run_ref=dataset_run_ref,
        item_run_ref=item_run_ref,
        item_id=item_id,
        binding_version=version,
        predecessor_binding_ref=predecessor,
        trace_candidate_decision_ref=_ref(
            "trace-candidate-decision",
            suffix,
        ),
        extracted_prompt_ref=_ref("extracted-user-prompt", suffix),
        inferred_intent_ref=_ref("inferred-user-intent", suffix),
        rewrite_candidate_ref=_ref("task-rewrite-candidate", suffix),
        audit=_audit(),
    )


def _stage(
    binding: FactoryItemRunBindingV2,
    *,
    version: int = 1,
    predecessor: ObjectRef | None = None,
    outcome: FactoryItemStageOutcomeV2 = (FactoryItemStageOutcomeV2.SUCCEEDED),
    reasons: tuple[str, ...] = (),
) -> FactoryItemStageHeadV2:
    return FactoryItemStageHeadV2.create(
        item_binding_ref=binding.to_ref(),
        item_run_ref=binding.item_run_ref,
        stage=FactoryItemStageV2.ATTACHMENT,
        stage_version=version,
        predecessor_head_ref=predecessor,
        dependency_result_refs=(_ref("r4-task-contract-set"),),
        result_ref=_ref("attachment-subgraph-result"),
        outcome=outcome,
        reason_codes=reasons,
        audit=_audit(),
    )


def _prepared(
    tmp_path: Path,
) -> tuple[
    FactoryControlStore,
    FactoryItemRunBindingV2,
]:
    store = FactoryControlStore(tmp_path / "factory-control.sqlite3")
    dataset = _create_run(store, "factory-run://dataset/example")
    child = _create_run(store, "factory-run://dataset/item-one")
    return store, _binding(dataset, child)


def _count(path: Path, table: str) -> int:
    with sqlite3.connect(path) as connection:
        return int(
            connection.execute(
                f"SELECT COUNT(*) FROM {table}",
            ).fetchone()[0]
        )


def test_item_binding_and_stage_heads_are_atomic_and_replayable(
    tmp_path: Path,
) -> None:
    store, binding = _prepared(tmp_path)

    committed = store.commit_item_binding(
        binding,
        idempotency_key="bind-item-one",
    )
    replay = store.commit_item_binding(
        binding,
        idempotency_key="bind-item-one",
    )
    assert replay == committed == binding
    assert (
        store.get_item_binding(
            "factory-run://dataset/example",
            binding.item_id,
        )
        == binding
    )
    assert store.list_item_bindings(
        "factory-run://dataset/example",
    ) == (binding,)
    assert store.get_item_run(binding).to_ref() == binding.item_run_ref
    planning = store.begin_item_planning(
        binding,
        idempotency_key="begin-item-one-planning",
        audit=_audit(),
    )
    replay_planning = store.begin_item_planning(
        binding,
        idempotency_key="begin-item-one-planning",
        audit=_audit(),
    )
    assert replay_planning == planning
    assert planning.status.value == "PLANNING"
    assert planning.run_version == 1
    assert store.get_item_run(binding) == planning

    stage = _stage(binding)
    material_ref = _ref("stage-material", "item-one-attachment")
    committed_stage = store.commit_item_stage_head(
        stage,
        idempotency_key="item-one-attachment-v1",
        material_ref=material_ref,
    )
    replay_stage = store.commit_item_stage_head(
        stage,
        idempotency_key="item-one-attachment-v1",
        material_ref=material_ref,
    )
    assert replay_stage == committed_stage == stage
    assert store.get_item_stage_material_ref(stage.to_ref()) == material_ref
    with pytest.raises(
        FactoryControlConflictError,
        match="idempotency",
    ):
        store.commit_item_stage_head(
            stage,
            idempotency_key="item-one-attachment-v1",
            material_ref=_ref(
                "stage-material",
                "changed",
            ),
        )
    assert (
        store.get_item_stage_head(
            binding.item_id,
            FactoryItemStageV2.ATTACHMENT,
        )
        == stage
    )
    assert store.list_item_stage_heads(binding.item_id) == (stage,)

    assert _count(store.path, "factory_item_run_bindings") == 1
    assert _count(store.path, "factory_item_binding_current_heads") == 1
    assert _count(store.path, "factory_item_stage_records") == 1
    assert _count(store.path, "factory_item_stage_current_heads") == 1
    assert _count(store.path, "factory_item_stage_material_refs") == 1


def test_item_planning_rejects_a_second_transition_key(
    tmp_path: Path,
) -> None:
    store, binding = _prepared(tmp_path)
    store.commit_item_binding(
        binding,
        idempotency_key="bind-item-one",
    )
    planning = store.begin_item_planning(
        binding,
        idempotency_key="begin-item-one-planning",
        audit=_audit(),
    )

    with pytest.raises(
        FactoryControlConflictError,
        match="created child run",
    ):
        store.begin_item_planning(
            binding,
            idempotency_key="different-planning-key",
            audit=_audit("different-actor"),
        )

    assert store.get_item_run(binding) == planning
    assert len(store.get_run_history(planning.run_id)) == 2


@pytest.mark.parametrize(
    "fault_point",
    (
        FactoryControlStoreFaultPoint.AFTER_RUN,
        FactoryControlStoreFaultPoint.AFTER_RUN_HEAD,
        FactoryControlStoreFaultPoint.AFTER_OUTBOX,
        FactoryControlStoreFaultPoint.AFTER_IDEMPOTENCY,
    ),
)
def test_item_planning_faults_roll_back_complete_transaction(
    tmp_path: Path,
    fault_point: FactoryControlStoreFaultPoint,
) -> None:
    path = tmp_path / f"planning-{fault_point.value}.sqlite3"
    store = FactoryControlStore(path)
    dataset = _create_run(
        store,
        "factory-run://dataset/planning-fault",
    )
    child = _create_run(
        store,
        "factory-run://dataset/planning-fault-item",
    )
    binding = _binding(
        dataset,
        child,
        item_id="item://dataset/planning-fault",
    )
    store.commit_item_binding(
        binding,
        idempotency_key="bind-planning-fault",
    )
    baseline_outbox = _count(path, "factory_control_outbox")
    baseline_idempotency = _count(
        path,
        "factory_control_idempotency",
    )
    crashing = FactoryControlStore(
        path,
        fault_injector=StaticFactoryControlStoreFaultInjector(
            crash_points=frozenset({fault_point}),
        ),
    )

    with pytest.raises(
        FactoryControlInjectedCrash,
        match=fault_point.value,
    ):
        crashing.begin_item_planning(
            binding,
            idempotency_key="begin-planning-fault",
            audit=_audit(),
        )

    assert _count(path, "factory_control_outbox") == baseline_outbox
    assert _count(path, "factory_control_idempotency") == baseline_idempotency
    assert len(store.get_run_history("factory-run://dataset/planning-fault-item")) == 1
    assert store.get_item_run(binding).status.value == "CREATED"

    recovered = FactoryControlStore(path)
    planning = recovered.begin_item_planning(
        binding,
        idempotency_key="begin-planning-fault",
        audit=_audit(),
    )
    assert planning.status.value == "PLANNING"
    assert planning.run_version == 1


def test_item_binding_successor_invalidates_only_that_item_stage_heads(
    tmp_path: Path,
) -> None:
    store, first = _prepared(tmp_path)
    second_child = _create_run(
        store,
        "factory-run://dataset/item-one-v2",
    )
    other_child = _create_run(
        store,
        "factory-run://dataset/item-two",
    )
    other = _binding(
        first.dataset_run_ref,
        other_child,
        item_id="item://dataset/two",
    )
    store.commit_item_binding(first, idempotency_key="bind-one-v1")
    store.commit_item_binding(other, idempotency_key="bind-two-v1")
    first_stage = store.commit_item_stage_head(
        _stage(first),
        idempotency_key="stage-one-v1",
    )
    other_stage = store.commit_item_stage_head(
        _stage(other),
        idempotency_key="stage-two-v1",
    )

    successor = _binding(
        first.dataset_run_ref,
        second_child,
        version=2,
        predecessor=first.to_ref(),
    )
    store.commit_item_binding(
        successor,
        idempotency_key="bind-one-v2",
    )

    assert (
        store.get_item_binding(
            "factory-run://dataset/example",
            first.item_id,
        )
        == successor
    )
    with pytest.raises(
        FactoryControlNotFoundError,
        match="item stage head",
    ):
        store.get_item_stage_head(
            first.item_id,
            FactoryItemStageV2.ATTACHMENT,
        )
    assert (
        store.get_item_stage_head(
            other.item_id,
            FactoryItemStageV2.ATTACHMENT,
        )
        == other_stage
    )
    assert first_stage.item_binding_ref == first.to_ref()


def test_item_store_rejects_stale_bindings_heads_and_changed_replays(
    tmp_path: Path,
) -> None:
    store, binding = _prepared(tmp_path)
    store.commit_item_binding(binding, idempotency_key="binding")

    changed = FactoryItemRunBindingV2.create(
        dataset_run_ref=binding.dataset_run_ref,
        item_run_ref=binding.item_run_ref,
        item_id=binding.item_id,
        binding_version=binding.binding_version,
        predecessor_binding_ref=binding.predecessor_binding_ref,
        trace_candidate_decision_ref=binding.trace_candidate_decision_ref,
        extracted_prompt_ref=binding.extracted_prompt_ref,
        inferred_intent_ref=binding.inferred_intent_ref,
        rewrite_candidate_ref=_ref(
            "task-rewrite-candidate",
            "changed",
        ),
        audit=_audit(),
    )
    with pytest.raises(
        FactoryControlConflictError,
        match="idempotency",
    ):
        store.commit_item_binding(
            changed,
            idempotency_key="binding",
        )
    with pytest.raises(
        FactoryControlConflictError,
        match="binding authority",
    ):
        store.commit_item_binding(
            changed,
            idempotency_key="different-key",
        )

    first_stage = store.commit_item_stage_head(
        _stage(binding),
        idempotency_key="stage",
    )
    stale_successor = _stage(
        binding,
        version=2,
        predecessor=_ref("factory-item-stage-head", "wrong"),
    )
    with pytest.raises(
        FactoryControlConflictError,
        match="predecessor",
    ):
        store.commit_item_stage_head(
            stale_successor,
            idempotency_key="stale-stage",
        )
    assert (
        store.get_item_stage_head(
            binding.item_id,
            FactoryItemStageV2.ATTACHMENT,
        )
        == first_stage
    )


@pytest.mark.parametrize(
    "fault_point",
    (
        FactoryControlStoreFaultPoint.AFTER_ITEM_BINDING,
        FactoryControlStoreFaultPoint.AFTER_ITEM_BINDING_HEAD,
        FactoryControlStoreFaultPoint.AFTER_OUTBOX,
        FactoryControlStoreFaultPoint.AFTER_IDEMPOTENCY,
    ),
)
def test_item_binding_faults_roll_back_complete_transaction(
    tmp_path: Path,
    fault_point: FactoryControlStoreFaultPoint,
) -> None:
    path = tmp_path / f"binding-{fault_point.value}.sqlite3"
    initial = FactoryControlStore(path)
    dataset = _create_run(initial, "factory-run://dataset/example")
    child = _create_run(initial, "factory-run://dataset/item-one")
    binding = _binding(dataset, child)
    store = FactoryControlStore(
        path,
        fault_injector=StaticFactoryControlStoreFaultInjector(
            crash_points=frozenset({fault_point}),
        ),
    )

    with pytest.raises(FactoryControlInjectedCrash, match=fault_point.value):
        store.commit_item_binding(
            binding,
            idempotency_key="fault-binding",
        )
    assert _count(path, "factory_item_run_bindings") == 0
    assert _count(path, "factory_item_binding_current_heads") == 0


@pytest.mark.parametrize(
    "fault_point",
    (
        FactoryControlStoreFaultPoint.AFTER_ITEM_STAGE,
        FactoryControlStoreFaultPoint.AFTER_ITEM_STAGE_HEAD,
        FactoryControlStoreFaultPoint.AFTER_OUTBOX,
        FactoryControlStoreFaultPoint.AFTER_IDEMPOTENCY,
    ),
)
def test_item_stage_faults_roll_back_complete_transaction(
    tmp_path: Path,
    fault_point: FactoryControlStoreFaultPoint,
) -> None:
    path = tmp_path / f"stage-{fault_point.value}.sqlite3"
    initial = FactoryControlStore(path)
    dataset = _create_run(initial, "factory-run://dataset/example")
    child = _create_run(initial, "factory-run://dataset/item-one")
    binding = _binding(dataset, child)
    initial.commit_item_binding(
        binding,
        idempotency_key="binding",
    )
    store = FactoryControlStore(
        path,
        fault_injector=StaticFactoryControlStoreFaultInjector(
            crash_points=frozenset({fault_point}),
        ),
    )
    stage = _stage(binding)
    material_ref = _ref("stage-material", fault_point.value)

    with pytest.raises(FactoryControlInjectedCrash, match=fault_point.value):
        store.commit_item_stage_head(
            stage,
            idempotency_key="fault-stage",
            material_ref=material_ref,
        )
    assert _count(path, "factory_item_stage_records") == 0
    assert _count(path, "factory_item_stage_current_heads") == 0
    assert _count(path, "factory_item_stage_material_refs") == 0

    recovered = FactoryControlStore(path)
    assert (
        recovered.commit_item_stage_head(
            stage,
            idempotency_key="fault-stage",
            material_ref=material_ref,
        )
        == stage
    )
    assert recovered.get_item_stage_material_ref(stage.to_ref()) == material_ref
    assert _count(path, "factory_item_stage_records") == 1
    assert _count(path, "factory_item_stage_current_heads") == 1
    assert _count(path, "factory_item_stage_material_refs") == 1


def test_item_head_rebuild_restores_only_current_binding_stages(
    tmp_path: Path,
) -> None:
    store, binding = _prepared(tmp_path)
    store.commit_item_binding(binding, idempotency_key="binding")
    stage = store.commit_item_stage_head(
        _stage(binding),
        idempotency_key="stage",
    )
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "DELETE FROM factory_item_binding_current_heads",
        )
        connection.execute(
            "DELETE FROM factory_item_stage_current_heads",
        )

    with pytest.raises(FactoryControlIntegrityError, match="binding head"):
        store.get_item_binding(
            "factory-run://dataset/example",
            binding.item_id,
        )
    rebuilt = store.rebuild_current_heads()
    assert rebuilt == FactoryControlHeadRebuild(
        run_head_count=2,
        plan_head_count=0,
        item_binding_head_count=1,
        item_stage_head_count=1,
    )
    assert (
        store.get_item_binding(
            "factory-run://dataset/example",
            binding.item_id,
        )
        == binding
    )
    assert (
        store.get_item_stage_head(
            binding.item_id,
            FactoryItemStageV2.ATTACHMENT,
        )
        == stage
    )


def test_item_immutable_drift_cannot_be_repaired_by_rebuild(
    tmp_path: Path,
) -> None:
    store, binding = _prepared(tmp_path)
    store.commit_item_binding(binding, idempotency_key="binding")
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            """
            UPDATE factory_item_run_bindings
            SET record_json = '{}'
            WHERE binding_object_id = ?
            """,
            (binding.object_id,),
        )
    with pytest.raises(
        FactoryControlIntegrityError,
        match="immutable factory item",
    ):
        store.rebuild_current_heads()


def _aggregate_ready(
    store: FactoryControlStore,
    binding: FactoryItemRunBindingV2,
) -> FactoryDatasetAggregateResultV2:
    store.commit_item_binding(
        binding,
        idempotency_key="aggregate-binding",
    )
    grading_head = FactoryItemStageHeadV2.create(
        item_binding_ref=binding.to_ref(),
        item_run_ref=binding.item_run_ref,
        stage=FactoryItemStageV2.GRADING_DESIGN,
        stage_version=1,
        predecessor_head_ref=None,
        dependency_result_refs=(_ref("criteria-rubric-result"),),
        result_ref=_ref("grading-design-result"),
        outcome=FactoryItemStageOutcomeV2.SUCCEEDED,
        reason_codes=(),
        audit=_audit(),
    )
    store.commit_item_stage_head(
        grading_head,
        idempotency_key="aggregate-grading-head",
    )
    return FactoryDatasetAggregateResultV2.create(
        dataset_run_ref=binding.dataset_run_ref,
        core_vertical_result_ref=_ref(
            "core-vertical-result",
            "aggregate",
        ),
        item_binding_refs=(binding.to_ref(),),
        candidate_binding_refs=(binding.to_ref(),),
        rejected_binding_refs=(),
        blocked_binding_refs=(),
        batch_quality_ref=_ref(
            "batch-quality-report",
            "aggregate",
        ),
        outcome=CandidateDatasetOutcomeV2.COMPLETE,
        reason_codes=(),
        audit=_audit(),
    )


def test_dataset_aggregate_is_atomic_replayable_and_rebuildable(
    tmp_path: Path,
) -> None:
    store, binding = _prepared(tmp_path)
    aggregate = _aggregate_ready(store, binding)
    material_ref = _ref(
        "batch-quality-material",
        "aggregate",
    )

    committed = store.commit_dataset_aggregate(
        aggregate,
        material_ref=material_ref,
        idempotency_key="commit-aggregate",
    )
    replay = store.commit_dataset_aggregate(
        aggregate,
        material_ref=material_ref,
        idempotency_key="commit-aggregate",
    )

    assert replay == committed == aggregate
    assert store.get_dataset_aggregate("factory-run://dataset/example") == aggregate
    assert store.get_dataset_aggregate_material_ref("factory-run://dataset/example") == material_ref
    with pytest.raises(
        FactoryControlConflictError,
        match="already exists",
    ):
        store.commit_dataset_aggregate(
            aggregate,
            material_ref=material_ref,
            idempotency_key="different-aggregate-key",
        )

    with sqlite3.connect(store.path) as connection:
        connection.execute("DELETE FROM factory_dataset_aggregate_current_heads")
    with pytest.raises(
        FactoryControlIntegrityError,
        match="aggregate head",
    ):
        store.get_dataset_aggregate("factory-run://dataset/example")
    rebuilt = store.rebuild_current_heads()
    assert rebuilt.dataset_aggregate_head_count == 1
    assert store.get_dataset_aggregate("factory-run://dataset/example") == aggregate


@pytest.mark.parametrize(
    "fault_point",
    (
        FactoryControlStoreFaultPoint.AFTER_DATASET_AGGREGATE,
        FactoryControlStoreFaultPoint.AFTER_DATASET_AGGREGATE_HEAD,
        FactoryControlStoreFaultPoint.AFTER_OUTBOX,
        FactoryControlStoreFaultPoint.AFTER_IDEMPOTENCY,
    ),
)
def test_dataset_aggregate_faults_roll_back_complete_transaction(
    tmp_path: Path,
    fault_point: FactoryControlStoreFaultPoint,
) -> None:
    path = tmp_path / f"aggregate-{fault_point.value}.sqlite3"
    store = FactoryControlStore(path)
    dataset = _create_run(
        store,
        "factory-run://dataset/aggregate-fault",
    )
    child = _create_run(
        store,
        "factory-run://dataset/aggregate-fault-item",
    )
    binding = _binding(
        dataset,
        child,
        item_id="item://dataset/aggregate-fault",
    )
    aggregate = _aggregate_ready(store, binding)
    material_ref = _ref(
        "batch-quality-material",
        fault_point.value,
    )
    baseline_outbox = _count(path, "factory_control_outbox")
    baseline_idempotency = _count(
        path,
        "factory_control_idempotency",
    )
    crashing = FactoryControlStore(
        path,
        fault_injector=StaticFactoryControlStoreFaultInjector(
            crash_points=frozenset({fault_point}),
        ),
    )

    with pytest.raises(
        FactoryControlInjectedCrash,
        match=fault_point.value,
    ):
        crashing.commit_dataset_aggregate(
            aggregate,
            material_ref=material_ref,
            idempotency_key="fault-aggregate",
        )

    assert (
        _count(
            path,
            "factory_dataset_aggregate_results",
        )
        == 0
    )
    assert (
        _count(
            path,
            "factory_dataset_aggregate_current_heads",
        )
        == 0
    )
    assert (
        _count(
            path,
            "factory_dataset_aggregate_material_refs",
        )
        == 0
    )
    assert _count(path, "factory_control_outbox") == baseline_outbox
    assert _count(path, "factory_control_idempotency") == baseline_idempotency
    recovered = FactoryControlStore(path)
    assert (
        recovered.commit_dataset_aggregate(
            aggregate,
            material_ref=material_ref,
            idempotency_key="fault-aggregate",
        )
        == aggregate
    )
