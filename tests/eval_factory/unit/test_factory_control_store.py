from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pytest

from eval_factory.agent_system.store import (
    FactoryControlConcurrencyError,
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
    CompiledDatasetBuildPlanV2,
    DatasetBuildPlanTaskV2,
    DatasetBuildPlanV2,
    EvaluationRequirementSpecV2,
    FactoryRunPolicyV2,
    PlanKindV2,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.orchestration.job_store import JobStore

HASH = "a" * 64
CREATE_FAULT_POINTS = (
    FactoryControlStoreFaultPoint.AFTER_POLICY,
    FactoryControlStoreFaultPoint.AFTER_REQUIREMENT,
    FactoryControlStoreFaultPoint.AFTER_RUN,
    FactoryControlStoreFaultPoint.AFTER_RUN_HEAD,
    FactoryControlStoreFaultPoint.AFTER_OUTBOX,
    FactoryControlStoreFaultPoint.AFTER_IDEMPOTENCY,
)
PLAN_FAULT_POINTS = (
    FactoryControlStoreFaultPoint.AFTER_PLAN,
    FactoryControlStoreFaultPoint.AFTER_COMPILED_PLAN,
    FactoryControlStoreFaultPoint.AFTER_RUN,
    FactoryControlStoreFaultPoint.AFTER_RUN_HEAD,
    FactoryControlStoreFaultPoint.AFTER_PLAN_HEAD,
    FactoryControlStoreFaultPoint.AFTER_OUTBOX,
    FactoryControlStoreFaultPoint.AFTER_IDEMPOTENCY,
)


def _ref(object_type: str, suffix: str = "example") -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://{suffix}/v2",
        object_version="v2",
        object_sha256=HASH,
    )


def _audit(actor: str = "factory-control-store-test") -> ContractAudit:
    return ContractAudit(
        created_at=datetime(2026, 8, 6, tzinfo=UTC),
        created_by=actor,
        governing_versions=(
            VersionBinding(
                component="graph-engineered-eval-factory",
                version="phase-a",
                sha256=HASH,
            ),
        ),
    )


def _policy() -> FactoryRunPolicyV2:
    return FactoryRunPolicyV2.create(
        policy_id="factory-run-policy://core-v1",
        allowed_task_kinds=("planning", "trace-extraction", "task-rewrite"),
        max_transitions=256,
        max_plan_revisions=8,
        max_agent_attempts=3,
        max_model_requests=1_000,
        max_model_tokens=2_000_000,
        max_cost_micro_usd=25_000_000,
        audit=_audit(),
    )


def _requirement(
    *,
    run_id: str = "factory-run://example",
    goal: str = "Build source-grounded rewrite candidates.",
) -> EvaluationRequirementSpecV2:
    return EvaluationRequirementSpecV2.create(
        requirement_spec_id=f"evaluation-requirement-spec://{run_id.rsplit('://', 1)[-1]}",
        run_id=run_id,
        source_ref=_ref("evaluation-requirement-source"),
        goals=(goal,),
        constraints=("Do not expose raw traces.",),
        assumptions=("Provider calls use an offline fake.",),
        open_questions=("Future quality validation requires 300 traces.",),
        requirement_version=1,
        audit=_audit(),
    )


def _plan(
    *,
    run_ref: ObjectRef,
    policy_ref: ObjectRef,
) -> tuple[DatasetBuildPlanV2, CompiledDatasetBuildPlanV2]:
    extract = DatasetBuildPlanTaskV2(
        task_key="extract",
        stage="core",
        task_kind="trace-extraction",
        agent_role="trace-extraction-agent",
        dependency_task_keys=(),
        input_object_types=("requirement-spec",),
        output_object_types=("extracted-user-prompt",),
        required_capability_ids=("capability://trace-extraction",),
        acceptance_check_refs=(_ref("acceptance-check", "extract"),),
        plan_review_kind=None,
        max_attempts=2,
        max_model_requests=1,
        max_model_tokens=4096,
        max_cost_micro_usd=100_000,
    )
    rewrite = DatasetBuildPlanTaskV2(
        task_key="rewrite",
        stage="core",
        task_kind="task-rewrite",
        agent_role="task-rewrite-agent",
        dependency_task_keys=("extract",),
        input_object_types=("extracted-user-prompt",),
        output_object_types=("task-rewrite-candidate",),
        required_capability_ids=("capability://task-rewrite",),
        acceptance_check_refs=(_ref("acceptance-check", "rewrite"),),
        plan_review_kind=PlanKindV2.TASK_REWRITE,
        max_attempts=2,
        max_model_requests=1,
        max_model_tokens=4096,
        max_cost_micro_usd=100_000,
    )
    plan = DatasetBuildPlanV2.create(
        plan_id="dataset-build-plan://example",
        run_ref=run_ref,
        plan_version=1,
        predecessor_plan_ref=None,
        goals=("Build source-grounded rewrite candidates.",),
        user_constraints=("Do not expose raw traces.",),
        assumptions=("Provider calls use an offline fake.",),
        unresolved_questions=("Future quality validation requires 300 traces.",),
        stage_order=("core",),
        tasks=(rewrite, extract),
        required_review_kinds=(PlanKindV2.GLOBAL_BUILD, PlanKindV2.TASK_REWRITE),
        total_model_requests=2,
        total_model_tokens=8192,
        total_cost_micro_usd=200_000,
        audit=_audit(),
    )
    compiled = CompiledDatasetBuildPlanV2.create(
        compiled_plan_id="compiled-dataset-build-plan://example",
        source_plan_ref=plan.to_ref(),
        policy_ref=policy_ref,
        tasks=plan.tasks,
        topological_task_keys=("extract", "rewrite"),
        audit=_audit(),
    )
    return plan, compiled


def _table_names(path: Path) -> tuple[str, ...]:
    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
        ).fetchall()
    return tuple(str(row[0]) for row in rows)


def test_create_run_is_atomic_replayable_and_separate_from_job_store(tmp_path: Path) -> None:
    job_store_path = tmp_path / "job-store.sqlite3"
    JobStore(job_store_path)
    job_schema_before = _table_names(job_store_path)

    store = FactoryControlStore(tmp_path / "factory-control.sqlite3")
    result = store.create_run(
        policy=_policy(),
        requirement=_requirement(),
        idempotency_key="create-run-example",
    )
    replay = store.create_run(
        policy=_policy(),
        requirement=_requirement(),
        idempotency_key="create-run-example",
    )

    assert replay == result
    assert store.get_run(result.run_id) == result
    assert store.get_run_by_ref(result.to_ref()) == result
    assert store.get_run_history(result.run_id) == (result,)
    assert store.get_policy(_policy().policy_id) == _policy()
    assert store.get_policy_by_ref(_policy().to_ref()) == _policy()
    assert store.get_requirement(_requirement().requirement_spec_id) == _requirement()
    assert store.get_requirement_by_ref(_requirement().to_ref()) == _requirement()
    assert _table_names(job_store_path) == job_schema_before
    assert "jobs" not in _table_names(store.path)
    assert {
        "factory_runs",
        "factory_run_current_heads",
        "dataset_build_plans",
        "compiled_dataset_build_plans",
        "agent_definitions",
        "agent_tasks",
        "agent_results",
        "plan_review_requests",
        "plan_decisions",
        "model_route_decisions",
        "gateway_receipt_refs",
        "dataset_delivery_manifests",
        "factory_control_idempotency",
        "factory_control_outbox",
    }.issubset(_table_names(store.path))


@pytest.mark.parametrize(
    "fault_point",
    CREATE_FAULT_POINTS,
)
def test_every_create_write_family_rolls_back(
    tmp_path: Path,
    fault_point: FactoryControlStoreFaultPoint,
) -> None:
    store = FactoryControlStore(
        tmp_path / f"{fault_point.value}.sqlite3",
        fault_injector=StaticFactoryControlStoreFaultInjector(crash_points=frozenset({fault_point})),
    )
    with pytest.raises(FactoryControlInjectedCrash, match=fault_point.value):
        store.create_run(
            policy=_policy(),
            requirement=_requirement(),
            idempotency_key="create-run-fault",
        )

    with sqlite3.connect(store.path) as connection:
        for table in (
            "factory_run_policies",
            "evaluation_requirement_specs",
            "factory_runs",
            "factory_run_current_heads",
            "factory_control_outbox",
            "factory_control_idempotency",
        ):
            assert connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone() == (0,)


def test_idempotency_and_run_business_authority_conflicts(tmp_path: Path) -> None:
    store = FactoryControlStore(tmp_path / "factory-control.sqlite3")
    store.create_run(
        policy=_policy(),
        requirement=_requirement(),
        idempotency_key="create-run-example",
    )

    with pytest.raises(FactoryControlConflictError, match="idempotency"):
        store.create_run(
            policy=_policy(),
            requirement=_requirement(goal="Changed request."),
            idempotency_key="create-run-example",
        )
    with pytest.raises(FactoryControlConflictError, match="already exists"):
        store.create_run(
            policy=_policy(),
            requirement=_requirement(),
            idempotency_key="different-command",
        )


def test_empty_store_reads_return_typed_not_found(tmp_path: Path) -> None:
    store = FactoryControlStore(tmp_path / "factory-control.sqlite3")
    with pytest.raises(FactoryControlNotFoundError, match="factory run not found"):
        store.get_run("factory-run://missing")
    with pytest.raises(FactoryControlNotFoundError, match="factory run not found"):
        store.get_run_history("factory-run://missing")
    with pytest.raises(FactoryControlNotFoundError, match="factory policy not found"):
        store.get_policy("factory-run-policy://missing")
    with pytest.raises(FactoryControlNotFoundError, match="factory requirement not found"):
        store.get_requirement("evaluation-requirement-spec://missing")


def test_policy_reuse_is_exact_across_runs(tmp_path: Path) -> None:
    store = FactoryControlStore(tmp_path / "factory-control.sqlite3")
    store.create_run(
        policy=_policy(),
        requirement=_requirement(),
        idempotency_key="create-run-one",
    )
    second = store.create_run(
        policy=_policy(),
        requirement=_requirement(run_id="factory-run://second"),
        idempotency_key="create-run-two",
    )
    assert second.policy_ref == _policy().to_ref()

    changed_policy = FactoryRunPolicyV2.create(
        policy_id="factory-run-policy://core-v1",
        allowed_task_kinds=("planning",),
        max_transitions=256,
        max_plan_revisions=8,
        max_agent_attempts=3,
        max_model_requests=1_000,
        max_model_tokens=2_000_000,
        max_cost_micro_usd=25_000_000,
        audit=_audit(),
    )
    with pytest.raises(FactoryControlConflictError, match="policy identity"):
        store.create_run(
            policy=changed_policy,
            requirement=_requirement(run_id="factory-run://third"),
            idempotency_key="create-run-three",
        )


def test_same_and_different_key_create_races_have_one_authority(tmp_path: Path) -> None:
    store = FactoryControlStore(tmp_path / "same-key.sqlite3")

    def create_same() -> str:
        return store.create_run(
            policy=_policy(),
            requirement=_requirement(),
            idempotency_key="same-key",
        ).object_id

    with ThreadPoolExecutor(max_workers=2) as executor:
        same_results = tuple(executor.map(lambda _: create_same(), range(2)))
    assert len(set(same_results)) == 1
    assert len(store.get_run_history("factory-run://example")) == 1

    different_store = FactoryControlStore(tmp_path / "different-key.sqlite3")

    def create_different(key: str) -> str:
        try:
            return different_store.create_run(
                policy=_policy(),
                requirement=_requirement(),
                idempotency_key=key,
            ).object_id
        except FactoryControlConflictError:
            return "CONFLICT"

    with ThreadPoolExecutor(max_workers=2) as executor:
        different_results = tuple(executor.map(create_different, ("key-a", "key-b")))
    assert different_results.count("CONFLICT") == 1
    assert len(different_store.get_run_history("factory-run://example")) == 1


def test_plan_commit_uses_optimistic_version_and_immutable_snapshots(tmp_path: Path) -> None:
    store = FactoryControlStore(tmp_path / "factory-control.sqlite3")
    initial = store.create_run(
        policy=_policy(),
        requirement=_requirement(),
        idempotency_key="create-run",
    )
    plan, compiled = _plan(run_ref=initial.to_ref(), policy_ref=_policy().to_ref())

    planned = store.commit_plan(
        run_id=initial.run_id,
        expected_run_version=0,
        plan=plan,
        compiled_plan=compiled,
        audit=_audit(),
        idempotency_key="commit-plan",
    )
    replay = store.commit_plan(
        run_id=initial.run_id,
        expected_run_version=0,
        plan=plan,
        compiled_plan=compiled,
        audit=_audit(),
        idempotency_key="commit-plan",
    )

    assert replay == planned
    assert planned.run_version == 1
    assert planned.current_plan_ref == plan.to_ref()
    assert planned.compiled_plan_ref == compiled.to_ref()
    assert store.get_plan(initial.run_id) == (plan, compiled)
    assert store.get_run_history(initial.run_id) == (initial, planned)

    with pytest.raises(FactoryControlConcurrencyError, match="expected run version"):
        store.commit_plan(
            run_id=initial.run_id,
            expected_run_version=0,
            plan=plan,
            compiled_plan=compiled,
            audit=_audit(),
            idempotency_key="stale-plan-command",
        )


@pytest.mark.parametrize("fault_point", PLAN_FAULT_POINTS)
def test_every_plan_write_family_rolls_back(
    tmp_path: Path,
    fault_point: FactoryControlStoreFaultPoint,
) -> None:
    path = tmp_path / f"plan-{fault_point.value}.sqlite3"
    initial_store = FactoryControlStore(path)
    initial = initial_store.create_run(
        policy=_policy(),
        requirement=_requirement(),
        idempotency_key="create-run",
    )
    plan, compiled = _plan(run_ref=initial.to_ref(), policy_ref=_policy().to_ref())
    store = FactoryControlStore(
        path,
        fault_injector=StaticFactoryControlStoreFaultInjector(crash_points=frozenset({fault_point})),
    )

    with pytest.raises(FactoryControlInjectedCrash, match=fault_point.value):
        store.commit_plan(
            run_id=initial.run_id,
            expected_run_version=0,
            plan=plan,
            compiled_plan=compiled,
            audit=_audit(),
            idempotency_key="commit-plan-fault",
        )

    assert store.get_run_history(initial.run_id) == (initial,)
    with pytest.raises(FactoryControlNotFoundError, match="current factory plan not found"):
        store.get_plan(initial.run_id)


def test_rebuild_restores_run_and_plan_heads_from_immutable_records(tmp_path: Path) -> None:
    store = FactoryControlStore(tmp_path / "factory-control.sqlite3")
    initial = store.create_run(
        policy=_policy(),
        requirement=_requirement(),
        idempotency_key="create-run",
    )
    plan, compiled = _plan(run_ref=initial.to_ref(), policy_ref=_policy().to_ref())
    planned = store.commit_plan(
        run_id=initial.run_id,
        expected_run_version=0,
        plan=plan,
        compiled_plan=compiled,
        audit=_audit(),
        idempotency_key="commit-plan",
    )
    with sqlite3.connect(store.path) as connection:
        connection.execute("DELETE FROM factory_run_current_heads")
        connection.execute("DELETE FROM plan_current_heads")

    with pytest.raises(FactoryControlIntegrityError, match="head is missing"):
        store.get_run(initial.run_id)
    rebuild = store.rebuild_current_heads()

    assert rebuild == FactoryControlHeadRebuild(run_head_count=1, plan_head_count=1)
    assert store.get_run(initial.run_id) == planned
    assert store.get_plan(initial.run_id) == (plan, compiled)


def test_head_drift_rejects_reads_and_explicit_rebuild_repairs_only_heads(tmp_path: Path) -> None:
    store = FactoryControlStore(tmp_path / "factory-control.sqlite3")
    run = store.create_run(
        policy=_policy(),
        requirement=_requirement(),
        idempotency_key="create-run",
    )

    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "UPDATE factory_run_current_heads SET run_version = 99 WHERE run_id = ?",
            (run.run_id,),
        )
    with pytest.raises(FactoryControlIntegrityError, match="head"):
        store.get_run(run.run_id)

    rebuild = store.rebuild_current_heads()
    assert rebuild == FactoryControlHeadRebuild(run_head_count=1, plan_head_count=0)
    assert store.get_run(run.run_id) == run

    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "UPDATE factory_runs SET record_json = '{}' WHERE run_id = ?",
            (run.run_id,),
        )
    with pytest.raises(FactoryControlIntegrityError, match="immutable"):
        store.rebuild_current_heads()
