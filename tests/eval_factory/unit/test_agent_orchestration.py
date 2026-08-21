from __future__ import annotations

import sqlite3
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from eval_factory.agent_system.planner import (
    DatasetBuildPlanCompiler,
    DatasetBuildPlanCompilerError,
)
from eval_factory.agent_system.registry import AgentRegistry, AgentRegistryError
from eval_factory.agent_system.store import FactoryControlStore
from eval_factory.agent_system.supervisor import (
    AgentCancellationStateV2,
    AgentLeaseV2,
    AgentRetryDecisionKindV2,
    AgentWorkerRef,
    AgentWorkEventKindV2,
    ExecutionSupervisor,
    ExecutionSupervisorError,
    StaleAgentLeaseError,
)
from eval_factory.agent_system.workspace import (
    AgentWorkspaceManager,
)
from eval_factory.contracts.agent_system_v2 import (
    AgentCapabilityV2,
    AgentDefinitionV2,
    AgentResultEnvelopeV2,
    AgentTaskOutcomeV2,
    AgentTaskV2,
    DatasetBuildPlanTaskV2,
    DatasetBuildPlanV2,
    EvaluationRequirementSpecV2,
    FactoryRunPolicyV2,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding

HASH = "a" * 64
NOW = datetime(2026, 8, 6, tzinfo=UTC)


@dataclass
class MutableClock:
    value: datetime

    def __call__(self) -> datetime:
        return self.value

    def advance(self, delta: timedelta) -> None:
        self.value += delta


def _ref(object_type: str, suffix: str = "example") -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://{suffix}/v2",
        object_version="v2",
        object_sha256=HASH,
    )


def _audit() -> ContractAudit:
    return ContractAudit(
        created_at=NOW,
        created_by="agent-orchestration-test",
        governing_versions=(
            VersionBinding(
                component="graph-engineered-eval-factory",
                version="supervisor-v1",
                sha256=HASH,
            ),
        ),
    )


def _capability(task_kind: str, input_type: str, output_type: str) -> AgentCapabilityV2:
    return AgentCapabilityV2.create(
        capability_id=f"agent-capability://{task_kind}",
        task_kinds=(task_kind,),
        input_object_types=(input_type,),
        output_object_types=(output_type,),
        model_capabilities=("structured-output", task_kind),
        tool_ids=(f"{task_kind}-tool",),
        data_purposes=("evaluation-dataset-construction",),
        data_classifications=("RESTRICTED_TRACE_DERIVED",),
        audit=_audit(),
    )


def _definition(
    task_kind: str,
    input_type: str,
    output_type: str,
) -> tuple[AgentCapabilityV2, AgentDefinitionV2]:
    capability = _capability(task_kind, input_type, output_type)
    definition = AgentDefinitionV2.create(
        agent_definition_id=f"agent-definition://{task_kind}",
        agent_role=f"{task_kind}-agent",
        agent_version="v1",
        capability_refs=(capability.to_ref(),),
        prompt_template_ref=_ref("prompt-template", task_kind),
        model_policy_ref=_ref("model-routing-policy"),
        tool_ids=(f"{task_kind}-tool",),
        data_purpose="evaluation-dataset-construction",
        allowed_data_classifications=("RESTRICTED_TRACE_DERIVED",),
        validator_refs=(_ref("validator", task_kind),),
        max_attempts=2,
        max_model_requests=2,
        max_model_tokens=4096,
        max_cost_micro_usd=100_000,
        workspace_isolated=True,
        network_allowed=False,
        audit=_audit(),
    )
    return capability, definition


def _registry() -> AgentRegistry:
    extract = _definition(
        "trace-extraction",
        "evaluation-requirement-spec",
        "extracted-user-prompt",
    )
    rewrite = _definition(
        "task-rewrite",
        "extracted-user-prompt",
        "task-rewrite-candidate",
    )
    return AgentRegistry(
        capabilities=(extract[0], rewrite[0]),
        definitions=(extract[1], rewrite[1]),
    )


def _policy() -> FactoryRunPolicyV2:
    return FactoryRunPolicyV2.create(
        policy_id="factory-run-policy://core",
        allowed_task_kinds=("task-rewrite", "trace-extraction"),
        max_transitions=256,
        max_plan_revisions=4,
        max_agent_attempts=2,
        max_model_requests=10,
        max_model_tokens=100_000,
        max_cost_micro_usd=1_000_000,
        audit=_audit(),
    )


def _plan(run_ref: ObjectRef) -> DatasetBuildPlanV2:
    extract = DatasetBuildPlanTaskV2(
        task_key="extract",
        stage="core",
        task_kind="trace-extraction",
        agent_role="trace-extraction-agent",
        dependency_task_keys=(),
        input_object_types=("evaluation-requirement-spec",),
        output_object_types=("extracted-user-prompt",),
        required_capability_ids=("agent-capability://trace-extraction",),
        acceptance_check_refs=(_ref("acceptance-check", "extract"),),
        plan_review_kind=None,
        max_attempts=2,
        max_model_requests=2,
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
        required_capability_ids=("agent-capability://task-rewrite",),
        acceptance_check_refs=(_ref("acceptance-check", "rewrite"),),
        plan_review_kind=None,
        max_attempts=2,
        max_model_requests=2,
        max_model_tokens=4096,
        max_cost_micro_usd=100_000,
    )
    return DatasetBuildPlanV2.create(
        plan_id="dataset-build-plan://core",
        run_ref=run_ref,
        plan_version=1,
        predecessor_plan_ref=None,
        goals=("Build source-grounded rewrite candidates.",),
        user_constraints=("Intent cannot replace the original prompt.",),
        assumptions=(),
        unresolved_questions=(),
        stage_order=("core",),
        tasks=(rewrite, extract),
        required_review_kinds=(),
        total_model_requests=4,
        total_model_tokens=8192,
        total_cost_micro_usd=200_000,
        audit=_audit(),
    )


def _plan_with_tasks(
    source: DatasetBuildPlanV2,
    *,
    suffix: str,
    tasks: tuple[DatasetBuildPlanTaskV2, ...],
) -> DatasetBuildPlanV2:
    return DatasetBuildPlanV2.create(
        plan_id=f"dataset-build-plan://{suffix}",
        run_ref=source.run_ref,
        plan_version=1,
        predecessor_plan_ref=None,
        goals=source.goals,
        user_constraints=source.user_constraints,
        assumptions=source.assumptions,
        unresolved_questions=source.unresolved_questions,
        stage_order=source.stage_order,
        tasks=tasks,
        required_review_kinds=source.required_review_kinds,
        total_model_requests=source.total_model_requests,
        total_model_tokens=source.total_model_tokens,
        total_cost_micro_usd=source.total_cost_micro_usd,
        audit=_audit(),
    )


def _control_store(tmp_path: Path) -> tuple[FactoryControlStore, str]:
    store = FactoryControlStore(tmp_path / "factory-control.sqlite3")
    requirement = EvaluationRequirementSpecV2.create(
        requirement_spec_id="evaluation-requirement-spec://core",
        run_id="factory-run://core",
        source_ref=_ref("evaluation-requirement-source"),
        goals=("Build source-grounded rewrite candidates.",),
        constraints=("Intent cannot replace the original prompt.",),
        assumptions=(),
        open_questions=(),
        requirement_version=1,
        audit=_audit(),
    )
    run = store.create_run(
        policy=_policy(),
        requirement=requirement,
        idempotency_key="create-run",
    )
    return store, run.run_id


def _supervisor(
    tmp_path: Path,
    store: FactoryControlStore,
    registry: AgentRegistry,
    *,
    clock: datetime | Callable[[], datetime] = NOW,
) -> ExecutionSupervisor:
    clock_callable = clock if callable(clock) else lambda: clock
    return ExecutionSupervisor(
        store,
        registry,
        workspace_manager=AgentWorkspaceManager(tmp_path / "agent-workspaces"),
        clock=clock_callable,
    )


def _materialized_extract(
    tmp_path: Path,
    clock: MutableClock,
) -> tuple[ExecutionSupervisor, AgentTaskV2]:
    store, run_id = _control_store(tmp_path)
    registry = _registry()
    current = store.get_run(run_id)
    plan = _plan(current.to_ref())
    compiled = DatasetBuildPlanCompiler(registry).compile(
        plan=plan,
        policy=_policy(),
        audit=_audit(),
    )
    store.commit_plan(
        run_id=run_id,
        expected_run_version=0,
        plan=plan,
        compiled_plan=compiled,
        audit=_audit(),
        idempotency_key="commit-plan",
    )
    supervisor = _supervisor(
        tmp_path,
        store,
        registry,
        clock=clock,
    )
    task = next(
        value
        for value in supervisor.materialize(
            run_id=run_id,
            compiled_plan=compiled,
            audit=_audit(),
        )
        if value.plan_task_key == "extract"
    )
    return supervisor, task


def _result_envelope(
    task: AgentTaskV2,
    lease: AgentLeaseV2,
    workspace_ref: ObjectRef,
) -> AgentResultEnvelopeV2:
    return AgentResultEnvelopeV2.create(
        result_id=(f"agent-result-envelope://extract/{lease.attempt}"),
        task_ref=task.to_ref(),
        agent_definition_ref=lease.agent_definition_ref,
        attempt=lease.attempt,
        lease_id=lease.lease_id,
        fencing_token=lease.fencing_token,
        workspace_receipt_ref=workspace_ref,
        outcome=AgentTaskOutcomeV2.SUCCEEDED,
        output_refs=(_ref("extracted-user-prompt"),),
        delegated_dataset_job_result_refs=(),
        gateway_receipt_refs=(_ref("gateway-receipt"),),
        validator_result_refs=(_ref("validator-result"),),
        failure_code=None,
        safe_metrics=(),
        audit=_audit(),
    )


def test_registry_and_compiler_enforce_capability_and_schema_closure() -> None:
    registry = _registry()
    plan = _plan(_ref("factory-run"))
    compiled = DatasetBuildPlanCompiler(registry).compile(
        plan=plan,
        policy=_policy(),
        audit=_audit(),
    )

    assert compiled.topological_task_keys == ("extract", "rewrite")
    assert registry.resolve("trace-extraction-agent", "trace-extraction").agent_role == (
        "trace-extraction-agent"
    )

    bad_task = plan.tasks[1].model_copy(
        update={"agent_role": "missing-agent"},
    )
    with pytest.raises(DatasetBuildPlanCompilerError, match="Agent"):
        DatasetBuildPlanCompiler(registry).compile(
            plan=DatasetBuildPlanV2.create(
                plan_id="dataset-build-plan://bad-agent",
                run_ref=plan.run_ref,
                plan_version=1,
                predecessor_plan_ref=None,
                goals=plan.goals,
                user_constraints=plan.user_constraints,
                assumptions=(),
                unresolved_questions=(),
                stage_order=("core",),
                tasks=(plan.tasks[0], bad_task),
                required_review_kinds=(),
                total_model_requests=4,
                total_model_tokens=8192,
                total_cost_micro_usd=200_000,
                audit=_audit(),
            ),
            policy=_policy(),
            audit=_audit(),
        )


def test_compiler_rejects_policy_budget_and_agent_schema_bypasses() -> None:
    compiler = DatasetBuildPlanCompiler(_registry())
    plan = _plan(_ref("factory-run"))

    disallowed_policy = FactoryRunPolicyV2.create(
        policy_id="factory-run-policy://disallowed-task",
        allowed_task_kinds=("trace-extraction",),
        max_transitions=256,
        max_plan_revisions=4,
        max_agent_attempts=2,
        max_model_requests=10,
        max_model_tokens=100_000,
        max_cost_micro_usd=1_000_000,
        audit=_audit(),
    )
    with pytest.raises(DatasetBuildPlanCompilerError, match="outside run policy"):
        compiler.compile(plan=plan, policy=disallowed_policy, audit=_audit())

    over_budget_policy = FactoryRunPolicyV2.create(
        policy_id="factory-run-policy://over-budget",
        allowed_task_kinds=("task-rewrite", "trace-extraction"),
        max_transitions=256,
        max_plan_revisions=4,
        max_agent_attempts=2,
        max_model_requests=3,
        max_model_tokens=100_000,
        max_cost_micro_usd=1_000_000,
        audit=_audit(),
    )
    with pytest.raises(DatasetBuildPlanCompilerError, match="run budget"):
        compiler.compile(plan=plan, policy=over_budget_policy, audit=_audit())

    attempt_policy = FactoryRunPolicyV2.create(
        policy_id="factory-run-policy://attempt-limit",
        allowed_task_kinds=("task-rewrite", "trace-extraction"),
        max_transitions=256,
        max_plan_revisions=4,
        max_agent_attempts=1,
        max_model_requests=10,
        max_model_tokens=100_000,
        max_cost_micro_usd=1_000_000,
        audit=_audit(),
    )
    with pytest.raises(DatasetBuildPlanCompilerError, match="attempt policy"):
        compiler.compile(plan=plan, policy=attempt_policy, audit=_audit())

    extract = next(task for task in plan.tasks if task.task_key == "extract")
    rewrite = next(task for task in plan.tasks if task.task_key == "rewrite")
    unsupported_capability = extract.model_copy(
        update={
            "required_capability_ids": (
                "agent-capability://trace-extraction",
                "agent-capability://unsupported",
            )
        }
    )
    with pytest.raises(DatasetBuildPlanCompilerError, match="unsupported Agent capability"):
        compiler.compile(
            plan=_plan_with_tasks(
                plan,
                suffix="unsupported-capability",
                tasks=(unsupported_capability, rewrite),
            ),
            policy=_policy(),
            audit=_audit(),
        )

    incompatible_input = extract.model_copy(update={"input_object_types": ("unsupported-input",)})
    with pytest.raises(DatasetBuildPlanCompilerError, match="input schema"):
        compiler.compile(
            plan=_plan_with_tasks(
                plan,
                suffix="unsupported-input",
                tasks=(incompatible_input, rewrite),
            ),
            policy=_policy(),
            audit=_audit(),
        )

    incompatible_output = rewrite.model_copy(update={"output_object_types": ("unsupported-output",)})
    with pytest.raises(DatasetBuildPlanCompilerError, match="output schema"):
        compiler.compile(
            plan=_plan_with_tasks(
                plan,
                suffix="unsupported-output",
                tasks=(extract, incompatible_output),
            ),
            policy=_policy(),
            audit=_audit(),
        )


def test_registry_rejects_definition_capability_and_write_overlap() -> None:
    capability, definition = _definition(
        "trace-extraction",
        "evaluation-requirement-spec",
        "extracted-user-prompt",
    )
    with pytest.raises(AgentRegistryError, match="unknown capability"):
        AgentRegistry(
            capabilities=(),
            definitions=(definition,),
        )
    unsupported_tool = AgentDefinitionV2.create(
        agent_definition_id="agent-definition://unsupported-tool",
        agent_role="unsupported-tool-agent",
        agent_version="v1",
        capability_refs=(capability.to_ref(),),
        prompt_template_ref=_ref("prompt-template", "trace-extraction"),
        model_policy_ref=_ref("model-routing-policy"),
        tool_ids=("unregistered-tool",),
        data_purpose="evaluation-dataset-construction",
        allowed_data_classifications=("RESTRICTED_TRACE_DERIVED",),
        validator_refs=(_ref("validator", "trace-extraction"),),
        max_attempts=2,
        max_model_requests=2,
        max_model_tokens=4096,
        max_cost_micro_usd=100_000,
        workspace_isolated=True,
        network_allowed=False,
        audit=_audit(),
    )
    with pytest.raises(AgentRegistryError, match="unsupported tool"):
        AgentRegistry(
            capabilities=(capability,),
            definitions=(unsupported_tool,),
        )
    duplicate = AgentDefinitionV2.create(
        agent_definition_id="agent-definition://duplicate",
        agent_role="other-trace-extraction-agent",
        agent_version="v1",
        capability_refs=(capability.to_ref(),),
        prompt_template_ref=_ref("prompt-template", "trace-extraction"),
        model_policy_ref=_ref("model-routing-policy"),
        tool_ids=("trace-extraction-tool",),
        data_purpose="evaluation-dataset-construction",
        allowed_data_classifications=("RESTRICTED_TRACE_DERIVED",),
        validator_refs=(_ref("validator", "trace-extraction"),),
        max_attempts=2,
        max_model_requests=2,
        max_model_tokens=4096,
        max_cost_micro_usd=100_000,
        workspace_isolated=True,
        network_allowed=False,
        audit=_audit(),
    )
    with pytest.raises(AgentRegistryError, match="write ownership"):
        AgentRegistry(
            capabilities=(capability,),
            definitions=(definition, duplicate),
        )


def test_supervisor_fences_workers_and_releases_dependency(tmp_path: Path) -> None:
    store, run_id = _control_store(tmp_path)
    registry = _registry()
    current = store.get_run(run_id)
    plan = _plan(current.to_ref())
    compiled = DatasetBuildPlanCompiler(registry).compile(
        plan=plan,
        policy=_policy(),
        audit=_audit(),
    )
    store.commit_plan(
        run_id=run_id,
        expected_run_version=0,
        plan=plan,
        compiled_plan=compiled,
        audit=_audit(),
        idempotency_key="commit-plan",
    )
    supervisor = _supervisor(
        tmp_path,
        store,
        registry,
    )
    tasks = supervisor.materialize(
        run_id=run_id,
        compiled_plan=compiled,
        audit=_audit(),
    )

    ready = supervisor.dispatch_ready(run_id)
    assert tuple(task.plan_task_key for task in ready) == ("extract",)
    extract = next(task for task in tasks if task.plan_task_key == "extract")
    lease = supervisor.acquire(
        extract.agent_task_id,
        AgentWorkerRef(worker_id="worker://one"),
        lease_duration=timedelta(minutes=5),
    )
    stale = lease.__class__(
        agent_task_ref=lease.agent_task_ref,
        agent_definition_ref=lease.agent_definition_ref,
        lease_id=lease.lease_id,
        worker=lease.worker,
        attempt=lease.attempt,
        fencing_token=lease.fencing_token + 1,
        acquired_at=lease.acquired_at,
        expires_at=lease.expires_at,
    )
    workspace = supervisor.workspace_manager.open(
        run_ref=extract.run_ref,
        lease=lease,
        audit=_audit(),
    )
    (workspace.path / "result.json").write_text(
        "{}",
        encoding="utf-8",
    )
    workspace_receipt = supervisor.workspace_manager.commit(
        workspace,
        audit=_audit(),
    )
    envelope = AgentResultEnvelopeV2.create(
        result_id="agent-result-envelope://extract",
        task_ref=extract.to_ref(),
        agent_definition_ref=lease.agent_definition_ref,
        attempt=1,
        lease_id=lease.lease_id,
        fencing_token=lease.fencing_token,
        workspace_receipt_ref=workspace_receipt.to_ref(),
        outcome=AgentTaskOutcomeV2.SUCCEEDED,
        output_refs=(_ref("extracted-user-prompt"),),
        delegated_dataset_job_result_refs=(),
        gateway_receipt_refs=(_ref("gateway-receipt"),),
        validator_result_refs=(_ref("validator-result"),),
        failure_code=None,
        safe_metrics=(),
        audit=_audit(),
    )
    active_envelope = _result_envelope(
        extract,
        lease,
        workspace.receipt.to_ref(),
    )
    with pytest.raises(StaleAgentLeaseError):
        supervisor.complete(lease=stale, envelope=envelope)
    with pytest.raises(
        ExecutionSupervisorError,
        match="workspace receipt",
    ):
        supervisor.complete(
            lease=lease,
            envelope=active_envelope,
        )
    assert supervisor.complete(lease=lease, envelope=envelope) == envelope
    with sqlite3.connect(store.path) as connection:
        assert connection.execute(
            """
            SELECT COUNT(*) FROM factory_control_outbox
            WHERE event_type = 'agent-result-completed'
            """
        ).fetchone() == (1,)
    assert tuple(task.plan_task_key for task in supervisor.dispatch_ready(run_id)) == ("rewrite",)


def test_two_workers_acquire_one_fenced_authority(tmp_path: Path) -> None:
    store, run_id = _control_store(tmp_path)
    registry = _registry()
    current = store.get_run(run_id)
    plan = _plan(current.to_ref())
    compiled = DatasetBuildPlanCompiler(registry).compile(
        plan=plan,
        policy=_policy(),
        audit=_audit(),
    )
    store.commit_plan(
        run_id=run_id,
        expected_run_version=0,
        plan=plan,
        compiled_plan=compiled,
        audit=_audit(),
        idempotency_key="commit-plan",
    )
    supervisor = _supervisor(
        tmp_path,
        store,
        registry,
    )
    extract = next(
        task
        for task in supervisor.materialize(
            run_id=run_id,
            compiled_plan=compiled,
            audit=_audit(),
        )
        if task.plan_task_key == "extract"
    )

    def acquire(worker: str) -> str:
        try:
            return supervisor.acquire(
                extract.agent_task_id,
                AgentWorkerRef(worker_id=worker),
                lease_duration=timedelta(minutes=5),
            ).lease_id
        except StaleAgentLeaseError:
            return "CONFLICT"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(acquire, ("worker://one", "worker://two")))
    assert results.count("CONFLICT") == 1


def test_supervisor_result_head_and_outbox_roll_back_together(
    tmp_path: Path,
) -> None:
    clock = MutableClock(NOW)
    supervisor, task = _materialized_extract(tmp_path, clock)
    lease = supervisor.acquire(
        task.agent_task_id,
        AgentWorkerRef(worker_id="worker://result-fault"),
        lease_duration=timedelta(minutes=5),
    )
    handle = supervisor.workspace_manager.open(
        run_ref=task.run_ref,
        lease=lease,
        audit=_audit(),
    )
    (handle.path / "result.json").write_text(
        "{}",
        encoding="utf-8",
    )
    committed = supervisor.workspace_manager.commit(
        handle,
        audit=_audit(),
    )
    envelope = _result_envelope(
        task,
        lease,
        committed.to_ref(),
    )
    with sqlite3.connect(supervisor.store.path) as connection:
        connection.execute(
            """
            CREATE TRIGGER fail_agent_result_outbox
            BEFORE INSERT ON factory_control_outbox
            WHEN NEW.event_type = 'agent-result-completed'
            BEGIN
                SELECT RAISE(ABORT, 'injected outbox failure');
            END
            """
        )

    with pytest.raises(
        StaleAgentLeaseError,
        match="authority raced",
    ):
        supervisor.complete(lease=lease, envelope=envelope)
    with sqlite3.connect(supervisor.store.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM agent_results").fetchone() == (0,)
        assert connection.execute(
            """
            SELECT state FROM agent_work_current_heads
            WHERE agent_task_id = ?
            """,
            (task.agent_task_id,),
        ).fetchone() == ("ACTIVE",)
        assert connection.execute(
            """
            SELECT COUNT(*) FROM factory_control_outbox
            WHERE event_type = 'agent-result-completed'
            """
        ).fetchone() == (0,)
        connection.execute("DROP TRIGGER fail_agent_result_outbox")

    assert supervisor.complete(lease=lease, envelope=envelope) == envelope


def test_supervisor_heartbeat_expiry_retry_and_exhaustion(
    tmp_path: Path,
) -> None:
    clock = MutableClock(NOW)
    supervisor, task = _materialized_extract(tmp_path, clock)
    worker = AgentWorkerRef(worker_id="worker://retry")
    lease = supervisor.acquire(
        task.agent_task_id,
        worker,
        lease_duration=timedelta(seconds=10),
        idempotency_key="acquire-attempt-one",
    )
    replay = supervisor.acquire(
        task.agent_task_id,
        worker,
        lease_duration=timedelta(seconds=10),
        idempotency_key="acquire-attempt-one",
    )
    assert replay == lease

    heartbeat = supervisor.heartbeat(
        lease=lease,
        worker=worker,
        expected_event_version=1,
        extension=timedelta(seconds=30),
        idempotency_key="heartbeat-attempt-one",
    )
    assert heartbeat.event_kind is AgentWorkEventKindV2.HEARTBEAT
    assert heartbeat.event_version == 2
    assert (
        supervisor.heartbeat(
            lease=lease,
            worker=worker,
            expected_event_version=1,
            extension=timedelta(seconds=30),
            idempotency_key="heartbeat-attempt-one",
        )
        == heartbeat
    )

    clock.advance(timedelta(seconds=11))
    with pytest.raises(
        ExecutionSupervisorError,
        match="not expired",
    ):
        supervisor.expire(
            lease=lease,
            retry_delay=timedelta(seconds=5),
            audit=_audit(),
            idempotency_key="early-expire",
        )
    clock.advance(timedelta(seconds=20))
    precommitted_quarantine = supervisor.workspace_manager.quarantine_lease(
        run_ref=task.run_ref,
        lease=lease,
        reason_code="LEASE_EXPIRED",
        audit=_audit(),
    )
    decision = supervisor.expire(
        lease=lease,
        retry_delay=timedelta(seconds=5),
        audit=_audit(),
        idempotency_key="expire-attempt-one",
    )
    assert decision.workspace_receipt_ref == precommitted_quarantine.to_ref()
    assert decision.decision is AgentRetryDecisionKindV2.RETRY_SCHEDULED
    assert decision.next_attempt == 2
    assert (
        supervisor.expire(
            lease=lease,
            retry_delay=timedelta(seconds=5),
            audit=_audit(),
            idempotency_key="expire-attempt-one",
        )
        == decision
    )
    assert supervisor.dispatch_ready("factory-run://core") == ()

    with pytest.raises(StaleAgentLeaseError, match="not eligible"):
        supervisor.acquire(
            task.agent_task_id,
            worker,
            lease_duration=timedelta(seconds=10),
            idempotency_key="acquire-attempt-two-early",
        )
    clock.advance(timedelta(seconds=5))
    assert tuple(value.agent_task_id for value in supervisor.dispatch_ready("factory-run://core")) == (
        task.agent_task_id,
    )
    second = supervisor.acquire(
        task.agent_task_id,
        worker,
        lease_duration=timedelta(seconds=10),
        idempotency_key="acquire-attempt-two",
    )
    assert second.attempt == 2
    assert second.fencing_token == 2

    clock.advance(timedelta(seconds=11))
    exhausted = supervisor.expire(
        lease=second,
        retry_delay=timedelta(0),
        audit=_audit(),
        idempotency_key="expire-attempt-two",
    )
    assert exhausted.decision is AgentRetryDecisionKindV2.EXHAUSTED
    assert exhausted.next_attempt is None
    assert supervisor.dispatch_ready("factory-run://core") == ()
    with pytest.raises(StaleAgentLeaseError, match="terminal"):
        supervisor.acquire(
            task.agent_task_id,
            worker,
            lease_duration=timedelta(seconds=10),
        )


def test_supervisor_cancel_quarantines_workspace_and_fences_late_result(
    tmp_path: Path,
) -> None:
    clock = MutableClock(NOW)
    supervisor, task = _materialized_extract(tmp_path, clock)
    worker = AgentWorkerRef(worker_id="worker://cancel")
    lease = supervisor.acquire(
        task.agent_task_id,
        worker,
        lease_duration=timedelta(minutes=5),
    )
    requested = supervisor.request_cancel(
        task_id=task.agent_task_id,
        requested_by="user://operator",
        reason_code="USER_CANCELLED",
        audit=_audit(),
        idempotency_key="request-cancel",
    )
    assert requested.state is AgentCancellationStateV2.REQUESTED
    assert supervisor.dispatch_ready("factory-run://core") == ()

    active = supervisor.workspace_manager.open(
        run_ref=task.run_ref,
        lease=lease,
        audit=_audit(),
    )
    late = _result_envelope(
        task,
        lease,
        active.receipt.to_ref(),
    )
    with pytest.raises(StaleAgentLeaseError):
        supervisor.complete(lease=lease, envelope=late)

    cancelled = supervisor.acknowledge_cancel(
        lease=lease,
        worker=worker,
        audit=_audit(),
        idempotency_key="acknowledge-cancel",
    )
    assert cancelled.state is AgentCancellationStateV2.CANCELLED
    assert cancelled.workspace_receipt_ref is not None
    assert not active.path.exists()
    assert supervisor.dispatch_ready("factory-run://core") == ()
    assert (
        supervisor.acknowledge_cancel(
            lease=lease,
            worker=worker,
            audit=_audit(),
            idempotency_key="acknowledge-cancel",
        )
        == cancelled
    )


def test_supervisor_changed_idempotency_request_is_rejected(
    tmp_path: Path,
) -> None:
    clock = MutableClock(NOW)
    supervisor, task = _materialized_extract(tmp_path, clock)
    supervisor.acquire(
        task.agent_task_id,
        AgentWorkerRef(worker_id="worker://one"),
        lease_duration=timedelta(minutes=5),
        idempotency_key="same-acquire-key",
    )

    with pytest.raises(
        ExecutionSupervisorError,
        match="idempotency",
    ):
        supervisor.acquire(
            task.agent_task_id,
            AgentWorkerRef(worker_id="worker://two"),
            lease_duration=timedelta(minutes=5),
            idempotency_key="same-acquire-key",
        )
