from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest
from test_attachment_agent_planning import (
    _audit,
    _policy,
    _ref,
    _work,
)
from test_domain_plan_review import (
    RUN_ID,
)
from test_domain_plan_review import (
    _setup as _domain_plan_setup,
)

from eval_factory.agent_system.attachment_planning import (
    AttachmentGenerationPlanCompiler,
)
from eval_factory.agent_system.attachment_registry import (
    AttachmentAgentRegistryConfig,
    build_attachment_agent_registry,
)
from eval_factory.agent_system.supervisor import (
    AgentWorkerRef,
    ExecutionSupervisor,
    ExecutionSupervisorError,
)
from eval_factory.agent_system.workspace import AgentWorkspaceManager
from eval_factory.contracts.agent_system_v2 import (
    AgentResultEnvelopeV2,
    AgentTaskOutcomeV2,
    AttachmentGenerationPlanV2,
    PlanKindV2,
)
from eval_factory.contracts.core import ObjectRef


def _registry(suffix: str = "current"):
    return build_attachment_agent_registry(
        config=AttachmentAgentRegistryConfig(
            mock_prompt_ref=_ref(
                "prompt-template",
                f"attachment-mock-{suffix}",
            ),
            quality_prompt_ref=_ref(
                "prompt-template",
                f"attachment-quality-{suffix}",
            ),
            solvability_prompt_ref=_ref(
                "prompt-template",
                f"attachment-solvability-{suffix}",
            ),
            mock_model_policy_ref=_ref(
                "model-routing-policy",
                f"attachment-mock-{suffix}",
            ),
            quality_model_policy_ref=_ref(
                "model-routing-policy",
                f"attachment-quality-{suffix}",
            ),
            solvability_model_policy_ref=_ref(
                "model-routing-policy",
                f"attachment-solvability-{suffix}",
            ),
        ),
        audit=_audit(),
    )


def _plan(
    run_ref: ObjectRef,
    *,
    suffix: str = "current",
) -> AttachmentGenerationPlanV2:
    works = (
        _supervised_work("work-a"),
        _supervised_work(
            "work-b",
            dependencies=("work-a",),
        ),
    )
    return AttachmentGenerationPlanV2.create(
        plan_id=f"attachment-generation-plan://{suffix}",
        run_ref=run_ref,
        plan_version=1,
        predecessor_plan_ref=None,
        producer_task_view_ref=_ref(
            "producer-task-view",
            suffix,
        ),
        evidence_bundle_ref=_ref(
            "evidence-bundle",
            suffix,
            version="v1",
        ),
        attachment_planning_context_ref=_ref(
            "attachment-planning-context",
            suffix,
        ),
        works=works,
        max_parallel_groups=2,
        quality_policy_ref=_ref(
            "attachment-quality-policy",
            suffix,
        ),
        solvability_policy_ref=_ref(
            "solvability-policy",
            suffix,
        ),
        total_model_requests=sum(work.max_model_requests for work in works),
        total_model_tokens=sum(work.max_model_tokens for work in works),
        total_cost_micro_usd=sum(work.max_cost_micro_usd for work in works),
        audit=_audit(),
    )


def _supervised_work(
    key: str,
    *,
    dependencies: tuple[str, ...] = (),
):
    return _work(
        key,
        dependencies=dependencies,
    ).model_copy(
        update={
            "max_model_requests": 0,
            "max_model_tokens": 0,
            "max_cost_micro_usd": 0,
        }
    )


def _materialized(
    tmp_path: Path,
):
    _, store, _ = _domain_plan_setup(tmp_path)
    run = store.get_run(RUN_ID)
    policy = store.get_policy(run.policy_ref.object_id)
    registry = _registry()
    plan = _plan(run.to_ref())
    compiled = AttachmentGenerationPlanCompiler(registry).compile(
        plan=plan,
        policy=policy,
        audit=_audit(),
    )
    store.commit_domain_plan(
        run_id=run.run_id,
        expected_run_version=run.run_version,
        plan_kind=PlanKindV2.ATTACHMENT_GENERATION,
        plan=plan,
        compiled_plan=compiled,
        audit=_audit(),
        idempotency_key="commit-attachment-plan",
    )
    supervisor = ExecutionSupervisor(
        store,
        registry,
        workspace_manager=AgentWorkspaceManager(tmp_path / "agent-workspaces"),
    )
    tasks = supervisor.materialize_attachment(
        run_id=run.run_id,
        compiled_plan=compiled,
        audit=_audit(),
    )
    return store, supervisor, plan, compiled, tasks


def test_supervisor_materializes_attachment_tasks_and_dependency(
    tmp_path: Path,
) -> None:
    _, supervisor, plan, compiled, tasks = _materialized(tmp_path)

    assert tuple(task.plan_task_key for task in tasks) == (
        "work-a",
        "work-b",
    )
    first, second = tasks
    assert first.compiled_plan_ref == compiled.to_ref()
    assert first.run_ref == supervisor.store.get_run(RUN_ID).to_ref()
    assert plan.attachment_planning_context_ref in first.input_refs
    assert plan.works[0].artifact_group_ref in first.input_refs
    assert first.to_ref() in second.input_refs
    assert tuple(
        task.plan_task_key
        for task in supervisor.dispatch_ready(
            RUN_ID,
            compiled_plan_ref=compiled.to_ref(),
        )
    ) == ("work-a",)

    worker = AgentWorkerRef(worker_id="worker://attachment-a")
    lease = supervisor.acquire(
        first.agent_task_id,
        worker,
        lease_duration=timedelta(minutes=5),
    )
    handle = supervisor.workspace_manager.open(
        run_ref=first.run_ref,
        lease=lease,
        audit=_audit(),
    )
    (handle.path / "group-result.json").write_text(
        "{}",
        encoding="utf-8",
    )
    committed = supervisor.workspace_manager.commit(
        handle,
        audit=_audit(),
    )
    envelope = AgentResultEnvelopeV2.create(
        result_id="agent-result-envelope://attachment-work-a",
        task_ref=first.to_ref(),
        agent_definition_ref=lease.agent_definition_ref,
        attempt=lease.attempt,
        lease_id=lease.lease_id,
        fencing_token=lease.fencing_token,
        workspace_receipt_ref=committed.to_ref(),
        outcome=AgentTaskOutcomeV2.SUCCEEDED,
        output_refs=(_ref("attachment-group-result", "work-a"),),
        delegated_dataset_job_result_refs=(),
        gateway_receipt_refs=(_ref("gateway-receipt", "attachment-work-a"),),
        validator_result_refs=(_ref("validator-result", "attachment-work-a"),),
        failure_code=None,
        safe_metrics=(),
        audit=_audit(),
    )
    supervisor.complete(
        lease=lease,
        envelope=envelope,
    )

    assert tuple(
        task.plan_task_key
        for task in supervisor.dispatch_ready(
            RUN_ID,
            compiled_plan_ref=compiled.to_ref(),
        )
    ) == ("work-b",)


def test_supervisor_attachment_materialization_replays_exactly(
    tmp_path: Path,
) -> None:
    _, supervisor, _, compiled, tasks = _materialized(tmp_path)

    replay = supervisor.materialize_attachment(
        run_id=RUN_ID,
        compiled_plan=compiled,
        audit=_audit(),
    )

    assert replay == tasks


def test_supervisor_rejects_stale_attachment_plan(
    tmp_path: Path,
) -> None:
    store, supervisor, _, _, _ = _materialized(tmp_path)
    current = store.get_run(RUN_ID)
    other_plan = _plan(current.to_ref(), suffix="other")
    other_compiled = AttachmentGenerationPlanCompiler(_registry()).compile(
        plan=other_plan,
        policy=_policy(),
        audit=_audit(),
    )

    with pytest.raises(
        ExecutionSupervisorError,
        match="stale attachment plan",
    ):
        supervisor.materialize_attachment(
            run_id=RUN_ID,
            compiled_plan=other_compiled,
            audit=_audit(),
        )


def test_supervisor_rejects_attachment_registry_assignment_drift(
    tmp_path: Path,
) -> None:
    store, _, _, compiled, _ = _materialized(tmp_path)
    supervisor = ExecutionSupervisor(
        store,
        _registry("changed"),
        workspace_manager=AgentWorkspaceManager(tmp_path / "changed-workspaces"),
    )

    with pytest.raises(
        ExecutionSupervisorError,
        match="assignment differs",
    ):
        supervisor.materialize_attachment(
            run_id=RUN_ID,
            compiled_plan=compiled,
            audit=_audit(),
        )
