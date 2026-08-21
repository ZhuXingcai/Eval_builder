from __future__ import annotations

from pathlib import Path

import pytest
from test_artifact_execution import (
    _audit,
    _ControlledExecutionFacade,
    _definition,
    _execution_plan,
    _ref,
    _routing_source,
)
from test_artifact_results import (
    _planning_result,
    _result_audit,
)

from eval_factory.agent_system.attachment_subgraph import (
    AttachmentR5SubgraphAdapter,
    AttachmentSubgraphError,
)
from eval_factory.attachment_planning import (
    ArtifactExecutionPlanningOutcome,
)
from eval_factory.contracts.agent_system_v2 import (
    AgentResultEnvelopeV2,
    AgentTaskOutcomeV2,
    AgentTaskStatusV2,
    AgentTaskV2,
    AttachmentGenerationPlanV2,
    AttachmentMockWorkV2,
    AttachmentSubgraphOutcomeV2,
)
from eval_factory.contracts.attachment_v2 import (
    artifact_execution_group_ref,
)
from eval_factory.contracts.core import ObjectRef

HASH = "a" * 64


def _object_ref(value) -> ObjectRef:
    return ObjectRef(
        object_type=value.object_type,
        object_id=value.object_id,
        object_version=value.object_version,
        object_sha256=value.object_sha256,
    )


def _attachment_plan(execution_plan) -> AttachmentGenerationPlanV2:
    works = tuple(
        AttachmentMockWorkV2(
            work_key=f"group-{index:02d}",
            artifact_group_ref=_object_ref(artifact_execution_group_ref(group)),
            artifact_ids=tuple(sorted(unit.artifact_id for unit in group.units)),
            agent_role="attachment-mock-agent",
            dependency_work_keys=(),
            input_object_types=("attachment-planning-context",),
            output_object_types=("attachment-group-result",),
            required_capability_ids=("agent-capability://attachment-mock",),
            allowed_tool_ids=("attachment-execution",),
            data_purposes=("attachment-production",),
            data_classifications=("RESTRICTED_TRACE_DERIVED",),
            workspace_policy_ref=_ref(
                "agent-workspace-policy",
                "isolated",
            ),
            acceptance_check_refs=(_ref("acceptance-check", f"group-{index:02d}"),),
            max_attempts=2,
            max_model_requests=0,
            max_model_tokens=0,
            max_cost_micro_usd=0,
        )
        for index, group in enumerate(execution_plan.groups)
    )
    return AttachmentGenerationPlanV2.create(
        plan_id="attachment-generation-plan://r5-adapter",
        run_ref=_ref("factory-run", "r5-adapter"),
        plan_version=1,
        predecessor_plan_ref=None,
        producer_task_view_ref=_ref(
            "producer-task-view",
            "current",
        ),
        evidence_bundle_ref=_ref(
            "evidence-bundle",
            "current",
            version="v1",
        ),
        attachment_planning_context_ref=_ref(
            "attachment-planning-context",
            "current",
        ),
        works=works,
        max_parallel_groups=len(works),
        quality_policy_ref=_ref(
            "attachment-quality-policy",
            "current",
        ),
        solvability_policy_ref=_ref(
            "solvability-policy",
            "current",
        ),
        total_model_requests=0,
        total_model_tokens=0,
        total_cost_micro_usd=0,
        audit=_result_audit(),
    )


@pytest.mark.asyncio
async def test_r5_adapter_runs_two_real_groups_in_parallel_and_replays(
    tmp_path: Path,
) -> None:
    artifacts = (
        ("artifact://a", "inputs/a.txt"),
        ("artifact://b", "inputs/b.txt"),
    )
    execution_plan = await _execution_plan(
        tmp_path,
        artifacts,
        (
            _definition("artifact://a"),
            _definition("artifact://b"),
        ),
    )
    routing_request, routing_result = _routing_source(artifacts)
    facade = _ControlledExecutionFacade(expected_parallelism=2)
    adapter = AttachmentR5SubgraphAdapter()
    plan = _attachment_plan(execution_plan)

    first = await adapter.execute(
        plan=plan,
        routing_request=routing_request,
        routing_result=routing_result,
        execution_result=_planning_result(execution_plan),
        facade=facade,
        audit=_result_audit(),
    )
    calls_after_first = facade.calls.copy()
    second = await adapter.execute(
        plan=plan,
        routing_request=routing_request,
        routing_result=routing_result,
        execution_result=_planning_result(execution_plan),
        facade=facade,
        audit=_result_audit(),
        prior_batch=first.execution_batch,
    )

    assert facade.max_active == 2
    assert facade.calls == calls_after_first
    assert second.execution_batch.succeeded_artifact_ids == ("artifact://a", "artifact://b")
    assert (
        second.reconstruction_result.candidate_output_refs
        == first.reconstruction_result.candidate_output_refs
    )


@pytest.mark.asyncio
async def test_r5_adapter_rejects_plan_group_drift(
    tmp_path: Path,
) -> None:
    artifacts = (("artifact://a", "inputs/a.txt"),)
    execution_plan = await _execution_plan(
        tmp_path,
        artifacts,
        (_definition("artifact://a"),),
    )
    routing_request, routing_result = _routing_source(artifacts)
    plan = _attachment_plan(execution_plan)
    drifted = plan.model_copy(
        update={"works": (plan.works[0].model_copy(update={"artifact_ids": ("artifact://other",)}),)}
    )

    with pytest.raises(
        AttachmentSubgraphError,
        match="differs",
    ):
        await AttachmentR5SubgraphAdapter().execute(
            plan=drifted,
            routing_request=routing_request,
            routing_result=routing_result,
            execution_result=_planning_result(execution_plan),
            facade=_ControlledExecutionFacade(),
            audit=_audit(),
        )


@pytest.mark.asyncio
async def test_r5_adapter_rejects_non_planned_execution(
    tmp_path: Path,
) -> None:
    artifacts = (("artifact://a", "inputs/a.txt"),)
    execution_plan = await _execution_plan(
        tmp_path,
        artifacts,
        (_definition("artifact://a"),),
    )
    routing_request, routing_result = _routing_source(artifacts)
    execution_result = _planning_result(execution_plan).model_copy(
        update={
            "outcome": ArtifactExecutionPlanningOutcome.NOT_REQUIRED,
            "world_ledger_snapshot": None,
            "execution_plan": None,
        }
    )

    with pytest.raises(
        AttachmentSubgraphError,
        match="requires a current",
    ):
        await AttachmentR5SubgraphAdapter().execute(
            plan=_attachment_plan(execution_plan),
            routing_request=routing_request,
            routing_result=routing_result,
            execution_result=execution_result,
            facade=_ControlledExecutionFacade(),
            audit=_audit(),
        )


def test_r5_adapter_join_has_exact_work_partitions() -> None:
    plan = AttachmentGenerationPlanV2.create(
        plan_id="attachment-generation-plan://join",
        run_ref=_ref("factory-run", "join"),
        plan_version=1,
        predecessor_plan_ref=None,
        producer_task_view_ref=_ref(
            "producer-task-view",
            "join",
        ),
        evidence_bundle_ref=_ref(
            "evidence-bundle",
            "join",
            version="v1",
        ),
        attachment_planning_context_ref=_ref(
            "attachment-planning-context",
            "join",
        ),
        works=(
            _join_work("group-a", "artifact://a"),
            _join_work("group-b", "artifact://b"),
        ),
        max_parallel_groups=2,
        quality_policy_ref=_ref(
            "attachment-quality-policy",
            "join",
        ),
        solvability_policy_ref=_ref(
            "solvability-policy",
            "join",
        ),
        total_model_requests=0,
        total_model_tokens=0,
        total_cost_micro_usd=0,
        audit=_audit(),
    )
    tasks = tuple(_join_task(plan, work) for work in plan.works)
    envelopes = (
        _join_envelope(tasks[0], AgentTaskOutcomeV2.SUCCEEDED),
        _join_envelope(
            tasks[1],
            AgentTaskOutcomeV2.RETRYABLE_FAILURE,
        ),
    )

    result = AttachmentR5SubgraphAdapter().join(
        plan=plan,
        tasks=tasks,
        envelopes=envelopes,
        audit=_audit(),
    )

    assert result.outcome is AttachmentSubgraphOutcomeV2.PARTIAL
    assert result.succeeded_work_result_refs == (envelopes[0].to_ref(),)
    assert result.retryable_work_result_refs == (envelopes[1].to_ref(),)


def _join_work(
    key: str,
    artifact_id: str,
) -> AttachmentMockWorkV2:
    return AttachmentMockWorkV2(
        work_key=key,
        artifact_group_ref=_ref(
            "artifact-execution-group",
            key,
        ),
        artifact_ids=(artifact_id,),
        agent_role="attachment-mock-agent",
        dependency_work_keys=(),
        input_object_types=("attachment-planning-context",),
        output_object_types=("attachment-group-result",),
        required_capability_ids=("agent-capability://attachment-mock",),
        allowed_tool_ids=("attachment-execution",),
        data_purposes=("attachment-production",),
        data_classifications=("RESTRICTED_TRACE_DERIVED",),
        workspace_policy_ref=_ref(
            "agent-workspace-policy",
            key,
        ),
        acceptance_check_refs=(_ref("acceptance-check", key),),
        max_attempts=2,
        max_model_requests=0,
        max_model_tokens=0,
        max_cost_micro_usd=0,
    )


def _join_task(
    plan: AttachmentGenerationPlanV2,
    work: AttachmentMockWorkV2,
) -> AgentTaskV2:
    return AgentTaskV2.create(
        agent_task_id=f"agent-task://{work.work_key}",
        run_ref=plan.run_ref,
        compiled_plan_ref=_ref(
            "compiled-attachment-generation-plan",
            "join",
        ),
        plan_task_key=work.work_key,
        task_kind="attachment-mock",
        agent_definition_ref=_ref(
            "agent-definition",
            work.work_key,
        ),
        input_refs=tuple(
            sorted(
                (
                    plan.attachment_planning_context_ref,
                    work.artifact_group_ref,
                ),
                key=lambda value: (
                    value.object_type,
                    value.object_id,
                    value.object_version,
                    value.object_sha256,
                ),
            )
        ),
        expected_output_object_types=("attachment-group-result",),
        required_capability_refs=(_ref("agent-capability", work.work_key),),
        status=AgentTaskStatusV2.READY,
        attempt=0,
        max_attempts=2,
        audit=_audit(),
    )


def _join_envelope(
    task: AgentTaskV2,
    outcome: AgentTaskOutcomeV2,
) -> AgentResultEnvelopeV2:
    succeeded = outcome is AgentTaskOutcomeV2.SUCCEEDED
    return AgentResultEnvelopeV2.create(
        result_id=f"agent-result-envelope://{task.plan_task_key}",
        task_ref=task.to_ref(),
        agent_definition_ref=task.agent_definition_ref,
        attempt=1,
        lease_id=f"agent-lease://{task.plan_task_key}",
        fencing_token=1,
        workspace_receipt_ref=_ref(
            "agent-workspace-receipt",
            task.plan_task_key,
        ),
        outcome=outcome,
        output_refs=(
            _ref(
                "attachment-group-result",
                task.plan_task_key,
            ),
        )
        if succeeded
        else (),
        delegated_dataset_job_result_refs=(),
        gateway_receipt_refs=(),
        validator_result_refs=(_ref("validator-result", task.plan_task_key),),
        failure_code=None if succeeded else "GROUP_RETRYABLE",
        safe_metrics=(),
        audit=_audit(),
    )
