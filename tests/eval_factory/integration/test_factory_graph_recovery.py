from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from eval_factory.agent_system.dataset_graph_handlers import (
    FactoryDatasetNodeCommand,
    FactoryDatasetNodeCommandRegistry,
)
from eval_factory.agent_system.graph import FactoryGraphState
from eval_factory.agent_system.graph_journal import (
    FactoryGraphJournalStore,
)
from eval_factory.agent_system.graph_runtime import (
    FactoryDatasetGraphRuntime,
)
from eval_factory.agent_system.graph_transition import (
    FactoryGraphTransitionService,
)
from eval_factory.agent_system.plan_review import PlanReviewService
from eval_factory.agent_system.store import FactoryControlStore
from eval_factory.contracts.agent_system_v2 import (
    CompiledDatasetBuildPlanV2,
    DatasetBuildPlanTaskV2,
    DatasetBuildPlanV2,
    EvaluationRequirementSpecV2,
    FactoryRunPolicyV2,
    FactoryRunStatusV2,
    FactoryRunV2,
    PlannerAssessmentV2,
)
from eval_factory.contracts.core import (
    ContractAudit,
    ObjectRef,
    VersionBinding,
)
from eval_factory.contracts.dataset_runtime_v2 import (
    FactoryDatasetNextActionV2,
    FactoryDatasetRunRequestV2,
    FactoryDatasetRunViewV2,
)
from eval_factory.harness.graph_models import (
    GraphCheckpointOutcomeV1,
    GraphCheckpointPhaseV1,
    HarnessGraphCheckpointV1,
    HarnessGraphExecutionBindingV1,
)
from eval_factory.harness.session_models import SessionEventV1

HASH = "a" * 64
NODES = (
    "requirement_planner",
    "review_global_plan",
    "compile_global_plan",
    "dispatch_ready_work",
    "validate_core_vertical",
    "planner_assessment",
    "assemble_core_output",
    "review_final_delivery",
    "export_core_output",
)


def _audit() -> ContractAudit:
    return ContractAudit(
        created_at=datetime(2026, 8, 20, tzinfo=UTC),
        created_by="factory-graph-integration-test",
        governing_versions=(
            VersionBinding(
                component="eval-harness-stage4",
                version="v1",
                sha256=HASH,
            ),
        ),
    )


def _ref(
    object_type: str,
    *,
    version: str = "v2",
) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://stage4-integration",
        object_version=version,
        object_sha256=HASH,
    )


@dataclass
class _Authority:
    store: FactoryControlStore
    binding: HarnessGraphExecutionBindingV1
    run_id: str

    def resolve_current(
        self,
        reference: ObjectRef,
    ) -> HarnessGraphExecutionBindingV1:
        assert reference == self.binding.to_ref()
        assert self.store.get_run(self.run_id).to_ref() == self.binding.factory_run_ref
        return self.binding

    def capture_current(
        self,
        reference: ObjectRef,
        *,
        audit: ContractAudit,
    ) -> HarnessGraphExecutionBindingV1:
        assert reference == self.binding.to_ref()
        current = self.store.get_run(self.run_id)
        self.binding = HarnessGraphExecutionBindingV1.create(
            binding_id=self.binding.binding_id,
            session_ref=self.binding.session_ref,
            requirement_ref=self.binding.requirement_ref,
            requirement_policy_ref=(self.binding.requirement_policy_ref),
            factory_run_ref=current.to_ref(),
            factory_request_ref=self.binding.factory_request_ref,
            factory_policy_ref=self.binding.factory_policy_ref,
            expected_factory_run_version=current.run_version,
            pack_manifest_ref=self.binding.pack_manifest_ref,
            composition_ref=self.binding.composition_ref,
            blueprint_ref=self.binding.blueprint_ref,
            team_ref=self.binding.team_ref,
            roster_ref=self.binding.roster_ref,
            task_graph_ref=self.binding.task_graph_ref,
            execution_authority_ref=(self.binding.execution_authority_ref),
            team_checkpoint_ref=self.binding.team_checkpoint_ref,
            expected_team_version=self.binding.expected_team_version,
            expected_task_graph_revision=(self.binding.expected_task_graph_revision),
            expected_authority_version=(self.binding.expected_authority_version),
            blackboard_head_refs=self.binding.blackboard_head_refs,
            thread_id=self.binding.thread_id,
            max_transitions=self.binding.max_transitions,
            audit=audit,
        )
        return self.binding


@dataclass
class _SessionReconciler:
    def reconcile(
        self,
        binding_id: str,
        *,
        audit: ContractAudit,
        limit: int = 500,
    ) -> tuple[SessionEventV1, ...]:
        del binding_id, audit, limit
        return ()


class _Continuation:
    def resume(
        self,
        *,
        graph_binding_ref: ObjectRef,
        expected_checkpoint_ref: ObjectRef,
        review_result_ref: ObjectRef,
        idempotency_key: str,
    ) -> HarnessGraphCheckpointV1:
        del (
            graph_binding_ref,
            expected_checkpoint_ref,
            review_result_ref,
            idempotency_key,
        )
        raise AssertionError("continuation is not used before review")


def _assessment(
    state: FactoryGraphState,
    view: FactoryDatasetRunViewV2,
    run: FactoryRunV2,
) -> PlannerAssessmentV2:
    del state, view, run
    raise AssertionError("assessment must not run before review")


def _view(
    *,
    request: FactoryDatasetRunRequestV2,
    store: FactoryControlStore,
    pending: tuple[ObjectRef, ...] = (),
) -> FactoryDatasetRunViewV2:
    run = store.get_run(request.dataset_run_id)
    return FactoryDatasetRunViewV2.create(
        request_ref=request.to_ref(),
        dataset_run_ref=run.to_ref(),
        status=run.status,
        pending_review_refs=pending,
        item_binding_refs=(),
        candidate_count=0,
        rejected_count=0,
        blocked_count=0,
        incomplete_count=0,
        aggregate_result_ref=None,
        delivery_manifest_ref=None,
        next_action=(
            FactoryDatasetNextActionV2.REVIEW_PLAN if pending else FactoryDatasetNextActionV2.ADVANCE
        ),
        audit=_audit(),
    )


@pytest.mark.asyncio
async def test_default_graph_commits_pre_post_and_waits_for_review(
    tmp_path: Path,
) -> None:
    store = FactoryControlStore(tmp_path / "factory.sqlite3")
    policy = FactoryRunPolicyV2.create(
        policy_id="policy.stage4-integration",
        allowed_task_kinds=("planning",),
        max_transitions=64,
        max_plan_revisions=4,
        max_agent_attempts=2,
        max_model_requests=10,
        max_model_tokens=10_000,
        max_cost_micro_usd=1_000_000,
        audit=_audit(),
    )
    requirement = EvaluationRequirementSpecV2.create(
        requirement_spec_id="requirement.stage4-integration",
        run_id="factory-run.stage4-integration",
        source_ref=_ref("evaluation-requirement-source"),
        goals=("Build evaluation data.",),
        constraints=(),
        assumptions=(),
        open_questions=(),
        requirement_version=1,
        audit=_audit(),
    )
    run = store.create_run(
        policy=policy,
        requirement=requirement,
        idempotency_key="create-run",
    )
    request = FactoryDatasetRunRequestV2.create(
        dataset_run_id=run.run_id,
        requirement_spec_ref=requirement.to_ref(),
        manifest_ref=_ref("trace-manifest"),
        source_authorization_ref=_ref(
            "trace-source-authorization",
        ),
        factory_policy_ref=policy.to_ref(),
        pipeline_policy_refs=(_ref("trace-policy"),),
        gateway_registry_refs=(_ref("agent-registry"),),
        output_target_ref=_ref("candidate-output-target"),
        idempotency_key="dataset-run-stage4-integration",
        max_transitions=64,
        audit=_audit(),
    )
    store.commit_dataset_request(
        request,
        idempotency_key="commit-request",
    )
    task = DatasetBuildPlanTaskV2(
        task_key="planning",
        stage="factory",
        task_kind="planning",
        agent_role="planner",
        dependency_task_keys=(),
        input_object_types=("evaluation-requirement-spec",),
        output_object_types=("dataset-build-plan",),
        required_capability_ids=("capability.requirement-planning",),
        acceptance_check_refs=(_ref("acceptance-check"),),
        plan_review_kind=None,
        max_attempts=2,
        max_model_requests=1,
        max_model_tokens=1_000,
        max_cost_micro_usd=100_000,
    )
    plan = DatasetBuildPlanV2.create(
        plan_id="plan.stage4-integration",
        run_ref=run.to_ref(),
        plan_version=1,
        predecessor_plan_ref=None,
        goals=requirement.goals,
        user_constraints=(),
        assumptions=(),
        unresolved_questions=(),
        stage_order=("factory",),
        tasks=(task,),
        required_review_kinds=(),
        total_model_requests=1,
        total_model_tokens=1_000,
        total_cost_micro_usd=100_000,
        audit=_audit(),
    )
    compiled = CompiledDatasetBuildPlanV2.create(
        compiled_plan_id="compiled-plan.stage4-integration",
        source_plan_ref=plan.to_ref(),
        policy_ref=policy.to_ref(),
        tasks=plan.tasks,
        topological_task_keys=(task.task_key,),
        audit=_audit(),
    )
    reviews = PlanReviewService(store)

    async def requirement_command(
        state: FactoryGraphState,
    ) -> FactoryDatasetRunViewV2:
        del state
        current = store.get_run(run.run_id)
        store.commit_plan(
            run_id=run.run_id,
            expected_run_version=current.run_version,
            plan=plan,
            compiled_plan=compiled,
            audit=_audit(),
            idempotency_key="commit-plan",
        )
        return _view(request=request, store=store)

    async def review_command(
        state: FactoryGraphState,
    ) -> FactoryDatasetRunViewV2:
        del state
        review = reviews.open_global_plan(
            run_id=run.run_id,
            title="Dataset build plan",
            summary_lines=("One task.",),
            editable_paths=(),
            warning_codes=(),
            requested_by="reviewer-stage4",
            idempotency_key="open-review",
            audit=_audit(),
        )
        return _view(
            request=request,
            store=store,
            pending=(review.request.to_ref(),),
        )

    async def unexpected(
        state: FactoryGraphState,
    ) -> FactoryDatasetRunViewV2:
        del state
        raise AssertionError("Graph advanced beyond global review")

    commands: dict[str, FactoryDatasetNodeCommand] = {node: unexpected for node in NODES}
    commands["requirement_planner"] = requirement_command
    commands["review_global_plan"] = review_command
    registry = FactoryDatasetNodeCommandRegistry(
        store=store,
        commands=commands,
    )
    binding = HarnessGraphExecutionBindingV1.create(
        binding_id="binding.stage4-integration",
        session_ref=_ref("harness-session", version="v1"),
        requirement_ref=requirement.to_ref(),
        requirement_policy_ref=_ref(
            "harness-requirement-policy",
            version="v1",
        ),
        factory_run_ref=run.to_ref(),
        factory_request_ref=request.to_ref(),
        factory_policy_ref=policy.to_ref(),
        expected_factory_run_version=run.run_version,
        pack_manifest_ref=_ref(
            "eval-pack-manifest",
            version="v1",
        ),
        composition_ref=_ref(
            "harness-composition",
            version="v1",
        ),
        blueprint_ref=_ref(
            "evaluation-blueprint",
            version="v1",
        ),
        team_ref=_ref("agent-team", version="v1"),
        roster_ref=_ref("team-roster", version="v1"),
        task_graph_ref=_ref(
            "team-task-graph",
            version="v1",
        ),
        execution_authority_ref=_ref(
            "execution-authority",
            version="v1",
        ),
        team_checkpoint_ref=_ref(
            "team-checkpoint",
            version="v1",
        ),
        expected_team_version=1,
        expected_task_graph_revision=1,
        expected_authority_version=1,
        blackboard_head_refs=(),
        thread_id="thread.stage4-integration",
        max_transitions=64,
        audit=_audit(),
    )
    authority = _Authority(store, binding, run.run_id)
    journal = FactoryGraphJournalStore(
        tmp_path / "journal.sqlite3",
    )
    transition = FactoryGraphTransitionService(
        journal=journal,
        authority=authority,
        audit=_audit(),
    )
    runtime = FactoryDatasetGraphRuntime(
        store=store,
        journal=journal,
        authority=authority,
        transition_service=transition,
        continuation_service=_Continuation(),
        session_reconciler=_SessionReconciler(),
        commands=registry,
        request=request,
        policy=policy,
        requirement=requirement,
        assessment_factory=_assessment,
        checkpoint_path=tmp_path / "checkpoint.sqlite3",
        audit=_audit(),
    )

    result = await runtime.start(
        binding,
        idempotency_key="start-graph",
    )

    assert result.state["run_status"] is FactoryRunStatusV2.WAITING_REVIEW
    assert result.state["current_node"] == "review_global_plan"
    checkpoints = journal.list_checkpoints(binding.binding_id)
    assert tuple(value.phase for value in checkpoints) == (
        GraphCheckpointPhaseV1.PRE_TRANSITION,
        GraphCheckpointPhaseV1.POST_TRANSITION,
        GraphCheckpointPhaseV1.PRE_TRANSITION,
        GraphCheckpointPhaseV1.POST_TRANSITION,
    )
    assert checkpoints[-1].outcome is GraphCheckpointOutcomeV1.WAITING_REVIEW
    assert store.get_run(run.run_id).run_version == 2
    assert (
        result.binding.factory_run_ref
        == store.get_run(
            run.run_id,
        ).to_ref()
    )
