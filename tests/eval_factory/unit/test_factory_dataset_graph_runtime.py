from __future__ import annotations

import inspect
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import eval_factory.agent_system.graph_runtime as graph_runtime_module
from eval_factory.agent_system.dataset_graph_handlers import (
    FactoryDatasetAdvanceCompatibilityCommands,
    FactoryDatasetGraphHandlers,
    FactoryDatasetNodeCommand,
    FactoryDatasetNodeCommandRegistry,
)
from eval_factory.agent_system.dataset_runtime import FactoryDatasetRuntime
from eval_factory.agent_system.graph import (
    FactoryGraphSignalV2,
    FactoryGraphState,
)
from eval_factory.agent_system.graph_journal import (
    FactoryGraphJournalStore,
)
from eval_factory.agent_system.graph_runtime import (
    FactoryDatasetGraphRun,
    FactoryDatasetGraphRuntime,
)
from eval_factory.agent_system.store import FactoryControlStore
from eval_factory.contracts.agent_system_v2 import (
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
        created_by="dataset-graph-runtime-test",
        governing_versions=(
            VersionBinding(
                component="eval-harness-stage4",
                version="v1",
                sha256=HASH,
            ),
        ),
    )


def _ref(object_type: str) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://stage4",
        object_version="v2",
        object_sha256=HASH,
    )


def _harness_ref(
    object_type: str,
    suffix: str = "stage4",
    *,
    version: str = "v1",
) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://{suffix}",
        object_version=version,
        object_sha256=HASH,
    )


def _inputs() -> tuple[
    FactoryRunPolicyV2,
    EvaluationRequirementSpecV2,
    FactoryDatasetRunRequestV2,
]:
    policy = FactoryRunPolicyV2.create(
        policy_id="factory-policy.stage4",
        allowed_task_kinds=("dataset-production",),
        max_transitions=64,
        max_plan_revisions=4,
        max_agent_attempts=2,
        max_model_requests=100,
        max_model_tokens=1_000_000,
        max_cost_micro_usd=100_000_000,
        audit=_audit(),
    )
    requirement = EvaluationRequirementSpecV2.create(
        requirement_spec_id="requirement.stage4",
        run_id="factory-run.stage4",
        source_ref=_ref("evaluation-requirement-source"),
        goals=("Build evaluation data.",),
        constraints=(),
        assumptions=(),
        open_questions=(),
        requirement_version=1,
        audit=_audit(),
    )
    request = FactoryDatasetRunRequestV2.create(
        dataset_run_id=requirement.run_id,
        requirement_spec_ref=requirement.to_ref(),
        manifest_ref=_ref("trace-manifest"),
        source_authorization_ref=_ref(
            "trace-source-authorization",
        ),
        factory_policy_ref=policy.to_ref(),
        pipeline_policy_refs=(_ref("trace-index-policy"),),
        gateway_registry_refs=(_ref("agent-registry"),),
        output_target_ref=_ref("candidate-output-target"),
        idempotency_key="dataset-run-stage4",
        max_transitions=64,
        audit=_audit(),
    )
    return policy, requirement, request


def _binding(
    *,
    run_ref: ObjectRef,
    request: FactoryDatasetRunRequestV2,
    policy: FactoryRunPolicyV2,
    requirement: EvaluationRequirementSpecV2,
) -> HarnessGraphExecutionBindingV1:
    return HarnessGraphExecutionBindingV1.create(
        binding_id="binding-stage4",
        session_ref=_harness_ref("harness-session"),
        requirement_ref=requirement.to_ref(),
        requirement_policy_ref=_harness_ref(
            "harness-requirement-policy",
        ),
        factory_run_ref=run_ref,
        factory_request_ref=request.to_ref(),
        factory_policy_ref=policy.to_ref(),
        expected_factory_run_version=0,
        pack_manifest_ref=_harness_ref("eval-pack-manifest"),
        composition_ref=_harness_ref("harness-composition"),
        blueprint_ref=_harness_ref("evaluation-blueprint"),
        team_ref=_harness_ref("agent-team"),
        roster_ref=_harness_ref("team-roster"),
        task_graph_ref=_harness_ref("team-task-graph"),
        execution_authority_ref=_harness_ref(
            "execution-authority",
        ),
        team_checkpoint_ref=_harness_ref("team-checkpoint"),
        expected_team_version=1,
        expected_task_graph_revision=1,
        expected_authority_version=1,
        blackboard_head_refs=(),
        thread_id="thread-stage4",
        max_transitions=64,
        audit=_audit(),
    )


@dataclass
class _Authority:
    binding: HarnessGraphExecutionBindingV1
    calls: int = 0

    def resolve_current(
        self,
        reference: ObjectRef,
    ) -> HarnessGraphExecutionBindingV1:
        assert reference == self.binding.to_ref()
        self.calls += 1
        return self.binding

    def capture_current(
        self,
        reference: ObjectRef,
        *,
        audit: ContractAudit,
    ) -> HarnessGraphExecutionBindingV1:
        del audit
        return self.resolve_current(reference)


@dataclass
class _Continuation:
    binding: HarnessGraphExecutionBindingV1
    calls: int = 0

    def resume(
        self,
        *,
        graph_binding_ref: ObjectRef,
        expected_checkpoint_ref: ObjectRef,
        review_result_ref: ObjectRef,
        idempotency_key: str,
    ) -> HarnessGraphCheckpointV1:
        del expected_checkpoint_ref, review_result_ref, idempotency_key
        assert graph_binding_ref == self.binding.to_ref()
        self.calls += 1
        return HarnessGraphCheckpointV1.create(
            checkpoint_id="checkpoint-stage4-reconciled",
            binding_ref=self.binding.to_ref(),
            phase=GraphCheckpointPhaseV1.RECONCILED,
            node="review_global_plan",
            transition_number=1,
            predecessor_checkpoint_ref=_harness_ref(
                "graph-checkpoint",
                "prior",
            ),
            factory_run_ref=self.binding.factory_run_ref,
            team_ref=self.binding.team_ref,
            team_checkpoint_ref=self.binding.team_checkpoint_ref,
            blackboard_head_refs=(),
            planner_assessment_ref=None,
            outcome=GraphCheckpointOutcomeV1.COMMITTED,
            reason_codes=(),
            audit=_audit(),
        )


@dataclass
class _SessionReconciler:
    calls: int = 0

    def reconcile(
        self,
        binding_id: str,
        *,
        audit: ContractAudit,
        limit: int = 500,
    ) -> tuple[SessionEventV1, ...]:
        del audit, limit
        assert binding_id == "binding-stage4"
        self.calls += 1
        return ()


class _FacadeRuntime(FactoryDatasetGraphRuntime):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.invocations: list[FactoryGraphState] = []

    async def _invoke(
        self,
        *,
        binding: HarnessGraphExecutionBindingV1,
        state: FactoryGraphState,
    ) -> FactoryGraphState:
        assert binding == self.authority.resolve_current(
            binding.to_ref(),
        )
        self.invocations.append(state)
        return state


def _commands(
    store: FactoryControlStore,
) -> FactoryDatasetNodeCommandRegistry:
    async def unused(
        state: FactoryGraphState,
    ) -> FactoryDatasetRunViewV2:
        del state
        raise AssertionError("facade test replaces Graph invocation")

    return FactoryDatasetNodeCommandRegistry(
        store=store,
        commands={node: unused for node in NODES},
    )


def _assessment(
    state: FactoryGraphState,
    view: FactoryDatasetRunViewV2,
    run: FactoryRunV2,
) -> PlannerAssessmentV2:
    del state, view, run
    raise AssertionError("facade test replaces Graph invocation")


def _facade(
    tmp_path: Path,
) -> tuple[
    _FacadeRuntime,
    HarnessGraphExecutionBindingV1,
    _Authority,
    _Continuation,
    _SessionReconciler,
]:
    policy, requirement, request = _inputs()
    store = FactoryControlStore(tmp_path / "factory.sqlite3")
    run = store.create_run(
        policy=policy,
        requirement=requirement,
        idempotency_key="create-run",
    )
    store.commit_dataset_request(
        request,
        idempotency_key="commit-request",
    )
    binding = _binding(
        run_ref=run.to_ref(),
        request=request,
        policy=policy,
        requirement=requirement,
    )
    authority = _Authority(binding)
    continuation = _Continuation(binding)
    reconciler = _SessionReconciler()
    runtime = _FacadeRuntime(
        store=store,
        journal=FactoryGraphJournalStore(
            tmp_path / "journal.sqlite3",
        ),
        authority=authority,
        transition_service=cast(Any, object()),
        continuation_service=continuation,
        session_reconciler=reconciler,
        commands=_commands(store),
        request=request,
        policy=policy,
        requirement=requirement,
        assessment_factory=_assessment,
        checkpoint_path=tmp_path / "checkpoint.sqlite3",
        audit=_audit(),
    )
    return runtime, binding, authority, continuation, reconciler


@pytest.mark.asyncio
async def test_handlers_execute_only_named_node_command(
    tmp_path: Path,
) -> None:
    policy, requirement, request = _inputs()
    store = FactoryControlStore(tmp_path / "factory.sqlite3")
    run = store.create_run(
        policy=policy,
        requirement=requirement,
        idempotency_key="create-run",
    )
    view = FactoryDatasetRunViewV2.create(
        request_ref=request.to_ref(),
        dataset_run_ref=run.to_ref(),
        status=run.status,
        pending_review_refs=(),
        item_binding_refs=(),
        candidate_count=0,
        rejected_count=0,
        blocked_count=0,
        incomplete_count=0,
        aggregate_result_ref=None,
        delivery_manifest_ref=None,
        next_action=FactoryDatasetNextActionV2.ADVANCE,
        audit=_audit(),
    )
    calls: list[str] = []

    def command(node: str) -> FactoryDatasetNodeCommand:
        async def execute(
            state: FactoryGraphState,
        ) -> FactoryDatasetRunViewV2:
            assert state["factory_run_id"] == request.dataset_run_id
            calls.append(node)
            return view

        return execute

    registry = FactoryDatasetNodeCommandRegistry(
        store=store,
        commands={node: command(node) for node in NODES},
    )
    handler = FactoryDatasetGraphHandlers(
        commands=registry,
        request=request,
        policy=policy,
        requirement=requirement,
        core_input=None,
        audit=_audit(),
    ).mapping()["requirement_planner"]
    raw = handler(
        {
            "factory_run_id": run.run_id,
            "expected_run_version": run.run_version,
            "transition_count": 1,
            "signal": FactoryGraphSignalV2.CONTINUE,
            "run_ref": run.to_ref(),
        }
    )
    assert inspect.isawaitable(raw)
    update = await raw

    assert calls == ["requirement_planner"]
    assert update["run_ref"] == run.to_ref()


def test_assemble_signal_enters_final_delivery_review() -> None:
    view = cast(
        FactoryDatasetRunViewV2,
        SimpleNamespace(status=FactoryRunStatusV2.RUNNING),
    )

    assert (
        FactoryDatasetGraphHandlers._signal(
            node="assemble_core_output",
            view=view,
        )
        is FactoryGraphSignalV2.CONTINUE
    )


@pytest.mark.asyncio
async def test_default_facade_start_show_reconcile_and_resume(
    tmp_path: Path,
) -> None:
    runtime, binding, authority, continuation, reconciler = _facade(tmp_path)

    started = await runtime.start(
        binding,
        idempotency_key="start-stage4",
    )
    shown = runtime.show(binding.to_ref())
    projected = runtime.reconcile(binding.to_ref())
    continued = await runtime.continue_current(
        binding.to_ref(),
    )
    resumed = await runtime.resume(
        graph_binding_ref=binding.to_ref(),
        expected_checkpoint_ref=_harness_ref(
            "graph-checkpoint",
            "waiting",
        ),
        review_result_ref=_ref("plan-review-result"),
        idempotency_key="resume-stage4",
    )

    assert isinstance(started, FactoryDatasetGraphRun)
    assert started.binding == binding
    assert started.state["graph_binding_ref"] == binding.to_ref()
    assert shown.binding == binding
    assert shown.checkpoint is None
    assert projected == ()
    assert continued.binding == binding
    assert continued.state["transition_count"] == 0
    assert resumed.binding == binding
    assert resumed.state["transition_count"] == 1
    assert resumed.state["graph_checkpoint_ref"] is not None
    assert len(runtime.invocations) == 3
    assert authority.calls >= 4
    assert continuation.calls == 1
    assert reconciler.calls == 1


@pytest.mark.asyncio
async def test_default_facade_rejects_binding_input_drift(
    tmp_path: Path,
) -> None:
    runtime, binding, *_ = _facade(tmp_path)
    changed = binding.model_copy(
        update={
            "requirement_ref": _ref(
                "evaluation-requirement-spec",
            ),
        },
    )

    with pytest.raises(
        Exception,
        match="differs from dataset runtime inputs",
    ):
        await runtime.start(
            changed,
            idempotency_key="changed-binding",
        )


def test_default_facade_requires_three_separate_store_paths(
    tmp_path: Path,
) -> None:
    policy, requirement, request = _inputs()
    store = FactoryControlStore(tmp_path / "factory.sqlite3")
    run = store.create_run(
        policy=policy,
        requirement=requirement,
        idempotency_key="create-run",
    )
    binding = _binding(
        run_ref=run.to_ref(),
        request=request,
        policy=policy,
        requirement=requirement,
    )

    with pytest.raises(ValueError, match="separate paths"):
        FactoryDatasetGraphRuntime(
            store=store,
            journal=FactoryGraphJournalStore(
                tmp_path / "journal.sqlite3",
            ),
            authority=_Authority(binding),
            transition_service=cast(Any, object()),
            continuation_service=_Continuation(binding),
            session_reconciler=_SessionReconciler(),
            commands=_commands(store),
            request=request,
            policy=policy,
            requirement=requirement,
            assessment_factory=_assessment,
            checkpoint_path=store.path,
            audit=_audit(),
        )


@pytest.mark.asyncio
async def test_facade_invokes_langgraph_with_bound_thread(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, binding, *_ = _facade(tmp_path)

    class _Compiled:
        async def ainvoke(
            self,
            state: FactoryGraphState,
            config: dict[str, object],
        ) -> FactoryGraphState:
            assert config == {
                "configurable": {
                    "thread_id": binding.thread_id,
                },
            }
            return state

    class _Graph:
        def __init__(
            self,
            *args: object,
            **kwargs: object,
        ) -> None:
            assert args
            assert kwargs["binding_resolver"] is runtime.authority
            assert kwargs["transition_journal"] is runtime.transition_service

        def compile(
            self,
            checkpointer: object,
        ) -> _Compiled:
            assert checkpointer is not None
            return _Compiled()

    monkeypatch.setattr(
        graph_runtime_module,
        "FactoryControlGraph",
        _Graph,
    )
    state = runtime._initial_state(binding)

    result = await FactoryDatasetGraphRuntime._invoke(
        runtime,
        binding=binding,
        state=state,
    )

    assert result == state
    assert runtime.checkpoint_path.exists()


def test_default_graph_runtime_rejects_advance_compatibility(
    tmp_path: Path,
) -> None:
    policy, requirement, request = _inputs()
    store = FactoryControlStore(tmp_path / "factory.sqlite3")
    compatibility = FactoryDatasetAdvanceCompatibilityCommands(
        runtime=cast(
            FactoryDatasetRuntime,
            SimpleNamespace(store=store),
        ),
        request=request,
        policy=policy,
        requirement=requirement,
        core_input=None,
        audit=_audit(),
    )

    with pytest.raises(ValueError, match="rejects advance"):
        FactoryDatasetGraphRuntime(
            store=store,
            journal=cast(Any, None),
            authority=cast(Any, None),
            transition_service=cast(Any, None),
            continuation_service=cast(Any, None),
            session_reconciler=cast(Any, None),
            commands=compatibility,
            request=request,
            policy=policy,
            requirement=requirement,
            assessment_factory=cast(Any, None),
            checkpoint_path=tmp_path / "checkpoint.sqlite3",
            audit=_audit(),
        )
