from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from eval_factory.agent_system.dataset_graph_handlers import (
    FactoryDatasetAdvanceCompatibilityCommands,
    FactoryDatasetGraphCommands,
    FactoryDatasetGraphHandlers,
    FactoryPlannerAssessmentFactory,
)
from eval_factory.agent_system.graph import (
    FactoryControlGraph,
    FactoryGraphDriftError,
    FactoryGraphSignalV2,
    FactoryGraphState,
    async_factory_graph_checkpointer,
)
from eval_factory.agent_system.graph_continuation import (
    FactoryGraphContinuationFactorySource,
    FactoryGraphContinuationReviewSource,
    FactoryGraphContinuationService,
)
from eval_factory.agent_system.graph_journal import (
    FactoryGraphJournalStore,
)
from eval_factory.agent_system.graph_projection import (
    FactoryGraphSessionEventSink,
    FactoryGraphSessionReconciler,
)
from eval_factory.agent_system.graph_transition import (
    FactoryGraphBindingAuthority,
    FactoryGraphTransitionService,
)
from eval_factory.agent_system.store import FactoryControlStore
from eval_factory.contracts.agent_system_v2 import (
    EvaluationRequirementSpecV2,
    FactoryRunPolicyV2,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.dataset_runtime_v2 import (
    FactoryDatasetRunRequestV2,
)
from eval_factory.harness.graph_models import (
    HarnessGraphCheckpointV1,
    HarnessGraphExecutionBindingV1,
)
from eval_factory.harness.session_models import SessionEventV1


@dataclass(frozen=True, slots=True)
class FactoryDatasetGraphRun:
    binding: HarnessGraphExecutionBindingV1
    checkpoint: HarnessGraphCheckpointV1 | None
    state: FactoryGraphState


class FactoryGraphContinuation(Protocol):
    def resume(
        self,
        *,
        graph_binding_ref: ObjectRef,
        expected_checkpoint_ref: ObjectRef,
        review_result_ref: ObjectRef,
        idempotency_key: str,
    ) -> HarnessGraphCheckpointV1: ...


class FactoryGraphSessionProjection(Protocol):
    def reconcile(
        self,
        binding_id: str,
        *,
        audit: ContractAudit,
        limit: int = 500,
    ) -> tuple[SessionEventV1, ...]: ...


class FactoryDatasetGraphRuntime:
    """Default product facade for one recoverable Factory Graph run."""

    def __init__(
        self,
        *,
        store: FactoryControlStore,
        journal: FactoryGraphJournalStore,
        authority: FactoryGraphBindingAuthority,
        transition_service: FactoryGraphTransitionService,
        continuation_service: FactoryGraphContinuation,
        session_reconciler: FactoryGraphSessionProjection,
        commands: FactoryDatasetGraphCommands,
        request: FactoryDatasetRunRequestV2,
        policy: FactoryRunPolicyV2,
        requirement: EvaluationRequirementSpecV2,
        assessment_factory: FactoryPlannerAssessmentFactory,
        checkpoint_path: Path,
        audit: ContractAudit,
    ) -> None:
        if isinstance(
            commands,
            FactoryDatasetAdvanceCompatibilityCommands,
        ):
            raise ValueError(
                "default Graph runtime rejects advance compatibility",
            )
        self.store = store
        self.journal = journal
        self.authority = authority
        self.transition_service = transition_service
        self.continuation_service = continuation_service
        self.session_reconciler = session_reconciler
        self.commands = commands
        self.request = request
        self.policy = policy
        self.requirement = requirement
        self.assessment_factory = assessment_factory
        self.checkpoint_path = checkpoint_path.expanduser().resolve()
        self.audit = audit
        paths = (
            self.store.path.expanduser().resolve(),
            self.journal.path.expanduser().resolve(),
            self.checkpoint_path,
        )
        if len(set(paths)) != len(paths):
            raise ValueError(
                "Graph checkpoint, journal, and Factory Store require separate paths",
            )
        self.handlers = FactoryDatasetGraphHandlers(
            commands=commands,
            request=request,
            policy=policy,
            requirement=requirement,
            core_input=None,
            audit=audit,
            assessment_factory=assessment_factory,
        )

    async def start(
        self,
        binding: HarnessGraphExecutionBindingV1,
        *,
        idempotency_key: str,
    ) -> FactoryDatasetGraphRun:
        if (
            binding.factory_request_ref != self.request.to_ref()
            or binding.factory_policy_ref != self.policy.to_ref()
            or binding.requirement_ref != self.requirement.to_ref()
        ):
            raise FactoryGraphDriftError(
                "Graph binding differs from dataset runtime inputs",
            )
        committed = self.journal.commit_binding(
            binding,
            idempotency_key=idempotency_key,
        )
        self.authority.resolve_current(committed.to_ref())
        state = self._initial_state(committed)
        result = await self._invoke(
            binding=committed,
            state=state,
        )
        return FactoryDatasetGraphRun(
            binding=self.journal.get_binding(committed.binding_id),
            checkpoint=self.journal.current_checkpoint(
                committed.binding_id,
            ),
            state=result,
        )

    async def resume(
        self,
        *,
        graph_binding_ref: ObjectRef,
        expected_checkpoint_ref: ObjectRef,
        review_result_ref: ObjectRef,
        idempotency_key: str,
    ) -> FactoryDatasetGraphRun:
        reconciled = self.continuation_service.resume(
            graph_binding_ref=graph_binding_ref,
            expected_checkpoint_ref=expected_checkpoint_ref,
            review_result_ref=review_result_ref,
            idempotency_key=idempotency_key,
        )
        binding = self.journal.get_binding_by_ref(
            reconciled.binding_ref,
        )
        state = self._initial_state(
            binding,
            checkpoint_ref=reconciled.to_ref(),
            transition_count=reconciled.transition_number,
        )
        result = await self._invoke(
            binding=binding,
            state=state,
        )
        return FactoryDatasetGraphRun(
            binding=self.journal.get_binding(binding.binding_id),
            checkpoint=self.journal.current_checkpoint(
                binding.binding_id,
            ),
            state=result,
        )

    async def continue_current(
        self,
        graph_binding_ref: ObjectRef,
    ) -> FactoryDatasetGraphRun:
        binding = self.authority.resolve_current(
            graph_binding_ref,
        )
        checkpoint = self.journal.current_checkpoint(
            binding.binding_id,
        )
        state = self._initial_state(
            binding,
            checkpoint_ref=(checkpoint.to_ref() if checkpoint is not None else None),
            transition_count=(checkpoint.transition_number if checkpoint is not None else 0),
        )
        result = await self._invoke(
            binding=binding,
            state=state,
        )
        return FactoryDatasetGraphRun(
            binding=self.journal.get_binding(binding.binding_id),
            checkpoint=self.journal.current_checkpoint(
                binding.binding_id,
            ),
            state=result,
        )

    def reconcile(
        self,
        graph_binding_ref: ObjectRef,
    ) -> tuple[SessionEventV1, ...]:
        binding = self.authority.resolve_current(
            graph_binding_ref,
        )
        self.journal.rebuild_current_heads()
        return self.session_reconciler.reconcile(
            binding.binding_id,
            audit=self.audit,
        )

    def show(
        self,
        graph_binding_ref: ObjectRef,
    ) -> FactoryDatasetGraphRun:
        binding = self.authority.resolve_current(
            graph_binding_ref,
        )
        checkpoint = self.journal.current_checkpoint(
            binding.binding_id,
        )
        state = self._initial_state(
            binding,
            checkpoint_ref=(checkpoint.to_ref() if checkpoint is not None else None),
            transition_count=(checkpoint.transition_number if checkpoint is not None else 0),
        )
        return FactoryDatasetGraphRun(
            binding=binding,
            checkpoint=checkpoint,
            state=state,
        )

    async def _invoke(
        self,
        *,
        binding: HarnessGraphExecutionBindingV1,
        state: FactoryGraphState,
    ) -> FactoryGraphState:
        async with async_factory_graph_checkpointer(
            self.checkpoint_path,
            control_store_path=self.store.path,
        ) as checkpointer:
            graph = FactoryControlGraph(
                self.store,
                handlers=self.handlers.mapping(),
                transition_limit=binding.max_transitions,
                binding_resolver=self.authority,
                transition_journal=self.transition_service,
            ).compile(checkpointer=checkpointer)
            result = await graph.ainvoke(
                state,
                {
                    "configurable": {
                        "thread_id": binding.thread_id,
                    },
                },
            )
        return cast(FactoryGraphState, result)

    def _initial_state(
        self,
        binding: HarnessGraphExecutionBindingV1,
        *,
        checkpoint_ref: ObjectRef | None = None,
        transition_count: int = 0,
    ) -> FactoryGraphState:
        run = self.store.get_run_by_ref(
            binding.factory_run_ref,
        )
        return {
            "factory_run_id": self.request.dataset_run_id,
            "expected_run_version": (binding.expected_factory_run_version),
            "transition_count": transition_count,
            "signal": FactoryGraphSignalV2.CONTINUE,
            "run_ref": binding.factory_run_ref,
            "run_status": run.status,
            "current_plan_ref": run.current_plan_ref,
            "compiled_plan_ref": run.compiled_plan_ref,
            "pending_review_ref": run.pending_review_ref,
            "planner_assessment_ref": run.planner_assessment_ref,
            "completion_ref": run.completion_ref,
            "delivery_manifest_ref": run.delivery_manifest_ref,
            "graph_binding_ref": binding.to_ref(),
            "graph_checkpoint_ref": checkpoint_ref,
            "session_ref": binding.session_ref,
            "team_ref": binding.team_ref,
            "team_checkpoint_ref": binding.team_checkpoint_ref,
            "blueprint_ref": binding.blueprint_ref,
            "composition_ref": binding.composition_ref,
            "expected_team_version": binding.expected_team_version,
            "expected_task_graph_revision": (binding.expected_task_graph_revision),
            "expected_authority_version": (binding.expected_authority_version),
        }


__all__ = [
    "FactoryDatasetGraphRun",
    "FactoryDatasetGraphRuntime",
    "FactoryGraphBindingAuthority",
    "FactoryGraphContinuation",
    "FactoryGraphContinuationFactorySource",
    "FactoryGraphContinuationReviewSource",
    "FactoryGraphContinuationService",
    "FactoryGraphSessionEventSink",
    "FactoryGraphSessionProjection",
    "FactoryGraphSessionReconciler",
    "FactoryGraphTransitionService",
]
