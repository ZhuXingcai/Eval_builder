from __future__ import annotations

from typing import Protocol

from eval_factory.agent_system.graph import (
    FactoryGraphDriftError,
    FactoryGraphNodeUpdate,
    FactoryGraphSignalV2,
    FactoryGraphState,
)
from eval_factory.agent_system.graph_journal import (
    FactoryGraphJournalStore,
)
from eval_factory.contracts.agent_system_v2 import FactoryRunStatusV2
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.harness.graph_models import (
    GraphCheckpointOutcomeV1,
    GraphCheckpointPhaseV1,
    HarnessGraphCheckpointV1,
    HarnessGraphExecutionBindingV1,
)


class FactoryGraphBindingAuthority(Protocol):
    def resolve_current(
        self,
        reference: ObjectRef,
    ) -> HarnessGraphExecutionBindingV1: ...

    def capture_current(
        self,
        reference: ObjectRef,
        *,
        audit: ContractAudit,
    ) -> HarnessGraphExecutionBindingV1: ...


class FactoryGraphTransitionService:
    """Owns exact pre/post journal records around one Graph command."""

    def __init__(
        self,
        *,
        journal: FactoryGraphJournalStore,
        authority: FactoryGraphBindingAuthority,
        audit: ContractAudit,
    ) -> None:
        self._journal = journal
        self._authority = authority
        self._audit = audit

    def begin_transition(
        self,
        *,
        node: str,
        state: FactoryGraphState,
    ) -> ObjectRef:
        binding_ref = self._required_ref(
            state,
            "graph_binding_ref",
        )
        binding = self._authority.resolve_current(binding_ref)
        prior = self._journal.current_checkpoint(binding.binding_id)
        if prior is not None and prior.phase is GraphCheckpointPhaseV1.PRE_TRANSITION:
            if prior.node != node or prior.binding_ref != binding.to_ref():
                raise FactoryGraphDriftError(
                    "unfinished Graph pre checkpoint belongs to another command",
                )
            return prior.to_ref()
        transition_number = 1 if prior is None else prior.transition_number + 1
        checkpoint = HarnessGraphCheckpointV1.create(
            checkpoint_id=self._checkpoint_id(
                binding=binding,
                node=node,
                transition_number=transition_number,
                phase=GraphCheckpointPhaseV1.PRE_TRANSITION,
            ),
            binding_ref=binding.to_ref(),
            phase=GraphCheckpointPhaseV1.PRE_TRANSITION,
            node=node,
            transition_number=transition_number,
            predecessor_checkpoint_ref=(prior.to_ref() if prior is not None else None),
            factory_run_ref=binding.factory_run_ref,
            team_ref=binding.team_ref,
            team_checkpoint_ref=binding.team_checkpoint_ref,
            blackboard_head_refs=binding.blackboard_head_refs,
            planner_assessment_ref=None,
            outcome=GraphCheckpointOutcomeV1.READY,
            reason_codes=(),
            audit=self._audit,
        )
        committed = self._journal.commit_checkpoint(
            checkpoint,
            idempotency_key=(f"{binding.binding_id}.{transition_number}.{node}.pre"),
        )
        return committed.to_ref()

    def complete_transition(
        self,
        *,
        node: str,
        state: FactoryGraphState,
        pre_checkpoint_ref: ObjectRef,
    ) -> FactoryGraphNodeUpdate:
        binding_ref = self._required_ref(
            state,
            "graph_binding_ref",
        )
        pre_checkpoint = self._journal.get_checkpoint_by_ref(
            pre_checkpoint_ref,
        )
        if pre_checkpoint.node != node:
            raise FactoryGraphDriftError(
                "Graph pre checkpoint belongs to another node",
            )
        successor = self._authority.capture_current(
            binding_ref,
            audit=self._audit,
        )
        successor = self._journal.commit_binding(
            successor,
            idempotency_key=(f"{successor.binding_id}.{successor.object_sha256}"),
        )
        transition_number = pre_checkpoint.transition_number
        outcome, reason_codes = self._post_outcome(state)
        checkpoint = HarnessGraphCheckpointV1.create(
            checkpoint_id=self._checkpoint_id(
                binding=successor,
                node=node,
                transition_number=transition_number,
                phase=GraphCheckpointPhaseV1.POST_TRANSITION,
            ),
            binding_ref=successor.to_ref(),
            phase=GraphCheckpointPhaseV1.POST_TRANSITION,
            node=node,
            transition_number=transition_number,
            predecessor_checkpoint_ref=pre_checkpoint_ref,
            factory_run_ref=successor.factory_run_ref,
            team_ref=successor.team_ref,
            team_checkpoint_ref=successor.team_checkpoint_ref,
            blackboard_head_refs=successor.blackboard_head_refs,
            planner_assessment_ref=state.get(
                "planner_assessment_ref",
            ),
            outcome=outcome,
            reason_codes=reason_codes,
            audit=self._audit,
        )
        committed = self._journal.commit_checkpoint(
            checkpoint,
            idempotency_key=(f"{successor.binding_id}.{transition_number}.{node}.post"),
        )
        return {
            "expected_authority_version": (successor.expected_authority_version),
            "expected_run_version": (successor.expected_factory_run_version),
            "expected_task_graph_revision": (successor.expected_task_graph_revision),
            "expected_team_version": successor.expected_team_version,
            "graph_binding_ref": successor.to_ref(),
            "graph_checkpoint_ref": committed.to_ref(),
            "run_ref": successor.factory_run_ref,
            "session_ref": successor.session_ref,
            "team_checkpoint_ref": successor.team_checkpoint_ref,
            "team_ref": successor.team_ref,
        }

    @staticmethod
    def _required_ref(
        state: FactoryGraphState,
        field_name: str,
    ) -> ObjectRef:
        value = state.get(field_name)
        if not isinstance(value, ObjectRef):
            raise FactoryGraphDriftError(
                f"Graph transition requires {field_name}",
            )
        return value

    @staticmethod
    def _post_outcome(
        state: FactoryGraphState,
    ) -> tuple[GraphCheckpointOutcomeV1, tuple[str, ...]]:
        signal = state.get("signal")
        if signal is FactoryGraphSignalV2.BLOCKED:
            return (
                GraphCheckpointOutcomeV1.BLOCKED,
                (state.get("error_code") or "GRAPH_SIGNAL_BLOCKED",),
            )
        if signal is FactoryGraphSignalV2.ESCALATE:
            return (
                GraphCheckpointOutcomeV1.VERIFICATION_REQUIRED,
                (state.get("error_code") or "GRAPH_ESCALATION_REQUIRED",),
            )
        if (
            signal is FactoryGraphSignalV2.WAIT
            and state.get("run_status") is FactoryRunStatusV2.WAITING_REVIEW
        ):
            return (
                GraphCheckpointOutcomeV1.WAITING_REVIEW,
                (),
            )
        return GraphCheckpointOutcomeV1.COMMITTED, ()

    @staticmethod
    def _checkpoint_id(
        *,
        binding: HarnessGraphExecutionBindingV1,
        node: str,
        transition_number: int,
        phase: GraphCheckpointPhaseV1,
    ) -> str:
        return f"{binding.binding_id}.{transition_number}.{node}.{phase.value.casefold()}"


__all__ = [
    "FactoryGraphBindingAuthority",
    "FactoryGraphTransitionService",
]
