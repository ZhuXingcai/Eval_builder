from __future__ import annotations

from eval_factory.agent_system.graph import FactoryGraphDriftError
from eval_factory.agent_system.graph_journal import (
    FactoryGraphJournalStore,
)
from eval_factory.agent_system.graph_transition import (
    FactoryGraphBindingAuthority,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.harness.graph_models import (
    GraphCheckpointOutcomeV1,
    GraphCheckpointPhaseV1,
    HarnessGraphCheckpointV1,
    HarnessGraphExecutionBindingV1,
)


class FactoryGraphRecoveryService:
    """Reconciles unfinished PRE records without inventing owner success."""

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

    def reconcile_pre(
        self,
        *,
        graph_binding_ref: ObjectRef,
        expected_pre_checkpoint_ref: ObjectRef,
        idempotency_key: str,
    ) -> HarnessGraphCheckpointV1:
        prior = self._journal.get_binding_by_ref(
            graph_binding_ref,
        )
        pre = self._journal.get_checkpoint_by_ref(
            expected_pre_checkpoint_ref,
        )
        current = self._journal.current_checkpoint(
            prior.binding_id,
        )
        if (
            current is not None
            and current.phase is GraphCheckpointPhaseV1.RECONCILED
            and current.predecessor_checkpoint_ref == expected_pre_checkpoint_ref
            and current.outcome is GraphCheckpointOutcomeV1.VERIFICATION_REQUIRED
            and current.reason_codes == ("OWNER_COMMIT_AFTER_PRE_REQUIRES_VERIFICATION",)
        ):
            return current
        if (
            current is None
            or current.to_ref() != expected_pre_checkpoint_ref
            or pre.phase is not GraphCheckpointPhaseV1.PRE_TRANSITION
            or pre.binding_ref != graph_binding_ref
        ):
            raise FactoryGraphDriftError(
                "Graph recovery PRE authority is stale",
            )
        try:
            self._authority.resolve_current(graph_binding_ref)
        except FactoryGraphDriftError:
            successor = self._authority.capture_current(
                graph_binding_ref,
                audit=self._audit,
            )
        else:
            return pre
        self._validate_pinned_authority(
            prior=prior,
            successor=successor,
        )
        successor = self._journal.commit_binding(
            successor,
            idempotency_key=(f"{idempotency_key}.binding.{successor.object_sha256}"),
        )
        reconciled = HarnessGraphCheckpointV1.create(
            checkpoint_id=(
                f"{successor.binding_id}.{pre.transition_number}.{pre.node}.verification-required"
            ),
            binding_ref=successor.to_ref(),
            phase=GraphCheckpointPhaseV1.RECONCILED,
            node=pre.node,
            transition_number=pre.transition_number,
            predecessor_checkpoint_ref=pre.to_ref(),
            factory_run_ref=successor.factory_run_ref,
            team_ref=successor.team_ref,
            team_checkpoint_ref=successor.team_checkpoint_ref,
            blackboard_head_refs=successor.blackboard_head_refs,
            planner_assessment_ref=None,
            outcome=(GraphCheckpointOutcomeV1.VERIFICATION_REQUIRED),
            reason_codes=("OWNER_COMMIT_AFTER_PRE_REQUIRES_VERIFICATION",),
            audit=self._audit,
        )
        return self._journal.commit_checkpoint(
            reconciled,
            idempotency_key=f"{idempotency_key}.checkpoint",
        )

    @staticmethod
    def _validate_pinned_authority(
        *,
        prior: HarnessGraphExecutionBindingV1,
        successor: HarnessGraphExecutionBindingV1,
    ) -> None:
        invariant_fields = (
            "binding_id",
            "session_ref",
            "requirement_ref",
            "requirement_policy_ref",
            "factory_request_ref",
            "factory_policy_ref",
            "pack_manifest_ref",
            "composition_ref",
            "blueprint_ref",
            "thread_id",
            "max_transitions",
        )
        if any(getattr(prior, field) != getattr(successor, field) for field in invariant_fields):
            raise FactoryGraphDriftError(
                "Graph recovery changed pinned authority",
            )


__all__ = ["FactoryGraphRecoveryService"]
