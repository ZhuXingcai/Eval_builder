from __future__ import annotations

from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.harness.contracts import sorted_refs
from eval_factory.team.models import (
    TeamConvergenceDecisionV1,
    TeamConvergenceOutcomeV1,
    TeamMemberV1,
    TeamMessageAudienceV1,
    TeamMessageKindV1,
    TeamMessageV1,
    TeamTaskStatusV1,
    TeamTaskV1,
)
from eval_factory.team.store import TeamSnapshot, TeamStore, TeamTaskStateError


class TeamCoordinator:
    """Makes deterministic control decisions without specialist authorship."""

    def __init__(self, store: TeamStore) -> None:
        self.store = store

    def converge(
        self,
        team_id: str,
        *,
        audit: ContractAudit,
        decision_key: str,
    ) -> TeamConvergenceDecisionV1:
        source = self.store.convergence_source(team_id)
        before = source.snapshot
        conflicts = source.conflicts
        work = source.work
        messages = source.messages
        challenges = _open_challenges(messages)

        if conflicts:
            return self._record(
                team_id=team_id,
                before=before,
                after=before,
                outcome=TeamConvergenceOutcomeV1.BLOCKED,
                blockers=tuple(value.to_ref() for value in conflicts),
                reason_codes=("OPEN_CONFLICT",),
                audit=audit,
                decision_key=decision_key,
            )

        runnable = tuple(
            value
            for value in work
            if value.status
            in {
                TeamTaskStatusV1.READY,
                TeamTaskStatusV1.CLAIMED,
                TeamTaskStatusV1.ACTIVE,
            }
        )
        if runnable:
            return self._record(
                team_id=team_id,
                before=before,
                after=before,
                outcome=TeamConvergenceOutcomeV1.CONTINUE,
                blockers=tuple(value.task.to_ref() for value in runnable),
                reason_codes=("WORK_REMAINS",),
                audit=audit,
                decision_key=decision_key,
            )

        if challenges:
            reset_task_ids = _challenge_rework_tasks(
                challenge=challenges[0],
                graph_tasks=before.graph.tasks,
                roster_members=before.roster.members,
            )
            try:
                after = self.store.revise_for_rework(
                    team_id,
                    expected_team_ref=before.team.to_ref(),
                    expected_graph_ref=before.graph.to_ref(),
                    expected_authority_ref=before.authority.to_ref(),
                    reset_task_ids=reset_task_ids,
                    audit=audit,
                    idempotency_key=f"{decision_key}.rework",
                )
            except TeamTaskStateError:
                return self._record(
                    team_id=team_id,
                    before=before,
                    after=before,
                    outcome=TeamConvergenceOutcomeV1.BLOCKED,
                    blockers=tuple(value.to_ref() for value in challenges),
                    reason_codes=("REWORK_BUDGET_EXHAUSTED",),
                    audit=audit,
                    decision_key=decision_key,
                )
            return self._record(
                team_id=team_id,
                before=before,
                after=after,
                outcome=TeamConvergenceOutcomeV1.REWORK,
                blockers=tuple(value.to_ref() for value in challenges),
                reason_codes=("DIRECT_CHALLENGE_REWORK",),
                audit=audit,
                decision_key=decision_key,
            )

        terminal_failures = tuple(
            value
            for value in work
            if value.status
            in {
                TeamTaskStatusV1.FAILED,
                TeamTaskStatusV1.CANCELLED,
                TeamTaskStatusV1.BLOCKED,
            }
        )
        if terminal_failures:
            return self._record(
                team_id=team_id,
                before=before,
                after=before,
                outcome=TeamConvergenceOutcomeV1.BLOCKED,
                blockers=tuple(value.task.to_ref() for value in terminal_failures),
                reason_codes=("TERMINAL_TASK_FAILURE",),
                audit=audit,
                decision_key=decision_key,
            )

        required_head_ids = tuple(
            head_id
            for task in before.graph.tasks
            if task.assigned_member_id != before.team.coordinator_member_id
            for head_id in task.output_artifact_head_ids
        )
        current_head_ids = {head.head_id for head in source.artifact_heads}
        all_completed = work and all(value.status is TeamTaskStatusV1.COMPLETED for value in work)
        missing_heads = tuple(head_id for head_id in required_head_ids if head_id not in current_head_ids)
        if all_completed and not missing_heads:
            return self._record(
                team_id=team_id,
                before=before,
                after=before,
                outcome=TeamConvergenceOutcomeV1.COMPLETE,
                blockers=(),
                reason_codes=(),
                audit=audit,
                decision_key=decision_key,
            )
        if all_completed:
            return self._record(
                team_id=team_id,
                before=before,
                after=before,
                outcome=TeamConvergenceOutcomeV1.BLOCKED,
                blockers=tuple(
                    task.to_ref()
                    for task in before.graph.tasks
                    if set(task.output_artifact_head_ids).intersection(
                        missing_heads,
                    )
                ),
                reason_codes=("REQUIRED_ARTIFACT_HEAD_MISSING",),
                audit=audit,
                decision_key=decision_key,
            )

        return self._record(
            team_id=team_id,
            before=before,
            after=before,
            outcome=TeamConvergenceOutcomeV1.BLOCKED,
            blockers=tuple(value.task.to_ref() for value in work),
            reason_codes=("NO_RUNNABLE_WORK",),
            audit=audit,
            decision_key=decision_key,
        )

    def _record(
        self,
        *,
        team_id: str,
        before: TeamSnapshot,
        after: TeamSnapshot,
        outcome: TeamConvergenceOutcomeV1,
        blockers: tuple[ObjectRef, ...],
        reason_codes: tuple[str, ...],
        audit: ContractAudit,
        decision_key: str,
    ) -> TeamConvergenceDecisionV1:
        checkpoint = self.store.create_checkpoint(
            team_id,
            audit=audit,
            idempotency_key=f"{decision_key}.checkpoint",
        )
        rework = outcome is TeamConvergenceOutcomeV1.REWORK
        decision = TeamConvergenceDecisionV1.create(
            decision_key=decision_key,
            team_ref=before.team.to_ref(),
            graph_ref=before.graph.to_ref(),
            authority_ref=before.authority.to_ref(),
            checkpoint_ref=checkpoint.to_ref(),
            outcome=outcome,
            blocking_refs=sorted_refs(blockers),
            successor_team_ref=(after.team.to_ref() if rework else None),
            successor_graph_ref=(after.graph.to_ref() if rework else None),
            successor_authority_ref=(after.authority.to_ref() if rework else None),
            reason_codes=reason_codes,
            audit=audit,
        )
        return self.store.record_convergence(
            decision,
            idempotency_key=f"{decision_key}.record",
        )


def _open_challenges(
    messages: tuple[TeamMessageV1, ...],
) -> tuple[TeamMessageV1, ...]:
    return tuple(
        challenge
        for challenge in messages
        if challenge.message_kind is TeamMessageKindV1.CHALLENGE
        and not any(
            response.reply_to_message_ref == challenge.to_ref()
            and response.sender_member_ref == challenge.sender_member_ref
            and response.task_ref == challenge.task_ref
            and response.message_kind
            in {
                TeamMessageKindV1.ACKNOWLEDGEMENT,
                TeamMessageKindV1.REVIEW_RESPONSE,
            }
            for response in messages
        )
    )


def _challenge_rework_tasks(
    *,
    challenge: TeamMessageV1,
    graph_tasks: tuple[TeamTaskV1, ...],
    roster_members: tuple[TeamMemberV1, ...],
) -> tuple[str, ...]:
    challenged = next(
        (
            task
            for task in graph_tasks
            if challenge.task_ref is not None and task.to_ref() == challenge.task_ref
        ),
        None,
    )
    sender = next(
        (member for member in roster_members if member.to_ref() == challenge.sender_member_ref),
        None,
    )
    if challenged is None or sender is None:
        raise TeamTaskStateError("challenge authority is stale")
    challenged_owner = next(
        (member for member in roster_members if member.member_id == challenged.assigned_member_id),
        None,
    )
    if (
        challenge.audience is not TeamMessageAudienceV1.DIRECT
        or challenged_owner is None
        or challenged_owner.to_ref() not in challenge.recipient_member_refs
    ):
        raise TeamTaskStateError(
            "challenge does not target the current task owner",
        )
    task_ids = {
        challenged.task_id,
        *(task.task_id for task in graph_tasks if task.assigned_member_id == sender.member_id),
    }
    return tuple(sorted(task_ids))


__all__ = ["TeamCoordinator"]
