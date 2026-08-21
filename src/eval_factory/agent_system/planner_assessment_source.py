from __future__ import annotations

from eval_factory.agent_system.graph import (
    FactoryGraphDriftError,
    FactoryGraphState,
)
from eval_factory.agent_system.planner_assessment import (
    FactoryPlannerAssessmentCompiler,
    FactoryPlannerAssessmentFacts,
)
from eval_factory.agent_system.store import FactoryControlStore
from eval_factory.contracts.agent_system_v2 import (
    FactoryRunV2,
    PlannerAssessmentV2,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.dataset_runtime_v2 import (
    FactoryDatasetRunViewV2,
)
from eval_factory.harness.contracts import sorted_refs
from eval_factory.team import TeamStore, TeamTaskStatusV1


class FactoryPlannerAssessmentSource:
    """Compiles assessment authority from current Factory and Team facts."""

    def __init__(
        self,
        *,
        factory_store: FactoryControlStore,
        team_store: TeamStore,
        team_id: str,
        audit: ContractAudit,
        compiler: FactoryPlannerAssessmentCompiler | None = None,
        terminal_task_kinds: frozenset[str] = frozenset(
            {"delivery"},
        ),
    ) -> None:
        self.factory_store = factory_store
        self.team_store = team_store
        self.team_id = team_id
        self.audit = audit
        self.compiler = compiler or FactoryPlannerAssessmentCompiler()
        self.terminal_task_kinds = terminal_task_kinds

    def __call__(
        self,
        state: FactoryGraphState,
        view: FactoryDatasetRunViewV2,
        run: FactoryRunV2,
    ) -> PlannerAssessmentV2:
        current = self.factory_store.get_run(run.run_id)
        if current != run or view.dataset_run_ref != run.to_ref() or state.get("run_ref") != run.to_ref():
            raise FactoryGraphDriftError(
                "PlannerAssessment source Factory authority drifted",
            )
        _plan, compiled = self.factory_store.get_plan(run.run_id)
        snapshot = self.team_store.get_snapshot(self.team_id)
        if state.get("team_ref") != snapshot.team.to_ref():
            raise FactoryGraphDriftError(
                "PlannerAssessment source Team authority drifted",
            )
        checkpoint = self.team_store.current_checkpoint(
            self.team_id,
        )
        if checkpoint is None or state.get("team_checkpoint_ref") != checkpoint.to_ref():
            raise FactoryGraphDriftError(
                "PlannerAssessment source checkpoint drifted",
            )
        heads = {
            value.head_id: value
            for value in self.team_store.current_artifact_heads(
                self.team_id,
            )
        }
        pending: list[ObjectRef] = []
        ready: list[ObjectRef] = []
        retryable: list[ObjectRef] = []
        terminal_failures: list[ObjectRef] = []
        invalidated: list[ObjectRef] = []
        approval: list[ObjectRef] = []
        unknown: list[ObjectRef] = []
        missing: list[ObjectRef] = []
        validator_refs: list[ObjectRef] = [checkpoint.to_ref()]
        for work in self.team_store.list_task_work(self.team_id):
            task_ref = work.task.to_ref()
            reason = work.latest_event.reason_code if work.latest_event is not None else None
            if work.latest_event is not None and work.latest_event.capability_result_ref is not None:
                validator_refs.append(
                    work.latest_event.capability_result_ref,
                )
            if work.task.task_kind in self.terminal_task_kinds:
                continue
            if reason is not None and "UNKNOWN" in reason:
                unknown.append(work.latest_event.to_ref() if work.latest_event is not None else task_ref)
            if reason is not None and ("APPROVAL" in reason or "AUTHORITY" in reason):
                approval.append(task_ref)
                continue
            if work.status is TeamTaskStatusV1.READY:
                ready.append(task_ref)
            elif work.status in {
                TeamTaskStatusV1.PENDING,
                TeamTaskStatusV1.CLAIMED,
                TeamTaskStatusV1.ACTIVE,
                TeamTaskStatusV1.BLOCKED,
            }:
                pending.append(task_ref)
            elif work.status is TeamTaskStatusV1.FAILED:
                if work.attempt < work.task.max_attempts:
                    retryable.append(task_ref)
                else:
                    terminal_failures.append(task_ref)
            elif work.status is TeamTaskStatusV1.CANCELLED:
                invalidated.append(task_ref)
            elif work.status is TeamTaskStatusV1.COMPLETED and any(
                head_id not in heads for head_id in work.task.output_artifact_head_ids
            ):
                missing.append(task_ref)
        conflicts = self.team_store.open_conflicts(self.team_id)
        facts = FactoryPlannerAssessmentFacts(
            run=run,
            compiled_plan=compiled,
            validator_result_refs=sorted_refs(validator_refs),
            pending_task_refs=sorted_refs(pending),
            ready_task_refs=sorted_refs(ready),
            retryable_task_refs=sorted_refs(retryable),
            terminal_failure_task_refs=sorted_refs(terminal_failures),
            invalidated_task_refs=sorted_refs(invalidated),
            approval_required_task_refs=sorted_refs(approval),
            unknown_outcome_refs=sorted_refs(unknown),
            open_conflict_refs=sorted_refs(value.to_ref() for value in conflicts),
            missing_output_refs=sorted_refs(missing),
            delivery_manifest_ref=run.delivery_manifest_ref,
        )
        return self.compiler.compile(
            facts,
            audit=self.audit,
        )


__all__ = ["FactoryPlannerAssessmentSource"]
