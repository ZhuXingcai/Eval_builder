from __future__ import annotations

import hashlib
from dataclasses import dataclass

from eval_factory.contracts.agent_system_v2 import (
    CompiledDatasetBuildPlanV2,
    FactoryRunV2,
    PlannerAssessmentActionV2,
    PlannerAssessmentV2,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.harness.contracts import sorted_refs


class FactoryPlannerAssessmentError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class FactoryPlannerAssessmentFacts:
    run: FactoryRunV2
    compiled_plan: CompiledDatasetBuildPlanV2
    validator_result_refs: tuple[ObjectRef, ...]
    pending_task_refs: tuple[ObjectRef, ...] = ()
    ready_task_refs: tuple[ObjectRef, ...] = ()
    retryable_task_refs: tuple[ObjectRef, ...] = ()
    terminal_failure_task_refs: tuple[ObjectRef, ...] = ()
    invalidated_task_refs: tuple[ObjectRef, ...] = ()
    approval_required_task_refs: tuple[ObjectRef, ...] = ()
    unknown_outcome_refs: tuple[ObjectRef, ...] = ()
    open_conflict_refs: tuple[ObjectRef, ...] = ()
    missing_output_refs: tuple[ObjectRef, ...] = ()
    delivery_manifest_ref: ObjectRef | None = None

    def __post_init__(self) -> None:
        for values in (
            self.validator_result_refs,
            self.pending_task_refs,
            self.ready_task_refs,
            self.retryable_task_refs,
            self.terminal_failure_task_refs,
            self.invalidated_task_refs,
            self.approval_required_task_refs,
            self.unknown_outcome_refs,
            self.open_conflict_refs,
            self.missing_output_refs,
        ):
            if values != sorted_refs(values):
                raise ValueError(
                    "Planner assessment fact refs must be sorted",
                )
        task_partitions = (
            set(self.pending_task_refs),
            set(self.ready_task_refs),
            set(self.retryable_task_refs),
            set(self.terminal_failure_task_refs),
            set(self.invalidated_task_refs),
            set(self.approval_required_task_refs),
        )
        if any(
            left & right
            for index, left in enumerate(task_partitions)
            for right in task_partitions[index + 1 :]
        ):
            raise ValueError(
                "Planner assessment task partitions must be disjoint",
            )


class FactoryPlannerAssessmentCompiler:
    def compile(
        self,
        facts: FactoryPlannerAssessmentFacts,
        *,
        audit: ContractAudit,
    ) -> PlannerAssessmentV2:
        self._validate_current(facts)
        action, affected, rationale = self._classify(facts)
        digest = hashlib.sha256(
            (
                f"{facts.run.object_sha256}:"
                f"{facts.compiled_plan.object_sha256}:"
                f"{action.value}:"
                f"{','.join(rationale)}"
            ).encode()
        ).hexdigest()
        return PlannerAssessmentV2.create(
            assessment_id=(f"planner-assessment://{facts.run.run_id}/{digest[:24]}"),
            run_ref=facts.run.to_ref(),
            compiled_plan_ref=facts.compiled_plan.to_ref(),
            validator_result_refs=facts.validator_result_refs,
            proposed_action=action,
            affected_task_refs=affected,
            rationale_codes=rationale,
            audit=audit,
        )

    @staticmethod
    def _validate_current(
        facts: FactoryPlannerAssessmentFacts,
    ) -> None:
        if facts.run.compiled_plan_ref != facts.compiled_plan.to_ref():
            raise FactoryPlannerAssessmentError(
                "Planner assessment compiled plan is not current",
            )
        if not facts.validator_result_refs:
            raise FactoryPlannerAssessmentError(
                "Planner assessment requires validator evidence",
            )
        if (
            facts.delivery_manifest_ref is not None
            and facts.run.delivery_manifest_ref != facts.delivery_manifest_ref
        ):
            raise FactoryPlannerAssessmentError(
                "Planner assessment delivery manifest is stale",
            )

    @staticmethod
    def _classify(
        facts: FactoryPlannerAssessmentFacts,
    ) -> tuple[
        PlannerAssessmentActionV2,
        tuple[ObjectRef, ...],
        tuple[str, ...],
    ]:
        if (
            facts.unknown_outcome_refs
            or facts.open_conflict_refs
            or facts.approval_required_task_refs
            or facts.terminal_failure_task_refs
        ):
            reasons: list[str] = []
            if facts.unknown_outcome_refs:
                reasons.append("UNKNOWN_OUTCOME_VERIFICATION_REQUIRED")
            if facts.open_conflict_refs:
                reasons.append("OPEN_TEAM_CONFLICT")
            if facts.approval_required_task_refs:
                reasons.append("USER_APPROVAL_REQUIRED")
            if facts.terminal_failure_task_refs:
                reasons.append("TERMINAL_WORK_FAILED")
            return (
                PlannerAssessmentActionV2.ESCALATE,
                sorted_refs(
                    (
                        *facts.approval_required_task_refs,
                        *facts.terminal_failure_task_refs,
                    )
                ),
                tuple(sorted(reasons)),
            )
        if facts.invalidated_task_refs:
            return (
                PlannerAssessmentActionV2.REPLAN,
                facts.invalidated_task_refs,
                ("CURRENT_PLAN_INVALIDATED",),
            )
        if facts.retryable_task_refs:
            return (
                PlannerAssessmentActionV2.RETRY,
                facts.retryable_task_refs,
                ("RETRYABLE_WORK_REMAINS",),
            )
        active = sorted_refs(
            (
                *facts.pending_task_refs,
                *facts.ready_task_refs,
            ),
        )
        if active:
            return (
                PlannerAssessmentActionV2.CONTINUE,
                active,
                ("APPROVED_WORK_REMAINS",),
            )
        if facts.missing_output_refs:
            return (
                PlannerAssessmentActionV2.CONTINUE,
                (),
                ("REQUIRED_OUTPUT_MISSING",),
            )
        if facts.delivery_manifest_ref is None:
            return (
                PlannerAssessmentActionV2.FINISH,
                (),
                ("EXECUTION_FRONTIER_COMPLETE",),
            )
        return (
            PlannerAssessmentActionV2.FINISH,
            (),
            ("ALL_REQUIRED_AUTHORITY_CURRENT",),
        )


__all__ = [
    "FactoryPlannerAssessmentCompiler",
    "FactoryPlannerAssessmentError",
    "FactoryPlannerAssessmentFacts",
]
