from __future__ import annotations

from enum import StrEnum
from typing import ClassVar, Literal, Self

from pydantic import Field, model_validator

from eval_factory.contracts.core import ContractAudit, Identifier, ObjectRef
from eval_factory.harness.contracts import (
    HarnessObjectV1,
    require_ref,
    require_sorted_unique,
    require_sorted_unique_refs,
    sorted_refs,
)


class GraphCheckpointPhaseV1(StrEnum):
    PRE_TRANSITION = "PRE_TRANSITION"
    POST_TRANSITION = "POST_TRANSITION"
    RECONCILED = "RECONCILED"


class GraphCheckpointOutcomeV1(StrEnum):
    READY = "READY"
    COMMITTED = "COMMITTED"
    WAITING_REVIEW = "WAITING_REVIEW"
    VERIFICATION_REQUIRED = "VERIFICATION_REQUIRED"
    BLOCKED = "BLOCKED"


class HarnessGraphExecutionBindingV1(HarnessObjectV1):
    schema_version: Literal["eval-harness/graph-execution-binding/v1"] = (
        "eval-harness/graph-execution-binding/v1"
    )
    OBJECT_TYPE: ClassVar[str] = "graph-execution-binding"

    binding_id: Identifier
    session_ref: ObjectRef
    requirement_ref: ObjectRef
    requirement_policy_ref: ObjectRef
    factory_run_ref: ObjectRef
    factory_request_ref: ObjectRef
    factory_policy_ref: ObjectRef
    expected_factory_run_version: int = Field(
        ge=0,
        le=100_000_000,
    )
    pack_manifest_ref: ObjectRef
    composition_ref: ObjectRef
    blueprint_ref: ObjectRef
    team_ref: ObjectRef
    roster_ref: ObjectRef
    task_graph_ref: ObjectRef
    execution_authority_ref: ObjectRef
    team_checkpoint_ref: ObjectRef
    expected_team_version: int = Field(ge=1, le=1_000_000_000)
    expected_task_graph_revision: int = Field(
        ge=1,
        le=1_000_000_000,
    )
    expected_authority_version: int = Field(
        ge=1,
        le=1_000_000_000,
    )
    blackboard_head_refs: tuple[ObjectRef, ...] = Field(
        default=(),
        max_length=100_000,
    )
    thread_id: Identifier
    max_transitions: int = Field(ge=1, le=100_000)

    @classmethod
    def create(  # type: ignore[override]
        cls,
        *,
        binding_id: str,
        session_ref: ObjectRef,
        requirement_ref: ObjectRef,
        requirement_policy_ref: ObjectRef,
        factory_run_ref: ObjectRef,
        factory_request_ref: ObjectRef,
        factory_policy_ref: ObjectRef,
        expected_factory_run_version: int,
        pack_manifest_ref: ObjectRef,
        composition_ref: ObjectRef,
        blueprint_ref: ObjectRef,
        team_ref: ObjectRef,
        roster_ref: ObjectRef,
        task_graph_ref: ObjectRef,
        execution_authority_ref: ObjectRef,
        team_checkpoint_ref: ObjectRef,
        expected_team_version: int,
        expected_task_graph_revision: int,
        expected_authority_version: int,
        blackboard_head_refs: tuple[ObjectRef, ...],
        thread_id: str,
        max_transitions: int,
        audit: ContractAudit,
    ) -> HarnessGraphExecutionBindingV1:
        return super().create(
            audit=audit,
            binding_id=binding_id,
            session_ref=session_ref,
            requirement_ref=requirement_ref,
            requirement_policy_ref=requirement_policy_ref,
            factory_run_ref=factory_run_ref,
            factory_request_ref=factory_request_ref,
            factory_policy_ref=factory_policy_ref,
            expected_factory_run_version=expected_factory_run_version,
            pack_manifest_ref=pack_manifest_ref,
            composition_ref=composition_ref,
            blueprint_ref=blueprint_ref,
            team_ref=team_ref,
            roster_ref=roster_ref,
            task_graph_ref=task_graph_ref,
            execution_authority_ref=execution_authority_ref,
            team_checkpoint_ref=team_checkpoint_ref,
            expected_team_version=expected_team_version,
            expected_task_graph_revision=expected_task_graph_revision,
            expected_authority_version=expected_authority_version,
            blackboard_head_refs=sorted_refs(blackboard_head_refs),
            thread_id=thread_id,
            max_transitions=max_transitions,
        )

    @model_validator(mode="after")
    def validate_binding(self) -> Self:
        expectations = (
            (self.session_ref, "harness-session", "session_ref"),
            (
                self.requirement_ref,
                "evaluation-requirement-spec",
                "requirement_ref",
            ),
            (
                self.requirement_policy_ref,
                "harness-requirement-policy",
                "requirement_policy_ref",
            ),
            (self.factory_run_ref, "factory-run", "factory_run_ref"),
            (
                self.factory_request_ref,
                "factory-dataset-run-request",
                "factory_request_ref",
            ),
            (
                self.factory_policy_ref,
                "factory-run-policy",
                "factory_policy_ref",
            ),
            (
                self.pack_manifest_ref,
                "eval-pack-manifest",
                "pack_manifest_ref",
            ),
            (
                self.composition_ref,
                "harness-composition",
                "composition_ref",
            ),
            (
                self.blueprint_ref,
                "evaluation-blueprint",
                "blueprint_ref",
            ),
            (self.team_ref, "agent-team", "team_ref"),
            (self.roster_ref, "team-roster", "roster_ref"),
            (
                self.task_graph_ref,
                "team-task-graph",
                "task_graph_ref",
            ),
            (
                self.execution_authority_ref,
                "execution-authority",
                "execution_authority_ref",
            ),
            (
                self.team_checkpoint_ref,
                "team-checkpoint",
                "team_checkpoint_ref",
            ),
        )
        for reference, object_type, label in expectations:
            require_ref(
                reference,
                object_type,
                label,
                object_version=reference.object_version,
            )
        require_sorted_unique_refs(
            self.blackboard_head_refs,
            "blackboard_head_refs",
        )
        if any(reference.object_type != "artifact-head" for reference in self.blackboard_head_refs):
            raise ValueError(
                "Graph binding Blackboard values must reference Artifact Heads",
            )
        return self


class HarnessGraphCheckpointV1(HarnessObjectV1):
    schema_version: Literal["eval-harness/graph-checkpoint/v1"] = "eval-harness/graph-checkpoint/v1"
    OBJECT_TYPE: ClassVar[str] = "graph-checkpoint"

    checkpoint_id: Identifier
    binding_ref: ObjectRef
    phase: GraphCheckpointPhaseV1
    node: Identifier
    transition_number: int = Field(ge=0, le=100_000)
    predecessor_checkpoint_ref: ObjectRef | None = None
    factory_run_ref: ObjectRef
    team_ref: ObjectRef
    team_checkpoint_ref: ObjectRef
    blackboard_head_refs: tuple[ObjectRef, ...] = Field(
        default=(),
        max_length=100_000,
    )
    planner_assessment_ref: ObjectRef | None = None
    outcome: GraphCheckpointOutcomeV1
    reason_codes: tuple[Identifier, ...] = Field(default=(), max_length=256)

    @classmethod
    def create(  # type: ignore[override]
        cls,
        *,
        checkpoint_id: str,
        binding_ref: ObjectRef,
        phase: GraphCheckpointPhaseV1,
        node: str,
        transition_number: int,
        predecessor_checkpoint_ref: ObjectRef | None,
        factory_run_ref: ObjectRef,
        team_ref: ObjectRef,
        team_checkpoint_ref: ObjectRef,
        blackboard_head_refs: tuple[ObjectRef, ...],
        planner_assessment_ref: ObjectRef | None,
        outcome: GraphCheckpointOutcomeV1,
        reason_codes: tuple[str, ...],
        audit: ContractAudit,
    ) -> HarnessGraphCheckpointV1:
        return super().create(
            audit=audit,
            checkpoint_id=checkpoint_id,
            binding_ref=binding_ref,
            phase=phase,
            node=node,
            transition_number=transition_number,
            predecessor_checkpoint_ref=predecessor_checkpoint_ref,
            factory_run_ref=factory_run_ref,
            team_ref=team_ref,
            team_checkpoint_ref=team_checkpoint_ref,
            blackboard_head_refs=sorted_refs(blackboard_head_refs),
            planner_assessment_ref=planner_assessment_ref,
            outcome=outcome,
            reason_codes=tuple(sorted(reason_codes)),
        )

    @model_validator(mode="after")
    def validate_checkpoint(self) -> Self:
        expectations = (
            (
                self.binding_ref,
                "graph-execution-binding",
                "binding_ref",
            ),
            (self.factory_run_ref, "factory-run", "factory_run_ref"),
            (self.team_ref, "agent-team", "team_ref"),
            (
                self.team_checkpoint_ref,
                "team-checkpoint",
                "team_checkpoint_ref",
            ),
        )
        for reference, object_type, label in expectations:
            require_ref(
                reference,
                object_type,
                label,
                object_version=reference.object_version,
            )
        if self.predecessor_checkpoint_ref is not None:
            require_ref(
                self.predecessor_checkpoint_ref,
                "graph-checkpoint",
                "predecessor_checkpoint_ref",
                object_version=self.predecessor_checkpoint_ref.object_version,
            )
        if self.planner_assessment_ref is not None:
            require_ref(
                self.planner_assessment_ref,
                "planner-assessment",
                "planner_assessment_ref",
                object_version=self.planner_assessment_ref.object_version,
            )
        require_sorted_unique_refs(
            self.blackboard_head_refs,
            "blackboard_head_refs",
        )
        if any(reference.object_type != "artifact-head" for reference in self.blackboard_head_refs):
            raise ValueError(
                "Graph checkpoint Blackboard values must reference Artifact Heads",
            )
        require_sorted_unique(self.reason_codes, "reason_codes")
        if self.phase is GraphCheckpointPhaseV1.PRE_TRANSITION:
            if self.outcome is not GraphCheckpointOutcomeV1.READY:
                raise ValueError(
                    "pre-transition checkpoint requires READY outcome",
                )
            if self.planner_assessment_ref is not None or self.reason_codes:
                raise ValueError(
                    "pre-transition checkpoint cannot claim result authority",
                )
        elif self.predecessor_checkpoint_ref is None:
            raise ValueError(
                "post/reconciled checkpoint requires a predecessor",
            )
        if (
            self.outcome
            in {
                GraphCheckpointOutcomeV1.VERIFICATION_REQUIRED,
                GraphCheckpointOutcomeV1.BLOCKED,
            }
            and not self.reason_codes
        ):
            raise ValueError(
                "blocked Graph checkpoint requires reason codes",
            )
        return self


__all__ = [
    "GraphCheckpointOutcomeV1",
    "GraphCheckpointPhaseV1",
    "HarnessGraphCheckpointV1",
    "HarnessGraphExecutionBindingV1",
]
