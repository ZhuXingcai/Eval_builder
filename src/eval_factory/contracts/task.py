from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from eval_factory.contracts.core import (
    ContractAudit,
    ContractModel,
    EvidenceRef,
    Identifier,
    ObjectRef,
    RelativePath,
    Sha256,
)


class EvidencePriority(StrEnum):
    DIRECT_OBSERVATION = "DIRECT_OBSERVATION"
    EXPLICIT_REQUIREMENT = "EXPLICIT_REQUIREMENT"
    INFERRED_DEPENDENCY = "INFERRED_DEPENDENCY"


class RequirementConflict(StrEnum):
    NONE = "NONE"
    RESOLVED_BY_HIGHER_PRIORITY = "RESOLVED_BY_HIGHER_PRIORITY"
    UNRESOLVED = "UNRESOLVED"


class RequirementLineage(ContractModel):
    schema_version: Literal["eval-factory/requirement-lineage/v1"] = "eval-factory/requirement-lineage/v1"
    requirement_id: Identifier
    statement: str = Field(min_length=1, max_length=4000)
    evidence_priority: EvidencePriority
    evidence: tuple[EvidenceRef, ...] = Field(min_length=1)
    conflict_status: RequirementConflict


class AttachmentCriticality(StrEnum):
    CRITICAL = "CRITICAL"
    REQUIRED = "REQUIRED"
    OPTIONAL = "OPTIONAL"


class AttachmentDependency(ContractModel):
    schema_version: Literal["eval-factory/attachment-dependency/v1"] = "eval-factory/attachment-dependency/v1"
    dependency_id: Identifier
    description: str = Field(min_length=1, max_length=2000)
    criticality: AttachmentCriticality
    evidence_priority: EvidencePriority
    evidence: tuple[EvidenceRef, ...] = Field(min_length=1)


class TaskEpisode(ContractModel):
    schema_version: Literal["eval-factory/task-episode/v1"] = "eval-factory/task-episode/v1"
    task_episode_id: Identifier
    trace_envelope_ref: ObjectRef
    segment_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    rationale_ref: ObjectRef
    evidence_bundle_ref: ObjectRef
    audit: ContractAudit


class SelectionFirewall(ContractModel):
    schema_version: Literal["eval-factory/selection-firewall/v1"] = "eval-factory/selection-firewall/v1"
    final_answer_excluded: bool
    completed_deliverable_excluded: bool
    private_reference_excluded: bool
    grader_rules_excluded: bool
    hidden_selection_signals_excluded: bool
    trajectory_specific_steps_excluded: bool

    @property
    def passed(self) -> bool:
        return all(
            (
                self.final_answer_excluded,
                self.completed_deliverable_excluded,
                self.private_reference_excluded,
                self.grader_rules_excluded,
                self.hidden_selection_signals_excluded,
                self.trajectory_specific_steps_excluded,
            )
        )


class TaskDraftStatus(StrEnum):
    CANDIDATE = "CANDIDATE"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    BLOCKED = "BLOCKED"


class TaskDraft(ContractModel):
    schema_version: Literal["eval-factory/task-draft/v1"] = "eval-factory/task-draft/v1"
    task_draft_id: Identifier
    selection_context_ref: ObjectRef
    task_episode_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    visible_prompt: str = Field(min_length=1, max_length=20000)
    task_intent: str = Field(min_length=1, max_length=4000)
    evaluation_claim: str = Field(min_length=1, max_length=4000)
    required_capabilities: tuple[Identifier, ...] = Field(min_length=1)
    allowed_tools: tuple[Identifier, ...]
    forbidden_outputs: tuple[str, ...] = Field(min_length=1)
    attachment_dependencies: tuple[AttachmentDependency, ...]
    requirement_lineage: tuple[RequirementLineage, ...] = Field(min_length=1)
    uncertainties: tuple[str, ...] = ()
    selection_firewall: SelectionFirewall
    status: TaskDraftStatus
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_candidate(self) -> TaskDraft:
        if self.status is TaskDraftStatus.CANDIDATE:
            if not self.selection_firewall.passed:
                raise ValueError("candidate TaskDraft must pass Selection Firewall")
            if any(
                lineage.conflict_status is RequirementConflict.UNRESOLVED
                for lineage in self.requirement_lineage
            ):
                raise ValueError("candidate TaskDraft cannot contain unresolved conflicts")
        return self


class QuerySpec(ContractModel):
    schema_version: Literal["eval-factory/query-spec/v1"] = "eval-factory/query-spec/v1"
    query_spec_id: Identifier
    task_draft_ref: ObjectRef
    prompt: str = Field(min_length=1, max_length=20000)
    attachment_dependency_ids: tuple[Identifier, ...]
    contestant_visible: Literal[True] = True
    prompt_sha256: Sha256
    audit: ContractAudit


class EnvironmentArtifact(ContractModel):
    schema_version: Literal["eval-factory/environment-artifact/v1"] = "eval-factory/environment-artifact/v1"
    artifact_id: Identifier
    relative_path: RelativePath
    media_type: str = Field(min_length=3, max_length=255)
    content_sha256: Sha256
    size_bytes: int = Field(ge=0)
    criticality: AttachmentCriticality


class EnvironmentSpec(ContractModel):
    schema_version: Literal["eval-factory/environment-spec/v1"] = "eval-factory/environment-spec/v1"
    environment_spec_id: Identifier
    artifacts: tuple[EnvironmentArtifact, ...]
    package_manifest_ref: ObjectRef
    package_sha256: Sha256
    input_state_only: Literal[True] = True
    audit: ContractAudit


class RubricVisibility(StrEnum):
    PUBLIC = "PUBLIC"
    EVALUATOR_ONLY = "EVALUATOR_ONLY"
    HUMAN_ONLY = "HUMAN_ONLY"


class RubricCriterion(ContractModel):
    schema_version: Literal["eval-factory/rubric-criterion/v1"] = "eval-factory/rubric-criterion/v1"
    criterion_id: Identifier
    judged_object: Identifier
    description: str = Field(min_length=1, max_length=4000)
    weight: float = Field(gt=0)
    reachability_evidence: tuple[EvidenceRef, ...] = Field(min_length=1)
    visibility: RubricVisibility
    evaluator_binding: Identifier
    approval_status: Literal["CANDIDATE", "APPROVED", "REJECTED"]


class RubricSet(ContractModel):
    schema_version: Literal["eval-factory/rubric-set/v1"] = "eval-factory/rubric-set/v1"
    rubric_set_id: Identifier
    query_spec_ref: ObjectRef
    criteria: tuple[RubricCriterion, ...] = Field(min_length=1)
    total_weight: float = Field(gt=0)
    imported_from_ref: ObjectRef | None = None
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_weight(self) -> RubricSet:
        observed = sum(item.weight for item in self.criteria)
        if abs(observed - self.total_weight) > 1e-9:
            raise ValueError("rubric total_weight must equal criterion weights")
        return self


class EvaluationFailureClass(StrEnum):
    CONTESTANT_FAILURE = "CONTESTANT_FAILURE"
    EVALUATOR_FAILURE = "EVALUATOR_FAILURE"
    ENVIRONMENT_FAILURE = "ENVIRONMENT_FAILURE"
    INDETERMINATE = "INDETERMINATE"


class EvaluatorSpec(ContractModel):
    schema_version: Literal["eval-factory/evaluator-spec/v1"] = "eval-factory/evaluator-spec/v1"
    evaluator_spec_id: Identifier
    rubric_set_ref: ObjectRef
    evaluator_type: Identifier
    evaluator_version: str = Field(min_length=1, max_length=128)
    input_contract_ref: ObjectRef
    output_contract_ref: ObjectRef
    failure_classes: frozenset[EvaluationFailureClass] = Field(min_length=4, max_length=4)
    timeout_seconds: int = Field(gt=0)
    audit: ContractAudit


class ReferenceMode(StrEnum):
    NONE = "NONE"
    STRUCTURED_EXPECTATIONS = "STRUCTURED_EXPECTATIONS"
    PRIVATE_ANSWER = "PRIVATE_ANSWER"
    TRACE_BEHAVIOR = "TRACE_BEHAVIOR"
    HUMAN_ONLY = "HUMAN_ONLY"


class ReferencePolicy(ContractModel):
    schema_version: Literal["eval-factory/reference-policy/v1"] = "eval-factory/reference-policy/v1"
    reference_policy_id: Identifier
    mode: ReferenceMode
    reference_refs: tuple[ObjectRef, ...] = ()
    contestant_access: Literal[False] = False
    attachment_producer_access: Literal[False] = False
    evaluator_access: bool
    human_only_access: bool
    policy_version: str = Field(min_length=1, max_length=128)
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_mode(self) -> ReferencePolicy:
        if self.mode is ReferenceMode.NONE and self.reference_refs:
            raise ValueError("NONE reference mode cannot include reference objects")
        if self.mode is not ReferenceMode.NONE and not self.reference_refs:
            raise ValueError("non-NONE reference mode requires reference objects")
        return self


class ToolRule(ContractModel):
    schema_version: Literal["eval-factory/tool-rule/v1"] = "eval-factory/tool-rule/v1"
    tool: Identifier
    action: Literal["ALLOW", "DENY"]
    constraints: tuple[str, ...] = ()


class ToolPolicy(ContractModel):
    schema_version: Literal["eval-factory/tool-policy/v1"] = "eval-factory/tool-policy/v1"
    tool_policy_id: Identifier
    rules: tuple[ToolRule, ...]
    default_action: Literal["DENY"] = "DENY"
    contestant_projection_ref: ObjectRef
    policy_version: str = Field(min_length=1, max_length=128)
    audit: ContractAudit


class ProducerTaskView(ContractModel):
    schema_version: Literal["eval-factory/producer-task-view/v1"] = "eval-factory/producer-task-view/v1"
    producer_task_view_id: Identifier
    query_instruction: str = Field(min_length=1, max_length=12000)
    attachment_dependencies: tuple[AttachmentDependency, ...]
    allowed_tools: tuple[Identifier, ...]
    safe_evidence_bundle_refs: tuple[ObjectRef, ...]
    forbidden_outputs: tuple[str, ...] = Field(min_length=1)
    projection_policy_ref: ObjectRef
    source_task_draft_sha256: Sha256
    audit: ContractAudit
