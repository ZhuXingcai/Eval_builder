from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from enum import StrEnum
from typing import Literal

from pydantic import Field, field_validator, model_validator

from eval_factory.contracts.approval import RewriteFidelity, TaskRewritePlan
from eval_factory.contracts.core import (
    ContractAudit,
    EvidenceRef,
    Identifier,
    ObjectRef,
    Sha256,
)
from eval_factory.contracts.core_v2 import ContractModelV2, canonical_value_v2
from eval_factory.contracts.task import (
    AttachmentCriticality,
    AttachmentDependency,
    EvaluationFailureClass,
    EvidencePriority,
    ReferenceMode,
    RequirementConflict,
    RubricVisibility,
)
from eval_factory.contracts.trace import ToolFamily

_DENIED_TASK_REFERENCE_MARKERS = frozenset(
    {
        "answer-bearing",
        "completed-deliverable",
        "configured-pii",
        "final-answer",
        "final-output",
        "grader-rule",
        "hidden-pass-condition",
        "hidden-selection-signal",
        "private-reference",
        "quarantine",
        "raw-trace",
        "raw-traj",
        "restricted-pii",
        "secret",
        "sensitive-pii",
        "trace-raw",
    }
)


class TaskEpisodeSegmentEvidenceBindingV2(ContractModelV2):
    schema_version: Literal["eval-factory/task-episode-segment-evidence-binding/v2"] = (
        "eval-factory/task-episode-segment-evidence-binding/v2"
    )
    segment_ref: ObjectRef
    evidence_ref_ids: tuple[Identifier, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_binding(self) -> TaskEpisodeSegmentEvidenceBindingV2:
        if self.segment_ref.object_type != "interaction-segment":
            raise ValueError("segment_ref must reference interaction-segment")
        if len(self.evidence_ref_ids) != len(set(self.evidence_ref_ids)):
            raise ValueError("evidence_ref_ids must be unique")
        return self


class TaskEpisodeV2(ContractModelV2):
    schema_version: Literal["eval-factory/task-episode/v2"] = "eval-factory/task-episode/v2"
    task_episode_id: Identifier
    selection_context_ref: ObjectRef
    trace_envelope_ref: ObjectRef
    segment_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    segment_evidence_bindings: tuple[TaskEpisodeSegmentEvidenceBindingV2, ...] = Field(min_length=1)
    rationale_ref: ObjectRef
    evidence_bundle_ref: ObjectRef
    model_profile: Identifier
    prompt_version: str = Field(min_length=1, max_length=128)
    policy_version: str = Field(min_length=1, max_length=128)
    task_episode_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_episode(self) -> TaskEpisodeV2:
        _require_ref_type(self.selection_context_ref, "selection-context", "selection_context_ref")
        _require_ref_type(self.trace_envelope_ref, "trace-envelope", "trace_envelope_ref")
        _require_ref_type(self.rationale_ref, "task-episode-rationale", "rationale_ref")
        _require_ref_type(self.evidence_bundle_ref, "evidence-bundle", "evidence_bundle_ref")
        for ref in self.segment_refs:
            _require_ref_type(ref, "interaction-segment", "segment_refs")
        segment_ids = [ref.object_id for ref in self.segment_refs]
        if len(segment_ids) != len(set(segment_ids)):
            raise ValueError("segment_refs must be unique")
        binding_refs = tuple(binding.segment_ref for binding in self.segment_evidence_bindings)
        if binding_refs != self.segment_refs:
            raise ValueError("segment_evidence_bindings must exactly match segment_refs in order")
        return self


def task_episode_ref(episode: TaskEpisodeV2) -> ObjectRef:
    return ObjectRef(
        object_type="task-episode",
        object_id=episode.task_episode_id,
        object_version="v2",
        object_sha256=episode.task_episode_sha256,
    )


class TaskDraftPromptSafetyStatusV2(StrEnum):
    PENDING = "PENDING"
    PASSED = "PASSED"
    BLOCKED = "BLOCKED"


class RubricJudgedObjectKindV2(StrEnum):
    CONTESTANT_RESPONSE = "CONTESTANT_RESPONSE"
    OUTPUT_ARTIFACT = "OUTPUT_ARTIFACT"
    WORKSPACE_STATE = "WORKSPACE_STATE"
    TOOL_BEHAVIOR = "TOOL_BEHAVIOR"
    STRUCTURED_VALUE = "STRUCTURED_VALUE"


class RubricSourceModeV2(StrEnum):
    IMPORTED = "IMPORTED"
    GENERATED = "GENERATED"


class EvaluatorExecutionModeV2(StrEnum):
    DETERMINISTIC = "DETERMINISTIC"
    MODEL = "MODEL"
    HUMAN_ONLY = "HUMAN_ONLY"


class EvaluatorFailureSignalV2(StrEnum):
    CONTESTANT_OUTPUT_MISSING = "CONTESTANT_OUTPUT_MISSING"
    CONTESTANT_OUTPUT_CONTRACT_VIOLATION = "CONTESTANT_OUTPUT_CONTRACT_VIOLATION"
    EVALUATOR_TIMEOUT = "EVALUATOR_TIMEOUT"
    EVALUATOR_RUNTIME_FAILURE = "EVALUATOR_RUNTIME_FAILURE"
    EVALUATOR_OUTPUT_CONTRACT_VIOLATION = "EVALUATOR_OUTPUT_CONTRACT_VIOLATION"
    ENVIRONMENT_STARTUP_FAILURE = "ENVIRONMENT_STARTUP_FAILURE"
    ENVIRONMENT_RUNTIME_FAILURE = "ENVIRONMENT_RUNTIME_FAILURE"
    REQUIRED_ENVIRONMENT_CAPABILITY_MISSING = "REQUIRED_ENVIRONMENT_CAPABILITY_MISSING"
    INSUFFICIENT_OBSERVATION = "INSUFFICIENT_OBSERVATION"
    CONFLICTING_FAILURE_SIGNALS = "CONFLICTING_FAILURE_SIGNALS"


class EvaluatorReferenceDataClassV2(StrEnum):
    NONE = "NONE"
    RESTRICTED_EVAL_CONTROL = "RESTRICTED_EVAL_CONTROL"
    PRIVATE_REFERENCE = "PRIVATE_REFERENCE"
    RESTRICTED_TRACE_SPAN = "RESTRICTED_TRACE_SPAN"
    HUMAN_ONLY_REFERENCE = "HUMAN_ONLY_REFERENCE"


class EvaluatorModelDomainV2(StrEnum):
    INTERNAL_APPROVED = "INTERNAL_APPROVED"
    INTERNAL_EVALUATOR_APPROVED = "INTERNAL_EVALUATOR_APPROVED"


class ToolRuleActionV2(StrEnum):
    ALLOW = "ALLOW"
    DENY = "DENY"


class ToolDenyReasonV2(StrEnum):
    NOT_REQUIRED = "NOT_REQUIRED"
    CATALOG_POLICY_DENIED = "CATALOG_POLICY_DENIED"


class PromptLeakageCategoryV2(StrEnum):
    FINAL_ANSWER = "FINAL_ANSWER"
    COMPLETED_DELIVERABLE = "COMPLETED_DELIVERABLE"
    PRIVATE_REFERENCE = "PRIVATE_REFERENCE"
    GRADER_RULE = "GRADER_RULE"
    HIDDEN_PASS_CONDITION = "HIDDEN_PASS_CONDITION"
    HIDDEN_SELECTION_SIGNAL = "HIDDEN_SELECTION_SIGNAL"
    TRAJECTORY_SPECIFIC_STEP = "TRAJECTORY_SPECIFIC_STEP"


TASK_PROMPT_SAFETY_REQUIRED_CATEGORIES_V2 = frozenset(PromptLeakageCategoryV2)


class PromptLeakageMatchKindV2(StrEnum):
    NORMALIZED_FULL_TEXT = "NORMALIZED_FULL_TEXT"
    TOKEN_WINDOW = "TOKEN_WINDOW"
    PATH_COMPONENT = "PATH_COMPONENT"


class PromptLeakageDetectorV2(StrEnum):
    DETERMINISTIC_FINGERPRINT = "DETERMINISTIC_FINGERPRINT"
    SEMANTIC_RESIDUAL = "SEMANTIC_RESIDUAL"


class TaskPromptSafetyCheckOutcomeV2(StrEnum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    NOT_EVALUATED = "NOT_EVALUATED"


class TaskPromptSafetyGateStatusV2(StrEnum):
    PASSED = "PASSED"
    BLOCKED = "BLOCKED"


class TaskRequirementLineageV2(ContractModelV2):
    schema_version: Literal["eval-factory/task-requirement-lineage/v2"] = (
        "eval-factory/task-requirement-lineage/v2"
    )
    requirement_id: Identifier
    statement: str = Field(min_length=1, max_length=4000)
    criticality: AttachmentCriticality
    evidence_priority: EvidencePriority
    evidence: tuple[EvidenceRef, ...] = Field(min_length=1)
    task_episode_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    conflict_status: RequirementConflict

    @model_validator(mode="after")
    def validate_lineage(self) -> TaskRequirementLineageV2:
        _require_unique(
            "requirement evidence IDs",
            tuple(item.evidence_ref_id for item in self.evidence),
        )
        if any(not _is_safe_task_ref(item.subject_ref) for item in self.evidence):
            raise ValueError("unsafe requirement evidence reference")
        for ref in self.task_episode_refs:
            _require_ref_type(ref, "task-episode", "task_episode_refs")
        _require_unique(
            "requirement task episode IDs",
            tuple(ref.object_id for ref in self.task_episode_refs),
        )
        return self


class TaskDraftV2(ContractModelV2):
    schema_version: Literal["eval-factory/task-draft/v2"] = "eval-factory/task-draft/v2"
    task_draft_id: Identifier
    task_version: int = Field(ge=1)
    supersedes_task_draft_ref: ObjectRef | None = None
    selection_context_ref: ObjectRef
    task_episode_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    visible_prompt: str = Field(min_length=1, max_length=20000)
    task_intent: str = Field(min_length=1, max_length=4000)
    evaluation_claim: str = Field(min_length=1, max_length=4000)
    required_capabilities: tuple[Identifier, ...] = Field(min_length=1)
    allowed_tools: tuple[Identifier, ...]
    forbidden_outputs: tuple[str, ...] = Field(min_length=1)
    attachment_dependencies: tuple[AttachmentDependency, ...]
    requirement_lineage: tuple[TaskRequirementLineageV2, ...] = Field(min_length=1)
    prompt_requirement_ids: tuple[Identifier, ...] = Field(min_length=1)
    uncertainties: tuple[str, ...] = ()
    prompt_safety_status: TaskDraftPromptSafetyStatusV2
    prompt_safety_gate_ref: ObjectRef | None = None
    model_profile: Identifier
    prompt_version: str = Field(min_length=1, max_length=128)
    policy_version: str = Field(min_length=1, max_length=128)
    task_draft_sha256: Sha256
    audit: ContractAudit

    @field_validator("prompt_safety_status", mode="before")
    @classmethod
    def parse_prompt_safety_status(
        cls,
        value: object,
    ) -> TaskDraftPromptSafetyStatusV2:
        if isinstance(value, TaskDraftPromptSafetyStatusV2):
            return value
        if isinstance(value, str):
            return TaskDraftPromptSafetyStatusV2(value)
        raise TypeError("prompt_safety_status must be a TaskDraftPromptSafetyStatusV2")

    @model_validator(mode="after")
    def validate_draft(self) -> TaskDraftV2:
        if self.task_version == 1 and self.supersedes_task_draft_ref is not None:
            raise ValueError("task version 1 cannot supersede another TaskDraft")
        if self.task_version > 1 and self.supersedes_task_draft_ref is None:
            raise ValueError("later task versions require a predecessor TaskDraft")
        if self.supersedes_task_draft_ref is not None:
            _require_ref_type(
                self.supersedes_task_draft_ref,
                "task-draft",
                "supersedes_task_draft_ref",
            )
        _require_ref_type(
            self.selection_context_ref,
            "selection-context",
            "selection_context_ref",
        )
        for ref in self.task_episode_refs:
            _require_ref_type(ref, "task-episode", "task_episode_refs")
        _require_unique(
            "TaskDraft task episode IDs",
            tuple(ref.object_id for ref in self.task_episode_refs),
        )
        _require_unique(
            "required_capabilities",
            self.required_capabilities,
        )
        _require_unique("allowed_tools", self.allowed_tools)
        _require_unique("forbidden_outputs", self.forbidden_outputs)

        dependency_ids = tuple(item.dependency_id for item in self.attachment_dependencies)
        _require_unique("attachment dependency IDs", dependency_ids)
        for dependency in self.attachment_dependencies:
            _require_unique(
                "attachment evidence IDs",
                tuple(item.evidence_ref_id for item in dependency.evidence),
            )
            if any(not _is_safe_task_ref(item.subject_ref) for item in dependency.evidence):
                raise ValueError("unsafe attachment evidence reference")
        requirement_ids = tuple(item.requirement_id for item in self.requirement_lineage)
        _require_unique("requirement IDs", requirement_ids)
        _require_unique("prompt requirement IDs", self.prompt_requirement_ids)
        requirement_id_set = set(requirement_ids)
        prompt_id_set = set(self.prompt_requirement_ids)
        if not prompt_id_set.issubset(requirement_id_set):
            raise ValueError("prompt requirement IDs must resolve to requirement lineage")
        critical_ids = {
            item.requirement_id
            for item in self.requirement_lineage
            if item.criticality is AttachmentCriticality.CRITICAL
        }
        if not critical_ids.issubset(prompt_id_set):
            raise ValueError("all critical requirements must be prompt requirements")
        if any(item.conflict_status is RequirementConflict.UNRESOLVED for item in self.requirement_lineage):
            raise ValueError("TaskDraft cannot contain unresolved requirement conflicts")

        episode_keys = {_ref_key(ref) for ref in self.task_episode_refs}
        for lineage in self.requirement_lineage:
            if any(_ref_key(ref) not in episode_keys for ref in lineage.task_episode_refs):
                raise ValueError("requirement task episodes must be present in TaskDraft task episodes")

        if self.prompt_safety_status is TaskDraftPromptSafetyStatusV2.PENDING:
            if self.prompt_safety_gate_ref is not None:
                raise ValueError("PENDING prompt safety cannot carry a gate ref")
        elif self.prompt_safety_gate_ref is None:
            raise ValueError("PASSED or BLOCKED prompt safety requires prompt_safety_gate_ref")
        else:
            _require_ref_type(
                self.prompt_safety_gate_ref,
                "task-prompt-safety-gate",
                "prompt_safety_gate_ref",
            )
        return self


def task_draft_ref(draft: TaskDraftV2) -> ObjectRef:
    return ObjectRef(
        object_type="task-draft",
        object_id=draft.task_draft_id,
        object_version="v2",
        object_sha256=draft.task_draft_sha256,
    )


def task_draft_carried_sha256(draft: TaskDraftV2) -> str:
    return task_draft_payload_sha256(
        {
            "task_version": draft.task_version,
            "supersedes_task_draft_ref": _maybe_ref_payload(draft.supersedes_task_draft_ref),
            "selection_context_ref": _ref_payload(draft.selection_context_ref),
            "task_episode_refs": [_ref_payload(ref) for ref in draft.task_episode_refs],
            "visible_prompt": draft.visible_prompt,
            "task_intent": draft.task_intent,
            "evaluation_claim": draft.evaluation_claim,
            "required_capabilities": list(draft.required_capabilities),
            "allowed_tools": list(draft.allowed_tools),
            "forbidden_outputs": list(draft.forbidden_outputs),
            "attachment_dependencies": [
                item.model_dump(mode="json", exclude_none=False) for item in draft.attachment_dependencies
            ],
            "requirement_lineage": [
                item.model_dump(mode="json", exclude_none=False) for item in draft.requirement_lineage
            ],
            "prompt_requirement_ids": list(draft.prompt_requirement_ids),
            "uncertainties": list(draft.uncertainties),
            "prompt_safety_status": draft.prompt_safety_status.value,
            "prompt_safety_gate_ref": _maybe_ref_payload(draft.prompt_safety_gate_ref),
            "model_profile": draft.model_profile,
            "prompt_version": draft.prompt_version,
            "policy_version": draft.policy_version,
        }
    )


def task_draft_payload_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        canonical_value_v2(dict(payload)),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


class RubricJudgedObjectV2(ContractModelV2):
    schema_version: Literal["eval-factory/rubric-judged-object/v2"] = "eval-factory/rubric-judged-object/v2"
    judged_object_id: Identifier
    kind: RubricJudgedObjectKindV2
    description: str = Field(min_length=1, max_length=2000)

    @field_validator("kind", mode="before")
    @classmethod
    def parse_kind(cls, value: object) -> RubricJudgedObjectKindV2:
        return _parse_enum(value, RubricJudgedObjectKindV2, "kind")


class RubricReachabilityV2(ContractModelV2):
    schema_version: Literal["eval-factory/rubric-reachability/v2"] = "eval-factory/rubric-reachability/v2"
    reachability_id: Identifier
    prompt_requirement_ids: tuple[Identifier, ...] = Field(min_length=1)
    attachment_dependency_ids: tuple[Identifier, ...] = ()
    allowed_tool_ids: tuple[Identifier, ...] = ()
    evidence: tuple[EvidenceRef, ...] = Field(min_length=1)
    capability_complete: Literal[True] = True
    reachability_sha256: Sha256

    @model_validator(mode="after")
    def validate_reachability(self) -> RubricReachabilityV2:
        _require_unique("prompt requirement IDs", self.prompt_requirement_ids)
        _require_unique("attachment dependency IDs", self.attachment_dependency_ids)
        _require_unique("allowed tool IDs", self.allowed_tool_ids)
        _require_unique(
            "evidence IDs",
            tuple(item.evidence_ref_id for item in self.evidence),
        )
        if any(not item.capability_complete for item in self.evidence):
            raise ValueError("rubric reachability evidence must be capability complete")
        if any(not _is_safe_task_ref(item.subject_ref) for item in self.evidence):
            raise ValueError("unsafe rubric reachability evidence reference")
        return self


class RubricCriterionV2(ContractModelV2):
    schema_version: Literal["eval-factory/rubric-criterion/v2"] = "eval-factory/rubric-criterion/v2"
    criterion_id: Identifier
    judged_object: RubricJudgedObjectV2
    description: str = Field(min_length=1, max_length=4000)
    weight: float = Field(gt=0)
    reachability: RubricReachabilityV2
    visibility: RubricVisibility
    evaluator_binding: Identifier
    approval_status: Literal["CANDIDATE"] = "CANDIDATE"
    criterion_sha256: Sha256

    @field_validator("visibility", mode="before")
    @classmethod
    def parse_visibility(cls, value: object) -> RubricVisibility:
        return _parse_enum(value, RubricVisibility, "visibility")


class RubricSetV2(ContractModelV2):
    schema_version: Literal["eval-factory/rubric-set/v2"] = "eval-factory/rubric-set/v2"
    rubric_set_id: Identifier
    rubric_version: int = Field(ge=1)
    supersedes_rubric_set_ref: ObjectRef | None = None
    task_draft_ref: ObjectRef
    source_mode: RubricSourceModeV2
    criteria: tuple[RubricCriterionV2, ...] = Field(min_length=1)
    total_weight: float = Field(gt=0)
    imported_from_ref: ObjectRef | None = None
    model_profile: Identifier | None = None
    prompt_version: str | None = Field(default=None, min_length=1, max_length=128)
    policy_version: str = Field(min_length=1, max_length=128)
    rubric_set_sha256: Sha256
    audit: ContractAudit

    @field_validator("source_mode", mode="before")
    @classmethod
    def parse_source_mode(cls, value: object) -> RubricSourceModeV2:
        return _parse_enum(value, RubricSourceModeV2, "source_mode")

    @model_validator(mode="after")
    def validate_rubric_set(self) -> RubricSetV2:
        _require_ref_type(self.task_draft_ref, "task-draft", "task_draft_ref")
        if self.task_draft_ref.object_version != "v2":
            raise ValueError("task_draft_ref must reference TaskDraft v2")
        if self.rubric_version == 1 and self.supersedes_rubric_set_ref is not None:
            raise ValueError("rubric version 1 cannot supersede another RubricSet")
        if self.rubric_version > 1 and self.supersedes_rubric_set_ref is None:
            raise ValueError("later rubric versions require a predecessor RubricSet")
        if self.supersedes_rubric_set_ref is not None:
            _require_ref_type(
                self.supersedes_rubric_set_ref,
                "rubric-set",
                "supersedes_rubric_set_ref",
            )
            if self.supersedes_rubric_set_ref.object_version != "v2":
                raise ValueError("supersedes_rubric_set_ref must reference RubricSet v2")
        if self.source_mode is RubricSourceModeV2.IMPORTED:
            if self.imported_from_ref is None:
                raise ValueError("IMPORTED RubricSet requires imported_from_ref")
            _require_ref_type(
                self.imported_from_ref,
                "imported-rubric",
                "imported_from_ref",
            )
            if self.model_profile is not None or self.prompt_version is not None:
                raise ValueError("IMPORTED RubricSet cannot carry model metadata")
        else:
            if self.imported_from_ref is not None:
                raise ValueError("GENERATED RubricSet cannot carry imported_from_ref")
            if self.model_profile is None or self.prompt_version is None:
                raise ValueError("GENERATED RubricSet requires model metadata")
        _require_unique(
            "criterion IDs",
            tuple(item.criterion_id for item in self.criteria),
        )
        _require_unique(
            "judged object IDs",
            tuple(item.judged_object.judged_object_id for item in self.criteria),
        )
        _require_unique(
            "criterion hashes",
            tuple(item.criterion_sha256 for item in self.criteria),
        )
        observed = sum(item.weight for item in self.criteria)
        if abs(observed - self.total_weight) > 1e-9:
            raise ValueError("rubric total_weight must equal criterion weights")
        return self


def rubric_reachability_carried_sha256(
    reachability: RubricReachabilityV2,
) -> str:
    return task_draft_payload_sha256(
        {
            "prompt_requirement_ids": list(reachability.prompt_requirement_ids),
            "attachment_dependency_ids": list(reachability.attachment_dependency_ids),
            "allowed_tool_ids": list(reachability.allowed_tool_ids),
            "evidence": [item.model_dump(mode="json", exclude_none=False) for item in reachability.evidence],
            "capability_complete": reachability.capability_complete,
        }
    )


def rubric_criterion_carried_sha256(criterion: RubricCriterionV2) -> str:
    return task_draft_payload_sha256(
        {
            "judged_object": criterion.judged_object.model_dump(
                mode="json",
                exclude_none=False,
            ),
            "description": criterion.description,
            "weight": criterion.weight,
            "reachability": criterion.reachability.model_dump(
                mode="json",
                exclude_none=False,
            ),
            "visibility": criterion.visibility.value,
            "evaluator_binding": criterion.evaluator_binding,
            "approval_status": criterion.approval_status,
        }
    )


def rubric_set_carried_sha256(rubric_set: RubricSetV2) -> str:
    return task_draft_payload_sha256(
        {
            "rubric_version": rubric_set.rubric_version,
            "supersedes_rubric_set_ref": _maybe_ref_payload(rubric_set.supersedes_rubric_set_ref),
            "task_draft_ref": _ref_payload(rubric_set.task_draft_ref),
            "source_mode": rubric_set.source_mode.value,
            "criteria": [item.model_dump(mode="json", exclude_none=False) for item in rubric_set.criteria],
            "total_weight": rubric_set.total_weight,
            "imported_from_ref": _maybe_ref_payload(rubric_set.imported_from_ref),
            "model_profile": rubric_set.model_profile,
            "prompt_version": rubric_set.prompt_version,
            "policy_version": rubric_set.policy_version,
        }
    )


def rubric_criterion_ref(criterion: RubricCriterionV2) -> ObjectRef:
    return ObjectRef(
        object_type="rubric-criterion",
        object_id=criterion.criterion_id,
        object_version="v2",
        object_sha256=criterion.criterion_sha256,
    )


def rubric_set_ref(rubric_set: RubricSetV2) -> ObjectRef:
    return ObjectRef(
        object_type="rubric-set",
        object_id=rubric_set.rubric_set_id,
        object_version="v2",
        object_sha256=rubric_set.rubric_set_sha256,
    )


class EvaluatorFailureRuleV2(ContractModelV2):
    schema_version: Literal["eval-factory/evaluator-failure-rule/v2"] = (
        "eval-factory/evaluator-failure-rule/v2"
    )
    rule_id: Identifier
    signal: EvaluatorFailureSignalV2
    failure_class: EvaluationFailureClass
    rule_sha256: Sha256

    @field_validator("signal", mode="before")
    @classmethod
    def parse_signal(cls, value: object) -> EvaluatorFailureSignalV2:
        return _parse_enum(value, EvaluatorFailureSignalV2, "signal")

    @field_validator("failure_class", mode="before")
    @classmethod
    def parse_failure_class(cls, value: object) -> EvaluationFailureClass:
        return _parse_enum(value, EvaluationFailureClass, "failure_class")


class EvaluatorBindingV2(ContractModelV2):
    schema_version: Literal["eval-factory/evaluator-binding/v2"] = "eval-factory/evaluator-binding/v2"
    evaluator_binding_id: Identifier
    criterion_ids: tuple[Identifier, ...] = Field(min_length=1)
    evaluator_type: Identifier
    execution_mode: EvaluatorExecutionModeV2
    evaluator_version: str = Field(min_length=1, max_length=128)
    input_contract_ref: ObjectRef
    output_contract_ref: ObjectRef
    evaluator_principal_id: Identifier | None = None
    model_profile_ref: ObjectRef | None = None
    reference_refs: tuple[ObjectRef, ...] = ()
    timeout_seconds: int = Field(gt=0)
    binding_sha256: Sha256

    @field_validator("execution_mode", mode="before")
    @classmethod
    def parse_execution_mode(cls, value: object) -> EvaluatorExecutionModeV2:
        return _parse_enum(value, EvaluatorExecutionModeV2, "execution_mode")

    @model_validator(mode="after")
    def validate_binding(self) -> EvaluatorBindingV2:
        _require_unique("criterion IDs", self.criterion_ids)
        _require_unique(
            "reference refs",
            tuple(_ref_key(ref) for ref in self.reference_refs),
        )
        _require_ref_type(
            self.input_contract_ref,
            "evaluator-input-contract",
            "input_contract_ref",
        )
        _require_ref_type(
            self.output_contract_ref,
            "evaluator-output-contract",
            "output_contract_ref",
        )
        if self.model_profile_ref is not None:
            _require_ref_type(
                self.model_profile_ref,
                "model-profile",
                "model_profile_ref",
            )
        if self.execution_mode is EvaluatorExecutionModeV2.DETERMINISTIC:
            if self.evaluator_principal_id is None:
                raise ValueError("DETERMINISTIC binding requires evaluator principal")
            if self.model_profile_ref is not None:
                raise ValueError("DETERMINISTIC binding cannot carry model profile")
        elif self.execution_mode is EvaluatorExecutionModeV2.MODEL:
            if self.evaluator_principal_id is None or self.model_profile_ref is None:
                raise ValueError("MODEL binding requires evaluator principal and model profile")
        elif self.evaluator_principal_id is not None or self.model_profile_ref is not None:
            raise ValueError("HUMAN_ONLY binding cannot carry automated principal or model profile")
        return self


class EvaluatorSpecV2(ContractModelV2):
    schema_version: Literal["eval-factory/evaluator-spec/v2"] = "eval-factory/evaluator-spec/v2"
    evaluator_spec_id: Identifier
    evaluator_spec_version: int = Field(ge=1)
    supersedes_evaluator_spec_ref: ObjectRef | None = None
    rubric_set_ref: ObjectRef
    bindings: tuple[EvaluatorBindingV2, ...] = Field(min_length=1)
    failure_rules: tuple[EvaluatorFailureRuleV2, ...] = Field(
        min_length=len(EvaluatorFailureSignalV2),
        max_length=len(EvaluatorFailureSignalV2),
    )
    policy_version: str = Field(min_length=1, max_length=128)
    evaluator_spec_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_evaluator_spec(self) -> EvaluatorSpecV2:
        _require_ref_type(self.rubric_set_ref, "rubric-set", "rubric_set_ref")
        if self.rubric_set_ref.object_version != "v2":
            raise ValueError("rubric_set_ref must reference RubricSet v2")
        if self.evaluator_spec_version == 1 and self.supersedes_evaluator_spec_ref is not None:
            raise ValueError("evaluator spec version 1 cannot supersede another EvaluatorSpec")
        if self.evaluator_spec_version > 1 and self.supersedes_evaluator_spec_ref is None:
            raise ValueError("later evaluator spec versions require a predecessor EvaluatorSpec")
        if self.supersedes_evaluator_spec_ref is not None:
            _require_ref_type(
                self.supersedes_evaluator_spec_ref,
                "evaluator-spec",
                "supersedes_evaluator_spec_ref",
            )
            if self.supersedes_evaluator_spec_ref.object_version != "v2":
                raise ValueError("supersedes_evaluator_spec_ref must reference EvaluatorSpec v2")
        _require_unique(
            "binding IDs",
            tuple(item.evaluator_binding_id for item in self.bindings),
        )
        _require_unique(
            "criterion IDs across bindings",
            tuple(criterion_id for binding in self.bindings for criterion_id in binding.criterion_ids),
        )
        _require_unique(
            "failure rule IDs",
            tuple(item.rule_id for item in self.failure_rules),
        )
        _require_unique(
            "failure rule hashes",
            tuple(item.rule_sha256 for item in self.failure_rules),
        )
        signals = tuple(item.signal for item in self.failure_rules)
        _require_unique("failure signals", signals)
        if frozenset(signals) != frozenset(EvaluatorFailureSignalV2):
            raise ValueError("failure signals must cover the complete required set")
        if frozenset(item.failure_class for item in self.failure_rules) != frozenset(EvaluationFailureClass):
            raise ValueError("failure rules must cover all evaluation failure classes")
        return self


class ReferencePolicyV2(ContractModelV2):
    schema_version: Literal["eval-factory/reference-policy/v2"] = "eval-factory/reference-policy/v2"
    reference_policy_id: Identifier
    reference_policy_version: int = Field(ge=1)
    supersedes_reference_policy_ref: ObjectRef | None = None
    evaluator_spec_ref: ObjectRef
    mode: ReferenceMode
    reference_data_class: EvaluatorReferenceDataClassV2
    reference_refs: tuple[ObjectRef, ...] = ()
    contestant_access: Literal[False] = False
    attachment_producer_access: Literal[False] = False
    evaluator_access: bool
    human_only_access: bool
    policy_version: str = Field(min_length=1, max_length=128)
    reference_policy_sha256: Sha256
    audit: ContractAudit

    @field_validator("mode", mode="before")
    @classmethod
    def parse_mode(cls, value: object) -> ReferenceMode:
        return _parse_enum(value, ReferenceMode, "mode")

    @field_validator("reference_data_class", mode="before")
    @classmethod
    def parse_reference_data_class(
        cls,
        value: object,
    ) -> EvaluatorReferenceDataClassV2:
        return _parse_enum(
            value,
            EvaluatorReferenceDataClassV2,
            "reference_data_class",
        )

    @model_validator(mode="after")
    def validate_reference_policy(self) -> ReferencePolicyV2:
        _require_ref_type(
            self.evaluator_spec_ref,
            "evaluator-spec",
            "evaluator_spec_ref",
        )
        if self.evaluator_spec_ref.object_version != "v2":
            raise ValueError("evaluator_spec_ref must reference EvaluatorSpec v2")
        if self.reference_policy_version == 1 and self.supersedes_reference_policy_ref is not None:
            raise ValueError("reference policy version 1 cannot supersede another ReferencePolicy")
        if self.reference_policy_version > 1 and self.supersedes_reference_policy_ref is None:
            raise ValueError("later reference policy versions require a predecessor ReferencePolicy")
        if self.supersedes_reference_policy_ref is not None:
            _require_ref_type(
                self.supersedes_reference_policy_ref,
                "reference-policy",
                "supersedes_reference_policy_ref",
            )
            if self.supersedes_reference_policy_ref.object_version != "v2":
                raise ValueError("supersedes_reference_policy_ref must reference ReferencePolicy v2")
        _require_unique(
            "reference refs",
            tuple(_ref_key(ref) for ref in self.reference_refs),
        )
        expected_data_class, expected_ref_type, evaluator_access, human_only_access = {
            ReferenceMode.NONE: (
                EvaluatorReferenceDataClassV2.NONE,
                None,
                False,
                False,
            ),
            ReferenceMode.STRUCTURED_EXPECTATIONS: (
                EvaluatorReferenceDataClassV2.RESTRICTED_EVAL_CONTROL,
                "structured-expectation",
                True,
                False,
            ),
            ReferenceMode.PRIVATE_ANSWER: (
                EvaluatorReferenceDataClassV2.PRIVATE_REFERENCE,
                "private-reference",
                True,
                False,
            ),
            ReferenceMode.TRACE_BEHAVIOR: (
                EvaluatorReferenceDataClassV2.RESTRICTED_TRACE_SPAN,
                "trace-behavior-reference",
                True,
                False,
            ),
            ReferenceMode.HUMAN_ONLY: (
                EvaluatorReferenceDataClassV2.HUMAN_ONLY_REFERENCE,
                "human-only-reference",
                False,
                True,
            ),
        }[self.mode]
        if self.reference_data_class is not expected_data_class:
            raise ValueError(f"{self.mode.value} requires {expected_data_class.value} reference data class")
        if expected_ref_type is None:
            if self.reference_refs:
                raise ValueError("NONE reference mode cannot carry reference refs")
        else:
            if not self.reference_refs:
                raise ValueError(f"{self.mode.value} requires reference refs")
            if any(ref.object_type != expected_ref_type for ref in self.reference_refs):
                raise ValueError(f"{self.mode.value} requires {expected_ref_type} refs")
        if self.evaluator_access is not evaluator_access:
            raise ValueError(f"{self.mode.value} evaluator_access must be {evaluator_access}")
        if self.human_only_access is not human_only_access:
            raise ValueError(f"{self.mode.value} human_only_access must be {human_only_access}")
        return self


class EvaluatorReferenceGrantV2(ContractModelV2):
    schema_version: Literal["eval-factory/evaluator-reference-grant/v2"] = (
        "eval-factory/evaluator-reference-grant/v2"
    )
    grant_id: Identifier
    evaluator_spec_ref: ObjectRef
    reference_policy_ref: ObjectRef
    evaluator_binding_id: Identifier
    evaluator_principal_id: Identifier
    model_profile_ref: ObjectRef | None = None
    model_domain_approval_ref: ObjectRef | None = None
    reference_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    purpose: Literal["EVALUATION"] = "EVALUATION"
    policy_version: str = Field(min_length=1, max_length=128)
    grant_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_grant(self) -> EvaluatorReferenceGrantV2:
        _require_ref_type(
            self.evaluator_spec_ref,
            "evaluator-spec",
            "evaluator_spec_ref",
        )
        if self.evaluator_spec_ref.object_version != "v2":
            raise ValueError("evaluator_spec_ref must reference EvaluatorSpec v2")
        _require_ref_type(
            self.reference_policy_ref,
            "reference-policy",
            "reference_policy_ref",
        )
        if self.reference_policy_ref.object_version != "v2":
            raise ValueError("reference_policy_ref must reference ReferencePolicy v2")
        _require_unique(
            "reference refs",
            tuple(_ref_key(ref) for ref in self.reference_refs),
        )
        if self.model_profile_ref is not None:
            _require_ref_type(
                self.model_profile_ref,
                "model-profile",
                "model_profile_ref",
            )
        if self.model_domain_approval_ref is not None:
            _require_ref_type(
                self.model_domain_approval_ref,
                "model-domain-approval",
                "model_domain_approval_ref",
            )
        if (self.model_profile_ref is None) != (self.model_domain_approval_ref is None):
            raise ValueError("model_profile_ref and model_domain_approval_ref must be paired")
        return self


def evaluator_failure_rule_carried_sha256(
    rule: EvaluatorFailureRuleV2,
) -> str:
    return task_draft_payload_sha256(
        {
            "signal": rule.signal.value,
            "failure_class": rule.failure_class.value,
        }
    )


def evaluator_binding_carried_sha256(binding: EvaluatorBindingV2) -> str:
    return task_draft_payload_sha256(
        {
            "evaluator_binding_id": binding.evaluator_binding_id,
            "criterion_ids": list(binding.criterion_ids),
            "evaluator_type": binding.evaluator_type,
            "execution_mode": binding.execution_mode.value,
            "evaluator_version": binding.evaluator_version,
            "input_contract_ref": _ref_payload(binding.input_contract_ref),
            "output_contract_ref": _ref_payload(binding.output_contract_ref),
            "evaluator_principal_id": binding.evaluator_principal_id,
            "model_profile_ref": _maybe_ref_payload(binding.model_profile_ref),
            "reference_refs": [_ref_payload(ref) for ref in binding.reference_refs],
            "timeout_seconds": binding.timeout_seconds,
        }
    )


def evaluator_spec_carried_sha256(evaluator_spec: EvaluatorSpecV2) -> str:
    return task_draft_payload_sha256(
        {
            "evaluator_spec_version": evaluator_spec.evaluator_spec_version,
            "supersedes_evaluator_spec_ref": _maybe_ref_payload(evaluator_spec.supersedes_evaluator_spec_ref),
            "rubric_set_ref": _ref_payload(evaluator_spec.rubric_set_ref),
            "bindings": [
                item.model_dump(mode="json", exclude_none=False) for item in evaluator_spec.bindings
            ],
            "failure_rules": [
                item.model_dump(mode="json", exclude_none=False) for item in evaluator_spec.failure_rules
            ],
            "policy_version": evaluator_spec.policy_version,
        }
    )


def reference_policy_carried_sha256(
    reference_policy: ReferencePolicyV2,
) -> str:
    return task_draft_payload_sha256(
        {
            "reference_policy_version": reference_policy.reference_policy_version,
            "supersedes_reference_policy_ref": _maybe_ref_payload(
                reference_policy.supersedes_reference_policy_ref
            ),
            "evaluator_spec_ref": _ref_payload(reference_policy.evaluator_spec_ref),
            "mode": reference_policy.mode.value,
            "reference_data_class": reference_policy.reference_data_class.value,
            "reference_refs": [_ref_payload(ref) for ref in reference_policy.reference_refs],
            "contestant_access": reference_policy.contestant_access,
            "attachment_producer_access": reference_policy.attachment_producer_access,
            "evaluator_access": reference_policy.evaluator_access,
            "human_only_access": reference_policy.human_only_access,
            "policy_version": reference_policy.policy_version,
        }
    )


def evaluator_reference_grant_carried_sha256(
    grant: EvaluatorReferenceGrantV2,
) -> str:
    return task_draft_payload_sha256(
        {
            "evaluator_spec_ref": _ref_payload(grant.evaluator_spec_ref),
            "reference_policy_ref": _ref_payload(grant.reference_policy_ref),
            "evaluator_binding_id": grant.evaluator_binding_id,
            "evaluator_principal_id": grant.evaluator_principal_id,
            "model_profile_ref": _maybe_ref_payload(grant.model_profile_ref),
            "model_domain_approval_ref": _maybe_ref_payload(grant.model_domain_approval_ref),
            "reference_refs": [_ref_payload(ref) for ref in grant.reference_refs],
            "purpose": grant.purpose,
            "policy_version": grant.policy_version,
        }
    )


def evaluator_spec_ref(evaluator_spec: EvaluatorSpecV2) -> ObjectRef:
    return ObjectRef(
        object_type="evaluator-spec",
        object_id=evaluator_spec.evaluator_spec_id,
        object_version="v2",
        object_sha256=evaluator_spec.evaluator_spec_sha256,
    )


def reference_policy_ref(reference_policy: ReferencePolicyV2) -> ObjectRef:
    return ObjectRef(
        object_type="reference-policy",
        object_id=reference_policy.reference_policy_id,
        object_version="v2",
        object_sha256=reference_policy.reference_policy_sha256,
    )


def evaluator_reference_grant_ref(
    grant: EvaluatorReferenceGrantV2,
) -> ObjectRef:
    return ObjectRef(
        object_type="evaluator-reference-grant",
        object_id=grant.grant_id,
        object_version="v2",
        object_sha256=grant.grant_sha256,
    )


class ToolRuleV2(ContractModelV2):
    schema_version: Literal["eval-factory/tool-rule/v2"] = "eval-factory/tool-rule/v2"
    rule_id: Identifier
    tool_id: Identifier
    tool_family: ToolFamily
    action: ToolRuleActionV2
    capability_ref: ObjectRef
    enforcement_profile_ref: ObjectRef
    constraint_profile_ref: ObjectRef
    deny_reason: ToolDenyReasonV2 | None = None
    rule_sha256: Sha256

    @field_validator("tool_family", mode="before")
    @classmethod
    def parse_tool_family(cls, value: object) -> ToolFamily:
        return _parse_enum(value, ToolFamily, "tool_family")

    @field_validator("action", mode="before")
    @classmethod
    def parse_action(cls, value: object) -> ToolRuleActionV2:
        return _parse_enum(value, ToolRuleActionV2, "action")

    @field_validator("deny_reason", mode="before")
    @classmethod
    def parse_deny_reason(cls, value: object) -> ToolDenyReasonV2 | None:
        if value is None:
            return None
        return _parse_enum(value, ToolDenyReasonV2, "deny_reason")

    @model_validator(mode="after")
    def validate_rule(self) -> ToolRuleV2:
        _require_ref_type(self.capability_ref, "tool-capability", "capability_ref")
        _require_ref_type(
            self.enforcement_profile_ref,
            "tool-enforcement-profile",
            "enforcement_profile_ref",
        )
        _require_ref_type(
            self.constraint_profile_ref,
            "tool-constraint-profile",
            "constraint_profile_ref",
        )
        if self.action is ToolRuleActionV2.ALLOW and self.deny_reason is not None:
            raise ValueError("ALLOW rule cannot carry deny_reason")
        if self.action is ToolRuleActionV2.DENY and self.deny_reason is None:
            raise ValueError("DENY rule requires deny_reason")
        return self


class ContestantToolRuleV2(ContractModelV2):
    schema_version: Literal["eval-factory/contestant-tool-rule/v2"] = "eval-factory/contestant-tool-rule/v2"
    rule_id: Identifier
    tool_id: Identifier
    tool_family: ToolFamily
    action: Literal["ALLOW"] = "ALLOW"
    contestant_descriptor_ref: ObjectRef
    contestant_constraint_profile_ref: ObjectRef
    rule_sha256: Sha256

    @field_validator("tool_family", mode="before")
    @classmethod
    def parse_tool_family(cls, value: object) -> ToolFamily:
        return _parse_enum(value, ToolFamily, "tool_family")

    @model_validator(mode="after")
    def validate_rule(self) -> ContestantToolRuleV2:
        _require_ref_type(
            self.contestant_descriptor_ref,
            "contestant-tool-descriptor",
            "contestant_descriptor_ref",
        )
        _require_ref_type(
            self.contestant_constraint_profile_ref,
            "contestant-tool-constraint-profile",
            "contestant_constraint_profile_ref",
        )
        return self


class ContestantToolPolicyV2(ContractModelV2):
    schema_version: Literal["eval-factory/contestant-tool-policy/v2"] = (
        "eval-factory/contestant-tool-policy/v2"
    )
    contestant_tool_policy_id: Identifier
    source_task_draft_sha256: Sha256
    rules: tuple[ContestantToolRuleV2, ...] = ()
    default_action: Literal["DENY"] = "DENY"
    policy_version: str = Field(min_length=1, max_length=128)
    contestant_tool_policy_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_policy(self) -> ContestantToolPolicyV2:
        _require_unique("contestant rule IDs", tuple(item.rule_id for item in self.rules))
        _require_unique("contestant tool IDs", tuple(item.tool_id for item in self.rules))
        _require_unique(
            "contestant rule hashes",
            tuple(item.rule_sha256 for item in self.rules),
        )
        if tuple(item.tool_id for item in self.rules) != tuple(sorted(item.tool_id for item in self.rules)):
            raise ValueError("contestant tool rules must be sorted by tool ID")
        return self


class ToolPolicyV2(ContractModelV2):
    schema_version: Literal["eval-factory/tool-policy/v2"] = "eval-factory/tool-policy/v2"
    tool_policy_id: Identifier
    tool_policy_version: int = Field(ge=1)
    supersedes_tool_policy_ref: ObjectRef | None = None
    task_draft_ref: ObjectRef
    rubric_set_ref: ObjectRef
    evaluator_spec_ref: ObjectRef
    tool_catalog_ref: ObjectRef
    control_boundary_enforcement_ref: ObjectRef
    rules: tuple[ToolRuleV2, ...] = Field(min_length=1)
    default_action: Literal["DENY"] = "DENY"
    contestant_projection_ref: ObjectRef
    policy_version: str = Field(min_length=1, max_length=128)
    tool_policy_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_policy(self) -> ToolPolicyV2:
        if self.tool_policy_version == 1 and self.supersedes_tool_policy_ref is not None:
            raise ValueError("tool policy version 1 cannot supersede another ToolPolicy")
        if self.tool_policy_version > 1 and self.supersedes_tool_policy_ref is None:
            raise ValueError("later tool policy versions require a predecessor ToolPolicy")
        if self.supersedes_tool_policy_ref is not None:
            _require_ref_type(
                self.supersedes_tool_policy_ref,
                "tool-policy",
                "supersedes_tool_policy_ref",
            )
            if self.supersedes_tool_policy_ref.object_version != "v2":
                raise ValueError("supersedes_tool_policy_ref must reference ToolPolicy v2")
        for ref, expected, display_name, field_name in (
            (self.task_draft_ref, "task-draft", "TaskDraft", "task_draft_ref"),
            (self.rubric_set_ref, "rubric-set", "RubricSet", "rubric_set_ref"),
            (
                self.evaluator_spec_ref,
                "evaluator-spec",
                "EvaluatorSpec",
                "evaluator_spec_ref",
            ),
        ):
            _require_ref_type(ref, expected, field_name)
            if ref.object_version != "v2":
                raise ValueError(f"{field_name} must reference {display_name} v2")
        _require_ref_type(
            self.tool_catalog_ref,
            "tool-capability-catalog",
            "tool_catalog_ref",
        )
        _require_ref_type(
            self.control_boundary_enforcement_ref,
            "prompt-boundary-enforcement",
            "control_boundary_enforcement_ref",
        )
        _require_ref_type(
            self.contestant_projection_ref,
            "contestant-tool-policy",
            "contestant_projection_ref",
        )
        if self.contestant_projection_ref.object_version != "v2":
            raise ValueError("contestant_projection_ref must reference ContestantToolPolicy v2")
        _require_unique("rule IDs", tuple(item.rule_id for item in self.rules))
        _require_unique("tool IDs", tuple(item.tool_id for item in self.rules))
        _require_unique("rule hashes", tuple(item.rule_sha256 for item in self.rules))
        if tuple(item.tool_id for item in self.rules) != tuple(sorted(item.tool_id for item in self.rules)):
            raise ValueError("tool rules must be sorted by tool ID")
        return self


def tool_rule_carried_sha256(rule: ToolRuleV2) -> str:
    return task_draft_payload_sha256(
        {
            "rule_id": rule.rule_id,
            "tool_id": rule.tool_id,
            "tool_family": rule.tool_family.value,
            "action": rule.action.value,
            "capability_ref": _ref_payload(rule.capability_ref),
            "enforcement_profile_ref": _ref_payload(rule.enforcement_profile_ref),
            "constraint_profile_ref": _ref_payload(rule.constraint_profile_ref),
            "deny_reason": rule.deny_reason.value if rule.deny_reason is not None else None,
        }
    )


def contestant_tool_rule_carried_sha256(rule: ContestantToolRuleV2) -> str:
    return task_draft_payload_sha256(
        {
            "rule_id": rule.rule_id,
            "tool_id": rule.tool_id,
            "tool_family": rule.tool_family.value,
            "action": rule.action,
            "contestant_descriptor_ref": _ref_payload(rule.contestant_descriptor_ref),
            "contestant_constraint_profile_ref": _ref_payload(rule.contestant_constraint_profile_ref),
        }
    )


def contestant_tool_policy_carried_sha256(policy: ContestantToolPolicyV2) -> str:
    return task_draft_payload_sha256(
        {
            "source_task_draft_sha256": policy.source_task_draft_sha256,
            "rules": [item.model_dump(mode="json", exclude_none=False) for item in policy.rules],
            "default_action": policy.default_action,
            "policy_version": policy.policy_version,
        }
    )


def tool_policy_carried_sha256(policy: ToolPolicyV2) -> str:
    return task_draft_payload_sha256(
        {
            "tool_policy_version": policy.tool_policy_version,
            "supersedes_tool_policy_ref": _maybe_ref_payload(policy.supersedes_tool_policy_ref),
            "task_draft_ref": _ref_payload(policy.task_draft_ref),
            "rubric_set_ref": _ref_payload(policy.rubric_set_ref),
            "evaluator_spec_ref": _ref_payload(policy.evaluator_spec_ref),
            "tool_catalog_ref": _ref_payload(policy.tool_catalog_ref),
            "control_boundary_enforcement_ref": _ref_payload(policy.control_boundary_enforcement_ref),
            "rules": [item.model_dump(mode="json", exclude_none=False) for item in policy.rules],
            "default_action": policy.default_action,
            "contestant_projection_ref": _ref_payload(policy.contestant_projection_ref),
            "policy_version": policy.policy_version,
        }
    )


def contestant_tool_policy_ref(policy: ContestantToolPolicyV2) -> ObjectRef:
    return ObjectRef(
        object_type="contestant-tool-policy",
        object_id=policy.contestant_tool_policy_id,
        object_version="v2",
        object_sha256=policy.contestant_tool_policy_sha256,
    )


def tool_policy_ref(policy: ToolPolicyV2) -> ObjectRef:
    return ObjectRef(
        object_type="tool-policy",
        object_id=policy.tool_policy_id,
        object_version="v2",
        object_sha256=policy.tool_policy_sha256,
    )


class ProducerAttachmentRequirementV2(ContractModelV2):
    schema_version: Literal["eval-factory/producer-attachment-requirement/v2"] = (
        "eval-factory/producer-attachment-requirement/v2"
    )
    dependency_id: Identifier
    description: str = Field(min_length=1, max_length=2000)
    criticality: AttachmentCriticality

    @field_validator("criticality", mode="before")
    @classmethod
    def parse_criticality(cls, value: object) -> AttachmentCriticality:
        return _parse_enum(value, AttachmentCriticality, "criticality")


class ProducerStorageAuthorizationV2(ContractModelV2):
    schema_version: Literal["eval-factory/producer-storage-authorization/v2"] = (
        "eval-factory/producer-storage-authorization/v2"
    )
    authorization_id: Identifier
    producer_principal_id: Identifier
    purpose: Literal["ATTACHMENT_PRODUCTION"] = "ATTACHMENT_PRODUCTION"
    source_task_draft_sha256: Sha256
    projection_policy_ref: ObjectRef
    evidence_bundle_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    authorized_subject_refs: tuple[ObjectRef, ...] = ()
    raw_store_access: Literal[False] = False
    canonical_store_access: Literal[False] = False
    quarantine_store_access: Literal[False] = False
    private_reference_store_access: Literal[False] = False
    credentials_issued: Literal[False] = False
    policy_version: str = Field(min_length=1, max_length=128)
    authorization_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_authorization(self) -> ProducerStorageAuthorizationV2:
        _require_ref_type(
            self.projection_policy_ref,
            "projection-policy",
            "projection_policy_ref",
        )
        for ref in self.evidence_bundle_refs:
            _require_ref_type(ref, "evidence-bundle", "evidence_bundle_refs")
        _require_unique(
            "evidence bundle refs",
            tuple(_ref_key(ref) for ref in self.evidence_bundle_refs),
        )
        _require_unique(
            "authorized subject refs",
            tuple(_ref_key(ref) for ref in self.authorized_subject_refs),
        )
        if tuple(_ref_key(ref) for ref in self.evidence_bundle_refs) != tuple(
            sorted(_ref_key(ref) for ref in self.evidence_bundle_refs)
        ):
            raise ValueError("evidence bundle refs must be sorted")
        if tuple(_ref_key(ref) for ref in self.authorized_subject_refs) != tuple(
            sorted(_ref_key(ref) for ref in self.authorized_subject_refs)
        ):
            raise ValueError("authorized subject refs must be sorted")
        if any(not _is_safe_task_ref(ref) for ref in self.authorized_subject_refs):
            raise ValueError("authorized subject refs must be safe projected subjects")
        if any(not ref.object_type.endswith("-projection") for ref in self.authorized_subject_refs):
            raise ValueError("authorized subject refs must reference projected evidence")
        return self


class ProducerTaskViewV2(ContractModelV2):
    schema_version: Literal["eval-factory/producer-task-view/v2"] = "eval-factory/producer-task-view/v2"
    producer_task_view_id: Identifier
    producer_task_view_version: int = Field(ge=1)
    supersedes_producer_task_view_ref: ObjectRef | None = None
    query_instruction: str = Field(min_length=1, max_length=20000)
    attachment_requirements: tuple[ProducerAttachmentRequirementV2, ...] = ()
    allowed_tools: tuple[Identifier, ...] = ()
    safe_evidence_bundle_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    forbidden_outputs: tuple[str, ...] = Field(min_length=1)
    projection_policy_ref: ObjectRef
    storage_authorization_ref: ObjectRef
    prompt_boundary_enforcement_ref: ObjectRef
    source_task_draft_sha256: Sha256
    source_contestant_tool_policy_sha256: Sha256
    source_contract_chain_sha256: Sha256
    policy_version: str = Field(min_length=1, max_length=128)
    producer_task_view_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_view(self) -> ProducerTaskViewV2:
        if self.producer_task_view_version == 1 and self.supersedes_producer_task_view_ref is not None:
            raise ValueError("producer task view version 1 cannot supersede another ProducerTaskView")
        if self.producer_task_view_version > 1 and self.supersedes_producer_task_view_ref is None:
            raise ValueError("later producer task view versions require a predecessor ProducerTaskView")
        if self.supersedes_producer_task_view_ref is not None:
            _require_ref_type(
                self.supersedes_producer_task_view_ref,
                "producer-task-view",
                "supersedes_producer_task_view_ref",
            )
            if self.supersedes_producer_task_view_ref.object_version != "v2":
                raise ValueError("supersedes_producer_task_view_ref must reference ProducerTaskView v2")
        for ref in self.safe_evidence_bundle_refs:
            _require_ref_type(
                ref,
                "evidence-bundle",
                "safe_evidence_bundle_refs",
            )
        _require_ref_type(
            self.projection_policy_ref,
            "projection-policy",
            "projection_policy_ref",
        )
        _require_ref_type(
            self.storage_authorization_ref,
            "producer-storage-authorization",
            "storage_authorization_ref",
        )
        if self.storage_authorization_ref.object_version != "v2":
            raise ValueError("storage_authorization_ref must reference ProducerStorageAuthorization v2")
        _require_ref_type(
            self.prompt_boundary_enforcement_ref,
            "prompt-boundary-enforcement",
            "prompt_boundary_enforcement_ref",
        )
        _require_unique(
            "attachment requirement IDs",
            tuple(item.dependency_id for item in self.attachment_requirements),
        )
        _require_unique("allowed tool IDs", self.allowed_tools)
        _require_unique(
            "safe evidence bundle refs",
            tuple(_ref_key(ref) for ref in self.safe_evidence_bundle_refs),
        )
        _require_unique("forbidden output values", self.forbidden_outputs)
        if tuple(item.dependency_id for item in self.attachment_requirements) != tuple(
            sorted(item.dependency_id for item in self.attachment_requirements)
        ):
            raise ValueError("attachment requirements must be sorted by dependency ID")
        if self.allowed_tools != tuple(sorted(self.allowed_tools)):
            raise ValueError("allowed tool IDs must be sorted")
        if tuple(_ref_key(ref) for ref in self.safe_evidence_bundle_refs) != tuple(
            sorted(_ref_key(ref) for ref in self.safe_evidence_bundle_refs)
        ):
            raise ValueError("safe evidence bundle refs must be sorted")
        return self


def producer_storage_authorization_carried_sha256(
    authorization: ProducerStorageAuthorizationV2,
) -> str:
    return task_draft_payload_sha256(
        {
            "producer_principal_id": authorization.producer_principal_id,
            "purpose": authorization.purpose,
            "source_task_draft_sha256": authorization.source_task_draft_sha256,
            "projection_policy_ref": _ref_payload(authorization.projection_policy_ref),
            "evidence_bundle_refs": [_ref_payload(ref) for ref in authorization.evidence_bundle_refs],
            "authorized_subject_refs": [_ref_payload(ref) for ref in authorization.authorized_subject_refs],
            "raw_store_access": authorization.raw_store_access,
            "canonical_store_access": authorization.canonical_store_access,
            "quarantine_store_access": authorization.quarantine_store_access,
            "private_reference_store_access": (authorization.private_reference_store_access),
            "credentials_issued": authorization.credentials_issued,
            "policy_version": authorization.policy_version,
        }
    )


def producer_task_view_carried_sha256(view: ProducerTaskViewV2) -> str:
    return task_draft_payload_sha256(
        {
            "producer_task_view_version": view.producer_task_view_version,
            "supersedes_producer_task_view_ref": _maybe_ref_payload(view.supersedes_producer_task_view_ref),
            "query_instruction": view.query_instruction,
            "attachment_requirements": [
                item.model_dump(mode="json", exclude_none=False) for item in view.attachment_requirements
            ],
            "allowed_tools": list(view.allowed_tools),
            "safe_evidence_bundle_refs": [_ref_payload(ref) for ref in view.safe_evidence_bundle_refs],
            "forbidden_outputs": list(view.forbidden_outputs),
            "projection_policy_ref": _ref_payload(view.projection_policy_ref),
            "storage_authorization_ref": _ref_payload(view.storage_authorization_ref),
            "prompt_boundary_enforcement_ref": _ref_payload(view.prompt_boundary_enforcement_ref),
            "source_task_draft_sha256": view.source_task_draft_sha256,
            "source_contestant_tool_policy_sha256": (view.source_contestant_tool_policy_sha256),
            "source_contract_chain_sha256": view.source_contract_chain_sha256,
            "policy_version": view.policy_version,
        }
    )


def producer_storage_authorization_ref(
    authorization: ProducerStorageAuthorizationV2,
) -> ObjectRef:
    return ObjectRef(
        object_type="producer-storage-authorization",
        object_id=authorization.authorization_id,
        object_version="v2",
        object_sha256=authorization.authorization_sha256,
    )


def producer_task_view_ref(view: ProducerTaskViewV2) -> ObjectRef:
    return ObjectRef(
        object_type="producer-task-view",
        object_id=view.producer_task_view_id,
        object_version="v2",
        object_sha256=view.producer_task_view_sha256,
    )


class PromptLeakageFingerprintV2(ContractModelV2):
    schema_version: Literal["eval-factory/prompt-leakage-fingerprint/v2"] = (
        "eval-factory/prompt-leakage-fingerprint/v2"
    )
    fingerprint_id: Identifier
    category: PromptLeakageCategoryV2
    source_subject_ref: ObjectRef
    source_provenance_decision_ref: ObjectRef
    classification_evidence_ref: ObjectRef
    match_kind: PromptLeakageMatchKindV2
    digest_sha256: Sha256
    token_count: int = Field(ge=1, le=20_000)
    normalization_version: str = Field(min_length=1, max_length=128)

    @field_validator("category", mode="before")
    @classmethod
    def parse_category(cls, value: object) -> PromptLeakageCategoryV2:
        return _parse_enum(value, PromptLeakageCategoryV2, "category")

    @field_validator("match_kind", mode="before")
    @classmethod
    def parse_match_kind(cls, value: object) -> PromptLeakageMatchKindV2:
        return _parse_enum(value, PromptLeakageMatchKindV2, "match_kind")

    @model_validator(mode="after")
    def validate_fingerprint(self) -> PromptLeakageFingerprintV2:
        _require_ref_type(
            self.source_provenance_decision_ref,
            "provenance-decision",
            "source_provenance_decision_ref",
        )
        _require_ref_type(
            self.classification_evidence_ref,
            "safety-classification",
            "classification_evidence_ref",
        )
        return self


def prompt_leakage_fingerprint_carried_sha256(
    fingerprint: PromptLeakageFingerprintV2,
) -> str:
    return task_draft_payload_sha256(
        {
            "category": fingerprint.category.value,
            "source_subject_ref": _ref_payload(fingerprint.source_subject_ref),
            "source_provenance_decision_ref": _ref_payload(fingerprint.source_provenance_decision_ref),
            "classification_evidence_ref": _ref_payload(fingerprint.classification_evidence_ref),
            "match_kind": fingerprint.match_kind.value,
            "digest_sha256": fingerprint.digest_sha256,
            "token_count": fingerprint.token_count,
            "normalization_version": fingerprint.normalization_version,
        }
    )


def prompt_leakage_fingerprint_ref(
    fingerprint: PromptLeakageFingerprintV2,
) -> ObjectRef:
    digest = prompt_leakage_fingerprint_carried_sha256(fingerprint)
    return ObjectRef(
        object_type="prompt-leakage-fingerprint",
        object_id=fingerprint.fingerprint_id,
        object_version="v2",
        object_sha256=digest,
    )


def validate_prompt_leakage_fingerprint_identity(
    fingerprint: PromptLeakageFingerprintV2,
) -> None:
    digest = prompt_leakage_fingerprint_carried_sha256(fingerprint)
    if fingerprint.fingerprint_id != (f"prompt-leakage-fingerprint://sha256/{digest}"):
        raise ValueError("prompt leakage fingerprint identity is stale")


class PromptLeakageReferenceSetV2(ContractModelV2):
    schema_version: Literal["eval-factory/prompt-leakage-reference-set/v2"] = (
        "eval-factory/prompt-leakage-reference-set/v2"
    )
    reference_set_id: Identifier
    trace_envelope_ref: ObjectRef
    fingerprints: tuple[PromptLeakageFingerprintV2, ...] = ()
    complete_categories: frozenset[PromptLeakageCategoryV2]
    fingerprint_policy_version: str = Field(min_length=1, max_length=128)
    reference_set_sha256: Sha256
    audit: ContractAudit

    @field_validator("complete_categories", mode="before")
    @classmethod
    def parse_complete_categories(
        cls,
        value: object,
    ) -> frozenset[PromptLeakageCategoryV2]:
        return _parse_enum_set(value, PromptLeakageCategoryV2, "complete_categories")

    @model_validator(mode="after")
    def validate_reference_set(self) -> PromptLeakageReferenceSetV2:
        _require_ref_type(
            self.trace_envelope_ref,
            "trace-envelope",
            "trace_envelope_ref",
        )
        _require_unique(
            "fingerprint IDs",
            tuple(item.fingerprint_id for item in self.fingerprints),
        )
        fingerprint_keys = tuple(
            (
                item.category,
                _ref_key(item.source_subject_ref),
                item.match_kind,
                item.digest_sha256,
                item.token_count,
            )
            for item in self.fingerprints
        )
        _require_unique("fingerprint digests", fingerprint_keys)
        return self


def prompt_leakage_reference_set_ref(
    reference_set: PromptLeakageReferenceSetV2,
) -> ObjectRef:
    return ObjectRef(
        object_type="prompt-leakage-reference-set",
        object_id=reference_set.reference_set_id,
        object_version="v2",
        object_sha256=reference_set.reference_set_sha256,
    )


class TaskPromptSafetyFindingV2(ContractModelV2):
    schema_version: Literal["eval-factory/task-prompt-safety-finding/v2"] = (
        "eval-factory/task-prompt-safety-finding/v2"
    )
    finding_id: Identifier
    category: PromptLeakageCategoryV2
    detector: PromptLeakageDetectorV2
    prompt_span_start: int = Field(ge=0)
    prompt_span_end: int = Field(gt=0)
    matched_fingerprint_ids: tuple[Identifier, ...] = ()
    rule_id: Identifier
    non_waivable: Literal[True] = True

    @field_validator("category", mode="before")
    @classmethod
    def parse_category(cls, value: object) -> PromptLeakageCategoryV2:
        return _parse_enum(value, PromptLeakageCategoryV2, "category")

    @field_validator("detector", mode="before")
    @classmethod
    def parse_detector(cls, value: object) -> PromptLeakageDetectorV2:
        return _parse_enum(value, PromptLeakageDetectorV2, "detector")

    @model_validator(mode="after")
    def validate_finding(self) -> TaskPromptSafetyFindingV2:
        if self.prompt_span_end <= self.prompt_span_start:
            raise ValueError("prompt finding span must be ordered")
        _require_unique(
            "matched fingerprint IDs",
            self.matched_fingerprint_ids,
        )
        if (
            self.detector is PromptLeakageDetectorV2.DETERMINISTIC_FINGERPRINT
            and not self.matched_fingerprint_ids
        ):
            raise ValueError("deterministic finding requires matched fingerprint IDs")
        if self.detector is PromptLeakageDetectorV2.SEMANTIC_RESIDUAL and self.matched_fingerprint_ids:
            raise ValueError("semantic finding cannot claim fingerprint matches")
        return self


class TaskPromptSafetyCheckV2(ContractModelV2):
    schema_version: Literal["eval-factory/task-prompt-safety-check/v2"] = (
        "eval-factory/task-prompt-safety-check/v2"
    )
    category: PromptLeakageCategoryV2
    outcome: TaskPromptSafetyCheckOutcomeV2
    finding_ids: tuple[Identifier, ...] = ()
    detector_ids: tuple[Identifier, ...] = ()

    @field_validator("category", mode="before")
    @classmethod
    def parse_category(cls, value: object) -> PromptLeakageCategoryV2:
        return _parse_enum(value, PromptLeakageCategoryV2, "category")

    @field_validator("outcome", mode="before")
    @classmethod
    def parse_outcome(cls, value: object) -> TaskPromptSafetyCheckOutcomeV2:
        return _parse_enum(value, TaskPromptSafetyCheckOutcomeV2, "outcome")

    @model_validator(mode="after")
    def validate_check(self) -> TaskPromptSafetyCheckV2:
        _require_unique("prompt safety check finding IDs", self.finding_ids)
        _require_unique("prompt safety check detector IDs", self.detector_ids)
        if self.outcome is TaskPromptSafetyCheckOutcomeV2.FAILED:
            if not self.finding_ids:
                raise ValueError("FAILED prompt safety check requires findings")
            if not self.detector_ids:
                raise ValueError("FAILED prompt safety check requires detectors")
        elif self.finding_ids:
            raise ValueError(f"{self.outcome.value} prompt safety check cannot carry findings")
        elif self.outcome is TaskPromptSafetyCheckOutcomeV2.PASSED and not self.detector_ids:
            raise ValueError("PASSED prompt safety check requires detectors")
        return self


class TaskPromptSafetyGateV2(ContractModelV2):
    schema_version: Literal["eval-factory/task-prompt-safety-gate/v2"] = (
        "eval-factory/task-prompt-safety-gate/v2"
    )
    task_prompt_safety_gate_id: Identifier
    source_task_draft_ref: ObjectRef
    leakage_reference_set_ref: ObjectRef
    deterministic_scan_ref: ObjectRef
    semantic_assessment_ref: ObjectRef | None = None
    visible_prompt_sha256: Sha256
    status: TaskPromptSafetyGateStatusV2
    checks: tuple[TaskPromptSafetyCheckV2, ...]
    findings: tuple[TaskPromptSafetyFindingV2, ...] = ()
    policy_version: str = Field(min_length=1, max_length=128)
    model_profile: Identifier | None = None
    prompt_version: str | None = Field(default=None, min_length=1, max_length=128)
    gate_sha256: Sha256
    audit: ContractAudit

    @field_validator("status", mode="before")
    @classmethod
    def parse_status(cls, value: object) -> TaskPromptSafetyGateStatusV2:
        return _parse_enum(value, TaskPromptSafetyGateStatusV2, "status")

    @model_validator(mode="after")
    def validate_gate(self) -> TaskPromptSafetyGateV2:
        _require_ref_type(
            self.source_task_draft_ref,
            "task-draft",
            "source_task_draft_ref",
        )
        _require_ref_type(
            self.leakage_reference_set_ref,
            "prompt-leakage-reference-set",
            "leakage_reference_set_ref",
        )
        _require_ref_type(
            self.deterministic_scan_ref,
            "task-prompt-deterministic-scan",
            "deterministic_scan_ref",
        )
        if self.semantic_assessment_ref is not None:
            _require_ref_type(
                self.semantic_assessment_ref,
                "task-prompt-safety-assessment",
                "semantic_assessment_ref",
            )
        categories = tuple(item.category for item in self.checks)
        _require_unique("prompt safety check categories", categories)
        if set(categories) != set(TASK_PROMPT_SAFETY_REQUIRED_CATEGORIES_V2):
            raise ValueError("gate requires exactly one check for every prompt safety category")
        finding_ids = tuple(item.finding_id for item in self.findings)
        _require_unique("prompt safety finding IDs", finding_ids)
        if self.status is TaskPromptSafetyGateStatusV2.PASSED and self.findings:
            raise ValueError("PASSED prompt safety gate cannot contain findings")
        if self.status is TaskPromptSafetyGateStatusV2.BLOCKED and not any(
            item.outcome is TaskPromptSafetyCheckOutcomeV2.FAILED for item in self.checks
        ):
            raise ValueError("BLOCKED prompt safety gate requires a failed check")
        finding_by_id = {item.finding_id: item for item in self.findings}
        referenced_finding_ids: set[str] = set()
        for check in self.checks:
            for finding_id in check.finding_ids:
                finding = finding_by_id.get(finding_id)
                if finding is None or finding.category is not check.category:
                    raise ValueError("prompt safety check finding binding is invalid")
                referenced_finding_ids.add(finding_id)
        if referenced_finding_ids != set(finding_ids):
            raise ValueError("every prompt safety finding must bind exactly one failed check")
        if (self.model_profile is None) != (self.prompt_version is None):
            raise ValueError("model profile and prompt version must be present together")
        if self.status is TaskPromptSafetyGateStatusV2.PASSED:
            if any(item.outcome is not TaskPromptSafetyCheckOutcomeV2.PASSED for item in self.checks):
                raise ValueError("PASSED prompt safety gate requires all checks passed")
            if self.semantic_assessment_ref is None:
                raise ValueError("PASSED prompt safety gate requires semantic assessment")
            if self.model_profile is None:
                raise ValueError("PASSED prompt safety gate requires model metadata")
        else:
            if not self.findings:
                raise ValueError("BLOCKED prompt safety gate requires findings")
        return self


def task_prompt_safety_gate_ref(gate: TaskPromptSafetyGateV2) -> ObjectRef:
    return ObjectRef(
        object_type="task-prompt-safety-gate",
        object_id=gate.task_prompt_safety_gate_id,
        object_version="v2",
        object_sha256=gate.gate_sha256,
    )


class TaskRewriteRebuildStageV2(StrEnum):
    TASK_DRAFT = "TASK_DRAFT"
    PROMPT_SAFETY = "PROMPT_SAFETY"
    RUBRIC = "RUBRIC"
    EVALUATOR_REFERENCE = "EVALUATOR_REFERENCE"
    TOOL_POLICY = "TOOL_POLICY"
    PRODUCER_VIEW = "PRODUCER_VIEW"


TASK_REWRITE_REBUILD_STAGES_V2 = tuple(TaskRewriteRebuildStageV2)
_TASK_REWRITE_INVALIDATED_OBJECT_TYPES = (
    "task-draft",
    "task-prompt-safety-gate",
    "rubric-set",
    "evaluator-spec",
    "reference-policy",
    "tool-policy",
    "contestant-tool-policy",
    "producer-storage-authorization",
    "producer-task-view",
)


class R4TaskContractSetV2(ContractModelV2):
    schema_version: Literal["eval-factory/r4-task-contract-set/v2"] = "eval-factory/r4-task-contract-set/v2"
    contract_set_id: Identifier
    task_draft_ref: ObjectRef
    task_prompt_safety_gate_ref: ObjectRef
    rubric_set_ref: ObjectRef
    evaluator_spec_ref: ObjectRef
    reference_policy_ref: ObjectRef
    tool_policy_ref: ObjectRef
    contestant_tool_policy_ref: ObjectRef
    producer_storage_authorization_ref: ObjectRef
    producer_task_view_ref: ObjectRef
    contract_set_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_contract_set(self) -> R4TaskContractSetV2:
        refs = (
            (self.task_draft_ref, "task-draft", "task_draft_ref"),
            (
                self.task_prompt_safety_gate_ref,
                "task-prompt-safety-gate",
                "task_prompt_safety_gate_ref",
            ),
            (self.rubric_set_ref, "rubric-set", "rubric_set_ref"),
            (self.evaluator_spec_ref, "evaluator-spec", "evaluator_spec_ref"),
            (
                self.reference_policy_ref,
                "reference-policy",
                "reference_policy_ref",
            ),
            (self.tool_policy_ref, "tool-policy", "tool_policy_ref"),
            (
                self.contestant_tool_policy_ref,
                "contestant-tool-policy",
                "contestant_tool_policy_ref",
            ),
            (
                self.producer_storage_authorization_ref,
                "producer-storage-authorization",
                "producer_storage_authorization_ref",
            ),
            (
                self.producer_task_view_ref,
                "producer-task-view",
                "producer_task_view_ref",
            ),
        )
        for ref, object_type, field_name in refs:
            _require_v2_ref(ref, object_type, field_name)
        return self


class TaskRewritePlanVersionV2(ContractModelV2):
    schema_version: Literal["eval-factory/task-rewrite-plan-version/v2"] = (
        "eval-factory/task-rewrite-plan-version/v2"
    )
    task_rewrite_plan_version_id: Identifier
    plan_version: int = Field(ge=1)
    supersedes_task_rewrite_plan_version_ref: ObjectRef | None = None
    plan: TaskRewritePlan
    basis_contract_set_ref: ObjectRef
    policy_version: str = Field(min_length=1, max_length=128)
    task_rewrite_plan_version_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_plan_version(self) -> TaskRewritePlanVersionV2:
        if self.plan_version == 1 and self.supersedes_task_rewrite_plan_version_ref is not None:
            raise ValueError("plan version 1 cannot supersede another plan version")
        if self.plan_version > 1 and self.supersedes_task_rewrite_plan_version_ref is None:
            raise ValueError("later plan versions require a predecessor")
        if self.supersedes_task_rewrite_plan_version_ref is not None:
            _require_v2_ref(
                self.supersedes_task_rewrite_plan_version_ref,
                "task-rewrite-plan-version",
                "supersedes_task_rewrite_plan_version_ref",
            )
        _require_ref_type(
            self.plan.selection_context_ref,
            "selection-context",
            "plan.selection_context_ref",
        )
        _require_v2_ref(
            self.basis_contract_set_ref,
            "r4-task-contract-set",
            "basis_contract_set_ref",
        )
        return self


class TaskRewriteExamplePreviewV2(ContractModelV2):
    schema_version: Literal["eval-factory/task-rewrite-example-preview/v2"] = (
        "eval-factory/task-rewrite-example-preview/v2"
    )
    example_id: Identifier
    kind: Literal["REWRITE"] = "REWRITE"
    input_summary: str = Field(min_length=1, max_length=4000)
    expected_treatment: str = Field(min_length=1, max_length=4000)


class TaskRewritePreviewSafetyGateV2(ContractModelV2):
    schema_version: Literal["eval-factory/task-rewrite-preview-safety-gate/v2"] = (
        "eval-factory/task-rewrite-preview-safety-gate/v2"
    )
    gate_id: Identifier
    preview_candidate_ref: ObjectRef
    leakage_reference_set_ref: ObjectRef
    preview_content_sha256: Sha256
    status: TaskPromptSafetyGateStatusV2
    checks: tuple[TaskPromptSafetyCheckV2, ...]
    findings: tuple[TaskPromptSafetyFindingV2, ...] = ()
    semantic_assessment_ref: ObjectRef | None = None
    model_profile: Identifier | None = None
    prompt_version: str | None = Field(default=None, min_length=1, max_length=128)
    policy_version: str = Field(min_length=1, max_length=128)
    gate_sha256: Sha256
    audit: ContractAudit

    @field_validator("status", mode="before")
    @classmethod
    def parse_status(cls, value: object) -> TaskPromptSafetyGateStatusV2:
        return _parse_enum(value, TaskPromptSafetyGateStatusV2, "status")

    @model_validator(mode="after")
    def validate_gate(self) -> TaskRewritePreviewSafetyGateV2:
        _require_v2_ref(
            self.preview_candidate_ref,
            "task-rewrite-preview-candidate",
            "preview_candidate_ref",
        )
        _require_v2_ref(
            self.leakage_reference_set_ref,
            "prompt-leakage-reference-set",
            "leakage_reference_set_ref",
        )
        if self.semantic_assessment_ref is not None:
            _require_ref_type(
                self.semantic_assessment_ref,
                "task-prompt-safety-assessment",
                "semantic_assessment_ref",
            )
        categories = tuple(item.category for item in self.checks)
        _require_unique("rewrite preview safety check categories", categories)
        if set(categories) != set(TASK_PROMPT_SAFETY_REQUIRED_CATEGORIES_V2):
            raise ValueError("rewrite preview gate requires exactly one check for every category")
        finding_ids = tuple(item.finding_id for item in self.findings)
        _require_unique("rewrite preview safety finding IDs", finding_ids)
        if self.status is TaskPromptSafetyGateStatusV2.PASSED and self.findings:
            raise ValueError("PASSED rewrite preview gate cannot contain findings")
        finding_by_id = {item.finding_id: item for item in self.findings}
        referenced_ids: set[str] = set()
        for check in self.checks:
            for finding_id in check.finding_ids:
                finding = finding_by_id.get(finding_id)
                if finding is None or finding.category is not check.category:
                    raise ValueError("rewrite preview safety finding binding is invalid")
                referenced_ids.add(finding_id)
        if referenced_ids != set(finding_ids):
            raise ValueError("every rewrite preview finding must bind exactly one failed check")
        if (self.model_profile is None) != (self.prompt_version is None):
            raise ValueError("model profile and prompt version must be paired")
        if self.status is TaskPromptSafetyGateStatusV2.PASSED:
            if any(item.outcome is not TaskPromptSafetyCheckOutcomeV2.PASSED for item in self.checks):
                raise ValueError("PASSED rewrite preview gate requires all checks passed")
            if self.semantic_assessment_ref is None:
                raise ValueError("PASSED rewrite preview gate requires semantic assessment")
            if self.model_profile is None:
                raise ValueError("PASSED rewrite preview gate requires model metadata")
        else:
            if not self.findings:
                raise ValueError("BLOCKED rewrite preview gate requires findings")
            if not any(item.outcome is TaskPromptSafetyCheckOutcomeV2.FAILED for item in self.checks):
                raise ValueError("BLOCKED rewrite preview gate requires a failed check")
        return self


class TaskRewritePlanPreviewV2(ContractModelV2):
    schema_version: Literal["eval-factory/task-rewrite-plan-preview/v2"] = (
        "eval-factory/task-rewrite-plan-preview/v2"
    )
    preview_id: Identifier
    source_plan_version_ref: ObjectRef
    target_capability: str = Field(min_length=1, max_length=4000)
    rewrite_style: str = Field(min_length=1, max_length=2000)
    fidelity: RewriteFidelity
    operational_noise_policy: str = Field(min_length=1, max_length=4000)
    examples: tuple[TaskRewriteExamplePreviewV2, ...] = Field(min_length=1)
    forbidden_content_rules: tuple[str, ...] = Field(min_length=1)
    expected_capability_impact: str = Field(min_length=1, max_length=2000)
    safety_gate_ref: ObjectRef
    projection_policy_version: str = Field(min_length=1, max_length=128)
    preview_sha256: Sha256
    audit: ContractAudit

    @field_validator("fidelity", mode="before")
    @classmethod
    def parse_fidelity(cls, value: object) -> RewriteFidelity:
        return _parse_enum(value, RewriteFidelity, "fidelity")

    @model_validator(mode="after")
    def validate_preview(self) -> TaskRewritePlanPreviewV2:
        _require_v2_ref(
            self.source_plan_version_ref,
            "task-rewrite-plan-version",
            "source_plan_version_ref",
        )
        _require_v2_ref(
            self.safety_gate_ref,
            "task-rewrite-preview-safety-gate",
            "safety_gate_ref",
        )
        _require_unique(
            "rewrite preview example IDs",
            tuple(item.example_id for item in self.examples),
        )
        _require_unique(
            "rewrite preview forbidden content rules",
            self.forbidden_content_rules,
        )
        return self


class TaskContractInvalidationV2(ContractModelV2):
    schema_version: Literal["eval-factory/task-contract-invalidation/v2"] = (
        "eval-factory/task-contract-invalidation/v2"
    )
    invalidation_id: Identifier
    source_plan_version_ref: ObjectRef
    replacement_plan_version_ref: ObjectRef
    source_contract_set_ref: ObjectRef
    changed_paths: tuple[str, ...] = Field(min_length=1)
    invalidated_object_refs: tuple[ObjectRef, ...] = Field(
        min_length=len(_TASK_REWRITE_INVALIDATED_OBJECT_TYPES),
        max_length=len(_TASK_REWRITE_INVALIDATED_OBJECT_TYPES),
    )
    preserved_object_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    required_rebuild_stages: tuple[TaskRewriteRebuildStageV2, ...] = Field(
        min_length=len(TASK_REWRITE_REBUILD_STAGES_V2),
        max_length=len(TASK_REWRITE_REBUILD_STAGES_V2),
    )
    hard_gate_revalidation_required: Literal[True] = True
    policy_version: str = Field(min_length=1, max_length=128)
    invalidation_sha256: Sha256
    audit: ContractAudit

    @field_validator("required_rebuild_stages", mode="before")
    @classmethod
    def parse_rebuild_stages(
        cls,
        value: object,
    ) -> tuple[TaskRewriteRebuildStageV2, ...]:
        if isinstance(value, (tuple, list)):
            return tuple(
                item if isinstance(item, TaskRewriteRebuildStageV2) else TaskRewriteRebuildStageV2(item)
                for item in value
            )
        raise TypeError("required_rebuild_stages must be a collection")

    @model_validator(mode="after")
    def validate_invalidation(self) -> TaskContractInvalidationV2:
        _require_v2_ref(
            self.source_plan_version_ref,
            "task-rewrite-plan-version",
            "source_plan_version_ref",
        )
        _require_v2_ref(
            self.replacement_plan_version_ref,
            "task-rewrite-plan-version",
            "replacement_plan_version_ref",
        )
        _require_v2_ref(
            self.source_contract_set_ref,
            "r4-task-contract-set",
            "source_contract_set_ref",
        )
        _require_unique("changed_paths", self.changed_paths)
        if self.changed_paths != tuple(sorted(self.changed_paths)):
            raise ValueError("changed_paths must be sorted")
        observed_types = tuple(item.object_type for item in self.invalidated_object_refs)
        if observed_types != _TASK_REWRITE_INVALIDATED_OBJECT_TYPES:
            raise ValueError("invalidated_object_refs must use the canonical invalidated order")
        if any(item.object_version != "v2" for item in self.invalidated_object_refs):
            raise ValueError("invalidated_object_refs must reference v2 objects")
        _require_unique(
            "invalidated object refs",
            tuple(_ref_key(item) for item in self.invalidated_object_refs),
        )
        _require_unique(
            "preserved object refs",
            tuple(_ref_key(item) for item in self.preserved_object_refs),
        )
        if self.required_rebuild_stages != TASK_REWRITE_REBUILD_STAGES_V2:
            raise ValueError("required_rebuild_stages must use the canonical rebuild order")
        return self


class TaskRewriteApplicationV2(ContractModelV2):
    schema_version: Literal["eval-factory/task-rewrite-application/v2"] = (
        "eval-factory/task-rewrite-application/v2"
    )
    application_id: Identifier
    source_plan_version_ref: ObjectRef
    replacement_plan_version_ref: ObjectRef
    invalidation_ref: ObjectRef
    source_contract_set_ref: ObjectRef
    replacement_contract_set_ref: ObjectRef
    candidate_state: Literal["CANDIDATE_TASK"] = "CANDIDATE_TASK"
    policy_version: str = Field(min_length=1, max_length=128)
    application_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_application(self) -> TaskRewriteApplicationV2:
        for ref, object_type, field_name in (
            (
                self.source_plan_version_ref,
                "task-rewrite-plan-version",
                "source_plan_version_ref",
            ),
            (
                self.replacement_plan_version_ref,
                "task-rewrite-plan-version",
                "replacement_plan_version_ref",
            ),
            (
                self.invalidation_ref,
                "task-contract-invalidation",
                "invalidation_ref",
            ),
            (
                self.source_contract_set_ref,
                "r4-task-contract-set",
                "source_contract_set_ref",
            ),
            (
                self.replacement_contract_set_ref,
                "r4-task-contract-set",
                "replacement_contract_set_ref",
            ),
        ):
            _require_v2_ref(ref, object_type, field_name)
        return self


def r4_task_contract_set_carried_sha256(
    contract_set: R4TaskContractSetV2,
) -> str:
    return task_draft_payload_sha256(
        {
            "task_draft_ref": _ref_payload(contract_set.task_draft_ref),
            "task_prompt_safety_gate_ref": _ref_payload(contract_set.task_prompt_safety_gate_ref),
            "rubric_set_ref": _ref_payload(contract_set.rubric_set_ref),
            "evaluator_spec_ref": _ref_payload(contract_set.evaluator_spec_ref),
            "reference_policy_ref": _ref_payload(contract_set.reference_policy_ref),
            "tool_policy_ref": _ref_payload(contract_set.tool_policy_ref),
            "contestant_tool_policy_ref": _ref_payload(contract_set.contestant_tool_policy_ref),
            "producer_storage_authorization_ref": _ref_payload(
                contract_set.producer_storage_authorization_ref
            ),
            "producer_task_view_ref": _ref_payload(contract_set.producer_task_view_ref),
        }
    )


def r4_task_contract_set_ref(
    contract_set: R4TaskContractSetV2,
) -> ObjectRef:
    return ObjectRef(
        object_type="r4-task-contract-set",
        object_id=contract_set.contract_set_id,
        object_version="v2",
        object_sha256=contract_set.contract_set_sha256,
    )


def task_rewrite_plan_carried_sha256(plan: TaskRewritePlan) -> str:
    return task_draft_payload_sha256(
        {
            "selection_context_ref": _ref_payload(plan.selection_context_ref),
            "target_capability": plan.target_capability,
            "rewrite_style": plan.rewrite_style,
            "fidelity": plan.fidelity.value,
            "operational_noise_policy": plan.operational_noise_policy,
            "examples": [
                {
                    "example_id": item.example_id,
                    "kind": item.kind.value,
                    "input_summary": item.input_summary,
                    "expected_treatment": item.expected_treatment,
                    "evidence_refs": [
                        evidence.model_dump(mode="json", exclude_none=False)
                        for evidence in sorted(
                            item.evidence_refs,
                            key=lambda evidence: evidence.evidence_ref_id,
                        )
                    ],
                }
                for item in plan.examples
            ],
            "forbidden_content_rules": sorted(plan.forbidden_content_rules),
            "expected_capability_impact": plan.expected_capability_impact,
        }
    )


def task_rewrite_plan_ref(plan: TaskRewritePlan) -> ObjectRef:
    digest = task_rewrite_plan_carried_sha256(plan)
    return ObjectRef(
        object_type="task-rewrite-plan",
        object_id=plan.task_rewrite_plan_id,
        object_version="v2",
        object_sha256=digest,
    )


def task_rewrite_plan_version_carried_sha256(
    version: TaskRewritePlanVersionV2,
) -> str:
    return task_draft_payload_sha256(
        {
            "plan_version": version.plan_version,
            "supersedes_task_rewrite_plan_version_ref": _maybe_ref_payload(
                version.supersedes_task_rewrite_plan_version_ref
            ),
            "plan_ref": _ref_payload(task_rewrite_plan_ref(version.plan)),
            "basis_contract_set_ref": _ref_payload(version.basis_contract_set_ref),
            "policy_version": version.policy_version,
        }
    )


def task_rewrite_plan_version_ref(
    version: TaskRewritePlanVersionV2,
) -> ObjectRef:
    return ObjectRef(
        object_type="task-rewrite-plan-version",
        object_id=version.task_rewrite_plan_version_id,
        object_version="v2",
        object_sha256=version.task_rewrite_plan_version_sha256,
    )


def task_rewrite_preview_safety_gate_carried_sha256(
    gate: TaskRewritePreviewSafetyGateV2,
) -> str:
    return task_draft_payload_sha256(
        {
            "preview_candidate_ref": _ref_payload(gate.preview_candidate_ref),
            "leakage_reference_set_ref": _ref_payload(gate.leakage_reference_set_ref),
            "preview_content_sha256": gate.preview_content_sha256,
            "status": gate.status.value,
            "checks": [item.model_dump(mode="json", exclude_none=False) for item in gate.checks],
            "findings": [item.model_dump(mode="json", exclude_none=False) for item in gate.findings],
            "semantic_assessment_ref": _maybe_ref_payload(gate.semantic_assessment_ref),
            "model_profile": gate.model_profile,
            "prompt_version": gate.prompt_version,
            "policy_version": gate.policy_version,
        }
    )


def task_rewrite_preview_safety_gate_ref(
    gate: TaskRewritePreviewSafetyGateV2,
) -> ObjectRef:
    return ObjectRef(
        object_type="task-rewrite-preview-safety-gate",
        object_id=gate.gate_id,
        object_version="v2",
        object_sha256=gate.gate_sha256,
    )


def task_rewrite_plan_preview_carried_sha256(
    preview: TaskRewritePlanPreviewV2,
) -> str:
    return task_draft_payload_sha256(
        {
            "source_plan_version_ref": _ref_payload(preview.source_plan_version_ref),
            "target_capability": preview.target_capability,
            "rewrite_style": preview.rewrite_style,
            "fidelity": preview.fidelity.value,
            "operational_noise_policy": preview.operational_noise_policy,
            "examples": [item.model_dump(mode="json", exclude_none=False) for item in preview.examples],
            "forbidden_content_rules": sorted(preview.forbidden_content_rules),
            "expected_capability_impact": (preview.expected_capability_impact),
            "safety_gate_ref": _ref_payload(preview.safety_gate_ref),
            "projection_policy_version": (preview.projection_policy_version),
        }
    )


def task_rewrite_plan_preview_ref(
    preview: TaskRewritePlanPreviewV2,
) -> ObjectRef:
    return ObjectRef(
        object_type="task-rewrite-plan-preview",
        object_id=preview.preview_id,
        object_version="v2",
        object_sha256=preview.preview_sha256,
    )


def task_contract_invalidation_carried_sha256(
    invalidation: TaskContractInvalidationV2,
) -> str:
    return task_draft_payload_sha256(
        {
            "source_plan_version_ref": _ref_payload(invalidation.source_plan_version_ref),
            "replacement_plan_version_ref": _ref_payload(invalidation.replacement_plan_version_ref),
            "source_contract_set_ref": _ref_payload(invalidation.source_contract_set_ref),
            "changed_paths": list(invalidation.changed_paths),
            "invalidated_object_refs": [_ref_payload(item) for item in invalidation.invalidated_object_refs],
            "preserved_object_refs": [
                _ref_payload(item)
                for item in sorted(
                    invalidation.preserved_object_refs,
                    key=_ref_key,
                )
            ],
            "required_rebuild_stages": [item.value for item in invalidation.required_rebuild_stages],
            "hard_gate_revalidation_required": (invalidation.hard_gate_revalidation_required),
            "policy_version": invalidation.policy_version,
        }
    )


def task_contract_invalidation_ref(
    invalidation: TaskContractInvalidationV2,
) -> ObjectRef:
    return ObjectRef(
        object_type="task-contract-invalidation",
        object_id=invalidation.invalidation_id,
        object_version="v2",
        object_sha256=invalidation.invalidation_sha256,
    )


def task_rewrite_application_carried_sha256(
    application: TaskRewriteApplicationV2,
) -> str:
    return task_draft_payload_sha256(
        {
            "source_plan_version_ref": _ref_payload(application.source_plan_version_ref),
            "replacement_plan_version_ref": _ref_payload(application.replacement_plan_version_ref),
            "invalidation_ref": _ref_payload(application.invalidation_ref),
            "source_contract_set_ref": _ref_payload(application.source_contract_set_ref),
            "replacement_contract_set_ref": _ref_payload(application.replacement_contract_set_ref),
            "candidate_state": application.candidate_state,
            "policy_version": application.policy_version,
        }
    )


def task_rewrite_application_ref(
    application: TaskRewriteApplicationV2,
) -> ObjectRef:
    return ObjectRef(
        object_type="task-rewrite-application",
        object_id=application.application_id,
        object_version="v2",
        object_sha256=application.application_sha256,
    )


def _require_ref_type(ref: ObjectRef, expected: str, field_name: str) -> None:
    if ref.object_type != expected:
        raise ValueError(f"{field_name} must reference {expected}")


def _require_v2_ref(
    ref: ObjectRef,
    expected: str,
    field_name: str,
) -> None:
    _require_ref_type(ref, expected, field_name)
    if ref.object_version != "v2":
        raise ValueError(f"{field_name} must reference {expected} v2")


def _require_unique(label: str, values: tuple[object, ...]) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must be unique")


def _ref_key(ref: ObjectRef) -> tuple[str, str, str, str]:
    return (
        ref.object_type,
        ref.object_id,
        ref.object_version,
        ref.object_sha256,
    )


def _is_safe_task_ref(ref: ObjectRef) -> bool:
    normalized = f"{ref.object_type}:{ref.object_id}".casefold().replace(
        "_",
        "-",
    )
    return not any(marker in normalized for marker in _DENIED_TASK_REFERENCE_MARKERS)


def _ref_payload(ref: ObjectRef) -> dict[str, object]:
    return ref.model_dump(mode="json", exclude_none=False)


def _maybe_ref_payload(ref: ObjectRef | None) -> dict[str, object] | None:
    if ref is None:
        return None
    return _ref_payload(ref)


def _parse_enum[T: StrEnum](
    value: object,
    enum_type: type[T],
    field_name: str,
) -> T:
    if isinstance(value, enum_type):
        return value
    if isinstance(value, str):
        return enum_type(value)
    raise TypeError(f"{field_name} must be a {enum_type.__name__}")


def _parse_enum_set[T: StrEnum](
    value: object,
    enum_type: type[T],
    field_name: str,
) -> frozenset[T]:
    if isinstance(value, (frozenset, set, tuple, list)):
        return frozenset(item if isinstance(item, enum_type) else enum_type(item) for item in value)
    raise TypeError(f"{field_name} must be a collection")
