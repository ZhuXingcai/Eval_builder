from __future__ import annotations

import hashlib
from enum import StrEnum
from typing import Literal

from pydantic import Field, field_validator, model_validator

from eval_factory.contracts.core import (
    ContractAudit,
    ContractModel,
    Identifier,
    ObjectRef,
    Sha256,
)
from eval_factory.contracts.safety import (
    ContentRiskLabel,
    Disposition,
    OriginClass,
    ProvenanceDecision,
    TaintLabel,
)
from eval_factory.contracts.task import AttachmentCriticality
from eval_factory.contracts.task_v2 import (
    PromptLeakageCategoryV2,
    TaskDraftV2,
    TaskPromptSafetyFindingV2,
    TaskPromptSafetyGateV2,
    task_prompt_safety_gate_ref,
)

TASK_PROMPT_SAFETY_POLICY_VERSION: Literal["task-prompt-safety/r4-04-v1"] = "task-prompt-safety/r4-04-v1"
PROMPT_LEAKAGE_FINGERPRINT_POLICY_VERSION: Literal["task-prompt-leakage-fingerprint/r4-04-v1"] = (
    "task-prompt-leakage-fingerprint/r4-04-v1"
)
PROMPT_LEAKAGE_NORMALIZATION_VERSION: Literal["task-prompt-leakage-normalization/r4-04-v1"] = (
    "task-prompt-leakage-normalization/r4-04-v1"
)


class TaskPromptSafetyPolicyError(RuntimeError):
    pass


class TaskPromptSafetyOutcome(StrEnum):
    PASSED = "PASSED"
    BLOCKED = "BLOCKED"
    ABSTAIN = "ABSTAIN"
    BLOCKED_CAPABILITY = "BLOCKED_CAPABILITY"


class TaskPromptSafetyReason(StrEnum):
    AMBIGUOUS_TRAJECTORY = "AMBIGUOUS_TRAJECTORY"
    INCOMPLETE_REFERENCE_COVERAGE = "INCOMPLETE_REFERENCE_COVERAGE"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    MISSING_SEMANTIC_ASSESSMENT = "MISSING_SEMANTIC_ASSESSMENT"


class RestrictedPromptLeakageSource(ContractModel):
    schema_version: Literal["eval-factory/restricted-prompt-leakage-source/r4-04"] = (
        "eval-factory/restricted-prompt-leakage-source/r4-04"
    )
    source_id: Identifier
    category: PromptLeakageCategoryV2
    source_subject_ref: ObjectRef
    source_provenance_decision: ProvenanceDecision
    classification_evidence_ref: ObjectRef
    source_text: str | None = Field(default=None, min_length=1, max_length=100_000)
    logical_paths: tuple[str, ...] = ()
    content_sha256: Sha256
    capability_complete: bool

    @field_validator("category", mode="before")
    @classmethod
    def parse_category(cls, value: object) -> PromptLeakageCategoryV2:
        if isinstance(value, PromptLeakageCategoryV2):
            return value
        if isinstance(value, str):
            return PromptLeakageCategoryV2(value)
        raise TypeError("category must be a PromptLeakageCategoryV2")

    @field_validator("logical_paths", mode="before")
    @classmethod
    def normalize_logical_paths(cls, value: object) -> tuple[str, ...]:
        if isinstance(value, (tuple, list)):
            return tuple(str(item).replace("\\", "/") for item in value)
        raise TypeError("logical_paths must be a collection")

    @model_validator(mode="after")
    def validate_source(self) -> RestrictedPromptLeakageSource:
        if self.source_text is None and not self.logical_paths:
            raise ValueError("restricted leakage source requires text or logical paths")
        if self.source_text is not None and self.logical_paths:
            raise ValueError("restricted leakage source must separate text and path material")
        if len(self.logical_paths) != len(set(self.logical_paths)):
            raise ValueError("restricted leakage source logical paths must be unique")
        if any(not item or item.startswith("/") or ".." in item.split("/") for item in self.logical_paths):
            raise ValueError("restricted leakage source paths must be normalized relative paths")
        decision = self.source_provenance_decision
        if decision.subject_ref != self.source_subject_ref:
            raise ValueError("source provenance decision subject is mismatched")
        if decision.subject_sha256 != self.source_subject_ref.object_sha256:
            raise ValueError("source provenance decision hash is mismatched")
        material = self.source_text if self.source_text is not None else "\n".join(self.logical_paths)
        observed_hash = hashlib.sha256(material.encode()).hexdigest()
        if observed_hash != self.content_sha256:
            raise ValueError("restricted leakage source content hash is mismatched")
        if self.source_subject_ref.object_sha256 != self.content_sha256:
            raise ValueError("restricted leakage source ref hash is mismatched")
        if self.classification_evidence_ref.object_type != "safety-classification":
            raise ValueError("classification_evidence_ref must reference safety-classification")
        if decision.disposition not in {
            Disposition.QUARANTINE,
            Disposition.REJECT,
        }:
            raise ValueError("restricted leakage source must remain quarantined or rejected")
        _validate_category_provenance(self.category, decision)
        return self


class PromptDeterministicScan(ContractModel):
    schema_version: Literal["eval-factory/task-prompt-deterministic-scan/r4-04"] = (
        "eval-factory/task-prompt-deterministic-scan/r4-04"
    )
    scan_id: Identifier
    task_draft_ref: ObjectRef
    leakage_reference_set_ref: ObjectRef
    visible_prompt_sha256: Sha256
    complete_categories: frozenset[PromptLeakageCategoryV2]
    findings: tuple[TaskPromptSafetyFindingV2, ...] = ()
    policy_version: Literal["task-prompt-safety/r4-04-v1"] = TASK_PROMPT_SAFETY_POLICY_VERSION
    scan_sha256: Sha256
    audit: ContractAudit

    @field_validator("complete_categories", mode="before")
    @classmethod
    def parse_complete_categories(
        cls,
        value: object,
    ) -> frozenset[PromptLeakageCategoryV2]:
        if isinstance(value, (frozenset, set, tuple, list)):
            return frozenset(
                item if isinstance(item, PromptLeakageCategoryV2) else PromptLeakageCategoryV2(item)
                for item in value
            )
        raise TypeError("complete_categories must be a collection")

    @model_validator(mode="after")
    def validate_scan(self) -> PromptDeterministicScan:
        _require_ref_type(self.task_draft_ref, "task-draft", "task_draft_ref")
        _require_ref_type(
            self.leakage_reference_set_ref,
            "prompt-leakage-reference-set",
            "leakage_reference_set_ref",
        )
        finding_ids = tuple(item.finding_id for item in self.findings)
        if len(finding_ids) != len(set(finding_ids)):
            raise ValueError("deterministic scan finding IDs must be unique")
        return self


class TaskPromptSafetyRequirementView(ContractModel):
    schema_version: Literal["eval-factory/task-prompt-safety-requirement-view/r4-04"] = (
        "eval-factory/task-prompt-safety-requirement-view/r4-04"
    )
    requirement_id: Identifier
    statement: str = Field(min_length=1, max_length=4000)
    criticality: AttachmentCriticality


class TaskPromptSafetyRequest(ContractModel):
    schema_version: Literal["eval-factory/task-prompt-safety-request/r4-04"] = (
        "eval-factory/task-prompt-safety-request/r4-04"
    )
    request_id: Identifier
    task_draft_ref: ObjectRef
    leakage_reference_set_ref: ObjectRef
    deterministic_scan_ref: ObjectRef
    prompt_boundary_enforcement_ref: ObjectRef
    visible_prompt: str = Field(min_length=1, max_length=20_000)
    task_intent: str = Field(min_length=1, max_length=4000)
    evaluation_claim: str = Field(min_length=1, max_length=4000)
    required_capabilities: tuple[Identifier, ...]
    allowed_tools: tuple[Identifier, ...]
    forbidden_outputs: tuple[str, ...]
    requirement_views: tuple[TaskPromptSafetyRequirementView, ...]
    complete_categories: frozenset[PromptLeakageCategoryV2]
    deterministic_scan: PromptDeterministicScan
    model_profile: Identifier
    prompt_version: str = Field(min_length=1, max_length=128)
    untrusted_data_marker: Literal[True] = True
    policy_version: Literal["task-prompt-safety/r4-04-v1"] = TASK_PROMPT_SAFETY_POLICY_VERSION
    request_sha256: Sha256
    audit: ContractAudit

    @field_validator("complete_categories", mode="before")
    @classmethod
    def parse_complete_categories(
        cls,
        value: object,
    ) -> frozenset[PromptLeakageCategoryV2]:
        if isinstance(value, (frozenset, set, tuple, list)):
            return frozenset(
                item if isinstance(item, PromptLeakageCategoryV2) else PromptLeakageCategoryV2(item)
                for item in value
            )
        raise TypeError("complete_categories must be a collection")

    @model_validator(mode="after")
    def validate_request(self) -> TaskPromptSafetyRequest:
        _require_ref_type(self.task_draft_ref, "task-draft", "task_draft_ref")
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
        _require_ref_type(
            self.prompt_boundary_enforcement_ref,
            "prompt-boundary-enforcement",
            "prompt_boundary_enforcement_ref",
        )
        if self.deterministic_scan_ref != deterministic_scan_ref(self.deterministic_scan):
            raise ValueError("deterministic scan ref is mismatched")
        if self.deterministic_scan.task_draft_ref != self.task_draft_ref:
            raise ValueError("deterministic scan TaskDraft ref is mismatched")
        if self.deterministic_scan.leakage_reference_set_ref != self.leakage_reference_set_ref:
            raise ValueError("deterministic scan reference set ref is mismatched")
        for values, label in (
            (self.required_capabilities, "required capabilities"),
            (self.allowed_tools, "allowed tools"),
            (self.forbidden_outputs, "forbidden outputs"),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{label} must be unique")
        requirement_ids = tuple(item.requirement_id for item in self.requirement_views)
        if len(requirement_ids) != len(set(requirement_ids)):
            raise ValueError("requirement view IDs must be unique")
        return self


class TaskPromptSafetyFindingFixture(ContractModel):
    schema_version: Literal["eval-factory/task-prompt-safety-finding-fixture/r4-04"] = (
        "eval-factory/task-prompt-safety-finding-fixture/r4-04"
    )
    category: PromptLeakageCategoryV2
    prompt_span_start: int = Field(ge=0)
    prompt_span_end: int = Field(gt=0)
    rule_id: Identifier

    @field_validator("category", mode="before")
    @classmethod
    def parse_category(cls, value: object) -> PromptLeakageCategoryV2:
        if isinstance(value, PromptLeakageCategoryV2):
            return value
        if isinstance(value, str):
            return PromptLeakageCategoryV2(value)
        raise TypeError("category must be a PromptLeakageCategoryV2")

    @model_validator(mode="after")
    def validate_span(self) -> TaskPromptSafetyFindingFixture:
        if self.prompt_span_end <= self.prompt_span_start:
            raise ValueError("semantic prompt finding span must be ordered")
        return self


class FakeTaskPromptSafetyFixture(ContractModel):
    schema_version: Literal["eval-factory/fake-task-prompt-safety-fixture/r4-04"] = (
        "eval-factory/fake-task-prompt-safety-fixture/r4-04"
    )
    fixture_id: Identifier
    outcome: TaskPromptSafetyOutcome
    findings: tuple[TaskPromptSafetyFindingFixture, ...] = ()
    unresolved_reasons: frozenset[TaskPromptSafetyReason] = frozenset()
    model_available: bool = True

    @field_validator("outcome", mode="before")
    @classmethod
    def parse_outcome(cls, value: object) -> TaskPromptSafetyOutcome:
        return _parse_outcome(value)

    @field_validator("unresolved_reasons", mode="before")
    @classmethod
    def parse_reasons(
        cls,
        value: object,
    ) -> frozenset[TaskPromptSafetyReason]:
        return _parse_reasons(value)

    @model_validator(mode="after")
    def validate_fixture(self) -> FakeTaskPromptSafetyFixture:
        _validate_outcome_shape(
            self.outcome,
            self.findings,
            self.unresolved_reasons,
        )
        if not self.model_available and (
            self.outcome is not TaskPromptSafetyOutcome.BLOCKED_CAPABILITY
            or TaskPromptSafetyReason.MODEL_UNAVAILABLE not in self.unresolved_reasons
        ):
            raise ValueError("unavailable model requires BLOCKED_CAPABILITY/MODEL_UNAVAILABLE")
        return self


class TaskPromptSafetyProposal(ContractModel):
    schema_version: Literal["eval-factory/task-prompt-safety-proposal/r4-04"] = (
        "eval-factory/task-prompt-safety-proposal/r4-04"
    )
    proposal_id: Identifier
    request_ref: ObjectRef
    outcome: TaskPromptSafetyOutcome
    findings: tuple[TaskPromptSafetyFindingFixture, ...] = ()
    unresolved_reasons: frozenset[TaskPromptSafetyReason] = frozenset()
    model_profile: Identifier
    prompt_version: str = Field(min_length=1, max_length=128)
    policy_version: Literal["task-prompt-safety/r4-04-v1"] = TASK_PROMPT_SAFETY_POLICY_VERSION
    proposal_sha256: Sha256
    audit: ContractAudit

    @field_validator("outcome", mode="before")
    @classmethod
    def parse_outcome(cls, value: object) -> TaskPromptSafetyOutcome:
        return _parse_outcome(value)

    @field_validator("unresolved_reasons", mode="before")
    @classmethod
    def parse_reasons(
        cls,
        value: object,
    ) -> frozenset[TaskPromptSafetyReason]:
        return _parse_reasons(value)

    @model_validator(mode="after")
    def validate_proposal(self) -> TaskPromptSafetyProposal:
        _require_ref_type(
            self.request_ref,
            "task-prompt-safety-request",
            "request_ref",
        )
        _validate_outcome_shape(
            self.outcome,
            self.findings,
            self.unresolved_reasons,
        )
        return self


class TaskPromptSafetyResult(ContractModel):
    schema_version: Literal["eval-factory/task-prompt-safety-result/r4-04"] = (
        "eval-factory/task-prompt-safety-result/r4-04"
    )
    result_id: Identifier
    request_ref: ObjectRef
    proposal_ref: ObjectRef | None = None
    outcome: TaskPromptSafetyOutcome
    task_prompt_safety_gate: TaskPromptSafetyGateV2 | None = None
    task_draft: TaskDraftV2 | None = None
    unresolved_reasons: frozenset[TaskPromptSafetyReason] = frozenset()
    policy_version: Literal["task-prompt-safety/r4-04-v1"] = TASK_PROMPT_SAFETY_POLICY_VERSION
    result_sha256: Sha256
    audit: ContractAudit

    @field_validator("outcome", mode="before")
    @classmethod
    def parse_outcome(cls, value: object) -> TaskPromptSafetyOutcome:
        return _parse_outcome(value)

    @field_validator("unresolved_reasons", mode="before")
    @classmethod
    def parse_reasons(
        cls,
        value: object,
    ) -> frozenset[TaskPromptSafetyReason]:
        return _parse_reasons(value)

    @model_validator(mode="after")
    def validate_result(self) -> TaskPromptSafetyResult:
        _require_ref_type(
            self.request_ref,
            "task-prompt-safety-request",
            "request_ref",
        )
        if self.proposal_ref is not None:
            _require_ref_type(
                self.proposal_ref,
                "task-prompt-safety-assessment",
                "proposal_ref",
            )
        if self.outcome in {
            TaskPromptSafetyOutcome.PASSED,
            TaskPromptSafetyOutcome.BLOCKED,
        }:
            if self.task_prompt_safety_gate is None or self.task_draft is None:
                raise ValueError("terminal prompt safety outcome requires gate and TaskDraft")
            if self.unresolved_reasons:
                raise ValueError("terminal prompt safety outcome cannot carry unresolved reasons")
            expected_status = "PASSED" if self.outcome is TaskPromptSafetyOutcome.PASSED else "BLOCKED"
            if self.task_prompt_safety_gate.status.value != expected_status:
                raise ValueError("result outcome and prompt safety gate status must match")
            if self.task_draft.prompt_safety_status.value != expected_status:
                raise ValueError("result outcome and TaskDraft safety status must match")
            if self.task_draft.prompt_safety_gate_ref != task_prompt_safety_gate_ref(
                self.task_prompt_safety_gate
            ):
                raise ValueError("result TaskDraft must bind its prompt safety gate")
        else:
            if self.task_prompt_safety_gate is not None or self.task_draft is not None:
                raise ValueError("unresolved prompt safety outcome cannot carry artifacts")
            if not self.unresolved_reasons:
                raise ValueError("unresolved prompt safety outcome requires reasons")
        return self


def deterministic_scan_ref(scan: PromptDeterministicScan) -> ObjectRef:
    return ObjectRef(
        object_type="task-prompt-deterministic-scan",
        object_id=scan.scan_id,
        object_version=scan.policy_version,
        object_sha256=scan.scan_sha256,
    )


def _validate_category_provenance(
    category: PromptLeakageCategoryV2,
    decision: ProvenanceDecision,
) -> None:
    if category in {
        PromptLeakageCategoryV2.FINAL_ANSWER,
        PromptLeakageCategoryV2.COMPLETED_DELIVERABLE,
    } and not (
        decision.origin_class is OriginClass.AGENT_GENERATED_FINAL
        or TaintLabel.FINAL_OUTPUT_DERIVED in decision.taint_labels
        or ContentRiskLabel.ANSWER_BEARING in decision.content_risk_labels
    ):
        raise ValueError("final-answer source lacks final-output provenance")
    if (
        category is PromptLeakageCategoryV2.PRIVATE_REFERENCE
        and TaintLabel.PRIVATE_REFERENCE_DERIVED not in decision.taint_labels
    ):
        raise ValueError("private-reference source lacks private-reference provenance")
    if (
        category is PromptLeakageCategoryV2.GRADER_RULE
        and TaintLabel.GRADER_RULE_DERIVED not in decision.taint_labels
    ):
        raise ValueError("grader-rule source lacks grader provenance")
    if (
        category is PromptLeakageCategoryV2.HIDDEN_PASS_CONDITION
        and ContentRiskLabel.HIDDEN_PASS_CONDITION not in decision.content_risk_labels
    ):
        raise ValueError("hidden-pass source lacks hidden-pass risk")


def _parse_outcome(value: object) -> TaskPromptSafetyOutcome:
    if isinstance(value, TaskPromptSafetyOutcome):
        return value
    if isinstance(value, str):
        return TaskPromptSafetyOutcome(value)
    raise TypeError("outcome must be a TaskPromptSafetyOutcome")


def _parse_reasons(value: object) -> frozenset[TaskPromptSafetyReason]:
    if isinstance(value, (frozenset, set, tuple, list)):
        return frozenset(
            item if isinstance(item, TaskPromptSafetyReason) else TaskPromptSafetyReason(item)
            for item in value
        )
    raise TypeError("unresolved_reasons must be a collection")


def _validate_outcome_shape(
    outcome: TaskPromptSafetyOutcome,
    findings: tuple[TaskPromptSafetyFindingFixture, ...],
    reasons: frozenset[TaskPromptSafetyReason],
) -> None:
    if outcome is TaskPromptSafetyOutcome.PASSED:
        if findings or reasons:
            raise ValueError("PASSED outcome cannot carry findings or unresolved reasons")
        return
    if outcome is TaskPromptSafetyOutcome.BLOCKED:
        if not findings or reasons:
            raise ValueError("BLOCKED outcome requires findings and no unresolved reasons")
        return
    if findings:
        raise ValueError(f"{outcome.value} outcome cannot carry findings")
    if not reasons:
        raise ValueError(f"{outcome.value} outcome requires unresolved reasons")
    if outcome is TaskPromptSafetyOutcome.BLOCKED_CAPABILITY and not reasons.intersection(
        {
            TaskPromptSafetyReason.INCOMPLETE_REFERENCE_COVERAGE,
            TaskPromptSafetyReason.MODEL_UNAVAILABLE,
            TaskPromptSafetyReason.MISSING_SEMANTIC_ASSESSMENT,
        }
    ):
        raise ValueError("BLOCKED_CAPABILITY requires a capability reason")


def _require_ref_type(ref: ObjectRef, expected: str, field_name: str) -> None:
    if ref.object_type != expected:
        raise ValueError(f"{field_name} must reference {expected}")
