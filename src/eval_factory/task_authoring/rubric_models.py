from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
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
from eval_factory.contracts.core_v2 import canonical_value_v2
from eval_factory.contracts.task import AttachmentCriticality, RubricVisibility
from eval_factory.contracts.task_v2 import (
    RubricJudgedObjectKindV2,
    RubricSetV2,
    RubricSourceModeV2,
)

RUBRIC_AUTHORING_POLICY_VERSION: Literal["rubric-authoring/r4-05-v1"] = "rubric-authoring/r4-05-v1"


class RubricAuthoringPolicyError(RuntimeError):
    pass


class RubricAuthoringOutcome(StrEnum):
    COMPILED = "COMPILED"
    ABSTAIN = "ABSTAIN"
    BLOCKED_REACHABILITY = "BLOCKED_REACHABILITY"
    BLOCKED_CAPABILITY = "BLOCKED_CAPABILITY"


class RubricAuthoringReason(StrEnum):
    AMBIGUOUS_RUBRIC = "AMBIGUOUS_RUBRIC"
    IMPORT_REJECTED = "IMPORT_REJECTED"
    INCOMPLETE_REACHABILITY_EVIDENCE = "INCOMPLETE_REACHABILITY_EVIDENCE"
    MISSING_ALLOWED_TOOL = "MISSING_ALLOWED_TOOL"
    MISSING_ATTACHMENT_DEPENDENCY = "MISSING_ATTACHMENT_DEPENDENCY"
    MISSING_CAPABILITY = "MISSING_CAPABILITY"
    MISSING_PROMPT_REQUIREMENT = "MISSING_PROMPT_REQUIREMENT"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"


class ImportedRubricStatus(StrEnum):
    USABLE = "USABLE"
    REJECTED = "REJECTED"
    AMBIGUOUS = "AMBIGUOUS"


class RubricTaskRequirementView(ContractModel):
    schema_version: Literal["eval-factory/rubric-task-requirement-view/r4-05"] = (
        "eval-factory/rubric-task-requirement-view/r4-05"
    )
    requirement_id: Identifier
    statement: str = Field(min_length=1, max_length=4000)
    criticality: AttachmentCriticality

    @field_validator("criticality", mode="before")
    @classmethod
    def parse_criticality(cls, value: object) -> AttachmentCriticality:
        return _parse_enum(value, AttachmentCriticality, "criticality")


class RubricAttachmentDependencyView(ContractModel):
    schema_version: Literal["eval-factory/rubric-attachment-dependency-view/r4-05"] = (
        "eval-factory/rubric-attachment-dependency-view/r4-05"
    )
    dependency_id: Identifier
    description: str = Field(min_length=1, max_length=2000)
    criticality: AttachmentCriticality

    @field_validator("criticality", mode="before")
    @classmethod
    def parse_criticality(cls, value: object) -> AttachmentCriticality:
        return _parse_enum(value, AttachmentCriticality, "criticality")


class RubricCandidateRequest(ContractModel):
    schema_version: Literal["eval-factory/rubric-candidate-request/r4-05"] = (
        "eval-factory/rubric-candidate-request/r4-05"
    )
    rubric_candidate_request_id: Identifier
    task_draft_ref: ObjectRef
    candidate_task_projection_ref: ObjectRef
    prompt_boundary_enforcement_ref: ObjectRef
    visible_prompt: str = Field(min_length=1, max_length=20000)
    task_intent: str = Field(min_length=1, max_length=4000)
    evaluation_claim: str = Field(min_length=1, max_length=4000)
    prompt_requirements: tuple[RubricTaskRequirementView, ...] = Field(min_length=1)
    attachment_dependencies: tuple[RubricAttachmentDependencyView, ...] = ()
    required_capability_ids: tuple[Identifier, ...] = Field(min_length=1)
    allowed_tool_ids: tuple[Identifier, ...] = ()
    forbidden_outputs: tuple[str, ...] = Field(min_length=1)
    allowed_evaluator_bindings: tuple[Identifier, ...] = Field(min_length=1)
    policy_version: Literal["rubric-authoring/r4-05-v1"] = RUBRIC_AUTHORING_POLICY_VERSION
    request_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_request(self) -> RubricCandidateRequest:
        _require_ref_type(self.task_draft_ref, "task-draft", "task_draft_ref")
        if self.task_draft_ref.object_version != "v2":
            raise ValueError("task_draft_ref must reference TaskDraft v2")
        _require_ref_type(
            self.candidate_task_projection_ref,
            "task-draft-rubric-input",
            "candidate_task_projection_ref",
        )
        if self.candidate_task_projection_ref.object_version != "v2":
            raise ValueError("candidate_task_projection_ref must reference rubric input v2")
        _require_ref_type(
            self.prompt_boundary_enforcement_ref,
            "prompt-boundary-enforcement",
            "prompt_boundary_enforcement_ref",
        )
        _require_unique(
            "prompt requirement",
            (item.requirement_id for item in self.prompt_requirements),
        )
        _require_unique(
            "attachment dependency",
            (item.dependency_id for item in self.attachment_dependencies),
        )
        _require_unique("required capability", self.required_capability_ids)
        _require_unique("allowed tool", self.allowed_tool_ids)
        _require_unique("forbidden output", self.forbidden_outputs)
        _require_unique("allowed evaluator binding", self.allowed_evaluator_bindings)
        return self


class RubricCriterionSelection(ContractModel):
    schema_version: Literal["eval-factory/rubric-criterion-selection/r4-05"] = (
        "eval-factory/rubric-criterion-selection/r4-05"
    )
    selection_id: Identifier
    judged_object_id: Identifier
    judged_object_kind: RubricJudgedObjectKindV2
    judged_object_description: str = Field(min_length=1, max_length=2000)
    description: str = Field(min_length=1, max_length=4000)
    weight: float = Field(gt=0)
    prompt_requirement_ids: tuple[Identifier, ...] = Field(min_length=1)
    attachment_dependency_ids: tuple[Identifier, ...] = ()
    allowed_tool_ids: tuple[Identifier, ...] = ()
    visibility: RubricVisibility
    evaluator_binding: Identifier

    @field_validator("judged_object_kind", mode="before")
    @classmethod
    def parse_judged_object_kind(cls, value: object) -> RubricJudgedObjectKindV2:
        return _parse_enum(value, RubricJudgedObjectKindV2, "judged_object_kind")

    @field_validator("visibility", mode="before")
    @classmethod
    def parse_visibility(cls, value: object) -> RubricVisibility:
        return _parse_enum(value, RubricVisibility, "visibility")

    @model_validator(mode="after")
    def validate_selection(self) -> RubricCriterionSelection:
        _require_unique("prompt requirement", self.prompt_requirement_ids)
        _require_unique("attachment dependency", self.attachment_dependency_ids)
        _require_unique("allowed tool", self.allowed_tool_ids)
        return self


class ImportedRubricDefinition(ContractModel):
    schema_version: Literal["eval-factory/imported-rubric-definition/r4-05"] = (
        "eval-factory/imported-rubric-definition/r4-05"
    )
    imported_rubric_ref: ObjectRef
    status: ImportedRubricStatus
    criteria: tuple[RubricCriterionSelection, ...] = Field(min_length=1)
    definition_sha256: Sha256
    audit: ContractAudit

    @field_validator("status", mode="before")
    @classmethod
    def parse_status(cls, value: object) -> ImportedRubricStatus:
        return _parse_enum(value, ImportedRubricStatus, "status")

    @model_validator(mode="after")
    def validate_import(self) -> ImportedRubricDefinition:
        _require_ref_type(
            self.imported_rubric_ref,
            "imported-rubric",
            "imported_rubric_ref",
        )
        _validate_unique_selections(self.criteria)
        return self


def imported_rubric_definition_sha256(
    imported: ImportedRubricDefinition,
) -> str:
    payload = {
        "imported_rubric_ref": imported.imported_rubric_ref.model_dump(
            mode="json",
            exclude_none=False,
        ),
        "status": imported.status.value,
        "criteria": [item.model_dump(mode="json", exclude_none=False) for item in imported.criteria],
    }
    encoded = json.dumps(
        canonical_value_v2(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


class FakeRubricGenerationFixture(ContractModel):
    schema_version: Literal["eval-factory/fake-rubric-generation-fixture/r4-05"] = (
        "eval-factory/fake-rubric-generation-fixture/r4-05"
    )
    fixture_id: Identifier
    outcome: RubricAuthoringOutcome
    criteria: tuple[RubricCriterionSelection, ...] = ()
    unresolved_reasons: frozenset[RubricAuthoringReason] = frozenset()
    model_available: bool = True

    @field_validator("outcome", mode="before")
    @classmethod
    def parse_outcome(cls, value: object) -> RubricAuthoringOutcome:
        return _parse_enum(value, RubricAuthoringOutcome, "outcome")

    @field_validator("unresolved_reasons", mode="before")
    @classmethod
    def parse_reasons(
        cls,
        value: object,
    ) -> frozenset[RubricAuthoringReason]:
        return _parse_enum_set(value, RubricAuthoringReason, "unresolved_reasons")

    @model_validator(mode="after")
    def validate_fixture(self) -> FakeRubricGenerationFixture:
        _validate_proposal_shape(
            self.outcome,
            self.criteria,
            self.unresolved_reasons,
        )
        if self.outcome is RubricAuthoringOutcome.BLOCKED_REACHABILITY:
            raise ValueError("semantic runner cannot author BLOCKED_REACHABILITY")
        if not self.model_available and (
            self.outcome is not RubricAuthoringOutcome.BLOCKED_CAPABILITY
            or RubricAuthoringReason.MODEL_UNAVAILABLE not in self.unresolved_reasons
        ):
            raise ValueError("unavailable model requires BLOCKED_CAPABILITY/MODEL_UNAVAILABLE")
        return self


class RubricCandidateProposal(ContractModel):
    schema_version: Literal["eval-factory/rubric-candidate-proposal/r4-05"] = (
        "eval-factory/rubric-candidate-proposal/r4-05"
    )
    rubric_candidate_proposal_id: Identifier
    request_ref: ObjectRef
    task_draft_ref: ObjectRef
    source_mode: RubricSourceModeV2
    outcome: RubricAuthoringOutcome
    criteria: tuple[RubricCriterionSelection, ...] = ()
    unresolved_reasons: frozenset[RubricAuthoringReason] = frozenset()
    imported_from_ref: ObjectRef | None = None
    model_profile: Identifier | None = None
    prompt_version: str | None = Field(default=None, min_length=1, max_length=128)
    policy_version: Literal["rubric-authoring/r4-05-v1"] = RUBRIC_AUTHORING_POLICY_VERSION
    proposal_sha256: Sha256
    audit: ContractAudit

    @field_validator("source_mode", mode="before")
    @classmethod
    def parse_source_mode(cls, value: object) -> RubricSourceModeV2:
        return _parse_enum(value, RubricSourceModeV2, "source_mode")

    @field_validator("outcome", mode="before")
    @classmethod
    def parse_outcome(cls, value: object) -> RubricAuthoringOutcome:
        return _parse_enum(value, RubricAuthoringOutcome, "outcome")

    @field_validator("unresolved_reasons", mode="before")
    @classmethod
    def parse_reasons(
        cls,
        value: object,
    ) -> frozenset[RubricAuthoringReason]:
        return _parse_enum_set(value, RubricAuthoringReason, "unresolved_reasons")

    @model_validator(mode="after")
    def validate_proposal(self) -> RubricCandidateProposal:
        _require_ref_type(
            self.request_ref,
            "rubric-candidate-request",
            "request_ref",
        )
        _require_ref_type(self.task_draft_ref, "task-draft", "task_draft_ref")
        if self.task_draft_ref.object_version != "v2":
            raise ValueError("task_draft_ref must reference TaskDraft v2")
        _validate_proposal_shape(
            self.outcome,
            self.criteria,
            self.unresolved_reasons,
        )
        if self.outcome is RubricAuthoringOutcome.BLOCKED_REACHABILITY:
            raise ValueError("proposal cannot claim compiler-owned reachability outcome")
        _validate_unique_selections(self.criteria)
        if self.source_mode is RubricSourceModeV2.IMPORTED:
            if self.imported_from_ref is None:
                raise ValueError("IMPORTED proposal requires imported_from_ref")
            _require_ref_type(
                self.imported_from_ref,
                "imported-rubric",
                "imported_from_ref",
            )
            if self.model_profile is not None or self.prompt_version is not None:
                raise ValueError("IMPORTED proposal cannot carry model metadata")
        else:
            if self.imported_from_ref is not None:
                raise ValueError("GENERATED proposal cannot carry imported_from_ref")
            if self.model_profile is None or self.prompt_version is None:
                raise ValueError("GENERATED proposal requires model metadata")
        return self


class RubricAuthoringResult(ContractModel):
    schema_version: Literal["eval-factory/rubric-authoring-result/r4-05"] = (
        "eval-factory/rubric-authoring-result/r4-05"
    )
    rubric_authoring_result_id: Identifier
    request_ref: ObjectRef
    proposal_ref: ObjectRef
    outcome: RubricAuthoringOutcome
    rubric_set: RubricSetV2 | None = None
    unresolved_reasons: frozenset[RubricAuthoringReason] = frozenset()
    policy_version: Literal["rubric-authoring/r4-05-v1"] = RUBRIC_AUTHORING_POLICY_VERSION
    result_sha256: Sha256
    audit: ContractAudit

    @field_validator("outcome", mode="before")
    @classmethod
    def parse_outcome(cls, value: object) -> RubricAuthoringOutcome:
        return _parse_enum(value, RubricAuthoringOutcome, "outcome")

    @field_validator("unresolved_reasons", mode="before")
    @classmethod
    def parse_reasons(
        cls,
        value: object,
    ) -> frozenset[RubricAuthoringReason]:
        return _parse_enum_set(value, RubricAuthoringReason, "unresolved_reasons")

    @model_validator(mode="after")
    def validate_result(self) -> RubricAuthoringResult:
        _require_ref_type(
            self.request_ref,
            "rubric-candidate-request",
            "request_ref",
        )
        _require_ref_type(
            self.proposal_ref,
            "rubric-candidate-proposal",
            "proposal_ref",
        )
        if self.outcome is RubricAuthoringOutcome.COMPILED:
            if self.rubric_set is None:
                raise ValueError("COMPILED result requires RubricSet")
            if self.unresolved_reasons:
                raise ValueError("COMPILED result cannot carry unresolved reasons")
        else:
            if self.rubric_set is not None:
                raise ValueError(f"{self.outcome.value} result cannot carry RubricSet")
            if not self.unresolved_reasons:
                raise ValueError(f"{self.outcome.value} result requires unresolved reasons")
        return self


def _validate_proposal_shape(
    outcome: RubricAuthoringOutcome,
    criteria: tuple[RubricCriterionSelection, ...],
    reasons: frozenset[RubricAuthoringReason],
) -> None:
    if outcome is RubricAuthoringOutcome.COMPILED:
        if not criteria:
            raise ValueError("COMPILED outcome requires criterion selections")
        if reasons:
            raise ValueError("COMPILED outcome cannot carry unresolved reasons")
        return
    if criteria:
        raise ValueError(f"{outcome.value} outcome cannot carry criterion selections")
    if not reasons:
        raise ValueError(f"{outcome.value} outcome requires unresolved reasons")


def _validate_unique_selections(
    criteria: tuple[RubricCriterionSelection, ...],
) -> None:
    _require_unique("criterion selection", (item.selection_id for item in criteria))
    _require_unique("judged object", (item.judged_object_id for item in criteria))


def _require_ref_type(ref: ObjectRef, expected: str, field_name: str) -> None:
    if ref.object_type != expected:
        raise ValueError(f"{field_name} must reference {expected}")


def _require_unique(label: str, values: Iterable[object]) -> None:
    seen: set[str] = set()
    for value in values:
        key = str(value)
        if key in seen:
            raise ValueError(f"{label} IDs must be unique")
        seen.add(key)


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
