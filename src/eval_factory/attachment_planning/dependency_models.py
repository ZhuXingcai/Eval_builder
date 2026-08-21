from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, field_validator, model_validator

from eval_factory.contracts.attachment_v2 import (
    PromptOnlyDependencyDiscoveryV2,
)
from eval_factory.contracts.core import (
    ContractAudit,
    ContractModel,
    Identifier,
    ObjectRef,
    RelativePath,
    Sha256,
)
from eval_factory.contracts.task import (
    AttachmentCriticality,
    EvidencePriority,
)

PROMPT_ONLY_DEPENDENCY_POLICY_VERSION: Literal["prompt-only-dependency/r5-03-v1"] = (
    "prompt-only-dependency/r5-03-v1"
)
PROMPT_ONLY_DEPENDENCY_PLANNING_POLICY_VERSION: Literal["prompt-only-dependency-planning/r5-03-v1"] = (
    "prompt-only-dependency-planning/r5-03-v1"
)


class PromptOnlyDependencyPolicyError(RuntimeError):
    pass


class PromptOnlyPathFactSource(StrEnum):
    REQUIREMENT_DESCRIPTION = "REQUIREMENT_DESCRIPTION"
    FORBIDDEN_OUTPUT = "FORBIDDEN_OUTPUT"


class PromptOnlyCandidateRole(StrEnum):
    INPUT_DEPENDENCY = "INPUT_DEPENDENCY"
    REQUESTED_OUTPUT = "REQUESTED_OUTPUT"
    COMPLETED_DELIVERABLE = "COMPLETED_DELIVERABLE"
    AMBIGUOUS = "AMBIGUOUS"


class PromptOnlyDependencyOutcome(StrEnum):
    RESOLVED = "RESOLVED"
    NOT_REQUIRED = "NOT_REQUIRED"
    ABSTAIN = "ABSTAIN"
    BLOCKED_CAPABILITY = "BLOCKED_CAPABILITY"
    BLOCKED_SAFETY = "BLOCKED_SAFETY"


class PromptOnlyDependencyReason(StrEnum):
    AMBIGUOUS_EXPLICIT_FACTS = "AMBIGUOUS_EXPLICIT_FACTS"
    AMBIGUOUS_INPUT_OUTPUT_ROLE = "AMBIGUOUS_INPUT_OUTPUT_ROLE"
    DIRECT_TARGET_REQUIRED = "DIRECT_TARGET_REQUIRED"
    FORBIDDEN_OUTPUT_MATCH = "FORBIDDEN_OUTPUT_MATCH"
    MEDIA_TYPE_UNRESOLVED = "MEDIA_TYPE_UNRESOLVED"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    REQUESTED_OUTPUT_PROPOSED = "REQUESTED_OUTPUT_PROPOSED"


class PromptOnlyPathFact(ContractModel):
    schema_version: Literal["eval-factory/prompt-only-path-fact/r5-03"] = (
        "eval-factory/prompt-only-path-fact/r5-03"
    )
    fact_id: Identifier
    attachment_dependency_id: Identifier | None = None
    source: PromptOnlyPathFactSource
    logical_path: RelativePath
    media_type: str | None = Field(
        default=None,
        min_length=3,
        max_length=255,
        pattern=r"^[a-z0-9.+-]+/[a-z0-9.+-]+$",
    )
    source_text_sha256: Sha256
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)

    @field_validator("source", mode="before")
    @classmethod
    def parse_source(
        cls,
        value: object,
    ) -> PromptOnlyPathFactSource:
        if isinstance(value, PromptOnlyPathFactSource):
            return value
        if isinstance(value, str):
            return PromptOnlyPathFactSource(value)
        raise TypeError("source must be a PromptOnlyPathFactSource")

    @model_validator(mode="after")
    def validate_fact(self) -> PromptOnlyPathFact:
        if self.char_end <= self.char_start:
            raise ValueError("path fact character range must be nonempty")
        if (
            self.source is PromptOnlyPathFactSource.REQUIREMENT_DESCRIPTION
            and self.attachment_dependency_id is None
        ):
            raise ValueError("requirement path fact requires attachment dependency ID")
        if (
            self.source is PromptOnlyPathFactSource.FORBIDDEN_OUTPUT
            and self.attachment_dependency_id is not None
        ):
            raise ValueError("forbidden-output path fact cannot bind a dependency")
        return self


class PromptOnlyDependencyView(ContractModel):
    schema_version: Literal["eval-factory/prompt-only-dependency-view/r5-03"] = (
        "eval-factory/prompt-only-dependency-view/r5-03"
    )
    dependency_id: Identifier
    description: str = Field(min_length=1, max_length=2000)
    criticality: AttachmentCriticality
    evidence_priority: EvidencePriority
    explicit_path_facts: tuple[PromptOnlyPathFact, ...] = ()
    existing_target_ref: ObjectRef | None = None

    @field_validator("criticality", mode="before")
    @classmethod
    def parse_criticality(
        cls,
        value: object,
    ) -> AttachmentCriticality:
        if isinstance(value, AttachmentCriticality):
            return value
        if isinstance(value, str):
            return AttachmentCriticality(value)
        raise TypeError("criticality must be an AttachmentCriticality")

    @field_validator("evidence_priority", mode="before")
    @classmethod
    def parse_priority(
        cls,
        value: object,
    ) -> EvidencePriority:
        if isinstance(value, EvidencePriority):
            return value
        if isinstance(value, str):
            return EvidencePriority(value)
        raise TypeError("evidence_priority must be an EvidencePriority")

    @model_validator(mode="after")
    def validate_view(self) -> PromptOnlyDependencyView:
        fact_ids = tuple(item.fact_id for item in self.explicit_path_facts)
        _require_unique("explicit path facts", fact_ids)
        if fact_ids != tuple(sorted(fact_ids)):
            raise ValueError("explicit path facts must be sorted")
        if any(item.attachment_dependency_id != self.dependency_id for item in self.explicit_path_facts):
            raise ValueError("explicit path facts must bind the dependency view")
        if self.existing_target_ref is not None:
            _require_ref_version(
                self.existing_target_ref,
                "artifact-evidence-target",
                "v2",
                "existing_target_ref",
            )
        return self


class PromptOnlyDependencyDiscoveryRequest(ContractModel):
    schema_version: Literal["eval-factory/prompt-only-dependency-discovery-request/r5-03"] = (
        "eval-factory/prompt-only-dependency-discovery-request/r5-03"
    )
    request_id: Identifier
    attachment_planning_context_ref: ObjectRef
    producer_task_view_ref: ObjectRef
    dependency_planning_context_ref: ObjectRef
    query_instruction: str = Field(min_length=1, max_length=20000)
    dependencies: tuple[PromptOnlyDependencyView, ...] = ()
    forbidden_outputs: tuple[str, ...] = Field(min_length=1)
    forbidden_path_facts: tuple[PromptOnlyPathFact, ...] = ()
    prompt_boundary_enforcement_ref: ObjectRef
    model_profile: Identifier
    prompt_version: str = Field(min_length=1, max_length=128)
    abstain_conditions: tuple[str, ...] = Field(min_length=1)
    policy_version: Literal["prompt-only-dependency/r5-03-v1"] = PROMPT_ONLY_DEPENDENCY_POLICY_VERSION
    request_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_request(self) -> PromptOnlyDependencyDiscoveryRequest:
        _require_ref_version(
            self.attachment_planning_context_ref,
            "attachment-planning-context",
            "v2",
            "attachment_planning_context_ref",
        )
        _require_ref_version(
            self.producer_task_view_ref,
            "producer-task-view",
            "v2",
            "producer_task_view_ref",
        )
        _require_ref_version(
            self.dependency_planning_context_ref,
            "prompt-only-dependency-planning-context",
            "v2",
            "dependency_planning_context_ref",
        )
        _require_ref_type(
            self.prompt_boundary_enforcement_ref,
            "prompt-boundary-enforcement",
            "prompt_boundary_enforcement_ref",
        )
        dependency_ids = tuple(item.dependency_id for item in self.dependencies)
        _require_unique("prompt-only dependency views", dependency_ids)
        if dependency_ids != tuple(sorted(dependency_ids)):
            raise ValueError("prompt-only dependency views must be sorted")
        forbidden_fact_ids = tuple(item.fact_id for item in self.forbidden_path_facts)
        _require_unique("forbidden path facts", forbidden_fact_ids)
        if forbidden_fact_ids != tuple(sorted(forbidden_fact_ids)):
            raise ValueError("forbidden path facts must be sorted")
        if any(
            item.source is not PromptOnlyPathFactSource.FORBIDDEN_OUTPUT for item in self.forbidden_path_facts
        ):
            raise ValueError("forbidden path facts must use forbidden-output source")
        _require_unique("forbidden outputs", self.forbidden_outputs)
        _require_unique("abstain conditions", self.abstain_conditions)
        return self


class PromptOnlyDependencyAssessment(ContractModel):
    schema_version: Literal["eval-factory/prompt-only-dependency-assessment/r5-03"] = (
        "eval-factory/prompt-only-dependency-assessment/r5-03"
    )
    dependency_id: Identifier
    candidate_role: PromptOnlyCandidateRole
    proposed_media_type: str | None = Field(
        default=None,
        min_length=3,
        max_length=255,
        pattern=r"^[a-z0-9.+-]+/[a-z0-9.+-]+$",
    )
    explanation_code: Identifier

    @field_validator("candidate_role", mode="before")
    @classmethod
    def parse_role(
        cls,
        value: object,
    ) -> PromptOnlyCandidateRole:
        if isinstance(value, PromptOnlyCandidateRole):
            return value
        if isinstance(value, str):
            return PromptOnlyCandidateRole(value)
        raise TypeError("candidate_role must be a PromptOnlyCandidateRole")


class FakePromptOnlyDependencyFixture(ContractModel):
    schema_version: Literal["eval-factory/fake-prompt-only-dependency-fixture/r5-03"] = (
        "eval-factory/fake-prompt-only-dependency-fixture/r5-03"
    )
    fixture_id: Identifier
    outcome: PromptOnlyDependencyOutcome
    assessments: tuple[PromptOnlyDependencyAssessment, ...] = ()
    unresolved_reasons: frozenset[PromptOnlyDependencyReason] = frozenset()
    model_available: bool = True

    @field_validator("outcome", mode="before")
    @classmethod
    def parse_outcome(
        cls,
        value: object,
    ) -> PromptOnlyDependencyOutcome:
        if isinstance(value, PromptOnlyDependencyOutcome):
            return value
        if isinstance(value, str):
            return PromptOnlyDependencyOutcome(value)
        raise TypeError("outcome must be a PromptOnlyDependencyOutcome")

    @field_validator("unresolved_reasons", mode="before")
    @classmethod
    def parse_reasons(
        cls,
        value: object,
    ) -> frozenset[PromptOnlyDependencyReason]:
        return _parse_reason_set(value)

    @model_validator(mode="after")
    def validate_fixture(self) -> FakePromptOnlyDependencyFixture:
        _validate_outcome_shape(
            self.outcome,
            self.assessments,
            self.unresolved_reasons,
        )
        _require_unique(
            "prompt-only dependency assessments",
            tuple(item.dependency_id for item in self.assessments),
        )
        if not self.model_available and (
            self.outcome is not PromptOnlyDependencyOutcome.BLOCKED_CAPABILITY
            or PromptOnlyDependencyReason.MODEL_UNAVAILABLE not in self.unresolved_reasons
        ):
            raise ValueError("unavailable model requires BLOCKED_CAPABILITY/MODEL_UNAVAILABLE")
        return self


class PromptOnlyDependencyDiscoveryProposal(ContractModel):
    schema_version: Literal["eval-factory/prompt-only-dependency-discovery-proposal/r5-03"] = (
        "eval-factory/prompt-only-dependency-discovery-proposal/r5-03"
    )
    proposal_id: Identifier
    request_ref: ObjectRef
    outcome: PromptOnlyDependencyOutcome
    assessments: tuple[PromptOnlyDependencyAssessment, ...] = ()
    unresolved_reasons: frozenset[PromptOnlyDependencyReason] = frozenset()
    model_profile: Identifier
    prompt_version: str = Field(min_length=1, max_length=128)
    policy_version: Literal["prompt-only-dependency/r5-03-v1"] = PROMPT_ONLY_DEPENDENCY_POLICY_VERSION
    proposal_sha256: Sha256
    audit: ContractAudit

    @field_validator("outcome", mode="before")
    @classmethod
    def parse_outcome(
        cls,
        value: object,
    ) -> PromptOnlyDependencyOutcome:
        if isinstance(value, PromptOnlyDependencyOutcome):
            return value
        if isinstance(value, str):
            return PromptOnlyDependencyOutcome(value)
        raise TypeError("outcome must be a PromptOnlyDependencyOutcome")

    @field_validator("unresolved_reasons", mode="before")
    @classmethod
    def parse_reasons(
        cls,
        value: object,
    ) -> frozenset[PromptOnlyDependencyReason]:
        return _parse_reason_set(value)

    @model_validator(mode="after")
    def validate_proposal(self) -> PromptOnlyDependencyDiscoveryProposal:
        _require_ref_type(
            self.request_ref,
            "prompt-only-dependency-request",
            "request_ref",
        )
        _validate_outcome_shape(
            self.outcome,
            self.assessments,
            self.unresolved_reasons,
        )
        _require_unique(
            "prompt-only dependency assessments",
            tuple(item.dependency_id for item in self.assessments),
        )
        return self


class PromptOnlyDependencyDiscoveryResult(ContractModel):
    schema_version: Literal["eval-factory/prompt-only-dependency-discovery-result/r5-03"] = (
        "eval-factory/prompt-only-dependency-discovery-result/r5-03"
    )
    result_id: Identifier
    request_ref: ObjectRef | None = None
    proposal_ref: ObjectRef | None = None
    outcome: PromptOnlyDependencyOutcome
    discovery: PromptOnlyDependencyDiscoveryV2 | None = None
    unresolved_dependency_ids: tuple[Identifier, ...] = ()
    reasons: frozenset[PromptOnlyDependencyReason] = frozenset()
    policy_version: Literal["prompt-only-dependency/r5-03-v1"] = PROMPT_ONLY_DEPENDENCY_POLICY_VERSION
    result_sha256: Sha256
    audit: ContractAudit

    @field_validator("outcome", mode="before")
    @classmethod
    def parse_outcome(
        cls,
        value: object,
    ) -> PromptOnlyDependencyOutcome:
        if isinstance(value, PromptOnlyDependencyOutcome):
            return value
        if isinstance(value, str):
            return PromptOnlyDependencyOutcome(value)
        raise TypeError("outcome must be a PromptOnlyDependencyOutcome")

    @field_validator("reasons", mode="before")
    @classmethod
    def parse_reasons(
        cls,
        value: object,
    ) -> frozenset[PromptOnlyDependencyReason]:
        return _parse_reason_set(value)

    @model_validator(mode="after")
    def validate_result(self) -> PromptOnlyDependencyDiscoveryResult:
        if self.request_ref is not None:
            _require_ref_type(
                self.request_ref,
                "prompt-only-dependency-request",
                "request_ref",
            )
        if self.proposal_ref is not None:
            _require_ref_type(
                self.proposal_ref,
                "prompt-only-dependency-proposal",
                "proposal_ref",
            )
        if self.unresolved_dependency_ids != tuple(sorted(set(self.unresolved_dependency_ids))):
            raise ValueError("unresolved dependency IDs must be unique and sorted")
        if self.outcome is PromptOnlyDependencyOutcome.RESOLVED:
            if self.discovery is None:
                raise ValueError("RESOLVED requires a discovery")
            if self.unresolved_dependency_ids or self.reasons:
                raise ValueError("RESOLVED cannot carry unresolved dependencies or reasons")
        elif self.outcome is PromptOnlyDependencyOutcome.NOT_REQUIRED:
            if self.discovery is not None or self.unresolved_dependency_ids or self.reasons:
                raise ValueError("NOT_REQUIRED cannot carry discovery, dependencies, or reasons")
        else:
            if self.discovery is not None:
                raise ValueError(f"{self.outcome.value} cannot carry partial discovery")
            if not self.unresolved_dependency_ids or not self.reasons:
                raise ValueError(f"{self.outcome.value} requires dependencies and reasons")
        return self


def _validate_outcome_shape(
    outcome: PromptOnlyDependencyOutcome,
    assessments: tuple[PromptOnlyDependencyAssessment, ...],
    reasons: frozenset[PromptOnlyDependencyReason],
) -> None:
    if outcome is PromptOnlyDependencyOutcome.RESOLVED:
        if not assessments:
            raise ValueError("RESOLVED semantic proposal requires assessments")
        if reasons:
            raise ValueError("RESOLVED cannot carry unresolved reasons")
        return
    if assessments:
        raise ValueError(f"{outcome.value} cannot carry dependency assessments")
    if outcome is PromptOnlyDependencyOutcome.NOT_REQUIRED:
        if reasons:
            raise ValueError("NOT_REQUIRED cannot carry reasons")
        return
    if not reasons:
        raise ValueError(f"{outcome.value} requires unresolved reasons")


def _parse_reason_set(
    value: object,
) -> frozenset[PromptOnlyDependencyReason]:
    if not isinstance(value, (tuple, list, set, frozenset)):
        raise TypeError("reasons must be a collection")
    return frozenset(PromptOnlyDependencyReason(item) for item in value)


def _require_unique(label: str, values: tuple[object, ...]) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must be unique")


def _require_ref_type(
    ref: ObjectRef,
    expected_type: str,
    field_name: str,
) -> None:
    if ref.object_type != expected_type:
        raise ValueError(f"{field_name} must reference {expected_type}")


def _require_ref_version(
    ref: ObjectRef,
    expected_type: str,
    expected_version: str,
    field_name: str,
) -> None:
    _require_ref_type(ref, expected_type, field_name)
    if ref.object_version != expected_version:
        raise ValueError(f"{field_name} must reference {expected_type} {expected_version}")
