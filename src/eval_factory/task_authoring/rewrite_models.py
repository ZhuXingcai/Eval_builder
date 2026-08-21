from __future__ import annotations

import hashlib
import json
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
from eval_factory.contracts.task_v2 import (
    ContestantToolPolicyV2,
    EvaluatorSpecV2,
    ProducerStorageAuthorizationV2,
    ProducerTaskViewV2,
    R4TaskContractSetV2,
    ReferencePolicyV2,
    RubricSetV2,
    TaskContractInvalidationV2,
    TaskDraftV2,
    TaskRewriteApplicationV2,
    TaskRewritePlanPreviewV2,
    TaskRewritePlanVersionV2,
    TaskRewritePreviewSafetyGateV2,
    ToolPolicyV2,
)

TASK_REWRITE_POLICY_VERSION: Literal["task-rewrite/r4-09-v1"] = "task-rewrite/r4-09-v1"
TASK_REWRITE_PREVIEW_POLICY_VERSION: Literal["task-rewrite-preview/r4-09-v1"] = (
    "task-rewrite-preview/r4-09-v1"
)


class TaskRewritePolicyError(RuntimeError):
    pass


class TaskRewritePlanCompileOutcome(StrEnum):
    PREVIEWED = "PREVIEWED"
    BLOCKED_SAFETY = "BLOCKED_SAFETY"
    BLOCKED_CAPABILITY = "BLOCKED_CAPABILITY"


class TaskRewritePlanCompileReason(StrEnum):
    INCOMPLETE_SAFETY_COVERAGE = "INCOMPLETE_SAFETY_COVERAGE"
    UNSAFE_PREVIEW = "UNSAFE_PREVIEW"
    MISSING_SEMANTIC_ASSESSMENT = "MISSING_SEMANTIC_ASSESSMENT"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"


class TaskRewritePlanCompilationResult(ContractModel):
    schema_version: Literal["eval-factory/task-rewrite-plan-compilation-result/r4-09"] = (
        "eval-factory/task-rewrite-plan-compilation-result/r4-09"
    )
    result_id: Identifier
    outcome: TaskRewritePlanCompileOutcome
    plan_version: TaskRewritePlanVersionV2 | None = None
    preview_safety_gate: TaskRewritePreviewSafetyGateV2 | None = None
    preview: TaskRewritePlanPreviewV2 | None = None
    unresolved_reasons: frozenset[TaskRewritePlanCompileReason] = frozenset()
    policy_version: Literal["task-rewrite/r4-09-v1"] = TASK_REWRITE_POLICY_VERSION
    result_sha256: Sha256
    audit: ContractAudit

    @field_validator("outcome", mode="before")
    @classmethod
    def parse_outcome(
        cls,
        value: object,
    ) -> TaskRewritePlanCompileOutcome:
        return _parse_enum(
            value,
            TaskRewritePlanCompileOutcome,
            "outcome",
        )

    @field_validator("unresolved_reasons", mode="before")
    @classmethod
    def parse_reasons(
        cls,
        value: object,
    ) -> frozenset[TaskRewritePlanCompileReason]:
        return _parse_enum_set(
            value,
            TaskRewritePlanCompileReason,
            "unresolved_reasons",
        )

    @model_validator(mode="after")
    def validate_result(self) -> TaskRewritePlanCompilationResult:
        if self.outcome is TaskRewritePlanCompileOutcome.PREVIEWED:
            if self.plan_version is None or self.preview_safety_gate is None or self.preview is None:
                raise ValueError("PREVIEWED requires plan version, safety gate, and preview")
            if self.unresolved_reasons:
                raise ValueError("PREVIEWED cannot carry unresolved reasons")
        elif self.outcome is TaskRewritePlanCompileOutcome.BLOCKED_SAFETY:
            if self.plan_version is not None or self.preview is not None:
                raise ValueError("BLOCKED_SAFETY cannot carry a plan version or preview")
            if self.preview_safety_gate is None:
                raise ValueError("BLOCKED_SAFETY requires a content-free safety gate")
            if not self.unresolved_reasons:
                raise ValueError(f"{self.outcome.value} requires unresolved reasons")
        else:
            if (
                self.plan_version is not None
                or self.preview_safety_gate is not None
                or self.preview is not None
            ):
                raise ValueError("BLOCKED_CAPABILITY cannot carry preview artifacts")
            if not self.unresolved_reasons:
                raise ValueError("BLOCKED_CAPABILITY requires unresolved reasons")
        return self


class TaskRewriteAdjustmentResult(ContractModel):
    schema_version: Literal["eval-factory/task-rewrite-adjustment-result/r4-09"] = (
        "eval-factory/task-rewrite-adjustment-result/r4-09"
    )
    result_id: Identifier
    outcome: TaskRewritePlanCompileOutcome
    source_plan_version: TaskRewritePlanVersionV2
    replacement_plan_version: TaskRewritePlanVersionV2 | None = None
    replacement_preview_safety_gate: TaskRewritePreviewSafetyGateV2 | None = None
    replacement_preview: TaskRewritePlanPreviewV2 | None = None
    invalidation: TaskContractInvalidationV2 | None = None
    unresolved_reasons: frozenset[TaskRewritePlanCompileReason] = frozenset()
    policy_version: Literal["task-rewrite/r4-09-v1"] = TASK_REWRITE_POLICY_VERSION
    result_sha256: Sha256
    audit: ContractAudit

    @field_validator("outcome", mode="before")
    @classmethod
    def parse_outcome(
        cls,
        value: object,
    ) -> TaskRewritePlanCompileOutcome:
        return _parse_enum(
            value,
            TaskRewritePlanCompileOutcome,
            "outcome",
        )

    @field_validator("unresolved_reasons", mode="before")
    @classmethod
    def parse_reasons(
        cls,
        value: object,
    ) -> frozenset[TaskRewritePlanCompileReason]:
        return _parse_enum_set(
            value,
            TaskRewritePlanCompileReason,
            "unresolved_reasons",
        )

    @model_validator(mode="after")
    def validate_result(self) -> TaskRewriteAdjustmentResult:
        values = (
            self.replacement_plan_version,
            self.replacement_preview_safety_gate,
            self.replacement_preview,
            self.invalidation,
        )
        if self.outcome is TaskRewritePlanCompileOutcome.PREVIEWED:
            if any(item is None for item in values):
                raise ValueError("PREVIEWED adjustment requires complete replacement artifacts")
            if self.unresolved_reasons:
                raise ValueError("PREVIEWED adjustment cannot carry unresolved reasons")
        elif self.outcome is TaskRewritePlanCompileOutcome.BLOCKED_SAFETY:
            if (
                self.replacement_plan_version is not None
                or self.replacement_preview is not None
                or self.invalidation is not None
            ):
                raise ValueError("BLOCKED_SAFETY cannot carry replacement plan, preview, or invalidation")
            if self.replacement_preview_safety_gate is None:
                raise ValueError("BLOCKED_SAFETY requires a content-free safety gate")
            if not self.unresolved_reasons:
                raise ValueError(f"{self.outcome.value} requires unresolved reasons")
        else:
            if any(item is not None for item in values):
                raise ValueError("BLOCKED_CAPABILITY cannot carry replacement artifacts")
            if not self.unresolved_reasons:
                raise ValueError("BLOCKED_CAPABILITY requires unresolved reasons")
        return self


class TaskRewritePromptOutcome(StrEnum):
    REWRITTEN = "REWRITTEN"
    ABSTAIN = "ABSTAIN"
    BLOCKED_CAPABILITY = "BLOCKED_CAPABILITY"


class TaskRewritePromptReason(StrEnum):
    AMBIGUOUS_REWRITE = "AMBIGUOUS_REWRITE"
    INCOMPLETE_REQUIREMENT_COVERAGE = "INCOMPLETE_REQUIREMENT_COVERAGE"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    ENVIRONMENT_STRATEGY_REQUIRED = "ENVIRONMENT_STRATEGY_REQUIRED"


class TaskRewritePromptRequest(ContractModel):
    schema_version: Literal["eval-factory/task-rewrite-prompt-request/r4-09"] = (
        "eval-factory/task-rewrite-prompt-request/r4-09"
    )
    request_id: Identifier
    source_task_draft_ref: ObjectRef
    replacement_plan_version_ref: ObjectRef
    replacement_preview_ref: ObjectRef
    replacement_preview_safety_gate_ref: ObjectRef
    prompt_boundary_enforcement_ref: ObjectRef
    source_visible_prompt: str = Field(min_length=1, max_length=20_000)
    target_capability: str = Field(min_length=1, max_length=4000)
    rewrite_style: str = Field(min_length=1, max_length=2000)
    fidelity: str = Field(min_length=1, max_length=64)
    operational_noise_policy: str = Field(min_length=1, max_length=4000)
    example_summaries: tuple[str, ...] = Field(min_length=1)
    example_treatments: tuple[str, ...] = Field(min_length=1)
    forbidden_content_rules: tuple[str, ...] = Field(min_length=1)
    expected_capability_impact: str = Field(min_length=1, max_length=2000)
    prompt_requirement_ids: tuple[Identifier, ...] = Field(min_length=1)
    model_profile: Identifier
    prompt_version: str = Field(min_length=1, max_length=128)
    untrusted_data_marker: Literal[True] = True
    policy_version: Literal["task-rewrite/r4-09-v1"] = TASK_REWRITE_POLICY_VERSION
    request_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_request(self) -> TaskRewritePromptRequest:
        _require_v2_ref(
            self.source_task_draft_ref,
            "task-draft",
            "source_task_draft_ref",
        )
        _require_v2_ref(
            self.replacement_plan_version_ref,
            "task-rewrite-plan-version",
            "replacement_plan_version_ref",
        )
        _require_v2_ref(
            self.replacement_preview_ref,
            "task-rewrite-plan-preview",
            "replacement_preview_ref",
        )
        _require_v2_ref(
            self.replacement_preview_safety_gate_ref,
            "task-rewrite-preview-safety-gate",
            "replacement_preview_safety_gate_ref",
        )
        _require_ref_type(
            self.prompt_boundary_enforcement_ref,
            "prompt-boundary-enforcement",
            "prompt_boundary_enforcement_ref",
        )
        _require_unique(
            "prompt requirement IDs",
            self.prompt_requirement_ids,
        )
        _require_unique(
            "forbidden content rules",
            self.forbidden_content_rules,
        )
        if len(self.example_summaries) != len(self.example_treatments):
            raise ValueError("rewrite example summaries and treatments must align")
        return self


class FakeTaskRewritePromptFixture(ContractModel):
    schema_version: Literal["eval-factory/fake-task-rewrite-prompt-fixture/r4-09"] = (
        "eval-factory/fake-task-rewrite-prompt-fixture/r4-09"
    )
    fixture_id: Identifier
    outcome: TaskRewritePromptOutcome
    visible_prompt: str | None = Field(
        default=None,
        min_length=1,
        max_length=20_000,
    )
    covered_prompt_requirement_ids: tuple[Identifier, ...] = ()
    unresolved_reasons: frozenset[TaskRewritePromptReason] = frozenset()
    model_available: bool = True

    @field_validator("outcome", mode="before")
    @classmethod
    def parse_outcome(
        cls,
        value: object,
    ) -> TaskRewritePromptOutcome:
        return _parse_enum(value, TaskRewritePromptOutcome, "outcome")

    @field_validator("unresolved_reasons", mode="before")
    @classmethod
    def parse_reasons(
        cls,
        value: object,
    ) -> frozenset[TaskRewritePromptReason]:
        return _parse_enum_set(
            value,
            TaskRewritePromptReason,
            "unresolved_reasons",
        )

    @model_validator(mode="after")
    def validate_fixture(self) -> FakeTaskRewritePromptFixture:
        _validate_prompt_outcome(
            self.outcome,
            self.visible_prompt,
            self.covered_prompt_requirement_ids,
            self.unresolved_reasons,
        )
        if not self.model_available and (
            self.outcome is not TaskRewritePromptOutcome.BLOCKED_CAPABILITY
            or TaskRewritePromptReason.MODEL_UNAVAILABLE not in self.unresolved_reasons
        ):
            raise ValueError("unavailable model requires BLOCKED_CAPABILITY/MODEL_UNAVAILABLE")
        return self


class TaskRewritePromptProposal(ContractModel):
    schema_version: Literal["eval-factory/task-rewrite-prompt-proposal/r4-09"] = (
        "eval-factory/task-rewrite-prompt-proposal/r4-09"
    )
    proposal_id: Identifier
    request_ref: ObjectRef
    outcome: TaskRewritePromptOutcome
    visible_prompt: str | None = Field(
        default=None,
        min_length=1,
        max_length=20_000,
    )
    covered_prompt_requirement_ids: tuple[Identifier, ...] = ()
    unresolved_reasons: frozenset[TaskRewritePromptReason] = frozenset()
    model_profile: Identifier
    prompt_version: str = Field(min_length=1, max_length=128)
    policy_version: Literal["task-rewrite/r4-09-v1"] = TASK_REWRITE_POLICY_VERSION
    proposal_sha256: Sha256
    audit: ContractAudit

    @field_validator("outcome", mode="before")
    @classmethod
    def parse_outcome(
        cls,
        value: object,
    ) -> TaskRewritePromptOutcome:
        return _parse_enum(value, TaskRewritePromptOutcome, "outcome")

    @field_validator("unresolved_reasons", mode="before")
    @classmethod
    def parse_reasons(
        cls,
        value: object,
    ) -> frozenset[TaskRewritePromptReason]:
        return _parse_enum_set(
            value,
            TaskRewritePromptReason,
            "unresolved_reasons",
        )

    @model_validator(mode="after")
    def validate_proposal(self) -> TaskRewritePromptProposal:
        _require_ref_type(
            self.request_ref,
            "task-rewrite-prompt-request",
            "request_ref",
        )
        _validate_prompt_outcome(
            self.outcome,
            self.visible_prompt,
            self.covered_prompt_requirement_ids,
            self.unresolved_reasons,
        )
        return self


class TaskRewritePromptResult(ContractModel):
    schema_version: Literal["eval-factory/task-rewrite-prompt-result/r4-09"] = (
        "eval-factory/task-rewrite-prompt-result/r4-09"
    )
    result_id: Identifier
    request_ref: ObjectRef
    proposal_ref: ObjectRef | None = None
    outcome: TaskRewritePromptOutcome
    task_draft: TaskDraftV2 | None = None
    unresolved_reasons: frozenset[TaskRewritePromptReason] = frozenset()
    policy_version: Literal["task-rewrite/r4-09-v1"] = TASK_REWRITE_POLICY_VERSION
    result_sha256: Sha256
    audit: ContractAudit

    @field_validator("outcome", mode="before")
    @classmethod
    def parse_outcome(
        cls,
        value: object,
    ) -> TaskRewritePromptOutcome:
        return _parse_enum(value, TaskRewritePromptOutcome, "outcome")

    @field_validator("unresolved_reasons", mode="before")
    @classmethod
    def parse_reasons(
        cls,
        value: object,
    ) -> frozenset[TaskRewritePromptReason]:
        return _parse_enum_set(
            value,
            TaskRewritePromptReason,
            "unresolved_reasons",
        )

    @model_validator(mode="after")
    def validate_result(self) -> TaskRewritePromptResult:
        if self.outcome is TaskRewritePromptOutcome.REWRITTEN:
            if self.task_draft is None:
                raise ValueError("REWRITTEN requires a TaskDraft")
            if self.unresolved_reasons:
                raise ValueError("REWRITTEN cannot carry unresolved reasons")
        else:
            if self.task_draft is not None:
                raise ValueError(f"{self.outcome.value} cannot carry a TaskDraft")
            if not self.unresolved_reasons:
                raise ValueError(f"{self.outcome.value} requires unresolved reasons")
        return self


class TaskRewriteApplicationResult(ContractModel):
    schema_version: Literal["eval-factory/task-rewrite-application-result/r4-09"] = (
        "eval-factory/task-rewrite-application-result/r4-09"
    )
    result_id: Identifier
    application: TaskRewriteApplicationV2
    replacement_contract_set: R4TaskContractSetV2
    rubric_set: RubricSetV2
    evaluator_spec: EvaluatorSpecV2
    reference_policy: ReferencePolicyV2
    tool_policy: ToolPolicyV2
    contestant_tool_policy: ContestantToolPolicyV2
    storage_authorization: ProducerStorageAuthorizationV2
    producer_task_view: ProducerTaskViewV2
    policy_version: Literal["task-rewrite/r4-09-v1"] = TASK_REWRITE_POLICY_VERSION
    result_sha256: Sha256
    audit: ContractAudit


def rewrite_payload_sha256(payload: object) -> str:
    encoded = json.dumps(
        canonical_value_v2(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_prompt_outcome(
    outcome: TaskRewritePromptOutcome,
    visible_prompt: str | None,
    covered_ids: tuple[str, ...],
    reasons: frozenset[TaskRewritePromptReason],
) -> None:
    _require_unique("covered prompt requirement IDs", covered_ids)
    if outcome is TaskRewritePromptOutcome.REWRITTEN:
        if visible_prompt is None or not covered_ids:
            raise ValueError("REWRITTEN requires prompt and requirement coverage")
        if reasons:
            raise ValueError("REWRITTEN cannot carry unresolved reasons")
    else:
        if visible_prompt is not None or covered_ids:
            raise ValueError(f"{outcome.value} cannot carry rewritten content")
        if not reasons:
            raise ValueError(f"{outcome.value} requires unresolved reasons")


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


def _require_ref_type(
    ref: ObjectRef,
    expected: str,
    field_name: str,
) -> None:
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
