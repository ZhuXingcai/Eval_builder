from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from eval_factory.contracts.core import (
    ContractAudit,
    EvidenceRef,
    Identifier,
    ObjectRef,
    ScalarValue,
)
from eval_factory.contracts.core_v2 import ContractModelV2


class ApprovalCheckpoint(StrEnum):
    LABEL_PLAN = "LABEL_PLAN"
    TASK_REWRITE_PLAN = "TASK_REWRITE_PLAN"
    ENVIRONMENT_STRATEGY = "ENVIRONMENT_STRATEGY"
    FINAL_DATASET_REVIEW = "FINAL_DATASET_REVIEW"


class ApprovalMode(StrEnum):
    NONE = "NONE"
    PLAN_GATES = "PLAN_GATES"
    FINAL_ONLY = "FINAL_ONLY"
    PLAN_AND_FINAL = "PLAN_AND_FINAL"
    CUSTOM = "CUSTOM"


PLAN_CHECKPOINTS = frozenset(
    {
        ApprovalCheckpoint.LABEL_PLAN,
        ApprovalCheckpoint.TASK_REWRITE_PLAN,
        ApprovalCheckpoint.ENVIRONMENT_STRATEGY,
    }
)
ALL_CHECKPOINTS = frozenset(ApprovalCheckpoint)


class FinalReviewScope(StrEnum):
    NONE = "NONE"
    PROMPTS = "PROMPTS"
    SELECTED_ITEMS = "SELECTED_ITEMS"
    FULL_DATASET = "FULL_DATASET"


class UserApprovalPolicy(ContractModelV2):
    schema_version: Literal["eval-factory/user-approval-policy/v2"] = "eval-factory/user-approval-policy/v2"
    policy_id: Identifier
    policy_version: str = Field(min_length=1, max_length=128)
    mode: ApprovalMode
    enabled_checkpoints: frozenset[ApprovalCheckpoint]
    final_review_scope: FinalReviewScope = FinalReviewScope.NONE
    explicit_unattended_choice: bool
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_mode(self) -> UserApprovalPolicy:
        expected = {
            ApprovalMode.NONE: frozenset(),
            ApprovalMode.PLAN_GATES: PLAN_CHECKPOINTS,
            ApprovalMode.FINAL_ONLY: frozenset({ApprovalCheckpoint.FINAL_DATASET_REVIEW}),
            ApprovalMode.PLAN_AND_FINAL: ALL_CHECKPOINTS,
        }
        if self.mode in expected and self.enabled_checkpoints != expected[self.mode]:
            raise ValueError(f"{self.mode} requires its canonical checkpoint set")
        if self.mode is ApprovalMode.CUSTOM and not self.enabled_checkpoints:
            raise ValueError("CUSTOM approval mode requires an explicit non-empty checkpoint set")
        has_final = ApprovalCheckpoint.FINAL_DATASET_REVIEW in self.enabled_checkpoints
        if has_final == (self.final_review_scope is FinalReviewScope.NONE):
            raise ValueError("final_review_scope must match FINAL_DATASET_REVIEW enablement")
        if self.mode is ApprovalMode.NONE and not self.explicit_unattended_choice:
            raise ValueError("NONE approval mode must be selected explicitly")
        return self


class ExampleKind(StrEnum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    AMBIGUOUS = "AMBIGUOUS"
    ABSTAIN = "ABSTAIN"
    REWRITE = "REWRITE"


class PlanExample(ContractModelV2):
    schema_version: Literal["eval-factory/plan-example/v2"] = "eval-factory/plan-example/v2"
    example_id: Identifier
    kind: ExampleKind
    input_summary: str = Field(min_length=1, max_length=4000)
    expected_treatment: str = Field(min_length=1, max_length=4000)
    evidence_refs: tuple[EvidenceRef, ...] = ()


class LabelPlan(ContractModelV2):
    schema_version: Literal["eval-factory/label-plan/v2"] = "eval-factory/label-plan/v2"
    label_plan_id: Identifier
    label_spec_ref: ObjectRef
    intent: str = Field(min_length=1, max_length=4000)
    boundary: str = Field(min_length=1, max_length=4000)
    deterministic_clause_refs: tuple[ObjectRef, ...]
    semantic_clause_refs: tuple[ObjectRef, ...]
    examples: tuple[PlanExample, ...] = Field(min_length=4)
    abstain_rules: tuple[str, ...] = Field(min_length=1)
    expected_model_path: tuple[Identifier, ...]
    estimated_model_requests_per_trace: int = Field(ge=0)
    blind_spots: tuple[str, ...] = Field(min_length=1)
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_examples(self) -> LabelPlan:
        required = {
            ExampleKind.POSITIVE,
            ExampleKind.NEGATIVE,
            ExampleKind.AMBIGUOUS,
            ExampleKind.ABSTAIN,
        }
        if not required.issubset({example.kind for example in self.examples}):
            raise ValueError("LabelPlan requires positive, negative, ambiguous, and abstain examples")
        if not self.deterministic_clause_refs and not self.semantic_clause_refs:
            raise ValueError("LabelPlan requires at least one deterministic or semantic clause")
        if not self.semantic_clause_refs and self.expected_model_path:
            raise ValueError("deterministic-only LabelPlan cannot declare a model path")
        return self


class RewriteFidelity(StrEnum):
    TRACE_FAITHFUL = "TRACE_FAITHFUL"
    CAPABILITY_PRESERVING = "CAPABILITY_PRESERVING"
    STANDARD_ENVIRONMENT = "STANDARD_ENVIRONMENT"


class TaskRewritePlan(ContractModelV2):
    schema_version: Literal["eval-factory/task-rewrite-plan/v2"] = "eval-factory/task-rewrite-plan/v2"
    task_rewrite_plan_id: Identifier
    selection_context_ref: ObjectRef
    target_capability: str = Field(min_length=1, max_length=4000)
    rewrite_style: str = Field(min_length=1, max_length=2000)
    fidelity: RewriteFidelity
    operational_noise_policy: str = Field(min_length=1, max_length=4000)
    examples: tuple[PlanExample, ...] = Field(min_length=1)
    forbidden_content_rules: tuple[str, ...] = Field(min_length=1)
    expected_capability_impact: str = Field(min_length=1, max_length=2000)
    audit: ContractAudit


class EnvironmentStrategyChoice(StrEnum):
    TRACE_FAITHFUL_MOCK = "TRACE_FAITHFUL_MOCK"
    REWRITE_STANDARD_ENV = "REWRITE_STANDARD_ENV"
    EXCLUDE_TASK = "EXCLUDE_TASK"


class QueryPackagingChoice(StrEnum):
    INCLUDE_QUERY_YAML = "INCLUDE_QUERY_YAML"
    OMIT_QUERY_YAML = "OMIT_QUERY_YAML"
    ASK_PER_TASK = "ASK_PER_TASK"


class EnvironmentAlternative(ContractModelV2):
    schema_version: Literal["eval-factory/environment-alternative/v2"] = (
        "eval-factory/environment-alternative/v2"
    )
    strategy: EnvironmentStrategyChoice
    feasible: bool
    capability_impact: str = Field(min_length=1, max_length=2000)
    blocking_reason: str | None = Field(default=None, min_length=1, max_length=2000)

    @model_validator(mode="after")
    def validate_feasibility(self) -> EnvironmentAlternative:
        if self.feasible == (self.blocking_reason is not None):
            raise ValueError("blocking_reason must be present exactly when an alternative is infeasible")
        return self


class EnvironmentRequirement(ContractModelV2):
    schema_version: Literal["eval-factory/environment-requirement/v2"] = (
        "eval-factory/environment-requirement/v2"
    )
    requirement_id: Identifier
    affected_subject_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    description: str = Field(min_length=1, max_length=4000)
    requires_special_account: bool
    requires_nonstandard_environment: bool
    evidence_refs: tuple[EvidenceRef, ...]
    alternatives: tuple[EnvironmentAlternative, ...] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def validate_requirement(self) -> EnvironmentRequirement:
        if not self.requires_special_account and not self.requires_nonstandard_environment:
            raise ValueError("EnvironmentRequirement must identify a special account or environment")
        choices = [alternative.strategy for alternative in self.alternatives]
        if len(set(choices)) != len(choices) or set(choices) != set(EnvironmentStrategyChoice):
            raise ValueError("EnvironmentRequirement must present each environment strategy once")
        if not any(alternative.feasible for alternative in self.alternatives):
            raise ValueError("EnvironmentRequirement must have at least one feasible strategy")
        return self


class EnvironmentStrategy(ContractModelV2):
    schema_version: Literal["eval-factory/environment-strategy/v2"] = "eval-factory/environment-strategy/v2"
    environment_strategy_id: Identifier
    requirements: tuple[EnvironmentRequirement, ...] = Field(min_length=1)
    query_packaging_options: frozenset[QueryPackagingChoice] = Field(min_length=3, max_length=3)
    recommended_strategy: EnvironmentStrategyChoice | None = None
    recommendation_reason: str | None = Field(default=None, min_length=1, max_length=2000)
    credential_fabrication_forbidden: Literal[True] = True
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_strategy(self) -> EnvironmentStrategy:
        if self.query_packaging_options != frozenset(QueryPackagingChoice):
            raise ValueError("EnvironmentStrategy must present every query packaging choice")
        if (self.recommended_strategy is None) != (self.recommendation_reason is None):
            raise ValueError("recommended strategy and reason must be provided together")
        return self


class UserDecision(StrEnum):
    ACCEPT = "ACCEPT"
    ADJUST = "ADJUST"
    REJECT = "REJECT"
    REQUEST_MORE_EXAMPLES = "REQUEST_MORE_EXAMPLES"
    DEFER = "DEFER"


DECISIONS_BY_CHECKPOINT = {
    ApprovalCheckpoint.LABEL_PLAN: frozenset(
        {
            UserDecision.ACCEPT,
            UserDecision.ADJUST,
            UserDecision.REJECT,
            UserDecision.REQUEST_MORE_EXAMPLES,
        }
    ),
    ApprovalCheckpoint.TASK_REWRITE_PLAN: frozenset(
        {
            UserDecision.ACCEPT,
            UserDecision.ADJUST,
            UserDecision.REJECT,
            UserDecision.REQUEST_MORE_EXAMPLES,
        }
    ),
    ApprovalCheckpoint.ENVIRONMENT_STRATEGY: frozenset(
        {UserDecision.ACCEPT, UserDecision.ADJUST, UserDecision.REJECT}
    ),
    ApprovalCheckpoint.FINAL_DATASET_REVIEW: frozenset(
        {UserDecision.ACCEPT, UserDecision.ADJUST, UserDecision.REJECT, UserDecision.DEFER}
    ),
}


class UserApprovalRequest(ContractModelV2):
    schema_version: Literal["eval-factory/user-approval-request/v2"] = "eval-factory/user-approval-request/v2"
    request_id: Identifier
    checkpoint: ApprovalCheckpoint
    requested_by: Identifier
    approval_policy_ref: ObjectRef
    subject_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    plan_ref: ObjectRef | None
    projection_ref: ObjectRef
    preview_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    available_decisions: frozenset[UserDecision]
    affected_stages: tuple[Identifier, ...] = Field(min_length=1)
    idempotency_key: Identifier
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_request(self) -> UserApprovalRequest:
        if self.available_decisions != DECISIONS_BY_CHECKPOINT[self.checkpoint]:
            raise ValueError("available decisions must match the checkpoint policy")
        if self.checkpoint is ApprovalCheckpoint.FINAL_DATASET_REVIEW:
            if self.plan_ref is not None:
                raise ValueError("final dataset review binds subjects, not a plan")
        elif self.plan_ref is None:
            raise ValueError("plan checkpoints require an exact plan reference")
        return self


class TypedAdjustment(ContractModelV2):
    schema_version: Literal["eval-factory/typed-adjustment/v2"] = "eval-factory/typed-adjustment/v2"
    target_path: str = Field(pattern=r"^[A-Za-z0-9_.\[\]-]{1,256}$")
    operation: Literal["SET", "ADD", "REMOVE", "REPLACE"]
    value: ScalarValue
    reason: str = Field(min_length=1, max_length=2000)


class EnvironmentScopeDecision(ContractModelV2):
    schema_version: Literal["eval-factory/environment-scope-decision/v2"] = (
        "eval-factory/environment-scope-decision/v2"
    )
    requirement_id: Identifier
    strategy: EnvironmentStrategyChoice


class InvalidationScope(ContractModelV2):
    schema_version: Literal["eval-factory/invalidation-scope/v2"] = "eval-factory/invalidation-scope/v2"
    object_refs: tuple[ObjectRef, ...] = ()
    stages: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def validate_scope(self) -> InvalidationScope:
        if not self.object_refs and not self.stages:
            raise ValueError("InvalidationScope cannot be empty")
        return self


class UserDecisionRecord(ContractModelV2):
    schema_version: Literal["eval-factory/user-decision-record/v2"] = "eval-factory/user-decision-record/v2"
    decision_record_id: Identifier
    request_ref: ObjectRef
    checkpoint: ApprovalCheckpoint
    approval_policy_ref: ObjectRef
    subject_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    plan_ref: ObjectRef | None
    projection_ref: ObjectRef
    authenticated_user: Identifier
    decision: UserDecision
    adjustments: tuple[TypedAdjustment, ...] = ()
    environment_decisions: tuple[EnvironmentScopeDecision, ...] = ()
    query_packaging: QueryPackagingChoice | None = None
    reason: str = Field(min_length=1, max_length=4000)
    resulting_object_ref: ObjectRef | None = None
    invalidation_scope: InvalidationScope | None = None
    hard_gate_override_requested: Literal[False] = False
    idempotency_key: Identifier
    decided_at: datetime
    record_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def validate_decision(self) -> UserDecisionRecord:
        if self.decision not in DECISIONS_BY_CHECKPOINT[self.checkpoint]:
            raise ValueError("decision is not allowed for checkpoint")
        if self.checkpoint is ApprovalCheckpoint.FINAL_DATASET_REVIEW:
            if self.plan_ref is not None:
                raise ValueError("final dataset review cannot bind a plan")
        elif self.plan_ref is None:
            raise ValueError("plan checkpoint decision requires a plan reference")
        if self.decision is UserDecision.ADJUST:
            if not self.adjustments or self.resulting_object_ref is None:
                raise ValueError("ADJUST requires typed adjustments and a resulting object")
            if self.invalidation_scope is None:
                raise ValueError("ADJUST requires an explicit invalidation scope")
        elif self.adjustments or self.invalidation_scope is not None:
            raise ValueError("only ADJUST may carry adjustments or invalidation scope")
        if self.decision is UserDecision.REQUEST_MORE_EXAMPLES and self.resulting_object_ref is None:
            raise ValueError("REQUEST_MORE_EXAMPLES requires a new request reference")
        is_environment = self.checkpoint is ApprovalCheckpoint.ENVIRONMENT_STRATEGY
        if (
            is_environment
            and self.decision in {UserDecision.ACCEPT, UserDecision.ADJUST}
            and (not self.environment_decisions or self.query_packaging is None)
        ):
            raise ValueError("environment acceptance requires strategy and query packaging choices")
        if not is_environment and (self.environment_decisions or self.query_packaging is not None):
            raise ValueError("environment choices are valid only at ENVIRONMENT_STRATEGY")
        return self
