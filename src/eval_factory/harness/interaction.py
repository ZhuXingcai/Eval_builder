from __future__ import annotations

from enum import StrEnum
from typing import ClassVar, Literal, Self

from pydantic import Field, model_validator

from eval_factory.contracts.core import ContractAudit, Identifier, ObjectRef
from eval_factory.contracts.core_v2 import ContractModelV2
from eval_factory.harness.capability import CapabilitySideEffectV1
from eval_factory.harness.contracts import (
    HarnessObjectV1,
    require_ref,
    require_sorted_unique,
    require_sorted_unique_refs,
    sorted_refs,
)


class PermissionOutcomeV1(StrEnum):
    ALLOW = "ALLOW"
    REQUIRE_APPROVAL = "REQUIRE_APPROVAL"
    DENY = "DENY"


class GraphMutationKindV1(StrEnum):
    NONE = "NONE"
    IN_SCOPE = "IN_SCOPE"
    AUTHORITY_WIDENING = "AUTHORITY_WIDENING"


class PermissionReasonV1(StrEnum):
    INSIDE_APPROVED_AUTHORITY = "INSIDE_APPROVED_AUTHORITY"
    UNKNOWN_PRINCIPAL = "UNKNOWN_PRINCIPAL"
    CAPABILITY_NOT_GRANTED = "CAPABILITY_NOT_GRANTED"
    PROVIDER_NOT_GRANTED = "PROVIDER_NOT_GRANTED"
    TASK_NOT_GRANTED = "TASK_NOT_GRANTED"
    DATA_SCOPE_NOT_GRANTED = "DATA_SCOPE_NOT_GRANTED"
    PURPOSE_NOT_GRANTED = "PURPOSE_NOT_GRANTED"
    CLASSIFICATION_NOT_GRANTED = "CLASSIFICATION_NOT_GRANTED"
    SIDE_EFFECT_NOT_GRANTED = "SIDE_EFFECT_NOT_GRANTED"
    SOURCE_ADMISSION = "SOURCE_ADMISSION"
    MODEL_BUDGET_WIDENING = "MODEL_BUDGET_WIDENING"
    EXTERNAL_EXECUTION = "EXTERNAL_EXECUTION"
    GRAPH_AUTHORITY_WIDENING = "GRAPH_AUTHORITY_WIDENING"
    NONDETERMINISTIC_MUTATION = "NONDETERMINISTIC_MUTATION"
    DESTRUCTIVE_ACTION = "DESTRUCTIVE_ACTION"
    RELEASE = "RELEASE"
    STALE_AUTHORITY = "STALE_AUTHORITY"
    WRONG_PERMISSION_POLICY = "WRONG_PERMISSION_POLICY"


class InteractionDecisionKindV1(StrEnum):
    APPROVE = "APPROVE"
    DENY = "DENY"


class MemberExecutionGrantV1(ContractModelV2):
    schema_version: Literal["eval-harness/member-execution-grant/v1"] = (
        "eval-harness/member-execution-grant/v1"
    )
    member_id: Identifier
    principal_ref: ObjectRef
    capability_definition_refs: tuple[ObjectRef, ...] = Field(
        min_length=1,
        max_length=256,
    )
    provider_binding_refs: tuple[ObjectRef, ...] = Field(
        min_length=1,
        max_length=256,
    )
    task_ids: tuple[Identifier, ...] = Field(
        default=(),
        max_length=10_000,
    )
    data_scope_refs: tuple[ObjectRef, ...] = Field(default=(), max_length=10_000)
    data_purposes: tuple[Identifier, ...] = Field(min_length=1, max_length=64)
    data_classifications: tuple[Identifier, ...] = Field(
        min_length=1,
        max_length=64,
    )
    allowed_side_effects: tuple[CapabilitySideEffectV1, ...] = Field(
        min_length=1,
        max_length=4,
    )

    @model_validator(mode="after")
    def validate_grant(self) -> Self:
        require_sorted_unique_refs(
            self.capability_definition_refs,
            "capability_definition_refs",
        )
        if any(ref.object_type != "harness-capability-definition" for ref in self.capability_definition_refs):
            raise ValueError("grant capabilities must reference Harness definitions")
        require_sorted_unique_refs(
            self.provider_binding_refs,
            "provider_binding_refs",
        )
        if any(ref.object_type != "harness-capability-provider" for ref in self.provider_binding_refs):
            raise ValueError("grant providers must reference Harness provider bindings")
        require_sorted_unique(self.task_ids, "task_ids")
        require_sorted_unique_refs(self.data_scope_refs, "data_scope_refs")
        require_sorted_unique(self.data_purposes, "data_purposes")
        require_sorted_unique(
            self.data_classifications,
            "data_classifications",
        )
        side_effects = tuple(value.value for value in self.allowed_side_effects)
        require_sorted_unique(side_effects, "allowed_side_effects")
        return self


class BalancedAutonomyPolicyV1(HarnessObjectV1):
    schema_version: Literal["eval-harness/balanced-autonomy-policy/v1"] = (
        "eval-harness/balanced-autonomy-policy/v1"
    )
    OBJECT_TYPE: ClassVar[str] = "balanced-autonomy-policy"

    policy_id: Identifier
    mode: Literal["BALANCED_AUTONOMY"] = "BALANCED_AUTONOMY"
    auto_run_read_only: Literal[True] = True
    auto_run_local_deterministic_write: Literal[True] = True
    auto_run_prebudgeted_model_calls: Literal[True] = True
    source_admission_requires_approval: Literal[True] = True
    external_execution_requires_approval: Literal[True] = True
    authority_widening_requires_approval: Literal[True] = True
    destructive_action_requires_approval: Literal[True] = True
    release_requires_approval: Literal[True] = True
    peer_messages_grant_authority: Literal[False] = False


class ExecutionAuthorityV1(HarnessObjectV1):
    schema_version: Literal["eval-harness/execution-authority/v1"] = "eval-harness/execution-authority/v1"
    OBJECT_TYPE: ClassVar[str] = "execution-authority"

    authority_id: Identifier
    authority_version: int = Field(ge=1, le=1_000_000_000)
    predecessor_authority_ref: ObjectRef | None = None
    team_id: Identifier
    team_incarnation_id: Identifier
    roster_ref: ObjectRef
    task_graph_ref: ObjectRef
    permission_policy_ref: ObjectRef
    grants: tuple[MemberExecutionGrantV1, ...] = Field(
        min_length=1,
        max_length=1_000,
    )
    max_model_requests: int = Field(ge=0, le=10_000_000)
    max_model_tokens: int = Field(ge=0, le=10_000_000_000)
    max_cost_micro_usd: int = Field(ge=0, le=10_000_000_000_000)
    used_model_requests: int = Field(ge=0, le=10_000_000)
    used_model_tokens: int = Field(ge=0, le=10_000_000_000)
    used_cost_micro_usd: int = Field(ge=0, le=10_000_000_000_000)

    @classmethod
    def create(  # type: ignore[override]
        cls,
        *,
        authority_id: str,
        authority_version: int,
        predecessor_authority_ref: ObjectRef | None,
        team_id: str,
        team_incarnation_id: str,
        roster_ref: ObjectRef,
        task_graph_ref: ObjectRef,
        permission_policy_ref: ObjectRef,
        grants: tuple[MemberExecutionGrantV1, ...],
        max_model_requests: int,
        max_model_tokens: int,
        max_cost_micro_usd: int,
        used_model_requests: int,
        used_model_tokens: int,
        used_cost_micro_usd: int,
        audit: ContractAudit,
    ) -> ExecutionAuthorityV1:
        return super().create(
            audit=audit,
            authority_id=authority_id,
            authority_version=authority_version,
            predecessor_authority_ref=predecessor_authority_ref,
            team_id=team_id,
            team_incarnation_id=team_incarnation_id,
            roster_ref=roster_ref,
            task_graph_ref=task_graph_ref,
            permission_policy_ref=permission_policy_ref,
            grants=tuple(sorted(grants, key=lambda value: value.member_id)),
            max_model_requests=max_model_requests,
            max_model_tokens=max_model_tokens,
            max_cost_micro_usd=max_cost_micro_usd,
            used_model_requests=used_model_requests,
            used_model_tokens=used_model_tokens,
            used_cost_micro_usd=used_cost_micro_usd,
        )

    @model_validator(mode="after")
    def validate_authority(self) -> Self:
        require_ref(self.roster_ref, "team-roster", "roster_ref")
        require_ref(self.task_graph_ref, "team-task-graph", "task_graph_ref")
        require_ref(
            self.permission_policy_ref,
            "balanced-autonomy-policy",
            "permission_policy_ref",
        )
        member_ids = tuple(grant.member_id for grant in self.grants)
        require_sorted_unique(member_ids, "grant member IDs")
        if self.authority_version == 1:
            if self.predecessor_authority_ref is not None:
                raise ValueError("first execution authority cannot have a predecessor")
        elif self.predecessor_authority_ref is None:
            raise ValueError("authority successor requires its predecessor")
        else:
            require_ref(
                self.predecessor_authority_ref,
                "execution-authority",
                "predecessor_authority_ref",
            )
        for used, maximum, label in (
            (self.used_model_requests, self.max_model_requests, "model requests"),
            (self.used_model_tokens, self.max_model_tokens, "model tokens"),
            (self.used_cost_micro_usd, self.max_cost_micro_usd, "model cost"),
        ):
            if used > maximum:
                raise ValueError(f"used {label} exceed approved authority")
        return self


class PermissionActionV1(HarnessObjectV1):
    schema_version: Literal["eval-harness/permission-action/v1"] = "eval-harness/permission-action/v1"
    OBJECT_TYPE: ClassVar[str] = "permission-action"

    action_id: Identifier
    authority_ref: ObjectRef
    permission_policy_ref: ObjectRef
    member_id: Identifier
    principal_ref: ObjectRef
    capability_definition_ref: ObjectRef
    provider_binding_ref: ObjectRef
    task_id: Identifier
    data_scope_refs: tuple[ObjectRef, ...] = Field(default=(), max_length=10_000)
    data_purpose: Identifier
    data_classification: Identifier
    side_effect: CapabilitySideEffectV1
    graph_mutation: GraphMutationKindV1
    local_execution: bool
    deterministic: bool
    source_admission: bool = False
    external_execution: bool = False
    destructive: bool = False
    release: bool = False
    model_requests_delta: int = Field(ge=0, le=10_000_000)
    model_tokens_delta: int = Field(ge=0, le=10_000_000_000)
    cost_micro_usd_delta: int = Field(ge=0, le=10_000_000_000_000)

    @classmethod
    def create(  # type: ignore[override]
        cls,
        *,
        action_id: str,
        authority_ref: ObjectRef,
        permission_policy_ref: ObjectRef,
        member_id: str,
        principal_ref: ObjectRef,
        capability_definition_ref: ObjectRef,
        provider_binding_ref: ObjectRef,
        task_id: str,
        data_scope_refs: tuple[ObjectRef, ...],
        data_purpose: str,
        data_classification: str,
        side_effect: CapabilitySideEffectV1,
        graph_mutation: GraphMutationKindV1,
        local_execution: bool,
        deterministic: bool,
        source_admission: bool,
        external_execution: bool,
        destructive: bool,
        release: bool,
        model_requests_delta: int,
        model_tokens_delta: int,
        cost_micro_usd_delta: int,
        audit: ContractAudit,
    ) -> PermissionActionV1:
        return super().create(
            audit=audit,
            action_id=action_id,
            authority_ref=authority_ref,
            permission_policy_ref=permission_policy_ref,
            member_id=member_id,
            principal_ref=principal_ref,
            capability_definition_ref=capability_definition_ref,
            provider_binding_ref=provider_binding_ref,
            task_id=task_id,
            data_scope_refs=sorted_refs(data_scope_refs),
            data_purpose=data_purpose,
            data_classification=data_classification,
            side_effect=side_effect,
            graph_mutation=graph_mutation,
            local_execution=local_execution,
            deterministic=deterministic,
            source_admission=source_admission,
            external_execution=external_execution,
            destructive=destructive,
            release=release,
            model_requests_delta=model_requests_delta,
            model_tokens_delta=model_tokens_delta,
            cost_micro_usd_delta=cost_micro_usd_delta,
        )

    @model_validator(mode="after")
    def validate_action(self) -> Self:
        require_ref(self.authority_ref, "execution-authority", "authority_ref")
        require_ref(
            self.permission_policy_ref,
            "balanced-autonomy-policy",
            "permission_policy_ref",
        )
        require_ref(
            self.capability_definition_ref,
            "harness-capability-definition",
            "capability_definition_ref",
        )
        require_ref(
            self.provider_binding_ref,
            "harness-capability-provider",
            "provider_binding_ref",
        )
        require_sorted_unique_refs(self.data_scope_refs, "data_scope_refs")
        if self.external_execution and self.local_execution:
            raise ValueError("execution cannot be both local and external")
        if self.side_effect is CapabilitySideEffectV1.EXTERNAL_EFFECT and not self.external_execution:
            raise ValueError("external side effect must declare external execution")
        if self.destructive and self.side_effect is CapabilitySideEffectV1.READ_ONLY:
            raise ValueError("read-only action cannot be destructive")
        return self


class PermissionDecisionV1(HarnessObjectV1):
    schema_version: Literal["eval-harness/permission-decision/v1"] = "eval-harness/permission-decision/v1"
    OBJECT_TYPE: ClassVar[str] = "permission-decision"

    decision_id: Identifier
    policy_ref: ObjectRef
    authority_ref: ObjectRef
    action_ref: ObjectRef
    outcome: PermissionOutcomeV1
    reason_codes: tuple[PermissionReasonV1, ...] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def validate_decision(self) -> Self:
        require_ref(self.policy_ref, "balanced-autonomy-policy", "policy_ref")
        require_ref(self.authority_ref, "execution-authority", "authority_ref")
        require_ref(self.action_ref, "permission-action", "action_ref")
        reason_values = tuple(reason.value for reason in self.reason_codes)
        require_sorted_unique(reason_values, "reason_codes")
        if self.outcome is PermissionOutcomeV1.ALLOW and self.reason_codes != (
            PermissionReasonV1.INSIDE_APPROVED_AUTHORITY,
        ):
            raise ValueError("allowed action requires the approved-authority reason")
        return self


class InteractionRequestV1(HarnessObjectV1):
    schema_version: Literal["eval-harness/interaction-request/v1"] = "eval-harness/interaction-request/v1"
    OBJECT_TYPE: ClassVar[str] = "interaction-request"

    request_id: Identifier
    authority_ref: ObjectRef
    action_ref: ObjectRef
    permission_decision_ref: ObjectRef
    requested_from: ObjectRef
    available_decisions: tuple[InteractionDecisionKindV1, ...] = (
        InteractionDecisionKindV1.APPROVE,
        InteractionDecisionKindV1.DENY,
    )

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        require_ref(self.authority_ref, "execution-authority", "authority_ref")
        require_ref(self.action_ref, "permission-action", "action_ref")
        require_ref(
            self.permission_decision_ref,
            "permission-decision",
            "permission_decision_ref",
        )
        if self.available_decisions != (
            InteractionDecisionKindV1.APPROVE,
            InteractionDecisionKindV1.DENY,
        ):
            raise ValueError("interaction request decisions are closed")
        return self


class InteractionDecisionV1(HarnessObjectV1):
    schema_version: Literal["eval-harness/interaction-decision/v1"] = "eval-harness/interaction-decision/v1"
    OBJECT_TYPE: ClassVar[str] = "interaction-decision"

    decision_id: Identifier
    request_ref: ObjectRef
    action_ref: ObjectRef
    authority_ref: ObjectRef
    decided_by: ObjectRef
    decision: InteractionDecisionKindV1
    successor_authority_ref: ObjectRef | None = None

    @model_validator(mode="after")
    def validate_decision(self) -> Self:
        require_ref(self.request_ref, "interaction-request", "request_ref")
        require_ref(self.action_ref, "permission-action", "action_ref")
        require_ref(self.authority_ref, "execution-authority", "authority_ref")
        if self.successor_authority_ref is not None:
            require_ref(
                self.successor_authority_ref,
                "execution-authority",
                "successor_authority_ref",
            )
        if self.decision is InteractionDecisionKindV1.DENY and self.successor_authority_ref is not None:
            raise ValueError("denial cannot create successor authority")
        return self


class BalancedAutonomyEvaluator:
    def evaluate(
        self,
        *,
        policy: BalancedAutonomyPolicyV1,
        authority: ExecutionAuthorityV1,
        action: PermissionActionV1,
        audit: ContractAudit,
    ) -> PermissionDecisionV1:
        outcome, reasons = self._classify(policy, authority, action)
        return PermissionDecisionV1.create(
            decision_id=f"permission-decision://{action.action_id}",
            policy_ref=policy.to_ref(),
            authority_ref=authority.to_ref(),
            action_ref=action.to_ref(),
            outcome=outcome,
            reason_codes=tuple(sorted(reasons, key=lambda reason: reason.value)),
            audit=audit,
        )

    @staticmethod
    def _classify(
        policy: BalancedAutonomyPolicyV1,
        authority: ExecutionAuthorityV1,
        action: PermissionActionV1,
    ) -> tuple[PermissionOutcomeV1, set[PermissionReasonV1]]:
        if action.permission_policy_ref != policy.to_ref():
            return PermissionOutcomeV1.DENY, {
                PermissionReasonV1.WRONG_PERMISSION_POLICY,
            }
        if action.authority_ref != authority.to_ref():
            return PermissionOutcomeV1.DENY, {
                PermissionReasonV1.STALE_AUTHORITY,
            }
        grant = next(
            (
                candidate
                for candidate in authority.grants
                if candidate.member_id == action.member_id and candidate.principal_ref == action.principal_ref
            ),
            None,
        )
        if grant is None:
            return PermissionOutcomeV1.DENY, {
                PermissionReasonV1.UNKNOWN_PRINCIPAL,
            }
        denied_reasons = _grant_denials(grant, action)
        if denied_reasons:
            return PermissionOutcomeV1.DENY, denied_reasons

        approval_reasons: set[PermissionReasonV1] = set()
        if action.source_admission:
            approval_reasons.add(PermissionReasonV1.SOURCE_ADMISSION)
        if action.external_execution or action.side_effect is CapabilitySideEffectV1.EXTERNAL_EFFECT:
            approval_reasons.add(PermissionReasonV1.EXTERNAL_EXECUTION)
        if action.graph_mutation is GraphMutationKindV1.AUTHORITY_WIDENING:
            approval_reasons.add(PermissionReasonV1.GRAPH_AUTHORITY_WIDENING)
        if action.destructive or action.side_effect is CapabilitySideEffectV1.IRREVERSIBLE_WRITE:
            approval_reasons.add(PermissionReasonV1.DESTRUCTIVE_ACTION)
        if action.release:
            approval_reasons.add(PermissionReasonV1.RELEASE)
        if not _inside_budget(authority, action):
            approval_reasons.add(PermissionReasonV1.MODEL_BUDGET_WIDENING)
        if action.side_effect is CapabilitySideEffectV1.REVERSIBLE_WRITE and not (
            action.local_execution
            and action.deterministic
            and action.graph_mutation in {GraphMutationKindV1.NONE, GraphMutationKindV1.IN_SCOPE}
        ):
            approval_reasons.add(PermissionReasonV1.NONDETERMINISTIC_MUTATION)
        if approval_reasons:
            return PermissionOutcomeV1.REQUIRE_APPROVAL, approval_reasons
        return PermissionOutcomeV1.ALLOW, {
            PermissionReasonV1.INSIDE_APPROVED_AUTHORITY,
        }


def _grant_denials(
    grant: MemberExecutionGrantV1,
    action: PermissionActionV1,
) -> set[PermissionReasonV1]:
    reasons: set[PermissionReasonV1] = set()
    if action.capability_definition_ref not in grant.capability_definition_refs:
        reasons.add(PermissionReasonV1.CAPABILITY_NOT_GRANTED)
    if action.provider_binding_ref not in grant.provider_binding_refs:
        reasons.add(PermissionReasonV1.PROVIDER_NOT_GRANTED)
    if action.task_id not in grant.task_ids:
        reasons.add(PermissionReasonV1.TASK_NOT_GRANTED)
    if not set(action.data_scope_refs).issubset(grant.data_scope_refs):
        reasons.add(PermissionReasonV1.DATA_SCOPE_NOT_GRANTED)
    if action.data_purpose not in grant.data_purposes:
        reasons.add(PermissionReasonV1.PURPOSE_NOT_GRANTED)
    if action.data_classification not in grant.data_classifications:
        reasons.add(PermissionReasonV1.CLASSIFICATION_NOT_GRANTED)
    if action.side_effect not in grant.allowed_side_effects:
        reasons.add(PermissionReasonV1.SIDE_EFFECT_NOT_GRANTED)
    return reasons


def _inside_budget(
    authority: ExecutionAuthorityV1,
    action: PermissionActionV1,
) -> bool:
    return (
        authority.used_model_requests + action.model_requests_delta <= authority.max_model_requests
        and authority.used_model_tokens + action.model_tokens_delta <= authority.max_model_tokens
        and authority.used_cost_micro_usd + action.cost_micro_usd_delta <= authority.max_cost_micro_usd
    )


__all__ = [
    "BalancedAutonomyEvaluator",
    "BalancedAutonomyPolicyV1",
    "ExecutionAuthorityV1",
    "GraphMutationKindV1",
    "InteractionDecisionKindV1",
    "InteractionDecisionV1",
    "InteractionRequestV1",
    "MemberExecutionGrantV1",
    "PermissionActionV1",
    "PermissionDecisionV1",
    "PermissionOutcomeV1",
    "PermissionReasonV1",
]
