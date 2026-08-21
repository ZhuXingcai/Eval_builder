from __future__ import annotations

from enum import StrEnum
from typing import Annotated, ClassVar, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from eval_factory.contracts.core import ContractAudit, Identifier, ObjectRef
from eval_factory.contracts.core_v2 import ContractModelV2
from eval_factory.harness.contracts import (
    HarnessObjectV1,
    require_ref,
    require_sorted_unique,
    require_sorted_unique_refs,
    sorted_refs,
)

CapabilityVersion = Annotated[
    str,
    StringConstraints(min_length=1, max_length=128),
]


class CapabilitySideEffectV1(StrEnum):
    READ_ONLY = "READ_ONLY"
    REVERSIBLE_WRITE = "REVERSIBLE_WRITE"
    IRREVERSIBLE_WRITE = "IRREVERSIBLE_WRITE"
    EXTERNAL_EFFECT = "EXTERNAL_EFFECT"


class CapabilityIdempotencyV1(StrEnum):
    PURE = "PURE"
    IDEMPOTENT = "IDEMPOTENT"
    IDEMPOTENCY_KEY_REQUIRED = "IDEMPOTENCY_KEY_REQUIRED"
    NON_IDEMPOTENT = "NON_IDEMPOTENT"


class CapabilityConsumerKindV1(StrEnum):
    AGENT_TOOL = "AGENT_TOOL"
    GRAPH_NODE = "GRAPH_NODE"
    HTTP_API = "HTTP_API"
    CLI = "CLI"
    UI_WORKSPACE = "UI_WORKSPACE"
    INTERNAL_SERVICE = "INTERNAL_SERVICE"


class CapabilityProviderKindV1(StrEnum):
    LOCAL_DETERMINISTIC = "LOCAL_DETERMINISTIC"
    LOCAL_MODEL_ASSISTED = "LOCAL_MODEL_ASSISTED"
    REMOTE_GOVERNED = "REMOTE_GOVERNED"


class CapabilityInvocationOutcomeV1(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    BLOCKED_CAPABILITY = "BLOCKED_CAPABILITY"
    BLOCKED_POLICY = "BLOCKED_POLICY"
    ABSTAINED = "ABSTAINED"
    FAILED = "FAILED"


class CapabilityDefinitionV1(HarnessObjectV1):
    schema_version: Literal["eval-harness/capability-definition/v1"] = "eval-harness/capability-definition/v1"
    OBJECT_TYPE: ClassVar[str] = "harness-capability-definition"

    capability_id: Identifier
    capability_version: CapabilityVersion
    request_schema_ref: ObjectRef
    result_schema_ref: ObjectRef
    input_artifact_roles: tuple[Identifier, ...] = Field(default=(), max_length=256)
    output_artifact_roles: tuple[Identifier, ...] = Field(min_length=1, max_length=256)
    data_purposes: tuple[Identifier, ...] = Field(min_length=1, max_length=64)
    data_classifications: tuple[Identifier, ...] = Field(min_length=1, max_length=64)
    side_effect: CapabilitySideEffectV1
    idempotency: CapabilityIdempotencyV1
    supported_consumers: tuple[CapabilityConsumerKindV1, ...] = Field(
        min_length=1,
        max_length=6,
    )
    legacy_agent_capability_ref: ObjectRef | None = None

    @classmethod
    def create(  # type: ignore[override]
        cls,
        *,
        capability_id: str,
        capability_version: str,
        request_schema_ref: ObjectRef,
        result_schema_ref: ObjectRef,
        input_artifact_roles: tuple[str, ...],
        output_artifact_roles: tuple[str, ...],
        data_purposes: tuple[str, ...],
        data_classifications: tuple[str, ...],
        side_effect: CapabilitySideEffectV1,
        idempotency: CapabilityIdempotencyV1,
        supported_consumers: tuple[CapabilityConsumerKindV1, ...],
        legacy_agent_capability_ref: ObjectRef | None,
        audit: ContractAudit,
    ) -> CapabilityDefinitionV1:
        return super().create(
            audit=audit,
            capability_id=capability_id,
            capability_version=capability_version,
            request_schema_ref=request_schema_ref,
            result_schema_ref=result_schema_ref,
            input_artifact_roles=tuple(sorted(input_artifact_roles)),
            output_artifact_roles=tuple(sorted(output_artifact_roles)),
            data_purposes=tuple(sorted(data_purposes)),
            data_classifications=tuple(sorted(data_classifications)),
            side_effect=side_effect,
            idempotency=idempotency,
            supported_consumers=tuple(sorted(set(supported_consumers), key=lambda item: item.value)),
            legacy_agent_capability_ref=legacy_agent_capability_ref,
        )

    @model_validator(mode="after")
    def validate_definition(self) -> Self:
        for value, label in (
            (self.request_schema_ref, "request_schema_ref"),
            (self.result_schema_ref, "result_schema_ref"),
        ):
            if value.object_type != "json-schema":
                raise ValueError(f"{label} must reference a JSON schema")
        for values, label in (
            (self.input_artifact_roles, "input_artifact_roles"),
            (self.output_artifact_roles, "output_artifact_roles"),
            (self.data_purposes, "data_purposes"),
            (self.data_classifications, "data_classifications"),
        ):
            require_sorted_unique(values, label)
        consumer_values = tuple(item.value for item in self.supported_consumers)
        require_sorted_unique(consumer_values, "supported_consumers")
        if self.legacy_agent_capability_ref is not None:
            require_ref(
                self.legacy_agent_capability_ref,
                "agent-capability",
                "legacy_agent_capability_ref",
                object_version="v2",
            )
        return self


class CapabilityProviderBindingV1(HarnessObjectV1):
    schema_version: Literal["eval-harness/capability-provider-binding/v1"] = (
        "eval-harness/capability-provider-binding/v1"
    )
    OBJECT_TYPE: ClassVar[str] = "harness-capability-provider"

    provider_id: Identifier
    provider_version: CapabilityVersion
    capability_definition_ref: ObjectRef
    implementation_id: Identifier
    provider_kind: CapabilityProviderKindV1
    supported_consumers: tuple[CapabilityConsumerKindV1, ...] = Field(
        min_length=1,
        max_length=6,
    )
    policy_refs: tuple[ObjectRef, ...] = Field(default=(), max_length=64)

    @classmethod
    def create(  # type: ignore[override]
        cls,
        *,
        provider_id: str,
        provider_version: str,
        capability_definition_ref: ObjectRef,
        implementation_id: str,
        provider_kind: CapabilityProviderKindV1,
        supported_consumers: tuple[CapabilityConsumerKindV1, ...],
        policy_refs: tuple[ObjectRef, ...],
        audit: ContractAudit,
    ) -> CapabilityProviderBindingV1:
        return super().create(
            audit=audit,
            provider_id=provider_id,
            provider_version=provider_version,
            capability_definition_ref=capability_definition_ref,
            implementation_id=implementation_id,
            provider_kind=provider_kind,
            supported_consumers=tuple(sorted(set(supported_consumers), key=lambda item: item.value)),
            policy_refs=sorted_refs(policy_refs),
        )

    @model_validator(mode="after")
    def validate_provider(self) -> Self:
        require_ref(
            self.capability_definition_ref,
            "harness-capability-definition",
            "capability_definition_ref",
        )
        consumer_values = tuple(item.value for item in self.supported_consumers)
        require_sorted_unique(consumer_values, "supported_consumers")
        require_sorted_unique_refs(self.policy_refs, "policy_refs")
        return self


class CapabilityConsumerBindingV1(HarnessObjectV1):
    schema_version: Literal["eval-harness/capability-consumer-binding/v1"] = (
        "eval-harness/capability-consumer-binding/v1"
    )
    OBJECT_TYPE: ClassVar[str] = "harness-capability-consumer"

    consumer_id: Identifier
    consumer_version: CapabilityVersion
    consumer_kind: CapabilityConsumerKindV1
    capability_definition_ref: ObjectRef
    provider_binding_ref: ObjectRef
    projection_ref: ObjectRef | None = None

    @model_validator(mode="after")
    def validate_consumer(self) -> Self:
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
        if self.projection_ref is not None:
            require_ref(
                self.projection_ref,
                "harness-projection-definition",
                "projection_ref",
            )
        return self


class CapabilityInvocationContextV1(ContractModelV2):
    schema_version: Literal["eval-harness/capability-invocation-context/v1"] = (
        "eval-harness/capability-invocation-context/v1"
    )
    session_ref: ObjectRef
    team_ref: ObjectRef | None = None
    member_ref: ObjectRef | None = None
    task_ref: ObjectRef | None = None
    authority_ref: ObjectRef
    principal_ref: ObjectRef
    data_purpose: Identifier
    data_classification: Identifier
    idempotency_key: Identifier

    @model_validator(mode="after")
    def validate_context(self) -> Self:
        require_ref(self.session_ref, "harness-session", "session_ref")
        require_ref(self.authority_ref, "execution-authority", "authority_ref")
        if self.team_ref is not None:
            require_ref(self.team_ref, "agent-team", "team_ref")
            if self.member_ref is None:
                raise ValueError("Team invocation requires a member ref")
        elif self.member_ref is not None or self.task_ref is not None:
            raise ValueError("member/task refs require Team invocation")
        if self.member_ref is not None:
            require_ref(self.member_ref, "team-member", "member_ref")
        if self.task_ref is not None:
            require_ref(self.task_ref, "team-task", "task_ref")
        return self


class CapabilityCallV1(HarnessObjectV1):
    schema_version: Literal["eval-harness/capability-call/v1"] = "eval-harness/capability-call/v1"
    OBJECT_TYPE: ClassVar[str] = "harness-capability-call"

    call_id: Identifier
    capability_definition_ref: ObjectRef
    provider_binding_ref: ObjectRef
    context: CapabilityInvocationContextV1
    request_ref: ObjectRef
    input_artifact_refs: tuple[ObjectRef, ...] = Field(default=(), max_length=10_000)

    @model_validator(mode="after")
    def validate_call(self) -> Self:
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
        require_sorted_unique_refs(self.input_artifact_refs, "input_artifact_refs")
        if any(ref.object_type != "artifact-envelope" for ref in self.input_artifact_refs):
            raise ValueError("capability inputs must reference artifact envelopes")
        return self


class CapabilityResultV1(HarnessObjectV1):
    schema_version: Literal["eval-harness/capability-result/v1"] = "eval-harness/capability-result/v1"
    OBJECT_TYPE: ClassVar[str] = "harness-capability-result"

    result_id: Identifier
    call_ref: ObjectRef
    outcome: CapabilityInvocationOutcomeV1
    canonical_result_ref: ObjectRef | None = None
    output_artifact_refs: tuple[ObjectRef, ...] = Field(default=(), max_length=10_000)
    validation_refs: tuple[ObjectRef, ...] = Field(default=(), max_length=10_000)
    failure_code: Identifier | None = None

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        require_ref(self.call_ref, "harness-capability-call", "call_ref")
        require_sorted_unique_refs(self.output_artifact_refs, "output_artifact_refs")
        require_sorted_unique_refs(self.validation_refs, "validation_refs")
        if any(ref.object_type != "artifact-envelope" for ref in self.output_artifact_refs):
            raise ValueError("capability outputs must reference artifact envelopes")
        if self.outcome is CapabilityInvocationOutcomeV1.SUCCEEDED:
            if self.canonical_result_ref is None or self.failure_code is not None:
                raise ValueError("successful capability result requires canonical result authority")
        elif self.canonical_result_ref is not None or self.output_artifact_refs:
            raise ValueError("non-success capability result cannot publish canonical output")
        elif self.failure_code is None:
            raise ValueError("non-success capability result requires a closed failure code")
        return self


__all__ = [
    "CapabilityCallV1",
    "CapabilityConsumerBindingV1",
    "CapabilityConsumerKindV1",
    "CapabilityDefinitionV1",
    "CapabilityIdempotencyV1",
    "CapabilityInvocationContextV1",
    "CapabilityInvocationOutcomeV1",
    "CapabilityProviderBindingV1",
    "CapabilityProviderKindV1",
    "CapabilityResultV1",
    "CapabilitySideEffectV1",
    "CapabilityVersion",
]
