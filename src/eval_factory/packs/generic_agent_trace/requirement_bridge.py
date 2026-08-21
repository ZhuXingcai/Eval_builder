from __future__ import annotations

from enum import StrEnum
from typing import ClassVar, Literal, Self

from pydantic import Field, model_validator

from eval_factory.contracts.agent_system_v2 import (
    EvaluationRequirementSpecV2,
    FactoryRunPolicyV2,
)
from eval_factory.contracts.core import ContractAudit, Identifier, ObjectRef
from eval_factory.contracts.core_v2 import ContractModelV2
from eval_factory.contracts.dataset_runtime_v2 import (
    FactoryDatasetRunRequestV2,
)
from eval_factory.harness.contracts import (
    HarnessObjectV1,
    require_sorted_unique,
    require_sorted_unique_refs,
)
from eval_factory.harness.runtime_models import (
    HarnessRequirementPolicyV1,
    ProviderEvidenceClassV1,
    RequirementInterpretationOutcomeV1,
    RequirementInterpretationV1,
)


class RequirementBridgeOutcomeV1(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    BLOCKED_INPUT = "BLOCKED_INPUT"


class RequirementBridgeFailureCodeV1(StrEnum):
    INTERPRETATION_NOT_READY = "INTERPRETATION_NOT_READY"
    INTERPRETATION_POLICY_STALE = "INTERPRETATION_POLICY_STALE"
    TARGET_CAPABILITY_UNSUPPORTED = "TARGET_CAPABILITY_UNSUPPORTED"
    SOURCE_AUTHORITY_INCOMPLETE = "SOURCE_AUTHORITY_INCOMPLETE"
    PIPELINE_POLICY_INCOMPLETE = "PIPELINE_POLICY_INCOMPLETE"
    GATEWAY_REGISTRY_INCOMPLETE = "GATEWAY_REGISTRY_INCOMPLETE"
    OUTPUT_TARGET_INCOMPLETE = "OUTPUT_TARGET_INCOMPLETE"


class GenericAgentRequirementBridgeInputV1(HarnessObjectV1):
    schema_version: Literal["generic-agent-trace/requirement-bridge-input/v1"] = (
        "generic-agent-trace/requirement-bridge-input/v1"
    )
    OBJECT_TYPE: ClassVar[str] = "generic-requirement-bridge-input"

    interpretation: RequirementInterpretationV1
    requirement_policy: HarnessRequirementPolicyV1
    dataset_run_id: Identifier
    requirement_source_ref: ObjectRef
    manifest_ref: ObjectRef
    source_authorization_ref: ObjectRef
    allowed_task_kinds: tuple[Identifier, ...] = Field(
        min_length=1,
        max_length=256,
    )
    pipeline_policy_refs: tuple[ObjectRef, ...] = Field(
        min_length=1,
        max_length=128,
    )
    gateway_registry_refs: tuple[ObjectRef, ...] = Field(
        min_length=1,
        max_length=32,
    )
    output_target_ref: ObjectRef
    max_transitions: int = Field(ge=1, le=100_000)
    max_plan_revisions: int = Field(ge=0, le=1_000)
    max_agent_attempts: int = Field(ge=1, le=100)
    idempotency_key: Identifier

    @model_validator(mode="after")
    def validate_input(self) -> Self:
        require_sorted_unique(
            self.allowed_task_kinds,
            "allowed_task_kinds",
        )
        require_sorted_unique_refs(
            self.pipeline_policy_refs,
            "pipeline_policy_refs",
        )
        require_sorted_unique_refs(
            self.gateway_registry_refs,
            "gateway_registry_refs",
        )
        return self


class GenericAgentRequirementBridgeResultV1(ContractModelV2):
    schema_version: Literal["generic-agent-trace/requirement-bridge-result/v1"] = (
        "generic-agent-trace/requirement-bridge-result/v1"
    )

    outcome: RequirementBridgeOutcomeV1
    interpretation_ref: ObjectRef
    evidence_class: ProviderEvidenceClassV1
    requirement: EvaluationRequirementSpecV2 | None = None
    factory_policy: FactoryRunPolicyV2 | None = None
    run_request: FactoryDatasetRunRequestV2 | None = None
    failure_codes: tuple[RequirementBridgeFailureCodeV1, ...] = Field(
        default=(),
        max_length=16,
    )

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        values = (
            self.requirement,
            self.factory_policy,
            self.run_request,
        )
        if self.outcome is RequirementBridgeOutcomeV1.SUCCEEDED:
            if any(value is None for value in values) or self.failure_codes:
                raise ValueError(
                    "successful requirement bridge requires complete Factory authority",
                )
        elif any(value is not None for value in values) or not self.failure_codes:
            raise ValueError(
                "blocked requirement bridge cannot create Factory authority",
            )
        if (
            tuple(
                sorted(set(self.failure_codes), key=lambda value: value.value),
            )
            != self.failure_codes
        ):
            raise ValueError("requirement bridge failure codes must be sorted")
        return self


class GenericAgentRequirementBridge:
    _TARGET_CAPABILITIES = frozenset(
        {
            "generic-agent-trace",
            "generic-agent-eval",
        },
    )

    def compile(
        self,
        bridge_input: GenericAgentRequirementBridgeInputV1,
        *,
        audit: ContractAudit,
    ) -> GenericAgentRequirementBridgeResultV1:
        failures = self._validate_input(bridge_input)
        if failures:
            return GenericAgentRequirementBridgeResultV1(
                outcome=RequirementBridgeOutcomeV1.BLOCKED_INPUT,
                interpretation_ref=bridge_input.interpretation.to_ref(),
                evidence_class=bridge_input.interpretation.evidence_class,
                failure_codes=tuple(
                    sorted(failures, key=lambda value: value.value),
                ),
            )
        interpretation = bridge_input.interpretation
        requirement = EvaluationRequirementSpecV2.create(
            requirement_spec_id=(f"evaluation-requirement-spec://harness/{interpretation.object_sha256}"),
            run_id=bridge_input.dataset_run_id,
            source_ref=bridge_input.requirement_source_ref,
            goals=interpretation.goals,
            constraints=interpretation.constraints,
            assumptions=interpretation.assumptions,
            open_questions=(),
            requirement_version=1,
            audit=audit,
        )
        harness_policy = bridge_input.requirement_policy
        factory_policy = FactoryRunPolicyV2.create(
            policy_id=(f"factory-run-policy://harness/{harness_policy.object_sha256}"),
            allowed_task_kinds=bridge_input.allowed_task_kinds,
            max_transitions=bridge_input.max_transitions,
            max_plan_revisions=bridge_input.max_plan_revisions,
            max_agent_attempts=bridge_input.max_agent_attempts,
            max_model_requests=harness_policy.max_model_requests,
            max_model_tokens=harness_policy.max_model_tokens,
            max_cost_micro_usd=harness_policy.max_cost_micro_usd,
            audit=audit,
        )
        run_request = FactoryDatasetRunRequestV2.create(
            dataset_run_id=bridge_input.dataset_run_id,
            requirement_spec_ref=requirement.to_ref(),
            manifest_ref=bridge_input.manifest_ref,
            source_authorization_ref=(bridge_input.source_authorization_ref),
            factory_policy_ref=factory_policy.to_ref(),
            pipeline_policy_refs=bridge_input.pipeline_policy_refs,
            gateway_registry_refs=bridge_input.gateway_registry_refs,
            output_target_ref=bridge_input.output_target_ref,
            idempotency_key=bridge_input.idempotency_key,
            max_transitions=bridge_input.max_transitions,
            audit=audit,
        )
        return GenericAgentRequirementBridgeResultV1(
            outcome=RequirementBridgeOutcomeV1.SUCCEEDED,
            interpretation_ref=interpretation.to_ref(),
            evidence_class=interpretation.evidence_class,
            requirement=requirement,
            factory_policy=factory_policy,
            run_request=run_request,
        )

    def _validate_input(
        self,
        bridge_input: GenericAgentRequirementBridgeInputV1,
    ) -> set[RequirementBridgeFailureCodeV1]:
        failures: set[RequirementBridgeFailureCodeV1] = set()
        interpretation = bridge_input.interpretation
        if interpretation.outcome is not RequirementInterpretationOutcomeV1.READY:
            failures.add(
                RequirementBridgeFailureCodeV1.INTERPRETATION_NOT_READY,
            )
        if bridge_input.requirement_policy.interpretation_ref != interpretation.to_ref():
            failures.add(
                RequirementBridgeFailureCodeV1.INTERPRETATION_POLICY_STALE,
            )
        if not set(interpretation.target_capabilities).issubset(
            self._TARGET_CAPABILITIES,
        ):
            failures.add(
                RequirementBridgeFailureCodeV1.TARGET_CAPABILITY_UNSUPPORTED,
            )
        if (
            bridge_input.requirement_source_ref.object_type != "evaluation-requirement-source"
            or bridge_input.requirement_source_ref.object_version != "v2"
            or bridge_input.manifest_ref.object_type != "trace-manifest"
            or bridge_input.manifest_ref.object_version != "v2"
            or bridge_input.source_authorization_ref.object_type != "trace-source-authorization"
            or bridge_input.source_authorization_ref.object_version != "v2"
        ):
            failures.add(
                RequirementBridgeFailureCodeV1.SOURCE_AUTHORITY_INCOMPLETE,
            )
        if any(
            not value.object_type.endswith("-policy") or value.object_version != "v2"
            for value in bridge_input.pipeline_policy_refs
        ):
            failures.add(
                RequirementBridgeFailureCodeV1.PIPELINE_POLICY_INCOMPLETE,
            )
        allowed_registries = {
            "agent-registry",
            "model-catalog",
            "prompt-registry",
            "rag-registry",
        }
        if any(
            value.object_type not in allowed_registries or value.object_version != "v2"
            for value in bridge_input.gateway_registry_refs
        ):
            failures.add(
                RequirementBridgeFailureCodeV1.GATEWAY_REGISTRY_INCOMPLETE,
            )
        if (
            bridge_input.output_target_ref.object_type != "candidate-output-target"
            or bridge_input.output_target_ref.object_version != "v2"
        ):
            failures.add(
                RequirementBridgeFailureCodeV1.OUTPUT_TARGET_INCOMPLETE,
            )
        return failures


__all__ = [
    "GenericAgentRequirementBridge",
    "GenericAgentRequirementBridgeInputV1",
    "GenericAgentRequirementBridgeResultV1",
    "RequirementBridgeFailureCodeV1",
    "RequirementBridgeOutcomeV1",
]
