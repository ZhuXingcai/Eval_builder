from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal, Self

from pydantic import Field, model_validator

from eval_factory.agent_system.private_store import (
    FactoryPrivateObjectStore,
)
from eval_factory.ai_gateway.protocols import AIGateway
from eval_factory.contracts.agent_system_v2 import (
    AttachmentQualityAssessmentV2,
    AttachmentSubgraphResultV2,
    SolvabilityAssessmentV2,
    SolvabilityOutcomeV2,
)
from eval_factory.contracts.ai_gateway_v2 import (
    GatewayInvocationRequestV2,
    GatewayInvocationStatusV2,
    ModelRouteDecisionV2,
    ModelRouteRequestV2,
    PromptTemplateV2,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.core_v2 import ContractModelV2


class SolvabilityAgentError(RuntimeError):
    pass


class SolvabilitySafeViewV1(ContractModelV2):
    schema_version: Literal["eval-factory/private-solvability-safe-view/v1"] = (
        "eval-factory/private-solvability-safe-view/v1"
    )
    attachment_subgraph_result_ref: ObjectRef
    quality_assessment_ref: ObjectRef
    evidence_refs: tuple[ObjectRef, ...] = Field(
        min_length=1,
        max_length=100_000,
    )

    @model_validator(mode="after")
    def validate_view(self) -> Self:
        if (
            self.attachment_subgraph_result_ref.object_type != "attachment-subgraph-result"
            or self.quality_assessment_ref.object_type != "attachment-quality-assessment"
        ):
            raise ValueError("solvability view requires current attachment refs")
        _require_safe_refs(self.evidence_refs)
        return self


class SolvabilityProposalV1(ContractModelV2):
    schema_version: Literal["eval-factory/private-solvability-proposal/v1"] = (
        "eval-factory/private-solvability-proposal/v1"
    )
    outcome: SolvabilityOutcomeV2
    evidence_refs: tuple[ObjectRef, ...] = Field(
        min_length=1,
        max_length=100_000,
    )
    reason_codes: tuple[str, ...] = Field(
        default=(),
        max_length=256,
    )

    @model_validator(mode="after")
    def validate_proposal(self) -> Self:
        _require_safe_refs(self.evidence_refs)
        if self.outcome is SolvabilityOutcomeV2.SOLVABLE:
            if self.reason_codes:
                raise ValueError("solvable proposal cannot retain reasons")
        elif not self.reason_codes:
            raise ValueError("non-solvable proposal requires reasons")
        return self


@dataclass(frozen=True, slots=True)
class SolvabilityAgentConfig:
    prompt: PromptTemplateV2
    agent_definition_ref: ObjectRef
    allowed_model_profile_refs: tuple[ObjectRef, ...]
    budget_reservation_ref: ObjectRef
    generator_model_profile_ref: ObjectRef


@dataclass(frozen=True, slots=True)
class SolvabilityAgentResult:
    assessment: SolvabilityAssessmentV2
    route: ModelRouteDecisionV2
    invocation_result_ref: ObjectRef


class GatewaySolvabilityAgent:
    def __init__(
        self,
        *,
        gateway: AIGateway,
        private_store: FactoryPrivateObjectStore,
        config: SolvabilityAgentConfig,
    ) -> None:
        self.gateway = gateway
        self.private_store = private_store
        self.config = config

    async def assess(
        self,
        *,
        task_ref: ObjectRef,
        subgraph_result: AttachmentSubgraphResultV2,
        quality: AttachmentQualityAssessmentV2,
        evidence_refs: tuple[ObjectRef, ...],
        audit: ContractAudit,
    ) -> SolvabilityAgentResult:
        if quality.attachment_subgraph_result_ref != subgraph_result.to_ref():
            raise SolvabilityAgentError("solvability quality does not bind the current attachment subgraph")
        safe_evidence_refs = _sorted_refs(evidence_refs)
        if not set(safe_evidence_refs).issubset(quality.validator_result_refs):
            raise SolvabilityAgentError("solvability evidence is outside quality validator authority")
        safe_view = SolvabilitySafeViewV1(
            attachment_subgraph_result_ref=(subgraph_result.to_ref()),
            quality_assessment_ref=quality.to_ref(),
            evidence_refs=safe_evidence_refs,
        )
        rendering_ref = self.private_store.put_model(
            object_type="prompt-rendering",
            value=safe_view,
        )
        route = self.gateway.route(
            self._route_request(
                task_ref=task_ref,
                rendering_ref=rendering_ref,
                audit=audit,
            )
        )
        invocation = GatewayInvocationRequestV2.create(
            invocation_request_id=(
                f"gateway-invocation-request://attachment-solvability/{_suffix(rendering_ref)}"
            ),
            agent_task_ref=task_ref,
            route_decision_ref=route.to_ref(),
            prompt_template_ref=self.config.prompt.to_ref(),
            prompt_rendering_ref=rendering_ref,
            output_schema_ref=self.config.prompt.output_schema_ref,
            rag_result_refs=(),
            idempotency_key=(f"gateway-invoke-attachment-solvability-{_suffix(rendering_ref)}"),
            audit=audit,
        )
        result = await self.gateway.invoke(
            invocation,
            route=route,
        )
        if result.status is not GatewayInvocationStatusV2.SUCCEEDED or result.output_ref is None:
            raise SolvabilityAgentError("solvability Gateway invocation did not succeed")
        proposal = self.private_store.get_model(
            result.output_ref,
            SolvabilityProposalV1,
        )
        if not set(proposal.evidence_refs).issubset(safe_view.evidence_refs):
            raise SolvabilityAgentError("solvability proposal widens evidence scope")
        assessment = SolvabilityAssessmentV2.create(
            assessment_id=(f"solvability-assessment://{subgraph_result.object_sha256}"),
            attachment_subgraph_result_ref=(subgraph_result.to_ref()),
            quality_assessment_ref=quality.to_ref(),
            route_decision_ref=route.to_ref(),
            gateway_receipt_ref=result.receipt_ref,
            evidence_refs=proposal.evidence_refs,
            outcome=proposal.outcome,
            reason_codes=proposal.reason_codes,
            audit=audit,
        )
        return SolvabilityAgentResult(
            assessment=assessment,
            route=route,
            invocation_result_ref=result.to_ref(),
        )

    def _route_request(
        self,
        *,
        task_ref: ObjectRef,
        rendering_ref: ObjectRef,
        audit: ContractAudit,
    ) -> ModelRouteRequestV2:
        return ModelRouteRequestV2.create(
            route_request_id=(f"model-route-request://attachment-solvability/{_suffix(rendering_ref)}"),
            agent_task_ref=task_ref,
            agent_definition_ref=(self.config.agent_definition_ref),
            task_kind="attachment-solvability",
            prompt_template_ref=self.config.prompt.to_ref(),
            required_capabilities=(self.config.prompt.required_model_capabilities),
            data_classification="RESTRICTED_TRACE_DERIVED",
            residency="LOCAL",
            input_token_budget=8_000,
            output_token_budget=2_000,
            max_cost_micro_usd=500_000,
            minimum_quality_basis_points=8000,
            allowed_model_profile_refs=(self.config.allowed_model_profile_refs),
            budget_reservation_ref=(self.config.budget_reservation_ref),
            route_version=1,
            predecessor_route_ref=None,
            failed_receipt_ref=None,
            generator_model_profile_ref=(self.config.generator_model_profile_ref),
            audit=audit,
        )


_DENIED_MARKERS = (
    "answer",
    "credential",
    "grader",
    "hidden",
    "private-reference",
    "prompt-body",
    "raw-trace",
    "runtime-transcript",
    "secret",
)

_ALLOWED_EVIDENCE_OBJECT_TYPES = {
    "deterministic-item-validation-result",
    "revision-deterministic-validation",
    "semantic-review-round-result",
}


def _require_safe_refs(values: tuple[ObjectRef, ...]) -> None:
    keys = tuple(_ref_key(value) for value in values)
    if tuple(sorted(keys)) != keys or len(set(keys)) != len(keys):
        raise ValueError("solvability evidence refs must be sorted and unique")
    for value in values:
        rendered = f"{value.object_type} {value.object_id}".casefold()
        if any(marker in rendered for marker in _DENIED_MARKERS):
            raise ValueError("solvability evidence cannot expose private material")
        if value.object_type not in _ALLOWED_EVIDENCE_OBJECT_TYPES:
            raise ValueError("solvability evidence must reference quality validator results")


def _sorted_refs(
    values: tuple[ObjectRef, ...],
) -> tuple[ObjectRef, ...]:
    unique = {_ref_key(value): value for value in values}
    return tuple(unique[key] for key in sorted(unique))


def _ref_key(
    value: ObjectRef,
) -> tuple[str, str, str, str]:
    return (
        value.object_type,
        value.object_id,
        value.object_version,
        value.object_sha256,
    )


def _suffix(value: ObjectRef) -> str:
    return hashlib.sha256(value.object_id.encode()).hexdigest()[:32]


__all__ = [
    "GatewaySolvabilityAgent",
    "SolvabilityAgentConfig",
    "SolvabilityAgentError",
    "SolvabilityAgentResult",
    "SolvabilityProposalV1",
    "SolvabilitySafeViewV1",
]
