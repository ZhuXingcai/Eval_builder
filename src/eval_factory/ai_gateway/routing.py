from __future__ import annotations

from eval_factory.ai_gateway.model_catalog import ModelCatalog, ModelCatalogError
from eval_factory.ai_gateway.price import estimate_max_cost_micro_usd
from eval_factory.contracts.ai_gateway_v2 import (
    ModelHealthSnapshotV2,
    ModelPriceScheduleV2,
    ModelQualityBaselineV2,
    ModelRouteCandidateV2,
    ModelRouteDecisionV2,
    ModelRouteRejectionCodeV2,
    ModelRouteRequestV2,
    ModelRoutingPolicyV2,
)
from eval_factory.contracts.core import ObjectRef


class ModelRouteError(RuntimeError):
    pass


class ModelRouteBlockedError(ModelRouteError):
    def __init__(self, candidates: tuple[ModelRouteCandidateV2, ...]) -> None:
        self.candidates = candidates
        super().__init__("no model satisfies the governed route request")


class ModelRouter:
    def __init__(
        self,
        catalog: ModelCatalog,
        policy: ModelRoutingPolicyV2,
    ) -> None:
        self.catalog = catalog
        self.policy = policy

    def route(
        self,
        request: ModelRouteRequestV2,
        *,
        predecessor: ModelRouteDecisionV2 | None = None,
    ) -> ModelRouteDecisionV2:
        self._validate_successor(request, predecessor)
        policy_models = {_ref_key(ref) for ref in self.policy.allowed_model_profile_refs}
        policy_agents = {_ref_key(ref) for ref in self.policy.allowed_agent_definition_refs}
        request_models = {_ref_key(ref) for ref in request.allowed_model_profile_refs}
        predecessor_model = predecessor.selected_model_profile_ref if predecessor is not None else None
        candidates: list[ModelRouteCandidateV2] = []
        eligible_facts: dict[
            tuple[str, str, str, str],
            tuple[
                ModelQualityBaselineV2,
                ModelHealthSnapshotV2,
                ModelPriceScheduleV2,
            ],
        ] = {}
        for reference in request.allowed_model_profile_refs:
            key = _ref_key(reference)
            try:
                facts = self.catalog.facts(reference, task_kind=request.task_kind)
            except ModelCatalogError:
                candidates.append(
                    _rejected(reference, ModelRouteRejectionCodeV2.GOVERNANCE_MODEL_NOT_ALLOWED)
                )
                continue
            profile = facts.profile
            governance_reason = self._governance_reason(
                request,
                profile_key=key,
                provider_id=profile.provider_id,
                classifications=profile.supported_data_classifications,
                residencies=profile.supported_residencies,
                availability=profile.availability,
                agent_allowed=_ref_key(request.agent_definition_ref) in policy_agents,
                policy_models=policy_models,
                request_models=request_models,
                predecessor_model=predecessor_model,
                reference=reference,
            )
            if governance_reason is not None:
                candidates.append(_rejected(reference, governance_reason))
                continue
            capability_reason = _capability_reason(
                request,
                capabilities=profile.capabilities,
                context_limit=profile.context_limit_tokens,
                output_limit=profile.output_limit_tokens,
            )
            if capability_reason is not None:
                candidates.append(_rejected(reference, capability_reason))
                continue
            if facts.quality is None:
                candidates.append(_rejected(reference, ModelRouteRejectionCodeV2.QUALITY_BASELINE_MISSING))
                continue
            if facts.quality.quality_basis_points < request.minimum_quality_basis_points:
                candidates.append(_rejected(reference, ModelRouteRejectionCodeV2.QUALITY_BELOW_MINIMUM))
                continue
            if facts.health is None:
                candidates.append(_rejected(reference, ModelRouteRejectionCodeV2.HEALTH_SNAPSHOT_MISSING))
                continue
            if facts.health.availability == "UNAVAILABLE" or facts.health.success_basis_points == 0:
                candidates.append(_rejected(reference, ModelRouteRejectionCodeV2.PROVIDER_UNAVAILABLE))
                continue
            if facts.price is None:
                candidates.append(_rejected(reference, ModelRouteRejectionCodeV2.PRICE_SCHEDULE_MISSING))
                continue
            estimated_cost = estimate_max_cost_micro_usd(request, facts.price)
            if estimated_cost > request.max_cost_micro_usd:
                candidates.append(_rejected(reference, ModelRouteRejectionCodeV2.BUDGET_EXCEEDED))
                continue
            score = (
                facts.quality.quality_basis_points * self.policy.quality_weight
                + facts.health.success_basis_points * self.policy.health_weight
                - facts.health.latency_milliseconds * self.policy.latency_weight
                - estimated_cost * self.policy.cost_weight
            )
            candidate = ModelRouteCandidateV2(
                model_profile_ref=reference,
                eligible=True,
                rejection_codes=(),
                quality_basis_points=facts.quality.quality_basis_points,
                health_basis_points=facts.health.success_basis_points,
                latency_milliseconds=facts.health.latency_milliseconds,
                estimated_cost_micro_usd=estimated_cost,
                score=score,
            )
            candidates.append(candidate)
            eligible_facts[key] = (facts.quality, facts.health, facts.price)
        candidates_tuple = tuple(sorted(candidates, key=lambda item: _ref_key(item.model_profile_ref)))
        eligible = [candidate for candidate in candidates_tuple if candidate.eligible]
        if not eligible:
            raise ModelRouteBlockedError(candidates_tuple)
        selected = max(
            eligible,
            key=lambda candidate: (
                candidate.score if candidate.score is not None else -(10**30),
                tuple(reversed(_ref_key(candidate.model_profile_ref))),
            ),
        )
        quality, health, price = eligible_facts[_ref_key(selected.model_profile_ref)]
        return ModelRouteDecisionV2.create(
            route_decision_id=(f"model-route-decision://{request.task_kind}/{request.route_version}"),
            request_ref=request.to_ref(),
            route_version=request.route_version,
            predecessor_route_ref=request.predecessor_route_ref,
            selected_model_profile_ref=selected.model_profile_ref,
            candidates=candidates_tuple,
            routing_policy_ref=self.policy.to_ref(),
            quality_baseline_ref=quality.to_ref(),
            health_snapshot_ref=health.to_ref(),
            price_schedule_ref=price.to_ref(),
            budget_reservation_ref=request.budget_reservation_ref,
            audit=request.audit,
        )

    def _governance_reason(
        self,
        request: ModelRouteRequestV2,
        *,
        profile_key: tuple[str, str, str, str],
        provider_id: str,
        classifications: tuple[str, ...],
        residencies: tuple[str, ...],
        availability: str,
        agent_allowed: bool,
        policy_models: set[tuple[str, str, str, str]],
        request_models: set[tuple[str, str, str, str]],
        predecessor_model: ObjectRef | None,
        reference: ObjectRef,
    ) -> ModelRouteRejectionCodeV2 | None:
        if not agent_allowed:
            return ModelRouteRejectionCodeV2.GOVERNANCE_AGENT_NOT_ALLOWED
        if profile_key not in policy_models or profile_key not in request_models:
            return ModelRouteRejectionCodeV2.GOVERNANCE_MODEL_NOT_ALLOWED
        if provider_id not in self.policy.allowed_provider_ids:
            return ModelRouteRejectionCodeV2.GOVERNANCE_MODEL_NOT_ALLOWED
        if request.data_classification not in classifications:
            return ModelRouteRejectionCodeV2.GOVERNANCE_DATA_CLASSIFICATION
        if request.residency not in residencies:
            return ModelRouteRejectionCodeV2.GOVERNANCE_RESIDENCY
        if availability == "UNAVAILABLE":
            return ModelRouteRejectionCodeV2.PROVIDER_UNAVAILABLE
        if predecessor_model is not None and reference == predecessor_model:
            return ModelRouteRejectionCodeV2.PREDECESSOR_MODEL_EXCLUDED
        if (
            request.generator_model_profile_ref is not None
            and reference == request.generator_model_profile_ref
        ):
            return ModelRouteRejectionCodeV2.GENERATOR_JUDGE_COLLISION
        return None

    @staticmethod
    def _validate_successor(
        request: ModelRouteRequestV2,
        predecessor: ModelRouteDecisionV2 | None,
    ) -> None:
        if request.route_version == 1:
            if predecessor is not None:
                raise ModelRouteError("first route cannot receive predecessor authority")
            return
        if predecessor is None or request.predecessor_route_ref != predecessor.to_ref():
            raise ModelRouteError("successor route requires the exact predecessor decision")


def _capability_reason(
    request: ModelRouteRequestV2,
    *,
    capabilities: tuple[str, ...],
    context_limit: int,
    output_limit: int,
) -> ModelRouteRejectionCodeV2 | None:
    if not set(request.required_capabilities).issubset(capabilities):
        return ModelRouteRejectionCodeV2.CAPABILITY_MISMATCH
    if request.input_token_budget + request.output_token_budget > context_limit:
        return ModelRouteRejectionCodeV2.CONTEXT_LIMIT
    if request.output_token_budget > output_limit:
        return ModelRouteRejectionCodeV2.OUTPUT_LIMIT
    return None


def _rejected(
    reference: ObjectRef,
    reason: ModelRouteRejectionCodeV2,
) -> ModelRouteCandidateV2:
    return ModelRouteCandidateV2(
        model_profile_ref=reference,
        eligible=False,
        rejection_codes=(reason,),
        quality_basis_points=None,
        health_basis_points=None,
        latency_milliseconds=None,
        estimated_cost_micro_usd=None,
        score=None,
    )


def _ref_key(value: ObjectRef) -> tuple[str, str, str, str]:
    return (
        value.object_type,
        value.object_id,
        value.object_version,
        value.object_sha256,
    )


__all__ = ["ModelRouteBlockedError", "ModelRouteError", "ModelRouter"]
