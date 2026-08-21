from __future__ import annotations

import hashlib

from eval_factory.agent_system.store import FactoryControlStore
from eval_factory.agent_system.trace_candidate_store import (
    FactoryTraceCandidateMaterialStore,
)
from eval_factory.contracts.agent_system_v2 import (
    CoreVerticalResultV2,
    EvaluationRequirementSpecV2,
    TraceCandidateDispositionV2,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.dataset_runtime_v2 import (
    FactoryDatasetRunRequestV2,
)
from eval_factory.harness.contracts import sorted_refs


class GenericAgentCandidateAggregateError(RuntimeError):
    pass


class GenericAgentCandidateAggregate:
    """Rebuilds the old Core fan-in contract from per-source authorities."""

    def __init__(
        self,
        *,
        request: FactoryDatasetRunRequestV2,
        requirement: EvaluationRequirementSpecV2,
        factory_store: FactoryControlStore,
        candidate_store: FactoryTraceCandidateMaterialStore,
        source_refs: tuple[ObjectRef, ...],
    ) -> None:
        self.request = request
        self.requirement = requirement
        self.factory_store = factory_store
        self.candidate_store = candidate_store
        self.source_refs = source_refs

    def build(self, *, audit: ContractAudit) -> CoreVerticalResultV2:
        materials = tuple(
            self.candidate_store.get(
                run_id=self.request.dataset_run_id,
                trace_source_ref=source_ref,
            )
            for source_ref in self.source_refs
        )
        candidates = []
        non_candidates = []
        blocked = []
        prompts = []
        intents = []
        rewrites = []
        planning = self.factory_store.get_planning_authority(
            self.request.dataset_run_id,
        )
        routes: set[ObjectRef] = {planning.route_decision_ref}
        for material in materials:
            prepared = material.preparation
            decision = prepared.decision
            if decision.disposition is TraceCandidateDispositionV2.CANDIDATE:
                candidates.append(decision.to_ref())
                if (
                    prepared.extracted_prompt is None
                    or prepared.inferred_intent is None
                    or prepared.rewrite_candidate is None
                ):
                    raise GenericAgentCandidateAggregateError(
                        "candidate authority is incomplete",
                    )
                prompts.append(prepared.extracted_prompt.to_ref())
                intents.append(prepared.inferred_intent.to_ref())
                rewrites.append(prepared.rewrite_candidate.to_ref())
            elif decision.disposition is TraceCandidateDispositionV2.NON_CANDIDATE:
                non_candidates.append(decision.to_ref())
            else:
                blocked.append(decision.to_ref())
            routes.update(prepared.route_refs)
        route_refs = sorted_refs(routes)
        if len(route_refs) < 1 + (2 * len(candidates)):
            raise GenericAgentCandidateAggregateError(
                "candidate aggregate lacks governed route authority",
            )
        digest = hashlib.sha256(
            "|".join(
                value.object_sha256
                for value in sorted_refs(material.authority.to_ref() for material in materials)
            ).encode()
        ).hexdigest()
        return CoreVerticalResultV2.create(
            result_id=f"core-vertical-result://harness/{digest}",
            manifest_ref=self.request.manifest_ref,
            requirement_spec_ref=self.requirement.to_ref(),
            candidate_decision_refs=sorted_refs(candidates),
            non_candidate_decision_refs=sorted_refs(non_candidates),
            blocked_decision_refs=sorted_refs(blocked),
            extracted_prompt_refs=sorted_refs(prompts),
            inferred_intent_refs=sorted_refs(intents),
            rewrite_candidate_refs=sorted_refs(rewrites),
            route_decision_refs=route_refs,
            source_count=len(materials),
            audit=audit,
        )


__all__ = [
    "GenericAgentCandidateAggregate",
    "GenericAgentCandidateAggregateError",
]
