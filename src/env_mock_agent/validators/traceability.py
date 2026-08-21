from __future__ import annotations

from env_mock_agent.schemas import (
    Criticality,
    FindingCategory,
    ReconstructionStrategy,
    Severity,
    ValidationFinding,
)
from env_mock_agent.validators.base import (
    ArtifactValidator,
    ValidationRequest,
    finding,
)

EVIDENCE_REQUIRED_STRATEGIES = {
    ReconstructionStrategy.SEARCH_DOWNLOAD,
    ReconstructionStrategy.SYNTHESIZE_GROUNDED,
    ReconstructionStrategy.MIXED,
}


class TraceabilityValidator(ArtifactValidator):
    name = "traceability"

    def validate(self, request: ValidationRequest) -> list[ValidationFinding]:
        dependency = request.dependency
        if dependency is None:
            return []
        evidence_by_id = {
            item.source_id: item
            for item in request.evidence
            if item.retrieval_error is None and (item.sha256 or item.relevant_excerpt)
        }
        referenced = [
            evidence_by_id[source_id]
            for source_id in request.plan.source_evidence_ids
            if source_id in evidence_by_id
        ]
        supported = [
            item
            for item in evidence_by_id.values()
            if dependency.dependency_id in item.supported_dependencies
        ]
        requires_evidence = (
            dependency.reconstruction_strategy in EVIDENCE_REQUIRED_STRATEGIES
            and dependency.criticality in {Criticality.CRITICAL, Criticality.HIGH}
        )
        if requires_evidence and not referenced and not supported:
            return [
                finding(
                    self.name,
                    request,
                    Severity.P1,
                    FindingCategory.TRACEABILITY,
                    "critical grounded dependency has no usable SourceEvidence",
                    evidence=[
                        f"dependency={dependency.dependency_id}",
                        f"strategy={dependency.reconstruction_strategy.value}",
                    ],
                    repair_action="retrieve an approved source or mark the dependency blocked",
                )
            ]
        missing = [
            source_id for source_id in request.plan.source_evidence_ids if source_id not in evidence_by_id
        ]
        return [
            finding(
                self.name,
                request,
                Severity.P1,
                FindingCategory.TRACEABILITY,
                f"referenced SourceEvidence is missing or unusable: {source_id}",
            )
            for source_id in missing
        ]
