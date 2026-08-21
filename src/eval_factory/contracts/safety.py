from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from eval_factory.contracts.core import (
    ContractAudit,
    ContractModel,
    EvidenceRef,
    Identifier,
    ObjectRef,
    RelativePath,
    Sha256,
)


class OriginClass(StrEnum):
    USER_SUPPLIED_INPUT = "USER_SUPPLIED_INPUT"
    PREEXISTING_WORKSPACE_INPUT = "PREEXISTING_WORKSPACE_INPUT"
    SYSTEM_OR_HARNESS_CONTEXT = "SYSTEM_OR_HARNESS_CONTEXT"
    AGENT_RETRIEVED_EXTERNAL = "AGENT_RETRIEVED_EXTERNAL"
    AGENT_GENERATED_INTERMEDIATE = "AGENT_GENERATED_INTERMEDIATE"
    AGENT_GENERATED_FINAL = "AGENT_GENERATED_FINAL"
    UNKNOWN = "UNKNOWN"


class TaintLabel(StrEnum):
    FINAL_OUTPUT_DERIVED = "FINAL_OUTPUT_DERIVED"
    PRIVATE_REFERENCE_DERIVED = "PRIVATE_REFERENCE_DERIVED"
    GRADER_RULE_DERIVED = "GRADER_RULE_DERIVED"
    UNKNOWN_DERIVATION = "UNKNOWN_DERIVATION"
    SENSITIVE_SOURCE_DERIVED = "SENSITIVE_SOURCE_DERIVED"
    UNTRUSTED_INSTRUCTION_DERIVED = "UNTRUSTED_INSTRUCTION_DERIVED"


class ContentRiskLabel(StrEnum):
    ANSWER_BEARING = "ANSWER_BEARING"
    HIDDEN_PASS_CONDITION = "HIDDEN_PASS_CONDITION"
    SECRET = "SECRET"
    RESTRICTED_PII = "RESTRICTED_PII"
    PROMPT_INJECTION = "PROMPT_INJECTION"
    UNSCANNABLE_CONTENT = "UNSCANNABLE_CONTENT"


class Visibility(StrEnum):
    PRIVATE_STORE = "PRIVATE_STORE"
    PRIVILEGED_AUDIT = "PRIVILEGED_AUDIT"
    STAGE_PROJECTION = "STAGE_PROJECTION"
    EVALUATOR_PROJECTION = "EVALUATOR_PROJECTION"
    CONTESTANT_VISIBLE = "CONTESTANT_VISIBLE"
    PUBLIC_RELEASE = "PUBLIC_RELEASE"


class Disposition(StrEnum):
    ALLOW_INPUT_EVIDENCE = "ALLOW_INPUT_EVIDENCE"
    ALLOW_STRUCTURE_ONLY = "ALLOW_STRUCTURE_ONLY"
    ALLOW_EXTERNAL_LEAD_ONLY = "ALLOW_EXTERNAL_LEAD_ONLY"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    QUARANTINE = "QUARANTINE"
    REJECT = "REJECT"


NON_WAIVABLE_TAINTS = {
    TaintLabel.FINAL_OUTPUT_DERIVED,
    TaintLabel.PRIVATE_REFERENCE_DERIVED,
    TaintLabel.GRADER_RULE_DERIVED,
}
NON_WAIVABLE_RISKS = {
    ContentRiskLabel.ANSWER_BEARING,
    ContentRiskLabel.HIDDEN_PASS_CONDITION,
    ContentRiskLabel.SECRET,
}
ALLOW_DISPOSITIONS = {
    Disposition.ALLOW_INPUT_EVIDENCE,
    Disposition.ALLOW_STRUCTURE_ONLY,
    Disposition.ALLOW_EXTERNAL_LEAD_ONLY,
}


class ProvenanceDecision(ContractModel):
    schema_version: Literal["eval-factory/provenance-decision/v1"] = "eval-factory/provenance-decision/v1"
    provenance_decision_id: Identifier
    subject_ref: ObjectRef
    origin_class: OriginClass
    taint_labels: frozenset[TaintLabel] = frozenset()
    content_risk_labels: frozenset[ContentRiskLabel] = frozenset()
    visibility: Visibility
    disposition: Disposition
    derived_from: tuple[ObjectRef, ...] = ()
    rule_ids: tuple[Identifier, ...] = Field(min_length=1)
    source_event_refs: tuple[ObjectRef, ...] = ()
    confidence: float = Field(ge=0, le=1)
    review_required: bool
    policy_version: str = Field(min_length=1, max_length=128)
    subject_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def enforce_hard_gates(self) -> ProvenanceDecision:
        non_waivable = bool(
            self.taint_labels & NON_WAIVABLE_TAINTS or self.content_risk_labels & NON_WAIVABLE_RISKS
        )
        fail_closed = (
            TaintLabel.UNKNOWN_DERIVATION in self.taint_labels
            or ContentRiskLabel.UNSCANNABLE_CONTENT in self.content_risk_labels
        )
        if (non_waivable or fail_closed) and self.disposition in ALLOW_DISPOSITIONS:
            raise ValueError("non-waivable, unknown, or unscannable subject cannot be allowed")
        return self


class TaintEdge(ContractModel):
    schema_version: Literal["eval-factory/taint-edge/v1"] = "eval-factory/taint-edge/v1"
    taint_edge_id: Identifier
    parent_refs: tuple[ObjectRef, ...]
    child_ref: ObjectRef
    operation: Identifier
    rule_id: Identifier
    transform_verified: bool
    audit: ContractAudit


class ProjectionPolicy(ContractModel):
    schema_version: Literal["eval-factory/projection-policy/v1"] = "eval-factory/projection-policy/v1"
    projection_policy_id: Identifier
    principal_type: Identifier
    purpose: Identifier
    recursive_allow_fields: tuple[str, ...] = Field(min_length=1)
    denied_object_types: tuple[Identifier, ...] = ()
    source_schema_versions: tuple[str, ...] = Field(min_length=1)
    policy_version: str = Field(min_length=1, max_length=128)


class EvidenceBundle(ContractModel):
    schema_version: Literal["eval-factory/evidence-bundle/v1"] = "eval-factory/evidence-bundle/v1"
    evidence_bundle_id: Identifier
    source_trace_id: Identifier
    trace_ir_version_id: Identifier
    consumer_stage: Identifier
    purpose: Identifier
    projection_policy_ref: ObjectRef
    evidence: tuple[EvidenceRef, ...]
    excluded_subject_refs: tuple[ObjectRef, ...] = ()
    returned_characters: int = Field(ge=0)
    max_characters: int = Field(ge=0)
    tainted_content_included: Literal[False] = False
    bundle_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def enforce_budget(self) -> EvidenceBundle:
        if self.returned_characters > self.max_characters:
            raise ValueError("returned characters exceed authorized budget")
        return self


class SourceEvidence(ContractModel):
    schema_version: Literal["eval-factory/source-evidence/v1"] = "eval-factory/source-evidence/v1"
    source_evidence_id: Identifier
    source_uri: str = Field(min_length=3, max_length=2048)
    retrieved_at: str = Field(min_length=1, max_length=128)
    retrieval_policy_version: str = Field(min_length=1, max_length=128)
    usage_basis: str = Field(min_length=1, max_length=2000)
    content_ref: ObjectRef
    content_sha256: Sha256
    supported_claim_ids: tuple[Identifier, ...] = Field(min_length=1)
    provenance_decision_ref: ObjectRef
    audit: ContractAudit


class PackageInventoryMember(ContractModel):
    schema_version: Literal["eval-factory/package-inventory-member/v1"] = (
        "eval-factory/package-inventory-member/v1"
    )
    normalized_path: RelativePath
    member_type: Literal["DIRECTORY", "FILE", "NESTED_MEMBER"]
    media_type: str | None = Field(default=None, max_length=255)
    size_bytes: int = Field(ge=0)
    content_sha256: Sha256 | None = None
    container_ref: str | None = Field(default=None, max_length=1024)


class ProvenanceManifestEntry(ContractModel):
    schema_version: Literal["eval-factory/provenance-manifest-entry/v1"] = (
        "eval-factory/provenance-manifest-entry/v1"
    )
    inventory_member: PackageInventoryMember
    provenance_decision_ref: ObjectRef
    derivation_closure_refs: tuple[ObjectRef, ...]


class ProvenanceManifest(ContractModel):
    schema_version: Literal["eval-factory/provenance-manifest/v1"] = "eval-factory/provenance-manifest/v1"
    provenance_manifest_id: Identifier
    package_sha256: Sha256
    inventory_ref: ObjectRef
    entries: tuple[ProvenanceManifestEntry, ...]
    exact_set_verified: bool
    policy_version: str = Field(min_length=1, max_length=128)
    audit: ContractAudit
