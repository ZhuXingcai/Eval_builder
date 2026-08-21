from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Literal

from pydantic import Field, field_validator, model_validator

from env_mock_agent.facade import (
    AttachmentExecutionFailureCodeV2,
    AttachmentExecutionRequestV2,
    AttachmentExecutionResultV2,
    AttachmentRouteCandidateKind,
    AttachmentRouteDecisionV2,
    AttachmentRouteOutcome,
    CapabilityToken,
    FacadeObjectRef,
    WorldLedgerSnapshotV2,
    attachment_execution_request_ref,
    attachment_execution_result_ref,
    attachment_route_decision_ref,
    validate_attachment_execution_request_identity,
    validate_attachment_execution_result_identity,
    validate_world_ledger_snapshot_identity,
    world_ledger_snapshot_ref,
)
from eval_factory.contracts.attachment import (
    ArtifactBuildResult,
    ArtifactBuildSpec,
    ArtifactBuildStatus,
    ArtifactEvidenceRow,
    AttachmentReconstructionResult,
    ReconstructionMode,
)
from eval_factory.contracts.core import (
    ContractAudit,
    EvidencePolarity,
    EvidenceRef,
    FailureClass,
    FailureRecord,
    Identifier,
    ObjectRef,
    RelativePath,
    Sha256,
)
from eval_factory.contracts.core_v2 import ContractModelV2, canonical_value_v2
from eval_factory.contracts.safety import (
    Disposition,
    OriginClass,
    ProvenanceDecision,
    SourceEvidence,
    Visibility,
)
from eval_factory.contracts.task import AttachmentCriticality, EvidencePriority

_DENIED_DEPENDENCY_REF_MARKERS = frozenset(
    {
        "answer-bearing",
        "completed-deliverable",
        "configured-pii",
        "final-answer",
        "final-output",
        "grader-rule",
        "hidden-pass-condition",
        "private-reference",
        "quarantine",
        "raw-trace",
        "raw-traj",
        "restricted-pii",
        "secret",
        "sensitive-pii",
        "trace-raw",
    }
)


class AttachmentPlanningContextV2(ContractModelV2):
    schema_version: Literal["eval-factory/attachment-planning-context/v2"] = (
        "eval-factory/attachment-planning-context/v2"
    )
    attachment_planning_context_id: Identifier
    producer_task_view_ref: ObjectRef
    producer_storage_authorization_ref: ObjectRef
    safe_evidence_bundle_ref: ObjectRef
    projection_policy_ref: ObjectRef
    source_trace_id: Identifier
    trace_ir_version_id: Identifier
    producer_principal_id: Identifier
    source_task_draft_sha256: Sha256
    source_contract_chain_sha256: Sha256
    policy_version: str = Field(min_length=1, max_length=128)
    attachment_planning_context_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_refs(self) -> AttachmentPlanningContextV2:
        _require_ref_type(
            self.producer_task_view_ref,
            "producer-task-view",
            "producer_task_view_ref",
        )
        if self.producer_task_view_ref.object_version != "v2":
            raise ValueError("producer_task_view_ref must reference ProducerTaskView v2")
        _require_ref_type(
            self.producer_storage_authorization_ref,
            "producer-storage-authorization",
            "producer_storage_authorization_ref",
        )
        if self.producer_storage_authorization_ref.object_version != "v2":
            raise ValueError(
                "producer_storage_authorization_ref must reference ProducerStorageAuthorization v2"
            )
        _require_ref_type(
            self.safe_evidence_bundle_ref,
            "evidence-bundle",
            "safe_evidence_bundle_ref",
        )
        if self.safe_evidence_bundle_ref.object_version != "v1":
            raise ValueError("safe_evidence_bundle_ref must reference EvidenceBundle v1")
        _require_ref_type(
            self.projection_policy_ref,
            "projection-policy",
            "projection_policy_ref",
        )
        return self


def attachment_planning_context_carried_sha256(
    context: AttachmentPlanningContextV2,
) -> str:
    payload = {
        "producer_task_view_ref": _ref_payload(context.producer_task_view_ref),
        "producer_storage_authorization_ref": _ref_payload(context.producer_storage_authorization_ref),
        "safe_evidence_bundle_ref": _ref_payload(context.safe_evidence_bundle_ref),
        "projection_policy_ref": _ref_payload(context.projection_policy_ref),
        "source_trace_id": context.source_trace_id,
        "trace_ir_version_id": context.trace_ir_version_id,
        "producer_principal_id": context.producer_principal_id,
        "source_task_draft_sha256": context.source_task_draft_sha256,
        "source_contract_chain_sha256": context.source_contract_chain_sha256,
        "policy_version": context.policy_version,
    }
    encoded = json.dumps(
        canonical_value_v2(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def attachment_planning_context_ref(
    context: AttachmentPlanningContextV2,
) -> ObjectRef:
    return ObjectRef(
        object_type="attachment-planning-context",
        object_id=context.attachment_planning_context_id,
        object_version="v2",
        object_sha256=context.attachment_planning_context_sha256,
    )


def _require_ref_type(
    ref: ObjectRef,
    expected_type: str,
    field_name: str,
) -> None:
    if ref.object_type != expected_type:
        raise ValueError(f"{field_name} must reference {expected_type}")


def _ref_payload(ref: ObjectRef) -> dict[str, object]:
    return ref.model_dump(mode="json", exclude_none=False)


ArtifactEvidenceAggregateModeV2 = Literal[
    "TRACE_RICH",
    "SKELETON_GUIDED",
    "PROMPT_ONLY",
    "MIXED",
    "BLOCKED",
]


class ArtifactEvidenceTargetV2(ContractModelV2):
    schema_version: Literal["eval-factory/artifact-evidence-target/v2"] = (
        "eval-factory/artifact-evidence-target/v2"
    )
    artifact_evidence_target_id: Identifier
    attachment_planning_context_ref: ObjectRef
    attachment_dependency_id: Identifier
    artifact_id: Identifier
    logical_path: RelativePath
    media_type: str = Field(
        min_length=3,
        max_length=255,
        pattern=r"^[a-z0-9.+-]+/[a-z0-9.+-]+$",
    )
    criticality: AttachmentCriticality
    requirement_evidence: tuple[EvidenceRef, ...] = Field(min_length=1)
    candidate_source_refs: tuple[ObjectRef, ...] = ()
    policy_version: str = Field(min_length=1, max_length=128)
    artifact_evidence_target_sha256: Sha256
    audit: ContractAudit

    @field_validator("criticality", mode="before")
    @classmethod
    def parse_criticality(
        cls,
        value: object,
    ) -> AttachmentCriticality:
        if isinstance(value, AttachmentCriticality):
            return value
        if isinstance(value, str):
            return AttachmentCriticality(value)
        raise TypeError("criticality must be an AttachmentCriticality")

    @model_validator(mode="after")
    def validate_target(self) -> ArtifactEvidenceTargetV2:
        _require_ref_version(
            self.attachment_planning_context_ref,
            "attachment-planning-context",
            "v2",
            "attachment_planning_context_ref",
        )
        evidence_ids = tuple(item.evidence_ref_id for item in self.requirement_evidence)
        _require_unique("requirement evidence IDs", evidence_ids)
        if evidence_ids != tuple(sorted(evidence_ids)):
            raise ValueError("requirement evidence must be sorted by evidence_ref_id")
        candidate_keys = tuple(_ref_key(ref) for ref in self.candidate_source_refs)
        _require_unique("candidate source refs", candidate_keys)
        if candidate_keys != tuple(sorted(candidate_keys)):
            raise ValueError("candidate source refs must be sorted")
        return self


class ArtifactEvidenceRowV2(ContractModelV2):
    schema_version: Literal["eval-factory/artifact-evidence-row/v2"] = "eval-factory/artifact-evidence-row/v2"
    artifact_evidence_row_id: Identifier
    attachment_planning_context_ref: ObjectRef
    artifact_evidence_target_ref: ObjectRef
    attachment_dependency_id: Identifier
    row: ArtifactEvidenceRow
    r2_row_sha256: Sha256
    r2_policy_version: str = Field(min_length=1, max_length=128)
    policy_version: str = Field(min_length=1, max_length=128)
    artifact_evidence_row_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_row(self) -> ArtifactEvidenceRowV2:
        _require_ref_version(
            self.attachment_planning_context_ref,
            "attachment-planning-context",
            "v2",
            "attachment_planning_context_ref",
        )
        _require_ref_version(
            self.artifact_evidence_target_ref,
            "artifact-evidence-target",
            "v2",
            "artifact_evidence_target_ref",
        )
        if self.r2_row_sha256 != self.row.canonical_sha256():
            raise ValueError("R2 row hash is stale or mismatched")
        return self


class ArtifactEvidenceMatrixV2(ContractModelV2):
    schema_version: Literal["eval-factory/artifact-evidence-matrix/v2"] = (
        "eval-factory/artifact-evidence-matrix/v2"
    )
    artifact_evidence_matrix_id: Identifier
    attachment_planning_context_ref: ObjectRef
    producer_task_view_ref: ObjectRef
    safe_evidence_bundle_ref: ObjectRef
    source_r2_matrix_ref: ObjectRef
    rows: tuple[ArtifactEvidenceRowV2, ...] = Field(min_length=1)
    aggregate_mode: ArtifactEvidenceAggregateModeV2
    r2_policy_version: str = Field(min_length=1, max_length=128)
    policy_version: str = Field(min_length=1, max_length=128)
    artifact_evidence_matrix_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_matrix(self) -> ArtifactEvidenceMatrixV2:
        _require_ref_version(
            self.attachment_planning_context_ref,
            "attachment-planning-context",
            "v2",
            "attachment_planning_context_ref",
        )
        _require_ref_version(
            self.producer_task_view_ref,
            "producer-task-view",
            "v2",
            "producer_task_view_ref",
        )
        _require_ref_version(
            self.safe_evidence_bundle_ref,
            "evidence-bundle",
            "v1",
            "safe_evidence_bundle_ref",
        )
        _require_ref_version(
            self.source_r2_matrix_ref,
            "artifact-evidence-matrix",
            "v1",
            "source_r2_matrix_ref",
        )
        _validate_matrix_rows(
            self.rows,
            self.attachment_planning_context_ref,
        )
        if any(item.r2_policy_version != self.r2_policy_version for item in self.rows):
            raise ValueError("artifact evidence row R2 policy must match the matrix R2 policy")
        if any(item.policy_version != self.policy_version for item in self.rows):
            raise ValueError("artifact evidence row policy must match the matrix policy")
        expected_aggregate = artifact_evidence_aggregate_mode_v2(self.rows)
        if self.aggregate_mode != expected_aggregate:
            raise ValueError("aggregate_mode does not match exact R5 row modes")
        return self


class PromptOnlyDependencyEvidenceBindingV2(ContractModelV2):
    schema_version: Literal["eval-factory/prompt-only-dependency-evidence-binding/v2"] = (
        "eval-factory/prompt-only-dependency-evidence-binding/v2"
    )
    attachment_dependency_id: Identifier
    evidence_priority: EvidencePriority
    evidence: tuple[EvidenceRef, ...] = Field(min_length=1)

    @field_validator("evidence_priority", mode="before")
    @classmethod
    def parse_evidence_priority(
        cls,
        value: object,
    ) -> EvidencePriority:
        if isinstance(value, EvidencePriority):
            return value
        if isinstance(value, str):
            return EvidencePriority(value)
        raise TypeError("evidence_priority must be an EvidencePriority")

    @model_validator(mode="after")
    def validate_binding(self) -> PromptOnlyDependencyEvidenceBindingV2:
        evidence_ids = tuple(item.evidence_ref_id for item in self.evidence)
        _require_unique("prompt-only dependency evidence IDs", evidence_ids)
        if evidence_ids != tuple(sorted(evidence_ids)):
            raise ValueError("prompt-only dependency evidence must be sorted by evidence_ref_id")
        if any(
            item.polarity is not EvidencePolarity.POSITIVE
            or not item.capability_complete
            or _unsafe_dependency_ref(item.subject_ref)
            for item in self.evidence
        ):
            raise ValueError("prompt-only dependency evidence is unsafe or incomplete")
        return self


class PromptOnlyDependencyPlanningContextV2(ContractModelV2):
    schema_version: Literal["eval-factory/prompt-only-dependency-planning-context/v2"] = (
        "eval-factory/prompt-only-dependency-planning-context/v2"
    )
    prompt_only_dependency_planning_context_id: Identifier
    attachment_planning_context_ref: ObjectRef
    producer_task_view_ref: ObjectRef
    task_prompt_safety_gate_ref: ObjectRef
    source_task_draft_sha256: Sha256
    source_trace_id: Identifier
    dependency_evidence_bindings: tuple[
        PromptOnlyDependencyEvidenceBindingV2,
        ...,
    ] = ()
    policy_version: str = Field(min_length=1, max_length=128)
    prompt_only_dependency_planning_context_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_context(self) -> PromptOnlyDependencyPlanningContextV2:
        _require_ref_version(
            self.attachment_planning_context_ref,
            "attachment-planning-context",
            "v2",
            "attachment_planning_context_ref",
        )
        _require_ref_version(
            self.producer_task_view_ref,
            "producer-task-view",
            "v2",
            "producer_task_view_ref",
        )
        _require_ref_version(
            self.task_prompt_safety_gate_ref,
            "task-prompt-safety-gate",
            "v2",
            "task_prompt_safety_gate_ref",
        )
        dependency_ids = tuple(item.attachment_dependency_id for item in self.dependency_evidence_bindings)
        _require_unique(
            "prompt-only dependency evidence bindings",
            dependency_ids,
        )
        if dependency_ids != tuple(sorted(dependency_ids)):
            raise ValueError("prompt-only dependency evidence bindings must be sorted")
        if any(
            span.source_trace_id != self.source_trace_id
            for binding in self.dependency_evidence_bindings
            for evidence in binding.evidence
            for span in evidence.source_spans
        ):
            raise ValueError("prompt-only dependency evidence source trace is mismatched")
        return self


class PromptOnlyDependencyDiscoveryV2(ContractModelV2):
    schema_version: Literal["eval-factory/prompt-only-dependency-discovery/v2"] = (
        "eval-factory/prompt-only-dependency-discovery/v2"
    )
    prompt_only_dependency_discovery_id: Identifier
    attachment_planning_context_ref: ObjectRef
    producer_task_view_ref: ObjectRef
    dependency_planning_context_ref: ObjectRef
    prompt_boundary_enforcement_ref: ObjectRef
    targets: tuple[ArtifactEvidenceTargetV2, ...] = Field(min_length=1)
    existing_target_refs: tuple[ObjectRef, ...] = ()
    discovered_target_refs: tuple[ObjectRef, ...] = ()
    deterministic_dependency_ids: tuple[Identifier, ...] = ()
    semantic_dependency_ids: tuple[Identifier, ...] = ()
    semantic_evaluated: bool
    model_profile: Identifier | None = None
    prompt_version: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
    )
    policy_version: str = Field(min_length=1, max_length=128)
    prompt_only_dependency_discovery_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_discovery(self) -> PromptOnlyDependencyDiscoveryV2:
        _require_ref_version(
            self.attachment_planning_context_ref,
            "attachment-planning-context",
            "v2",
            "attachment_planning_context_ref",
        )
        _require_ref_version(
            self.producer_task_view_ref,
            "producer-task-view",
            "v2",
            "producer_task_view_ref",
        )
        _require_ref_version(
            self.dependency_planning_context_ref,
            "prompt-only-dependency-planning-context",
            "v2",
            "dependency_planning_context_ref",
        )
        _require_ref_type(
            self.prompt_boundary_enforcement_ref,
            "prompt-boundary-enforcement",
            "prompt_boundary_enforcement_ref",
        )
        _validate_discovery_targets(
            targets=self.targets,
            context_ref=self.attachment_planning_context_ref,
            existing_refs=self.existing_target_refs,
            discovered_refs=self.discovered_target_refs,
            deterministic_ids=self.deterministic_dependency_ids,
            semantic_ids=self.semantic_dependency_ids,
        )
        if self.semantic_evaluated:
            if self.model_profile is None or self.prompt_version is None:
                raise ValueError("semantic discovery requires model profile and prompt version")
        elif self.model_profile is not None or self.prompt_version is not None:
            raise ValueError("deterministic discovery cannot carry semantic model metadata")
        return self


class PublicSourceRetrievalPolicyV2(ContractModelV2):
    schema_version: Literal["eval-factory/public-source-retrieval-policy/v2"] = (
        "eval-factory/public-source-retrieval-policy/v2"
    )
    public_source_retrieval_policy_id: Identifier
    approved_search_provider_ids: tuple[Identifier, ...] = ()
    approved_fetch_provider_ids: tuple[Identifier, ...] = Field(min_length=1)
    allowed_schemes: tuple[str, ...] = Field(min_length=1)
    allowed_host_suffixes: tuple[str, ...] = Field(min_length=1)
    max_search_results: int = Field(ge=1, le=100)
    max_fetch_bytes: int = Field(ge=1)
    query_egress_policy_ref: ObjectRef
    network_policy_ref: ObjectRef
    source_usage_policy_ref: ObjectRef
    safety_scan_policy_ref: ObjectRef
    license_policy_ref: ObjectRef
    policy_version: str = Field(min_length=1, max_length=128)
    public_source_retrieval_policy_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_policy(self) -> PublicSourceRetrievalPolicyV2:
        for label, values in (
            (
                "approved search provider IDs",
                self.approved_search_provider_ids,
            ),
            (
                "approved fetch provider IDs",
                self.approved_fetch_provider_ids,
            ),
            ("allowed schemes", self.allowed_schemes),
            ("allowed host suffixes", self.allowed_host_suffixes),
        ):
            _require_unique(label, values)
            if values != tuple(sorted(values)):
                raise ValueError(f"{label} must be sorted")
        if any(item not in {"http", "https"} for item in self.allowed_schemes):
            raise ValueError("allowed schemes must contain only http or https")
        for suffix in self.allowed_host_suffixes:
            if (
                suffix != suffix.casefold()
                or suffix.startswith((".", "*"))
                or "/" in suffix
                or ":" in suffix
                or suffix in {"localhost", "local"}
                or suffix.endswith((".localhost", ".local"))
            ):
                raise ValueError("allowed host suffixes must be normalized public DNS suffixes")
        for ref, expected_type, field_name in (
            (
                self.query_egress_policy_ref,
                "query-egress-policy",
                "query_egress_policy_ref",
            ),
            (
                self.network_policy_ref,
                "network-policy",
                "network_policy_ref",
            ),
            (
                self.source_usage_policy_ref,
                "source-usage-policy",
                "source_usage_policy_ref",
            ),
            (
                self.safety_scan_policy_ref,
                "safety-scan-policy",
                "safety_scan_policy_ref",
            ),
            (
                self.license_policy_ref,
                "source-license-policy",
                "license_policy_ref",
            ),
        ):
            _require_ref_type(ref, expected_type, field_name)
        return self


class SourceEvidenceClaimBindingV2(ContractModelV2):
    schema_version: Literal["eval-factory/source-evidence-claim-binding/v2"] = (
        "eval-factory/source-evidence-claim-binding/v2"
    )
    artifact_evidence_target_ref: ObjectRef
    attachment_dependency_id: Identifier
    supported_claim_ids: tuple[Identifier, ...] = Field(min_length=1)
    usage_basis: Literal["FACTS_ONLY", "STRUCTURE_AND_STYLE_ONLY"]

    @model_validator(mode="after")
    def validate_binding(self) -> SourceEvidenceClaimBindingV2:
        _require_ref_version(
            self.artifact_evidence_target_ref,
            "artifact-evidence-target",
            "v2",
            "artifact_evidence_target_ref",
        )
        _require_unique("supported claim IDs", self.supported_claim_ids)
        if self.supported_claim_ids != tuple(sorted(self.supported_claim_ids)):
            raise ValueError("supported claim IDs must be sorted")
        if self.supported_claim_ids != (self.attachment_dependency_id,):
            raise ValueError("supported claim IDs must equal the exact attachment dependency ID")
        return self


class PublicSourceSafetyAssessmentV2(ContractModelV2):
    schema_version: Literal["eval-factory/public-source-safety-assessment/v2"] = (
        "eval-factory/public-source-safety-assessment/v2"
    )
    public_source_safety_assessment_id: Identifier
    fetch_result_ref: ObjectRef
    content_ref: ObjectRef
    content_sha256: Sha256
    secret_scan_ref: ObjectRef
    pii_scan_ref: ObjectRef
    prompt_injection_scan_ref: ObjectRef
    answer_leakage_scan_ref: ObjectRef
    license_assessment_ref: ObjectRef
    passed: Literal[True]
    policy_version: str = Field(min_length=1, max_length=128)
    public_source_safety_assessment_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_assessment(self) -> PublicSourceSafetyAssessmentV2:
        _require_ref_version(
            self.fetch_result_ref,
            "public-source-fetch-result",
            "v2",
            "fetch_result_ref",
        )
        _require_ref_version(
            self.content_ref,
            "public-source-content",
            "v1",
            "content_ref",
        )
        if self.content_ref.object_sha256 != self.content_sha256:
            raise ValueError("content_ref must bind content_sha256")
        for ref, expected_type, field_name in (
            (self.secret_scan_ref, "secret-scan-result", "secret_scan_ref"),
            (
                self.pii_scan_ref,
                "configured-pii-scan-result",
                "pii_scan_ref",
            ),
            (
                self.prompt_injection_scan_ref,
                "prompt-injection-scan-result",
                "prompt_injection_scan_ref",
            ),
            (
                self.answer_leakage_scan_ref,
                "answer-leakage-scan-result",
                "answer_leakage_scan_ref",
            ),
            (
                self.license_assessment_ref,
                "source-license-assessment",
                "license_assessment_ref",
            ),
        ):
            _require_ref_type(ref, expected_type, field_name)
            if ref.object_sha256 != self.content_sha256:
                raise ValueError(f"{field_name} must bind content_sha256")
        return self


class SourceEvidenceV2(ContractModelV2):
    schema_version: Literal["eval-factory/source-evidence/v2"] = "eval-factory/source-evidence/v2"
    source_evidence_id: Identifier
    attachment_planning_context_ref: ObjectRef
    producer_task_view_ref: ObjectRef
    prompt_only_dependency_discovery_ref: ObjectRef
    retrieval_policy_ref: ObjectRef
    external_lead_ref: ObjectRef | None = None
    external_lead_decision_ref: ObjectRef | None = None
    search_result_ref: ObjectRef | None = None
    fetch_result_ref: ObjectRef
    safety_assessment_ref: ObjectRef
    claim_bindings: tuple[SourceEvidenceClaimBindingV2, ...] = Field(min_length=1)
    source_evidence_v1: SourceEvidence
    content_provenance_decision: ProvenanceDecision
    policy_version: str = Field(min_length=1, max_length=128)
    source_evidence_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_evidence(self) -> SourceEvidenceV2:
        _require_ref_version(
            self.attachment_planning_context_ref,
            "attachment-planning-context",
            "v2",
            "attachment_planning_context_ref",
        )
        _require_ref_version(
            self.producer_task_view_ref,
            "producer-task-view",
            "v2",
            "producer_task_view_ref",
        )
        _require_ref_version(
            self.prompt_only_dependency_discovery_ref,
            "prompt-only-dependency-discovery",
            "v2",
            "prompt_only_dependency_discovery_ref",
        )
        _require_ref_version(
            self.retrieval_policy_ref,
            "public-source-retrieval-policy",
            "v2",
            "retrieval_policy_ref",
        )
        _require_ref_version(
            self.fetch_result_ref,
            "public-source-fetch-result",
            "v2",
            "fetch_result_ref",
        )
        _require_ref_version(
            self.safety_assessment_ref,
            "public-source-safety-assessment",
            "v2",
            "safety_assessment_ref",
        )
        lead_route = self.external_lead_ref is not None and self.external_lead_decision_ref is not None
        partial_lead_route = (self.external_lead_ref is None) != (self.external_lead_decision_ref is None)
        search_route = self.search_result_ref is not None
        if partial_lead_route or lead_route == search_route:
            raise ValueError("source evidence route must be exactly one complete lead or search route")
        if lead_route:
            assert self.external_lead_ref is not None
            assert self.external_lead_decision_ref is not None
            _require_ref_type(
                self.external_lead_decision_ref,
                "provenance-decision",
                "external_lead_decision_ref",
            )
        else:
            assert self.search_result_ref is not None
            _require_ref_version(
                self.search_result_ref,
                "public-source-search-result",
                "v2",
                "search_result_ref",
            )
        binding_keys = tuple(
            (
                _ref_key(item.artifact_evidence_target_ref),
                item.attachment_dependency_id,
            )
            for item in self.claim_bindings
        )
        _require_unique("source evidence claim bindings", binding_keys)
        if binding_keys != tuple(sorted(binding_keys)):
            raise ValueError("source evidence claim bindings must be sorted")
        claims = tuple(
            sorted({claim for binding in self.claim_bindings for claim in binding.supported_claim_ids})
        )
        frozen = self.source_evidence_v1
        decision = self.content_provenance_decision
        if (
            frozen.retrieval_policy_version != self.policy_version
            or frozen.content_ref != decision.subject_ref
            or frozen.content_sha256 != decision.subject_sha256
            or frozen.supported_claim_ids != claims
        ):
            raise ValueError("frozen SourceEvidence facts must match v2 policy, claims, and provenance")
        decision_ref = ObjectRef(
            object_type="provenance-decision",
            object_id=decision.provenance_decision_id,
            object_version="v1",
            object_sha256=decision.canonical_sha256(),
        )
        if frozen.provenance_decision_ref != decision_ref:
            raise ValueError("frozen SourceEvidence provenance ref must match the fresh decision")
        if (
            decision.origin_class is not OriginClass.SYSTEM_OR_HARNESS_CONTEXT
            or decision.visibility is not Visibility.STAGE_PROJECTION
            or decision.disposition is not Disposition.ALLOW_INPUT_EVIDENCE
            or decision.rule_ids != ("public-source-refetch/r5-04-v1",)
            or decision.taint_labels
            or decision.content_risk_labels
            or decision.review_required
        ):
            raise ValueError("fresh public source provenance decision is not safe or current")
        return self


class SourceEvidenceSetV2(ContractModelV2):
    schema_version: Literal["eval-factory/source-evidence-set/v2"] = "eval-factory/source-evidence-set/v2"
    source_evidence_set_id: Identifier
    attachment_planning_context_ref: ObjectRef
    producer_task_view_ref: ObjectRef
    prompt_only_dependency_discovery_ref: ObjectRef
    retrieval_policy_ref: ObjectRef
    retrieval_request_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    source_evidence: tuple[SourceEvidenceV2, ...] = Field(min_length=1)
    covered_target_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    not_required_target_refs: tuple[ObjectRef, ...] = ()
    policy_version: str = Field(min_length=1, max_length=128)
    source_evidence_set_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_set(self) -> SourceEvidenceSetV2:
        for ref, expected_type, field_name in (
            (
                self.attachment_planning_context_ref,
                "attachment-planning-context",
                "attachment_planning_context_ref",
            ),
            (
                self.producer_task_view_ref,
                "producer-task-view",
                "producer_task_view_ref",
            ),
            (
                self.prompt_only_dependency_discovery_ref,
                "prompt-only-dependency-discovery",
                "prompt_only_dependency_discovery_ref",
            ),
            (
                self.retrieval_policy_ref,
                "public-source-retrieval-policy",
                "retrieval_policy_ref",
            ),
        ):
            _require_ref_version(ref, expected_type, "v2", field_name)
        request_keys = tuple(_ref_key(ref) for ref in self.retrieval_request_refs)
        _require_unique("retrieval request refs", request_keys)
        if request_keys != tuple(sorted(request_keys)):
            raise ValueError("retrieval request refs must be sorted")
        if any(
            ref.object_type
            not in {
                "public-source-search-request",
                "public-source-fetch-request",
            }
            or ref.object_version != "v2"
            for ref in self.retrieval_request_refs
        ):
            raise ValueError("retrieval request refs must reference search/fetch request v2")
        evidence_ids = tuple(item.source_evidence_id for item in self.source_evidence)
        _require_unique("source evidence IDs", evidence_ids)
        if evidence_ids != tuple(sorted(evidence_ids)):
            raise ValueError("source evidence must be sorted by ID")
        for item in self.source_evidence:
            if (
                item.attachment_planning_context_ref != self.attachment_planning_context_ref
                or item.producer_task_view_ref != self.producer_task_view_ref
                or item.prompt_only_dependency_discovery_ref != self.prompt_only_dependency_discovery_ref
                or item.retrieval_policy_ref != self.retrieval_policy_ref
                or item.policy_version != self.policy_version
            ):
                raise ValueError("source evidence must bind the aggregate source graph")
        for label, refs in (
            ("covered target refs", self.covered_target_refs),
            ("not-required target refs", self.not_required_target_refs),
        ):
            keys = tuple(_ref_key(ref) for ref in refs)
            _require_unique(label, keys)
            if keys != tuple(sorted(keys)):
                raise ValueError(f"{label} must be sorted")
            for ref in refs:
                _require_ref_version(
                    ref,
                    "artifact-evidence-target",
                    "v2",
                    label,
                )
        covered = {_ref_key(ref) for ref in self.covered_target_refs}
        not_required = {_ref_key(ref) for ref in self.not_required_target_refs}
        if covered & not_required:
            raise ValueError("covered and not-required target refs must be disjoint")
        binding_targets = {
            _ref_key(binding.artifact_evidence_target_ref)
            for item in self.source_evidence
            for binding in item.claim_bindings
        }
        if binding_targets != covered:
            raise ValueError("covered target refs must exactly equal claim binding targets")
        return self


class ArtifactRouteKindV2(StrEnum):
    PROVIDER = "PROVIDER"
    RUNTIME = "RUNTIME"


class ArtifactRouteEntryOutcomeV2(StrEnum):
    ROUTED_PROVIDER = "ROUTED_PROVIDER"
    ROUTED_RUNTIME = "ROUTED_RUNTIME"
    BLOCKED_CAPABILITY = "BLOCKED_CAPABILITY"
    BLOCKED_POLICY = "BLOCKED_POLICY"


class ArtifactRoutingAggregateOutcomeV2(StrEnum):
    ROUTED = "ROUTED"
    PARTIALLY_ROUTED = "PARTIALLY_ROUTED"
    BLOCKED_CAPABILITY = "BLOCKED_CAPABILITY"
    BLOCKED_POLICY = "BLOCKED_POLICY"
    NOT_REQUIRED = "NOT_REQUIRED"


class ArtifactRoutingReasonV2(StrEnum):
    BUILD_CONTRACT_MISSING = "BUILD_CONTRACT_MISSING"
    EVIDENCE_MODE_BLOCKED = "EVIDENCE_MODE_BLOCKED"
    ROUTE_UNAVAILABLE = "ROUTE_UNAVAILABLE"


class ArtifactRoutingPolicyV2(ContractModelV2):
    schema_version: Literal["eval-factory/artifact-routing-policy/v2"] = (
        "eval-factory/artifact-routing-policy/v2"
    )
    artifact_routing_policy_id: Identifier
    deterministic_provider_first: Literal[True]
    approved_provider_ids: tuple[Identifier, ...] = Field(min_length=1)
    runtime_order: tuple[Identifier, ...] = Field(min_length=1)
    runtime_model_profile_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    provider_capability_policy_ref: ObjectRef
    runtime_capability_policy_ref: ObjectRef
    validator_policy_ref: ObjectRef
    silent_degradation_allowed: Literal[False]
    policy_version: str = Field(min_length=1, max_length=128)
    artifact_routing_policy_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_policy(self) -> ArtifactRoutingPolicyV2:
        _require_unique("approved provider IDs", self.approved_provider_ids)
        if self.approved_provider_ids != tuple(sorted(self.approved_provider_ids)):
            raise ValueError("approved provider IDs must be sorted")
        _require_unique("runtime order", self.runtime_order)
        expected_order = (
            "claude_agent_sdk",
            "claude_code_cli",
            "pi_rpc",
        )
        if self.runtime_order != expected_order:
            raise ValueError("runtime order must be Claude Agent SDK, Claude Code CLI, then Pi")
        if len(self.runtime_model_profile_refs) != len(self.runtime_order):
            raise ValueError("runtime model profile refs must align with runtime order")
        model_keys = tuple(_ref_key(ref) for ref in self.runtime_model_profile_refs)
        _require_unique("runtime model profile refs", model_keys)
        for ref in self.runtime_model_profile_refs:
            _require_ref_type(ref, "model-profile", "runtime_model_profile_refs")
        for ref, expected_type, field_name in (
            (
                self.provider_capability_policy_ref,
                "provider-capability-policy",
                "provider_capability_policy_ref",
            ),
            (
                self.runtime_capability_policy_ref,
                "runtime-capability-policy",
                "runtime_capability_policy_ref",
            ),
            (
                self.validator_policy_ref,
                "validator-policy",
                "validator_policy_ref",
            ),
        ):
            _require_ref_type(ref, expected_type, field_name)
        return self


class ArtifactBuildContractV2(ContractModelV2):
    schema_version: Literal["eval-factory/artifact-build-contract/v2"] = (
        "eval-factory/artifact-build-contract/v2"
    )
    artifact_build_contract_id: Identifier
    attachment_planning_context_ref: ObjectRef
    producer_task_view_ref: ObjectRef
    artifact_evidence_matrix_ref: ObjectRef
    artifact_evidence_row_ref: ObjectRef
    artifact_evidence_target_ref: ObjectRef
    attachment_dependency_id: Identifier
    artifact_id: Identifier
    logical_path: RelativePath
    media_type: str = Field(
        min_length=3,
        max_length=255,
        pattern=r"^[a-z0-9.+-]+/[a-z0-9.+-]+$",
    )
    asset_type: CapabilityToken
    mode: ReconstructionMode
    criticality: AttachmentCriticality
    content_contract_ref: ObjectRef
    render_contract_ref: ObjectRef
    provider_payload_ref: ObjectRef | None = None
    source_evidence_set_ref: ObjectRef | None = None
    required_provider_capability_ids: tuple[Identifier, ...]
    required_runtime_tools: tuple[CapabilityToken, ...]
    runtime_role: Identifier
    runtime_resume_required: bool
    validator_ids: tuple[Identifier, ...] = Field(min_length=1)
    policy_version: str = Field(min_length=1, max_length=128)
    artifact_build_contract_sha256: Sha256
    audit: ContractAudit

    @field_validator("mode", mode="before")
    @classmethod
    def parse_mode(cls, value: object) -> ReconstructionMode:
        if isinstance(value, ReconstructionMode):
            return value
        if isinstance(value, str):
            return ReconstructionMode(value)
        raise TypeError("mode must be a ReconstructionMode")

    @field_validator("criticality", mode="before")
    @classmethod
    def parse_criticality(
        cls,
        value: object,
    ) -> AttachmentCriticality:
        if isinstance(value, AttachmentCriticality):
            return value
        if isinstance(value, str):
            return AttachmentCriticality(value)
        raise TypeError("criticality must be an AttachmentCriticality")

    @model_validator(mode="after")
    def validate_contract(self) -> ArtifactBuildContractV2:
        for ref, expected_type, expected_version, field_name in (
            (
                self.attachment_planning_context_ref,
                "attachment-planning-context",
                "v2",
                "attachment_planning_context_ref",
            ),
            (
                self.producer_task_view_ref,
                "producer-task-view",
                "v2",
                "producer_task_view_ref",
            ),
            (
                self.artifact_evidence_matrix_ref,
                "artifact-evidence-matrix",
                "v2",
                "artifact_evidence_matrix_ref",
            ),
            (
                self.artifact_evidence_row_ref,
                "artifact-evidence-row",
                "v2",
                "artifact_evidence_row_ref",
            ),
            (
                self.artifact_evidence_target_ref,
                "artifact-evidence-target",
                "v2",
                "artifact_evidence_target_ref",
            ),
            (
                self.content_contract_ref,
                "artifact-content-contract",
                "v2",
                "content_contract_ref",
            ),
            (
                self.render_contract_ref,
                "artifact-render-contract",
                "v2",
                "render_contract_ref",
            ),
        ):
            _require_ref_version(ref, expected_type, expected_version, field_name)
        if self.provider_payload_ref is not None:
            _require_ref_version(
                self.provider_payload_ref,
                "attachment-provider-payload",
                "v2",
                "provider_payload_ref",
            )
        if self.source_evidence_set_ref is not None:
            _require_ref_version(
                self.source_evidence_set_ref,
                "source-evidence-set",
                "v2",
                "source_evidence_set_ref",
            )
        if self.mode is ReconstructionMode.BLOCKED:
            raise ValueError("artifact build contract cannot use BLOCKED mode")
        for label, values in (
            (
                "required provider capability IDs",
                self.required_provider_capability_ids,
            ),
            ("required runtime tools", self.required_runtime_tools),
            ("validator IDs", self.validator_ids),
        ):
            _require_unique(label, values)
            if values != tuple(sorted(values)):
                raise ValueError(f"{label} must be sorted")
        return self


class ArtifactBuildSpecV2(ContractModelV2):
    schema_version: Literal["eval-factory/artifact-build-spec/v2"] = "eval-factory/artifact-build-spec/v2"
    artifact_build_spec_v2_id: Identifier
    attachment_planning_context_ref: ObjectRef
    artifact_evidence_matrix_ref: ObjectRef
    artifact_evidence_target_ref: ObjectRef
    artifact_build_contract_ref: ObjectRef
    artifact_routing_policy_ref: ObjectRef
    facade_route_request_ref: ObjectRef
    facade_route_decision_ref: ObjectRef
    selected_route_kind: ArtifactRouteKindV2
    selected_model_profile_ref: ObjectRef | None = None
    source_evidence_set_ref: ObjectRef | None = None
    build_spec: ArtifactBuildSpec
    policy_version: str = Field(min_length=1, max_length=128)
    artifact_build_spec_v2_sha256: Sha256
    audit: ContractAudit

    @field_validator("selected_route_kind", mode="before")
    @classmethod
    def parse_route_kind(cls, value: object) -> ArtifactRouteKindV2:
        if isinstance(value, ArtifactRouteKindV2):
            return value
        if isinstance(value, str):
            return ArtifactRouteKindV2(value)
        raise TypeError("selected_route_kind must be an ArtifactRouteKindV2")

    @model_validator(mode="after")
    def validate_spec(self) -> ArtifactBuildSpecV2:
        for ref, expected_type, expected_version, field_name in (
            (
                self.attachment_planning_context_ref,
                "attachment-planning-context",
                "v2",
                "attachment_planning_context_ref",
            ),
            (
                self.artifact_evidence_matrix_ref,
                "artifact-evidence-matrix",
                "v2",
                "artifact_evidence_matrix_ref",
            ),
            (
                self.artifact_evidence_target_ref,
                "artifact-evidence-target",
                "v2",
                "artifact_evidence_target_ref",
            ),
            (
                self.artifact_build_contract_ref,
                "artifact-build-contract",
                "v2",
                "artifact_build_contract_ref",
            ),
            (
                self.artifact_routing_policy_ref,
                "artifact-routing-policy",
                "v2",
                "artifact_routing_policy_ref",
            ),
            (
                self.facade_route_request_ref,
                "attachment-route-request",
                "v2",
                "facade_route_request_ref",
            ),
            (
                self.facade_route_decision_ref,
                "attachment-route-decision",
                "v2",
                "facade_route_decision_ref",
            ),
        ):
            _require_ref_version(ref, expected_type, expected_version, field_name)
        if self.source_evidence_set_ref is not None:
            _require_ref_version(
                self.source_evidence_set_ref,
                "source-evidence-set",
                "v2",
                "source_evidence_set_ref",
            )
        provider_route = self.selected_route_kind is ArtifactRouteKindV2.PROVIDER
        if provider_route:
            if (
                len(self.build_spec.provider_preference) != 1
                or self.build_spec.runtime_preference
                or self.selected_model_profile_ref is not None
            ):
                raise ValueError("provider route requires one provider and no runtime/model profile")
        else:
            if (
                self.build_spec.provider_preference
                or len(self.build_spec.runtime_preference) != 1
                or self.selected_model_profile_ref is None
            ):
                raise ValueError("runtime route requires one runtime and one model profile")
            _require_ref_type(
                self.selected_model_profile_ref,
                "model-profile",
                "selected_model_profile_ref",
            )
        return self


class ArtifactRoutePlanEntryV2(ContractModelV2):
    schema_version: Literal["eval-factory/artifact-route-plan-entry/v2"] = (
        "eval-factory/artifact-route-plan-entry/v2"
    )
    artifact_evidence_target_ref: ObjectRef
    attachment_dependency_id: Identifier
    artifact_id: Identifier
    criticality: AttachmentCriticality
    outcome: ArtifactRouteEntryOutcomeV2
    facade_route_request_ref: ObjectRef | None = None
    facade_route_decision: AttachmentRouteDecisionV2 | None = None
    build_spec: ArtifactBuildSpecV2 | None = None
    reasons: frozenset[ArtifactRoutingReasonV2] = frozenset()

    @field_validator("criticality", mode="before")
    @classmethod
    def parse_criticality(
        cls,
        value: object,
    ) -> AttachmentCriticality:
        if isinstance(value, AttachmentCriticality):
            return value
        if isinstance(value, str):
            return AttachmentCriticality(value)
        raise TypeError("criticality must be an AttachmentCriticality")

    @field_validator("outcome", mode="before")
    @classmethod
    def parse_outcome(cls, value: object) -> ArtifactRouteEntryOutcomeV2:
        if isinstance(value, ArtifactRouteEntryOutcomeV2):
            return value
        if isinstance(value, str):
            return ArtifactRouteEntryOutcomeV2(value)
        raise TypeError("outcome must be an ArtifactRouteEntryOutcomeV2")

    @field_validator("reasons", mode="before")
    @classmethod
    def parse_reasons(
        cls,
        value: object,
    ) -> frozenset[ArtifactRoutingReasonV2]:
        if not isinstance(value, (tuple, list, set, frozenset)):
            raise TypeError("reasons must be a collection")
        return frozenset(ArtifactRoutingReasonV2(item) for item in value)

    @model_validator(mode="after")
    def validate_entry(self) -> ArtifactRoutePlanEntryV2:
        _require_ref_version(
            self.artifact_evidence_target_ref,
            "artifact-evidence-target",
            "v2",
            "artifact_evidence_target_ref",
        )
        routed = self.outcome in {
            ArtifactRouteEntryOutcomeV2.ROUTED_PROVIDER,
            ArtifactRouteEntryOutcomeV2.ROUTED_RUNTIME,
        }
        if routed:
            if (
                self.facade_route_request_ref is None
                or self.facade_route_decision is None
                or self.build_spec is None
                or self.reasons
            ):
                raise ValueError("routed entry requires exact facade facts and one build spec")
            _require_ref_version(
                self.facade_route_request_ref,
                "attachment-route-request",
                "v2",
                "facade_route_request_ref",
            )
            expected_kind = (
                ArtifactRouteKindV2.PROVIDER
                if self.outcome is ArtifactRouteEntryOutcomeV2.ROUTED_PROVIDER
                else ArtifactRouteKindV2.RUNTIME
            )
            expected_outcome = (
                AttachmentRouteOutcome.SELECTED_PROVIDER
                if expected_kind is ArtifactRouteKindV2.PROVIDER
                else AttachmentRouteOutcome.SELECTED_RUNTIME
            )
            expected_candidate_kind = (
                AttachmentRouteCandidateKind.PROVIDER
                if expected_kind is ArtifactRouteKindV2.PROVIDER
                else AttachmentRouteCandidateKind.RUNTIME
            )
            if (
                self.build_spec.selected_route_kind is not expected_kind
                or self.facade_route_decision.outcome is not expected_outcome
                or self.facade_route_decision.selected_kind is not expected_candidate_kind
            ):
                raise ValueError("routed entry outcome does not match facade decision and build spec")
        else:
            if self.build_spec is not None or not self.reasons:
                raise ValueError("blocked entry requires reasons and cannot carry a build spec")
            if (self.facade_route_request_ref is None) != (self.facade_route_decision is None):
                raise ValueError("blocked entry facade request and decision must be present together")
            if self.facade_route_request_ref is not None:
                _require_ref_version(
                    self.facade_route_request_ref,
                    "attachment-route-request",
                    "v2",
                    "facade_route_request_ref",
                )
                assert self.facade_route_decision is not None
                if self.facade_route_decision.outcome is not AttachmentRouteOutcome.BLOCKED_CAPABILITY:
                    raise ValueError("blocked facade decision must be BLOCKED_CAPABILITY")
        return self


class ArtifactRoutingPlanV2(ContractModelV2):
    schema_version: Literal["eval-factory/artifact-routing-plan/v2"] = "eval-factory/artifact-routing-plan/v2"
    artifact_routing_plan_id: Identifier
    attachment_planning_context_ref: ObjectRef
    producer_task_view_ref: ObjectRef
    artifact_evidence_matrix_ref: ObjectRef
    artifact_routing_policy_ref: ObjectRef
    entries: tuple[ArtifactRoutePlanEntryV2, ...] = Field(min_length=1)
    aggregate_outcome: ArtifactRoutingAggregateOutcomeV2
    routed_artifact_ids: tuple[Identifier, ...]
    blocked_required_artifact_ids: tuple[Identifier, ...]
    blocked_optional_artifact_ids: tuple[Identifier, ...]
    policy_version: str = Field(min_length=1, max_length=128)
    artifact_routing_plan_sha256: Sha256
    audit: ContractAudit

    @field_validator("aggregate_outcome", mode="before")
    @classmethod
    def parse_aggregate_outcome(
        cls,
        value: object,
    ) -> ArtifactRoutingAggregateOutcomeV2:
        if isinstance(value, ArtifactRoutingAggregateOutcomeV2):
            return value
        if isinstance(value, str):
            return ArtifactRoutingAggregateOutcomeV2(value)
        raise TypeError("aggregate_outcome must be an ArtifactRoutingAggregateOutcomeV2")

    @model_validator(mode="after")
    def validate_plan(self) -> ArtifactRoutingPlanV2:
        for ref, expected_type, field_name in (
            (
                self.attachment_planning_context_ref,
                "attachment-planning-context",
                "attachment_planning_context_ref",
            ),
            (
                self.producer_task_view_ref,
                "producer-task-view",
                "producer_task_view_ref",
            ),
            (
                self.artifact_evidence_matrix_ref,
                "artifact-evidence-matrix",
                "artifact_evidence_matrix_ref",
            ),
            (
                self.artifact_routing_policy_ref,
                "artifact-routing-policy",
                "artifact_routing_policy_ref",
            ),
        ):
            _require_ref_version(ref, expected_type, "v2", field_name)
        artifact_ids = tuple(item.artifact_id for item in self.entries)
        _require_unique("routing plan artifact IDs", artifact_ids)
        expected_routed = tuple(
            sorted(
                item.artifact_id
                for item in self.entries
                if item.outcome
                in {
                    ArtifactRouteEntryOutcomeV2.ROUTED_PROVIDER,
                    ArtifactRouteEntryOutcomeV2.ROUTED_RUNTIME,
                }
            )
        )
        expected_required = tuple(
            sorted(
                item.artifact_id
                for item in self.entries
                if item.outcome
                in {
                    ArtifactRouteEntryOutcomeV2.BLOCKED_CAPABILITY,
                    ArtifactRouteEntryOutcomeV2.BLOCKED_POLICY,
                }
                and item.criticality is not AttachmentCriticality.OPTIONAL
            )
        )
        expected_optional = tuple(
            sorted(
                item.artifact_id
                for item in self.entries
                if item.outcome
                in {
                    ArtifactRouteEntryOutcomeV2.BLOCKED_CAPABILITY,
                    ArtifactRouteEntryOutcomeV2.BLOCKED_POLICY,
                }
                and item.criticality is AttachmentCriticality.OPTIONAL
            )
        )
        if (
            self.routed_artifact_ids != expected_routed
            or self.blocked_required_artifact_ids != expected_required
            or self.blocked_optional_artifact_ids != expected_optional
        ):
            raise ValueError("routing plan inventories must exactly classify entries")
        expected_outcome = _artifact_routing_aggregate_outcome(self.entries)
        if self.aggregate_outcome is not expected_outcome:
            raise ValueError("routing plan aggregate outcome does not match entries")
        return self


class ArtifactExecutionReceiptOutcomeV2(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    BLOCKED_CAPABILITY = "BLOCKED_CAPABILITY"
    BLOCKED_POLICY = "BLOCKED_POLICY"
    RETRYABLE_FAILURE = "RETRYABLE_FAILURE"
    TERMINAL_FAILURE = "TERMINAL_FAILURE"
    BLOCKED_DEPENDENCY = "BLOCKED_DEPENDENCY"


class ArtifactExecutionUnitV2(ContractModelV2):
    schema_version: Literal["eval-factory/artifact-execution-unit/v2"] = (
        "eval-factory/artifact-execution-unit/v2"
    )
    artifact_execution_unit_id: Identifier
    artifact_id: Identifier
    logical_path: RelativePath
    build_contract: ArtifactBuildContractV2
    build_spec: ArtifactBuildSpecV2
    facade_route_decision: AttachmentRouteDecisionV2
    dependency_artifact_ids: tuple[Identifier, ...]
    locked_fact_ids: tuple[Identifier, ...]
    order_index: int = Field(ge=0)
    policy_version: Literal["artifact-execution/r5-06-v1"] = "artifact-execution/r5-06-v1"
    artifact_execution_unit_sha256: Sha256

    @model_validator(mode="after")
    def validate_unit(self) -> ArtifactExecutionUnitV2:
        if self.artifact_id != self.build_contract.artifact_id:
            raise ValueError("execution unit artifact must match build contract")
        if (
            self.logical_path != self.build_contract.logical_path
            or self.logical_path != self.build_spec.build_spec.relative_path
        ):
            raise ValueError("execution unit path must match build contract and spec")
        if self.build_spec.artifact_build_contract_ref != artifact_build_contract_ref(self.build_contract):
            raise ValueError("execution unit build spec must bind its exact contract")
        observed_decision_ref = _object_ref_from_facade(
            attachment_route_decision_ref(self.facade_route_decision)
        )
        if self.build_spec.facade_route_decision_ref != observed_decision_ref:
            raise ValueError("execution unit must bind the exact route decision")
        selected_provider = self.build_spec.build_spec.provider_preference
        selected_runtime = self.build_spec.build_spec.runtime_preference
        if self.build_spec.selected_route_kind is ArtifactRouteKindV2.PROVIDER:
            if (
                self.facade_route_decision.selected_kind is not AttachmentRouteCandidateKind.PROVIDER
                or self.facade_route_decision.selected_id
                != (selected_provider[0] if selected_provider else None)
            ):
                raise ValueError("provider execution unit route is mismatched")
        elif (
            self.facade_route_decision.selected_kind is not AttachmentRouteCandidateKind.RUNTIME
            or self.facade_route_decision.selected_id != (selected_runtime[0] if selected_runtime else None)
        ):
            raise ValueError("runtime execution unit route is mismatched")
        _require_sorted_unique_values(
            "dependency artifact IDs",
            self.dependency_artifact_ids,
        )
        _require_sorted_unique_values("locked fact IDs", self.locked_fact_ids)
        if self.artifact_id in self.dependency_artifact_ids:
            raise ValueError("execution unit cannot depend on itself")
        return self


class ArtifactExecutionGroupV2(ContractModelV2):
    schema_version: Literal["eval-factory/artifact-execution-group/v2"] = (
        "eval-factory/artifact-execution-group/v2"
    )
    artifact_execution_group_id: Identifier
    units: tuple[ArtifactExecutionUnitV2, ...] = Field(min_length=1)
    locked_fact_ids: tuple[Identifier, ...]
    policy_version: Literal["artifact-execution/r5-06-v1"] = "artifact-execution/r5-06-v1"
    artifact_execution_group_sha256: Sha256

    @model_validator(mode="after")
    def validate_group(self) -> ArtifactExecutionGroupV2:
        artifact_ids = tuple(item.artifact_id for item in self.units)
        _require_unique("execution group artifact IDs", artifact_ids)
        if tuple(item.order_index for item in self.units) != tuple(range(len(self.units))):
            raise ValueError("execution group order indexes must be contiguous")
        expected_facts = tuple(sorted({fact_id for unit in self.units for fact_id in unit.locked_fact_ids}))
        if self.locked_fact_ids != expected_facts:
            raise ValueError("execution group locked facts must equal unit union")
        seen: set[str] = set()
        for unit in self.units:
            if not set(unit.dependency_artifact_ids) <= seen:
                raise ValueError("execution group dependencies must precede their unit")
            seen.add(unit.artifact_id)
        return self


class ArtifactExecutionPlanV2(ContractModelV2):
    schema_version: Literal["eval-factory/artifact-execution-plan/v2"] = (
        "eval-factory/artifact-execution-plan/v2"
    )
    artifact_execution_plan_id: Identifier
    artifact_routing_plan_ref: ObjectRef
    world_ledger_snapshot: WorldLedgerSnapshotV2
    groups: tuple[ArtifactExecutionGroupV2, ...] = Field(min_length=1)
    routed_artifact_ids: tuple[Identifier, ...] = Field(min_length=1)
    selected_provider_ids: tuple[Identifier, ...]
    selected_runtime_ids: tuple[Identifier, ...]
    policy_version: Literal["artifact-execution/r5-06-v1"] = "artifact-execution/r5-06-v1"
    artifact_execution_plan_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_plan(self) -> ArtifactExecutionPlanV2:
        _require_ref_version(
            self.artifact_routing_plan_ref,
            "artifact-routing-plan",
            "v2",
            "artifact_routing_plan_ref",
        )
        validate_world_ledger_snapshot_identity(self.world_ledger_snapshot)
        artifact_ids = tuple(unit.artifact_id for group in self.groups for unit in group.units)
        _require_unique("execution plan artifact IDs", artifact_ids)
        if self.routed_artifact_ids != tuple(sorted(artifact_ids)):
            raise ValueError("execution plan routed inventory must equal group units")
        providers = tuple(
            sorted(
                {
                    unit.facade_route_decision.selected_id
                    for group in self.groups
                    for unit in group.units
                    if unit.build_spec.selected_route_kind is ArtifactRouteKindV2.PROVIDER
                    and unit.facade_route_decision.selected_id is not None
                }
            )
        )
        runtimes = tuple(
            sorted(
                {
                    unit.facade_route_decision.selected_id
                    for group in self.groups
                    for unit in group.units
                    if unit.build_spec.selected_route_kind is ArtifactRouteKindV2.RUNTIME
                    and unit.facade_route_decision.selected_id is not None
                }
            )
        )
        if self.selected_provider_ids != providers or self.selected_runtime_ids != runtimes:
            raise ValueError("execution plan route inventories are not exact")
        return self


class ArtifactExecutionReceiptV2(ContractModelV2):
    schema_version: Literal["eval-factory/artifact-execution-receipt/v2"] = (
        "eval-factory/artifact-execution-receipt/v2"
    )
    artifact_execution_receipt_id: Identifier
    artifact_execution_plan_ref: ObjectRef
    artifact_execution_group_ref: ObjectRef
    artifact_execution_unit_ref: ObjectRef
    artifact_id: Identifier
    attempt: int = Field(ge=1)
    outcome: ArtifactExecutionReceiptOutcomeV2
    facade_request: AttachmentExecutionRequestV2 | None = None
    facade_result: AttachmentExecutionResultV2 | None = None
    failed_dependency_receipt_refs: tuple[ObjectRef, ...] = ()
    retry_of_receipt_ref: ObjectRef | None = None
    policy_version: Literal["artifact-execution/r5-06-v1"] = "artifact-execution/r5-06-v1"
    artifact_execution_receipt_sha256: Sha256
    audit: ContractAudit

    @field_validator("outcome", mode="before")
    @classmethod
    def parse_outcome(
        cls,
        value: object,
    ) -> ArtifactExecutionReceiptOutcomeV2:
        if isinstance(value, ArtifactExecutionReceiptOutcomeV2):
            return value
        if isinstance(value, str):
            return ArtifactExecutionReceiptOutcomeV2(value)
        raise TypeError("outcome must be an ArtifactExecutionReceiptOutcomeV2")

    @model_validator(mode="after")
    def validate_receipt(self) -> ArtifactExecutionReceiptV2:
        for ref, expected_type, field_name in (
            (
                self.artifact_execution_plan_ref,
                "artifact-execution-plan",
                "artifact_execution_plan_ref",
            ),
            (
                self.artifact_execution_group_ref,
                "artifact-execution-group",
                "artifact_execution_group_ref",
            ),
            (
                self.artifact_execution_unit_ref,
                "artifact-execution-unit",
                "artifact_execution_unit_ref",
            ),
        ):
            _require_ref_version(ref, expected_type, "v2", field_name)
        _validate_receipt_refs(self.failed_dependency_receipt_refs)
        if self.retry_of_receipt_ref is not None:
            _require_ref_version(
                self.retry_of_receipt_ref,
                "artifact-execution-receipt",
                "v2",
                "retry_of_receipt_ref",
            )
        if self.outcome is ArtifactExecutionReceiptOutcomeV2.BLOCKED_DEPENDENCY:
            if (
                self.facade_request is not None
                or self.facade_result is not None
                or not self.failed_dependency_receipt_refs
            ):
                raise ValueError("dependency block requires failed dependency refs and no facade call")
            return self
        if self.facade_request is None or self.facade_result is None or self.failed_dependency_receipt_refs:
            raise ValueError("attempted receipt requires facade request/result and no dependency block")
        validate_attachment_execution_request_identity(self.facade_request)
        validate_attachment_execution_result_identity(self.facade_result)
        expected_result_request_ref = _object_ref_from_facade(
            attachment_execution_request_ref(self.facade_request)
        )
        observed_result_request_ref = _object_ref_from_facade(self.facade_result.execution_request_ref)
        if expected_result_request_ref != observed_result_request_ref:
            raise ValueError("facade result must bind the exact execution request")
        if (
            self.artifact_id != self.facade_request.artifact_id
            or self.artifact_id != self.facade_result.artifact_id
            or self.attempt != self.facade_request.attempt
            or self.attempt != self.facade_result.attempt
        ):
            raise ValueError("receipt artifact/attempt must match facade facts")
        expected_outcome = ArtifactExecutionReceiptOutcomeV2(self.facade_result.status.value)
        if self.outcome is not expected_outcome:
            raise ValueError("receipt outcome must match facade result")
        return self


class ArtifactExecutionBatchV2(ContractModelV2):
    schema_version: Literal["eval-factory/artifact-execution-batch/v2"] = (
        "eval-factory/artifact-execution-batch/v2"
    )
    artifact_execution_batch_id: Identifier
    artifact_execution_plan_ref: ObjectRef
    prior_batch_ref: ObjectRef | None = None
    receipts: tuple[ArtifactExecutionReceiptV2, ...] = Field(min_length=1)
    succeeded_artifact_ids: tuple[Identifier, ...]
    retryable_artifact_ids: tuple[Identifier, ...]
    terminal_artifact_ids: tuple[Identifier, ...]
    dependency_blocked_artifact_ids: tuple[Identifier, ...]
    world_ledger_snapshot_ref: ObjectRef
    policy_version: Literal["artifact-execution/r5-06-v1"] = "artifact-execution/r5-06-v1"
    artifact_execution_batch_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_batch(self) -> ArtifactExecutionBatchV2:
        _require_ref_version(
            self.artifact_execution_plan_ref,
            "artifact-execution-plan",
            "v2",
            "artifact_execution_plan_ref",
        )
        if self.prior_batch_ref is not None:
            _require_ref_version(
                self.prior_batch_ref,
                "artifact-execution-batch",
                "v2",
                "prior_batch_ref",
            )
        _require_ref_version(
            self.world_ledger_snapshot_ref,
            "world-ledger-snapshot",
            "v2",
            "world_ledger_snapshot_ref",
        )
        artifact_ids = tuple(item.artifact_id for item in self.receipts)
        _require_unique("execution batch artifact IDs", artifact_ids)
        expected_succeeded = _receipt_artifact_ids(
            self.receipts,
            {ArtifactExecutionReceiptOutcomeV2.SUCCEEDED},
        )
        expected_retryable = _receipt_artifact_ids(
            self.receipts,
            {ArtifactExecutionReceiptOutcomeV2.RETRYABLE_FAILURE},
        )
        expected_terminal = _receipt_artifact_ids(
            self.receipts,
            {
                ArtifactExecutionReceiptOutcomeV2.BLOCKED_CAPABILITY,
                ArtifactExecutionReceiptOutcomeV2.BLOCKED_POLICY,
                ArtifactExecutionReceiptOutcomeV2.TERMINAL_FAILURE,
            },
        )
        expected_dependency = _receipt_artifact_ids(
            self.receipts,
            {ArtifactExecutionReceiptOutcomeV2.BLOCKED_DEPENDENCY},
        )
        if (
            self.succeeded_artifact_ids != expected_succeeded
            or self.retryable_artifact_ids != expected_retryable
            or self.terminal_artifact_ids != expected_terminal
            or self.dependency_blocked_artifact_ids != expected_dependency
        ):
            raise ValueError("execution batch inventories must classify receipts")
        return self


class ArtifactBuildResultOutcomeV2(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    BLOCKED_CAPABILITY = "BLOCKED_CAPABILITY"
    BLOCKED_POLICY = "BLOCKED_POLICY"
    RETRYABLE_FAILURE = "RETRYABLE_FAILURE"
    TERMINAL_FAILURE = "TERMINAL_FAILURE"
    BLOCKED_DEPENDENCY = "BLOCKED_DEPENDENCY"


class ArtifactBuildResultProjectionGapV2(StrEnum):
    BUILD_SPEC_UNAVAILABLE = "BUILD_SPEC_UNAVAILABLE"
    DEPENDENCY_NOT_ATTEMPTED = "DEPENDENCY_NOT_ATTEMPTED"
    WORKER_VERSION_UNAVAILABLE = "WORKER_VERSION_UNAVAILABLE"
    WORKER_VERSION_UNREPRESENTABLE = "WORKER_VERSION_UNREPRESENTABLE"


class AttachmentReconstructionOutcomeV2(StrEnum):
    NOT_REQUIRED = "NOT_REQUIRED"
    PENDING_VALIDATION = "PENDING_VALIDATION"
    PARTIAL_FAILURE = "PARTIAL_FAILURE"
    RETRYABLE_FAILURE = "RETRYABLE_FAILURE"
    BLOCKED = "BLOCKED"


class ArtifactBuildResultV2(ContractModelV2):
    schema_version: Literal["eval-factory/artifact-build-result/v2"] = "eval-factory/artifact-build-result/v2"
    artifact_build_result_v2_id: Identifier
    artifact_routing_plan_ref: ObjectRef
    route_entry: ArtifactRoutePlanEntryV2
    artifact_execution_plan_ref: ObjectRef | None = None
    execution_receipt: ArtifactExecutionReceiptV2 | None = None
    outcome: ArtifactBuildResultOutcomeV2
    retryable: bool
    frozen_result: ArtifactBuildResult | None = None
    frozen_projection_gap: ArtifactBuildResultProjectionGapV2 | None = None
    policy_version: Literal["artifact-results/r5-07-v1"] = "artifact-results/r5-07-v1"
    artifact_build_result_v2_sha256: Sha256
    audit: ContractAudit

    @field_validator("outcome", mode="before")
    @classmethod
    def parse_outcome(
        cls,
        value: object,
    ) -> ArtifactBuildResultOutcomeV2:
        if isinstance(value, ArtifactBuildResultOutcomeV2):
            return value
        if isinstance(value, str):
            return ArtifactBuildResultOutcomeV2(value)
        raise TypeError("outcome must be an ArtifactBuildResultOutcomeV2")

    @field_validator("frozen_projection_gap", mode="before")
    @classmethod
    def parse_projection_gap(
        cls,
        value: object,
    ) -> ArtifactBuildResultProjectionGapV2 | None:
        if value is None or isinstance(value, ArtifactBuildResultProjectionGapV2):
            return value
        if isinstance(value, str):
            return ArtifactBuildResultProjectionGapV2(value)
        raise TypeError("frozen_projection_gap must be an ArtifactBuildResultProjectionGapV2")

    @model_validator(mode="after")
    def validate_result(self) -> ArtifactBuildResultV2:
        _require_ref_version(
            self.artifact_routing_plan_ref,
            "artifact-routing-plan",
            "v2",
            "artifact_routing_plan_ref",
        )
        routed = self.route_entry.outcome in {
            ArtifactRouteEntryOutcomeV2.ROUTED_PROVIDER,
            ArtifactRouteEntryOutcomeV2.ROUTED_RUNTIME,
        }
        if routed:
            if self.artifact_execution_plan_ref is None or self.execution_receipt is None:
                raise ValueError("routed artifact result requires execution plan and receipt")
            _require_ref_version(
                self.artifact_execution_plan_ref,
                "artifact-execution-plan",
                "v2",
                "artifact_execution_plan_ref",
            )
            if (
                self.execution_receipt.artifact_execution_plan_ref != self.artifact_execution_plan_ref
                or self.execution_receipt.artifact_id != self.route_entry.artifact_id
            ):
                raise ValueError("artifact result receipt must bind the exact plan and artifact")
            expected_outcome = ArtifactBuildResultOutcomeV2(self.execution_receipt.outcome.value)
            _validate_result_receipt_route(self.route_entry, self.execution_receipt)
        else:
            if self.artifact_execution_plan_ref is not None or self.execution_receipt is not None:
                raise ValueError("routing-blocked artifact result cannot carry execution facts")
            expected_outcome = ArtifactBuildResultOutcomeV2(self.route_entry.outcome.value)
        if self.outcome is not expected_outcome:
            raise ValueError("artifact result outcome does not match routing or execution truth")
        if self.retryable is not (self.outcome is ArtifactBuildResultOutcomeV2.RETRYABLE_FAILURE):
            raise ValueError("artifact result retryable flag must match direct retry outcome")

        expected_gap = artifact_build_result_projection_gap_v2(
            route_entry=self.route_entry,
            receipt=self.execution_receipt,
        )
        if expected_gap is None:
            if self.frozen_result is None or self.frozen_projection_gap is not None:
                raise ValueError("representable artifact result requires frozen result and no projection gap")
            _validate_frozen_artifact_result(self)
        elif self.frozen_result is not None or self.frozen_projection_gap is not expected_gap:
            raise ValueError("artifact result frozen projection gap is not exact")

        expected_audit_refs = [self.artifact_routing_plan_ref]
        if self.artifact_execution_plan_ref is not None:
            expected_audit_refs.append(self.artifact_execution_plan_ref)
        if self.execution_receipt is not None:
            expected_audit_refs.append(artifact_execution_receipt_ref(self.execution_receipt))
        _validate_artifact_result_audit(
            self.audit,
            tuple(expected_audit_refs),
        )
        return self


class AttachmentReconstructionResultV2(ContractModelV2):
    schema_version: Literal["eval-factory/attachment-reconstruction-result/v2"] = (
        "eval-factory/attachment-reconstruction-result/v2"
    )
    attachment_reconstruction_result_v2_id: Identifier
    routing_request_ref: ObjectRef
    routing_compilation_result_ref: ObjectRef
    attachment_planning_context_ref: ObjectRef
    producer_task_view_ref: ObjectRef
    evidence_matrix_ref: ObjectRef | None
    artifact_routing_plan_ref: ObjectRef | None
    artifact_execution_plan_ref: ObjectRef | None
    artifact_execution_batch_ref: ObjectRef | None
    artifact_results: tuple[ArtifactBuildResultV2, ...]
    artifact_result_refs: tuple[ObjectRef, ...]
    outcome: AttachmentReconstructionOutcomeV2
    candidate_output_refs: tuple[ObjectRef, ...]
    accepted_artifact_refs: tuple[ObjectRef, ...]
    failed_artifact_ids: tuple[Identifier, ...]
    resumable_artifact_ids: tuple[Identifier, ...]
    dependency_blocked_artifact_ids: tuple[Identifier, ...]
    required_incomplete_artifact_ids: tuple[Identifier, ...]
    optional_incomplete_artifact_ids: tuple[Identifier, ...]
    frozen_result: AttachmentReconstructionResult | None
    frozen_projection_gap: Literal["PRE_VALIDATION"]
    environment_spec_ref: None
    provenance_manifest_ref: None
    quality_report_ref: None
    package_sha256: None
    input_state_only: None
    policy_version: Literal["artifact-results/r5-07-v1"] = "artifact-results/r5-07-v1"
    attachment_reconstruction_result_v2_sha256: Sha256
    audit: ContractAudit

    @field_validator("outcome", mode="before")
    @classmethod
    def parse_outcome(
        cls,
        value: object,
    ) -> AttachmentReconstructionOutcomeV2:
        if isinstance(value, AttachmentReconstructionOutcomeV2):
            return value
        if isinstance(value, str):
            return AttachmentReconstructionOutcomeV2(value)
        raise TypeError("outcome must be an AttachmentReconstructionOutcomeV2")

    @model_validator(mode="after")
    def validate_result(self) -> AttachmentReconstructionResultV2:
        for ref, expected_type, expected_version, field_name in (
            (
                self.routing_request_ref,
                "artifact-routing-request",
                "r5-05",
                "routing_request_ref",
            ),
            (
                self.routing_compilation_result_ref,
                "artifact-routing-compilation-result",
                "r5-05",
                "routing_compilation_result_ref",
            ),
            (
                self.attachment_planning_context_ref,
                "attachment-planning-context",
                "v2",
                "attachment_planning_context_ref",
            ),
            (
                self.producer_task_view_ref,
                "producer-task-view",
                "v2",
                "producer_task_view_ref",
            ),
        ):
            _require_ref_version(ref, expected_type, expected_version, field_name)
        if self.evidence_matrix_ref is not None:
            _require_ref_version(
                self.evidence_matrix_ref,
                "artifact-evidence-matrix",
                "v2",
                "evidence_matrix_ref",
            )
        for optional_ref, expected_type, field_name in (
            (
                self.artifact_routing_plan_ref,
                "artifact-routing-plan",
                "artifact_routing_plan_ref",
            ),
            (
                self.artifact_execution_plan_ref,
                "artifact-execution-plan",
                "artifact_execution_plan_ref",
            ),
            (
                self.artifact_execution_batch_ref,
                "artifact-execution-batch",
                "artifact_execution_batch_ref",
            ),
        ):
            if optional_ref is not None:
                _require_ref_version(
                    optional_ref,
                    expected_type,
                    "v2",
                    field_name,
                )

        artifact_ids = tuple(item.route_entry.artifact_id for item in self.artifact_results)
        _require_sorted_unique_values("artifact result IDs", artifact_ids)
        expected_result_refs = tuple(artifact_build_result_v2_ref(item) for item in self.artifact_results)
        if self.artifact_result_refs != expected_result_refs:
            raise ValueError("artifact result refs must exactly match nested results")
        if self.artifact_routing_plan_ref is None:
            if (
                self.evidence_matrix_ref is not None
                or self.artifact_execution_plan_ref is not None
                or self.artifact_execution_batch_ref is not None
                or self.artifact_results
                or self.outcome is not AttachmentReconstructionOutcomeV2.NOT_REQUIRED
            ):
                raise ValueError("NOT_REQUIRED result cannot contain artifact work")
        else:
            if not self.artifact_results:
                raise ValueError("routing result requires complete artifact results")
            if self.evidence_matrix_ref is None:
                raise ValueError("routing result requires evidence matrix")
            if any(
                item.artifact_routing_plan_ref != self.artifact_routing_plan_ref
                for item in self.artifact_results
            ):
                raise ValueError("artifact results must bind the aggregate routing plan")
            has_routed = any(
                item.route_entry.outcome
                in {
                    ArtifactRouteEntryOutcomeV2.ROUTED_PROVIDER,
                    ArtifactRouteEntryOutcomeV2.ROUTED_RUNTIME,
                }
                for item in self.artifact_results
            )
            if has_routed != (
                self.artifact_execution_plan_ref is not None and self.artifact_execution_batch_ref is not None
            ):
                raise ValueError("routed aggregate requires execution plan and batch together")
            if has_routed and any(
                item.route_entry.outcome
                in {
                    ArtifactRouteEntryOutcomeV2.ROUTED_PROVIDER,
                    ArtifactRouteEntryOutcomeV2.ROUTED_RUNTIME,
                }
                and item.artifact_execution_plan_ref != self.artifact_execution_plan_ref
                for item in self.artifact_results
            ):
                raise ValueError("routed artifact results must bind aggregate execution plan")

        expected_candidates = _candidate_output_refs(self.artifact_results)
        if self.candidate_output_refs != expected_candidates:
            raise ValueError("candidate output refs must equal successful execution outputs")
        if self.accepted_artifact_refs:
            raise ValueError("accepted artifact refs remain empty before validation")

        expected_failed = _artifact_result_ids(
            self.artifact_results,
            {
                ArtifactBuildResultOutcomeV2.BLOCKED_CAPABILITY,
                ArtifactBuildResultOutcomeV2.BLOCKED_POLICY,
                ArtifactBuildResultOutcomeV2.RETRYABLE_FAILURE,
                ArtifactBuildResultOutcomeV2.TERMINAL_FAILURE,
            },
        )
        expected_resumable = _artifact_result_ids(
            self.artifact_results,
            {ArtifactBuildResultOutcomeV2.RETRYABLE_FAILURE},
        )
        expected_dependency = _artifact_result_ids(
            self.artifact_results,
            {ArtifactBuildResultOutcomeV2.BLOCKED_DEPENDENCY},
        )
        expected_required = tuple(
            item.route_entry.artifact_id
            for item in self.artifact_results
            if item.outcome is not ArtifactBuildResultOutcomeV2.SUCCEEDED
            and item.route_entry.criticality is not AttachmentCriticality.OPTIONAL
        )
        expected_optional = tuple(
            item.route_entry.artifact_id
            for item in self.artifact_results
            if item.outcome is not ArtifactBuildResultOutcomeV2.SUCCEEDED
            and item.route_entry.criticality is AttachmentCriticality.OPTIONAL
        )
        if self.failed_artifact_ids != expected_failed:
            raise ValueError("failed artifact IDs must equal direct failure roots")
        if self.resumable_artifact_ids != expected_resumable:
            raise ValueError("resumable artifact IDs must equal direct retry roots")
        if self.dependency_blocked_artifact_ids != expected_dependency:
            raise ValueError("dependency-blocked artifact IDs are not exact")
        if (
            self.required_incomplete_artifact_ids != expected_required
            or self.optional_incomplete_artifact_ids != expected_optional
        ):
            raise ValueError("required and optional incomplete artifact IDs are not exact")

        expected_outcome = attachment_reconstruction_outcome_v2(
            self.artifact_results,
            has_routing_plan=self.artifact_routing_plan_ref is not None,
        )
        if self.outcome is not expected_outcome:
            raise ValueError("attachment reconstruction outcome is not exact")
        if self.frozen_result is not None:
            raise ValueError("frozen aggregate result is unavailable before validation")

        expected_audit_refs = [
            self.routing_request_ref,
            self.routing_compilation_result_ref,
            *self.artifact_result_refs,
        ]
        for optional_ref in (
            self.artifact_routing_plan_ref,
            self.artifact_execution_plan_ref,
            self.artifact_execution_batch_ref,
        ):
            if optional_ref is not None:
                expected_audit_refs.append(optional_ref)
        _validate_artifact_result_audit(
            self.audit,
            tuple(expected_audit_refs),
        )
        return self


def artifact_routing_policy_carried_sha256(
    policy: ArtifactRoutingPolicyV2,
) -> str:
    return _payload_sha256(
        policy.model_dump(
            mode="json",
            exclude={
                "artifact_routing_policy_id",
                "artifact_routing_policy_sha256",
                "audit",
            },
            exclude_none=False,
        )
    )


def artifact_routing_policy_ref(
    policy: ArtifactRoutingPolicyV2,
) -> ObjectRef:
    return ObjectRef(
        object_type="artifact-routing-policy",
        object_id=policy.artifact_routing_policy_id,
        object_version="v2",
        object_sha256=policy.artifact_routing_policy_sha256,
    )


def artifact_build_contract_carried_sha256(
    contract: ArtifactBuildContractV2,
) -> str:
    return _payload_sha256(
        contract.model_dump(
            mode="json",
            exclude={
                "artifact_build_contract_id",
                "artifact_build_contract_sha256",
                "audit",
            },
            exclude_none=False,
        )
    )


def artifact_build_contract_ref(
    contract: ArtifactBuildContractV2,
) -> ObjectRef:
    return ObjectRef(
        object_type="artifact-build-contract",
        object_id=contract.artifact_build_contract_id,
        object_version="v2",
        object_sha256=contract.artifact_build_contract_sha256,
    )


def artifact_build_spec_v2_carried_sha256(
    spec: ArtifactBuildSpecV2,
) -> str:
    return _payload_sha256(
        spec.model_dump(
            mode="json",
            exclude={
                "artifact_build_spec_v2_id",
                "artifact_build_spec_v2_sha256",
                "audit",
                "build_spec",
            },
            exclude_none=False,
        )
        | {
            "build_spec": spec.build_spec.model_dump(
                mode="json",
                exclude={"audit"},
                exclude_none=False,
            )
        }
    )


def artifact_build_spec_v2_ref(
    spec: ArtifactBuildSpecV2,
) -> ObjectRef:
    return ObjectRef(
        object_type="artifact-build-spec",
        object_id=spec.artifact_build_spec_v2_id,
        object_version="v2",
        object_sha256=spec.artifact_build_spec_v2_sha256,
    )


def artifact_route_plan_entry_carried_sha256(
    entry: ArtifactRoutePlanEntryV2,
) -> str:
    return _payload_sha256(_artifact_route_plan_entry_payload(entry))


def artifact_routing_plan_carried_sha256(
    plan: ArtifactRoutingPlanV2,
) -> str:
    return _payload_sha256(
        {
            "attachment_planning_context_ref": _ref_payload(plan.attachment_planning_context_ref),
            "producer_task_view_ref": _ref_payload(plan.producer_task_view_ref),
            "artifact_evidence_matrix_ref": _ref_payload(plan.artifact_evidence_matrix_ref),
            "artifact_routing_policy_ref": _ref_payload(plan.artifact_routing_policy_ref),
            "entries": [_artifact_route_plan_entry_payload(entry) for entry in plan.entries],
            "aggregate_outcome": plan.aggregate_outcome.value,
            "routed_artifact_ids": list(plan.routed_artifact_ids),
            "blocked_required_artifact_ids": list(plan.blocked_required_artifact_ids),
            "blocked_optional_artifact_ids": list(plan.blocked_optional_artifact_ids),
            "policy_version": plan.policy_version,
        }
    )


def artifact_routing_plan_ref(
    plan: ArtifactRoutingPlanV2,
) -> ObjectRef:
    return ObjectRef(
        object_type="artifact-routing-plan",
        object_id=plan.artifact_routing_plan_id,
        object_version="v2",
        object_sha256=plan.artifact_routing_plan_sha256,
    )


def artifact_execution_unit_carried_sha256(
    unit: ArtifactExecutionUnitV2,
) -> str:
    return _payload_sha256(
        {
            "artifact_id": unit.artifact_id,
            "logical_path": unit.logical_path,
            "build_contract_ref": _ref_payload(artifact_build_contract_ref(unit.build_contract)),
            "build_spec_ref": _ref_payload(artifact_build_spec_v2_ref(unit.build_spec)),
            "facade_route_decision_ref": _ref_payload(
                _object_ref_from_facade(attachment_route_decision_ref(unit.facade_route_decision))
            ),
            "dependency_artifact_ids": list(unit.dependency_artifact_ids),
            "locked_fact_ids": list(unit.locked_fact_ids),
            "order_index": unit.order_index,
            "policy_version": unit.policy_version,
        }
    )


def artifact_execution_unit_ref(
    unit: ArtifactExecutionUnitV2,
) -> ObjectRef:
    return ObjectRef(
        object_type="artifact-execution-unit",
        object_id=unit.artifact_execution_unit_id,
        object_version="v2",
        object_sha256=unit.artifact_execution_unit_sha256,
    )


def artifact_execution_group_carried_sha256(
    group: ArtifactExecutionGroupV2,
) -> str:
    return _payload_sha256(
        {
            "unit_refs": [_ref_payload(artifact_execution_unit_ref(unit)) for unit in group.units],
            "locked_fact_ids": list(group.locked_fact_ids),
            "policy_version": group.policy_version,
        }
    )


def artifact_execution_group_ref(
    group: ArtifactExecutionGroupV2,
) -> ObjectRef:
    return ObjectRef(
        object_type="artifact-execution-group",
        object_id=group.artifact_execution_group_id,
        object_version="v2",
        object_sha256=group.artifact_execution_group_sha256,
    )


def artifact_execution_plan_carried_sha256(
    plan: ArtifactExecutionPlanV2,
) -> str:
    return _payload_sha256(
        {
            "artifact_routing_plan_ref": _ref_payload(plan.artifact_routing_plan_ref),
            "world_ledger_snapshot_ref": _ref_payload(
                _object_ref_from_facade(world_ledger_snapshot_ref(plan.world_ledger_snapshot))
            ),
            "group_refs": [_ref_payload(artifact_execution_group_ref(group)) for group in plan.groups],
            "routed_artifact_ids": list(plan.routed_artifact_ids),
            "selected_provider_ids": list(plan.selected_provider_ids),
            "selected_runtime_ids": list(plan.selected_runtime_ids),
            "policy_version": plan.policy_version,
        }
    )


def artifact_execution_plan_ref(
    plan: ArtifactExecutionPlanV2,
) -> ObjectRef:
    return ObjectRef(
        object_type="artifact-execution-plan",
        object_id=plan.artifact_execution_plan_id,
        object_version="v2",
        object_sha256=plan.artifact_execution_plan_sha256,
    )


def artifact_execution_receipt_carried_sha256(
    receipt: ArtifactExecutionReceiptV2,
) -> str:
    return _payload_sha256(
        {
            "artifact_execution_plan_ref": _ref_payload(receipt.artifact_execution_plan_ref),
            "artifact_execution_group_ref": _ref_payload(receipt.artifact_execution_group_ref),
            "artifact_execution_unit_ref": _ref_payload(receipt.artifact_execution_unit_ref),
            "artifact_id": receipt.artifact_id,
            "attempt": receipt.attempt,
            "outcome": receipt.outcome.value,
            "facade_request_ref": (
                _ref_payload(
                    _object_ref_from_facade(attachment_execution_request_ref(receipt.facade_request))
                )
                if receipt.facade_request is not None
                else None
            ),
            "facade_result_ref": (
                _ref_payload(_object_ref_from_facade(attachment_execution_result_ref(receipt.facade_result)))
                if receipt.facade_result is not None
                else None
            ),
            "failed_dependency_receipt_refs": [
                _ref_payload(ref) for ref in receipt.failed_dependency_receipt_refs
            ],
            "retry_of_receipt_ref": (
                _ref_payload(receipt.retry_of_receipt_ref)
                if receipt.retry_of_receipt_ref is not None
                else None
            ),
            "policy_version": receipt.policy_version,
        }
    )


def artifact_execution_receipt_ref(
    receipt: ArtifactExecutionReceiptV2,
) -> ObjectRef:
    return ObjectRef(
        object_type="artifact-execution-receipt",
        object_id=receipt.artifact_execution_receipt_id,
        object_version="v2",
        object_sha256=receipt.artifact_execution_receipt_sha256,
    )


def artifact_execution_batch_carried_sha256(
    batch: ArtifactExecutionBatchV2,
) -> str:
    return _payload_sha256(
        {
            "artifact_execution_plan_ref": _ref_payload(batch.artifact_execution_plan_ref),
            "prior_batch_ref": (
                _ref_payload(batch.prior_batch_ref) if batch.prior_batch_ref is not None else None
            ),
            "receipt_refs": [
                _ref_payload(artifact_execution_receipt_ref(receipt)) for receipt in batch.receipts
            ],
            "succeeded_artifact_ids": list(batch.succeeded_artifact_ids),
            "retryable_artifact_ids": list(batch.retryable_artifact_ids),
            "terminal_artifact_ids": list(batch.terminal_artifact_ids),
            "dependency_blocked_artifact_ids": list(batch.dependency_blocked_artifact_ids),
            "world_ledger_snapshot_ref": _ref_payload(batch.world_ledger_snapshot_ref),
            "policy_version": batch.policy_version,
        }
    )


def artifact_execution_batch_ref(
    batch: ArtifactExecutionBatchV2,
) -> ObjectRef:
    return ObjectRef(
        object_type="artifact-execution-batch",
        object_id=batch.artifact_execution_batch_id,
        object_version="v2",
        object_sha256=batch.artifact_execution_batch_sha256,
    )


def artifact_build_result_v1_carried_sha256(
    result: ArtifactBuildResult,
) -> str:
    return _payload_sha256(_frozen_artifact_result_payload(result))


def artifact_execution_failure_record_v2(
    outcome: ArtifactBuildResultOutcomeV2,
    failure_code: AttachmentExecutionFailureCodeV2,
) -> FailureRecord:
    if outcome is ArtifactBuildResultOutcomeV2.BLOCKED_CAPABILITY:
        failure_class = FailureClass.CAPABILITY
        message = "Artifact execution capability was unavailable."
    elif outcome is ArtifactBuildResultOutcomeV2.BLOCKED_POLICY:
        failure_class = FailureClass.POLICY
        message = "Artifact execution was blocked by policy."
    elif outcome in {
        ArtifactBuildResultOutcomeV2.RETRYABLE_FAILURE,
        ArtifactBuildResultOutcomeV2.TERMINAL_FAILURE,
    }:
        failure_class = FailureClass.ENVIRONMENT
        message = "Artifact execution did not complete."
    else:
        raise ValueError("successful or dependency-blocked result cannot have execution failure")
    return FailureRecord(
        failure_class=failure_class,
        code=failure_code.value,
        message=message,
        retryable=(outcome is ArtifactBuildResultOutcomeV2.RETRYABLE_FAILURE),
        evidence_refs=(),
        detail=(),
    )


def artifact_build_result_v2_carried_sha256(
    result: ArtifactBuildResultV2,
) -> str:
    return _payload_sha256(
        {
            "artifact_routing_plan_ref": _ref_payload(result.artifact_routing_plan_ref),
            "route_entry": _artifact_route_plan_entry_payload(result.route_entry),
            "artifact_execution_plan_ref": (
                _ref_payload(result.artifact_execution_plan_ref)
                if result.artifact_execution_plan_ref is not None
                else None
            ),
            "execution_receipt_ref": (
                _ref_payload(artifact_execution_receipt_ref(result.execution_receipt))
                if result.execution_receipt is not None
                else None
            ),
            "outcome": result.outcome.value,
            "retryable": result.retryable,
            "frozen_result": (
                _frozen_artifact_result_payload(result.frozen_result)
                if result.frozen_result is not None
                else None
            ),
            "frozen_projection_gap": (
                result.frozen_projection_gap.value if result.frozen_projection_gap is not None else None
            ),
            "policy_version": result.policy_version,
        }
    )


def artifact_build_result_v2_ref(
    result: ArtifactBuildResultV2,
) -> ObjectRef:
    return ObjectRef(
        object_type="artifact-build-result",
        object_id=result.artifact_build_result_v2_id,
        object_version="v2",
        object_sha256=result.artifact_build_result_v2_sha256,
    )


def validate_artifact_build_result_v2_identity(
    result: ArtifactBuildResultV2,
) -> None:
    digest = artifact_build_result_v2_carried_sha256(result)
    if (
        result.artifact_build_result_v2_sha256 != digest
        or result.artifact_build_result_v2_id != f"artifact-build-result://sha256/{digest}"
    ):
        raise ValueError("artifact build result v2 identity is stale or invalid")


def attachment_reconstruction_result_v2_carried_sha256(
    result: AttachmentReconstructionResultV2,
) -> str:
    return _payload_sha256(
        {
            "routing_request_ref": _ref_payload(result.routing_request_ref),
            "routing_compilation_result_ref": _ref_payload(result.routing_compilation_result_ref),
            "attachment_planning_context_ref": _ref_payload(result.attachment_planning_context_ref),
            "producer_task_view_ref": _ref_payload(result.producer_task_view_ref),
            "evidence_matrix_ref": (
                _ref_payload(result.evidence_matrix_ref) if result.evidence_matrix_ref is not None else None
            ),
            "artifact_routing_plan_ref": (
                _ref_payload(result.artifact_routing_plan_ref)
                if result.artifact_routing_plan_ref is not None
                else None
            ),
            "artifact_execution_plan_ref": (
                _ref_payload(result.artifact_execution_plan_ref)
                if result.artifact_execution_plan_ref is not None
                else None
            ),
            "artifact_execution_batch_ref": (
                _ref_payload(result.artifact_execution_batch_ref)
                if result.artifact_execution_batch_ref is not None
                else None
            ),
            "artifact_result_refs": [_ref_payload(ref) for ref in result.artifact_result_refs],
            "outcome": result.outcome.value,
            "candidate_output_refs": [_ref_payload(ref) for ref in result.candidate_output_refs],
            "accepted_artifact_refs": [_ref_payload(ref) for ref in result.accepted_artifact_refs],
            "failed_artifact_ids": list(result.failed_artifact_ids),
            "resumable_artifact_ids": list(result.resumable_artifact_ids),
            "dependency_blocked_artifact_ids": list(result.dependency_blocked_artifact_ids),
            "required_incomplete_artifact_ids": list(result.required_incomplete_artifact_ids),
            "optional_incomplete_artifact_ids": list(result.optional_incomplete_artifact_ids),
            "frozen_result": None,
            "frozen_projection_gap": result.frozen_projection_gap,
            "environment_spec_ref": None,
            "provenance_manifest_ref": None,
            "quality_report_ref": None,
            "package_sha256": None,
            "input_state_only": None,
            "policy_version": result.policy_version,
        }
    )


def attachment_reconstruction_result_v2_ref(
    result: AttachmentReconstructionResultV2,
) -> ObjectRef:
    return ObjectRef(
        object_type="attachment-reconstruction-result",
        object_id=result.attachment_reconstruction_result_v2_id,
        object_version="v2",
        object_sha256=result.attachment_reconstruction_result_v2_sha256,
    )


def validate_attachment_reconstruction_result_v2_identity(
    result: AttachmentReconstructionResultV2,
) -> None:
    digest = attachment_reconstruction_result_v2_carried_sha256(result)
    if (
        result.attachment_reconstruction_result_v2_sha256 != digest
        or result.attachment_reconstruction_result_v2_id
        != f"attachment-reconstruction-result://sha256/{digest}"
    ):
        raise ValueError("attachment reconstruction result v2 identity is stale or invalid")


def public_source_retrieval_policy_carried_sha256(
    policy: PublicSourceRetrievalPolicyV2,
) -> str:
    return _payload_sha256(
        policy.model_dump(
            mode="json",
            exclude={
                "public_source_retrieval_policy_id",
                "public_source_retrieval_policy_sha256",
                "audit",
            },
            exclude_none=False,
        )
    )


def public_source_retrieval_policy_ref(
    policy: PublicSourceRetrievalPolicyV2,
) -> ObjectRef:
    return ObjectRef(
        object_type="public-source-retrieval-policy",
        object_id=policy.public_source_retrieval_policy_id,
        object_version="v2",
        object_sha256=policy.public_source_retrieval_policy_sha256,
    )


def public_source_safety_assessment_carried_sha256(
    assessment: PublicSourceSafetyAssessmentV2,
) -> str:
    return _payload_sha256(
        assessment.model_dump(
            mode="json",
            exclude={
                "public_source_safety_assessment_id",
                "public_source_safety_assessment_sha256",
                "audit",
            },
            exclude_none=False,
        )
    )


def public_source_safety_assessment_ref(
    assessment: PublicSourceSafetyAssessmentV2,
) -> ObjectRef:
    return ObjectRef(
        object_type="public-source-safety-assessment",
        object_id=assessment.public_source_safety_assessment_id,
        object_version="v2",
        object_sha256=assessment.public_source_safety_assessment_sha256,
    )


def source_evidence_v2_carried_sha256(
    evidence: SourceEvidenceV2,
) -> str:
    return _payload_sha256(_source_evidence_payload(evidence))


def source_evidence_v2_ref(
    evidence: SourceEvidenceV2,
) -> ObjectRef:
    return ObjectRef(
        object_type="source-evidence",
        object_id=evidence.source_evidence_id,
        object_version="v2",
        object_sha256=evidence.source_evidence_sha256,
    )


def source_evidence_set_carried_sha256(
    evidence_set: SourceEvidenceSetV2,
) -> str:
    return _payload_sha256(
        {
            "attachment_planning_context_ref": _ref_payload(evidence_set.attachment_planning_context_ref),
            "producer_task_view_ref": _ref_payload(evidence_set.producer_task_view_ref),
            "prompt_only_dependency_discovery_ref": _ref_payload(
                evidence_set.prompt_only_dependency_discovery_ref
            ),
            "retrieval_policy_ref": _ref_payload(evidence_set.retrieval_policy_ref),
            "retrieval_request_refs": [_ref_payload(ref) for ref in evidence_set.retrieval_request_refs],
            "source_evidence_refs": [
                _ref_payload(source_evidence_v2_ref(item)) for item in evidence_set.source_evidence
            ],
            "covered_target_refs": [_ref_payload(ref) for ref in evidence_set.covered_target_refs],
            "not_required_target_refs": [_ref_payload(ref) for ref in evidence_set.not_required_target_refs],
            "policy_version": evidence_set.policy_version,
        }
    )


def source_evidence_set_ref(
    evidence_set: SourceEvidenceSetV2,
) -> ObjectRef:
    return ObjectRef(
        object_type="source-evidence-set",
        object_id=evidence_set.source_evidence_set_id,
        object_version="v2",
        object_sha256=evidence_set.source_evidence_set_sha256,
    )


def _source_evidence_payload(
    evidence: SourceEvidenceV2,
) -> dict[str, object]:
    frozen_payload = evidence.source_evidence_v1.model_dump(
        mode="json",
        exclude={"audit"},
        exclude_none=False,
    )
    provenance_ref_payload = _ref_payload(evidence.source_evidence_v1.provenance_decision_ref)
    provenance_ref_payload.pop("object_sha256")
    frozen_payload["provenance_decision_ref"] = provenance_ref_payload
    return {
        "attachment_planning_context_ref": _ref_payload(evidence.attachment_planning_context_ref),
        "producer_task_view_ref": _ref_payload(evidence.producer_task_view_ref),
        "prompt_only_dependency_discovery_ref": _ref_payload(evidence.prompt_only_dependency_discovery_ref),
        "retrieval_policy_ref": _ref_payload(evidence.retrieval_policy_ref),
        "external_lead_ref": (
            None if evidence.external_lead_ref is None else _ref_payload(evidence.external_lead_ref)
        ),
        "external_lead_decision_ref": (
            None
            if evidence.external_lead_decision_ref is None
            else _behavior_ref_payload(evidence.external_lead_decision_ref)
        ),
        "search_result_ref": (
            None if evidence.search_result_ref is None else _ref_payload(evidence.search_result_ref)
        ),
        "fetch_result_ref": _ref_payload(evidence.fetch_result_ref),
        "safety_assessment_ref": _ref_payload(evidence.safety_assessment_ref),
        "claim_bindings": [
            item.model_dump(mode="json", exclude_none=False) for item in evidence.claim_bindings
        ],
        "source_evidence_v1": frozen_payload,
        "content_provenance_decision": (
            _provenance_decision_behavior_payload(evidence.content_provenance_decision)
        ),
        "policy_version": evidence.policy_version,
    }


def _provenance_decision_behavior_payload(
    decision: ProvenanceDecision,
) -> dict[str, object]:
    payload = decision.model_dump(
        mode="json",
        exclude={"audit"},
        exclude_none=False,
    )
    payload["derived_from"] = [_behavior_ref_payload(ref) for ref in decision.derived_from]
    return payload


def _behavior_ref_payload(ref: ObjectRef) -> dict[str, object]:
    payload = _ref_payload(ref)
    if ref.object_type == "provenance-decision":
        payload.pop("object_sha256")
    return payload


def _artifact_route_plan_entry_payload(
    entry: ArtifactRoutePlanEntryV2,
) -> dict[str, object]:
    return {
        "artifact_evidence_target_ref": _ref_payload(entry.artifact_evidence_target_ref),
        "attachment_dependency_id": entry.attachment_dependency_id,
        "artifact_id": entry.artifact_id,
        "criticality": entry.criticality.value,
        "outcome": entry.outcome.value,
        "facade_route_request_ref": (
            None if entry.facade_route_request_ref is None else _ref_payload(entry.facade_route_request_ref)
        ),
        "facade_route_decision": (
            None
            if entry.facade_route_decision is None
            else entry.facade_route_decision.model_dump(
                mode="json",
                exclude_none=False,
            )
        ),
        "build_spec": (
            None
            if entry.build_spec is None
            else {
                "artifact_build_spec_v2_sha256": (entry.build_spec.artifact_build_spec_v2_sha256),
                "artifact_build_spec_v2_id": (entry.build_spec.artifact_build_spec_v2_id),
            }
        ),
        "reasons": sorted(item.value for item in entry.reasons),
    }


def _artifact_routing_aggregate_outcome(
    entries: tuple[ArtifactRoutePlanEntryV2, ...],
) -> ArtifactRoutingAggregateOutcomeV2:
    if any(
        item.outcome is ArtifactRouteEntryOutcomeV2.BLOCKED_POLICY
        and item.criticality is not AttachmentCriticality.OPTIONAL
        for item in entries
    ):
        return ArtifactRoutingAggregateOutcomeV2.BLOCKED_POLICY
    if any(
        item.outcome is ArtifactRouteEntryOutcomeV2.BLOCKED_CAPABILITY
        and item.criticality is not AttachmentCriticality.OPTIONAL
        for item in entries
    ):
        return ArtifactRoutingAggregateOutcomeV2.BLOCKED_CAPABILITY
    if any(
        item.outcome
        in {
            ArtifactRouteEntryOutcomeV2.BLOCKED_CAPABILITY,
            ArtifactRouteEntryOutcomeV2.BLOCKED_POLICY,
        }
        for item in entries
    ):
        return ArtifactRoutingAggregateOutcomeV2.PARTIALLY_ROUTED
    return ArtifactRoutingAggregateOutcomeV2.ROUTED


def artifact_evidence_target_carried_sha256(
    target: ArtifactEvidenceTargetV2,
) -> str:
    return _payload_sha256(
        {
            "attachment_planning_context_ref": _ref_payload(target.attachment_planning_context_ref),
            "attachment_dependency_id": target.attachment_dependency_id,
            "artifact_id": target.artifact_id,
            "logical_path": target.logical_path,
            "media_type": target.media_type,
            "criticality": target.criticality.value,
            "requirement_evidence": [
                item.model_dump(mode="json", exclude_none=False) for item in target.requirement_evidence
            ],
            "candidate_source_refs": [_ref_payload(ref) for ref in target.candidate_source_refs],
            "policy_version": target.policy_version,
        }
    )


def artifact_evidence_row_carried_sha256(
    row: ArtifactEvidenceRowV2,
) -> str:
    return _payload_sha256(
        {
            "attachment_planning_context_ref": _ref_payload(row.attachment_planning_context_ref),
            "artifact_evidence_target_ref": _ref_payload(row.artifact_evidence_target_ref),
            "attachment_dependency_id": row.attachment_dependency_id,
            "row": row.row.model_dump(
                mode="json",
                exclude_none=False,
            ),
            "r2_row_sha256": row.r2_row_sha256,
            "r2_policy_version": row.r2_policy_version,
            "policy_version": row.policy_version,
        }
    )


def artifact_evidence_matrix_carried_sha256(
    matrix: ArtifactEvidenceMatrixV2,
) -> str:
    return _payload_sha256(
        {
            "attachment_planning_context_ref": _ref_payload(matrix.attachment_planning_context_ref),
            "producer_task_view_ref": _ref_payload(matrix.producer_task_view_ref),
            "safe_evidence_bundle_ref": _ref_payload(matrix.safe_evidence_bundle_ref),
            "source_r2_matrix_ref": _ref_payload(matrix.source_r2_matrix_ref),
            "rows": [
                item.model_dump(
                    mode="json",
                    exclude={"audit"},
                    exclude_none=False,
                )
                for item in matrix.rows
            ],
            "aggregate_mode": matrix.aggregate_mode,
            "r2_policy_version": matrix.r2_policy_version,
            "policy_version": matrix.policy_version,
        }
    )


def artifact_evidence_target_ref(
    target: ArtifactEvidenceTargetV2,
) -> ObjectRef:
    return ObjectRef(
        object_type="artifact-evidence-target",
        object_id=target.artifact_evidence_target_id,
        object_version="v2",
        object_sha256=target.artifact_evidence_target_sha256,
    )


def artifact_evidence_row_ref(
    row: ArtifactEvidenceRowV2,
) -> ObjectRef:
    return ObjectRef(
        object_type="artifact-evidence-row",
        object_id=row.artifact_evidence_row_id,
        object_version="v2",
        object_sha256=row.artifact_evidence_row_sha256,
    )


def artifact_evidence_matrix_ref(
    matrix: ArtifactEvidenceMatrixV2,
) -> ObjectRef:
    return ObjectRef(
        object_type="artifact-evidence-matrix",
        object_id=matrix.artifact_evidence_matrix_id,
        object_version="v2",
        object_sha256=matrix.artifact_evidence_matrix_sha256,
    )


def artifact_evidence_aggregate_mode_v2(
    rows: tuple[ArtifactEvidenceRowV2, ...],
) -> ArtifactEvidenceAggregateModeV2:
    modes = {item.row.selected_mode for item in rows}
    if len(modes) == 1:
        return next(iter(modes)).value
    if ReconstructionMode.BLOCKED in modes:
        return "BLOCKED"
    return "MIXED"


def prompt_only_dependency_planning_context_carried_sha256(
    context: PromptOnlyDependencyPlanningContextV2,
) -> str:
    return _payload_sha256(
        {
            "attachment_planning_context_ref": _ref_payload(context.attachment_planning_context_ref),
            "producer_task_view_ref": _ref_payload(context.producer_task_view_ref),
            "task_prompt_safety_gate_ref": _ref_payload(context.task_prompt_safety_gate_ref),
            "source_task_draft_sha256": context.source_task_draft_sha256,
            "source_trace_id": context.source_trace_id,
            "dependency_evidence_bindings": [
                item.model_dump(mode="json", exclude_none=False)
                for item in context.dependency_evidence_bindings
            ],
            "policy_version": context.policy_version,
        }
    )


def prompt_only_dependency_planning_context_ref(
    context: PromptOnlyDependencyPlanningContextV2,
) -> ObjectRef:
    return ObjectRef(
        object_type="prompt-only-dependency-planning-context",
        object_id=context.prompt_only_dependency_planning_context_id,
        object_version="v2",
        object_sha256=(context.prompt_only_dependency_planning_context_sha256),
    )


def prompt_only_dependency_discovery_carried_sha256(
    discovery: PromptOnlyDependencyDiscoveryV2,
) -> str:
    return _payload_sha256(
        {
            "attachment_planning_context_ref": _ref_payload(discovery.attachment_planning_context_ref),
            "producer_task_view_ref": _ref_payload(discovery.producer_task_view_ref),
            "dependency_planning_context_ref": _ref_payload(discovery.dependency_planning_context_ref),
            "prompt_boundary_enforcement_ref": _ref_payload(discovery.prompt_boundary_enforcement_ref),
            "targets": [
                item.model_dump(
                    mode="json",
                    exclude={"audit"},
                    exclude_none=False,
                )
                for item in discovery.targets
            ],
            "existing_target_refs": [_ref_payload(ref) for ref in discovery.existing_target_refs],
            "discovered_target_refs": [_ref_payload(ref) for ref in discovery.discovered_target_refs],
            "deterministic_dependency_ids": list(discovery.deterministic_dependency_ids),
            "semantic_dependency_ids": list(discovery.semantic_dependency_ids),
            "semantic_evaluated": discovery.semantic_evaluated,
            "model_profile": discovery.model_profile,
            "prompt_version": discovery.prompt_version,
            "policy_version": discovery.policy_version,
        }
    )


def prompt_only_dependency_discovery_ref(
    discovery: PromptOnlyDependencyDiscoveryV2,
) -> ObjectRef:
    return ObjectRef(
        object_type="prompt-only-dependency-discovery",
        object_id=discovery.prompt_only_dependency_discovery_id,
        object_version="v2",
        object_sha256=discovery.prompt_only_dependency_discovery_sha256,
    )


def validate_artifact_evidence_target_identity(
    target: ArtifactEvidenceTargetV2,
) -> None:
    digest = artifact_evidence_target_carried_sha256(target)
    if (
        target.artifact_evidence_target_sha256 != digest
        or target.artifact_evidence_target_id != f"artifact-evidence-target://sha256/{digest}"
    ):
        raise ValueError("artifact evidence target identity is stale or invalid")


def _validate_matrix_rows(
    rows: tuple[ArtifactEvidenceRowV2, ...],
    context_ref: ObjectRef,
) -> None:
    _require_unique(
        "artifact evidence row IDs",
        tuple(item.artifact_evidence_row_id for item in rows),
    )
    _require_unique(
        "artifact evidence row hashes",
        tuple(item.artifact_evidence_row_sha256 for item in rows),
    )
    _require_unique(
        "artifact evidence target refs",
        tuple(_ref_key(item.artifact_evidence_target_ref) for item in rows),
    )
    _require_unique(
        "attachment dependency IDs",
        tuple(item.attachment_dependency_id for item in rows),
    )
    _require_unique(
        "artifact IDs",
        tuple(item.row.artifact_id for item in rows),
    )
    _require_unique(
        "artifact logical paths",
        tuple(item.row.logical_path for item in rows),
    )
    if any(item.attachment_planning_context_ref != context_ref for item in rows):
        raise ValueError("all artifact evidence rows must bind the matrix planning context")
    observed_order = tuple((item.row.logical_path, item.row.artifact_id) for item in rows)
    if observed_order != tuple(sorted(observed_order)):
        raise ValueError("artifact evidence rows must be sorted by path and artifact ID")


def _validate_discovery_targets(
    *,
    targets: tuple[ArtifactEvidenceTargetV2, ...],
    context_ref: ObjectRef,
    existing_refs: tuple[ObjectRef, ...],
    discovered_refs: tuple[ObjectRef, ...],
    deterministic_ids: tuple[str, ...],
    semantic_ids: tuple[str, ...],
) -> None:
    target_refs = tuple(artifact_evidence_target_ref(item) for item in targets)
    target_ref_keys = tuple(_ref_key(ref) for ref in target_refs)
    dependency_ids = tuple(item.attachment_dependency_id for item in targets)
    artifact_ids = tuple(item.artifact_id for item in targets)
    logical_paths = tuple(item.logical_path for item in targets)
    _require_unique("prompt-only target refs", target_ref_keys)
    _require_unique("prompt-only dependency IDs", dependency_ids)
    _require_unique("prompt-only artifact IDs", artifact_ids)
    _require_unique("prompt-only logical paths", logical_paths)
    if any(item.attachment_planning_context_ref != context_ref for item in targets):
        raise ValueError("prompt-only targets must bind the discovery planning context")
    observed_order = tuple((item.logical_path, item.artifact_id) for item in targets)
    if observed_order != tuple(sorted(observed_order)):
        raise ValueError("prompt-only targets must be sorted by path and artifact ID")
    for label, refs in (
        ("existing target refs", existing_refs),
        ("discovered target refs", discovered_refs),
    ):
        keys = tuple(_ref_key(ref) for ref in refs)
        _require_unique(label, keys)
        if keys != tuple(sorted(keys)):
            raise ValueError(f"{label} must be sorted")
        for ref in refs:
            _require_ref_version(
                ref,
                "artifact-evidence-target",
                "v2",
                label,
            )
    existing_keys = {_ref_key(ref) for ref in existing_refs}
    discovered_keys = {_ref_key(ref) for ref in discovered_refs}
    if existing_keys & discovered_keys:
        raise ValueError("existing and discovered target refs must be disjoint")
    if existing_keys | discovered_keys != set(target_ref_keys):
        raise ValueError("existing and discovered target refs must exactly cover targets")
    for label, values in (
        ("deterministic dependency IDs", deterministic_ids),
        ("semantic dependency IDs", semantic_ids),
    ):
        _require_unique(label, values)
        if values != tuple(sorted(values)):
            raise ValueError(f"{label} must be sorted")
    if set(deterministic_ids) & set(semantic_ids):
        raise ValueError("deterministic and semantic dependency IDs must be disjoint")
    discovered_dependency_ids = {
        item.attachment_dependency_id
        for item, ref in zip(targets, target_refs, strict=True)
        if _ref_key(ref) in discovered_keys
    }
    if set(deterministic_ids) | set(semantic_ids) != discovered_dependency_ids:
        raise ValueError("deterministic and semantic IDs must cover discovered targets")


def _validate_result_receipt_route(
    entry: ArtifactRoutePlanEntryV2,
    receipt: ArtifactExecutionReceiptV2,
) -> None:
    receipt_digest = artifact_execution_receipt_carried_sha256(receipt)
    if (
        receipt.artifact_execution_receipt_sha256 != receipt_digest
        or receipt.artifact_execution_receipt_id != f"artifact-execution-receipt://sha256/{receipt_digest}"
    ):
        raise ValueError("artifact result receipt identity is stale or invalid")
    if entry.build_spec is None or entry.facade_route_decision is None:
        raise ValueError("routed artifact result requires exact build and route facts")
    spec_digest = artifact_build_spec_v2_carried_sha256(entry.build_spec)
    if (
        entry.build_spec.artifact_build_spec_v2_sha256 != spec_digest
        or entry.build_spec.artifact_build_spec_v2_id != f"artifact-build-spec://sha256/{spec_digest}"
    ):
        raise ValueError("artifact result build spec identity is stale or invalid")
    if receipt.outcome is ArtifactExecutionReceiptOutcomeV2.BLOCKED_DEPENDENCY:
        return
    if receipt.facade_request is None or receipt.facade_result is None:
        raise ValueError("attempted artifact result requires facade request and result")
    validate_attachment_execution_request_identity(receipt.facade_request)
    validate_attachment_execution_result_identity(receipt.facade_result)
    expected_build_ref = _object_ref_from_facade(receipt.facade_request.build_spec_ref)
    if expected_build_ref != artifact_build_spec_v2_ref(entry.build_spec):
        raise ValueError("artifact result receipt build spec is mismatched")
    if (
        _object_ref_from_facade(receipt.facade_request.execution_plan_ref)
        != receipt.artifact_execution_plan_ref
    ):
        raise ValueError("artifact result facade request execution plan is mismatched")
    expected_route_kind = (
        "PROVIDER" if entry.build_spec.selected_route_kind is ArtifactRouteKindV2.PROVIDER else "RUNTIME"
    )
    selected_id = entry.facade_route_decision.selected_id
    if (
        receipt.facade_request.selected_route_kind.value != expected_route_kind
        or receipt.facade_result.selected_route_kind.value != expected_route_kind
        or receipt.facade_request.selected_route_id != selected_id
        or receipt.facade_result.selected_route_id != selected_id
    ):
        raise ValueError("artifact result receipt selected route is mismatched")


def artifact_build_result_projection_gap_v2(
    *,
    route_entry: ArtifactRoutePlanEntryV2,
    receipt: ArtifactExecutionReceiptV2 | None,
) -> ArtifactBuildResultProjectionGapV2 | None:
    if route_entry.build_spec is None:
        return ArtifactBuildResultProjectionGapV2.BUILD_SPEC_UNAVAILABLE
    if receipt is None or receipt.outcome is ArtifactExecutionReceiptOutcomeV2.BLOCKED_DEPENDENCY:
        return ArtifactBuildResultProjectionGapV2.DEPENDENCY_NOT_ATTEMPTED
    if receipt.facade_result is None or receipt.facade_result.worker_version is None:
        return ArtifactBuildResultProjectionGapV2.WORKER_VERSION_UNAVAILABLE
    if len(receipt.facade_result.worker_version) > 128:
        return ArtifactBuildResultProjectionGapV2.WORKER_VERSION_UNREPRESENTABLE
    return None


def _validate_frozen_artifact_result(
    result: ArtifactBuildResultV2,
) -> None:
    frozen = result.frozen_result
    entry = result.route_entry
    receipt = result.execution_receipt
    if (
        frozen is None
        or entry.build_spec is None
        or receipt is None
        or receipt.facade_result is None
        or receipt.facade_result.worker_version is None
    ):
        raise ValueError("frozen artifact result requires complete observed execution facts")
    facade_result = receipt.facade_result
    expected_lineage = (
        result.artifact_routing_plan_ref,
        artifact_build_spec_v2_ref(entry.build_spec),
        receipt.artifact_execution_plan_ref,
        receipt.artifact_execution_unit_ref,
        artifact_execution_receipt_ref(receipt),
        _object_ref_from_facade(attachment_execution_result_ref(facade_result)),
    )
    expected_output_ref = (
        _object_ref_from_facade(facade_result.output_ref) if facade_result.output_ref is not None else None
    )
    expected_failure = None
    if result.outcome is not ArtifactBuildResultOutcomeV2.SUCCEEDED:
        if facade_result.failure_code is None:
            raise ValueError("failed frozen artifact result requires failure code")
        expected_failure = artifact_execution_failure_record_v2(
            result.outcome,
            facade_result.failure_code,
        )
    provider = (
        facade_result.selected_route_id
        if entry.build_spec.selected_route_kind is ArtifactRouteKindV2.PROVIDER
        else None
    )
    runtime = (
        facade_result.selected_route_id
        if entry.build_spec.selected_route_kind is ArtifactRouteKindV2.RUNTIME
        else None
    )
    if (
        frozen.build_spec_ref != artifact_build_spec_v2_ref(entry.build_spec)
        or frozen.build_spec_sha256 != entry.build_spec.artifact_build_spec_v2_sha256
        or frozen.status is not ArtifactBuildStatus(result.outcome.value)
        or frozen.output_ref != expected_output_ref
        or frozen.output_sha256 != facade_result.output_sha256
        or frozen.lineage_refs != expected_lineage
        or frozen.validation_result_refs
        or frozen.finding_refs
        or frozen.worker_version != facade_result.worker_version
        or frozen.provider != provider
        or frozen.runtime != runtime
        or frozen.failure != expected_failure
    ):
        raise ValueError("frozen artifact result does not match exact execution facts")
    frozen_digest = artifact_build_result_v1_carried_sha256(frozen)
    if frozen.artifact_build_result_id != (f"artifact-build-result://sha256/{frozen_digest}"):
        raise ValueError("frozen artifact result identity is stale or invalid")
    _validate_artifact_result_audit(frozen.audit, expected_lineage)


def _frozen_artifact_result_payload(
    result: ArtifactBuildResult,
) -> dict[str, object]:
    return result.model_dump(
        mode="json",
        exclude={
            "artifact_build_result_id",
            "audit",
        },
        exclude_none=False,
    )


def _candidate_output_refs(
    results: tuple[ArtifactBuildResultV2, ...],
) -> tuple[ObjectRef, ...]:
    refs = tuple(
        _object_ref_from_facade(item.execution_receipt.facade_result.output_ref)
        for item in results
        if item.outcome is ArtifactBuildResultOutcomeV2.SUCCEEDED
        and item.execution_receipt is not None
        and item.execution_receipt.facade_result is not None
        and item.execution_receipt.facade_result.output_ref is not None
    )
    return tuple(sorted(refs, key=_ref_key))


def _artifact_result_ids(
    results: tuple[ArtifactBuildResultV2, ...],
    outcomes: set[ArtifactBuildResultOutcomeV2],
) -> tuple[str, ...]:
    return tuple(sorted(item.route_entry.artifact_id for item in results if item.outcome in outcomes))


def attachment_reconstruction_outcome_v2(
    results: tuple[ArtifactBuildResultV2, ...],
    *,
    has_routing_plan: bool,
) -> AttachmentReconstructionOutcomeV2:
    if not has_routing_plan:
        return AttachmentReconstructionOutcomeV2.NOT_REQUIRED
    succeeded = any(item.outcome is ArtifactBuildResultOutcomeV2.SUCCEEDED for item in results)
    incomplete = any(item.outcome is not ArtifactBuildResultOutcomeV2.SUCCEEDED for item in results)
    if succeeded and incomplete:
        return AttachmentReconstructionOutcomeV2.PARTIAL_FAILURE
    if succeeded:
        return AttachmentReconstructionOutcomeV2.PENDING_VALIDATION
    direct = tuple(
        item for item in results if item.outcome is not ArtifactBuildResultOutcomeV2.BLOCKED_DEPENDENCY
    )
    if direct and all(item.outcome is ArtifactBuildResultOutcomeV2.RETRYABLE_FAILURE for item in direct):
        return AttachmentReconstructionOutcomeV2.RETRYABLE_FAILURE
    return AttachmentReconstructionOutcomeV2.BLOCKED


def _validate_artifact_result_audit(
    audit: ContractAudit,
    refs: tuple[ObjectRef, ...],
) -> None:
    unique = {_ref_key(ref): ref for ref in refs}
    expected = tuple(unique[key] for key in sorted(unique))
    if audit.input_refs != expected:
        raise ValueError("artifact result audit lineage is stale or mismatched")
    if not any(
        binding.component == "artifact-results" and binding.version == "r5-07"
        for binding in audit.governing_versions
    ):
        raise ValueError("artifact result audit governing version is missing")


def _unsafe_dependency_ref(ref: ObjectRef) -> bool:
    normalized = f"{ref.object_type}:{ref.object_id}".casefold().replace(
        "_",
        "-",
    )
    return any(marker in normalized for marker in _DENIED_DEPENDENCY_REF_MARKERS)


def _payload_sha256(payload: object) -> str:
    encoded = json.dumps(
        canonical_value_v2(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _require_ref_version(
    ref: ObjectRef,
    expected_type: str,
    expected_version: str,
    field_name: str,
) -> None:
    _require_ref_type(ref, expected_type, field_name)
    if ref.object_version != expected_version:
        raise ValueError(f"{field_name} must reference {expected_type} {expected_version}")


def _require_unique(
    field_name: str,
    values: tuple[object, ...],
) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must be unique")


def _require_sorted_unique_values(
    field_name: str,
    values: tuple[str, ...],
) -> None:
    _require_unique(field_name, values)
    if values != tuple(sorted(values)):
        raise ValueError(f"{field_name} must be sorted")


def _validate_receipt_refs(refs: tuple[ObjectRef, ...]) -> None:
    keys = tuple(_ref_key(ref) for ref in refs)
    _require_unique("failed dependency receipt refs", keys)
    if keys != tuple(sorted(keys)):
        raise ValueError("failed dependency receipt refs must be sorted")
    for ref in refs:
        _require_ref_version(
            ref,
            "artifact-execution-receipt",
            "v2",
            "failed_dependency_receipt_refs",
        )


def _receipt_artifact_ids(
    receipts: tuple[ArtifactExecutionReceiptV2, ...],
    outcomes: set[ArtifactExecutionReceiptOutcomeV2],
) -> tuple[str, ...]:
    return tuple(sorted(receipt.artifact_id for receipt in receipts if receipt.outcome in outcomes))


def _object_ref_from_facade(ref: FacadeObjectRef) -> ObjectRef:
    return ObjectRef(
        object_type=ref.object_type,
        object_id=ref.object_id,
        object_version=ref.object_version,
        object_sha256=ref.object_sha256,
    )


def _ref_key(ref: ObjectRef) -> tuple[str, str, str, str]:
    return (
        ref.object_type,
        ref.object_id,
        ref.object_version,
        ref.object_sha256,
    )
