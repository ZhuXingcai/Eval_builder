from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import ClassVar, Literal, Self

from pydantic import Field, model_validator

from eval_factory.contracts.core import (
    ContractAudit,
    Identifier,
    ObjectRef,
    Sha256,
    TypedAttribute,
)
from eval_factory.contracts.core_v2 import (
    ContractModelV2,
    canonical_value_v2,
)
from eval_factory.contracts.external_stability_v2 import (
    ExternalRealTraceStabilityOutcomeV2,
)
from eval_factory.contracts.label_quality_v2 import (
    LabelQualityEvidenceClassV2,
    LabelQualityOutcomeV2,
)

EXTERNAL_EVIDENCE_POLICY_VERSION: Literal["external-evidence/r8-10-v1"] = "external-evidence/r8-10-v1"

_RUNTIME_COUNT_KEYS = (
    "claude_code",
    "codex",
    "hermes",
)
_CURATED_RUNTIME_COUNT_KEYS = ("claude_curated",)
_EVIDENCE_CLASS_COUNT_KEYS = (
    "DEVELOPMENT",
    "REAL",
    "REPLAY",
    "SYNTHETIC",
)
_DENIED_REF_MARKERS = (
    "answer-bearing",
    "credential",
    "final-answer",
    "final-output",
    "grader-rule",
    "hidden-condition",
    "private-reference",
    "quarantine",
    "raw-trace",
    "runtime-transcript",
    "secret",
)


class ExternalEvidenceAdmissionOutcomeV2(StrEnum):
    ADMITTED = "ADMITTED"
    BLOCKED = "BLOCKED"


class ExternalEvidenceAdmissionReasonV2(StrEnum):
    NONE = "NONE"
    INVALID_PACKAGE = "INVALID_PACKAGE"
    CORPUS_INSUFFICIENT = "CORPUS_INSUFFICIENT"
    PARTITION_OVERLAP = "PARTITION_OVERLAP"
    MATERIAL_INTEGRITY_FAILED = "MATERIAL_INTEGRITY_FAILED"
    REQUIREMENT_STALE = "REQUIREMENT_STALE"
    REFERENCE_POLICY_STALE = "REFERENCE_POLICY_STALE"
    OBSERVATION_POLICY_STALE = "OBSERVATION_POLICY_STALE"


class ExternalReferenceAuthoringModeV2(StrEnum):
    AUTOMATIC_INDEPENDENT = "AUTOMATIC_INDEPENDENT"
    IMPORTED_INDEPENDENT = "IMPORTED_INDEPENDENT"


class _ExternalEvidenceObjectV2(ContractModelV2):
    object_id: Identifier
    object_sha256: Sha256
    audit: ContractAudit

    OBJECT_TYPE: ClassVar[str]
    OBJECT_VERSION: ClassVar[str] = "v2"

    @classmethod
    def _create(
        cls,
        *,
        audit: ContractAudit,
        values: dict[str, object],
    ) -> Self:
        provisional = cls.model_construct(
            object_id=f"{cls.OBJECT_TYPE}://pending",
            object_sha256="0" * 64,
            audit=audit,
            **values,  # type: ignore[arg-type]
        )
        refs = _collect_refs(provisional)
        safe_audit = _audit_with_refs(audit, refs)
        provisional = cls.model_construct(
            object_id=f"{cls.OBJECT_TYPE}://pending",
            object_sha256="0" * 64,
            audit=safe_audit,
            **values,  # type: ignore[arg-type]
        )
        digest = _carried_sha256(provisional)
        return cls(
            object_id=f"{cls.OBJECT_TYPE}://sha256/{digest}",
            object_sha256=digest,
            audit=safe_audit,
            **values,
        )

    @model_validator(mode="after")
    def validate_derived_identity(self) -> Self:
        digest = _carried_sha256(self)
        if self.object_sha256 != digest or self.object_id != f"{self.OBJECT_TYPE}://sha256/{digest}":
            raise ValueError("external evidence object identity is stale")
        refs = _collect_refs(self)
        _require_safe_refs(refs)
        if self.audit.input_refs != refs:
            raise ValueError("external evidence audit refs differ from object refs")
        return self

    def to_ref(self) -> ObjectRef:
        return ObjectRef(
            object_type=self.OBJECT_TYPE,
            object_id=self.object_id,
            object_version=self.OBJECT_VERSION,
            object_sha256=self.object_sha256,
        )


class ExternalEvidencePackageManifestV2(_ExternalEvidenceObjectV2):
    schema_version: Literal["eval-factory/external-evidence-package-manifest/v2"] = (
        "eval-factory/external-evidence-package-manifest/v2"
    )
    OBJECT_TYPE: ClassVar[str] = "external-evidence-package-manifest"

    requirement_source_ref: ObjectRef
    requirement_spec_ref: ObjectRef
    source_authorization_ref: ObjectRef
    source_population_ref: ObjectRef
    inventory_commitment_ref: ObjectRef
    train_partition_ref: ObjectRef
    development_partition_ref: ObjectRef
    label_spec_refs: tuple[ObjectRef, ...] = Field(
        min_length=3,
        max_length=3,
    )
    reference_policy_ref: ObjectRef
    observation_policy_ref: ObjectRef
    adapter_name: Literal[
        "curated_trajectory_v1",
        "runtime_snapshot_v1",
    ]
    adapter_version: Literal["1.0.0"]
    source_count: int = Field(ge=100, le=100_000)
    unique_trace_count: int = Field(ge=100, le=100_000)
    unique_raw_hash_count: int = Field(ge=100, le=100_000)
    total_raw_bytes: int = Field(ge=1, le=1_000_000_000_000)
    runtime_counts: tuple[TypedAttribute, ...] = Field(
        min_length=1,
        max_length=3,
    )
    inventory_sha256: Sha256
    policy_version: Literal["external-evidence/r8-10-v1"] = EXTERNAL_EVIDENCE_POLICY_VERSION

    @classmethod
    def create(
        cls,
        *,
        requirement_source_ref: ObjectRef,
        requirement_spec_ref: ObjectRef,
        source_authorization_ref: ObjectRef,
        source_population_ref: ObjectRef,
        inventory_commitment_ref: ObjectRef,
        train_partition_ref: ObjectRef,
        development_partition_ref: ObjectRef,
        label_spec_refs: tuple[ObjectRef, ...],
        reference_policy_ref: ObjectRef,
        observation_policy_ref: ObjectRef,
        adapter_name: Literal[
            "curated_trajectory_v1",
            "runtime_snapshot_v1",
        ],
        adapter_version: Literal["1.0.0"],
        source_count: int,
        unique_trace_count: int,
        unique_raw_hash_count: int,
        total_raw_bytes: int,
        runtime_counts: tuple[TypedAttribute, ...],
        inventory_sha256: str,
        audit: ContractAudit,
    ) -> ExternalEvidencePackageManifestV2:
        return cls._create(
            audit=audit,
            values={
                "requirement_source_ref": requirement_source_ref,
                "requirement_spec_ref": requirement_spec_ref,
                "source_authorization_ref": source_authorization_ref,
                "source_population_ref": source_population_ref,
                "inventory_commitment_ref": inventory_commitment_ref,
                "train_partition_ref": train_partition_ref,
                "development_partition_ref": development_partition_ref,
                "label_spec_refs": _sorted_refs(label_spec_refs),
                "reference_policy_ref": reference_policy_ref,
                "observation_policy_ref": observation_policy_ref,
                "adapter_name": adapter_name,
                "adapter_version": adapter_version,
                "source_count": source_count,
                "unique_trace_count": unique_trace_count,
                "unique_raw_hash_count": unique_raw_hash_count,
                "total_raw_bytes": total_raw_bytes,
                "runtime_counts": _sorted_attributes(runtime_counts),
                "inventory_sha256": inventory_sha256,
                "policy_version": EXTERNAL_EVIDENCE_POLICY_VERSION,
            },
        )

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        _require_ref(
            self.requirement_source_ref,
            "evaluation-requirement-source",
            "v2",
            "requirement_source_ref",
        )
        _require_ref(
            self.requirement_spec_ref,
            "evaluation-requirement-spec",
            "v2",
            "requirement_spec_ref",
        )
        _require_ref(
            self.source_authorization_ref,
            "external-source-authorization",
            "v2",
            "source_authorization_ref",
        )
        _require_ref(
            self.source_population_ref,
            "external-source-population",
            "v2",
            "source_population_ref",
        )
        _require_ref(
            self.inventory_commitment_ref,
            "external-corpus-inventory",
            "private-v1",
            "inventory_commitment_ref",
        )
        _require_ref(
            self.train_partition_ref,
            "external-partition-manifest",
            "v2",
            "train_partition_ref",
        )
        _require_ref(
            self.development_partition_ref,
            "external-partition-manifest",
            "v2",
            "development_partition_ref",
        )
        _require_sorted_unique_refs(
            self.label_spec_refs,
            "label_spec_refs",
        )
        for ref in self.label_spec_refs:
            _require_ref(ref, "label-spec", "v2", "label_spec_refs")
        _require_ref(
            self.reference_policy_ref,
            "external-reference-authoring-policy",
            "v2",
            "reference_policy_ref",
        )
        _require_ref(
            self.observation_policy_ref,
            "external-observation-policy",
            "v2",
            "observation_policy_ref",
        )
        if self.source_count != self.unique_trace_count or self.source_count != self.unique_raw_hash_count:
            raise ValueError("external package source counts must be exact")
        _require_attribute_counts(
            self.runtime_counts,
            expected_keys=(
                _CURATED_RUNTIME_COUNT_KEYS
                if self.adapter_name == "curated_trajectory_v1"
                else _RUNTIME_COUNT_KEYS
            ),
            expected_total=self.source_count,
            label="runtime_counts",
        )
        if self.inventory_commitment_ref.object_sha256 != self.inventory_sha256:
            raise ValueError("inventory commitment hash differs from manifest")
        return self


class ExternalEvidenceAdmissionReportV2(_ExternalEvidenceObjectV2):
    schema_version: Literal["eval-factory/external-evidence-admission-report/v2"] = (
        "eval-factory/external-evidence-admission-report/v2"
    )
    OBJECT_TYPE: ClassVar[str] = "external-evidence-admission-report"

    package_manifest_ref: ObjectRef
    material_closure_ref: ObjectRef | None = None
    outcome: ExternalEvidenceAdmissionOutcomeV2
    reason_codes: tuple[ExternalEvidenceAdmissionReasonV2, ...] = Field(
        min_length=1,
    )
    source_count: int = Field(ge=0, le=100_000)
    admitted_source_count: int = Field(ge=0, le=100_000)
    rejected_source_count: int = Field(ge=0, le=100_000)
    unique_trace_count: int = Field(ge=0, le=100_000)
    unique_raw_hash_count: int = Field(ge=0, le=100_000)
    duplicate_source_count: int = Field(ge=0, le=100_000)
    overlap_source_count: int = Field(ge=0, le=100_000)
    total_raw_bytes: int = Field(ge=0, le=1_000_000_000_000)
    manifest_verified: bool
    material_verified: bool
    policy_version: Literal["external-evidence/r8-10-v1"] = EXTERNAL_EVIDENCE_POLICY_VERSION

    @classmethod
    def create(
        cls,
        *,
        package_manifest_ref: ObjectRef,
        material_closure_ref: ObjectRef | None,
        outcome: ExternalEvidenceAdmissionOutcomeV2,
        reason_codes: tuple[ExternalEvidenceAdmissionReasonV2, ...],
        source_count: int,
        admitted_source_count: int,
        rejected_source_count: int,
        unique_trace_count: int,
        unique_raw_hash_count: int,
        duplicate_source_count: int,
        overlap_source_count: int,
        total_raw_bytes: int,
        manifest_verified: bool,
        material_verified: bool,
        audit: ContractAudit,
    ) -> ExternalEvidenceAdmissionReportV2:
        return cls._create(
            audit=audit,
            values={
                "package_manifest_ref": package_manifest_ref,
                "material_closure_ref": material_closure_ref,
                "outcome": outcome,
                "reason_codes": tuple(sorted(set(reason_codes), key=lambda value: value.value)),
                "source_count": source_count,
                "admitted_source_count": admitted_source_count,
                "rejected_source_count": rejected_source_count,
                "unique_trace_count": unique_trace_count,
                "unique_raw_hash_count": unique_raw_hash_count,
                "duplicate_source_count": duplicate_source_count,
                "overlap_source_count": overlap_source_count,
                "total_raw_bytes": total_raw_bytes,
                "manifest_verified": manifest_verified,
                "material_verified": material_verified,
                "policy_version": EXTERNAL_EVIDENCE_POLICY_VERSION,
            },
        )

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        _require_ref(
            self.package_manifest_ref,
            "external-evidence-package-manifest",
            "v2",
            "package_manifest_ref",
        )
        if self.source_count != (self.admitted_source_count + self.rejected_source_count):
            raise ValueError("admission source partition is not exact")
        if self.reason_codes != tuple(sorted(set(self.reason_codes), key=lambda value: value.value)):
            raise ValueError("admission reasons must be sorted and unique")
        if self.outcome is ExternalEvidenceAdmissionOutcomeV2.ADMITTED:
            if (
                self.material_closure_ref is None
                or self.reason_codes != (ExternalEvidenceAdmissionReasonV2.NONE,)
                or self.admitted_source_count != self.source_count
                or self.rejected_source_count != 0
                or self.unique_trace_count != self.source_count
                or self.unique_raw_hash_count != self.source_count
                or self.duplicate_source_count != 0
                or self.overlap_source_count != 0
                or not self.manifest_verified
                or not self.material_verified
            ):
                raise ValueError("admitted external evidence is incomplete")
            _require_ref(
                self.material_closure_ref,
                "external-evidence-material-closure",
                "private-v1",
                "material_closure_ref",
            )
        else:
            if (
                self.material_closure_ref is not None
                or self.reason_codes == (ExternalEvidenceAdmissionReasonV2.NONE,)
                or ExternalEvidenceAdmissionReasonV2.NONE in self.reason_codes
            ):
                raise ValueError("blocked admission cannot carry accepted material")
        return self


class ExternalLabelObservationSummaryV2(_ExternalEvidenceObjectV2):
    schema_version: Literal["eval-factory/external-label-observation-summary/v2"] = (
        "eval-factory/external-label-observation-summary/v2"
    )
    OBJECT_TYPE: ClassVar[str] = "external-label-observation-summary"

    package_manifest_ref: ObjectRef
    source_population_ref: ObjectRef
    reference_closure_ref: ObjectRef
    observation_closure_ref: ObjectRef
    label_spec_refs: tuple[ObjectRef, ...] = Field(
        min_length=3,
        max_length=3,
    )
    reference_policy_ref: ObjectRef
    observation_policy_ref: ObjectRef
    authoring_mode: ExternalReferenceAuthoringModeV2
    reference_authority_committed: Literal[True] = True
    observation_reference_access_denied: Literal[True] = True
    total_pair_count: int = Field(ge=1, le=30_000_000)
    reference_ready_count: int = Field(ge=0, le=30_000_000)
    reference_abstained_count: int = Field(ge=0, le=30_000_000)
    reference_blocked_count: int = Field(ge=0, le=30_000_000)
    observation_complete_count: int = Field(ge=0, le=30_000_000)
    observation_missing_count: int = Field(ge=0, le=30_000_000)
    observation_blocked_count: int = Field(ge=0, le=30_000_000)
    policy_version: Literal["external-evidence/r8-10-v1"] = EXTERNAL_EVIDENCE_POLICY_VERSION

    @classmethod
    def create(
        cls,
        *,
        package_manifest_ref: ObjectRef,
        source_population_ref: ObjectRef,
        reference_closure_ref: ObjectRef,
        observation_closure_ref: ObjectRef,
        label_spec_refs: tuple[ObjectRef, ...],
        reference_policy_ref: ObjectRef,
        observation_policy_ref: ObjectRef,
        authoring_mode: ExternalReferenceAuthoringModeV2,
        reference_authority_committed: Literal[True],
        observation_reference_access_denied: Literal[True],
        total_pair_count: int,
        reference_ready_count: int,
        reference_abstained_count: int,
        reference_blocked_count: int,
        observation_complete_count: int,
        observation_missing_count: int,
        observation_blocked_count: int,
        audit: ContractAudit,
    ) -> ExternalLabelObservationSummaryV2:
        return cls._create(
            audit=audit,
            values={
                "package_manifest_ref": package_manifest_ref,
                "source_population_ref": source_population_ref,
                "reference_closure_ref": reference_closure_ref,
                "observation_closure_ref": observation_closure_ref,
                "label_spec_refs": _sorted_refs(label_spec_refs),
                "reference_policy_ref": reference_policy_ref,
                "observation_policy_ref": observation_policy_ref,
                "authoring_mode": authoring_mode,
                "reference_authority_committed": (reference_authority_committed),
                "observation_reference_access_denied": (observation_reference_access_denied),
                "total_pair_count": total_pair_count,
                "reference_ready_count": reference_ready_count,
                "reference_abstained_count": reference_abstained_count,
                "reference_blocked_count": reference_blocked_count,
                "observation_complete_count": observation_complete_count,
                "observation_missing_count": observation_missing_count,
                "observation_blocked_count": observation_blocked_count,
                "policy_version": EXTERNAL_EVIDENCE_POLICY_VERSION,
            },
        )

    @model_validator(mode="after")
    def validate_summary(self) -> Self:
        _require_ref(
            self.package_manifest_ref,
            "external-evidence-package-manifest",
            "v2",
            "package_manifest_ref",
        )
        _require_ref(
            self.source_population_ref,
            "external-source-population",
            "v2",
            "source_population_ref",
        )
        _require_ref(
            self.reference_closure_ref,
            "external-reference-set",
            "private-v1",
            "reference_closure_ref",
        )
        _require_ref(
            self.observation_closure_ref,
            "blind-label-observation-set",
            "private-v1",
            "observation_closure_ref",
        )
        _require_sorted_unique_refs(
            self.label_spec_refs,
            "label_spec_refs",
        )
        for ref in self.label_spec_refs:
            _require_ref(ref, "label-spec", "v2", "label_spec_refs")
        _require_ref(
            self.reference_policy_ref,
            "external-reference-authoring-policy",
            "v2",
            "reference_policy_ref",
        )
        _require_ref(
            self.observation_policy_ref,
            "external-observation-policy",
            "v2",
            "observation_policy_ref",
        )
        if (
            self.reference_ready_count + self.reference_abstained_count + self.reference_blocked_count
            != self.total_pair_count
        ):
            raise ValueError("reference counts do not cover all label pairs")
        if (
            self.observation_complete_count + self.observation_missing_count + self.observation_blocked_count
            != self.total_pair_count
        ):
            raise ValueError("observation counts do not cover all label pairs")
        return self


class ExternalSc010Sc011EvidenceIndexV2(_ExternalEvidenceObjectV2):
    schema_version: Literal["eval-factory/external-sc010-sc011-evidence-index/v2"] = (
        "eval-factory/external-sc010-sc011-evidence-index/v2"
    )
    OBJECT_TYPE: ClassVar[str] = "external-sc010-sc011-evidence-index"

    package_manifest_ref: ObjectRef
    admission_report_ref: ObjectRef
    requirement_spec_ref: ObjectRef
    observation_summary_ref: ObjectRef
    freeze_result_ref: ObjectRef
    dataset_manifest_ref: ObjectRef | None = None
    label_quality_report_ref: ObjectRef
    label_quality_outcome: LabelQualityOutcomeV2
    label_evidence_class: LabelQualityEvidenceClassV2
    label_report_satisfies_sc_010: bool
    stability_report_ref: ObjectRef
    stability_outcome: ExternalRealTraceStabilityOutcomeV2
    unique_real_trace_count: int = Field(ge=0, le=100_000)
    observed_stability_threshold_met: bool
    evidence_class_counts: tuple[TypedAttribute, ...] = Field(
        min_length=4,
        max_length=4,
    )
    satisfies_sc_010: bool
    satisfies_sc_011: bool
    authorizes_sc_012: Literal[False] = False
    authorizes_sc_013: Literal[False] = False
    authorizes_sc_014: Literal[False] = False
    authorizes_sc_015: Literal[False] = False
    authorizes_approval: Literal[False] = False
    authorizes_attestation: Literal[False] = False
    authorizes_production_release: Literal[False] = False
    policy_version: Literal["external-evidence/r8-10-v1"] = EXTERNAL_EVIDENCE_POLICY_VERSION

    @classmethod
    def create(
        cls,
        *,
        package_manifest_ref: ObjectRef,
        admission_report_ref: ObjectRef,
        requirement_spec_ref: ObjectRef,
        observation_summary_ref: ObjectRef,
        freeze_result_ref: ObjectRef,
        dataset_manifest_ref: ObjectRef | None,
        label_quality_report_ref: ObjectRef,
        label_quality_outcome: LabelQualityOutcomeV2,
        label_evidence_class: LabelQualityEvidenceClassV2,
        label_report_satisfies_sc_010: bool,
        stability_report_ref: ObjectRef,
        stability_outcome: ExternalRealTraceStabilityOutcomeV2,
        unique_real_trace_count: int,
        observed_stability_threshold_met: bool,
        evidence_class_counts: tuple[TypedAttribute, ...],
        audit: ContractAudit,
    ) -> ExternalSc010Sc011EvidenceIndexV2:
        satisfies_sc_010 = _satisfies_sc_010(
            label_quality_outcome=label_quality_outcome,
            label_evidence_class=label_evidence_class,
            label_report_satisfies_sc_010=label_report_satisfies_sc_010,
            dataset_manifest_ref=dataset_manifest_ref,
        )
        satisfies_sc_011 = _satisfies_sc_011(
            stability_outcome=stability_outcome,
            unique_real_trace_count=unique_real_trace_count,
            observed_stability_threshold_met=(observed_stability_threshold_met),
        )
        return cls._create(
            audit=audit,
            values={
                "package_manifest_ref": package_manifest_ref,
                "admission_report_ref": admission_report_ref,
                "requirement_spec_ref": requirement_spec_ref,
                "observation_summary_ref": observation_summary_ref,
                "freeze_result_ref": freeze_result_ref,
                "dataset_manifest_ref": dataset_manifest_ref,
                "label_quality_report_ref": label_quality_report_ref,
                "label_quality_outcome": label_quality_outcome,
                "label_evidence_class": label_evidence_class,
                "label_report_satisfies_sc_010": (label_report_satisfies_sc_010),
                "stability_report_ref": stability_report_ref,
                "stability_outcome": stability_outcome,
                "unique_real_trace_count": unique_real_trace_count,
                "observed_stability_threshold_met": (observed_stability_threshold_met),
                "evidence_class_counts": _sorted_attributes(evidence_class_counts),
                "satisfies_sc_010": satisfies_sc_010,
                "satisfies_sc_011": satisfies_sc_011,
                "authorizes_sc_012": False,
                "authorizes_sc_013": False,
                "authorizes_sc_014": False,
                "authorizes_sc_015": False,
                "authorizes_approval": False,
                "authorizes_attestation": False,
                "authorizes_production_release": False,
                "policy_version": EXTERNAL_EVIDENCE_POLICY_VERSION,
            },
        )

    @model_validator(mode="after")
    def validate_index(self) -> Self:
        _require_ref(
            self.package_manifest_ref,
            "external-evidence-package-manifest",
            "v2",
            "package_manifest_ref",
        )
        _require_ref(
            self.admission_report_ref,
            "external-evidence-admission-report",
            "v2",
            "admission_report_ref",
        )
        _require_ref(
            self.requirement_spec_ref,
            "evaluation-requirement-spec",
            "v2",
            "requirement_spec_ref",
        )
        _require_ref(
            self.observation_summary_ref,
            "external-label-observation-summary",
            "v2",
            "observation_summary_ref",
        )
        _require_ref(
            self.freeze_result_ref,
            "independent-label-test-set-freeze-result",
            "v2",
            "freeze_result_ref",
        )
        if self.dataset_manifest_ref is not None:
            _require_ref(
                self.dataset_manifest_ref,
                "independent-label-test-set-manifest",
                "v2",
                "dataset_manifest_ref",
            )
        _require_ref(
            self.label_quality_report_ref,
            "label-quality-evaluation-report",
            "v2",
            "label_quality_report_ref",
        )
        _require_ref(
            self.stability_report_ref,
            "external-real-trace-stability-report",
            "v2",
            "stability_report_ref",
        )
        _require_attribute_counts(
            self.evidence_class_counts,
            expected_keys=_EVIDENCE_CLASS_COUNT_KEYS,
            expected_total=None,
            label="evidence_class_counts",
        )
        counts = {
            value.key: _attribute_count(value, "evidence_class_counts")
            for value in self.evidence_class_counts
        }
        if counts["REAL"] != self.unique_real_trace_count:
            raise ValueError("real evidence count differs from stability count")
        expected_sc_010 = _satisfies_sc_010(
            label_quality_outcome=self.label_quality_outcome,
            label_evidence_class=self.label_evidence_class,
            label_report_satisfies_sc_010=(self.label_report_satisfies_sc_010),
            dataset_manifest_ref=self.dataset_manifest_ref,
        )
        expected_sc_011 = _satisfies_sc_011(
            stability_outcome=self.stability_outcome,
            unique_real_trace_count=self.unique_real_trace_count,
            observed_stability_threshold_met=(self.observed_stability_threshold_met),
        )
        if self.satisfies_sc_010 != expected_sc_010 or self.satisfies_sc_011 != expected_sc_011:
            raise ValueError("external evidence SC predicates are stale")
        if self.label_report_satisfies_sc_010 and not expected_sc_010:
            raise ValueError("label report SC-010 claim lacks production authority")
        return self


def _satisfies_sc_010(
    *,
    label_quality_outcome: LabelQualityOutcomeV2,
    label_evidence_class: LabelQualityEvidenceClassV2,
    label_report_satisfies_sc_010: bool,
    dataset_manifest_ref: ObjectRef | None,
) -> bool:
    return (
        label_quality_outcome is LabelQualityOutcomeV2.PASSED
        and label_evidence_class is LabelQualityEvidenceClassV2.PRODUCTION_INDEPENDENT_TEST
        and label_report_satisfies_sc_010
        and dataset_manifest_ref is not None
    )


def _satisfies_sc_011(
    *,
    stability_outcome: ExternalRealTraceStabilityOutcomeV2,
    unique_real_trace_count: int,
    observed_stability_threshold_met: bool,
) -> bool:
    return (
        stability_outcome is ExternalRealTraceStabilityOutcomeV2.PASSED
        and unique_real_trace_count >= 100
        and observed_stability_threshold_met
    )


def _carried_sha256(
    value: _ExternalEvidenceObjectV2,
) -> str:
    payload = value.model_dump(
        mode="python",
        exclude={
            "schema_version",
            "object_id",
            "object_sha256",
            "audit",
        },
    )
    return hashlib.sha256(
        json.dumps(
            canonical_value_v2(payload),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _collect_refs(
    value: ContractModelV2,
) -> tuple[ObjectRef, ...]:
    refs: list[ObjectRef] = []

    def visit(item: object) -> None:
        if isinstance(item, ObjectRef):
            refs.append(item)
        elif isinstance(item, ContractModelV2):
            for field_name in type(item).model_fields:
                if field_name in {
                    "audit",
                    "object_id",
                    "object_sha256",
                }:
                    continue
                visit(getattr(item, field_name))
        elif isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, tuple | list | set | frozenset):
            for child in item:
                visit(child)

    visit(value)
    return _sorted_refs(tuple(refs))


def _audit_with_refs(
    audit: ContractAudit,
    refs: tuple[ObjectRef, ...],
) -> ContractAudit:
    return ContractAudit(
        created_at=audit.created_at,
        created_by=audit.created_by,
        governing_versions=audit.governing_versions,
        input_refs=refs,
    )


def _require_ref(
    value: ObjectRef,
    object_type: str,
    object_version: str,
    label: str,
) -> None:
    if value.object_type != object_type or value.object_version != object_version:
        raise ValueError(f"{label} must reference {object_type}/{object_version}")


def _require_safe_refs(
    refs: tuple[ObjectRef, ...],
) -> None:
    for ref in refs:
        rendered = f"{ref.object_type} {ref.object_id}".casefold()
        if any(marker in rendered for marker in _DENIED_REF_MARKERS):
            raise ValueError("external evidence refs contain sensitive material")


def _require_sorted_unique_refs(
    values: tuple[ObjectRef, ...],
    label: str,
) -> None:
    if values != _sorted_refs(values):
        raise ValueError(f"{label} must be sorted and unique")


def _sorted_refs(
    values: tuple[ObjectRef, ...],
) -> tuple[ObjectRef, ...]:
    return tuple(sorted(set(values), key=_ref_key))


def _ref_key(
    value: ObjectRef,
) -> tuple[str, str, str, str]:
    return (
        value.object_type,
        value.object_id,
        value.object_version,
        value.object_sha256,
    )


def _sorted_attributes(
    values: tuple[TypedAttribute, ...],
) -> tuple[TypedAttribute, ...]:
    return tuple(sorted(values, key=lambda value: value.key))


def _attribute_count(
    value: TypedAttribute,
    label: str,
) -> int:
    if isinstance(value.value, bool) or not isinstance(value.value, int) or value.value < 0:
        raise ValueError(f"{label} values must be non-negative integers")
    return value.value


def _require_attribute_counts(
    values: tuple[TypedAttribute, ...],
    *,
    expected_keys: tuple[str, ...],
    expected_total: int | None,
    label: str,
) -> None:
    keys = tuple(value.key for value in values)
    if keys != expected_keys or len(keys) != len(set(keys)):
        raise ValueError(f"{label} keys differ from the closed inventory")
    total = sum(_attribute_count(value, label) for value in values)
    if expected_total is not None and total != expected_total:
        raise ValueError(f"{label} counts do not match the source total")


__all__ = [
    "EXTERNAL_EVIDENCE_POLICY_VERSION",
    "ExternalEvidenceAdmissionOutcomeV2",
    "ExternalEvidenceAdmissionReasonV2",
    "ExternalEvidenceAdmissionReportV2",
    "ExternalEvidencePackageManifestV2",
    "ExternalLabelObservationSummaryV2",
    "ExternalReferenceAuthoringModeV2",
    "ExternalSc010Sc011EvidenceIndexV2",
]
