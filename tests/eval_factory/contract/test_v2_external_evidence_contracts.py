from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from eval_factory.contracts.core import (
    ContractAudit,
    ObjectRef,
    TypedAttribute,
    VersionBinding,
)
from eval_factory.contracts.external_evidence_v2 import (
    EXTERNAL_EVIDENCE_POLICY_VERSION,
    ExternalEvidenceAdmissionOutcomeV2,
    ExternalEvidenceAdmissionReasonV2,
    ExternalEvidenceAdmissionReportV2,
    ExternalEvidencePackageManifestV2,
    ExternalLabelObservationSummaryV2,
    ExternalReferenceAuthoringModeV2,
    ExternalSc010Sc011EvidenceIndexV2,
)
from eval_factory.contracts.external_stability_v2 import (
    ExternalRealTraceStabilityOutcomeV2,
)
from eval_factory.contracts.label_quality_v2 import (
    LabelQualityEvidenceClassV2,
    LabelQualityOutcomeV2,
)

NOW = datetime(2026, 8, 9, tzinfo=UTC)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _ref(
    object_type: str,
    suffix: str,
    *,
    version: str = "v2",
) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://r8-10/{suffix}",
        object_version=version,
        object_sha256=_digest(f"{object_type}:{suffix}:{version}"),
    )


def _audit(
    *,
    created_at: datetime = NOW,
) -> ContractAudit:
    return ContractAudit(
        created_at=created_at,
        created_by="r8-10-contract-test",
        governing_versions=(
            VersionBinding(
                component="external-evidence",
                version=EXTERNAL_EVIDENCE_POLICY_VERSION,
            ),
        ),
    )


def _label_refs() -> tuple[ObjectRef, ...]:
    return (
        _ref("label-spec", "powershell"),
        _ref("label-spec", "recovery"),
        _ref("label-spec", "search"),
    )


def _runtime_counts() -> tuple[TypedAttribute, ...]:
    return (
        TypedAttribute(key="claude_code", value=499),
        TypedAttribute(key="codex", value=128),
        TypedAttribute(key="hermes", value=147),
    )


def _count(value: TypedAttribute) -> int:
    assert isinstance(value.value, int)
    assert not isinstance(value.value, bool)
    return value.value


def _package() -> ExternalEvidencePackageManifestV2:
    inventory_sha256 = "f61c4b0cbe7f7f7aa99df991ffc87a67e865d918df922402276523249da8d2cc"
    return ExternalEvidencePackageManifestV2.create(
        requirement_source_ref=_ref(
            "evaluation-requirement-source",
            "requirement",
        ),
        requirement_spec_ref=_ref(
            "evaluation-requirement-spec",
            "requirement",
        ),
        source_authorization_ref=_ref(
            "external-source-authorization",
            "sources",
        ),
        source_population_ref=_ref(
            "external-source-population",
            "sources",
        ),
        inventory_commitment_ref=ObjectRef(
            object_type="external-corpus-inventory",
            object_id="external-corpus-inventory://r8-10/sources",
            object_version="private-v1",
            object_sha256=inventory_sha256,
        ),
        train_partition_ref=_ref(
            "external-partition-manifest",
            "train",
        ),
        development_partition_ref=_ref(
            "external-partition-manifest",
            "development",
        ),
        label_spec_refs=_label_refs(),
        reference_policy_ref=_ref(
            "external-reference-authoring-policy",
            "reference",
        ),
        observation_policy_ref=_ref(
            "external-observation-policy",
            "observation",
        ),
        adapter_name="runtime_snapshot_v1",
        adapter_version="1.0.0",
        source_count=774,
        unique_trace_count=774,
        unique_raw_hash_count=774,
        total_raw_bytes=207_506_845,
        runtime_counts=_runtime_counts(),
        inventory_sha256=inventory_sha256,
        audit=_audit(),
    )


def _curated_package() -> ExternalEvidencePackageManifestV2:
    inventory_sha256 = _digest("curated-inventory")
    return ExternalEvidencePackageManifestV2.create(
        requirement_source_ref=_ref(
            "evaluation-requirement-source",
            "requirement",
        ),
        requirement_spec_ref=_ref(
            "evaluation-requirement-spec",
            "requirement",
        ),
        source_authorization_ref=_ref(
            "external-source-authorization",
            "curated-sources",
        ),
        source_population_ref=_ref(
            "external-source-population",
            "curated-sources",
        ),
        inventory_commitment_ref=ObjectRef(
            object_type="external-corpus-inventory",
            object_id="external-corpus-inventory://r8-10/curated-sources",
            object_version="private-v1",
            object_sha256=inventory_sha256,
        ),
        train_partition_ref=_ref(
            "external-partition-manifest",
            "curated-train",
        ),
        development_partition_ref=_ref(
            "external-partition-manifest",
            "curated-development",
        ),
        label_spec_refs=_label_refs(),
        reference_policy_ref=_ref(
            "external-reference-authoring-policy",
            "reference",
        ),
        observation_policy_ref=_ref(
            "external-observation-policy",
            "observation",
        ),
        adapter_name="curated_trajectory_v1",
        adapter_version="1.0.0",
        source_count=158,
        unique_trace_count=158,
        unique_raw_hash_count=158,
        total_raw_bytes=74_710_670,
        runtime_counts=(TypedAttribute(key="claude_curated", value=158),),
        inventory_sha256=inventory_sha256,
        audit=_audit(),
    )


def _admission(
    package: ExternalEvidencePackageManifestV2 | None = None,
) -> ExternalEvidenceAdmissionReportV2:
    active = package or _package()
    return ExternalEvidenceAdmissionReportV2.create(
        package_manifest_ref=active.to_ref(),
        material_closure_ref=_ref(
            "external-evidence-material-closure",
            "admitted",
            version="private-v1",
        ),
        outcome=ExternalEvidenceAdmissionOutcomeV2.ADMITTED,
        reason_codes=(ExternalEvidenceAdmissionReasonV2.NONE,),
        source_count=774,
        admitted_source_count=774,
        rejected_source_count=0,
        unique_trace_count=774,
        unique_raw_hash_count=774,
        duplicate_source_count=0,
        overlap_source_count=0,
        total_raw_bytes=207_506_845,
        manifest_verified=True,
        material_verified=True,
        audit=_audit(),
    )


def _observations(
    package: ExternalEvidencePackageManifestV2 | None = None,
) -> ExternalLabelObservationSummaryV2:
    active = package or _package()
    return ExternalLabelObservationSummaryV2.create(
        package_manifest_ref=active.to_ref(),
        source_population_ref=active.source_population_ref,
        reference_closure_ref=_ref(
            "external-reference-set",
            "reference",
            version="private-v1",
        ),
        observation_closure_ref=_ref(
            "blind-label-observation-set",
            "observations",
            version="private-v1",
        ),
        label_spec_refs=active.label_spec_refs,
        reference_policy_ref=active.reference_policy_ref,
        observation_policy_ref=active.observation_policy_ref,
        authoring_mode=(ExternalReferenceAuthoringModeV2.AUTOMATIC_INDEPENDENT),
        reference_authority_committed=True,
        observation_reference_access_denied=True,
        total_pair_count=2_322,
        reference_ready_count=1_548,
        reference_abstained_count=0,
        reference_blocked_count=774,
        observation_complete_count=1_543,
        observation_missing_count=0,
        observation_blocked_count=779,
        audit=_audit(),
    )


def _index(
    *,
    label_outcome: LabelQualityOutcomeV2 = LabelQualityOutcomeV2.PASSED,
    label_satisfies_sc_010: bool = True,
    stability_outcome: ExternalRealTraceStabilityOutcomeV2 = (ExternalRealTraceStabilityOutcomeV2.PASSED),
    observed_threshold_met: bool = True,
    unique_real_trace_count: int = 774,
) -> ExternalSc010Sc011EvidenceIndexV2:
    package = _package()
    admission = _admission(package)
    observations = _observations(package)
    return ExternalSc010Sc011EvidenceIndexV2.create(
        package_manifest_ref=package.to_ref(),
        admission_report_ref=admission.to_ref(),
        requirement_spec_ref=package.requirement_spec_ref,
        observation_summary_ref=observations.to_ref(),
        freeze_result_ref=_ref(
            "independent-label-test-set-freeze-result",
            "freeze",
        ),
        dataset_manifest_ref=(
            _ref("independent-label-test-set-manifest", "dataset")
            if label_outcome is LabelQualityOutcomeV2.PASSED
            else None
        ),
        label_quality_report_ref=_ref(
            "label-quality-evaluation-report",
            "quality",
        ),
        label_quality_outcome=label_outcome,
        label_evidence_class=(LabelQualityEvidenceClassV2.PRODUCTION_INDEPENDENT_TEST),
        label_report_satisfies_sc_010=label_satisfies_sc_010,
        stability_report_ref=_ref(
            "external-real-trace-stability-report",
            "stability",
        ),
        stability_outcome=stability_outcome,
        unique_real_trace_count=unique_real_trace_count,
        observed_stability_threshold_met=observed_threshold_met,
        evidence_class_counts=(
            TypedAttribute(key="DEVELOPMENT", value=91),
            TypedAttribute(key="REAL", value=unique_real_trace_count),
            TypedAttribute(key="REPLAY", value=unique_real_trace_count),
            TypedAttribute(key="SYNTHETIC", value=0),
        ),
        audit=_audit(),
    )


def test_package_admission_and_observation_contracts_are_strict() -> None:
    package = _package()
    admission = _admission(package)
    observations = _observations(package)

    assert package.source_count == 774
    assert sum(_count(value) for value in package.runtime_counts) == 774
    assert admission.outcome is ExternalEvidenceAdmissionOutcomeV2.ADMITTED
    assert admission.reason_codes == (ExternalEvidenceAdmissionReasonV2.NONE,)
    assert observations.authoring_mode is (ExternalReferenceAuthoringModeV2.AUTOMATIC_INDEPENDENT)
    assert observations.reference_authority_committed is True
    assert observations.observation_reference_access_denied is True
    assert package.audit.input_refs
    assert admission.audit.input_refs
    assert observations.audit.input_refs
    with pytest.raises(ValidationError):
        package.source_count = 1


def test_package_accepts_curated_trajectory_adapter_without_runtime_aliases() -> None:
    package = _curated_package()

    assert package.adapter_name == "curated_trajectory_v1"
    assert package.runtime_counts == (TypedAttribute(key="claude_curated", value=158),)


def test_external_index_derives_only_sc_010_and_sc_011() -> None:
    value = _index()

    assert value.satisfies_sc_010 is True
    assert value.satisfies_sc_011 is True
    assert value.authorizes_sc_012 is False
    assert value.authorizes_sc_013 is False
    assert value.authorizes_sc_014 is False
    assert value.authorizes_sc_015 is False
    assert value.authorizes_approval is False
    assert value.authorizes_attestation is False
    assert value.authorizes_production_release is False


@pytest.mark.parametrize(
    ("updates", "sc_010", "sc_011"),
    [
        (
            {
                "label_outcome": LabelQualityOutcomeV2.FAILED,
                "label_satisfies_sc_010": False,
            },
            False,
            True,
        ),
        (
            {
                "stability_outcome": (ExternalRealTraceStabilityOutcomeV2.STABILITY_THRESHOLD_NOT_MET),
                "observed_threshold_met": False,
            },
            True,
            False,
        ),
        (
            {
                "unique_real_trace_count": 99,
                "stability_outcome": (ExternalRealTraceStabilityOutcomeV2.STATISTICAL_GATE_PENDING),
            },
            True,
            False,
        ),
    ],
)
def test_external_index_preserves_partial_evidence(
    updates: dict[str, object],
    sc_010: bool,
    sc_011: bool,
) -> None:
    value = _index(**updates)  # type: ignore[arg-type]

    assert value.satisfies_sc_010 is sc_010
    assert value.satisfies_sc_011 is sc_011
    assert value.authorizes_attestation is False


def test_contracts_reject_count_drift_authority_promotion_and_unknown_fields() -> None:
    package = _package()
    with pytest.raises(ValidationError, match="source counts"):
        ExternalEvidencePackageManifestV2.create(
            requirement_source_ref=package.requirement_source_ref,
            requirement_spec_ref=package.requirement_spec_ref,
            source_authorization_ref=package.source_authorization_ref,
            source_population_ref=package.source_population_ref,
            inventory_commitment_ref=package.inventory_commitment_ref,
            train_partition_ref=package.train_partition_ref,
            development_partition_ref=package.development_partition_ref,
            label_spec_refs=package.label_spec_refs,
            reference_policy_ref=package.reference_policy_ref,
            observation_policy_ref=package.observation_policy_ref,
            adapter_name=package.adapter_name,
            adapter_version=package.adapter_version,
            source_count=package.source_count,
            unique_trace_count=package.unique_trace_count,
            unique_raw_hash_count=773,
            total_raw_bytes=package.total_raw_bytes,
            runtime_counts=package.runtime_counts,
            inventory_sha256=package.inventory_sha256,
            audit=package.audit,
        )

    index = _index()
    promoted = index.model_dump(mode="python")
    promoted["authorizes_sc_012"] = True
    with pytest.raises(ValidationError):
        ExternalSc010Sc011EvidenceIndexV2.model_validate(promoted)

    unknown = _admission().model_dump(mode="python")
    unknown["source_members"] = ["forbidden"]
    with pytest.raises(ValidationError):
        ExternalEvidenceAdmissionReportV2.model_validate(unknown)


def test_identity_is_stable_across_audit_time_and_input_order() -> None:
    first = _package()
    second = ExternalEvidencePackageManifestV2.create(
        requirement_source_ref=first.requirement_source_ref,
        requirement_spec_ref=first.requirement_spec_ref,
        source_authorization_ref=first.source_authorization_ref,
        source_population_ref=first.source_population_ref,
        inventory_commitment_ref=first.inventory_commitment_ref,
        train_partition_ref=first.train_partition_ref,
        development_partition_ref=first.development_partition_ref,
        label_spec_refs=tuple(reversed(first.label_spec_refs)),
        reference_policy_ref=first.reference_policy_ref,
        observation_policy_ref=first.observation_policy_ref,
        adapter_name=first.adapter_name,
        adapter_version=first.adapter_version,
        source_count=first.source_count,
        unique_trace_count=first.unique_trace_count,
        unique_raw_hash_count=first.unique_raw_hash_count,
        total_raw_bytes=first.total_raw_bytes,
        runtime_counts=tuple(reversed(first.runtime_counts)),
        inventory_sha256=first.inventory_sha256,
        audit=_audit(created_at=datetime(2026, 8, 10, tzinfo=UTC)),
    )

    assert second.object_id == first.object_id
    assert second.object_sha256 == first.object_sha256
    assert second.canonical_sha256() != first.canonical_sha256()


def test_public_values_contain_no_member_truth_or_release_overclaim() -> None:
    serialized = (
        _package().model_dump_json()
        + _admission().model_dump_json()
        + _observations().model_dump_json()
        + _index().model_dump_json()
    ).casefold()
    for forbidden in (
        "source_trace_id",
        "expected_decision",
        "annotation_ref",
        "physical_path",
        "credential",
        "final_output",
        "grader_rule",
        "hidden_condition",
        "model_payload",
        "exception_text",
    ):
        assert forbidden not in serialized
