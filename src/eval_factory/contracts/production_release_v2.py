from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from env_mock_agent.facade import (
    FacadeObjectRef,
    LHWorkspaceExportMemberV2,
    LHWorkspaceExportOutcomeV2,
    LHWorkspaceExportResultV2,
    lh_workspace_export_result_ref,
    validate_lh_workspace_export_result_identity,
)
from eval_factory.contracts.core import (
    ContractAudit,
    Identifier,
    ObjectRef,
    Sha256,
)
from eval_factory.contracts.core_v2 import ContractModelV2, canonical_value_v2
from eval_factory.contracts.orchestration import ItemStatus
from eval_factory.contracts.release import ReleaseChannel
from eval_factory.contracts.release_projection_v2 import (
    evaluation_item_v2_ref,
    release_decision_v2_ref,
    validate_evaluation_item_v2_identity,
    validate_release_decision_v2_identity,
)
from eval_factory.contracts.release_v2 import (
    EvaluationItemV2,
    ReleaseActionV2,
    ReleaseDecisionV2,
    ReleaseStateV2,
)

PRODUCTION_RELEASE_POLICY_VERSION: Literal["production-release/r8-09-v1"] = "production-release/r8-09-v1"
PRODUCTION_ATTESTATION_POLICY_VERSION: Literal["production-readiness-attestation/r8-08-v1"] = (
    "production-readiness-attestation/r8-08-v1"
)

_DENIED_REF_MARKERS = frozenset(
    {
        "approved-by",
        "credential",
        "final-answer",
        "final-output",
        "grader-rule",
        "hidden-condition",
        "private-reference",
        "proof-value",
        "raw-trace",
        "raw-traj",
        "secret",
        "store-path",
    }
)


class ProductionReleaseEvidenceClassV2(StrEnum):
    PRODUCTION_VERIFIED = "PRODUCTION_VERIFIED"
    MECHANISM_VALIDATION_ONLY = "MECHANISM_VALIDATION_ONLY"
    REPOSITORY_PENDING_ONLY = "REPOSITORY_PENDING_ONLY"


class ProductionReleaseOutcomeV2(StrEnum):
    PUBLISHED = "PUBLISHED"
    PRODUCTION_RELEASE_BLOCKED = "PRODUCTION_RELEASE_BLOCKED"


class ProductionReleaseReasonCodeV2(StrEnum):
    NONE = "NONE"
    ATTESTATION_PENDING = "ATTESTATION_PENDING"
    ATTESTATION_NOT_ISSUED = "ATTESTATION_NOT_ISSUED"
    ATTESTATION_NOT_CURRENT = "ATTESTATION_NOT_CURRENT"
    ATTESTATION_EXPIRED = "ATTESTATION_EXPIRED"
    MECHANISM_ONLY = "MECHANISM_ONLY"


class ProductionReleasePolicyV2(ContractModelV2):
    schema_version: Literal["eval-factory/production-release-policy/v2"] = (
        "eval-factory/production-release-policy/v2"
    )
    policy_id: Identifier
    evidence_class: ProductionReleaseEvidenceClassV2
    expected_system_version: str = Field(min_length=1, max_length=128)
    base_contract_manifest_ref: ObjectRef
    overlay_contract_manifest_ref: ObjectRef
    required_schema_manifest_refs: tuple[ObjectRef, ...] = Field(
        min_length=2,
        max_length=128,
    )
    baseline_attested_policy_refs: tuple[ObjectRef, ...] = Field(
        min_length=1,
        max_length=1_000,
    )
    required_attestation_policy_version: Literal["production-readiness-attestation/r8-08-v1"] = (
        PRODUCTION_ATTESTATION_POLICY_VERSION
    )
    release_profile_decision_ref: ObjectRef
    release_profile_decision_sha256: Sha256
    repository_evidence_ref: ObjectRef | None = None
    repository_attestation_result_ref: ObjectRef | None = None
    allowed_production_registry_ids: frozenset[Identifier] = Field(
        min_length=1,
        max_length=100,
    )
    forbidden_nonproduction_registry_ids: frozenset[Identifier] = Field(
        min_length=1,
        max_length=1_000,
    )
    allowed_profile: Literal["LH"] = "LH"
    allowed_profile_version: Literal["v1"] = "v1"
    max_items: int = Field(ge=1, le=100_000)
    max_workspace_members_per_item: int = Field(ge=1, le=100_000)
    max_workspace_bytes_per_item: int = Field(ge=1, le=5_000_000_000)
    max_manifest_refs: int = Field(ge=1, le=1_000_000)
    max_registry_bytes: int = Field(ge=1, le=1_000_000_000)
    max_report_bytes: int = Field(ge=1, le=100_000_000)
    policy_version: Literal["production-release/r8-09-v1"] = PRODUCTION_RELEASE_POLICY_VERSION
    policy_sha256: Sha256
    audit: ContractAudit

    @field_validator("evidence_class", mode="before")
    @classmethod
    def parse_evidence_class(
        cls,
        value: object,
    ) -> ProductionReleaseEvidenceClassV2:
        return _parse_enum(value, ProductionReleaseEvidenceClassV2, "evidence_class")

    @model_validator(mode="after")
    def validate_policy(self) -> Self:
        _require_ref(
            self.base_contract_manifest_ref,
            "contract-manifest",
            "v1",
            "base_contract_manifest_ref",
        )
        _require_ref(
            self.overlay_contract_manifest_ref,
            "contract-manifest",
            "v2",
            "overlay_contract_manifest_ref",
        )
        _require_sorted_unique_refs(
            self.required_schema_manifest_refs,
            "required_schema_manifest_refs",
        )
        _require_sorted_unique_refs(
            self.baseline_attested_policy_refs,
            "baseline_attested_policy_refs",
        )
        _require_ref(
            self.release_profile_decision_ref,
            "release-profile-decision",
            "v1",
            "release_profile_decision_ref",
        )
        if self.release_profile_decision_ref.object_sha256 != self.release_profile_decision_sha256:
            raise ValueError("release-profile decision digest differs from ref")
        pending_values = (
            self.repository_evidence_ref,
            self.repository_attestation_result_ref,
        )
        if self.evidence_class is ProductionReleaseEvidenceClassV2.REPOSITORY_PENDING_ONLY:
            if any(value is None for value in pending_values):
                raise ValueError("repository-pending policy requires exact pending refs")
            assert self.repository_evidence_ref is not None
            assert self.repository_attestation_result_ref is not None
            _require_ref(
                self.repository_evidence_ref,
                "production-release-repository-evidence",
                "json/v1",
                "repository_evidence_ref",
            )
            _require_ref(
                self.repository_attestation_result_ref,
                "production-readiness-attestation-result",
                "v2",
                "repository_attestation_result_ref",
            )
        elif any(value is not None for value in pending_values):
            raise ValueError("production-verified policy cannot bind repository pending refs")
        if self.allowed_production_registry_ids & self.forbidden_nonproduction_registry_ids:
            raise ValueError("production and non-production registries must be disjoint")
        if any("production" not in value.casefold() for value in self.allowed_production_registry_ids):
            raise ValueError("production registry IDs require an explicit namespace")
        refs = self.refs
        _validate_audit(self.audit, refs, "production release policy")
        _validate_identity(
            self.policy_id,
            self.policy_sha256,
            "production-release-policy",
            production_release_policy_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        evidence_class: ProductionReleaseEvidenceClassV2,
        expected_system_version: str,
        base_contract_manifest_ref: ObjectRef,
        overlay_contract_manifest_ref: ObjectRef,
        required_schema_manifest_refs: tuple[ObjectRef, ...],
        baseline_attested_policy_refs: tuple[ObjectRef, ...],
        required_attestation_policy_version: Literal["production-readiness-attestation/r8-08-v1"],
        release_profile_decision_ref: ObjectRef,
        release_profile_decision_sha256: str,
        repository_evidence_ref: ObjectRef | None,
        repository_attestation_result_ref: ObjectRef | None,
        allowed_production_registry_ids: frozenset[str],
        forbidden_nonproduction_registry_ids: frozenset[str],
        max_items: int,
        max_workspace_members_per_item: int,
        max_workspace_bytes_per_item: int,
        max_manifest_refs: int,
        max_registry_bytes: int,
        max_report_bytes: int,
        audit: ContractAudit,
    ) -> ProductionReleasePolicyV2:
        schemas = _sorted_refs(required_schema_manifest_refs)
        policies = _sorted_refs(baseline_attested_policy_refs)
        refs = (
            base_contract_manifest_ref,
            overlay_contract_manifest_ref,
            *schemas,
            *policies,
            release_profile_decision_ref,
            *((repository_evidence_ref,) if repository_evidence_ref else ()),
            *((repository_attestation_result_ref,) if repository_attestation_result_ref else ()),
        )
        value = cls(
            policy_id="production-release-policy://pending",
            evidence_class=evidence_class,
            expected_system_version=expected_system_version,
            base_contract_manifest_ref=base_contract_manifest_ref,
            overlay_contract_manifest_ref=overlay_contract_manifest_ref,
            required_schema_manifest_refs=schemas,
            baseline_attested_policy_refs=policies,
            required_attestation_policy_version=required_attestation_policy_version,
            release_profile_decision_ref=release_profile_decision_ref,
            release_profile_decision_sha256=release_profile_decision_sha256,
            repository_evidence_ref=repository_evidence_ref,
            repository_attestation_result_ref=repository_attestation_result_ref,
            allowed_production_registry_ids=allowed_production_registry_ids,
            forbidden_nonproduction_registry_ids=(forbidden_nonproduction_registry_ids),
            max_items=max_items,
            max_workspace_members_per_item=max_workspace_members_per_item,
            max_workspace_bytes_per_item=max_workspace_bytes_per_item,
            max_manifest_refs=max_manifest_refs,
            max_registry_bytes=max_registry_bytes,
            max_report_bytes=max_report_bytes,
            policy_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        return _finalize(
            value,
            "policy_id",
            "policy_sha256",
            "production-release-policy",
            production_release_policy_v2_carried_sha256(value),
        )

    @property
    def refs(self) -> tuple[ObjectRef, ...]:
        return (
            self.base_contract_manifest_ref,
            self.overlay_contract_manifest_ref,
            *self.required_schema_manifest_refs,
            *self.baseline_attested_policy_refs,
            self.release_profile_decision_ref,
            *((self.repository_evidence_ref,) if self.repository_evidence_ref else ()),
            *((self.repository_attestation_result_ref,) if self.repository_attestation_result_ref else ()),
        )

    def to_ref(self) -> ObjectRef:
        return production_release_policy_v2_ref(self)


class ProductionAttestationAuthorityV2(ContractModelV2):
    schema_version: Literal["eval-factory/production-attestation-authority/v2"] = (
        "eval-factory/production-attestation-authority/v2"
    )
    authority_id: Identifier
    attestation_result_ref: ObjectRef
    current_projection_ref: ObjectRef
    frozen_attestation_ref: ObjectRef
    version_set_ref: ObjectRef
    prerequisite_ref: ObjectRef
    attestation_series_id: Identifier
    attestation_version: int = Field(ge=1, le=1_000_000)
    valid_from: datetime
    valid_until: datetime
    verified_at: datetime
    authority_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_authority(self) -> Self:
        for ref, object_type, version, label in (
            (
                self.attestation_result_ref,
                "production-readiness-attestation-result",
                "v2",
                "attestation_result_ref",
            ),
            (
                self.current_projection_ref,
                "production-readiness-attestation-projection",
                "v2",
                "current_projection_ref",
            ),
            (
                self.frozen_attestation_ref,
                "production-readiness-attestation",
                "v1",
                "frozen_attestation_ref",
            ),
            (
                self.version_set_ref,
                "production-readiness-version-set",
                "v2",
                "version_set_ref",
            ),
            (
                self.prerequisite_ref,
                "production-readiness-attestation-prerequisite",
                "v2",
                "prerequisite_ref",
            ),
        ):
            _require_ref(ref, object_type, version, label)
        for value, label in (
            (self.valid_from, "valid_from"),
            (self.valid_until, "valid_until"),
            (self.verified_at, "verified_at"),
        ):
            _require_aware(value, label)
        if self.valid_until <= self.valid_from or not self.valid_from <= self.verified_at < self.valid_until:
            raise ValueError("production attestation authority window is invalid")
        refs = self.refs
        _validate_audit(self.audit, refs, "production attestation authority")
        _validate_identity(
            self.authority_id,
            self.authority_sha256,
            "production-attestation-authority",
            production_attestation_authority_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        attestation_result_ref: ObjectRef,
        current_projection_ref: ObjectRef,
        frozen_attestation_ref: ObjectRef,
        version_set_ref: ObjectRef,
        prerequisite_ref: ObjectRef,
        attestation_series_id: str,
        attestation_version: int,
        valid_from: datetime,
        valid_until: datetime,
        verified_at: datetime,
        audit: ContractAudit,
    ) -> ProductionAttestationAuthorityV2:
        refs = (
            attestation_result_ref,
            current_projection_ref,
            frozen_attestation_ref,
            version_set_ref,
            prerequisite_ref,
        )
        value = cls(
            authority_id="production-attestation-authority://pending",
            attestation_result_ref=attestation_result_ref,
            current_projection_ref=current_projection_ref,
            frozen_attestation_ref=frozen_attestation_ref,
            version_set_ref=version_set_ref,
            prerequisite_ref=prerequisite_ref,
            attestation_series_id=attestation_series_id,
            attestation_version=attestation_version,
            valid_from=valid_from,
            valid_until=valid_until,
            verified_at=verified_at,
            authority_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        return _finalize(
            value,
            "authority_id",
            "authority_sha256",
            "production-attestation-authority",
            production_attestation_authority_v2_carried_sha256(value),
        )

    @property
    def refs(self) -> tuple[ObjectRef, ...]:
        return (
            self.attestation_result_ref,
            self.current_projection_ref,
            self.frozen_attestation_ref,
            self.version_set_ref,
            self.prerequisite_ref,
        )

    def to_ref(self) -> ObjectRef:
        return production_attestation_authority_v2_ref(self)


class ProductionReleaseItemManifestV2(ContractModelV2):
    schema_version: Literal["eval-factory/production-release-item-manifest/v2"] = (
        "eval-factory/production-release-item-manifest/v2"
    )
    item_manifest_id: Identifier
    job_id: Identifier
    item_id: Identifier
    approved_release_result_ref: ObjectRef
    approved_projection_ref: ObjectRef
    release_subject_ref: ObjectRef
    predecessor_decision_ref: ObjectRef
    predecessor_evaluation_item_ref: ObjectRef
    source_nonproduction_publication_ref: ObjectRef | None = None
    query_spec_ref: ObjectRef
    rubric_set_ref: ObjectRef
    environment_spec_ref: ObjectRef
    provenance_manifest_ref: ObjectRef
    quality_report_ref: ObjectRef
    final_package_manifest_ref: ObjectRef
    package_sha256: Sha256
    query_yaml_sha256: Sha256
    rubrics_json_sha256: Sha256
    expected_workspace_members: tuple[LHWorkspaceExportMemberV2, ...]
    attestation_authority_ref: ObjectRef
    channel: Literal[ReleaseChannel.PRODUCTION] = ReleaseChannel.PRODUCTION
    registry: Identifier
    export_profile: Literal["LH"] = "LH"
    export_profile_version: Literal["v1"] = "v1"
    policy_ref: ObjectRef
    policy_version: Literal["production-release/r8-09-v1"] = PRODUCTION_RELEASE_POLICY_VERSION
    item_manifest_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        for ref, object_type, version, label in (
            (
                self.approved_release_result_ref,
                "release-projection-result",
                "v2",
                "approved_release_result_ref",
            ),
            (
                self.approved_projection_ref,
                "item-release-projection",
                "v2",
                "approved_projection_ref",
            ),
            (
                self.release_subject_ref,
                "evaluation-item-release-subject",
                "v2",
                "release_subject_ref",
            ),
            (
                self.predecessor_decision_ref,
                "release-decision",
                "v2",
                "predecessor_decision_ref",
            ),
            (
                self.predecessor_evaluation_item_ref,
                "evaluation-item",
                "v2",
                "predecessor_evaluation_item_ref",
            ),
            (self.query_spec_ref, "query-spec", "v2", "query_spec_ref"),
            (self.rubric_set_ref, "rubric-set", "v2", "rubric_set_ref"),
            (
                self.environment_spec_ref,
                "environment-spec",
                "v2",
                "environment_spec_ref",
            ),
            (
                self.provenance_manifest_ref,
                "provenance-manifest",
                "v2",
                "provenance_manifest_ref",
            ),
            (
                self.quality_report_ref,
                "quality-report",
                "v2",
                "quality_report_ref",
            ),
            (
                self.final_package_manifest_ref,
                "final-package-manifest",
                "v2",
                "final_package_manifest_ref",
            ),
            (
                self.attestation_authority_ref,
                "production-attestation-authority",
                "v2",
                "attestation_authority_ref",
            ),
            (
                self.policy_ref,
                "production-release-policy",
                "v2",
                "policy_ref",
            ),
        ):
            _require_ref(ref, object_type, version, label)
        if self.source_nonproduction_publication_ref is not None:
            _require_ref(
                self.source_nonproduction_publication_ref,
                "release-publication-result",
                "v2",
                "source_nonproduction_publication_ref",
            )
        _require_sorted_unique_members(self.expected_workspace_members)
        refs = self.refs
        _validate_audit(self.audit, refs, "production release item manifest")
        _validate_identity(
            self.item_manifest_id,
            self.item_manifest_sha256,
            "production-release-item-manifest",
            production_release_item_manifest_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        job_id: str,
        item_id: str,
        approved_release_result_ref: ObjectRef,
        approved_projection_ref: ObjectRef,
        release_subject_ref: ObjectRef,
        predecessor_decision_ref: ObjectRef,
        predecessor_evaluation_item_ref: ObjectRef,
        source_nonproduction_publication_ref: ObjectRef | None,
        query_spec_ref: ObjectRef,
        rubric_set_ref: ObjectRef,
        environment_spec_ref: ObjectRef,
        provenance_manifest_ref: ObjectRef,
        quality_report_ref: ObjectRef,
        final_package_manifest_ref: ObjectRef,
        package_sha256: str,
        query_yaml_sha256: str,
        rubrics_json_sha256: str,
        expected_workspace_members: tuple[LHWorkspaceExportMemberV2, ...],
        attestation_authority_ref: ObjectRef,
        registry: str,
        policy_ref: ObjectRef,
        audit: ContractAudit,
    ) -> ProductionReleaseItemManifestV2:
        members = _sorted_members(expected_workspace_members)
        refs = (
            approved_release_result_ref,
            approved_projection_ref,
            release_subject_ref,
            predecessor_decision_ref,
            predecessor_evaluation_item_ref,
            *((source_nonproduction_publication_ref,) if source_nonproduction_publication_ref else ()),
            query_spec_ref,
            rubric_set_ref,
            environment_spec_ref,
            provenance_manifest_ref,
            quality_report_ref,
            final_package_manifest_ref,
            attestation_authority_ref,
            policy_ref,
        )
        value = cls(
            item_manifest_id="production-release-item-manifest://pending",
            job_id=job_id,
            item_id=item_id,
            approved_release_result_ref=approved_release_result_ref,
            approved_projection_ref=approved_projection_ref,
            release_subject_ref=release_subject_ref,
            predecessor_decision_ref=predecessor_decision_ref,
            predecessor_evaluation_item_ref=predecessor_evaluation_item_ref,
            source_nonproduction_publication_ref=(source_nonproduction_publication_ref),
            query_spec_ref=query_spec_ref,
            rubric_set_ref=rubric_set_ref,
            environment_spec_ref=environment_spec_ref,
            provenance_manifest_ref=provenance_manifest_ref,
            quality_report_ref=quality_report_ref,
            final_package_manifest_ref=final_package_manifest_ref,
            package_sha256=package_sha256,
            query_yaml_sha256=query_yaml_sha256,
            rubrics_json_sha256=rubrics_json_sha256,
            expected_workspace_members=members,
            attestation_authority_ref=attestation_authority_ref,
            registry=registry,
            policy_ref=policy_ref,
            item_manifest_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        return _finalize(
            value,
            "item_manifest_id",
            "item_manifest_sha256",
            "production-release-item-manifest",
            production_release_item_manifest_v2_carried_sha256(value),
        )

    @property
    def refs(self) -> tuple[ObjectRef, ...]:
        return (
            self.approved_release_result_ref,
            self.approved_projection_ref,
            self.release_subject_ref,
            self.predecessor_decision_ref,
            self.predecessor_evaluation_item_ref,
            *(
                (self.source_nonproduction_publication_ref,)
                if self.source_nonproduction_publication_ref
                else ()
            ),
            self.query_spec_ref,
            self.rubric_set_ref,
            self.environment_spec_ref,
            self.provenance_manifest_ref,
            self.quality_report_ref,
            self.final_package_manifest_ref,
            self.attestation_authority_ref,
            self.policy_ref,
        )

    def to_ref(self) -> ObjectRef:
        return production_release_item_manifest_v2_ref(self)


class ProductionReleaseManifestV2(ContractModelV2):
    schema_version: Literal["eval-factory/production-release-manifest/v2"] = (
        "eval-factory/production-release-manifest/v2"
    )
    manifest_id: Identifier
    job_id: Identifier
    item_ids: tuple[Identifier, ...] = Field(min_length=1)
    item_manifests: tuple[ProductionReleaseItemManifestV2, ...] = Field(min_length=1)
    item_manifest_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    base_contract_manifest_ref: ObjectRef
    overlay_contract_manifest_ref: ObjectRef
    release_profile_decision_ref: ObjectRef
    attestation_authority_ref: ObjectRef
    channel: Literal[ReleaseChannel.PRODUCTION] = ReleaseChannel.PRODUCTION
    registry: Identifier
    export_profile: Literal["LH"] = "LH"
    export_profile_version: Literal["v1"] = "v1"
    policy_ref: ObjectRef
    policy_version: Literal["production-release/r8-09-v1"] = PRODUCTION_RELEASE_POLICY_VERSION
    manifest_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        expected = tuple(sorted(self.item_manifests, key=lambda value: value.item_id))
        if expected != self.item_manifests:
            raise ValueError("production item manifests must be canonical")
        if self.item_ids != tuple(value.item_id for value in expected):
            raise ValueError("production item IDs differ from item manifests")
        expected_refs = tuple(value.to_ref() for value in expected)
        if self.item_manifest_refs != expected_refs:
            raise ValueError("production item manifest refs are not exact")
        if len(set(self.item_ids)) != len(self.item_ids):
            raise ValueError("production item IDs must be unique")
        for item in expected:
            if (
                item.job_id != self.job_id
                or item.registry != self.registry
                or item.policy_ref != self.policy_ref
                or item.attestation_authority_ref != self.attestation_authority_ref
            ):
                raise ValueError("production item manifest authority is mixed")
        for ref, object_type, version, label in (
            (
                self.base_contract_manifest_ref,
                "contract-manifest",
                "v1",
                "base_contract_manifest_ref",
            ),
            (
                self.overlay_contract_manifest_ref,
                "contract-manifest",
                "v2",
                "overlay_contract_manifest_ref",
            ),
            (
                self.release_profile_decision_ref,
                "release-profile-decision",
                "v1",
                "release_profile_decision_ref",
            ),
            (
                self.attestation_authority_ref,
                "production-attestation-authority",
                "v2",
                "attestation_authority_ref",
            ),
            (
                self.policy_ref,
                "production-release-policy",
                "v2",
                "policy_ref",
            ),
        ):
            _require_ref(ref, object_type, version, label)
        refs = self.refs
        _validate_audit(self.audit, refs, "production release manifest")
        _validate_identity(
            self.manifest_id,
            self.manifest_sha256,
            "production-release-manifest",
            production_release_manifest_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        job_id: str,
        item_manifests: tuple[ProductionReleaseItemManifestV2, ...],
        base_contract_manifest_ref: ObjectRef,
        overlay_contract_manifest_ref: ObjectRef,
        release_profile_decision_ref: ObjectRef,
        attestation_authority_ref: ObjectRef,
        registry: str,
        policy_ref: ObjectRef,
        audit: ContractAudit,
    ) -> ProductionReleaseManifestV2:
        items = tuple(sorted(item_manifests, key=lambda value: value.item_id))
        refs = (
            *(value.to_ref() for value in items),
            base_contract_manifest_ref,
            overlay_contract_manifest_ref,
            release_profile_decision_ref,
            attestation_authority_ref,
            policy_ref,
        )
        value = cls(
            manifest_id="production-release-manifest://pending",
            job_id=job_id,
            item_ids=tuple(item.item_id for item in items),
            item_manifests=items,
            item_manifest_refs=tuple(item.to_ref() for item in items),
            base_contract_manifest_ref=base_contract_manifest_ref,
            overlay_contract_manifest_ref=overlay_contract_manifest_ref,
            release_profile_decision_ref=release_profile_decision_ref,
            attestation_authority_ref=attestation_authority_ref,
            registry=registry,
            policy_ref=policy_ref,
            manifest_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        return _finalize(
            value,
            "manifest_id",
            "manifest_sha256",
            "production-release-manifest",
            production_release_manifest_v2_carried_sha256(value),
        )

    @property
    def refs(self) -> tuple[ObjectRef, ...]:
        return (
            *self.item_manifest_refs,
            self.base_contract_manifest_ref,
            self.overlay_contract_manifest_ref,
            self.release_profile_decision_ref,
            self.attestation_authority_ref,
            self.policy_ref,
        )

    def to_ref(self) -> ObjectRef:
        return production_release_manifest_v2_ref(self)


class ProductionLHExportReceiptV2(ContractModelV2):
    schema_version: Literal["eval-factory/production-lh-export-receipt/v2"] = (
        "eval-factory/production-lh-export-receipt/v2"
    )
    receipt_id: Identifier
    release_manifest_ref: ObjectRef
    item_manifest_ref: ObjectRef
    workspace_export_result: LHWorkspaceExportResultV2
    workspace_export_result_ref: ObjectRef
    query_yaml_sha256: Sha256
    rubrics_json_sha256: Sha256
    workspace_sha256: Sha256
    bundle_sha256: Sha256
    bundle_file_count: int = Field(ge=3, le=100_003)
    bundle_total_bytes: int = Field(ge=1, le=5_000_000_000)
    policy_ref: ObjectRef
    receipt_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_receipt(self) -> Self:
        validate_lh_workspace_export_result_identity(self.workspace_export_result)
        if self.workspace_export_result.outcome is not LHWorkspaceExportOutcomeV2.EXPORTED:
            raise ValueError("production receipt requires exported workspace")
        facade_ref = _object_ref(lh_workspace_export_result_ref(self.workspace_export_result))
        if (
            self.workspace_export_result_ref != facade_ref
            or self.workspace_sha256 != self.workspace_export_result.workspace_sha256
        ):
            raise ValueError("production workspace result is stale")
        for ref, object_type, version, label in (
            (
                self.release_manifest_ref,
                "production-release-manifest",
                "v2",
                "release_manifest_ref",
            ),
            (
                self.item_manifest_ref,
                "production-release-item-manifest",
                "v2",
                "item_manifest_ref",
            ),
            (
                self.workspace_export_result_ref,
                "lh-workspace-export-result",
                "v2",
                "workspace_export_result_ref",
            ),
            (
                self.policy_ref,
                "production-release-policy",
                "v2",
                "policy_ref",
            ),
        ):
            _require_ref(ref, object_type, version, label)
        refs = self.refs
        _validate_audit(self.audit, refs, "production LH export receipt")
        _validate_identity(
            self.receipt_id,
            self.receipt_sha256,
            "production-lh-export-receipt",
            production_lh_export_receipt_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        release_manifest_ref: ObjectRef,
        item_manifest_ref: ObjectRef,
        workspace_export_result: LHWorkspaceExportResultV2,
        query_yaml_sha256: str,
        rubrics_json_sha256: str,
        bundle_sha256: str,
        bundle_file_count: int,
        bundle_total_bytes: int,
        policy_ref: ObjectRef,
        audit: ContractAudit,
    ) -> ProductionLHExportReceiptV2:
        result_ref = _object_ref(lh_workspace_export_result_ref(workspace_export_result))
        refs = (
            release_manifest_ref,
            item_manifest_ref,
            result_ref,
            policy_ref,
        )
        value = cls(
            receipt_id="production-lh-export-receipt://pending",
            release_manifest_ref=release_manifest_ref,
            item_manifest_ref=item_manifest_ref,
            workspace_export_result=workspace_export_result,
            workspace_export_result_ref=result_ref,
            query_yaml_sha256=query_yaml_sha256,
            rubrics_json_sha256=rubrics_json_sha256,
            workspace_sha256=workspace_export_result.workspace_sha256 or "",
            bundle_sha256=bundle_sha256,
            bundle_file_count=bundle_file_count,
            bundle_total_bytes=bundle_total_bytes,
            policy_ref=policy_ref,
            receipt_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        return _finalize(
            value,
            "receipt_id",
            "receipt_sha256",
            "production-lh-export-receipt",
            production_lh_export_receipt_v2_carried_sha256(value),
        )

    @property
    def refs(self) -> tuple[ObjectRef, ...]:
        return (
            self.release_manifest_ref,
            self.item_manifest_ref,
            self.workspace_export_result_ref,
            self.policy_ref,
        )

    def to_ref(self) -> ObjectRef:
        return production_lh_export_receipt_v2_ref(self)


class ProductionRegistryEntryV2(ContractModelV2):
    schema_version: Literal["eval-factory/production-registry-entry/v2"] = (
        "eval-factory/production-registry-entry/v2"
    )
    registry_entry_id: Identifier
    release_manifest_ref: ObjectRef
    job_id: Identifier
    channel: Literal[ReleaseChannel.PRODUCTION] = ReleaseChannel.PRODUCTION
    registry: Identifier
    item_ids: tuple[Identifier, ...] = Field(min_length=1)
    item_manifest_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    export_receipt_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    published_release_decision_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    published_evaluation_item_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    bundle_sha256s: tuple[Sha256, ...] = Field(min_length=1)
    attestation_authority_ref: ObjectRef
    published_at: datetime
    policy_ref: ObjectRef
    entry_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_entry(self) -> Self:
        count = len(self.item_ids)
        if self.item_ids != tuple(sorted(set(self.item_ids))) or any(
            len(values) != count
            for values in (
                self.item_manifest_refs,
                self.export_receipt_refs,
                self.published_release_decision_refs,
                self.published_evaluation_item_refs,
                self.bundle_sha256s,
            )
        ):
            raise ValueError("production registry Item coverage is not exact")
        _require_ref(
            self.release_manifest_ref,
            "production-release-manifest",
            "v2",
            "release_manifest_ref",
        )
        for values, object_type, label in (
            (
                self.item_manifest_refs,
                "production-release-item-manifest",
                "item_manifest_refs",
            ),
            (
                self.export_receipt_refs,
                "production-lh-export-receipt",
                "export_receipt_refs",
            ),
            (
                self.published_release_decision_refs,
                "release-decision",
                "published_release_decision_refs",
            ),
            (
                self.published_evaluation_item_refs,
                "evaluation-item",
                "published_evaluation_item_refs",
            ),
        ):
            for ref in values:
                _require_ref(ref, object_type, "v2", label)
        _require_ref(
            self.attestation_authority_ref,
            "production-attestation-authority",
            "v2",
            "attestation_authority_ref",
        )
        _require_ref(
            self.policy_ref,
            "production-release-policy",
            "v2",
            "policy_ref",
        )
        _require_aware(self.published_at, "published_at")
        refs = self.refs
        _validate_audit(self.audit, refs, "production registry entry")
        _validate_identity(
            self.registry_entry_id,
            self.entry_sha256,
            "production-registry-entry",
            production_registry_entry_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        release_manifest_ref: ObjectRef,
        job_id: str,
        registry: str,
        item_ids: tuple[str, ...],
        item_manifest_refs: tuple[ObjectRef, ...],
        export_receipt_refs: tuple[ObjectRef, ...],
        published_release_decision_refs: tuple[ObjectRef, ...],
        published_evaluation_item_refs: tuple[ObjectRef, ...],
        bundle_sha256s: tuple[str, ...],
        attestation_authority_ref: ObjectRef,
        published_at: datetime,
        policy_ref: ObjectRef,
        audit: ContractAudit,
    ) -> ProductionRegistryEntryV2:
        refs = (
            release_manifest_ref,
            *item_manifest_refs,
            *export_receipt_refs,
            *published_release_decision_refs,
            *published_evaluation_item_refs,
            attestation_authority_ref,
            policy_ref,
        )
        value = cls(
            registry_entry_id="production-registry-entry://pending",
            release_manifest_ref=release_manifest_ref,
            job_id=job_id,
            registry=registry,
            item_ids=tuple(sorted(set(item_ids))),
            item_manifest_refs=item_manifest_refs,
            export_receipt_refs=export_receipt_refs,
            published_release_decision_refs=published_release_decision_refs,
            published_evaluation_item_refs=published_evaluation_item_refs,
            bundle_sha256s=bundle_sha256s,
            attestation_authority_ref=attestation_authority_ref,
            published_at=published_at,
            policy_ref=policy_ref,
            entry_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        return _finalize(
            value,
            "registry_entry_id",
            "entry_sha256",
            "production-registry-entry",
            production_registry_entry_v2_carried_sha256(value),
        )

    @property
    def refs(self) -> tuple[ObjectRef, ...]:
        return (
            self.release_manifest_ref,
            *self.item_manifest_refs,
            *self.export_receipt_refs,
            *self.published_release_decision_refs,
            *self.published_evaluation_item_refs,
            self.attestation_authority_ref,
            self.policy_ref,
        )

    def to_ref(self) -> ObjectRef:
        return production_registry_entry_v2_ref(self)


class ProductionPublishedItemProjectionV2(ContractModelV2):
    schema_version: Literal["eval-factory/production-published-item-projection/v2"] = (
        "eval-factory/production-published-item-projection/v2"
    )
    projection_id: Identifier
    job_id: Identifier
    item_id: Identifier
    chain_id: Identifier
    approved_result_ref: ObjectRef
    approved_projection_ref: ObjectRef
    source_nonproduction_publication_ref: ObjectRef | None = None
    release_manifest_ref: ObjectRef
    export_receipt_ref: ObjectRef
    registry_entry_ref: ObjectRef
    publish_decision_ref: ObjectRef
    published_evaluation_item_ref: ObjectRef
    attestation_authority_ref: ObjectRef
    item_status: Literal[ItemStatus.RELEASED] = ItemStatus.RELEASED
    policy_version: Literal["production-release/r8-09-v1"] = PRODUCTION_RELEASE_POLICY_VERSION
    projection_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_projection(self) -> Self:
        for ref, object_type, version, label in (
            (
                self.approved_result_ref,
                "release-projection-result",
                "v2",
                "approved_result_ref",
            ),
            (
                self.approved_projection_ref,
                "item-release-projection",
                "v2",
                "approved_projection_ref",
            ),
            (
                self.release_manifest_ref,
                "production-release-manifest",
                "v2",
                "release_manifest_ref",
            ),
            (
                self.export_receipt_ref,
                "production-lh-export-receipt",
                "v2",
                "export_receipt_ref",
            ),
            (
                self.registry_entry_ref,
                "production-registry-entry",
                "v2",
                "registry_entry_ref",
            ),
            (
                self.publish_decision_ref,
                "release-decision",
                "v2",
                "publish_decision_ref",
            ),
            (
                self.published_evaluation_item_ref,
                "evaluation-item",
                "v2",
                "published_evaluation_item_ref",
            ),
            (
                self.attestation_authority_ref,
                "production-attestation-authority",
                "v2",
                "attestation_authority_ref",
            ),
        ):
            _require_ref(ref, object_type, version, label)
        if self.source_nonproduction_publication_ref is not None:
            _require_ref(
                self.source_nonproduction_publication_ref,
                "release-publication-result",
                "v2",
                "source_nonproduction_publication_ref",
            )
        refs = self.refs
        _validate_audit(self.audit, refs, "production published Item projection")
        _validate_identity(
            self.projection_id,
            self.projection_sha256,
            "production-published-item-projection",
            production_published_item_projection_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        job_id: str,
        item_id: str,
        chain_id: str,
        approved_result_ref: ObjectRef,
        approved_projection_ref: ObjectRef,
        source_nonproduction_publication_ref: ObjectRef | None,
        release_manifest_ref: ObjectRef,
        export_receipt_ref: ObjectRef,
        registry_entry_ref: ObjectRef,
        publish_decision_ref: ObjectRef,
        published_evaluation_item_ref: ObjectRef,
        attestation_authority_ref: ObjectRef,
        audit: ContractAudit,
    ) -> ProductionPublishedItemProjectionV2:
        refs = (
            approved_result_ref,
            approved_projection_ref,
            *((source_nonproduction_publication_ref,) if source_nonproduction_publication_ref else ()),
            release_manifest_ref,
            export_receipt_ref,
            registry_entry_ref,
            publish_decision_ref,
            published_evaluation_item_ref,
            attestation_authority_ref,
        )
        value = cls(
            projection_id="production-published-item-projection://pending",
            job_id=job_id,
            item_id=item_id,
            chain_id=chain_id,
            approved_result_ref=approved_result_ref,
            approved_projection_ref=approved_projection_ref,
            source_nonproduction_publication_ref=(source_nonproduction_publication_ref),
            release_manifest_ref=release_manifest_ref,
            export_receipt_ref=export_receipt_ref,
            registry_entry_ref=registry_entry_ref,
            publish_decision_ref=publish_decision_ref,
            published_evaluation_item_ref=published_evaluation_item_ref,
            attestation_authority_ref=attestation_authority_ref,
            projection_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        return _finalize(
            value,
            "projection_id",
            "projection_sha256",
            "production-published-item-projection",
            production_published_item_projection_v2_carried_sha256(value),
        )

    @property
    def refs(self) -> tuple[ObjectRef, ...]:
        return (
            self.approved_result_ref,
            self.approved_projection_ref,
            *(
                (self.source_nonproduction_publication_ref,)
                if self.source_nonproduction_publication_ref
                else ()
            ),
            self.release_manifest_ref,
            self.export_receipt_ref,
            self.registry_entry_ref,
            self.publish_decision_ref,
            self.published_evaluation_item_ref,
            self.attestation_authority_ref,
        )

    def to_ref(self) -> ObjectRef:
        return production_published_item_projection_v2_ref(self)


class ProductionReleaseResultV2(ContractModelV2):
    schema_version: Literal["eval-factory/production-release-result/v2"] = (
        "eval-factory/production-release-result/v2"
    )
    result_id: Identifier
    evidence_class: ProductionReleaseEvidenceClassV2
    outcome: ProductionReleaseOutcomeV2
    reason_codes: tuple[ProductionReleaseReasonCodeV2, ...] = Field(min_length=1)
    policy_ref: ObjectRef
    repository_evidence_ref: ObjectRef | None = None
    attestation_authority: ProductionAttestationAuthorityV2 | None = None
    attestation_authority_ref: ObjectRef | None = None
    release_manifest: ProductionReleaseManifestV2 | None = None
    release_manifest_ref: ObjectRef | None = None
    export_receipts: tuple[ProductionLHExportReceiptV2, ...] = ()
    published_decisions: tuple[ReleaseDecisionV2, ...] = ()
    published_items: tuple[EvaluationItemV2, ...] = ()
    item_projections: tuple[ProductionPublishedItemProjectionV2, ...] = ()
    registry_entry: ProductionRegistryEntryV2 | None = None
    registry_entry_ref: ObjectRef | None = None
    approved_source_result_refs: tuple[ObjectRef, ...] = ()
    satisfies_sc_015: bool
    policy_version: Literal["production-release/r8-09-v1"] = PRODUCTION_RELEASE_POLICY_VERSION
    result_sha256: Sha256
    audit: ContractAudit

    @field_validator("evidence_class", mode="before")
    @classmethod
    def parse_evidence_class(
        cls,
        value: object,
    ) -> ProductionReleaseEvidenceClassV2:
        return _parse_enum(value, ProductionReleaseEvidenceClassV2, "evidence_class")

    @field_validator("outcome", mode="before")
    @classmethod
    def parse_outcome(cls, value: object) -> ProductionReleaseOutcomeV2:
        return _parse_enum(value, ProductionReleaseOutcomeV2, "outcome")

    @field_validator("reason_codes", mode="before")
    @classmethod
    def parse_reasons(
        cls,
        value: object,
    ) -> tuple[ProductionReleaseReasonCodeV2, ...]:
        if not isinstance(value, (tuple, list, set, frozenset)):
            raise TypeError("reason_codes must be a collection")
        return tuple(
            item if isinstance(item, ProductionReleaseReasonCodeV2) else ProductionReleaseReasonCodeV2(item)
            for item in value
        )

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        _require_ref(
            self.policy_ref,
            "production-release-policy",
            "v2",
            "policy_ref",
        )
        if len(set(self.reason_codes)) != len(self.reason_codes):
            raise ValueError("production release reason codes must be unique")
        published_values = (
            self.attestation_authority,
            self.attestation_authority_ref,
            self.release_manifest,
            self.release_manifest_ref,
            self.registry_entry,
            self.registry_entry_ref,
        )
        published_collections = (
            self.export_receipts,
            self.published_decisions,
            self.published_items,
            self.item_projections,
            self.approved_source_result_refs,
        )
        if self.outcome is ProductionReleaseOutcomeV2.PRODUCTION_RELEASE_BLOCKED:
            if (
                self.reason_codes == (ProductionReleaseReasonCodeV2.NONE,)
                or ProductionReleaseReasonCodeV2.NONE in self.reason_codes
                or any(value is not None for value in published_values)
                or any(published_collections)
                or self.satisfies_sc_015
            ):
                raise ValueError("blocked production release carries authority")
            if self.evidence_class is ProductionReleaseEvidenceClassV2.REPOSITORY_PENDING_ONLY:
                if self.repository_evidence_ref is None:
                    raise ValueError("repository blocked result requires evidence ref")
                _require_ref(
                    self.repository_evidence_ref,
                    "production-release-repository-evidence",
                    "json/v1",
                    "repository_evidence_ref",
                )
            elif self.repository_evidence_ref is not None:
                raise ValueError("non-repository blocked result carries repository ref")
        else:
            if (
                self.evidence_class is not ProductionReleaseEvidenceClassV2.PRODUCTION_VERIFIED
                or self.reason_codes != (ProductionReleaseReasonCodeV2.NONE,)
                or any(value is None for value in published_values)
                or any(not values for values in published_collections)
                or self.repository_evidence_ref is not None
                or not self.satisfies_sc_015
            ):
                raise ValueError("published production release closure is incomplete")
            self._validate_published()
        refs = self.refs
        _validate_audit(self.audit, refs, "production release result")
        _validate_identity(
            self.result_id,
            self.result_sha256,
            "production-release-result",
            production_release_result_v2_carried_sha256(self),
        )
        return self

    def _validate_published(self) -> None:
        assert self.attestation_authority is not None
        assert self.attestation_authority_ref is not None
        assert self.release_manifest is not None
        assert self.release_manifest_ref is not None
        assert self.registry_entry is not None
        assert self.registry_entry_ref is not None
        if self.attestation_authority_ref != self.attestation_authority.to_ref():
            raise ValueError("attestation authority ref differs from nested value")
        if self.release_manifest_ref != self.release_manifest.to_ref():
            raise ValueError("release manifest ref differs from nested value")
        if self.registry_entry_ref != self.registry_entry.to_ref():
            raise ValueError("registry entry ref differs from nested value")
        count = len(self.release_manifest.item_ids)
        if any(
            len(values) != count
            for values in (
                self.export_receipts,
                self.published_decisions,
                self.published_items,
                self.item_projections,
                self.approved_source_result_refs,
            )
        ):
            raise ValueError("published production coverage is not exact")
        for decision in self.published_decisions:
            validate_release_decision_v2_identity(decision)
            if (
                decision.channel is not ReleaseChannel.PRODUCTION
                or decision.action is not ReleaseActionV2.PUBLISH
                or decision.state is not ReleaseStateV2.RELEASED
                or decision.production_attestation_ref != self.attestation_authority.frozen_attestation_ref
            ):
                raise ValueError("published decision lacks exact production authority")
        for item in self.published_items:
            validate_evaluation_item_v2_identity(item)
        for receipt in self.export_receipts:
            validate_production_lh_export_receipt_v2_identity(receipt)
        for projection in self.item_projections:
            validate_production_published_item_projection_v2_identity(projection)
        if (
            self.registry_entry.release_manifest_ref != self.release_manifest_ref
            or self.registry_entry.attestation_authority_ref != self.attestation_authority_ref
            or self.registry_entry.item_ids != self.release_manifest.item_ids
            or self.registry_entry.item_manifest_refs != self.release_manifest.item_manifest_refs
            or self.registry_entry.export_receipt_refs
            != tuple(value.to_ref() for value in self.export_receipts)
            or self.registry_entry.published_release_decision_refs
            != tuple(release_decision_v2_ref(value) for value in self.published_decisions)
            or self.registry_entry.published_evaluation_item_refs
            != tuple(evaluation_item_v2_ref(value) for value in self.published_items)
        ):
            raise ValueError("production registry bindings are not exact")
        for manifest, receipt, decision, item, projection, source_ref in zip(
            self.release_manifest.item_manifests,
            self.export_receipts,
            self.published_decisions,
            self.published_items,
            self.item_projections,
            self.approved_source_result_refs,
            strict=True,
        ):
            if (
                manifest.approved_release_result_ref != source_ref
                or receipt.release_manifest_ref != self.release_manifest_ref
                or receipt.item_manifest_ref != manifest.to_ref()
                or decision.previous_decision_ref != manifest.predecessor_decision_ref
                or decision.release_subject_sha256 != _sha_from_ref(manifest.release_subject_ref)
                or decision.package_manifest_ref != manifest.final_package_manifest_ref
                or decision.package_sha256 != manifest.package_sha256
                or item.release_decision_ref != release_decision_v2_ref(decision)
                or projection.approved_result_ref != source_ref
                or projection.release_manifest_ref != self.release_manifest_ref
                or projection.export_receipt_ref != receipt.to_ref()
                or projection.registry_entry_ref != self.registry_entry_ref
                or projection.publish_decision_ref != release_decision_v2_ref(decision)
                or projection.published_evaluation_item_ref != evaluation_item_v2_ref(item)
                or projection.attestation_authority_ref != self.attestation_authority_ref
            ):
                raise ValueError("production Item publication bindings are not exact")

    @classmethod
    def create_blocked(
        cls,
        *,
        policy_ref: ObjectRef,
        evidence_class: ProductionReleaseEvidenceClassV2,
        reason_codes: tuple[ProductionReleaseReasonCodeV2, ...],
        repository_evidence_ref: ObjectRef | None,
        audit: ContractAudit,
    ) -> ProductionReleaseResultV2:
        refs = (
            policy_ref,
            *((repository_evidence_ref,) if repository_evidence_ref else ()),
        )
        value = cls(
            result_id="production-release-result://pending",
            evidence_class=evidence_class,
            outcome=ProductionReleaseOutcomeV2.PRODUCTION_RELEASE_BLOCKED,
            reason_codes=reason_codes,
            policy_ref=policy_ref,
            repository_evidence_ref=repository_evidence_ref,
            satisfies_sc_015=False,
            result_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        return _finalize(
            value,
            "result_id",
            "result_sha256",
            "production-release-result",
            production_release_result_v2_carried_sha256(value),
        )

    @classmethod
    def create_published(
        cls,
        *,
        policy_ref: ObjectRef,
        attestation_authority: ProductionAttestationAuthorityV2,
        release_manifest: ProductionReleaseManifestV2,
        export_receipts: tuple[ProductionLHExportReceiptV2, ...],
        published_decisions: tuple[ReleaseDecisionV2, ...],
        published_items: tuple[EvaluationItemV2, ...],
        item_projections: tuple[ProductionPublishedItemProjectionV2, ...],
        registry_entry: ProductionRegistryEntryV2,
        approved_source_result_refs: tuple[ObjectRef, ...],
        audit: ContractAudit,
    ) -> ProductionReleaseResultV2:
        refs = (
            policy_ref,
            attestation_authority.to_ref(),
            release_manifest.to_ref(),
            *(value.to_ref() for value in export_receipts),
            *(release_decision_v2_ref(value) for value in published_decisions),
            *(evaluation_item_v2_ref(value) for value in published_items),
            *(value.to_ref() for value in item_projections),
            registry_entry.to_ref(),
            *approved_source_result_refs,
        )
        value = cls(
            result_id="production-release-result://pending",
            evidence_class=ProductionReleaseEvidenceClassV2.PRODUCTION_VERIFIED,
            outcome=ProductionReleaseOutcomeV2.PUBLISHED,
            reason_codes=(ProductionReleaseReasonCodeV2.NONE,),
            policy_ref=policy_ref,
            attestation_authority=attestation_authority,
            attestation_authority_ref=attestation_authority.to_ref(),
            release_manifest=release_manifest,
            release_manifest_ref=release_manifest.to_ref(),
            export_receipts=export_receipts,
            published_decisions=published_decisions,
            published_items=published_items,
            item_projections=item_projections,
            registry_entry=registry_entry,
            registry_entry_ref=registry_entry.to_ref(),
            approved_source_result_refs=approved_source_result_refs,
            satisfies_sc_015=True,
            result_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        return _finalize(
            value,
            "result_id",
            "result_sha256",
            "production-release-result",
            production_release_result_v2_carried_sha256(value),
        )

    @property
    def refs(self) -> tuple[ObjectRef, ...]:
        return (
            self.policy_ref,
            *((self.repository_evidence_ref,) if self.repository_evidence_ref else ()),
            *((self.attestation_authority_ref,) if self.attestation_authority_ref else ()),
            *((self.release_manifest_ref,) if self.release_manifest_ref else ()),
            *(value.to_ref() for value in self.export_receipts),
            *(release_decision_v2_ref(value) for value in self.published_decisions),
            *(evaluation_item_v2_ref(value) for value in self.published_items),
            *(value.to_ref() for value in self.item_projections),
            *((self.registry_entry_ref,) if self.registry_entry_ref else ()),
            *self.approved_source_result_refs,
        )

    def to_ref(self) -> ObjectRef:
        return production_release_result_v2_ref(self)


def production_release_policy_v2_carried_sha256(
    value: ProductionReleasePolicyV2,
) -> str:
    return _carried(value, {"policy_id", "policy_sha256", "audit"})


def production_release_policy_v2_ref(
    value: ProductionReleasePolicyV2,
) -> ObjectRef:
    validate_production_release_policy_v2_identity(value)
    return _ref("production-release-policy", value.policy_id, value.policy_sha256)


def production_attestation_authority_v2_carried_sha256(
    value: ProductionAttestationAuthorityV2,
) -> str:
    return _carried(value, {"authority_id", "authority_sha256", "audit"})


def production_attestation_authority_v2_ref(
    value: ProductionAttestationAuthorityV2,
) -> ObjectRef:
    validate_production_attestation_authority_v2_identity(value)
    return _ref(
        "production-attestation-authority",
        value.authority_id,
        value.authority_sha256,
    )


def production_release_item_manifest_v2_carried_sha256(
    value: ProductionReleaseItemManifestV2,
) -> str:
    return _carried(value, {"item_manifest_id", "item_manifest_sha256", "audit"})


def production_release_item_manifest_v2_ref(
    value: ProductionReleaseItemManifestV2,
) -> ObjectRef:
    validate_production_release_item_manifest_v2_identity(value)
    return _ref(
        "production-release-item-manifest",
        value.item_manifest_id,
        value.item_manifest_sha256,
    )


def production_release_manifest_v2_carried_sha256(
    value: ProductionReleaseManifestV2,
) -> str:
    return _carried(value, {"manifest_id", "manifest_sha256", "audit"})


def production_release_manifest_v2_ref(
    value: ProductionReleaseManifestV2,
) -> ObjectRef:
    validate_production_release_manifest_v2_identity(value)
    return _ref(
        "production-release-manifest",
        value.manifest_id,
        value.manifest_sha256,
    )


def production_lh_export_receipt_v2_carried_sha256(
    value: ProductionLHExportReceiptV2,
) -> str:
    return _carried(value, {"receipt_id", "receipt_sha256", "audit"})


def production_lh_export_receipt_v2_ref(
    value: ProductionLHExportReceiptV2,
) -> ObjectRef:
    validate_production_lh_export_receipt_v2_identity(value)
    return _ref(
        "production-lh-export-receipt",
        value.receipt_id,
        value.receipt_sha256,
    )


def production_registry_entry_v2_carried_sha256(
    value: ProductionRegistryEntryV2,
) -> str:
    return _carried(value, {"registry_entry_id", "entry_sha256", "audit"})


def production_registry_entry_v2_ref(
    value: ProductionRegistryEntryV2,
) -> ObjectRef:
    validate_production_registry_entry_v2_identity(value)
    return _ref(
        "production-registry-entry",
        value.registry_entry_id,
        value.entry_sha256,
    )


def production_published_item_projection_v2_carried_sha256(
    value: ProductionPublishedItemProjectionV2,
) -> str:
    return _carried(value, {"projection_id", "projection_sha256", "audit"})


def production_published_item_projection_v2_ref(
    value: ProductionPublishedItemProjectionV2,
) -> ObjectRef:
    validate_production_published_item_projection_v2_identity(value)
    return _ref(
        "production-published-item-projection",
        value.projection_id,
        value.projection_sha256,
    )


def production_release_result_v2_carried_sha256(
    value: ProductionReleaseResultV2,
) -> str:
    return _carried(value, {"result_id", "result_sha256", "audit"})


def production_release_result_v2_ref(
    value: ProductionReleaseResultV2,
) -> ObjectRef:
    validate_production_release_result_v2_identity(value)
    return _ref("production-release-result", value.result_id, value.result_sha256)


def validate_production_release_policy_v2_identity(
    value: ProductionReleasePolicyV2,
) -> None:
    _validate_identity(
        value.policy_id,
        value.policy_sha256,
        "production-release-policy",
        production_release_policy_v2_carried_sha256(value),
        allow_pending=False,
    )


def validate_production_attestation_authority_v2_identity(
    value: ProductionAttestationAuthorityV2,
) -> None:
    _validate_identity(
        value.authority_id,
        value.authority_sha256,
        "production-attestation-authority",
        production_attestation_authority_v2_carried_sha256(value),
        allow_pending=False,
    )


def validate_production_release_item_manifest_v2_identity(
    value: ProductionReleaseItemManifestV2,
) -> None:
    _validate_identity(
        value.item_manifest_id,
        value.item_manifest_sha256,
        "production-release-item-manifest",
        production_release_item_manifest_v2_carried_sha256(value),
        allow_pending=False,
    )


def validate_production_release_manifest_v2_identity(
    value: ProductionReleaseManifestV2,
) -> None:
    _validate_identity(
        value.manifest_id,
        value.manifest_sha256,
        "production-release-manifest",
        production_release_manifest_v2_carried_sha256(value),
        allow_pending=False,
    )


def validate_production_lh_export_receipt_v2_identity(
    value: ProductionLHExportReceiptV2,
) -> None:
    _validate_identity(
        value.receipt_id,
        value.receipt_sha256,
        "production-lh-export-receipt",
        production_lh_export_receipt_v2_carried_sha256(value),
        allow_pending=False,
    )


def validate_production_registry_entry_v2_identity(
    value: ProductionRegistryEntryV2,
) -> None:
    _validate_identity(
        value.registry_entry_id,
        value.entry_sha256,
        "production-registry-entry",
        production_registry_entry_v2_carried_sha256(value),
        allow_pending=False,
    )


def validate_production_published_item_projection_v2_identity(
    value: ProductionPublishedItemProjectionV2,
) -> None:
    _validate_identity(
        value.projection_id,
        value.projection_sha256,
        "production-published-item-projection",
        production_published_item_projection_v2_carried_sha256(value),
        allow_pending=False,
    )


def validate_production_release_result_v2_identity(
    value: ProductionReleaseResultV2,
) -> None:
    _validate_identity(
        value.result_id,
        value.result_sha256,
        "production-release-result",
        production_release_result_v2_carried_sha256(value),
        allow_pending=False,
    )


def _sorted_members(
    values: tuple[LHWorkspaceExportMemberV2, ...],
) -> tuple[LHWorkspaceExportMemberV2, ...]:
    return tuple(
        sorted(
            values,
            key=lambda value: (
                value.normalized_path,
                value.member_type.value,
                value.media_type,
                value.content_sha256,
                value.size_bytes,
            ),
        )
    )


def _require_sorted_unique_members(
    values: tuple[LHWorkspaceExportMemberV2, ...],
) -> None:
    if values != _sorted_members(values):
        raise ValueError("workspace members must be canonical")
    keys = tuple((value.normalized_path, value.member_type.value, value.content_sha256) for value in values)
    if len(set(keys)) != len(keys):
        raise ValueError("workspace members must be unique")


def _sorted_refs(values: tuple[ObjectRef, ...]) -> tuple[ObjectRef, ...]:
    return tuple(sorted(values, key=_ref_key))


def _require_sorted_unique_refs(
    values: tuple[ObjectRef, ...],
    label: str,
) -> None:
    if values != _sorted_refs(values) or len(set(_ref_key(value) for value in values)) != len(values):
        raise ValueError(f"{label} must be canonical and unique")
    for value in values:
        _require_safe_ref(value, label)


def _require_ref(
    value: ObjectRef,
    object_type: str,
    object_version: str,
    label: str,
) -> None:
    if value.object_type != object_type or value.object_version != object_version:
        raise ValueError(f"{label} has invalid object type or version")
    _require_safe_ref(value, label)


def _require_safe_ref(value: ObjectRef, label: str) -> None:
    serialized = ":".join(
        (
            value.object_type,
            value.object_id,
            value.object_version,
        )
    ).casefold()
    if any(marker in serialized for marker in _DENIED_REF_MARKERS):
        raise ValueError(f"{label} contains a forbidden reference surface")


def _ref_key(value: ObjectRef) -> tuple[str, str, str, str]:
    return (
        value.object_type,
        value.object_id,
        value.object_version,
        value.object_sha256,
    )


def _safe_audit(
    audit: ContractAudit,
    refs: tuple[ObjectRef, ...],
) -> ContractAudit:
    _require_aware(audit.created_at, "audit.created_at")
    return audit.model_copy(update={"input_refs": _sorted_refs(refs)})


def _validate_audit(
    audit: ContractAudit,
    refs: tuple[ObjectRef, ...],
    label: str,
) -> None:
    _require_aware(audit.created_at, "audit.created_at")
    expected = _sorted_refs(refs)
    if audit.input_refs != expected:
        raise ValueError(f"{label} audit refs are not exact")
    for ref in audit.input_refs:
        _require_safe_ref(ref, f"{label} audit")


def _require_aware(value: datetime, label: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware")


def _parse_enum[EnumT: StrEnum](
    value: object,
    enum_type: type[EnumT],
    label: str,
) -> EnumT:
    if isinstance(value, enum_type):
        return value
    if isinstance(value, str):
        return enum_type(value)
    raise TypeError(f"{label} must be a closed string enum")


def _object_ref(value: FacadeObjectRef) -> ObjectRef:
    return ObjectRef(
        object_type=value.object_type,
        object_id=value.object_id,
        object_version=value.object_version,
        object_sha256=value.object_sha256,
    )


def _ref(
    object_type: str,
    object_id: str,
    object_sha256: str,
) -> ObjectRef:
    if object_id.endswith("://pending") or object_sha256 == "0" * 64:
        raise ValueError("cannot create a ref for a pending identity")
    return ObjectRef(
        object_type=object_type,
        object_id=object_id,
        object_version="v2",
        object_sha256=object_sha256,
    )


def _carried(
    value: ContractModelV2,
    exclude: set[str],
) -> str:
    payload = canonical_value_v2(
        value.model_dump(mode="python", exclude=exclude),
    )
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _finalize[ModelT: ContractModelV2](
    value: ModelT,
    id_field: str,
    sha_field: str,
    prefix: str,
    digest: str,
) -> ModelT:
    result = value.model_copy(
        update={
            id_field: f"{prefix}://sha256/{digest}",
            sha_field: digest,
        }
    )
    return type(value).model_validate(result.model_dump(mode="python"))


def _validate_identity(
    object_id: str,
    object_sha256: str,
    prefix: str,
    observed: str,
    *,
    allow_pending: bool = True,
) -> None:
    if allow_pending and object_id == f"{prefix}://pending" and object_sha256 == "0" * 64:
        return
    if object_sha256 != observed or object_id != f"{prefix}://sha256/{observed}":
        raise ValueError(f"{prefix} identity is stale")


def _sha_from_ref(value: ObjectRef) -> str:
    return value.object_sha256


__all__ = [
    "PRODUCTION_ATTESTATION_POLICY_VERSION",
    "PRODUCTION_RELEASE_POLICY_VERSION",
    "ProductionAttestationAuthorityV2",
    "ProductionLHExportReceiptV2",
    "ProductionPublishedItemProjectionV2",
    "ProductionRegistryEntryV2",
    "ProductionReleaseEvidenceClassV2",
    "ProductionReleaseItemManifestV2",
    "ProductionReleaseManifestV2",
    "ProductionReleaseOutcomeV2",
    "ProductionReleasePolicyV2",
    "ProductionReleaseReasonCodeV2",
    "ProductionReleaseResultV2",
    "production_attestation_authority_v2_carried_sha256",
    "production_attestation_authority_v2_ref",
    "production_lh_export_receipt_v2_carried_sha256",
    "production_lh_export_receipt_v2_ref",
    "production_published_item_projection_v2_carried_sha256",
    "production_published_item_projection_v2_ref",
    "production_registry_entry_v2_carried_sha256",
    "production_registry_entry_v2_ref",
    "production_release_item_manifest_v2_carried_sha256",
    "production_release_item_manifest_v2_ref",
    "production_release_manifest_v2_carried_sha256",
    "production_release_manifest_v2_ref",
    "production_release_policy_v2_carried_sha256",
    "production_release_policy_v2_ref",
    "production_release_result_v2_carried_sha256",
    "production_release_result_v2_ref",
    "validate_production_attestation_authority_v2_identity",
    "validate_production_lh_export_receipt_v2_identity",
    "validate_production_published_item_projection_v2_identity",
    "validate_production_registry_entry_v2_identity",
    "validate_production_release_item_manifest_v2_identity",
    "validate_production_release_manifest_v2_identity",
    "validate_production_release_policy_v2_identity",
    "validate_production_release_result_v2_identity",
]
