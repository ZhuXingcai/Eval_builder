from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, model_validator

from eval_factory.contracts.core import (
    ContractAudit,
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
    Visibility,
)
from eval_factory.contracts.validation_v2 import (
    CandidatePackageInventoryEntryV2,
    candidate_package_inventory_entry_ref,
    validate_candidate_package_inventory_entry_identity,
)

ITEM_QUALITY_POLICY_VERSION: Literal["item-quality/r5-10-v1"] = "item-quality/r5-10-v1"
_PACKAGE_PROVENANCE_RULE_ID = "item-quality-package/r5-10-v1"
_PROVENANCE_DECISION_POLICY_VERSION = "provenance-decision-table/r2-01-v1"


class ItemQualityOutcomeV2(StrEnum):
    PASSED = "PASSED"
    REQUIRES_REPAIR = "REQUIRES_REPAIR"
    REJECTED = "REJECTED"
    BLOCKED = "BLOCKED"
    UPSTREAM_INCOMPLETE = "UPSTREAM_INCOMPLETE"


class ItemQualityFailureCodeV2(StrEnum):
    PACKAGE_INVENTORY_MISSING = "PACKAGE_INVENTORY_MISSING"
    PACKAGE_INVENTORY_MISMATCH = "PACKAGE_INVENTORY_MISMATCH"
    PACKAGE_MEMBER_INVALID = "PACKAGE_MEMBER_INVALID"
    PACKAGE_PATH_COLLISION = "PACKAGE_PATH_COLLISION"
    PACKAGE_CONTENT_HASH_MISSING = "PACKAGE_CONTENT_HASH_MISSING"
    PACKAGE_HASH_MISMATCH = "PACKAGE_HASH_MISMATCH"
    PROVENANCE_INCOMPLETE = "PROVENANCE_INCOMPLETE"
    PROVENANCE_UNSAFE = "PROVENANCE_UNSAFE"
    ENVIRONMENT_INCOMPLETE = "ENVIRONMENT_INCOMPLETE"


class FinalPackageManifestV2(ContractModelV2):
    schema_version: Literal["eval-factory/final-package-manifest/v2"] = (
        "eval-factory/final-package-manifest/v2"
    )
    final_package_manifest_id: Identifier
    candidate_revision_ref: ObjectRef
    deterministic_validation_ref: ObjectRef
    source_deterministic_validation_ref: ObjectRef
    candidate_inventory_ref: ObjectRef | None = None
    artifact_version_refs: tuple[ObjectRef, ...] = ()
    output_refs: tuple[ObjectRef, ...] = ()
    artifact_validation_result_refs: tuple[ObjectRef, ...] = ()
    entries: tuple[CandidatePackageInventoryEntryV2, ...] = ()
    entry_refs: tuple[ObjectRef, ...] = ()
    accepted_artifact_refs: tuple[ObjectRef, ...] = ()
    package_sha256: Sha256
    input_state_only: Literal[True] = True
    policy_version: Literal["item-quality/r5-10-v1"] = ITEM_QUALITY_POLICY_VERSION
    manifest_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        _require_ref(
            self.candidate_revision_ref,
            "attachment-candidate-revision",
            "v2",
            "candidate_revision_ref",
        )
        _require_ref(
            self.deterministic_validation_ref,
            "revision-deterministic-validation",
            "v2",
            "deterministic_validation_ref",
        )
        _require_ref(
            self.source_deterministic_validation_ref,
            "deterministic-item-validation-result",
            "v2",
            "source_deterministic_validation_ref",
        )
        if self.candidate_inventory_ref is not None:
            _require_ref(
                self.candidate_inventory_ref,
                "candidate-package-inventory",
                "v2",
                "candidate_inventory_ref",
            )
        for label, refs, object_type in (
            (
                "final package artifact version refs",
                self.artifact_version_refs,
                "candidate-artifact-version",
            ),
            ("final package output refs", self.output_refs, "attachment-output"),
            (
                "final package validation refs",
                self.artifact_validation_result_refs,
                "artifact-deterministic-validation-result",
            ),
            (
                "final package accepted artifact refs",
                self.accepted_artifact_refs,
                "candidate-artifact-version",
            ),
        ):
            _require_sorted_unique_refs(label, refs)
            for ref in refs:
                _require_ref(ref, object_type, "v2", label)
        for item in self.entries:
            validate_candidate_package_inventory_entry_identity(item)
        expected_entry_refs = tuple(candidate_package_inventory_entry_ref(item) for item in self.entries)
        _require_sorted_unique_refs(
            "final package entry refs",
            expected_entry_refs,
        )
        for ref in expected_entry_refs:
            _require_ref(
                ref,
                "candidate-package-inventory-entry",
                "v2",
                "entry_refs",
            )
        if self.entry_refs != expected_entry_refs:
            raise ValueError("final package entry refs must match nested entries")
        entry_output_refs = _sorted_refs(tuple(item.output_ref for item in self.entries))
        entry_validation_refs = _sorted_refs(
            tuple(item.artifact_validation_result_ref for item in self.entries)
        )
        if (
            entry_output_refs != self.output_refs
            or entry_validation_refs != self.artifact_validation_result_refs
        ):
            raise ValueError("final package entries must exactly cover output and validation refs")
        if self.accepted_artifact_refs != self.artifact_version_refs:
            raise ValueError("final package accepted refs must equal candidate artifact versions")
        if self.candidate_inventory_ref is None and (
            self.entries
            or self.artifact_version_refs
            or self.output_refs
            or self.artifact_validation_result_refs
        ):
            raise ValueError("final package without candidate inventory must be the exact empty set")
        expected_package_sha = final_package_content_sha256(self.entries)
        if self.package_sha256 != expected_package_sha:
            raise ValueError("final package content hash is stale")
        _validate_audit(
            self.audit,
            (
                self.candidate_revision_ref,
                self.deterministic_validation_ref,
                self.source_deterministic_validation_ref,
                *((self.candidate_inventory_ref,) if self.candidate_inventory_ref else ()),
                *self.artifact_version_refs,
                *self.output_refs,
                *self.artifact_validation_result_refs,
                *self.entry_refs,
            ),
            "final package manifest",
        )
        _validate_identity(
            object_id=self.final_package_manifest_id,
            object_sha256=self.manifest_sha256,
            expected_prefix="final-package-manifest",
            observed=final_package_manifest_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        candidate_revision_ref: ObjectRef,
        deterministic_validation_ref: ObjectRef,
        source_deterministic_validation_ref: ObjectRef,
        candidate_inventory_ref: ObjectRef | None,
        artifact_version_refs: tuple[ObjectRef, ...],
        output_refs: tuple[ObjectRef, ...],
        artifact_validation_result_refs: tuple[ObjectRef, ...],
        entries: tuple[CandidatePackageInventoryEntryV2, ...],
        accepted_artifact_refs: tuple[ObjectRef, ...],
        audit: ContractAudit,
    ) -> FinalPackageManifestV2:
        sorted_entries = tuple(
            sorted(
                entries,
                key=lambda item: _ref_key(candidate_package_inventory_entry_ref(item)),
            )
        )
        value = cls(
            final_package_manifest_id="final-package-manifest://pending",
            candidate_revision_ref=candidate_revision_ref,
            deterministic_validation_ref=deterministic_validation_ref,
            source_deterministic_validation_ref=source_deterministic_validation_ref,
            candidate_inventory_ref=candidate_inventory_ref,
            artifact_version_refs=_sorted_refs(artifact_version_refs),
            output_refs=_sorted_refs(output_refs),
            artifact_validation_result_refs=_sorted_refs(artifact_validation_result_refs),
            entries=sorted_entries,
            entry_refs=tuple(candidate_package_inventory_entry_ref(item) for item in sorted_entries),
            accepted_artifact_refs=_sorted_refs(accepted_artifact_refs),
            package_sha256=final_package_content_sha256(sorted_entries),
            manifest_sha256="0" * 64,
            audit=audit,
        )
        return _finalize(
            value,
            id_field="final_package_manifest_id",
            hash_field="manifest_sha256",
            prefix="final-package-manifest",
            digest=final_package_manifest_carried_sha256(value),
        )


class PackageMemberProvenanceV2(ContractModelV2):
    schema_version: Literal["eval-factory/package-member-provenance/v2"] = (
        "eval-factory/package-member-provenance/v2"
    )
    package_member_provenance_id: Identifier
    final_package_manifest_ref: ObjectRef
    inventory_entry: CandidatePackageInventoryEntryV2
    inventory_entry_ref: ObjectRef
    inventory_member_ref: ObjectRef
    candidate_artifact_version_ref: ObjectRef
    output_ref: ObjectRef
    build_spec_ref: ObjectRef
    artifact_validation_result_ref: ObjectRef
    derivation_closure_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    provenance_decision: ProvenanceDecision
    provenance_decision_ref: ObjectRef
    policy_version: Literal["item-quality/r5-10-v1"] = ITEM_QUALITY_POLICY_VERSION
    binding_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_binding(self) -> Self:
        _require_ref(
            self.final_package_manifest_ref,
            "final-package-manifest",
            "v2",
            "final_package_manifest_ref",
        )
        expected_entry_ref = candidate_package_inventory_entry_ref(self.inventory_entry)
        validate_candidate_package_inventory_entry_identity(self.inventory_entry)
        if self.inventory_entry_ref != expected_entry_ref:
            raise ValueError("package member provenance entry ref must match nested entry")
        if self.inventory_member_ref != self.inventory_entry.inventory_member_ref:
            raise ValueError("package member provenance member ref must match inventory entry")
        for ref, object_type, field_name in (
            (
                self.inventory_entry_ref,
                "candidate-package-inventory-entry",
                "inventory_entry_ref",
            ),
            (
                self.inventory_member_ref,
                "attachment-inventory-member",
                "inventory_member_ref",
            ),
            (
                self.candidate_artifact_version_ref,
                "candidate-artifact-version",
                "candidate_artifact_version_ref",
            ),
            (self.output_ref, "attachment-output", "output_ref"),
            (self.build_spec_ref, "artifact-build-spec", "build_spec_ref"),
            (
                self.artifact_validation_result_ref,
                "artifact-deterministic-validation-result",
                "artifact_validation_result_ref",
            ),
        ):
            _require_ref(ref, object_type, "v2", field_name)
        if (
            self.inventory_entry.output_ref != self.output_ref
            or self.inventory_entry.build_spec_ref != self.build_spec_ref
            or self.inventory_entry.artifact_validation_result_ref != self.artifact_validation_result_ref
        ):
            raise ValueError("package member provenance does not match inventory ownership")
        _require_sorted_unique_refs(
            "package member derivation closure",
            self.derivation_closure_refs,
        )
        expected_decision_ref = provenance_decision_stable_ref(self.provenance_decision)
        if self.provenance_decision_ref != expected_decision_ref:
            raise ValueError("package member provenance decision ref is stale")
        decision = self.provenance_decision
        if (
            decision.subject_ref != self.inventory_member_ref
            or decision.subject_sha256 != self.inventory_member_ref.object_sha256
            or decision.origin_class is not OriginClass.SYSTEM_OR_HARNESS_CONTEXT
            or decision.visibility is not Visibility.CONTESTANT_VISIBLE
            or decision.disposition is not Disposition.ALLOW_INPUT_EVIDENCE
            or decision.taint_labels
            or decision.content_risk_labels
            or decision.review_required
            or decision.derived_from != self.derivation_closure_refs
            or decision.rule_ids != (_PACKAGE_PROVENANCE_RULE_ID,)
            or decision.source_event_refs
            or decision.confidence != 1.0
            or decision.policy_version != _PROVENANCE_DECISION_POLICY_VERSION
        ):
            raise ValueError("package member provenance decision is not safe and exact")
        _validate_audit(
            self.audit,
            (
                self.final_package_manifest_ref,
                self.inventory_entry_ref,
                self.inventory_member_ref,
                self.candidate_artifact_version_ref,
                self.output_ref,
                self.build_spec_ref,
                self.artifact_validation_result_ref,
                *self.derivation_closure_refs,
                self.provenance_decision_ref,
            ),
            "package member provenance",
        )
        _validate_identity(
            object_id=self.package_member_provenance_id,
            object_sha256=self.binding_sha256,
            expected_prefix="package-member-provenance",
            observed=package_member_provenance_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        final_package_manifest_ref: ObjectRef,
        inventory_entry: CandidatePackageInventoryEntryV2,
        candidate_artifact_version_ref: ObjectRef,
        output_ref: ObjectRef,
        build_spec_ref: ObjectRef,
        artifact_validation_result_ref: ObjectRef,
        derivation_closure_refs: tuple[ObjectRef, ...],
        provenance_decision: ProvenanceDecision,
        audit: ContractAudit,
    ) -> PackageMemberProvenanceV2:
        value = cls(
            package_member_provenance_id=("package-member-provenance://pending"),
            final_package_manifest_ref=final_package_manifest_ref,
            inventory_entry=inventory_entry,
            inventory_entry_ref=candidate_package_inventory_entry_ref(inventory_entry),
            inventory_member_ref=inventory_entry.inventory_member_ref,
            candidate_artifact_version_ref=candidate_artifact_version_ref,
            output_ref=output_ref,
            build_spec_ref=build_spec_ref,
            artifact_validation_result_ref=artifact_validation_result_ref,
            derivation_closure_refs=_sorted_refs(derivation_closure_refs),
            provenance_decision=provenance_decision,
            provenance_decision_ref=provenance_decision_stable_ref(provenance_decision),
            binding_sha256="0" * 64,
            audit=audit,
        )
        return _finalize(
            value,
            id_field="package_member_provenance_id",
            hash_field="binding_sha256",
            prefix="package-member-provenance",
            digest=package_member_provenance_carried_sha256(value),
        )


class ProvenanceManifestV2(ContractModelV2):
    schema_version: Literal["eval-factory/provenance-manifest/v2"] = "eval-factory/provenance-manifest/v2"
    provenance_manifest_id: Identifier
    final_package_manifest_ref: ObjectRef
    candidate_revision_ref: ObjectRef
    package_sha256: Sha256
    package_inventory_entry_refs: tuple[ObjectRef, ...]
    entries: tuple[PackageMemberProvenanceV2, ...]
    entry_refs: tuple[ObjectRef, ...]
    exact_set_verified: Literal[True] = True
    policy_version: Literal["item-quality/r5-10-v1"] = ITEM_QUALITY_POLICY_VERSION
    provenance_manifest_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        _require_ref(
            self.final_package_manifest_ref,
            "final-package-manifest",
            "v2",
            "final_package_manifest_ref",
        )
        _require_ref(
            self.candidate_revision_ref,
            "attachment-candidate-revision",
            "v2",
            "candidate_revision_ref",
        )
        _require_sorted_unique_refs(
            "provenance package inventory entry refs",
            self.package_inventory_entry_refs,
        )
        for ref in self.package_inventory_entry_refs:
            _require_ref(
                ref,
                "candidate-package-inventory-entry",
                "v2",
                "package_inventory_entry_refs",
            )
        expected_entry_refs = tuple(package_member_provenance_ref(item) for item in self.entries)
        if self.entry_refs != expected_entry_refs:
            raise ValueError("provenance manifest entry refs must match nested entries")
        if len(self.entry_refs) != len(set(self.entry_refs)):
            raise ValueError("provenance manifest entry refs must be unique")
        for ref in self.entry_refs:
            _require_ref(
                ref,
                "package-member-provenance",
                "v2",
                "entry_refs",
            )
        observed_inventory_refs = tuple(item.inventory_entry_ref for item in self.entries)
        if observed_inventory_refs != self.package_inventory_entry_refs:
            raise ValueError("provenance manifest must exactly cover package inventory entries")
        if any(item.final_package_manifest_ref != self.final_package_manifest_ref for item in self.entries):
            raise ValueError("provenance manifest entries belong to another package")
        _validate_audit(
            self.audit,
            (
                self.final_package_manifest_ref,
                self.candidate_revision_ref,
                *self.package_inventory_entry_refs,
                *self.entry_refs,
            ),
            "provenance manifest",
        )
        _validate_identity(
            object_id=self.provenance_manifest_id,
            object_sha256=self.provenance_manifest_sha256,
            expected_prefix="provenance-manifest",
            observed=provenance_manifest_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        final_package_manifest_ref: ObjectRef,
        candidate_revision_ref: ObjectRef,
        package_sha256: str,
        package_inventory_entry_refs: tuple[ObjectRef, ...],
        entries: tuple[PackageMemberProvenanceV2, ...],
        audit: ContractAudit,
    ) -> ProvenanceManifestV2:
        sorted_entries = tuple(
            sorted(
                entries,
                key=lambda item: _ref_key(item.inventory_entry_ref),
            )
        )
        value = cls(
            provenance_manifest_id="provenance-manifest://pending",
            final_package_manifest_ref=final_package_manifest_ref,
            candidate_revision_ref=candidate_revision_ref,
            package_sha256=package_sha256,
            package_inventory_entry_refs=_sorted_refs(package_inventory_entry_refs),
            entries=sorted_entries,
            entry_refs=tuple(package_member_provenance_ref(item) for item in sorted_entries),
            provenance_manifest_sha256="0" * 64,
            audit=audit,
        )
        return _finalize(
            value,
            id_field="provenance_manifest_id",
            hash_field="provenance_manifest_sha256",
            prefix="provenance-manifest",
            digest=provenance_manifest_v2_carried_sha256(value),
        )


class EnvironmentArtifactV2(ContractModelV2):
    schema_version: Literal["eval-factory/environment-artifact/v2"] = "eval-factory/environment-artifact/v2"
    artifact_id: Identifier
    attachment_dependency_id: Identifier
    candidate_artifact_version_ref: ObjectRef
    output_ref: ObjectRef
    logical_path: RelativePath
    media_type: str = Field(min_length=3, max_length=255)
    content_sha256: Sha256
    size_bytes: int = Field(ge=0)
    artifact_validation_result_ref: ObjectRef
    package_inventory_entry_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    artifact_sha256: Sha256

    @model_validator(mode="after")
    def validate_artifact(self) -> Self:
        _require_ref(
            self.candidate_artifact_version_ref,
            "candidate-artifact-version",
            "v2",
            "candidate_artifact_version_ref",
        )
        _require_ref(
            self.output_ref,
            "attachment-output",
            "v2",
            "output_ref",
        )
        _require_ref(
            self.artifact_validation_result_ref,
            "artifact-deterministic-validation-result",
            "v2",
            "artifact_validation_result_ref",
        )
        if self.output_ref.object_sha256 != self.content_sha256:
            raise ValueError("environment artifact content hash must match output ref")
        _require_sorted_unique_refs(
            "environment artifact package entry refs",
            self.package_inventory_entry_refs,
        )
        for ref in self.package_inventory_entry_refs:
            _require_ref(
                ref,
                "candidate-package-inventory-entry",
                "v2",
                "package_inventory_entry_refs",
            )
        _validate_identity(
            object_id=(f"environment-artifact://sha256/{self.artifact_sha256}"),
            object_sha256=self.artifact_sha256,
            expected_prefix="environment-artifact",
            observed=environment_artifact_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        artifact_id: str,
        attachment_dependency_id: str,
        candidate_artifact_version_ref: ObjectRef,
        output_ref: ObjectRef,
        logical_path: str,
        media_type: str,
        size_bytes: int,
        artifact_validation_result_ref: ObjectRef,
        package_inventory_entry_refs: tuple[ObjectRef, ...],
    ) -> EnvironmentArtifactV2:
        value = cls(
            artifact_id=artifact_id,
            attachment_dependency_id=attachment_dependency_id,
            candidate_artifact_version_ref=candidate_artifact_version_ref,
            output_ref=output_ref,
            logical_path=logical_path,
            media_type=media_type,
            content_sha256=output_ref.object_sha256,
            size_bytes=size_bytes,
            artifact_validation_result_ref=artifact_validation_result_ref,
            package_inventory_entry_refs=_sorted_refs(package_inventory_entry_refs),
            artifact_sha256="0" * 64,
        )
        digest = environment_artifact_carried_sha256(value)
        return value.model_copy(update={"artifact_sha256": digest})


class EnvironmentSpecV2(ContractModelV2):
    schema_version: Literal["eval-factory/environment-spec/v2"] = "eval-factory/environment-spec/v2"
    environment_spec_id: Identifier
    candidate_revision_ref: ObjectRef
    final_package_manifest_ref: ObjectRef
    provenance_manifest_ref: ObjectRef
    candidate_artifact_version_refs: tuple[ObjectRef, ...]
    artifacts: tuple[EnvironmentArtifactV2, ...]
    artifact_refs: tuple[ObjectRef, ...]
    package_sha256: Sha256
    input_state_only: Literal[True] = True
    policy_version: Literal["item-quality/r5-10-v1"] = ITEM_QUALITY_POLICY_VERSION
    environment_spec_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        for ref, object_type, field_name in (
            (
                self.candidate_revision_ref,
                "attachment-candidate-revision",
                "candidate_revision_ref",
            ),
            (
                self.final_package_manifest_ref,
                "final-package-manifest",
                "final_package_manifest_ref",
            ),
            (
                self.provenance_manifest_ref,
                "provenance-manifest",
                "provenance_manifest_ref",
            ),
        ):
            _require_ref(ref, object_type, "v2", field_name)
        _require_sorted_unique_refs(
            "environment candidate artifact refs",
            self.candidate_artifact_version_refs,
        )
        for ref in self.candidate_artifact_version_refs:
            _require_ref(
                ref,
                "candidate-artifact-version",
                "v2",
                "candidate_artifact_version_refs",
            )
        expected_artifact_refs = tuple(environment_artifact_ref(item) for item in self.artifacts)
        if self.artifact_refs != expected_artifact_refs:
            raise ValueError("environment artifact refs must match nested artifacts")
        for ref in self.artifact_refs:
            _require_ref(
                ref,
                "environment-artifact",
                "v2",
                "artifact_refs",
            )
        observed_candidate_refs = tuple(item.candidate_artifact_version_ref for item in self.artifacts)
        if observed_candidate_refs != self.candidate_artifact_version_refs:
            raise ValueError("environment artifacts must exactly cover candidate artifacts")
        _validate_audit(
            self.audit,
            (
                self.candidate_revision_ref,
                self.final_package_manifest_ref,
                self.provenance_manifest_ref,
                *self.candidate_artifact_version_refs,
                *self.artifact_refs,
            ),
            "environment spec",
        )
        _validate_identity(
            object_id=self.environment_spec_id,
            object_sha256=self.environment_spec_sha256,
            expected_prefix="environment-spec",
            observed=environment_spec_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        candidate_revision_ref: ObjectRef,
        final_package_manifest_ref: ObjectRef,
        provenance_manifest_ref: ObjectRef,
        candidate_artifact_version_refs: tuple[ObjectRef, ...],
        artifacts: tuple[EnvironmentArtifactV2, ...],
        package_sha256: str,
        audit: ContractAudit,
    ) -> EnvironmentSpecV2:
        sorted_artifacts = tuple(
            sorted(
                artifacts,
                key=lambda item: _ref_key(item.candidate_artifact_version_ref),
            )
        )
        value = cls(
            environment_spec_id="environment-spec://pending",
            candidate_revision_ref=candidate_revision_ref,
            final_package_manifest_ref=final_package_manifest_ref,
            provenance_manifest_ref=provenance_manifest_ref,
            candidate_artifact_version_refs=_sorted_refs(candidate_artifact_version_refs),
            artifacts=sorted_artifacts,
            artifact_refs=tuple(environment_artifact_ref(item) for item in sorted_artifacts),
            package_sha256=package_sha256,
            environment_spec_sha256="0" * 64,
            audit=audit,
        )
        return _finalize(
            value,
            id_field="environment_spec_id",
            hash_field="environment_spec_sha256",
            prefix="environment-spec",
            digest=environment_spec_v2_carried_sha256(value),
        )


class QualityReportV2(ContractModelV2):
    schema_version: Literal["eval-factory/quality-report/v2"] = "eval-factory/quality-report/v2"
    quality_report_id: Identifier
    r4_task_contract_set_ref: ObjectRef
    review_policy_ref: ObjectRef
    semantic_workflow_result_ref: ObjectRef
    candidate_revision_ref: ObjectRef
    deterministic_validation_ref: ObjectRef
    source_deterministic_validation_ref: ObjectRef
    candidate_inventory_ref: ObjectRef | None = None
    artifact_version_refs: tuple[ObjectRef, ...] = ()
    output_refs: tuple[ObjectRef, ...] = ()
    artifact_validation_result_refs: tuple[ObjectRef, ...] = ()
    semantic_round_result_refs: tuple[ObjectRef, ...] = ()
    stage_result_refs: tuple[ObjectRef, ...] = ()
    repair_plan_refs: tuple[ObjectRef, ...] = ()
    repair_result_refs: tuple[ObjectRef, ...] = ()
    current_finding_refs: tuple[ObjectRef, ...] = ()
    stale_finding_refs: tuple[ObjectRef, ...] = ()
    resolution_refs: tuple[ObjectRef, ...] = ()
    accepted_artifact_refs: tuple[ObjectRef, ...] = ()
    final_package_manifest_ref: ObjectRef | None = None
    provenance_manifest_ref: ObjectRef | None = None
    environment_spec_ref: ObjectRef | None = None
    package_sha256: Sha256 | None = None
    input_state_only: Literal[True] | None = None
    outcome: ItemQualityOutcomeV2
    failure_code: ItemQualityFailureCodeV2 | None = None
    open_p0_count: int = Field(ge=0)
    open_p1_count: int = Field(ge=0)
    unresolved_non_waivable_count: int = Field(ge=0)
    approvable: bool
    policy_version: Literal["item-quality/r5-10-v1"] = ITEM_QUALITY_POLICY_VERSION
    quality_report_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        for ref, object_type, field_name in (
            (
                self.r4_task_contract_set_ref,
                "r4-task-contract-set",
                "r4_task_contract_set_ref",
            ),
            (
                self.review_policy_ref,
                "semantic-review-policy",
                "review_policy_ref",
            ),
            (
                self.semantic_workflow_result_ref,
                "semantic-review-workflow-result",
                "semantic_workflow_result_ref",
            ),
            (
                self.candidate_revision_ref,
                "attachment-candidate-revision",
                "candidate_revision_ref",
            ),
            (
                self.deterministic_validation_ref,
                "revision-deterministic-validation",
                "deterministic_validation_ref",
            ),
            (
                self.source_deterministic_validation_ref,
                "deterministic-item-validation-result",
                "source_deterministic_validation_ref",
            ),
        ):
            _require_ref(ref, object_type, "v2", field_name)
        if self.candidate_inventory_ref is not None:
            _require_ref(
                self.candidate_inventory_ref,
                "candidate-package-inventory",
                "v2",
                "candidate_inventory_ref",
            )
        for label, refs in (
            ("quality artifact version refs", self.artifact_version_refs),
            ("quality output refs", self.output_refs),
            (
                "quality artifact validation refs",
                self.artifact_validation_result_refs,
            ),
            ("quality repair plan refs", self.repair_plan_refs),
            ("quality repair result refs", self.repair_result_refs),
            ("quality current finding refs", self.current_finding_refs),
            ("quality stale finding refs", self.stale_finding_refs),
            ("quality resolution refs", self.resolution_refs),
            ("quality accepted artifact refs", self.accepted_artifact_refs),
        ):
            _require_sorted_unique_refs(label, refs)
        for label, refs, object_type, object_version in (
            (
                "quality artifact version refs",
                self.artifact_version_refs,
                "candidate-artifact-version",
                "v2",
            ),
            (
                "quality output refs",
                self.output_refs,
                "attachment-output",
                "v2",
            ),
            (
                "quality artifact validation refs",
                self.artifact_validation_result_refs,
                "artifact-deterministic-validation-result",
                "v2",
            ),
            (
                "quality repair plan refs",
                self.repair_plan_refs,
                "targeted-repair-plan",
                "v2",
            ),
            (
                "quality repair result refs",
                self.repair_result_refs,
                "repaired-artifact-build-result",
                "v2",
            ),
            (
                "quality resolution refs",
                self.resolution_refs,
                "semantic-finding-resolution",
                "v2",
            ),
            (
                "quality accepted artifact refs",
                self.accepted_artifact_refs,
                "candidate-artifact-version",
                "v2",
            ),
        ):
            for ref in refs:
                _require_ref(
                    ref,
                    object_type,
                    object_version,
                    label,
                )
        for ref in self.semantic_round_result_refs:
            _require_ref(
                ref,
                "semantic-review-round-result",
                "v2",
                "semantic_round_result_refs",
            )
        for ref in self.stage_result_refs:
            _require_ref(
                ref,
                "stage-result",
                "record/v1",
                "stage_result_refs",
            )
        for label, refs in (
            ("current_finding_refs", self.current_finding_refs),
            ("stale_finding_refs", self.stale_finding_refs),
        ):
            for ref in refs:
                _require_ref_one_of(
                    ref,
                    (
                        "deterministic-validation-finding",
                        "semantic-review-finding",
                    ),
                    "v2",
                    label,
                )
        if len(self.semantic_round_result_refs) != len(set(self.semantic_round_result_refs)):
            raise ValueError("quality semantic round refs must be unique")
        if len(self.stage_result_refs) != len(set(self.stage_result_refs)):
            raise ValueError("quality stage result refs must be unique")
        if len(self.semantic_round_result_refs) != len(self.stage_result_refs):
            raise ValueError("quality round and StageResult refs must align")
        if set(self.current_finding_refs) & set(self.stale_finding_refs):
            raise ValueError("quality current and stale findings must be disjoint")
        final_fields = (
            self.final_package_manifest_ref,
            self.provenance_manifest_ref,
            self.environment_spec_ref,
            self.package_sha256,
            self.input_state_only,
        )
        if self.approvable:
            if (
                self.outcome is not ItemQualityOutcomeV2.PASSED
                or self.failure_code is not None
                or self.current_finding_refs
                or self.open_p0_count
                or self.open_p1_count
                or self.unresolved_non_waivable_count
                or len(self.semantic_round_result_refs) != 3
                or any(item is None for item in final_fields)
                or self.accepted_artifact_refs != self.artifact_version_refs
            ):
                raise ValueError("approvable quality report requires complete passing package truth")
        elif any(item is not None for item in final_fields) or self.accepted_artifact_refs:
            raise ValueError("non-approvable quality report cannot carry final package truth")
        if self.outcome is ItemQualityOutcomeV2.PASSED and not self.approvable:
            raise ValueError("PASSED quality report must be approvable")
        if self.failure_code is not None and self.outcome is not ItemQualityOutcomeV2.BLOCKED:
            raise ValueError("quality package failure code requires BLOCKED outcome")
        optional_final_refs: tuple[
            tuple[ObjectRef | None, str, str],
            ...,
        ] = (
            (
                self.final_package_manifest_ref,
                "final-package-manifest",
                "final_package_manifest_ref",
            ),
            (
                self.provenance_manifest_ref,
                "provenance-manifest",
                "provenance_manifest_ref",
            ),
            (
                self.environment_spec_ref,
                "environment-spec",
                "environment_spec_ref",
            ),
        )
        for optional_ref, object_type, field_name in optional_final_refs:
            if optional_ref is not None:
                _require_ref(
                    optional_ref,
                    object_type,
                    "v2",
                    field_name,
                )
        _validate_audit(
            self.audit,
            _quality_report_refs(self),
            "quality report",
        )
        _validate_identity(
            object_id=self.quality_report_id,
            object_sha256=self.quality_report_sha256,
            expected_prefix="quality-report",
            observed=quality_report_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        r4_task_contract_set_ref: ObjectRef,
        review_policy_ref: ObjectRef,
        semantic_workflow_result_ref: ObjectRef,
        candidate_revision_ref: ObjectRef,
        deterministic_validation_ref: ObjectRef,
        source_deterministic_validation_ref: ObjectRef,
        candidate_inventory_ref: ObjectRef | None,
        artifact_version_refs: tuple[ObjectRef, ...],
        output_refs: tuple[ObjectRef, ...],
        artifact_validation_result_refs: tuple[ObjectRef, ...],
        semantic_round_result_refs: tuple[ObjectRef, ...],
        stage_result_refs: tuple[ObjectRef, ...],
        repair_plan_refs: tuple[ObjectRef, ...],
        repair_result_refs: tuple[ObjectRef, ...],
        current_finding_refs: tuple[ObjectRef, ...],
        stale_finding_refs: tuple[ObjectRef, ...],
        resolution_refs: tuple[ObjectRef, ...],
        accepted_artifact_refs: tuple[ObjectRef, ...],
        final_package_manifest_ref: ObjectRef | None,
        provenance_manifest_ref: ObjectRef | None,
        environment_spec_ref: ObjectRef | None,
        package_sha256: str | None,
        input_state_only: Literal[True] | None,
        outcome: ItemQualityOutcomeV2,
        failure_code: ItemQualityFailureCodeV2 | None,
        open_p0_count: int,
        open_p1_count: int,
        unresolved_non_waivable_count: int,
        approvable: bool,
        audit: ContractAudit,
    ) -> QualityReportV2:
        value = cls(
            quality_report_id="quality-report://pending",
            r4_task_contract_set_ref=r4_task_contract_set_ref,
            review_policy_ref=review_policy_ref,
            semantic_workflow_result_ref=semantic_workflow_result_ref,
            candidate_revision_ref=candidate_revision_ref,
            deterministic_validation_ref=deterministic_validation_ref,
            source_deterministic_validation_ref=(source_deterministic_validation_ref),
            candidate_inventory_ref=candidate_inventory_ref,
            artifact_version_refs=_sorted_refs(artifact_version_refs),
            output_refs=_sorted_refs(output_refs),
            artifact_validation_result_refs=_sorted_refs(artifact_validation_result_refs),
            semantic_round_result_refs=semantic_round_result_refs,
            stage_result_refs=stage_result_refs,
            repair_plan_refs=_sorted_refs(repair_plan_refs),
            repair_result_refs=_sorted_refs(repair_result_refs),
            current_finding_refs=_sorted_refs(current_finding_refs),
            stale_finding_refs=_sorted_refs(stale_finding_refs),
            resolution_refs=_sorted_refs(resolution_refs),
            accepted_artifact_refs=_sorted_refs(accepted_artifact_refs),
            final_package_manifest_ref=final_package_manifest_ref,
            provenance_manifest_ref=provenance_manifest_ref,
            environment_spec_ref=environment_spec_ref,
            package_sha256=package_sha256,
            input_state_only=input_state_only,
            outcome=outcome,
            failure_code=failure_code,
            open_p0_count=open_p0_count,
            open_p1_count=open_p1_count,
            unresolved_non_waivable_count=unresolved_non_waivable_count,
            approvable=approvable,
            quality_report_sha256="0" * 64,
            audit=audit,
        )
        return _finalize(
            value,
            id_field="quality_report_id",
            hash_field="quality_report_sha256",
            prefix="quality-report",
            digest=quality_report_v2_carried_sha256(value),
        )


class ItemQualityCompilationResultV2(ContractModelV2):
    schema_version: Literal["eval-factory/item-quality-compilation-result/v2"] = (
        "eval-factory/item-quality-compilation-result/v2"
    )
    item_quality_compilation_result_id: Identifier
    quality_report: QualityReportV2
    quality_report_ref: ObjectRef
    final_package_manifest: FinalPackageManifestV2 | None = None
    final_package_manifest_ref: ObjectRef | None = None
    provenance_manifest: ProvenanceManifestV2 | None = None
    provenance_manifest_ref: ObjectRef | None = None
    environment_spec: EnvironmentSpecV2 | None = None
    environment_spec_ref: ObjectRef | None = None
    accepted_artifact_refs: tuple[ObjectRef, ...] = ()
    package_sha256: Sha256 | None = None
    input_state_only: Literal[True] | None = None
    evaluation_item_ref: None = None
    batch_quality_report_ref: None = None
    release_decision_ref: None = None
    policy_version: Literal["item-quality/r5-10-v1"] = ITEM_QUALITY_POLICY_VERSION
    result_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if self.quality_report_ref != quality_report_v2_ref(self.quality_report):
            raise ValueError("item quality report ref must match nested report")
        expected_package_ref = (
            final_package_manifest_ref(self.final_package_manifest)
            if self.final_package_manifest is not None
            else None
        )
        expected_provenance_ref = (
            provenance_manifest_v2_ref(self.provenance_manifest)
            if self.provenance_manifest is not None
            else None
        )
        expected_environment_ref = (
            environment_spec_v2_ref(self.environment_spec) if self.environment_spec is not None else None
        )
        if (
            self.final_package_manifest_ref != expected_package_ref
            or self.provenance_manifest_ref != expected_provenance_ref
            or self.environment_spec_ref != expected_environment_ref
        ):
            raise ValueError("item quality nested package refs are not exact")
        nested_values = (
            self.final_package_manifest,
            self.provenance_manifest,
            self.environment_spec,
        )
        if self.quality_report.approvable:
            if (
                any(item is None for item in nested_values)
                or self.accepted_artifact_refs != self.quality_report.accepted_artifact_refs
                or self.package_sha256 != self.quality_report.package_sha256
                or self.input_state_only is not True
            ):
                raise ValueError("approvable item quality result requires complete package truth")
            assert self.final_package_manifest is not None
            assert self.provenance_manifest is not None
            assert self.environment_spec is not None
            package = self.final_package_manifest
            provenance = self.provenance_manifest
            environment = self.environment_spec
            if (
                self.quality_report.final_package_manifest_ref != expected_package_ref
                or self.quality_report.provenance_manifest_ref != expected_provenance_ref
                or self.quality_report.environment_spec_ref != expected_environment_ref
            ):
                raise ValueError("item quality report does not match nested package refs")
            if (
                package.candidate_revision_ref != self.quality_report.candidate_revision_ref
                or package.deterministic_validation_ref != self.quality_report.deterministic_validation_ref
                or package.source_deterministic_validation_ref
                != self.quality_report.source_deterministic_validation_ref
                or package.candidate_inventory_ref != self.quality_report.candidate_inventory_ref
                or package.artifact_version_refs != self.quality_report.artifact_version_refs
                or package.package_sha256 != provenance.package_sha256
                or package.package_sha256 != environment.package_sha256
                or package.package_sha256 != self.quality_report.package_sha256
                or package.candidate_revision_ref != provenance.candidate_revision_ref
                or package.candidate_revision_ref != environment.candidate_revision_ref
                or provenance.final_package_manifest_ref != expected_package_ref
                or environment.final_package_manifest_ref != expected_package_ref
                or environment.provenance_manifest_ref != expected_provenance_ref
                or package.entry_refs != provenance.package_inventory_entry_refs
                or package.artifact_version_refs != environment.candidate_artifact_version_refs
                or package.accepted_artifact_refs != self.accepted_artifact_refs
                or package.output_refs != self.quality_report.output_refs
                or package.artifact_validation_result_refs
                != self.quality_report.artifact_validation_result_refs
            ):
                raise ValueError("item quality package values are cross-subject")
            package_entry_by_ref = dict(
                zip(
                    package.entry_refs,
                    package.entries,
                    strict=True,
                )
            )
            provenance_by_entry_ref = {item.inventory_entry_ref: item for item in provenance.entries}
            provenance_candidate_refs = _sorted_refs(
                tuple(item.candidate_artifact_version_ref for item in provenance.entries)
            )
            environment_entry_refs = tuple(
                entry_ref
                for artifact in environment.artifacts
                for entry_ref in artifact.package_inventory_entry_refs
            )
            if (
                provenance_candidate_refs != package.artifact_version_refs
                or len(environment_entry_refs) != len(set(environment_entry_refs))
                or _sorted_refs(environment_entry_refs) != package.entry_refs
            ):
                raise ValueError("item quality package ownership sets are not exact")
            for artifact in environment.artifacts:
                for entry_ref in artifact.package_inventory_entry_refs:
                    entry = package_entry_by_ref[entry_ref]
                    binding = provenance_by_entry_ref[entry_ref]
                    if (
                        entry.output_ref != artifact.output_ref
                        or entry.artifact_validation_result_ref != artifact.artifact_validation_result_ref
                        or binding.candidate_artifact_version_ref != artifact.candidate_artifact_version_ref
                        or binding.output_ref != artifact.output_ref
                        or binding.artifact_validation_result_ref != artifact.artifact_validation_result_ref
                    ):
                        raise ValueError("item quality package ownership is cross-artifact")
        elif (
            any(item is not None for item in nested_values)
            or self.accepted_artifact_refs
            or self.package_sha256 is not None
            or self.input_state_only is not None
        ):
            raise ValueError("non-approvable item quality result cannot carry package truth")
        _validate_audit(
            self.audit,
            (
                self.quality_report_ref,
                *((expected_package_ref,) if expected_package_ref else ()),
                *((expected_provenance_ref,) if expected_provenance_ref else ()),
                *((expected_environment_ref,) if expected_environment_ref else ()),
                *self.accepted_artifact_refs,
            ),
            "item quality compilation result",
        )
        _validate_identity(
            object_id=self.item_quality_compilation_result_id,
            object_sha256=self.result_sha256,
            expected_prefix="item-quality-compilation-result",
            observed=item_quality_compilation_result_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        quality_report: QualityReportV2,
        final_package_manifest: FinalPackageManifestV2 | None,
        provenance_manifest: ProvenanceManifestV2 | None,
        environment_spec: EnvironmentSpecV2 | None,
        audit: ContractAudit,
    ) -> ItemQualityCompilationResultV2:
        package_ref = (
            final_package_manifest_ref(final_package_manifest) if final_package_manifest is not None else None
        )
        provenance_ref = (
            provenance_manifest_v2_ref(provenance_manifest) if provenance_manifest is not None else None
        )
        environment_ref = environment_spec_v2_ref(environment_spec) if environment_spec is not None else None
        value = cls(
            item_quality_compilation_result_id=("item-quality-compilation-result://pending"),
            quality_report=quality_report,
            quality_report_ref=quality_report_v2_ref(quality_report),
            final_package_manifest=final_package_manifest,
            final_package_manifest_ref=package_ref,
            provenance_manifest=provenance_manifest,
            provenance_manifest_ref=provenance_ref,
            environment_spec=environment_spec,
            environment_spec_ref=environment_ref,
            accepted_artifact_refs=quality_report.accepted_artifact_refs,
            package_sha256=quality_report.package_sha256,
            input_state_only=quality_report.input_state_only,
            result_sha256="0" * 64,
            audit=audit,
        )
        return _finalize(
            value,
            id_field="item_quality_compilation_result_id",
            hash_field="result_sha256",
            prefix="item-quality-compilation-result",
            digest=item_quality_compilation_result_carried_sha256(value),
        )


def final_package_content_sha256(
    entries: tuple[CandidatePackageInventoryEntryV2, ...],
) -> str:
    payload = [
        {
            "container_ref": (_ref_payload(item.container_ref) if item.container_ref is not None else None),
            "normalized_path": item.inventory_member.normalized_path,
            "member_type": item.inventory_member.member_type,
            "media_type": item.inventory_member.media_type,
            "size_bytes": item.inventory_member.size_bytes,
            "content_sha256": item.inventory_member.content_sha256,
        }
        for item in sorted(
            entries,
            key=lambda entry: (
                entry.container_ref.object_id if entry.container_ref is not None else "",
                entry.inventory_member.normalized_path,
                entry.inventory_member.member_type,
            ),
        )
    ]
    return _payload_sha256(payload)


def final_package_manifest_carried_sha256(
    value: FinalPackageManifestV2,
) -> str:
    return _payload_sha256(
        {
            "candidate_revision_ref": _ref_payload(value.candidate_revision_ref),
            "deterministic_validation_ref": _ref_payload(value.deterministic_validation_ref),
            "source_deterministic_validation_ref": _ref_payload(value.source_deterministic_validation_ref),
            "candidate_inventory_ref": _maybe_ref_payload(value.candidate_inventory_ref),
            "artifact_version_refs": [_ref_payload(ref) for ref in value.artifact_version_refs],
            "output_refs": [_ref_payload(ref) for ref in value.output_refs],
            "artifact_validation_result_refs": [
                _ref_payload(ref) for ref in value.artifact_validation_result_refs
            ],
            "entry_refs": [_ref_payload(ref) for ref in value.entry_refs],
            "accepted_artifact_refs": [_ref_payload(ref) for ref in value.accepted_artifact_refs],
            "package_sha256": value.package_sha256,
            "input_state_only": value.input_state_only,
            "policy_version": value.policy_version,
        }
    )


def final_package_manifest_ref(
    value: FinalPackageManifestV2,
) -> ObjectRef:
    return ObjectRef(
        object_type="final-package-manifest",
        object_id=value.final_package_manifest_id,
        object_version="v2",
        object_sha256=value.manifest_sha256,
    )


def provenance_decision_stable_ref(
    value: ProvenanceDecision,
) -> ObjectRef:
    return ObjectRef(
        object_type="provenance-decision",
        object_id=value.provenance_decision_id,
        object_version=value.policy_version,
        object_sha256=_payload_sha256(_provenance_decision_payload(value)),
    )


def package_member_provenance_carried_sha256(
    value: PackageMemberProvenanceV2,
) -> str:
    return _payload_sha256(
        {
            "final_package_manifest_ref": _ref_payload(value.final_package_manifest_ref),
            "inventory_entry_ref": _ref_payload(value.inventory_entry_ref),
            "inventory_member_ref": _ref_payload(value.inventory_member_ref),
            "candidate_artifact_version_ref": _ref_payload(value.candidate_artifact_version_ref),
            "output_ref": _ref_payload(value.output_ref),
            "build_spec_ref": _ref_payload(value.build_spec_ref),
            "artifact_validation_result_ref": _ref_payload(value.artifact_validation_result_ref),
            "derivation_closure_refs": [_ref_payload(ref) for ref in value.derivation_closure_refs],
            "provenance_decision": _provenance_decision_payload(value.provenance_decision),
            "provenance_decision_ref": _ref_payload(value.provenance_decision_ref),
            "policy_version": value.policy_version,
        }
    )


def package_member_provenance_ref(
    value: PackageMemberProvenanceV2,
) -> ObjectRef:
    return ObjectRef(
        object_type="package-member-provenance",
        object_id=value.package_member_provenance_id,
        object_version="v2",
        object_sha256=value.binding_sha256,
    )


def provenance_manifest_v2_carried_sha256(
    value: ProvenanceManifestV2,
) -> str:
    return _payload_sha256(
        {
            "final_package_manifest_ref": _ref_payload(value.final_package_manifest_ref),
            "candidate_revision_ref": _ref_payload(value.candidate_revision_ref),
            "package_sha256": value.package_sha256,
            "package_inventory_entry_refs": [_ref_payload(ref) for ref in value.package_inventory_entry_refs],
            "entry_refs": [_ref_payload(ref) for ref in value.entry_refs],
            "exact_set_verified": value.exact_set_verified,
            "policy_version": value.policy_version,
        }
    )


def provenance_manifest_v2_ref(
    value: ProvenanceManifestV2,
) -> ObjectRef:
    return ObjectRef(
        object_type="provenance-manifest",
        object_id=value.provenance_manifest_id,
        object_version="v2",
        object_sha256=value.provenance_manifest_sha256,
    )


def environment_artifact_carried_sha256(
    value: EnvironmentArtifactV2,
) -> str:
    return _payload_sha256(
        value.model_dump(
            mode="json",
            exclude={"artifact_sha256"},
            exclude_none=False,
        )
    )


def environment_artifact_ref(
    value: EnvironmentArtifactV2,
) -> ObjectRef:
    return ObjectRef(
        object_type="environment-artifact",
        object_id=(f"environment-artifact://sha256/{value.artifact_sha256}"),
        object_version="v2",
        object_sha256=value.artifact_sha256,
    )


def environment_spec_v2_carried_sha256(
    value: EnvironmentSpecV2,
) -> str:
    return _payload_sha256(
        {
            "candidate_revision_ref": _ref_payload(value.candidate_revision_ref),
            "final_package_manifest_ref": _ref_payload(value.final_package_manifest_ref),
            "provenance_manifest_ref": _ref_payload(value.provenance_manifest_ref),
            "candidate_artifact_version_refs": [
                _ref_payload(ref) for ref in value.candidate_artifact_version_refs
            ],
            "artifact_refs": [_ref_payload(ref) for ref in value.artifact_refs],
            "package_sha256": value.package_sha256,
            "input_state_only": value.input_state_only,
            "policy_version": value.policy_version,
        }
    )


def environment_spec_v2_ref(value: EnvironmentSpecV2) -> ObjectRef:
    return ObjectRef(
        object_type="environment-spec",
        object_id=value.environment_spec_id,
        object_version="v2",
        object_sha256=value.environment_spec_sha256,
    )


def quality_report_v2_carried_sha256(
    value: QualityReportV2,
) -> str:
    return _payload_sha256(
        value.model_dump(
            mode="json",
            exclude={
                "quality_report_id",
                "quality_report_sha256",
                "audit",
            },
            exclude_none=False,
        )
    )


def quality_report_v2_ref(value: QualityReportV2) -> ObjectRef:
    return ObjectRef(
        object_type="quality-report",
        object_id=value.quality_report_id,
        object_version="v2",
        object_sha256=value.quality_report_sha256,
    )


def item_quality_compilation_result_carried_sha256(
    value: ItemQualityCompilationResultV2,
) -> str:
    return _payload_sha256(
        {
            "quality_report_ref": _ref_payload(value.quality_report_ref),
            "final_package_manifest_ref": _maybe_ref_payload(value.final_package_manifest_ref),
            "provenance_manifest_ref": _maybe_ref_payload(value.provenance_manifest_ref),
            "environment_spec_ref": _maybe_ref_payload(value.environment_spec_ref),
            "accepted_artifact_refs": [_ref_payload(ref) for ref in value.accepted_artifact_refs],
            "package_sha256": value.package_sha256,
            "input_state_only": value.input_state_only,
            "evaluation_item_ref": None,
            "batch_quality_report_ref": None,
            "release_decision_ref": None,
            "policy_version": value.policy_version,
        }
    )


def item_quality_compilation_result_ref(
    value: ItemQualityCompilationResultV2,
) -> ObjectRef:
    return ObjectRef(
        object_type="item-quality-compilation-result",
        object_id=value.item_quality_compilation_result_id,
        object_version="v2",
        object_sha256=value.result_sha256,
    )


def _quality_report_refs(
    value: QualityReportV2,
) -> tuple[ObjectRef, ...]:
    return (
        value.r4_task_contract_set_ref,
        value.review_policy_ref,
        value.semantic_workflow_result_ref,
        value.candidate_revision_ref,
        value.deterministic_validation_ref,
        value.source_deterministic_validation_ref,
        *((value.candidate_inventory_ref,) if value.candidate_inventory_ref else ()),
        *value.artifact_version_refs,
        *value.output_refs,
        *value.artifact_validation_result_refs,
        *value.semantic_round_result_refs,
        *value.stage_result_refs,
        *value.repair_plan_refs,
        *value.repair_result_refs,
        *value.current_finding_refs,
        *value.stale_finding_refs,
        *value.resolution_refs,
        *value.accepted_artifact_refs,
        *((value.final_package_manifest_ref,) if value.final_package_manifest_ref else ()),
        *((value.provenance_manifest_ref,) if value.provenance_manifest_ref else ()),
        *((value.environment_spec_ref,) if value.environment_spec_ref else ()),
    )


def _provenance_decision_payload(
    value: ProvenanceDecision,
) -> dict[str, object]:
    return value.model_dump(
        mode="json",
        exclude={"audit"},
        exclude_none=False,
    )


def _ref_payload(value: ObjectRef) -> dict[str, object]:
    return value.model_dump(mode="json", exclude_none=False)


def _maybe_ref_payload(
    value: ObjectRef | None,
) -> dict[str, object] | None:
    return _ref_payload(value) if value is not None else None


def _payload_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            canonical_value_v2(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _ref_key(value: ObjectRef) -> tuple[str, str, str, str]:
    return (
        value.object_type,
        value.object_id,
        value.object_version,
        value.object_sha256,
    )


def _sorted_refs(values: tuple[ObjectRef, ...]) -> tuple[ObjectRef, ...]:
    unique = {_ref_key(value): value for value in values}
    return tuple(unique[key] for key in sorted(unique))


def _require_sorted_unique_refs(
    label: str,
    values: tuple[ObjectRef, ...],
) -> None:
    if values != _sorted_refs(values) or len(values) != len(set(values)):
        raise ValueError(f"{label} must be sorted and unique")


def _require_ref(
    value: ObjectRef,
    object_type: str,
    object_version: str,
    field_name: str,
) -> None:
    if value.object_type != object_type or value.object_version != object_version:
        raise ValueError(f"{field_name} must reference {object_type} {object_version}")


def _require_ref_one_of(
    value: ObjectRef,
    object_types: tuple[str, ...],
    object_version: str,
    field_name: str,
) -> None:
    if value.object_type not in object_types or value.object_version != object_version:
        raise ValueError(f"{field_name} must reference one of {object_types} {object_version}")


def _validate_audit(
    audit: ContractAudit,
    expected_refs: tuple[ObjectRef, ...],
    label: str,
) -> None:
    expected = _sorted_refs(expected_refs)
    if audit.input_refs != expected:
        raise ValueError(f"{label} audit refs are incomplete")


def _validate_identity(
    *,
    object_id: str,
    object_sha256: str,
    expected_prefix: str,
    observed: str,
) -> None:
    if object_id.endswith("://pending") or object_sha256 == "0" * 64:
        return
    if object_sha256 != observed or object_id != f"{expected_prefix}://sha256/{observed}":
        raise ValueError(f"{expected_prefix} identity is stale")


def _finalize[ModelT: ContractModelV2](
    value: ModelT,
    *,
    id_field: str,
    hash_field: str,
    prefix: str,
    digest: str,
) -> ModelT:
    return value.model_copy(
        update={
            id_field: f"{prefix}://sha256/{digest}",
            hash_field: digest,
        }
    )
