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

RELEASE_PUBLICATION_POLICY_VERSION: Literal["release-publication/r7-09-v1"] = "release-publication/r7-09-v1"


class ReleasePublicationPhaseV2(StrEnum):
    PUBLISHED = "PUBLISHED"


class ReleasePublicationPolicyV2(ContractModelV2):
    schema_version: Literal["eval-factory/release-publication-policy/v2"] = (
        "eval-factory/release-publication-policy/v2"
    )
    policy_id: Identifier
    max_items: int = Field(ge=1, le=100_000)
    max_workspace_members_per_item: int = Field(ge=1, le=100_000)
    max_workspace_bytes_per_item: int = Field(
        ge=1,
        le=5_000_000_000,
    )
    max_manifest_refs: int = Field(ge=1, le=1_000_000)
    canary_registry_ids: frozenset[Identifier] = Field(min_length=1)
    internal_review_registry_ids: frozenset[Identifier] = Field(min_length=1)
    allowed_profile: Literal["LH"] = "LH"
    allowed_profile_version: Literal["v1"] = "v1"
    query_policy_version: Literal["lh-query-packaging/r7-09-v1"] = "lh-query-packaging/r7-09-v1"
    policy_version: Literal["release-publication/r7-09-v1"] = RELEASE_PUBLICATION_POLICY_VERSION
    policy_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_policy(self) -> Self:
        if self.canary_registry_ids & self.internal_review_registry_ids:
            raise ValueError("channel registry allowlists must be disjoint")
        if any(
            "production" in value.casefold()
            for value in (
                *self.canary_registry_ids,
                *self.internal_review_registry_ids,
            )
        ):
            raise ValueError("R7-09 policy cannot contain production")
        _validate_audit(self.audit, (), "release publication policy")
        _validate_identity(
            self.policy_id,
            self.policy_sha256,
            "release-publication-policy",
            release_publication_policy_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        max_items: int,
        max_workspace_members_per_item: int,
        max_workspace_bytes_per_item: int,
        max_manifest_refs: int,
        canary_registry_ids: frozenset[str],
        internal_review_registry_ids: frozenset[str],
        audit: ContractAudit,
    ) -> ReleasePublicationPolicyV2:
        value = cls(
            policy_id="release-publication-policy://pending",
            max_items=max_items,
            max_workspace_members_per_item=(max_workspace_members_per_item),
            max_workspace_bytes_per_item=max_workspace_bytes_per_item,
            max_manifest_refs=max_manifest_refs,
            canary_registry_ids=canary_registry_ids,
            internal_review_registry_ids=internal_review_registry_ids,
            policy_sha256="0" * 64,
            audit=_safe_audit(audit, ()),
        )
        return _finalize(
            value,
            "policy_id",
            "policy_sha256",
            "release-publication-policy",
            release_publication_policy_v2_carried_sha256(value),
        )

    def to_ref(self) -> ObjectRef:
        return release_publication_policy_v2_ref(self)


class LHReleaseItemManifestV2(ContractModelV2):
    schema_version: Literal["eval-factory/lh-release-item-manifest/v2"] = (
        "eval-factory/lh-release-item-manifest/v2"
    )
    item_manifest_id: Identifier
    job_id: Identifier
    item_id: Identifier
    approved_release_result_ref: ObjectRef
    approved_evaluation_item_ref: ObjectRef
    approved_release_decision_ref: ObjectRef
    release_subject_ref: ObjectRef
    query_spec_ref: ObjectRef
    rubric_set_ref: ObjectRef
    environment_spec_ref: ObjectRef
    provenance_manifest_ref: ObjectRef
    quality_report_ref: ObjectRef
    final_package_manifest_ref: ObjectRef
    package_sha256: Sha256
    query_yaml_included: Literal[True] = True
    query_yaml_sha256: Sha256
    rubrics_json_sha256: Sha256
    expected_workspace_members: tuple[
        LHWorkspaceExportMemberV2,
        ...,
    ]
    channel: ReleaseChannel
    registry: Identifier
    export_profile: Literal["LH"] = "LH"
    export_profile_version: Literal["v1"] = "v1"
    policy_version: Literal["release-publication/r7-09-v1"] = RELEASE_PUBLICATION_POLICY_VERSION
    item_manifest_sha256: Sha256
    audit: ContractAudit

    @field_validator("channel", mode="before")
    @classmethod
    def parse_channel(cls, value: object) -> ReleaseChannel:
        return _parse_enum(value, ReleaseChannel, "channel")

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        refs = (
            (self.approved_release_result_ref, "release-projection-result"),
            (self.approved_evaluation_item_ref, "evaluation-item"),
            (self.approved_release_decision_ref, "release-decision"),
            (
                self.release_subject_ref,
                "evaluation-item-release-subject",
            ),
            (self.query_spec_ref, "query-spec"),
            (self.rubric_set_ref, "rubric-set"),
            (self.environment_spec_ref, "environment-spec"),
            (self.provenance_manifest_ref, "provenance-manifest"),
            (self.quality_report_ref, "quality-report"),
            (self.final_package_manifest_ref, "final-package-manifest"),
        )
        for ref, object_type in refs:
            _require_ref(ref, object_type, "v2", object_type)
        if self.channel not in {
            ReleaseChannel.CANARY,
            ReleaseChannel.INTERNAL_REVIEW,
        }:
            raise ValueError("R7-09 item manifest must be non-production")
        if "production" in self.registry.casefold():
            raise ValueError("R7-09 item manifest cannot target production")
        _require_sorted_unique_members(self.expected_workspace_members)
        _validate_audit(
            self.audit,
            tuple(ref for ref, _ in refs),
            "LH release item manifest",
        )
        _validate_identity(
            self.item_manifest_id,
            self.item_manifest_sha256,
            "lh-release-item-manifest",
            lh_release_item_manifest_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        job_id: str,
        item_id: str,
        approved_release_result_ref: ObjectRef,
        approved_evaluation_item_ref: ObjectRef,
        approved_release_decision_ref: ObjectRef,
        release_subject_ref: ObjectRef,
        query_spec_ref: ObjectRef,
        rubric_set_ref: ObjectRef,
        environment_spec_ref: ObjectRef,
        provenance_manifest_ref: ObjectRef,
        quality_report_ref: ObjectRef,
        final_package_manifest_ref: ObjectRef,
        package_sha256: str,
        query_yaml_sha256: str,
        rubrics_json_sha256: str,
        expected_workspace_members: tuple[
            LHWorkspaceExportMemberV2,
            ...,
        ],
        channel: ReleaseChannel,
        registry: str,
        audit: ContractAudit,
    ) -> LHReleaseItemManifestV2:
        refs = (
            approved_release_result_ref,
            approved_evaluation_item_ref,
            approved_release_decision_ref,
            release_subject_ref,
            query_spec_ref,
            rubric_set_ref,
            environment_spec_ref,
            provenance_manifest_ref,
            quality_report_ref,
            final_package_manifest_ref,
        )
        value = cls(
            item_manifest_id="lh-release-item-manifest://pending",
            job_id=job_id,
            item_id=item_id,
            approved_release_result_ref=approved_release_result_ref,
            approved_evaluation_item_ref=approved_evaluation_item_ref,
            approved_release_decision_ref=approved_release_decision_ref,
            release_subject_ref=release_subject_ref,
            query_spec_ref=query_spec_ref,
            rubric_set_ref=rubric_set_ref,
            environment_spec_ref=environment_spec_ref,
            provenance_manifest_ref=provenance_manifest_ref,
            quality_report_ref=quality_report_ref,
            final_package_manifest_ref=final_package_manifest_ref,
            package_sha256=package_sha256,
            query_yaml_sha256=query_yaml_sha256,
            rubrics_json_sha256=rubrics_json_sha256,
            expected_workspace_members=_sorted_members(expected_workspace_members),
            channel=channel,
            registry=registry,
            item_manifest_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        return _finalize(
            value,
            "item_manifest_id",
            "item_manifest_sha256",
            "lh-release-item-manifest",
            lh_release_item_manifest_v2_carried_sha256(value),
        )

    def to_ref(self) -> ObjectRef:
        return lh_release_item_manifest_v2_ref(self)


class NonProductionReleaseManifestV2(ContractModelV2):
    schema_version: Literal["eval-factory/nonproduction-release-manifest/v2"] = (
        "eval-factory/nonproduction-release-manifest/v2"
    )
    manifest_id: Identifier
    job_id: Identifier
    approved_item_ids: tuple[Identifier, ...] = Field(min_length=1)
    rejected_item_ids: tuple[Identifier, ...] = ()
    item_manifests: tuple[LHReleaseItemManifestV2, ...] = Field(min_length=1)
    item_manifest_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    channel: ReleaseChannel
    registry: Identifier
    export_profile: Literal["LH"] = "LH"
    export_profile_version: Literal["v1"] = "v1"
    base_contract_manifest_ref: ObjectRef
    overlay_contract_manifest_ref: ObjectRef
    release_profile_decision_ref: ObjectRef
    policy_ref: ObjectRef
    policy_version: Literal["release-publication/r7-09-v1"] = RELEASE_PUBLICATION_POLICY_VERSION
    manifest_sha256: Sha256
    audit: ContractAudit

    @field_validator("channel", mode="before")
    @classmethod
    def parse_channel(cls, value: object) -> ReleaseChannel:
        return _parse_enum(value, ReleaseChannel, "channel")

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        if self.approved_item_ids != tuple(
            sorted(set(self.approved_item_ids))
        ) or self.rejected_item_ids != tuple(sorted(set(self.rejected_item_ids))):
            raise ValueError("release Item partitions must be canonical")
        if set(self.approved_item_ids) & set(self.rejected_item_ids):
            raise ValueError("release Item partitions must be disjoint")
        expected = tuple(sorted(self.item_manifests, key=lambda value: value.item_id))
        if expected != self.item_manifests:
            raise ValueError("release item manifests must be canonical")
        for item_manifest in expected:
            validate_lh_release_item_manifest_v2_identity(item_manifest)
        if tuple(value.item_id for value in expected) != (self.approved_item_ids):
            raise ValueError("release manifests must cover approved Items")
        expected_refs = tuple(value.to_ref() for value in expected)
        if self.item_manifest_refs != expected_refs:
            raise ValueError("release manifest refs are not exact")
        if any(
            value.job_id != self.job_id
            or value.channel is not self.channel
            or value.registry != self.registry
            for value in expected
        ):
            raise ValueError("release item manifest authority is mixed")
        if (
            self.channel
            not in {
                ReleaseChannel.CANARY,
                ReleaseChannel.INTERNAL_REVIEW,
            }
            or "production" in self.registry.casefold()
        ):
            raise ValueError("R7-09 release manifest must be non-production")
        for ref, object_type in (
            (self.base_contract_manifest_ref, "contract-manifest"),
            (self.overlay_contract_manifest_ref, "contract-manifest"),
            (
                self.release_profile_decision_ref,
                "release-profile-decision",
            ),
            (self.policy_ref, "release-publication-policy"),
        ):
            _require_ref(ref, object_type, ref.object_version, object_type)
        _validate_audit(
            self.audit,
            (
                *self.item_manifest_refs,
                self.base_contract_manifest_ref,
                self.overlay_contract_manifest_ref,
                self.release_profile_decision_ref,
                self.policy_ref,
            ),
            "non-production release manifest",
        )
        _validate_identity(
            self.manifest_id,
            self.manifest_sha256,
            "nonproduction-release-manifest",
            nonproduction_release_manifest_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        job_id: str,
        approved_item_ids: tuple[str, ...],
        rejected_item_ids: tuple[str, ...],
        item_manifests: tuple[LHReleaseItemManifestV2, ...],
        channel: ReleaseChannel,
        registry: str,
        base_contract_manifest_ref: ObjectRef,
        overlay_contract_manifest_ref: ObjectRef,
        release_profile_decision_ref: ObjectRef,
        policy_ref: ObjectRef,
        audit: ContractAudit,
    ) -> NonProductionReleaseManifestV2:
        manifests = tuple(sorted(item_manifests, key=lambda value: value.item_id))
        refs = (
            *(value.to_ref() for value in manifests),
            base_contract_manifest_ref,
            overlay_contract_manifest_ref,
            release_profile_decision_ref,
            policy_ref,
        )
        value = cls(
            manifest_id="nonproduction-release-manifest://pending",
            job_id=job_id,
            approved_item_ids=tuple(sorted(set(approved_item_ids))),
            rejected_item_ids=tuple(sorted(set(rejected_item_ids))),
            item_manifests=manifests,
            item_manifest_refs=tuple(value.to_ref() for value in manifests),
            channel=channel,
            registry=registry,
            base_contract_manifest_ref=base_contract_manifest_ref,
            overlay_contract_manifest_ref=overlay_contract_manifest_ref,
            release_profile_decision_ref=release_profile_decision_ref,
            policy_ref=policy_ref,
            manifest_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        return _finalize(
            value,
            "manifest_id",
            "manifest_sha256",
            "nonproduction-release-manifest",
            nonproduction_release_manifest_v2_carried_sha256(value),
        )

    def to_ref(self) -> ObjectRef:
        return nonproduction_release_manifest_v2_ref(self)


class LHExportReceiptV2(ContractModelV2):
    schema_version: Literal["eval-factory/lh-export-receipt/v2"] = "eval-factory/lh-export-receipt/v2"
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
            raise ValueError("LH export receipt requires exported workspace")
        facade_ref = _object_ref(lh_workspace_export_result_ref(self.workspace_export_result))
        if (
            self.workspace_export_result_ref != facade_ref
            or self.workspace_sha256 != self.workspace_export_result.workspace_sha256
        ):
            raise ValueError("LH export workspace result is stale")
        for ref, object_type in (
            (
                self.release_manifest_ref,
                "nonproduction-release-manifest",
            ),
            (self.item_manifest_ref, "lh-release-item-manifest"),
            (
                self.workspace_export_result_ref,
                "lh-workspace-export-result",
            ),
            (self.policy_ref, "release-publication-policy"),
        ):
            _require_ref(ref, object_type, "v2", object_type)
        _validate_audit(
            self.audit,
            (
                self.release_manifest_ref,
                self.item_manifest_ref,
                self.workspace_export_result_ref,
                self.policy_ref,
            ),
            "LH export receipt",
        )
        _validate_identity(
            self.receipt_id,
            self.receipt_sha256,
            "lh-export-receipt",
            lh_export_receipt_v2_carried_sha256(self),
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
    ) -> LHExportReceiptV2:
        result_ref = _object_ref(lh_workspace_export_result_ref(workspace_export_result))
        refs = (
            release_manifest_ref,
            item_manifest_ref,
            result_ref,
            policy_ref,
        )
        value = cls(
            receipt_id="lh-export-receipt://pending",
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
            "lh-export-receipt",
            lh_export_receipt_v2_carried_sha256(value),
        )

    def to_ref(self) -> ObjectRef:
        return lh_export_receipt_v2_ref(self)


class NonProductionRegistryEntryV2(ContractModelV2):
    schema_version: Literal["eval-factory/nonproduction-registry-entry/v2"] = (
        "eval-factory/nonproduction-registry-entry/v2"
    )
    registry_entry_id: Identifier
    release_manifest_ref: ObjectRef
    job_id: Identifier
    channel: ReleaseChannel
    registry: Identifier
    item_ids: tuple[Identifier, ...] = Field(min_length=1)
    item_manifest_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    export_receipt_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    published_release_decision_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    published_evaluation_item_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    bundle_sha256s: tuple[Sha256, ...] = Field(min_length=1)
    published_at: str = Field(min_length=1, max_length=128)
    policy_ref: ObjectRef
    entry_sha256: Sha256
    audit: ContractAudit

    @field_validator("channel", mode="before")
    @classmethod
    def parse_channel(cls, value: object) -> ReleaseChannel:
        return _parse_enum(value, ReleaseChannel, "channel")

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
            raise ValueError("registry entry Item coverage is not exact")
        if (
            self.channel
            not in {
                ReleaseChannel.CANARY,
                ReleaseChannel.INTERNAL_REVIEW,
            }
            or "production" in self.registry.casefold()
        ):
            raise ValueError("R7 registry entry cannot target production")
        _require_ref(
            self.release_manifest_ref,
            "nonproduction-release-manifest",
            "v2",
            "release_manifest_ref",
        )
        for values, object_type, field_name in (
            (
                self.item_manifest_refs,
                "lh-release-item-manifest",
                "item_manifest_refs",
            ),
            (
                self.export_receipt_refs,
                "lh-export-receipt",
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
                _require_ref(ref, object_type, "v2", field_name)
        _require_ref(
            self.policy_ref,
            "release-publication-policy",
            "v2",
            "policy_ref",
        )
        _validate_timestamp(self.published_at)
        _validate_audit(
            self.audit,
            (
                self.release_manifest_ref,
                *self.item_manifest_refs,
                *self.export_receipt_refs,
                *self.published_release_decision_refs,
                *self.published_evaluation_item_refs,
                self.policy_ref,
            ),
            "non-production registry entry",
        )
        _validate_identity(
            self.registry_entry_id,
            self.entry_sha256,
            "nonproduction-registry-entry",
            nonproduction_registry_entry_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        release_manifest_ref: ObjectRef,
        job_id: str,
        channel: ReleaseChannel,
        registry: str,
        item_ids: tuple[str, ...],
        item_manifest_refs: tuple[ObjectRef, ...],
        export_receipt_refs: tuple[ObjectRef, ...],
        published_release_decision_refs: tuple[ObjectRef, ...],
        published_evaluation_item_refs: tuple[ObjectRef, ...],
        bundle_sha256s: tuple[str, ...],
        published_at: str,
        policy_ref: ObjectRef,
        audit: ContractAudit,
    ) -> NonProductionRegistryEntryV2:
        refs = (
            release_manifest_ref,
            *item_manifest_refs,
            *export_receipt_refs,
            *published_release_decision_refs,
            *published_evaluation_item_refs,
            policy_ref,
        )
        value = cls(
            registry_entry_id="nonproduction-registry-entry://pending",
            release_manifest_ref=release_manifest_ref,
            job_id=job_id,
            channel=channel,
            registry=registry,
            item_ids=tuple(sorted(set(item_ids))),
            item_manifest_refs=item_manifest_refs,
            export_receipt_refs=export_receipt_refs,
            published_release_decision_refs=(published_release_decision_refs),
            published_evaluation_item_refs=(published_evaluation_item_refs),
            bundle_sha256s=bundle_sha256s,
            published_at=published_at,
            policy_ref=policy_ref,
            entry_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        return _finalize(
            value,
            "registry_entry_id",
            "entry_sha256",
            "nonproduction-registry-entry",
            nonproduction_registry_entry_v2_carried_sha256(value),
        )

    def to_ref(self) -> ObjectRef:
        return nonproduction_registry_entry_v2_ref(self)


class PublishedItemProjectionV2(ContractModelV2):
    schema_version: Literal["eval-factory/published-item-projection/v2"] = (
        "eval-factory/published-item-projection/v2"
    )
    projection_id: Identifier
    job_id: Identifier
    item_id: Identifier
    chain_id: Identifier
    approved_result_ref: ObjectRef
    approved_projection_ref: ObjectRef
    release_manifest_ref: ObjectRef
    export_receipt_ref: ObjectRef
    registry_entry_ref: ObjectRef
    publish_decision_ref: ObjectRef
    published_evaluation_item_ref: ObjectRef
    item_status: Literal[ItemStatus.RELEASED] = ItemStatus.RELEASED
    policy_version: Literal["release-publication/r7-09-v1"] = RELEASE_PUBLICATION_POLICY_VERSION
    projection_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_projection(self) -> Self:
        for ref, object_type in (
            (self.approved_result_ref, "release-projection-result"),
            (self.approved_projection_ref, "item-release-projection"),
            (
                self.release_manifest_ref,
                "nonproduction-release-manifest",
            ),
            (self.export_receipt_ref, "lh-export-receipt"),
            (
                self.registry_entry_ref,
                "nonproduction-registry-entry",
            ),
            (self.publish_decision_ref, "release-decision"),
            (
                self.published_evaluation_item_ref,
                "evaluation-item",
            ),
        ):
            _require_ref(ref, object_type, "v2", object_type)
        _validate_audit(
            self.audit,
            (
                self.approved_result_ref,
                self.approved_projection_ref,
                self.release_manifest_ref,
                self.export_receipt_ref,
                self.registry_entry_ref,
                self.publish_decision_ref,
                self.published_evaluation_item_ref,
            ),
            "published Item projection",
        )
        _validate_identity(
            self.projection_id,
            self.projection_sha256,
            "published-item-projection",
            published_item_projection_v2_carried_sha256(self),
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
        release_manifest_ref: ObjectRef,
        export_receipt_ref: ObjectRef,
        registry_entry_ref: ObjectRef,
        publish_decision_ref: ObjectRef,
        published_evaluation_item_ref: ObjectRef,
        audit: ContractAudit,
    ) -> PublishedItemProjectionV2:
        refs = (
            approved_result_ref,
            approved_projection_ref,
            release_manifest_ref,
            export_receipt_ref,
            registry_entry_ref,
            publish_decision_ref,
            published_evaluation_item_ref,
        )
        value = cls(
            projection_id="published-item-projection://pending",
            job_id=job_id,
            item_id=item_id,
            chain_id=chain_id,
            approved_result_ref=approved_result_ref,
            approved_projection_ref=approved_projection_ref,
            release_manifest_ref=release_manifest_ref,
            export_receipt_ref=export_receipt_ref,
            registry_entry_ref=registry_entry_ref,
            publish_decision_ref=publish_decision_ref,
            published_evaluation_item_ref=published_evaluation_item_ref,
            projection_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        return _finalize(
            value,
            "projection_id",
            "projection_sha256",
            "published-item-projection",
            published_item_projection_v2_carried_sha256(value),
        )

    def to_ref(self) -> ObjectRef:
        return published_item_projection_v2_ref(self)


class ReleasePublicationResultV2(ContractModelV2):
    schema_version: Literal["eval-factory/release-publication-result/v2"] = (
        "eval-factory/release-publication-result/v2"
    )
    result_id: Identifier
    phase: Literal[ReleasePublicationPhaseV2.PUBLISHED] = ReleasePublicationPhaseV2.PUBLISHED
    release_manifest: NonProductionReleaseManifestV2
    export_receipts: tuple[LHExportReceiptV2, ...] = Field(min_length=1)
    published_decisions: tuple[ReleaseDecisionV2, ...] = Field(min_length=1)
    published_items: tuple[EvaluationItemV2, ...] = Field(min_length=1)
    item_projections: tuple[PublishedItemProjectionV2, ...] = Field(min_length=1)
    registry_entry: NonProductionRegistryEntryV2
    approved_source_result_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    policy_ref: ObjectRef
    result_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        validate_nonproduction_release_manifest_v2_identity(self.release_manifest)
        validate_nonproduction_registry_entry_v2_identity(self.registry_entry)
        count = len(self.release_manifest.approved_item_ids)
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
            raise ValueError("publication result coverage is not exact")
        for ref in self.approved_source_result_refs:
            _require_ref(
                ref,
                "release-projection-result",
                "v2",
                "approved_source_result_refs",
            )
        for decision in self.published_decisions:
            validate_release_decision_v2_identity(decision)
            if (
                decision.action is not ReleaseActionV2.PUBLISH
                or decision.state is not ReleaseStateV2.RELEASED
                or decision.production_attestation_ref is not None
            ):
                raise ValueError("publication requires non-production PUBLISH decisions")
        for item in self.published_items:
            validate_evaluation_item_v2_identity(item)
        for receipt in self.export_receipts:
            validate_lh_export_receipt_v2_identity(receipt)
        for projection in self.item_projections:
            validate_published_item_projection_v2_identity(projection)
        if self.approved_source_result_refs != tuple(
            item.approved_release_result_ref for item in self.release_manifest.item_manifests
        ):
            raise ValueError("publication source refs differ from manifest")
        if self.registry_entry.item_ids != self.release_manifest.approved_item_ids:
            raise ValueError("registry entry differs from publication")
        _validate_publication_bindings(self)
        _validate_audit(
            self.audit,
            (
                self.release_manifest.to_ref(),
                *(value.to_ref() for value in self.export_receipts),
                *(release_decision_v2_ref(value) for value in self.published_decisions),
                *(evaluation_item_v2_ref(value) for value in self.published_items),
                *(value.to_ref() for value in self.item_projections),
                self.registry_entry.to_ref(),
                *self.approved_source_result_refs,
                self.policy_ref,
            ),
            "release publication result",
        )
        _validate_identity(
            self.result_id,
            self.result_sha256,
            "release-publication-result",
            release_publication_result_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        release_manifest: NonProductionReleaseManifestV2,
        export_receipts: tuple[LHExportReceiptV2, ...],
        published_decisions: tuple[ReleaseDecisionV2, ...],
        published_items: tuple[EvaluationItemV2, ...],
        item_projections: tuple[PublishedItemProjectionV2, ...],
        registry_entry: NonProductionRegistryEntryV2,
        approved_source_result_refs: tuple[ObjectRef, ...],
        policy_ref: ObjectRef,
        audit: ContractAudit,
    ) -> ReleasePublicationResultV2:
        refs = (
            release_manifest.to_ref(),
            *(value.to_ref() for value in export_receipts),
            *(release_decision_v2_ref(value) for value in published_decisions),
            *(evaluation_item_v2_ref(value) for value in published_items),
            *(value.to_ref() for value in item_projections),
            registry_entry.to_ref(),
            *approved_source_result_refs,
            policy_ref,
        )
        value = cls(
            result_id="release-publication-result://pending",
            release_manifest=release_manifest,
            export_receipts=export_receipts,
            published_decisions=published_decisions,
            published_items=published_items,
            item_projections=item_projections,
            registry_entry=registry_entry,
            approved_source_result_refs=approved_source_result_refs,
            policy_ref=policy_ref,
            result_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        return _finalize(
            value,
            "result_id",
            "result_sha256",
            "release-publication-result",
            release_publication_result_v2_carried_sha256(value),
        )

    def to_ref(self) -> ObjectRef:
        return release_publication_result_v2_ref(self)


def release_publication_policy_v2_carried_sha256(
    value: ReleasePublicationPolicyV2,
) -> str:
    return _carried(value, {"policy_id", "policy_sha256", "audit"})


def release_publication_policy_v2_ref(
    value: ReleasePublicationPolicyV2,
) -> ObjectRef:
    validate_release_publication_policy_v2_identity(value)
    return _ref(
        "release-publication-policy",
        value.policy_id,
        value.policy_sha256,
    )


def lh_release_item_manifest_v2_carried_sha256(
    value: LHReleaseItemManifestV2,
) -> str:
    return _carried(
        value,
        {"item_manifest_id", "item_manifest_sha256", "audit"},
    )


def lh_release_item_manifest_v2_ref(
    value: LHReleaseItemManifestV2,
) -> ObjectRef:
    validate_lh_release_item_manifest_v2_identity(value)
    return _ref(
        "lh-release-item-manifest",
        value.item_manifest_id,
        value.item_manifest_sha256,
    )


def nonproduction_release_manifest_v2_carried_sha256(
    value: NonProductionReleaseManifestV2,
) -> str:
    return _carried(value, {"manifest_id", "manifest_sha256", "audit"})


def nonproduction_release_manifest_v2_ref(
    value: NonProductionReleaseManifestV2,
) -> ObjectRef:
    validate_nonproduction_release_manifest_v2_identity(value)
    return _ref(
        "nonproduction-release-manifest",
        value.manifest_id,
        value.manifest_sha256,
    )


def lh_export_receipt_v2_carried_sha256(
    value: LHExportReceiptV2,
) -> str:
    return _carried(value, {"receipt_id", "receipt_sha256", "audit"})


def lh_export_receipt_v2_ref(value: LHExportReceiptV2) -> ObjectRef:
    validate_lh_export_receipt_v2_identity(value)
    return _ref(
        "lh-export-receipt",
        value.receipt_id,
        value.receipt_sha256,
    )


def nonproduction_registry_entry_v2_carried_sha256(
    value: NonProductionRegistryEntryV2,
) -> str:
    return _carried(
        value,
        {"registry_entry_id", "entry_sha256", "audit"},
    )


def nonproduction_registry_entry_v2_ref(
    value: NonProductionRegistryEntryV2,
) -> ObjectRef:
    validate_nonproduction_registry_entry_v2_identity(value)
    return _ref(
        "nonproduction-registry-entry",
        value.registry_entry_id,
        value.entry_sha256,
    )


def published_item_projection_v2_carried_sha256(
    value: PublishedItemProjectionV2,
) -> str:
    return _carried(
        value,
        {"projection_id", "projection_sha256", "audit"},
    )


def published_item_projection_v2_ref(
    value: PublishedItemProjectionV2,
) -> ObjectRef:
    validate_published_item_projection_v2_identity(value)
    return _ref(
        "published-item-projection",
        value.projection_id,
        value.projection_sha256,
    )


def release_publication_result_v2_carried_sha256(
    value: ReleasePublicationResultV2,
) -> str:
    return _carried(value, {"result_id", "result_sha256", "audit"})


def release_publication_result_v2_ref(
    value: ReleasePublicationResultV2,
) -> ObjectRef:
    validate_release_publication_result_v2_identity(value)
    return _ref(
        "release-publication-result",
        value.result_id,
        value.result_sha256,
    )


def validate_release_publication_policy_v2_identity(
    value: ReleasePublicationPolicyV2,
) -> None:
    _validate_identity(
        value.policy_id,
        value.policy_sha256,
        "release-publication-policy",
        release_publication_policy_v2_carried_sha256(value),
        allow_pending=False,
    )


def validate_lh_release_item_manifest_v2_identity(
    value: LHReleaseItemManifestV2,
) -> None:
    _validate_identity(
        value.item_manifest_id,
        value.item_manifest_sha256,
        "lh-release-item-manifest",
        lh_release_item_manifest_v2_carried_sha256(value),
        allow_pending=False,
    )


def validate_nonproduction_release_manifest_v2_identity(
    value: NonProductionReleaseManifestV2,
) -> None:
    _validate_identity(
        value.manifest_id,
        value.manifest_sha256,
        "nonproduction-release-manifest",
        nonproduction_release_manifest_v2_carried_sha256(value),
        allow_pending=False,
    )


def validate_lh_export_receipt_v2_identity(
    value: LHExportReceiptV2,
) -> None:
    _validate_identity(
        value.receipt_id,
        value.receipt_sha256,
        "lh-export-receipt",
        lh_export_receipt_v2_carried_sha256(value),
        allow_pending=False,
    )


def validate_nonproduction_registry_entry_v2_identity(
    value: NonProductionRegistryEntryV2,
) -> None:
    _validate_identity(
        value.registry_entry_id,
        value.entry_sha256,
        "nonproduction-registry-entry",
        nonproduction_registry_entry_v2_carried_sha256(value),
        allow_pending=False,
    )


def validate_published_item_projection_v2_identity(
    value: PublishedItemProjectionV2,
) -> None:
    _validate_identity(
        value.projection_id,
        value.projection_sha256,
        "published-item-projection",
        published_item_projection_v2_carried_sha256(value),
        allow_pending=False,
    )


def validate_release_publication_result_v2_identity(
    value: ReleasePublicationResultV2,
) -> None:
    _validate_identity(
        value.result_id,
        value.result_sha256,
        "release-publication-result",
        release_publication_result_v2_carried_sha256(value),
        allow_pending=False,
    )


def _validate_publication_bindings(
    value: ReleasePublicationResultV2,
) -> None:
    manifest = value.release_manifest
    manifest_ref = manifest.to_ref()
    registry = value.registry_entry
    registry_ref = registry.to_ref()
    receipt_refs = tuple(receipt.to_ref() for receipt in value.export_receipts)
    decision_refs = tuple(release_decision_v2_ref(decision) for decision in value.published_decisions)
    item_refs = tuple(evaluation_item_v2_ref(item) for item in value.published_items)
    if (
        value.policy_ref != manifest.policy_ref
        or registry.policy_ref != value.policy_ref
        or registry.release_manifest_ref != manifest_ref
        or registry.job_id != manifest.job_id
        or registry.channel is not manifest.channel
        or registry.registry != manifest.registry
        or registry.item_manifest_refs != manifest.item_manifest_refs
        or registry.export_receipt_refs != receipt_refs
        or registry.published_release_decision_refs != decision_refs
        or registry.published_evaluation_item_refs != item_refs
        or registry.bundle_sha256s != tuple(receipt.bundle_sha256 for receipt in value.export_receipts)
    ):
        raise ValueError("publication registry bindings are not exact")

    for item_manifest, receipt, decision, item, projection, source_ref in zip(
        manifest.item_manifests,
        value.export_receipts,
        value.published_decisions,
        value.published_items,
        value.item_projections,
        value.approved_source_result_refs,
        strict=True,
    ):
        receipt_ref = receipt.to_ref()
        decision_ref = release_decision_v2_ref(decision)
        item_ref = evaluation_item_v2_ref(item)
        expected_output_refs = tuple(
            ref
            for _, ref in sorted(
                {
                    (
                        member.output_ref.object_type,
                        member.output_ref.object_id,
                        member.output_ref.object_version,
                        member.output_ref.object_sha256,
                    ): member.output_ref
                    for member in item_manifest.expected_workspace_members
                }.items()
            )
        )
        if (
            item_manifest.job_id != manifest.job_id
            or item_manifest.approved_release_result_ref != source_ref
            or item_manifest.item_id != decision.item_id
            or item_manifest.query_spec_ref != decision.components.query_spec_ref
            or item_manifest.rubric_set_ref != decision.components.rubric_set_ref
            or item_manifest.environment_spec_ref != decision.components.environment_spec_ref
            or item_manifest.provenance_manifest_ref != decision.components.provenance_manifest_ref
            or item_manifest.quality_report_ref != decision.components.quality_report_ref
            or item_manifest.final_package_manifest_ref != decision.package_manifest_ref
            or item_manifest.package_sha256 != decision.package_sha256
        ):
            raise ValueError("publication item manifest differs from approved authority")
        if (
            receipt.release_manifest_ref != manifest_ref
            or receipt.item_manifest_ref != item_manifest.to_ref()
            or receipt.query_yaml_sha256 != item_manifest.query_yaml_sha256
            or receipt.rubrics_json_sha256 != item_manifest.rubrics_json_sha256
            or receipt.workspace_export_result.observed_members != item_manifest.expected_workspace_members
            or receipt.workspace_export_result.copied_output_refs != expected_output_refs
            or receipt.policy_ref != value.policy_ref
        ):
            raise ValueError("publication receipt differs from manifest authority")
        if (
            decision.previous_decision_ref != item_manifest.approved_release_decision_ref
            or decision.channel is not item_manifest.channel
            or decision.registry != item_manifest.registry
            or decision.export_profile != item_manifest.export_profile
            or decision.export_profile_version != item_manifest.export_profile_version
            or decision.production_attestation_ref is not None
            or decision.decided_at.isoformat() != registry.published_at
            or decision.audit.input_refs
            != tuple(
                sorted(
                    {
                        item_manifest.release_subject_ref,
                        item_manifest.approved_release_decision_ref,
                        item_manifest.approved_evaluation_item_ref,
                    },
                    key=_ref_key,
                )
            )
        ):
            raise ValueError("PUBLISH decision differs from approved authority")
        if (
            item.item_version != decision.item_version
            or item.components != decision.components
            or item.user_approval_policy_ref != decision.user_approval_policy_ref
            or item.release_decision_ref != decision_ref
            or item.audit.input_refs
            != tuple(
                sorted(
                    {
                        item_manifest.approved_evaluation_item_ref,
                        decision_ref,
                    },
                    key=_ref_key,
                )
            )
        ):
            raise ValueError("published EvaluationItem differs from approved Item")
        if (
            projection.job_id != manifest.job_id
            or projection.item_id != item_manifest.item_id
            or projection.chain_id != decision.chain_id
            or projection.approved_result_ref != source_ref
            or projection.release_manifest_ref != manifest_ref
            or projection.export_receipt_ref != receipt_ref
            or projection.registry_entry_ref != registry_ref
            or projection.publish_decision_ref != decision_ref
            or projection.published_evaluation_item_ref != item_ref
        ):
            raise ValueError("published Item projection bindings are not exact")


def _ref(
    object_type: str,
    object_id: str,
    object_sha256: str,
) -> ObjectRef:
    if object_id.endswith("://pending") or object_sha256 == "0" * 64:
        raise ValueError(f"{object_type} identity is pending")
    return ObjectRef(
        object_type=object_type,
        object_id=object_id,
        object_version="v2",
        object_sha256=object_sha256,
    )


def _object_ref(value: FacadeObjectRef) -> ObjectRef:
    return ObjectRef(
        object_type=value.object_type,
        object_id=value.object_id,
        object_version=value.object_version,
        object_sha256=value.object_sha256,
    )


def _carried(value: ContractModelV2, exclude: set[str]) -> str:
    return _payload_sha256(
        value.model_dump(
            mode="python",
            exclude=exclude,
            exclude_none=False,
        )
    )


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


def _finalize[ModelT: ContractModelV2](
    value: ModelT,
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


def _require_ref(
    value: ObjectRef,
    object_type: str,
    object_version: str,
    field_name: str,
) -> None:
    if value.object_type != object_type or value.object_version != object_version:
        raise ValueError(f"{field_name} must reference {object_type} {object_version}")


def _safe_audit(
    audit: ContractAudit,
    refs: tuple[ObjectRef, ...],
) -> ContractAudit:
    return audit.model_copy(update={"input_refs": tuple(sorted(set(refs), key=_ref_key))})


def _validate_audit(
    audit: ContractAudit,
    refs: tuple[ObjectRef, ...],
    label: str,
) -> None:
    if audit.input_refs != tuple(sorted(set(refs), key=_ref_key)):
        raise ValueError(f"{label} audit refs are incomplete")


def _validate_timestamp(value: str) -> None:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("published_at must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("published_at must be timezone-aware")


def _ref_key(value: ObjectRef) -> tuple[str, str, str, str]:
    return (
        value.object_type,
        value.object_id,
        value.object_version,
        value.object_sha256,
    )


def _member_key(
    value: LHWorkspaceExportMemberV2,
) -> tuple[str, str, str, str]:
    return (
        value.container_ref.object_id if value.container_ref is not None else "",
        value.normalized_path,
        value.member_type.value,
        value.output_ref.object_id,
    )


def _sorted_members(
    values: tuple[LHWorkspaceExportMemberV2, ...],
) -> tuple[LHWorkspaceExportMemberV2, ...]:
    return tuple(sorted(values, key=_member_key))


def _require_sorted_unique_members(
    values: tuple[LHWorkspaceExportMemberV2, ...],
) -> None:
    if values != _sorted_members(values) or len({_member_key(value) for value in values}) != len(values):
        raise ValueError("workspace members must be sorted and unique")


def _parse_enum[EnumT: StrEnum](
    value: object,
    enum_type: type[EnumT],
    field_name: str,
) -> EnumT:
    if isinstance(value, enum_type):
        return value
    if isinstance(value, str):
        return enum_type(value)
    raise TypeError(f"{field_name} must be a {enum_type.__name__}")


__all__ = [
    "RELEASE_PUBLICATION_POLICY_VERSION",
    "LHExportReceiptV2",
    "LHReleaseItemManifestV2",
    "NonProductionRegistryEntryV2",
    "NonProductionReleaseManifestV2",
    "PublishedItemProjectionV2",
    "ReleasePublicationPhaseV2",
    "ReleasePublicationPolicyV2",
    "ReleasePublicationResultV2",
    "lh_export_receipt_v2_ref",
    "lh_release_item_manifest_v2_ref",
    "nonproduction_registry_entry_v2_ref",
    "nonproduction_release_manifest_v2_ref",
    "published_item_projection_v2_ref",
    "release_publication_policy_v2_ref",
    "release_publication_result_v2_ref",
    "validate_lh_export_receipt_v2_identity",
    "validate_lh_release_item_manifest_v2_identity",
    "validate_nonproduction_registry_entry_v2_identity",
    "validate_nonproduction_release_manifest_v2_identity",
    "validate_published_item_projection_v2_identity",
    "validate_release_publication_policy_v2_identity",
    "validate_release_publication_result_v2_identity",
]
