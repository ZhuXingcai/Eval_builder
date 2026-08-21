from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Literal, Self
from uuid import uuid4

from pydantic import Field, ValidationError, model_validator

from eval_factory.contracts.core import Identifier, ObjectRef, Sha256
from eval_factory.contracts.core_v2 import ContractModelV2, canonical_value_v2
from eval_factory.contracts.production_release_v2 import (
    ProductionAttestationAuthorityV2,
    ProductionLHExportReceiptV2,
    ProductionPublishedItemProjectionV2,
    ProductionRegistryEntryV2,
    ProductionReleaseItemManifestV2,
    ProductionReleaseManifestV2,
    ProductionReleaseOutcomeV2,
    ProductionReleasePolicyV2,
    ProductionReleaseResultV2,
    production_attestation_authority_v2_ref,
    production_lh_export_receipt_v2_ref,
    production_published_item_projection_v2_ref,
    production_registry_entry_v2_ref,
    production_release_item_manifest_v2_ref,
    production_release_manifest_v2_ref,
    production_release_policy_v2_ref,
    production_release_result_v2_ref,
    validate_production_attestation_authority_v2_identity,
    validate_production_lh_export_receipt_v2_identity,
    validate_production_published_item_projection_v2_identity,
    validate_production_registry_entry_v2_identity,
    validate_production_release_item_manifest_v2_identity,
    validate_production_release_manifest_v2_identity,
    validate_production_release_policy_v2_identity,
    validate_production_release_result_v2_identity,
)
from eval_factory.dataset.production_release import (
    ProductionReleaseCompilation,
    ProductionReleaseCompiler,
    ProductionReleaseConflictError,
    ProductionReleaseIntegrityError,
)
from eval_factory.storage import (
    ContentAddressedByteStore,
    ContentAddressedByteStoreConflictError,
    ContentAddressedByteStoreCorruptionError,
)


class ProductionReleaseReportCodec(StrEnum):
    POLICY = "policy"
    EVIDENCE = "evidence"
    ATTESTATION_AUTHORITY = "attestation-authority"
    ITEM_MANIFEST = "item-manifest"
    RELEASE_MANIFEST = "release-manifest"
    RECEIPT = "receipt"
    REGISTRY_ENTRY = "registry-entry"
    PUBLISHED_PROJECTION = "published-projection"
    RESULT = "result"


class _ProductionReleaseEnvelopeV1(ContractModelV2):
    schema_version: Literal["eval-factory/production-release-envelope/private-v1"] = (
        "eval-factory/production-release-envelope/private-v1"
    )
    codec: ProductionReleaseReportCodec
    object_ref: ObjectRef
    content_ref: ObjectRef
    canonical_size_bytes: int = Field(ge=2)

    @model_validator(mode="after")
    def validate_envelope(self) -> Self:
        if (
            self.content_ref.object_type != "production-release-content"
            or self.content_ref.object_version != "private-v1"
            or self.content_ref.object_id
            != f"production-release-content://sha256/{self.content_ref.object_sha256}"
        ):
            raise ValueError("production release content ref is invalid")
        expected_type = {
            ProductionReleaseReportCodec.POLICY: "production-release-policy",
            ProductionReleaseReportCodec.EVIDENCE: ("production-release-repository-evidence"),
            ProductionReleaseReportCodec.ATTESTATION_AUTHORITY: ("production-attestation-authority"),
            ProductionReleaseReportCodec.ITEM_MANIFEST: ("production-release-item-manifest"),
            ProductionReleaseReportCodec.RELEASE_MANIFEST: "production-release-manifest",
            ProductionReleaseReportCodec.RECEIPT: "production-lh-export-receipt",
            ProductionReleaseReportCodec.REGISTRY_ENTRY: "production-registry-entry",
            ProductionReleaseReportCodec.PUBLISHED_PROJECTION: ("production-published-item-projection"),
            ProductionReleaseReportCodec.RESULT: "production-release-result",
        }[self.codec]
        if self.object_ref.object_type != expected_type:
            raise ValueError("production release envelope codec is invalid")
        return self


class ProductionReleaseAcceptedRunV1(ContractModelV2):
    schema_version: Literal["eval-factory/production-release-accepted-run/private-v1"] = (
        "eval-factory/production-release-accepted-run/private-v1"
    )
    accepted_run_id: Identifier
    acceptance_key: Identifier
    request_sha256: Sha256
    policy_ref: ObjectRef
    repository_evidence_ref: ObjectRef
    result_ref: ObjectRef
    accepted_run_sha256: Sha256

    @model_validator(mode="after")
    def validate_run(self) -> Self:
        expected = _accepted_sha256(
            acceptance_key=self.acceptance_key,
            request_sha256=self.request_sha256,
            policy_ref=self.policy_ref,
            repository_evidence_ref=self.repository_evidence_ref,
            result_ref=self.result_ref,
        )
        if (
            self.policy_ref.object_type != "production-release-policy"
            or self.repository_evidence_ref.object_type != "production-release-repository-evidence"
            or self.result_ref.object_type != "production-release-result"
            or self.accepted_run_sha256 != expected
            or self.accepted_run_id != f"production-release-accepted-run://sha256/{expected}"
        ):
            raise ValueError("production release accepted run identity is stale")
        return self

    @classmethod
    def create(
        cls,
        *,
        acceptance_key: str,
        request_sha256: str,
        policy_ref: ObjectRef,
        repository_evidence_ref: ObjectRef,
        result_ref: ObjectRef,
    ) -> ProductionReleaseAcceptedRunV1:
        digest = _accepted_sha256(
            acceptance_key=acceptance_key,
            request_sha256=request_sha256,
            policy_ref=policy_ref,
            repository_evidence_ref=repository_evidence_ref,
            result_ref=result_ref,
        )
        return cls(
            accepted_run_id=(f"production-release-accepted-run://sha256/{digest}"),
            acceptance_key=acceptance_key,
            request_sha256=request_sha256,
            policy_ref=policy_ref,
            repository_evidence_ref=repository_evidence_ref,
            result_ref=result_ref,
            accepted_run_sha256=digest,
        )


class ProductionReleaseReportStore:
    def __init__(
        self,
        root: Path,
        *,
        max_report_bytes: int,
    ) -> None:
        if max_report_bytes < 2:
            raise ProductionReleaseIntegrityError("production release report limit is invalid")
        resolved = _prepare_store_root(root)
        self.root = resolved
        self.max_report_bytes = max_report_bytes
        self._cas = ContentAddressedByteStore(resolved / "cas")
        self._envelopes = resolved / "envelopes"
        self._acceptances = resolved / "acceptances"

    def put_policy(self, value: ProductionReleasePolicyV2) -> ObjectRef:
        ref = value.to_ref()
        return self._put_contract(
            ProductionReleaseReportCodec.POLICY,
            value=value,
            ref=ref,
            validate=validate_production_release_policy_v2_identity,
        )

    def put_attestation_authority(
        self,
        value: ProductionAttestationAuthorityV2,
    ) -> ObjectRef:
        ref = value.to_ref()
        return self._put_contract(
            ProductionReleaseReportCodec.ATTESTATION_AUTHORITY,
            value=value,
            ref=ref,
            validate=validate_production_attestation_authority_v2_identity,
        )

    def put_item_manifest(
        self,
        value: ProductionReleaseItemManifestV2,
    ) -> ObjectRef:
        ref = value.to_ref()
        return self._put_contract(
            ProductionReleaseReportCodec.ITEM_MANIFEST,
            value=value,
            ref=ref,
            validate=validate_production_release_item_manifest_v2_identity,
        )

    def put_release_manifest(
        self,
        value: ProductionReleaseManifestV2,
    ) -> ObjectRef:
        ref = value.to_ref()
        return self._put_contract(
            ProductionReleaseReportCodec.RELEASE_MANIFEST,
            value=value,
            ref=ref,
            validate=validate_production_release_manifest_v2_identity,
        )

    def put_receipt(
        self,
        value: ProductionLHExportReceiptV2,
    ) -> ObjectRef:
        ref = value.to_ref()
        return self._put_contract(
            ProductionReleaseReportCodec.RECEIPT,
            value=value,
            ref=ref,
            validate=validate_production_lh_export_receipt_v2_identity,
        )

    def put_registry_entry(
        self,
        value: ProductionRegistryEntryV2,
    ) -> ObjectRef:
        ref = value.to_ref()
        return self._put_contract(
            ProductionReleaseReportCodec.REGISTRY_ENTRY,
            value=value,
            ref=ref,
            validate=validate_production_registry_entry_v2_identity,
        )

    def put_published_projection(
        self,
        value: ProductionPublishedItemProjectionV2,
    ) -> ObjectRef:
        ref = value.to_ref()
        return self._put_contract(
            ProductionReleaseReportCodec.PUBLISHED_PROJECTION,
            value=value,
            ref=ref,
            validate=validate_production_published_item_projection_v2_identity,
        )

    def put_result(self, value: ProductionReleaseResultV2) -> ObjectRef:
        ref = value.to_ref()
        return self._put_contract(
            ProductionReleaseReportCodec.RESULT,
            value=value,
            ref=ref,
            validate=validate_production_release_result_v2_identity,
        )

    def get_policy(self, ref: ObjectRef) -> ProductionReleasePolicyV2:
        return self._get_contract(
            ProductionReleaseReportCodec.POLICY,
            ref=ref,
            model_type=ProductionReleasePolicyV2,
            validate=validate_production_release_policy_v2_identity,
            resolve_ref=production_release_policy_v2_ref,
        )

    def get_attestation_authority(
        self,
        ref: ObjectRef,
    ) -> ProductionAttestationAuthorityV2:
        return self._get_contract(
            ProductionReleaseReportCodec.ATTESTATION_AUTHORITY,
            ref=ref,
            model_type=ProductionAttestationAuthorityV2,
            validate=validate_production_attestation_authority_v2_identity,
            resolve_ref=production_attestation_authority_v2_ref,
        )

    def get_item_manifest(
        self,
        ref: ObjectRef,
    ) -> ProductionReleaseItemManifestV2:
        return self._get_contract(
            ProductionReleaseReportCodec.ITEM_MANIFEST,
            ref=ref,
            model_type=ProductionReleaseItemManifestV2,
            validate=validate_production_release_item_manifest_v2_identity,
            resolve_ref=production_release_item_manifest_v2_ref,
        )

    def get_release_manifest(
        self,
        ref: ObjectRef,
    ) -> ProductionReleaseManifestV2:
        return self._get_contract(
            ProductionReleaseReportCodec.RELEASE_MANIFEST,
            ref=ref,
            model_type=ProductionReleaseManifestV2,
            validate=validate_production_release_manifest_v2_identity,
            resolve_ref=production_release_manifest_v2_ref,
        )

    def get_receipt(
        self,
        ref: ObjectRef,
    ) -> ProductionLHExportReceiptV2:
        return self._get_contract(
            ProductionReleaseReportCodec.RECEIPT,
            ref=ref,
            model_type=ProductionLHExportReceiptV2,
            validate=validate_production_lh_export_receipt_v2_identity,
            resolve_ref=production_lh_export_receipt_v2_ref,
        )

    def get_registry_entry(
        self,
        ref: ObjectRef,
    ) -> ProductionRegistryEntryV2:
        return self._get_contract(
            ProductionReleaseReportCodec.REGISTRY_ENTRY,
            ref=ref,
            model_type=ProductionRegistryEntryV2,
            validate=validate_production_registry_entry_v2_identity,
            resolve_ref=production_registry_entry_v2_ref,
        )

    def get_published_projection(
        self,
        ref: ObjectRef,
    ) -> ProductionPublishedItemProjectionV2:
        return self._get_contract(
            ProductionReleaseReportCodec.PUBLISHED_PROJECTION,
            ref=ref,
            model_type=ProductionPublishedItemProjectionV2,
            validate=validate_production_published_item_projection_v2_identity,
            resolve_ref=production_published_item_projection_v2_ref,
        )

    def get_result(self, ref: ObjectRef) -> ProductionReleaseResultV2:
        return self._get_contract(
            ProductionReleaseReportCodec.RESULT,
            ref=ref,
            model_type=ProductionReleaseResultV2,
            validate=validate_production_release_result_v2_identity,
            resolve_ref=production_release_result_v2_ref,
        )

    def persist_repository(
        self,
        *,
        acceptance_key: str,
        request_sha256: str,
        payload: bytes,
        compilation: ProductionReleaseCompilation,
    ) -> ProductionReleaseAcceptedRunV1:
        policy = compilation.policy
        result = compilation.result
        if policy.repository_evidence_ref is None or compilation.repository_evidence is None:
            raise ProductionReleaseIntegrityError("repository compilation closure is incomplete")
        expected = ProductionReleaseCompiler().compile_repository_blocked(
            payload=payload,
            policy=policy,
        )
        if expected != compilation:
            raise ProductionReleaseIntegrityError("repository result differs from deterministic compilation")
        accepted = ProductionReleaseAcceptedRunV1.create(
            acceptance_key=acceptance_key,
            request_sha256=request_sha256,
            policy_ref=policy.to_ref(),
            repository_evidence_ref=policy.repository_evidence_ref,
            result_ref=result.to_ref(),
        )
        existing = self._read_acceptance_optional(acceptance_key)
        if existing is not None:
            if existing.request_sha256 != request_sha256:
                raise ProductionReleaseConflictError("accepted production release request changed")
            if existing != accepted:
                raise ProductionReleaseConflictError("accepted production release authority changed")
            self._verify_repository(existing)
            return existing
        self._put_model(
            ProductionReleaseReportCodec.POLICY,
            policy.to_ref(),
            policy.canonical_json(),
        )
        self._put_bytes(
            ProductionReleaseReportCodec.EVIDENCE,
            policy.repository_evidence_ref,
            payload,
        )
        self._put_model(
            ProductionReleaseReportCodec.RESULT,
            result.to_ref(),
            result.canonical_json(),
        )
        authoritative = self._write_acceptance(accepted)
        if authoritative != accepted:
            raise ProductionReleaseConflictError("production release first authority conflicted")
        self._verify_repository(authoritative)
        return authoritative

    def get_accepted_result(
        self,
        acceptance_key: str,
    ) -> ProductionReleaseResultV2:
        accepted = self._read_acceptance(acceptance_key)
        return self._verify_repository(accepted)

    def _verify_repository(
        self,
        accepted: ProductionReleaseAcceptedRunV1,
    ) -> ProductionReleaseResultV2:
        policy_payload = self._get_bytes(
            ProductionReleaseReportCodec.POLICY,
            accepted.policy_ref,
        )
        result_payload = self._get_bytes(
            ProductionReleaseReportCodec.RESULT,
            accepted.result_ref,
        )
        evidence_payload = self._get_bytes(
            ProductionReleaseReportCodec.EVIDENCE,
            accepted.repository_evidence_ref,
        )
        try:
            policy = ProductionReleasePolicyV2.model_validate_json(policy_payload)
            validate_production_release_policy_v2_identity(policy)
            result = ProductionReleaseResultV2.model_validate_json(result_payload)
            validate_production_release_result_v2_identity(result)
        except (ValidationError, ValueError) as exc:
            raise ProductionReleaseIntegrityError("stored production release value is invalid") from exc
        if (
            production_release_policy_v2_ref(policy) != accepted.policy_ref
            or production_release_result_v2_ref(result) != accepted.result_ref
            or policy.repository_evidence_ref != accepted.repository_evidence_ref
            or policy.canonical_json() != policy_payload
            or result.canonical_json() != result_payload
        ):
            raise ProductionReleaseIntegrityError("stored production release closure differs from acceptance")
        expected = ProductionReleaseCompiler().compile_repository_blocked(
            payload=evidence_payload,
            policy=policy,
        )
        if expected.result != result:
            raise ProductionReleaseIntegrityError("stored production release result differs from replay")
        return result

    def _put_model(
        self,
        codec: ProductionReleaseReportCodec,
        ref: ObjectRef,
        payload: bytes,
    ) -> None:
        self._put_bytes(codec, ref, payload)

    def _put_contract[ValueT: ContractModelV2](
        self,
        codec: ProductionReleaseReportCodec,
        *,
        value: ValueT,
        ref: ObjectRef,
        validate: Callable[[ValueT], None],
    ) -> ObjectRef:
        validate(value)
        self._put_model(codec, ref, value.canonical_json())
        return ref

    def _get_contract[ValueT: ContractModelV2](
        self,
        codec: ProductionReleaseReportCodec,
        *,
        ref: ObjectRef,
        model_type: type[ValueT],
        validate: Callable[[ValueT], None],
        resolve_ref: Callable[[ValueT], ObjectRef],
    ) -> ValueT:
        payload = self._get_bytes(codec, ref)
        try:
            value = model_type.model_validate_json(payload)
            validate(value)
        except (ValidationError, ValueError) as exc:
            raise ProductionReleaseIntegrityError("stored production release value is invalid") from exc
        if resolve_ref(value) != ref or value.canonical_json() != payload:
            raise ProductionReleaseIntegrityError("stored production release value differs from ref")
        return value

    def _put_bytes(
        self,
        codec: ProductionReleaseReportCodec,
        ref: ObjectRef,
        payload: bytes,
    ) -> None:
        if len(payload) < 2 or len(payload) > self.max_report_bytes:
            raise ProductionReleaseIntegrityError("production release value exceeds byte limits")
        existing = self._read_envelope_optional(codec, ref)
        if existing is not None:
            if self._read_payload(existing) != payload:
                raise ProductionReleaseConflictError("production release envelope content changed")
            return
        digest = hashlib.sha256(payload).hexdigest()
        content_ref = ObjectRef(
            object_type="production-release-content",
            object_id=f"production-release-content://sha256/{digest}",
            object_version="private-v1",
            object_sha256=digest,
        )
        envelope = _ProductionReleaseEnvelopeV1(
            codec=codec,
            object_ref=ref,
            content_ref=content_ref,
            canonical_size_bytes=len(payload),
        )
        _require_safe_store_member(
            self.root,
            self._cas.blob_path(digest),
        )
        try:
            self._cas.write(
                object_id=content_ref.object_id,
                digest=digest,
                value=payload,
            )
        except ContentAddressedByteStoreConflictError as exc:
            raise ProductionReleaseConflictError("production release CAS write conflicted") from exc
        path = self._envelope_path(codec, ref)
        _write_exclusive(path, envelope.canonical_json())
        stored = self._read_envelope(codec, ref)
        if stored != envelope:
            raise ProductionReleaseConflictError("production release envelope first authority conflicted")

    def _get_bytes(
        self,
        codec: ProductionReleaseReportCodec,
        ref: ObjectRef,
    ) -> bytes:
        envelope = self._read_envelope(codec, ref)
        return self._read_payload(envelope)

    def _read_payload(
        self,
        envelope: _ProductionReleaseEnvelopeV1,
    ) -> bytes:
        _require_safe_store_member(
            self.root,
            self._cas.blob_path(envelope.content_ref.object_sha256),
        )
        try:
            payload = self._cas.read(
                object_id=envelope.content_ref.object_id,
                digest=envelope.content_ref.object_sha256,
            )
        except ContentAddressedByteStoreCorruptionError as exc:
            raise ProductionReleaseIntegrityError("production release content is missing or corrupt") from exc
        if len(payload) != envelope.canonical_size_bytes or len(payload) > self.max_report_bytes:
            raise ProductionReleaseIntegrityError("production release content size is invalid")
        return payload

    def _read_envelope_optional(
        self,
        codec: ProductionReleaseReportCodec,
        ref: ObjectRef,
    ) -> _ProductionReleaseEnvelopeV1 | None:
        path = self._envelope_path(codec, ref)
        if not path.exists():
            return None
        return self._read_envelope(codec, ref)

    def _read_envelope(
        self,
        codec: ProductionReleaseReportCodec,
        ref: ObjectRef,
    ) -> _ProductionReleaseEnvelopeV1:
        path = self._envelope_path(codec, ref)
        if path.is_symlink() or not path.is_file():
            raise ProductionReleaseIntegrityError("production release envelope is missing or unsafe")
        try:
            value = _ProductionReleaseEnvelopeV1.model_validate_json(path.read_bytes())
        except (OSError, ValidationError, ValueError) as exc:
            raise ProductionReleaseIntegrityError("production release envelope is invalid") from exc
        if value.codec is not codec or value.object_ref != ref:
            raise ProductionReleaseIntegrityError(
                "production release envelope differs from requested authority"
            )
        return value

    def _envelope_path(
        self,
        codec: ProductionReleaseReportCodec,
        ref: ObjectRef,
    ) -> Path:
        path = self._envelopes / codec.value / "sha256" / ref.object_sha256[:2] / ref.object_sha256
        _require_safe_store_member(self.root, path)
        return path

    def _write_acceptance(
        self,
        value: ProductionReleaseAcceptedRunV1,
    ) -> ProductionReleaseAcceptedRunV1:
        path = self._acceptance_path(value.acceptance_key)
        written = _write_exclusive(path, value.canonical_json())
        observed = self._read_acceptance(value.acceptance_key)
        if not written and observed != value:
            raise ProductionReleaseConflictError("production release acceptance key already has authority")
        return observed

    def _read_acceptance_optional(
        self,
        acceptance_key: str,
    ) -> ProductionReleaseAcceptedRunV1 | None:
        path = self._acceptance_path(acceptance_key)
        if not path.exists():
            return None
        return self._read_acceptance(acceptance_key)

    def _read_acceptance(
        self,
        acceptance_key: str,
    ) -> ProductionReleaseAcceptedRunV1:
        path = self._acceptance_path(acceptance_key)
        if path.is_symlink() or not path.is_file():
            raise ProductionReleaseIntegrityError("production release acceptance is missing or unsafe")
        try:
            value = ProductionReleaseAcceptedRunV1.model_validate_json(path.read_bytes())
        except (OSError, ValidationError, ValueError) as exc:
            raise ProductionReleaseIntegrityError("production release acceptance is invalid") from exc
        if value.acceptance_key != acceptance_key:
            raise ProductionReleaseIntegrityError("production release acceptance key differs from index")
        return value

    def _acceptance_path(self, acceptance_key: str) -> Path:
        digest = hashlib.sha256(acceptance_key.encode()).hexdigest()
        path = self._acceptances / "sha256" / digest[:2] / digest
        _require_safe_store_member(self.root, path)
        return path


class ProductionRegistryStore:
    def __init__(
        self,
        root: Path,
        *,
        max_registry_bytes: int,
    ) -> None:
        if "production" not in {part.casefold() for part in root.expanduser().parts}:
            raise ProductionReleaseIntegrityError(
                "production registry root requires an explicit production namespace"
            )
        self._store = ProductionReleaseReportStore(
            root,
            max_report_bytes=max_registry_bytes,
        )
        self.root = self._store.root

    def put_closure(
        self,
        *,
        policy: ProductionReleasePolicyV2,
        result: ProductionReleaseResultV2,
    ) -> ObjectRef:
        if result.outcome is not ProductionReleaseOutcomeV2.PUBLISHED or result.policy_ref != policy.to_ref():
            raise ProductionReleaseIntegrityError(
                "production registry accepts only complete published closure"
            )
        authority = result.attestation_authority
        manifest = result.release_manifest
        registry_entry = result.registry_entry
        if authority is None or manifest is None or registry_entry is None:
            raise ProductionReleaseIntegrityError(
                "production registry accepts only complete published closure"
            )
        self._store.put_policy(policy)
        self._store.put_attestation_authority(authority)
        for item_manifest in manifest.item_manifests:
            self._store.put_item_manifest(item_manifest)
        self._store.put_release_manifest(manifest)
        for receipt in result.export_receipts:
            self._store.put_receipt(receipt)
        self._store.put_registry_entry(registry_entry)
        for projection in result.item_projections:
            self._store.put_published_projection(projection)
        result_ref = self._store.put_result(result)
        if self.get_result(result_ref) != result:
            raise ProductionReleaseIntegrityError("production registry closure differs after write")
        return result_ref

    def get_result(self, ref: ObjectRef) -> ProductionReleaseResultV2:
        result = self._store.get_result(ref)
        if result.outcome is not ProductionReleaseOutcomeV2.PUBLISHED:
            raise ProductionReleaseIntegrityError("production registry result is not published")
        policy = self._store.get_policy(result.policy_ref)
        authority = result.attestation_authority
        manifest = result.release_manifest
        registry_entry = result.registry_entry
        if (
            authority is None
            or result.attestation_authority_ref is None
            or manifest is None
            or result.release_manifest_ref is None
            or registry_entry is None
            or result.registry_entry_ref is None
        ):
            raise ProductionReleaseIntegrityError("production registry closure is incomplete")
        stored_authority = self._store.get_attestation_authority(result.attestation_authority_ref)
        stored_manifest = self._store.get_release_manifest(result.release_manifest_ref)
        stored_item_manifests = tuple(
            self._store.get_item_manifest(item_ref) for item_ref in stored_manifest.item_manifest_refs
        )
        stored_receipts = tuple(
            self._store.get_receipt(receipt.to_ref()) for receipt in result.export_receipts
        )
        stored_registry_entry = self._store.get_registry_entry(result.registry_entry_ref)
        stored_projections = tuple(
            self._store.get_published_projection(projection.to_ref())
            for projection in result.item_projections
        )
        if (
            result.policy_ref != policy.to_ref()
            or stored_authority != authority
            or stored_manifest != manifest
            or stored_item_manifests != manifest.item_manifests
            or stored_receipts != result.export_receipts
            or stored_registry_entry != registry_entry
            or stored_projections != result.item_projections
        ):
            raise ProductionReleaseIntegrityError("production registry stored values differ from result")
        return result


def _accepted_sha256(
    *,
    acceptance_key: str,
    request_sha256: str,
    policy_ref: ObjectRef,
    repository_evidence_ref: ObjectRef,
    result_ref: ObjectRef,
) -> str:
    payload = canonical_value_v2(
        {
            "acceptance_key": acceptance_key,
            "request_sha256": request_sha256,
            "policy_ref": policy_ref,
            "repository_evidence_ref": repository_evidence_ref,
            "result_ref": result_ref,
        }
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


def _prepare_store_root(root: Path) -> Path:
    expanded = root.expanduser()
    absolute = expanded.absolute()
    if any(value.is_symlink() for value in (absolute, *absolute.parents)) or (
        expanded.exists() and not expanded.is_dir()
    ):
        raise ProductionReleaseIntegrityError("production release report root must be a real directory")
    expanded.mkdir(mode=0o700, parents=True, exist_ok=True)
    return expanded.resolve(strict=True)


def _require_safe_store_member(root: Path, path: Path) -> None:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ProductionReleaseIntegrityError("production release store member escapes root") from exc
    if not relative.parts:
        raise ProductionReleaseIntegrityError("production release store member is not a file")
    current = root
    for index, part in enumerate(relative.parts):
        current = current / part
        if current.is_symlink():
            raise ProductionReleaseIntegrityError("production release store member uses symlink")
        if current.exists():
            is_leaf = index == len(relative.parts) - 1
            if (is_leaf and not current.is_file()) or (not is_leaf and not current.is_dir()):
                raise ProductionReleaseIntegrityError("production release store member has invalid file type")


def _write_exclusive(path: Path, payload: bytes) -> bool:
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise ProductionReleaseIntegrityError("production release authority path is unsafe")
        return False
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            return False
    finally:
        temporary.unlink(missing_ok=True)
    return True


__all__ = [
    "ProductionRegistryStore",
    "ProductionReleaseAcceptedRunV1",
    "ProductionReleaseReportCodec",
    "ProductionReleaseReportStore",
]
