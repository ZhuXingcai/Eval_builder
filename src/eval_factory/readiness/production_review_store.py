from __future__ import annotations

import hashlib
import os
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal, Protocol, Self
from uuid import uuid4

from pydantic import Field, ValidationError, model_validator

from eval_factory.contracts.core import ContractAudit, Identifier, ObjectRef
from eval_factory.contracts.core_v2 import ContractModelV2
from eval_factory.contracts.production_readiness_review_v2 import (
    OperationsSLOAssessmentV2,
    OperationsSLOPolicyV2,
    ProductionReadinessReviewEvidenceClassV2,
    ProductionReadinessReviewPolicyV2,
    ProductionReadinessReviewReportV2,
    operations_slo_assessment_v2_ref,
    operations_slo_policy_v2_ref,
    production_readiness_review_policy_v2_ref,
    production_readiness_review_report_v2_ref,
    validate_operations_slo_assessment_v2_identity,
    validate_operations_slo_policy_v2_identity,
    validate_production_readiness_review_policy_v2_identity,
    validate_production_readiness_review_report_v2_identity,
)
from eval_factory.readiness.production_review import (
    ProductionReadinessReviewer,
)
from eval_factory.readiness.production_review_builder import (
    ProductionReadinessReviewBuilder,
    ProductionReadinessReviewCompilation,
)
from eval_factory.readiness.production_review_models import (
    ProductionApprovalBundleV1,
    ProductionApprovalProofV1,
    ProductionReadinessAcceptedRunV1,
    ProductionReadinessReviewRequestV1,
    TrustedProductionApprovalRegistryV1,
    production_approval_bundle_v1_ref,
    production_approval_proof_v1_ref,
    production_readiness_accepted_run_v1_ref,
    production_readiness_review_request_v1_ref,
    trusted_production_approval_registry_v1_ref,
)
from eval_factory.storage import (
    ContentAddressedByteStore,
    ContentAddressedByteStoreConflictError,
    ContentAddressedByteStoreCorruptionError,
)


class ProductionReadinessStoreError(RuntimeError):
    pass


class ProductionReadinessStoreTypeError(ProductionReadinessStoreError):
    pass


class ProductionReadinessStoreConflictError(ProductionReadinessStoreError):
    pass


class ProductionReadinessStoreIntegrityError(ProductionReadinessStoreError):
    pass


class ProductionReadinessStoreLimitError(ProductionReadinessStoreError):
    pass


class ProductionReadinessStoreInjectedCrash(ProductionReadinessStoreError):
    pass


class ProductionReadinessStoreCodec(StrEnum):
    REGISTRY = "registry"
    APPROVAL_PROOF = "approval-proof"
    APPROVAL_BUNDLE = "approval-bundle"
    REVIEW_REQUEST = "review-request"
    ACCEPTED_RUN = "accepted-run"
    REVIEW_POLICY = "review-policy"
    SLO_POLICY = "slo-policy"
    SLO_ASSESSMENT = "slo-assessment"
    REPORT = "report"


class ProductionReadinessStoreFaultPoint(StrEnum):
    AFTER_CAS_WRITE = "after_cas_write"
    AFTER_ENVELOPE_WRITE = "after_envelope_write"
    BEFORE_ACCEPTANCE_INDEX = "before_acceptance_index"


class ProductionReadinessStoreFaultInjector(Protocol):
    def maybe_raise(
        self,
        point: ProductionReadinessStoreFaultPoint,
        *,
        object_ref: ObjectRef,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class StaticProductionReadinessStoreFaultInjector:
    crash_points: frozenset[ProductionReadinessStoreFaultPoint] = frozenset()

    def maybe_raise(
        self,
        point: ProductionReadinessStoreFaultPoint,
        *,
        object_ref: ObjectRef,
    ) -> None:
        del object_ref
        if point in self.crash_points:
            raise ProductionReadinessStoreInjectedCrash(
                f"injected production readiness store crash at {point.value}"
            )


@dataclass(frozen=True, slots=True)
class ProductionReadinessStoreWrite:
    object_ref: ObjectRef
    written: bool
    content_blob_written: bool
    canonical_size_bytes: int


class _ProductionReadinessEnvelopeV1(ContractModelV2):
    schema_version: Literal["eval-factory/production-readiness-envelope/private-v1"] = (
        "eval-factory/production-readiness-envelope/private-v1"
    )
    object_ref: ObjectRef
    codec: ProductionReadinessStoreCodec
    content_blob_ref: ObjectRef
    canonical_size_bytes: int = Field(ge=2)

    @model_validator(mode="after")
    def validate_envelope(self) -> Self:
        expected_id = (
            f"production-readiness-content://{self.codec.value}/sha256/{self.content_blob_ref.object_sha256}"
        )
        if (
            self.content_blob_ref.object_type != "production-readiness-content"
            or self.content_blob_ref.object_version != "private-v1"
            or self.content_blob_ref.object_id != expected_id
        ):
            raise ValueError("production readiness content ref is invalid")
        return self


class _ProductionReadinessAcceptanceIndexV1(ContractModelV2):
    schema_version: Literal["eval-factory/production-readiness-acceptance-index/private-v1"] = (
        "eval-factory/production-readiness-acceptance-index/private-v1"
    )
    acceptance_key: Identifier
    accepted_run_ref: ObjectRef

    @model_validator(mode="after")
    def validate_index(self) -> Self:
        if (
            self.accepted_run_ref.object_type != "production-readiness-accepted-run"
            or self.accepted_run_ref.object_version != "private-v1"
        ):
            raise ValueError("acceptance index must reference accepted run")
        return self


class _EnvelopeByteStore:
    def __init__(
        self,
        root: Path,
        *,
        max_bytes: int,
        fault_injector: ProductionReadinessStoreFaultInjector | None,
    ) -> None:
        if max_bytes < 2:
            raise ProductionReadinessStoreLimitError("production readiness byte limit is invalid")
        candidate = root.expanduser()
        if candidate.is_symlink() or (candidate.exists() and not candidate.is_dir()):
            raise ProductionReadinessStoreTypeError(
                "production readiness root must be a non-symlink directory"
            )
        self.root = candidate.resolve()
        self.max_bytes = max_bytes
        self.fault_injector = fault_injector
        self._cas = ContentAddressedByteStore(self.root / "cas")
        self._envelopes = self.root / "envelopes"

    def put(
        self,
        *,
        ref: ObjectRef,
        codec: ProductionReadinessStoreCodec,
        payload: bytes,
    ) -> ProductionReadinessStoreWrite:
        if len(payload) > self.max_bytes:
            raise ProductionReadinessStoreLimitError("production readiness value exceeds byte limit")
        existing = self._existing(ref, codec)
        if existing is not None:
            return existing
        digest = hashlib.sha256(payload).hexdigest()
        content_ref = ObjectRef(
            object_type="production-readiness-content",
            object_id=(f"production-readiness-content://{codec.value}/sha256/{digest}"),
            object_version="private-v1",
            object_sha256=digest,
        )
        envelope = _ProductionReadinessEnvelopeV1(
            object_ref=ref,
            codec=codec,
            content_blob_ref=content_ref,
            canonical_size_bytes=len(payload),
        )
        _require_safe_store_member(
            self._cas.root,
            self._cas.blob_path(digest),
        )
        try:
            content_written = self._cas.write(
                object_id=content_ref.object_id,
                digest=digest,
                value=payload,
            )
        except ContentAddressedByteStoreConflictError as exc:
            raise ProductionReadinessStoreConflictError("production readiness CAS write conflicted") from exc
        self._fault(ProductionReadinessStoreFaultPoint.AFTER_CAS_WRITE, ref)
        written = self._write_envelope(envelope)
        self._fault(
            ProductionReadinessStoreFaultPoint.AFTER_ENVELOPE_WRITE,
            ref,
        )
        if not written:
            existing = self._existing(ref, codec)
            if existing is None:
                raise ProductionReadinessStoreIntegrityError("production readiness envelope disappeared")
            return existing
        self.get(ref=ref, codec=codec)
        return ProductionReadinessStoreWrite(
            object_ref=ref,
            written=True,
            content_blob_written=content_written,
            canonical_size_bytes=len(payload),
        )

    def get(
        self,
        *,
        ref: ObjectRef,
        codec: ProductionReadinessStoreCodec,
    ) -> bytes:
        envelope = self._read_envelope(ref, codec)
        _require_safe_store_member(
            self._cas.root,
            self._cas.blob_path(envelope.content_blob_ref.object_sha256),
        )
        try:
            payload = self._cas.read(
                object_id=envelope.content_blob_ref.object_id,
                digest=envelope.content_blob_ref.object_sha256,
            )
        except ContentAddressedByteStoreCorruptionError as exc:
            raise ProductionReadinessStoreIntegrityError(
                "production readiness content is missing or corrupt"
            ) from exc
        if len(payload) > self.max_bytes or len(payload) != envelope.canonical_size_bytes:
            raise ProductionReadinessStoreLimitError("stored production readiness byte size is invalid")
        return payload

    def _existing(
        self,
        ref: ObjectRef,
        codec: ProductionReadinessStoreCodec,
    ) -> ProductionReadinessStoreWrite | None:
        path = self._envelope_path(ref, codec)
        if not path.exists():
            return None
        envelope = self._read_envelope(ref, codec)
        self.get(ref=ref, codec=codec)
        return ProductionReadinessStoreWrite(
            object_ref=ref,
            written=False,
            content_blob_written=False,
            canonical_size_bytes=envelope.canonical_size_bytes,
        )

    def _write_envelope(
        self,
        envelope: _ProductionReadinessEnvelopeV1,
    ) -> bool:
        path = self._envelope_path(envelope.object_ref, envelope.codec)
        _require_safe_store_member(self._envelopes, path)
        if path.exists():
            self._read_envelope(envelope.object_ref, envelope.codec)
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        _require_safe_store_member(self._envelopes, path)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as handle:
                handle.write(envelope.canonical_json() + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                self._read_envelope(envelope.object_ref, envelope.codec)
                return False
        finally:
            temporary.unlink(missing_ok=True)
        return True

    def _read_envelope(
        self,
        ref: ObjectRef,
        codec: ProductionReadinessStoreCodec,
    ) -> _ProductionReadinessEnvelopeV1:
        path = self._envelope_path(ref, codec)
        _require_safe_store_member(self._envelopes, path)
        if not path.exists():
            raise ProductionReadinessStoreIntegrityError("production readiness envelope is missing")
        try:
            encoded = path.read_bytes()
            envelope = _ProductionReadinessEnvelopeV1.model_validate_json(encoded)
        except (OSError, ValidationError, ValueError) as exc:
            raise ProductionReadinessStoreIntegrityError("production readiness envelope is invalid") from exc
        if encoded != envelope.canonical_json() + b"\n":
            raise ProductionReadinessStoreIntegrityError("production readiness envelope is not canonical")
        if envelope.object_ref != ref or envelope.codec is not codec:
            raise ProductionReadinessStoreIntegrityError("production readiness envelope authority differs")
        return envelope

    def _envelope_path(
        self,
        ref: ObjectRef,
        codec: ProductionReadinessStoreCodec,
    ) -> Path:
        return self._envelopes / codec.value / "sha256" / ref.object_sha256[:2] / f"{ref.object_sha256}.json"

    def _fault(
        self,
        point: ProductionReadinessStoreFaultPoint,
        ref: ObjectRef,
    ) -> None:
        if self.fault_injector is not None:
            self.fault_injector.maybe_raise(point, object_ref=ref)


_PrivateMaterial = (
    TrustedProductionApprovalRegistryV1
    | ProductionApprovalProofV1
    | ProductionApprovalBundleV1
    | ProductionReadinessReviewRequestV1
    | ProductionReadinessAcceptedRunV1
)


class ProductionReadinessMaterialStore:
    def __init__(
        self,
        root: Path,
        *,
        max_private_bytes: int,
        max_members: int,
        fault_injector: ProductionReadinessStoreFaultInjector | None = None,
    ) -> None:
        if max_members < 4 or max_members > 1_000_000:
            raise ProductionReadinessStoreLimitError("production readiness member limit is invalid")
        self.root = root.expanduser()
        self.max_members = max_members
        self._store = _EnvelopeByteStore(
            self.root,
            max_bytes=max_private_bytes,
            fault_injector=fault_injector,
        )
        self.root = self._store.root
        self._authorities = self.root / "authorities"

    def put_registry(
        self,
        value: TrustedProductionApprovalRegistryV1,
    ) -> ProductionReadinessStoreWrite:
        return self._put(
            value,
            value.to_ref(),
            ProductionReadinessStoreCodec.REGISTRY,
            len(value.authorities),
        )

    def get_registry(
        self,
        ref: ObjectRef,
    ) -> TrustedProductionApprovalRegistryV1:
        return self._get(
            ref,
            ProductionReadinessStoreCodec.REGISTRY,
            TrustedProductionApprovalRegistryV1,
        )

    def put_proof(
        self,
        value: ProductionApprovalProofV1,
    ) -> ProductionReadinessStoreWrite:
        return self._put(
            value,
            value.to_ref(),
            ProductionReadinessStoreCodec.APPROVAL_PROOF,
            1,
        )

    def get_proof(
        self,
        ref: ObjectRef,
    ) -> ProductionApprovalProofV1:
        return self._get(
            ref,
            ProductionReadinessStoreCodec.APPROVAL_PROOF,
            ProductionApprovalProofV1,
        )

    def put_bundle(
        self,
        value: ProductionApprovalBundleV1,
    ) -> ProductionReadinessStoreWrite:
        return self._put(
            value,
            value.to_ref(),
            ProductionReadinessStoreCodec.APPROVAL_BUNDLE,
            len(value.proofs),
        )

    def get_bundle(
        self,
        ref: ObjectRef,
    ) -> ProductionApprovalBundleV1:
        return self._get(
            ref,
            ProductionReadinessStoreCodec.APPROVAL_BUNDLE,
            ProductionApprovalBundleV1,
        )

    def put_request(
        self,
        value: ProductionReadinessReviewRequestV1,
    ) -> ProductionReadinessStoreWrite:
        return self._put(
            value,
            value.to_ref(),
            ProductionReadinessStoreCodec.REVIEW_REQUEST,
            _member_count(value),
        )

    def get_request(
        self,
        ref: ObjectRef,
    ) -> ProductionReadinessReviewRequestV1:
        return self._get(
            ref,
            ProductionReadinessStoreCodec.REVIEW_REQUEST,
            ProductionReadinessReviewRequestV1,
        )

    def put_accepted_run(
        self,
        value: ProductionReadinessAcceptedRunV1,
    ) -> ProductionReadinessStoreWrite:
        return self._put(
            value,
            value.to_ref(),
            ProductionReadinessStoreCodec.ACCEPTED_RUN,
            1,
        )

    def get_accepted_run(
        self,
        ref: ObjectRef,
    ) -> ProductionReadinessAcceptedRunV1:
        return self._get(
            ref,
            ProductionReadinessStoreCodec.ACCEPTED_RUN,
            ProductionReadinessAcceptedRunV1,
        )

    def find_acceptance(
        self,
        acceptance_key: str,
    ) -> ProductionReadinessAcceptedRunV1 | None:
        path = self._acceptance_path(acceptance_key)
        _require_safe_store_member(self._authorities, path)
        if not path.exists():
            return None
        return self._read_acceptance(path, acceptance_key)

    def get_acceptance(
        self,
        acceptance_key: str,
    ) -> ProductionReadinessAcceptedRunV1:
        value = self.find_acceptance(acceptance_key)
        if value is None:
            raise ProductionReadinessStoreIntegrityError("production readiness acceptance was not found")
        return value

    def publish_acceptance(
        self,
        value: ProductionReadinessAcceptedRunV1,
    ) -> ProductionReadinessAcceptedRunV1:
        path = self._acceptance_path(value.acceptance_key)
        _require_safe_store_member(self._authorities, path)
        existing = self.find_acceptance(value.acceptance_key)
        if existing is not None:
            return existing
        path.parent.mkdir(parents=True, exist_ok=True)
        _require_safe_store_member(self._authorities, path)
        index = _ProductionReadinessAcceptanceIndexV1(
            acceptance_key=value.acceptance_key,
            accepted_run_ref=value.to_ref(),
        )
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as handle:
                handle.write(index.canonical_json() + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            with suppress(FileExistsError):
                os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return self._read_acceptance(path, value.acceptance_key)

    def _read_acceptance(
        self,
        path: Path,
        acceptance_key: str,
    ) -> ProductionReadinessAcceptedRunV1:
        try:
            encoded = path.read_bytes()
            index = _ProductionReadinessAcceptanceIndexV1.model_validate_json(encoded)
        except (OSError, ValidationError, ValueError) as exc:
            raise ProductionReadinessStoreIntegrityError(
                "production readiness acceptance index is invalid"
            ) from exc
        if encoded != index.canonical_json() + b"\n" or index.acceptance_key != acceptance_key:
            raise ProductionReadinessStoreIntegrityError("production readiness acceptance index differs")
        accepted = self.get_accepted_run(index.accepted_run_ref)
        if accepted.acceptance_key != acceptance_key:
            raise ProductionReadinessStoreIntegrityError("accepted run differs from acceptance index")
        return accepted

    def _acceptance_path(self, acceptance_key: str) -> Path:
        digest = hashlib.sha256(acceptance_key.encode()).hexdigest()
        return self._authorities / "sha256" / digest[:2] / f"{digest}.json"

    def _put(
        self,
        value: _PrivateMaterial,
        ref: ObjectRef,
        codec: ProductionReadinessStoreCodec,
        member_count: int,
    ) -> ProductionReadinessStoreWrite:
        if member_count > self.max_members:
            raise ProductionReadinessStoreLimitError("production readiness material exceeds member limit")
        return self._store.put(
            ref=ref,
            codec=codec,
            payload=value.canonical_json(),
        )

    def _get[ModelT: ContractModelV2](
        self,
        ref: ObjectRef,
        codec: ProductionReadinessStoreCodec,
        model_type: type[ModelT],
    ) -> ModelT:
        _require_private_ref(ref, codec)
        payload = self._store.get(ref=ref, codec=codec)
        try:
            value = model_type.model_validate_json(payload)
            observed = _private_ref(value)
        except (ValidationError, ValueError) as exc:
            raise ProductionReadinessStoreIntegrityError(
                "stored production readiness material is invalid"
            ) from exc
        if observed != ref or value.canonical_json() != payload:
            raise ProductionReadinessStoreIntegrityError(
                "stored production readiness material differs from ref"
            )
        if _member_count(value) > self.max_members:
            raise ProductionReadinessStoreLimitError(
                "stored production readiness material exceeds member limit"
            )
        return value


_PublicMaterial = (
    ProductionReadinessReviewPolicyV2
    | OperationsSLOPolicyV2
    | OperationsSLOAssessmentV2
    | ProductionReadinessReviewReportV2
)


class ProductionReadinessReportStore:
    def __init__(
        self,
        root: Path,
        *,
        max_report_bytes: int,
        fault_injector: ProductionReadinessStoreFaultInjector | None = None,
    ) -> None:
        self._store = _EnvelopeByteStore(
            root,
            max_bytes=max_report_bytes,
            fault_injector=fault_injector,
        )
        self.root = self._store.root

    def put_policy(
        self,
        value: ProductionReadinessReviewPolicyV2,
    ) -> ProductionReadinessStoreWrite:
        return self._put(
            value,
            value.to_ref(),
            ProductionReadinessStoreCodec.REVIEW_POLICY,
        )

    def get_policy(
        self,
        ref: ObjectRef,
    ) -> ProductionReadinessReviewPolicyV2:
        return self._get(
            ref,
            ProductionReadinessStoreCodec.REVIEW_POLICY,
            ProductionReadinessReviewPolicyV2,
        )

    def put_slo_policy(
        self,
        value: OperationsSLOPolicyV2,
    ) -> ProductionReadinessStoreWrite:
        return self._put(
            value,
            value.to_ref(),
            ProductionReadinessStoreCodec.SLO_POLICY,
        )

    def get_slo_policy(
        self,
        ref: ObjectRef,
    ) -> OperationsSLOPolicyV2:
        return self._get(
            ref,
            ProductionReadinessStoreCodec.SLO_POLICY,
            OperationsSLOPolicyV2,
        )

    def put_slo_assessment(
        self,
        value: OperationsSLOAssessmentV2,
    ) -> ProductionReadinessStoreWrite:
        return self._put(
            value,
            value.to_ref(),
            ProductionReadinessStoreCodec.SLO_ASSESSMENT,
        )

    def get_slo_assessment(
        self,
        ref: ObjectRef,
    ) -> OperationsSLOAssessmentV2:
        return self._get(
            ref,
            ProductionReadinessStoreCodec.SLO_ASSESSMENT,
            OperationsSLOAssessmentV2,
        )

    def put(
        self,
        value: ProductionReadinessReviewReportV2,
    ) -> ProductionReadinessStoreWrite:
        return self._put(
            value,
            value.to_ref(),
            ProductionReadinessStoreCodec.REPORT,
        )

    def get(
        self,
        ref: ObjectRef,
    ) -> ProductionReadinessReviewReportV2:
        return self._get(
            ref,
            ProductionReadinessStoreCodec.REPORT,
            ProductionReadinessReviewReportV2,
        )

    def _put(
        self,
        value: _PublicMaterial,
        ref: ObjectRef,
        codec: ProductionReadinessStoreCodec,
    ) -> ProductionReadinessStoreWrite:
        _validate_public(value)
        return self._store.put(
            ref=ref,
            codec=codec,
            payload=value.canonical_json(),
        )

    def _get[ModelT: ContractModelV2](
        self,
        ref: ObjectRef,
        codec: ProductionReadinessStoreCodec,
        model_type: type[ModelT],
    ) -> ModelT:
        _require_public_ref(ref, codec)
        payload = self._store.get(ref=ref, codec=codec)
        try:
            value = model_type.model_validate_json(payload)
            _validate_public(value)
            observed = _public_ref(value)
        except (ValidationError, ValueError) as exc:
            raise ProductionReadinessStoreIntegrityError(
                "stored public production readiness value is invalid"
            ) from exc
        if observed != ref or value.canonical_json() != payload:
            raise ProductionReadinessStoreIntegrityError(
                "stored public production readiness value differs from ref"
            )
        return value


class ProductionReadinessAcceptedRunStore:
    def __init__(
        self,
        *,
        material_store: ProductionReadinessMaterialStore,
        report_store: ProductionReadinessReportStore,
        fault_injector: ProductionReadinessStoreFaultInjector | None = None,
    ) -> None:
        _require_disjoint_roots(material_store.root, report_store.root)
        self.material_store = material_store
        self.report_store = report_store
        self.fault_injector = fault_injector

    def persist(
        self,
        *,
        acceptance_key: str,
        request_sha256: str,
        compilation: ProductionReadinessReviewCompilation,
        report: ProductionReadinessReviewReportV2,
        audit: ContractAudit,
        previous_acceptance_key: str | None = None,
    ) -> ProductionReadinessAcceptedRunV1:
        existing = self.material_store.find_acceptance(acceptance_key)
        if existing is not None:
            if existing.request_sha256 != request_sha256:
                raise ProductionReadinessStoreConflictError("production readiness acceptance request changed")
            if existing.report_ref != report.to_ref():
                raise ProductionReadinessStoreConflictError("production readiness accepted report changed")
            stored_report = self._verify_closure(existing)
            self._verify_previous_authority(
                stored_report,
                previous_acceptance_key=previous_acceptance_key,
            )
            return existing
        expected_report = ProductionReadinessReviewer().evaluate(
            compilation=compilation,
            policy=compilation.policy,
            audit=report.audit,
        )
        if expected_report != report:
            raise ProductionReadinessStoreIntegrityError(
                "production readiness report differs from private evaluation"
            )
        self._verify_previous_authority(
            expected_report,
            previous_acceptance_key=previous_acceptance_key,
        )
        self.report_store.put_policy(compilation.policy)
        registry_ref: ObjectRef | None = None
        request_ref: ObjectRef | None = None
        bundle_ref: ObjectRef | None = None
        if compilation.request is not None:
            registry = compilation.trusted_registry
            if registry is None:
                raise ProductionReadinessStoreIntegrityError("full review registry is missing")
            request = compilation.request
            self.material_store.put_registry(registry)
            if request.approval_bundle is not None:
                for proof in request.approval_bundle.proofs:
                    self.material_store.put_proof(proof)
                self.material_store.put_bundle(request.approval_bundle)
                bundle_ref = request.approval_bundle.to_ref()
            self.material_store.put_request(request)
            if request.operations_slo_policy is not None:
                self.report_store.put_slo_policy(request.operations_slo_policy)
            if request.operations_slo_assessment is not None:
                self.report_store.put_slo_assessment(request.operations_slo_assessment)
            registry_ref = registry.to_ref()
            request_ref = request.to_ref()
        elif compilation.trusted_registry is not None or compilation.approval_bundle is not None:
            raise ProductionReadinessStoreIntegrityError(
                "pending compilation carries private review authority"
            )
        self.report_store.put(report)
        accepted = ProductionReadinessAcceptedRunV1.create(
            acceptance_key=acceptance_key,
            request_sha256=request_sha256,
            report_ref=report.to_ref(),
            registry_ref=registry_ref,
            request_ref=request_ref,
            approval_bundle_ref=bundle_ref,
            audit=audit,
        )
        self.material_store.put_accepted_run(accepted)
        self._fault(
            ProductionReadinessStoreFaultPoint.BEFORE_ACCEPTANCE_INDEX,
            accepted.to_ref(),
        )
        authoritative = self.material_store.publish_acceptance(accepted)
        if authoritative.request_sha256 != request_sha256 or authoritative.to_ref() != accepted.to_ref():
            raise ProductionReadinessStoreConflictError("production readiness first authority conflicted")
        self._verify_closure(authoritative)
        return authoritative

    def get_accepted_report(
        self,
        acceptance_key: str,
        *,
        previous_acceptance_key: str | None = None,
    ) -> ProductionReadinessReviewReportV2:
        report = self._verify_closure(self.material_store.get_acceptance(acceptance_key))
        self._verify_previous_authority(
            report,
            previous_acceptance_key=previous_acceptance_key,
        )
        return report

    def _verify_previous_authority(
        self,
        report: ProductionReadinessReviewReportV2,
        *,
        previous_acceptance_key: str | None,
    ) -> None:
        if report.review_version == 1:
            if previous_acceptance_key is not None:
                raise ProductionReadinessStoreConflictError("review version 1 cannot bind previous authority")
            return
        if previous_acceptance_key is None:
            raise ProductionReadinessStoreIntegrityError(
                "later review version requires accepted previous authority"
            )
        previous = self._verify_closure(self.material_store.get_acceptance(previous_acceptance_key))
        if (
            previous.review_series_id != report.review_series_id
            or previous.review_version != report.review_version - 1
            or report.previous_report_ref != previous.to_ref()
        ):
            raise ProductionReadinessStoreConflictError(
                "previous review authority differs from accepted series"
            )

    def _verify_closure(
        self,
        accepted: ProductionReadinessAcceptedRunV1,
    ) -> ProductionReadinessReviewReportV2:
        report = self.report_store.get(accepted.report_ref)
        policy = self.report_store.get_policy(report.policy_ref)
        if accepted.request_ref is None:
            if (
                accepted.registry_ref is not None
                or accepted.approval_bundle_ref is not None
                or report.evidence_class
                is not ProductionReadinessReviewEvidenceClassV2.REPOSITORY_PENDING_ONLY
                or report.private_request_closure_sha256 is not None
                or report.private_approval_closure_sha256 is not None
            ):
                raise ProductionReadinessStoreIntegrityError("pending acceptance carries private authority")
            return report
        if accepted.registry_ref is None:
            raise ProductionReadinessStoreIntegrityError("full acceptance registry is missing")
        registry = self.material_store.get_registry(accepted.registry_ref)
        request = self.material_store.get_request(accepted.request_ref)
        bundle = request.approval_bundle
        if bundle is None:
            if accepted.approval_bundle_ref is not None:
                raise ProductionReadinessStoreIntegrityError("accepted approval bundle differs from request")
        else:
            if accepted.approval_bundle_ref != bundle.to_ref():
                raise ProductionReadinessStoreIntegrityError("accepted approval bundle differs from request")
            stored_bundle = self.material_store.get_bundle(bundle.to_ref())
            if stored_bundle != bundle:
                raise ProductionReadinessStoreIntegrityError("stored approval bundle differs from request")
            for proof in bundle.proofs:
                if self.material_store.get_proof(proof.to_ref()) != proof:
                    raise ProductionReadinessStoreIntegrityError("stored approval proof differs from bundle")
        if (
            request.operations_slo_policy is not None
            and self.report_store.get_slo_policy(request.operations_slo_policy.to_ref())
            != request.operations_slo_policy
        ):
            raise ProductionReadinessStoreIntegrityError("stored SLO policy differs from request")
        if (
            request.operations_slo_assessment is not None
            and self.report_store.get_slo_assessment(request.operations_slo_assessment.to_ref())
            != request.operations_slo_assessment
        ):
            raise ProductionReadinessStoreIntegrityError("stored SLO assessment differs from request")
        compilation = ProductionReadinessReviewBuilder().compile_full(
            policy=policy,
            request=request,
            trusted_registry=registry,
        )
        expected = ProductionReadinessReviewer().evaluate(
            compilation=compilation,
            policy=policy,
            audit=report.audit,
        )
        if expected != report:
            raise ProductionReadinessStoreIntegrityError(
                "accepted report differs from stored private closure"
            )
        return report

    def _fault(
        self,
        point: ProductionReadinessStoreFaultPoint,
        ref: ObjectRef,
    ) -> None:
        if self.fault_injector is not None:
            self.fault_injector.maybe_raise(point, object_ref=ref)


def _private_ref(value: ContractModelV2) -> ObjectRef:
    if isinstance(value, TrustedProductionApprovalRegistryV1):
        return trusted_production_approval_registry_v1_ref(value)
    if isinstance(value, ProductionApprovalProofV1):
        return production_approval_proof_v1_ref(value)
    if isinstance(value, ProductionApprovalBundleV1):
        return production_approval_bundle_v1_ref(value)
    if isinstance(value, ProductionReadinessReviewRequestV1):
        return production_readiness_review_request_v1_ref(value)
    if isinstance(value, ProductionReadinessAcceptedRunV1):
        return production_readiness_accepted_run_v1_ref(value)
    raise ProductionReadinessStoreTypeError("production readiness private type is unsupported")


def _public_ref(value: ContractModelV2) -> ObjectRef:
    if isinstance(value, ProductionReadinessReviewPolicyV2):
        return production_readiness_review_policy_v2_ref(value)
    if isinstance(value, OperationsSLOPolicyV2):
        return operations_slo_policy_v2_ref(value)
    if isinstance(value, OperationsSLOAssessmentV2):
        return operations_slo_assessment_v2_ref(value)
    if isinstance(value, ProductionReadinessReviewReportV2):
        return production_readiness_review_report_v2_ref(value)
    raise ProductionReadinessStoreTypeError("production readiness public type is unsupported")


def _validate_public(value: ContractModelV2) -> None:
    if isinstance(value, ProductionReadinessReviewPolicyV2):
        validate_production_readiness_review_policy_v2_identity(value)
    elif isinstance(value, OperationsSLOPolicyV2):
        validate_operations_slo_policy_v2_identity(value)
    elif isinstance(value, OperationsSLOAssessmentV2):
        validate_operations_slo_assessment_v2_identity(value)
    elif isinstance(value, ProductionReadinessReviewReportV2):
        validate_production_readiness_review_report_v2_identity(value)
    else:
        raise ProductionReadinessStoreTypeError("production readiness public type is unsupported")


def _member_count(value: ContractModelV2) -> int:
    if isinstance(value, TrustedProductionApprovalRegistryV1):
        return len(value.authorities)
    if isinstance(value, ProductionApprovalBundleV1):
        return len(value.proofs)
    if isinstance(value, ProductionReadinessReviewRequestV1):
        return 4 + (len(value.approval_bundle.proofs) if value.approval_bundle is not None else 0)
    return 1


def _require_private_ref(
    ref: ObjectRef,
    codec: ProductionReadinessStoreCodec,
) -> None:
    expected = {
        ProductionReadinessStoreCodec.REGISTRY: ("production-approval-authority-registry"),
        ProductionReadinessStoreCodec.APPROVAL_PROOF: ("production-approval-proof"),
        ProductionReadinessStoreCodec.APPROVAL_BUNDLE: ("production-approval-bundle"),
        ProductionReadinessStoreCodec.REVIEW_REQUEST: ("production-readiness-review-request"),
        ProductionReadinessStoreCodec.ACCEPTED_RUN: ("production-readiness-accepted-run"),
    }.get(codec)
    if expected is None or ref.object_type != expected or ref.object_version != "private-v1":
        raise ProductionReadinessStoreTypeError("production readiness private ref has wrong type")


def _require_public_ref(
    ref: ObjectRef,
    codec: ProductionReadinessStoreCodec,
) -> None:
    expected = {
        ProductionReadinessStoreCodec.REVIEW_POLICY: ("production-readiness-review-policy"),
        ProductionReadinessStoreCodec.SLO_POLICY: "operations-slo-policy",
        ProductionReadinessStoreCodec.SLO_ASSESSMENT: ("operations-slo-assessment"),
        ProductionReadinessStoreCodec.REPORT: ("production-readiness-review-report"),
    }.get(codec)
    if expected is None or ref.object_type != expected or ref.object_version != "v2":
        raise ProductionReadinessStoreTypeError("production readiness public ref has wrong type")


def _require_safe_store_member(root: Path, path: Path) -> None:
    if root.is_symlink() or (root.exists() and not root.is_dir()):
        raise ProductionReadinessStoreTypeError("production readiness store root has invalid file type")
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ProductionReadinessStoreTypeError("production readiness store member escapes root") from exc
    if not relative.parts:
        raise ProductionReadinessStoreTypeError("production readiness store member is not a file")
    current = root
    for index, part in enumerate(relative.parts):
        current = current / part
        if current.is_symlink():
            raise ProductionReadinessStoreTypeError("production readiness store member uses symlink")
        if current.exists():
            is_leaf = index == len(relative.parts) - 1
            if (is_leaf and not current.is_file()) or (not is_leaf and not current.is_dir()):
                raise ProductionReadinessStoreTypeError(
                    "production readiness store member has invalid file type"
                )


def _require_disjoint_roots(first: Path, second: Path) -> None:
    first_resolved = first.resolve()
    second_resolved = second.resolve()
    if (
        first_resolved == second_resolved
        or first_resolved in second_resolved.parents
        or second_resolved in first_resolved.parents
    ):
        raise ProductionReadinessStoreTypeError("production readiness store roots overlap")


__all__ = [
    "ProductionReadinessAcceptedRunStore",
    "ProductionReadinessMaterialStore",
    "ProductionReadinessReportStore",
    "ProductionReadinessStoreCodec",
    "ProductionReadinessStoreConflictError",
    "ProductionReadinessStoreError",
    "ProductionReadinessStoreFaultPoint",
    "ProductionReadinessStoreInjectedCrash",
    "ProductionReadinessStoreIntegrityError",
    "ProductionReadinessStoreLimitError",
    "ProductionReadinessStoreTypeError",
    "ProductionReadinessStoreWrite",
    "StaticProductionReadinessStoreFaultInjector",
]
