from __future__ import annotations

import fcntl
import hashlib
import os
from collections.abc import Generator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol, cast
from uuid import uuid4

from pydantic import ValidationError

from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.core_v2 import ContractModelV2
from eval_factory.contracts.production_attestation_v2 import (
    ProductionReadinessAttestationPolicyV2,
    ProductionReadinessAttestationPrerequisiteV2,
    ProductionReadinessAttestationProjectionV2,
    ProductionReadinessAttestationResultV2,
    ProductionReadinessAttestationStateV2,
    ProductionReadinessInvalidationPolicyV2,
    ProductionReadinessInvalidationRecordV2,
    ProductionReadinessVersionSetV2,
    validate_production_readiness_attestation_policy_v2_identity,
    validate_production_readiness_attestation_prerequisite_v2_identity,
    validate_production_readiness_attestation_projection_v2_identity,
    validate_production_readiness_attestation_result_v2_identity,
    validate_production_readiness_invalidation_policy_v2_identity,
    validate_production_readiness_invalidation_record_v2_identity,
    validate_production_readiness_version_set_v2_identity,
)
from eval_factory.contracts.release import ProductionReadinessAttestation
from eval_factory.readiness.production_attestation import (
    ProductionAttestationEvaluator,
    production_readiness_attestation_v1_ref,
    validate_production_readiness_attestation_v1_identity,
)
from eval_factory.readiness.production_attestation_builder import (
    ProductionAttestationBuilder,
    ProductionAttestationCompilation,
)
from eval_factory.readiness.production_attestation_models import (
    AttestationIssuanceBundleV1,
    AttestationIssuanceProofV1,
    ProductionReadinessAttestationAcceptedRunV1,
    ProductionReadinessAttestationRequestV1,
    ProductionReadinessSC014ClosureV1,
    TrustedAttestationIssuerRegistryV1,
)
from eval_factory.storage import (
    ContentAddressedByteStore,
    ContentAddressedByteStoreConflictError,
    ContentAddressedByteStoreCorruptionError,
)


class ProductionAttestationStoreError(RuntimeError):
    pass


class ProductionAttestationStoreTypeError(ProductionAttestationStoreError):
    pass


class ProductionAttestationStoreConflictError(ProductionAttestationStoreError):
    pass


class ProductionAttestationStoreIntegrityError(ProductionAttestationStoreError):
    pass


class ProductionAttestationStoreLimitError(ProductionAttestationStoreError):
    pass


class ProductionAttestationStoreInjectedCrash(ProductionAttestationStoreError):
    pass


class ProductionAttestationStoreCodec(StrEnum):
    ISSUER_REGISTRY = "issuer-registry"
    ISSUANCE_PROOF = "issuance-proof"
    ISSUANCE_BUNDLE = "issuance-bundle"
    SC014_CLOSURE = "sc014-closure"
    REQUEST = "request"
    FROZEN_ATTESTATION = "frozen-attestation"
    ACCEPTED_RUN = "accepted-run"
    POLICY = "policy"
    INVALIDATION_POLICY = "invalidation-policy"
    VERSION_SET = "version-set"
    PREREQUISITE = "prerequisite"
    PROJECTION = "projection"
    INVALIDATION_RECORD = "invalidation-record"
    RESULT = "result"


class ProductionAttestationStoreFaultPoint(StrEnum):
    AFTER_CAS_WRITE = "after_cas_write"
    AFTER_ENVELOPE_WRITE = "after_envelope_write"
    BEFORE_ACCEPTANCE_INDEX = "before_acceptance_index"


class ProductionAttestationStoreFaultInjector(Protocol):
    def maybe_raise(
        self,
        point: ProductionAttestationStoreFaultPoint,
        *,
        object_ref: ObjectRef,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class StaticProductionAttestationStoreFaultInjector:
    crash_points: frozenset[ProductionAttestationStoreFaultPoint] = frozenset()

    def maybe_raise(
        self,
        point: ProductionAttestationStoreFaultPoint,
        *,
        object_ref: ObjectRef,
    ) -> None:
        del object_ref
        if point in self.crash_points:
            raise ProductionAttestationStoreInjectedCrash(
                f"injected production attestation store crash at {point.value}"
            )


@dataclass(frozen=True, slots=True)
class ProductionAttestationStoreWrite:
    object_ref: ObjectRef
    written: bool
    canonical_size_bytes: int


class _Envelope(ContractModelV2):
    schema_version: str = "eval-factory/production-attestation-envelope/private-v1"
    object_ref: ObjectRef
    codec: ProductionAttestationStoreCodec
    content_ref: ObjectRef
    canonical_size_bytes: int


class _AcceptanceIndex(ContractModelV2):
    schema_version: str = "eval-factory/production-attestation-acceptance-index/private-v1"
    acceptance_key: str
    accepted_run_ref: ObjectRef


class _CurrentProjectionIndex(ContractModelV2):
    schema_version: str = "eval-factory/production-attestation-current-projection/private-v1"
    series_key: str
    projection_ref: ObjectRef


class _EnvelopeStore:
    def __init__(
        self,
        root: Path,
        *,
        max_bytes: int,
        fault_injector: ProductionAttestationStoreFaultInjector | None,
    ) -> None:
        if max_bytes < 2:
            raise ProductionAttestationStoreLimitError("production attestation byte limit is invalid")
        candidate = root.expanduser()
        if candidate.is_symlink() or (candidate.exists() and not candidate.is_dir()):
            raise ProductionAttestationStoreTypeError(
                "production attestation root must be a non-symlink directory"
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
        codec: ProductionAttestationStoreCodec,
        payload: bytes,
    ) -> ProductionAttestationStoreWrite:
        if len(payload) > self.max_bytes:
            raise ProductionAttestationStoreLimitError("production attestation value exceeds byte limit")
        path = self._path(ref, codec)
        if path.exists():
            self.get(ref=ref, codec=codec)
            return ProductionAttestationStoreWrite(ref, False, len(payload))
        digest = hashlib.sha256(payload).hexdigest()
        content_ref = ObjectRef(
            object_type="production-attestation-content",
            object_id=f"production-attestation-content://{codec.value}/sha256/{digest}",
            object_version="private-v1",
            object_sha256=digest,
        )
        envelope = _Envelope(
            object_ref=ref,
            codec=codec,
            content_ref=content_ref,
            canonical_size_bytes=len(payload),
        )
        _safe_member(self._cas.root, self._cas.blob_path(digest))
        try:
            self._cas.write(
                object_id=content_ref.object_id,
                digest=digest,
                value=payload,
            )
        except ContentAddressedByteStoreConflictError as exc:
            raise ProductionAttestationStoreConflictError(
                "production attestation CAS write conflicted"
            ) from exc
        self._fault(ProductionAttestationStoreFaultPoint.AFTER_CAS_WRITE, ref)
        self._write_exclusive(path, envelope.canonical_json() + b"\n")
        self._fault(
            ProductionAttestationStoreFaultPoint.AFTER_ENVELOPE_WRITE,
            ref,
        )
        self.get(ref=ref, codec=codec)
        return ProductionAttestationStoreWrite(ref, True, len(payload))

    def get(
        self,
        *,
        ref: ObjectRef,
        codec: ProductionAttestationStoreCodec,
    ) -> bytes:
        path = self._path(ref, codec)
        _safe_member(self._envelopes, path)
        if not path.exists():
            raise ProductionAttestationStoreIntegrityError("production attestation envelope is missing")
        try:
            encoded = path.read_bytes()
            envelope = _Envelope.model_validate_json(encoded)
        except (OSError, ValidationError, ValueError) as exc:
            raise ProductionAttestationStoreIntegrityError(
                "production attestation envelope is invalid"
            ) from exc
        if (
            encoded != envelope.canonical_json() + b"\n"
            or envelope.object_ref != ref
            or envelope.codec is not codec
        ):
            raise ProductionAttestationStoreIntegrityError(
                "production attestation envelope authority differs"
            )
        try:
            payload = self._cas.read(
                object_id=envelope.content_ref.object_id,
                digest=envelope.content_ref.object_sha256,
            )
        except ContentAddressedByteStoreCorruptionError as exc:
            raise ProductionAttestationStoreIntegrityError(
                "production attestation content is missing or corrupt"
            ) from exc
        if len(payload) != envelope.canonical_size_bytes or len(payload) > self.max_bytes:
            raise ProductionAttestationStoreLimitError("stored production attestation byte size is invalid")
        return payload

    def _path(
        self,
        ref: ObjectRef,
        codec: ProductionAttestationStoreCodec,
    ) -> Path:
        return self._envelopes / codec.value / "sha256" / ref.object_sha256[:2] / f"{ref.object_sha256}.json"

    def _write_exclusive(self, path: Path, payload: bytes) -> None:
        _safe_member(self._envelopes, path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            with suppress(FileExistsError):
                os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def _fault(
        self,
        point: ProductionAttestationStoreFaultPoint,
        ref: ObjectRef,
    ) -> None:
        if self.fault_injector is not None:
            self.fault_injector.maybe_raise(point, object_ref=ref)


_PrivateValue = (
    TrustedAttestationIssuerRegistryV1
    | AttestationIssuanceProofV1
    | AttestationIssuanceBundleV1
    | ProductionReadinessSC014ClosureV1
    | ProductionReadinessAttestationRequestV1
    | ProductionReadinessAttestation
    | ProductionReadinessAttestationAcceptedRunV1
)


class ProductionAttestationMaterialStore:
    def __init__(
        self,
        root: Path,
        *,
        max_private_bytes: int,
        max_members: int,
        fault_injector: ProductionAttestationStoreFaultInjector | None = None,
    ) -> None:
        if max_members < 2:
            raise ProductionAttestationStoreLimitError("production attestation member limit is invalid")
        self._store = _EnvelopeStore(
            root,
            max_bytes=max_private_bytes,
            fault_injector=fault_injector,
        )
        self.root = self._store.root
        self.max_members = max_members
        self._acceptances = self.root / "acceptances"

    def put(self, value: _PrivateValue) -> ProductionAttestationStoreWrite:
        ref, codec = _private_ref_codec(value)
        if _member_count(value) > self.max_members:
            raise ProductionAttestationStoreLimitError("production attestation material exceeds member limit")
        return self._store.put(
            ref=ref,
            codec=codec,
            payload=value.canonical_json(),
        )

    def get[ModelT: _PrivateValue](
        self,
        ref: ObjectRef,
        codec: ProductionAttestationStoreCodec,
        model_type: type[ModelT],
    ) -> ModelT:
        payload = self._store.get(ref=ref, codec=codec)
        try:
            value = model_type.model_validate_json(payload)
            observed, observed_codec = _private_ref_codec(value)
        except (ValidationError, ValueError) as exc:
            raise ProductionAttestationStoreIntegrityError(
                "stored private attestation material is invalid"
            ) from exc
        if observed != ref or observed_codec is not codec or value.canonical_json() != payload:
            raise ProductionAttestationStoreIntegrityError(
                "stored private attestation material differs from ref"
            )
        return cast(ModelT, value)

    def find_acceptance(
        self,
        acceptance_key: str,
    ) -> ProductionReadinessAttestationAcceptedRunV1 | None:
        path = self._acceptance_path(acceptance_key)
        _safe_member(self._acceptances, path)
        if not path.exists():
            return None
        return self._read_acceptance(path, acceptance_key)

    def publish_acceptance(
        self,
        value: ProductionReadinessAttestationAcceptedRunV1,
    ) -> ProductionReadinessAttestationAcceptedRunV1:
        path = self._acceptance_path(value.acceptance_key)
        existing = self.find_acceptance(value.acceptance_key)
        if existing is not None:
            return existing
        index = _AcceptanceIndex(
            acceptance_key=value.acceptance_key,
            accepted_run_ref=value.to_ref(),
        )
        _write_atomic_index(path, index.canonical_json() + b"\n", replace=False)
        return self._read_acceptance(path, value.acceptance_key)

    def get_acceptance(
        self,
        acceptance_key: str,
    ) -> ProductionReadinessAttestationAcceptedRunV1:
        value = self.find_acceptance(acceptance_key)
        if value is None:
            raise ProductionAttestationStoreIntegrityError("production attestation acceptance is missing")
        return value

    def _read_acceptance(
        self,
        path: Path,
        acceptance_key: str,
    ) -> ProductionReadinessAttestationAcceptedRunV1:
        try:
            encoded = path.read_bytes()
            index = _AcceptanceIndex.model_validate_json(encoded)
        except (OSError, ValidationError, ValueError) as exc:
            raise ProductionAttestationStoreIntegrityError(
                "production attestation acceptance index is invalid"
            ) from exc
        if encoded != index.canonical_json() + b"\n" or index.acceptance_key != acceptance_key:
            raise ProductionAttestationStoreIntegrityError("production attestation acceptance index differs")
        accepted = self.get(
            index.accepted_run_ref,
            ProductionAttestationStoreCodec.ACCEPTED_RUN,
            ProductionReadinessAttestationAcceptedRunV1,
        )
        if accepted.acceptance_key != acceptance_key:
            raise ProductionAttestationStoreIntegrityError("accepted attestation differs from index")
        return accepted

    def _acceptance_path(self, acceptance_key: str) -> Path:
        digest = hashlib.sha256(acceptance_key.encode()).hexdigest()
        return self._acceptances / "sha256" / digest[:2] / f"{digest}.json"


_PublicValue = (
    ProductionReadinessAttestationPolicyV2
    | ProductionReadinessInvalidationPolicyV2
    | ProductionReadinessVersionSetV2
    | ProductionReadinessAttestationPrerequisiteV2
    | ProductionReadinessAttestationProjectionV2
    | ProductionReadinessInvalidationRecordV2
    | ProductionReadinessAttestationResultV2
)


class ProductionAttestationReportStore:
    def __init__(
        self,
        root: Path,
        *,
        max_report_bytes: int,
        fault_injector: ProductionAttestationStoreFaultInjector | None = None,
    ) -> None:
        self._store = _EnvelopeStore(
            root,
            max_bytes=max_report_bytes,
            fault_injector=fault_injector,
        )
        self.root = self._store.root
        self._heads = self.root / "current-projections"

    def put(self, value: _PublicValue) -> ProductionAttestationStoreWrite:
        _validate_public(value)
        ref, codec = _public_ref_codec(value)
        return self._store.put(
            ref=ref,
            codec=codec,
            payload=value.canonical_json(),
        )

    def get[ModelT: _PublicValue](
        self,
        ref: ObjectRef,
        codec: ProductionAttestationStoreCodec,
        model_type: type[ModelT],
    ) -> ModelT:
        payload = self._store.get(ref=ref, codec=codec)
        try:
            value = model_type.model_validate_json(payload)
            _validate_public(value)
            observed, observed_codec = _public_ref_codec(value)
        except (ValidationError, ValueError) as exc:
            raise ProductionAttestationStoreIntegrityError(
                "stored public attestation value is invalid"
            ) from exc
        if observed != ref or observed_codec is not codec or value.canonical_json() != payload:
            raise ProductionAttestationStoreIntegrityError("stored public attestation value differs from ref")
        return cast(ModelT, value)

    def publish_current_projection(
        self,
        *,
        series_key: str,
        projection: ProductionReadinessAttestationProjectionV2,
        expected_prior_ref: ObjectRef | None,
    ) -> ProductionReadinessAttestationProjectionV2:
        if series_key != projection.attestation_series_id:
            raise ProductionAttestationStoreTypeError("current projection series key differs from projection")
        self.put(projection)
        path = self._head_path(series_key)
        with _index_lock(self._heads, path):
            existing = self.find_current_projection(series_key)
            if existing is not None:
                if existing.to_ref() == projection.to_ref():
                    return existing
                if expected_prior_ref is None or existing.to_ref() != expected_prior_ref:
                    raise ProductionAttestationStoreConflictError("current attestation projection conflicted")
                _validate_projection_transition(existing, projection)
            else:
                if expected_prior_ref is not None:
                    raise ProductionAttestationStoreIntegrityError(
                        "prior current attestation projection is missing"
                    )
                if projection.attestation_version != 1 or projection.projection_revision != 1:
                    raise ProductionAttestationStoreIntegrityError(
                        "initial current projection must start at version one"
                    )
            index = _CurrentProjectionIndex(
                series_key=series_key,
                projection_ref=projection.to_ref(),
            )
            _write_atomic_index(
                path,
                index.canonical_json() + b"\n",
                replace=True,
            )
        return self.get(
            projection.to_ref(),
            ProductionAttestationStoreCodec.PROJECTION,
            ProductionReadinessAttestationProjectionV2,
        )

    def find_current_projection(
        self,
        series_key: str,
    ) -> ProductionReadinessAttestationProjectionV2 | None:
        path = self._head_path(series_key)
        _safe_member(self._heads, path)
        if not path.exists():
            return None
        try:
            encoded = path.read_bytes()
            index = _CurrentProjectionIndex.model_validate_json(encoded)
        except (OSError, ValidationError, ValueError) as exc:
            raise ProductionAttestationStoreIntegrityError(
                "current attestation projection index is invalid"
            ) from exc
        if encoded != index.canonical_json() + b"\n" or index.series_key != series_key:
            raise ProductionAttestationStoreIntegrityError("current attestation projection index differs")
        projection = self.get(
            index.projection_ref,
            ProductionAttestationStoreCodec.PROJECTION,
            ProductionReadinessAttestationProjectionV2,
        )
        if projection.attestation_series_id != series_key:
            raise ProductionAttestationStoreIntegrityError("current projection differs from series index")
        return projection

    @contextmanager
    def hold_current_projection(
        self,
        series_key: str,
    ) -> Generator[ProductionReadinessAttestationProjectionV2 | None, None, None]:
        path = self._head_path(series_key)
        with _index_lock(self._heads, path, shared=True):
            yield self.find_current_projection(series_key)

    def publish_invalidation(
        self,
        *,
        series_key: str,
        invalidation_policy: ProductionReadinessInvalidationPolicyV2,
        prior: ProductionReadinessAttestationProjectionV2,
        invalidation_record: ProductionReadinessInvalidationRecordV2,
        successor: ProductionReadinessAttestationProjectionV2,
    ) -> ProductionReadinessAttestationProjectionV2:
        stored_policy = self.get(
            invalidation_policy.to_ref(),
            ProductionAttestationStoreCodec.INVALIDATION_POLICY,
            ProductionReadinessInvalidationPolicyV2,
        )
        if stored_policy != invalidation_policy:
            raise ProductionAttestationStoreIntegrityError(
                "stored invalidation policy differs from publication authority"
            )
        if (
            prior.projection_revision >= invalidation_policy.max_projection_revisions
            or len(invalidation_record.triggers)
            + int(invalidation_record.explicit_revocation_ref is not None)
            > invalidation_policy.max_invalidation_refs
        ):
            raise ProductionAttestationStoreLimitError("invalidation publication exceeds policy limit")
        if (
            invalidation_record.prior_projection_ref != prior.to_ref()
            or successor.previous_projection_ref != prior.to_ref()
            or successor.invalidation_record_ref != invalidation_record.to_ref()
        ):
            raise ProductionAttestationStoreIntegrityError(
                "invalidation successor closure differs from prior authority"
            )
        self.put(invalidation_record)
        self.put(successor)
        return self.publish_current_projection(
            series_key=series_key,
            projection=successor,
            expected_prior_ref=prior.to_ref(),
        )

    def _head_path(self, series_key: str) -> Path:
        digest = hashlib.sha256(series_key.encode()).hexdigest()
        return self._heads / "sha256" / digest[:2] / f"{digest}.json"


class ProductionAttestationAcceptedRunStore:
    def __init__(
        self,
        *,
        material_store: ProductionAttestationMaterialStore,
        report_store: ProductionAttestationReportStore,
        fault_injector: ProductionAttestationStoreFaultInjector | None = None,
    ) -> None:
        _disjoint(material_store.root, report_store.root)
        self.material_store = material_store
        self.report_store = report_store
        self.fault_injector = fault_injector

    def persist(
        self,
        *,
        acceptance_key: str,
        request_sha256: str,
        compilation: ProductionAttestationCompilation,
        result: ProductionReadinessAttestationResultV2,
        frozen: ProductionReadinessAttestation | None,
        audit: ContractAudit,
        previous_acceptance_key: str | None = None,
    ) -> ProductionReadinessAttestationAcceptedRunV1:
        existing = self.material_store.find_acceptance(acceptance_key)
        if existing is not None:
            if existing.request_sha256 != request_sha256:
                raise ProductionAttestationStoreConflictError("accepted attestation request changed")
            if existing.result_ref != result.to_ref():
                raise ProductionAttestationStoreConflictError("accepted attestation result changed")
            stored_result = self._verify(existing)
            self._verify_previous_authority(
                stored_result,
                request=compilation.request,
                previous_acceptance_key=previous_acceptance_key,
            )
            if stored_result.projection is not None:
                if compilation.request is None:
                    raise ProductionAttestationStoreIntegrityError("issued projection has no private request")
                self.report_store.publish_current_projection(
                    series_key=stored_result.attestation_series_id,
                    projection=stored_result.projection,
                    expected_prior_ref=(compilation.request.previous_projection_ref),
                )
            return existing
        expected_result, expected_frozen = ProductionAttestationEvaluator().evaluate(
            compilation=compilation,
            audit=result.audit,
        )
        if expected_result != result or expected_frozen != frozen:
            raise ProductionAttestationStoreIntegrityError(
                "attestation result differs from private evaluation"
            )
        self._verify_previous_authority(
            expected_result,
            request=compilation.request,
            previous_acceptance_key=previous_acceptance_key,
        )
        registry_ref = request_ref = bundle_ref = frozen_ref = projection_ref = None
        if compilation.request is not None:
            registry = compilation.trusted_registry
            bundle = compilation.issuance_bundle
            if registry is None or bundle is None:
                raise ProductionAttestationStoreIntegrityError(
                    "full attestation private request closure is incomplete"
                )
            self.material_store.put(registry)
            for proof in bundle.proofs:
                self.material_store.put(proof)
            private_values: tuple[_PrivateValue, ...] = (
                bundle,
                compilation.request.sc_014_closure,
                compilation.request,
            )
            for private_value in private_values:
                self.material_store.put(private_value)
            registry_ref = registry.to_ref()
            request_ref = compilation.request.to_ref()
            bundle_ref = bundle.to_ref()
            if result.projection is None:
                if frozen is not None:
                    raise ProductionAttestationStoreIntegrityError(
                        "non-issued attestation carries frozen authority"
                    )
            else:
                if frozen is None:
                    raise ProductionAttestationStoreIntegrityError(
                        "issued attestation frozen authority is missing"
                    )
                self.material_store.put(frozen)
                frozen_ref = production_readiness_attestation_v1_ref(frozen)
                projection_ref = result.projection.to_ref()
        elif (
            compilation.trusted_registry is not None
            or compilation.issuance_bundle is not None
            or frozen is not None
            or result.projection is not None
        ):
            raise ProductionAttestationStoreIntegrityError(
                "repository pending compilation carries private authority"
            )
        public_values: tuple[_PublicValue, ...] = (
            compilation.policy,
            compilation.invalidation_policy,
            compilation.version_set,
            compilation.prerequisite,
            *(tuple((result.projection,)) if result.projection is not None else ()),
            result,
        )
        for public_value in public_values:
            self.report_store.put(public_value)
        accepted = ProductionReadinessAttestationAcceptedRunV1.create(
            acceptance_key=acceptance_key,
            request_sha256=request_sha256,
            result_ref=result.to_ref(),
            registry_ref=registry_ref,
            request_ref=request_ref,
            bundle_ref=bundle_ref,
            frozen_attestation_ref=frozen_ref,
            current_projection_ref=projection_ref,
            audit=audit,
        )
        self.material_store.put(accepted)
        self._fault(
            ProductionAttestationStoreFaultPoint.BEFORE_ACCEPTANCE_INDEX,
            accepted.to_ref(),
        )
        authoritative = self.material_store.publish_acceptance(accepted)
        if authoritative.to_ref() != accepted.to_ref():
            raise ProductionAttestationStoreConflictError("production attestation first authority conflicted")
        if result.projection is not None:
            if compilation.request is None:
                raise ProductionAttestationStoreIntegrityError("issued projection has no private request")
            self.report_store.publish_current_projection(
                series_key=result.attestation_series_id,
                projection=result.projection,
                expected_prior_ref=compilation.request.previous_projection_ref,
            )
        self._verify(authoritative)
        return authoritative

    def get_accepted_result(
        self,
        acceptance_key: str,
        *,
        previous_acceptance_key: str | None = None,
    ) -> ProductionReadinessAttestationResultV2:
        accepted = self.material_store.get_acceptance(acceptance_key)
        result = self._verify(accepted)
        request = (
            self.material_store.get(
                accepted.request_ref,
                ProductionAttestationStoreCodec.REQUEST,
                ProductionReadinessAttestationRequestV1,
            )
            if accepted.request_ref is not None
            else None
        )
        self._verify_previous_authority(
            result,
            request=request,
            previous_acceptance_key=previous_acceptance_key,
        )
        return result

    def replay_accepted_result(
        self,
        acceptance_key: str,
        *,
        previous_acceptance_key: str | None = None,
    ) -> ProductionReadinessAttestationResultV2:
        result = self.get_accepted_result(
            acceptance_key,
            previous_acceptance_key=previous_acceptance_key,
        )
        if result.projection is None:
            return result
        accepted = self.material_store.get_acceptance(acceptance_key)
        if accepted.request_ref is None:
            raise ProductionAttestationStoreIntegrityError("issued acceptance has no private request")
        request = self.material_store.get(
            accepted.request_ref,
            ProductionAttestationStoreCodec.REQUEST,
            ProductionReadinessAttestationRequestV1,
        )
        current = self.report_store.find_current_projection(result.attestation_series_id)
        if current is None:
            self.report_store.publish_current_projection(
                series_key=result.attestation_series_id,
                projection=result.projection,
                expected_prior_ref=request.previous_projection_ref,
            )
        elif current.attestation_version < result.attestation_version or (
            current.attestation_version == result.attestation_version
            and current.projection_revision == result.projection.projection_revision
            and current.to_ref() != result.projection.to_ref()
        ):
            raise ProductionAttestationStoreConflictError("current projection differs from accepted replay")
        return result

    def _verify_previous_authority(
        self,
        result: ProductionReadinessAttestationResultV2,
        *,
        request: ProductionReadinessAttestationRequestV1 | None,
        previous_acceptance_key: str | None,
    ) -> None:
        if result.attestation_version == 1:
            if previous_acceptance_key is not None:
                raise ProductionAttestationStoreConflictError(
                    "attestation version 1 cannot bind previous authority"
                )
            return
        if request is None or previous_acceptance_key is None:
            raise ProductionAttestationStoreIntegrityError(
                "later attestation version requires accepted previous authority"
            )
        previous = self._verify(self.material_store.get_acceptance(previous_acceptance_key))
        if previous.projection is None or request.previous_projection_ref is None:
            raise ProductionAttestationStoreConflictError("previous attestation has no issued projection")
        requested_projection = self.report_store.get(
            request.previous_projection_ref,
            ProductionAttestationStoreCodec.PROJECTION,
            ProductionReadinessAttestationProjectionV2,
        )
        initial_projection = previous.projection
        projection_matches = requested_projection.to_ref() == (initial_projection.to_ref()) or (
            requested_projection.previous_projection_ref == initial_projection.to_ref()
            and requested_projection.attestation_series_id == initial_projection.attestation_series_id
            and requested_projection.attestation_version == initial_projection.attestation_version
            and requested_projection.frozen_attestation_ref == initial_projection.frozen_attestation_ref
            and requested_projection.version_set_ref == initial_projection.version_set_ref
            and requested_projection.prerequisite_ref == initial_projection.prerequisite_ref
            and requested_projection.issuer_registry_ref == initial_projection.issuer_registry_ref
        )
        if (
            previous.attestation_series_id != result.attestation_series_id
            or previous.attestation_version != result.attestation_version - 1
            or result.previous_result_ref != previous.to_ref()
            or not projection_matches
        ):
            raise ProductionAttestationStoreConflictError(
                "previous attestation authority differs from accepted series"
            )

    def _verify(
        self,
        accepted: ProductionReadinessAttestationAcceptedRunV1,
    ) -> ProductionReadinessAttestationResultV2:
        result = self.report_store.get(
            accepted.result_ref,
            ProductionAttestationStoreCodec.RESULT,
            ProductionReadinessAttestationResultV2,
        )
        policy = self.report_store.get(
            result.policy_ref,
            ProductionAttestationStoreCodec.POLICY,
            ProductionReadinessAttestationPolicyV2,
        )
        invalidation_policy = self.report_store.get(
            policy.invalidation_policy_ref,
            ProductionAttestationStoreCodec.INVALIDATION_POLICY,
            ProductionReadinessInvalidationPolicyV2,
        )
        if invalidation_policy != policy.invalidation_policy:
            raise ProductionAttestationStoreIntegrityError(
                "stored invalidation policy differs from attestation policy"
            )
        version_set = self.report_store.get(
            result.version_set.to_ref(),
            ProductionAttestationStoreCodec.VERSION_SET,
            ProductionReadinessVersionSetV2,
        )
        prerequisite = self.report_store.get(
            result.prerequisite.to_ref(),
            ProductionAttestationStoreCodec.PREREQUISITE,
            ProductionReadinessAttestationPrerequisiteV2,
        )
        if version_set != result.version_set or prerequisite != result.prerequisite:
            raise ProductionAttestationStoreIntegrityError(
                "stored attestation public closure differs from result"
            )
        if accepted.request_ref is None:
            if any(
                item is not None
                for item in (
                    accepted.registry_ref,
                    accepted.bundle_ref,
                    accepted.frozen_attestation_ref,
                    accepted.current_projection_ref,
                    result.projection,
                )
            ):
                raise ProductionAttestationStoreIntegrityError(
                    "pending attestation acceptance carries private authority"
                )
            return result
        assert accepted.registry_ref is not None
        assert accepted.bundle_ref is not None
        registry = self.material_store.get(
            accepted.registry_ref,
            ProductionAttestationStoreCodec.ISSUER_REGISTRY,
            TrustedAttestationIssuerRegistryV1,
        )
        request = self.material_store.get(
            accepted.request_ref,
            ProductionAttestationStoreCodec.REQUEST,
            ProductionReadinessAttestationRequestV1,
        )
        bundle = self.material_store.get(
            accepted.bundle_ref,
            ProductionAttestationStoreCodec.ISSUANCE_BUNDLE,
            AttestationIssuanceBundleV1,
        )
        if request.issuance_bundle != bundle:
            raise ProductionAttestationStoreIntegrityError("stored issuance bundle differs from request")
        for proof in bundle.proofs:
            stored = self.material_store.get(
                proof.to_ref(),
                ProductionAttestationStoreCodec.ISSUANCE_PROOF,
                AttestationIssuanceProofV1,
            )
            if stored != proof:
                raise ProductionAttestationStoreIntegrityError("stored issuance proof differs from bundle")
        closure = self.material_store.get(
            request.sc_014_closure.to_ref(),
            ProductionAttestationStoreCodec.SC014_CLOSURE,
            ProductionReadinessSC014ClosureV1,
        )
        if closure != request.sc_014_closure:
            raise ProductionAttestationStoreIntegrityError("stored SC-014 closure differs from request")
        frozen: ProductionReadinessAttestation | None = None
        if result.projection is None:
            if accepted.frozen_attestation_ref is not None or accepted.current_projection_ref is not None:
                raise ProductionAttestationStoreIntegrityError(
                    "non-issued acceptance carries frozen authority"
                )
        else:
            if (
                accepted.frozen_attestation_ref is None
                or accepted.current_projection_ref != result.projection.to_ref()
            ):
                raise ProductionAttestationStoreIntegrityError(
                    "issued acceptance projection closure is incomplete"
                )
            projection = self.report_store.get(
                accepted.current_projection_ref,
                ProductionAttestationStoreCodec.PROJECTION,
                ProductionReadinessAttestationProjectionV2,
            )
            if projection != result.projection:
                raise ProductionAttestationStoreIntegrityError("stored projection differs from result")
            frozen = self.material_store.get(
                accepted.frozen_attestation_ref,
                ProductionAttestationStoreCodec.FROZEN_ATTESTATION,
                ProductionReadinessAttestation,
            )
        compilation = ProductionAttestationBuilder().compile_full(
            policy=policy,
            request=request,
            trusted_registry=registry,
        )
        expected, expected_frozen = ProductionAttestationEvaluator().evaluate(
            compilation=compilation,
            audit=result.audit,
        )
        if expected != result or expected_frozen != frozen:
            raise ProductionAttestationStoreIntegrityError(
                "accepted attestation differs from stored private closure"
            )
        return result

    def _fault(
        self,
        point: ProductionAttestationStoreFaultPoint,
        ref: ObjectRef,
    ) -> None:
        if self.fault_injector is not None:
            self.fault_injector.maybe_raise(point, object_ref=ref)


def _private_ref_codec(
    value: _PrivateValue,
) -> tuple[ObjectRef, ProductionAttestationStoreCodec]:
    if isinstance(value, TrustedAttestationIssuerRegistryV1):
        return value.to_ref(), ProductionAttestationStoreCodec.ISSUER_REGISTRY
    if isinstance(value, AttestationIssuanceProofV1):
        return value.to_ref(), ProductionAttestationStoreCodec.ISSUANCE_PROOF
    if isinstance(value, AttestationIssuanceBundleV1):
        return value.to_ref(), ProductionAttestationStoreCodec.ISSUANCE_BUNDLE
    if isinstance(value, ProductionReadinessSC014ClosureV1):
        return value.to_ref(), ProductionAttestationStoreCodec.SC014_CLOSURE
    if isinstance(value, ProductionReadinessAttestationRequestV1):
        return value.to_ref(), ProductionAttestationStoreCodec.REQUEST
    if isinstance(value, ProductionReadinessAttestation):
        validate_production_readiness_attestation_v1_identity(value)
        return (
            production_readiness_attestation_v1_ref(value),
            ProductionAttestationStoreCodec.FROZEN_ATTESTATION,
        )
    if isinstance(value, ProductionReadinessAttestationAcceptedRunV1):
        return value.to_ref(), ProductionAttestationStoreCodec.ACCEPTED_RUN
    raise ProductionAttestationStoreTypeError("unsupported private production attestation type")


def _public_ref_codec(
    value: _PublicValue,
) -> tuple[ObjectRef, ProductionAttestationStoreCodec]:
    if isinstance(value, ProductionReadinessAttestationPolicyV2):
        return value.to_ref(), ProductionAttestationStoreCodec.POLICY
    if isinstance(value, ProductionReadinessInvalidationPolicyV2):
        return value.to_ref(), ProductionAttestationStoreCodec.INVALIDATION_POLICY
    if isinstance(value, ProductionReadinessVersionSetV2):
        return value.to_ref(), ProductionAttestationStoreCodec.VERSION_SET
    if isinstance(value, ProductionReadinessAttestationPrerequisiteV2):
        return value.to_ref(), ProductionAttestationStoreCodec.PREREQUISITE
    if isinstance(value, ProductionReadinessAttestationProjectionV2):
        return value.to_ref(), ProductionAttestationStoreCodec.PROJECTION
    if isinstance(value, ProductionReadinessInvalidationRecordV2):
        return value.to_ref(), ProductionAttestationStoreCodec.INVALIDATION_RECORD
    if isinstance(value, ProductionReadinessAttestationResultV2):
        return value.to_ref(), ProductionAttestationStoreCodec.RESULT
    raise ProductionAttestationStoreTypeError("unsupported public production attestation type")


def _validate_public(value: _PublicValue) -> None:
    if isinstance(value, ProductionReadinessAttestationPolicyV2):
        validate_production_readiness_attestation_policy_v2_identity(value)
    elif isinstance(value, ProductionReadinessInvalidationPolicyV2):
        validate_production_readiness_invalidation_policy_v2_identity(value)
    elif isinstance(value, ProductionReadinessVersionSetV2):
        validate_production_readiness_version_set_v2_identity(value)
    elif isinstance(value, ProductionReadinessAttestationPrerequisiteV2):
        validate_production_readiness_attestation_prerequisite_v2_identity(value)
    elif isinstance(value, ProductionReadinessAttestationProjectionV2):
        validate_production_readiness_attestation_projection_v2_identity(value)
    elif isinstance(value, ProductionReadinessInvalidationRecordV2):
        validate_production_readiness_invalidation_record_v2_identity(value)
    elif isinstance(value, ProductionReadinessAttestationResultV2):
        validate_production_readiness_attestation_result_v2_identity(value)
    else:
        raise ProductionAttestationStoreTypeError("unsupported public production attestation type")


def _member_count(value: _PrivateValue) -> int:
    if isinstance(value, TrustedAttestationIssuerRegistryV1):
        return len(value.issuers)
    if isinstance(value, AttestationIssuanceBundleV1):
        return len(value.proofs)
    if isinstance(value, ProductionReadinessAttestationRequestV1):
        return len(value.issuance_bundle.proofs) + 5
    return 1


def _write_atomic_index(path: Path, payload: bytes, *, replace: bool) -> None:
    _safe_member(path.parents[2], path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            with suppress(FileExistsError):
                os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _index_lock(
    root: Path,
    index_path: Path,
    *,
    shared: bool = False,
) -> Generator[None, None, None]:
    digest = hashlib.sha256(str(index_path.relative_to(root)).encode()).hexdigest()
    lock_path = root / "locks" / digest[:2] / f"{digest}.lock"
    _safe_member(root, lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        lock_path,
        os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "a+b", closefd=True) as handle:
            fcntl.flock(
                handle.fileno(),
                fcntl.LOCK_SH if shared else fcntl.LOCK_EX,
            )
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except BaseException:
        with suppress(OSError):
            os.close(descriptor)
        raise


def _validate_projection_transition(
    prior: ProductionReadinessAttestationProjectionV2,
    successor: ProductionReadinessAttestationProjectionV2,
) -> None:
    if successor.attestation_series_id != prior.attestation_series_id:
        raise ProductionAttestationStoreConflictError("successor projection changed attestation series")
    if successor.attestation_version == prior.attestation_version:
        if (
            successor.projection_revision != prior.projection_revision + 1
            or successor.previous_projection_ref != prior.to_ref()
        ):
            raise ProductionAttestationStoreConflictError("successor projection revision is not contiguous")
        return
    if (
        successor.attestation_version != prior.attestation_version + 1
        or successor.projection_revision != 1
        or successor.previous_projection_ref is not None
        or successor.state is not ProductionReadinessAttestationStateV2.ACTIVE
    ):
        raise ProductionAttestationStoreConflictError("successor attestation version is not contiguous")


def _safe_member(root: Path, path: Path) -> None:
    if root.is_symlink() or (root.exists() and not root.is_dir()):
        raise ProductionAttestationStoreTypeError("production attestation root has invalid file type")
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ProductionAttestationStoreTypeError("production attestation member escapes root") from exc
    current = root
    for index, part in enumerate(relative.parts):
        current = current / part
        if current.is_symlink():
            raise ProductionAttestationStoreTypeError("production attestation member uses symlink")
        if current.exists():
            leaf = index == len(relative.parts) - 1
            if (leaf and not current.is_file()) or (not leaf and not current.is_dir()):
                raise ProductionAttestationStoreTypeError(
                    "production attestation member has invalid file type"
                )


def _disjoint(first: Path, second: Path) -> None:
    left = first.resolve()
    right = second.resolve()
    if left == right or left in right.parents or right in left.parents:
        raise ProductionAttestationStoreTypeError("production attestation roots overlap")


__all__ = [
    "ProductionAttestationAcceptedRunStore",
    "ProductionAttestationMaterialStore",
    "ProductionAttestationReportStore",
    "ProductionAttestationStoreCodec",
    "ProductionAttestationStoreConflictError",
    "ProductionAttestationStoreError",
    "ProductionAttestationStoreFaultPoint",
    "ProductionAttestationStoreInjectedCrash",
    "ProductionAttestationStoreIntegrityError",
    "ProductionAttestationStoreLimitError",
    "ProductionAttestationStoreTypeError",
    "ProductionAttestationStoreWrite",
    "StaticProductionAttestationStoreFaultInjector",
]
