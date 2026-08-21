from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal, Protocol, Self
from uuid import uuid4

from pydantic import Field, ValidationError, model_validator

from eval_factory.contracts.core import ObjectRef
from eval_factory.contracts.core_v2 import ContractModelV2
from eval_factory.contracts.external_evidence_v2 import (
    ExternalEvidenceAdmissionReportV2,
    ExternalEvidencePackageManifestV2,
    ExternalLabelObservationSummaryV2,
    ExternalSc010Sc011EvidenceIndexV2,
)
from eval_factory.storage import (
    ContentAddressedByteStore,
    ContentAddressedByteStoreConflictError,
    ContentAddressedByteStoreCorruptionError,
)


class ExternalEvidenceAuthorityStoreError(RuntimeError):
    pass


class ExternalEvidenceAuthorityStoreTypeError(
    ExternalEvidenceAuthorityStoreError,
):
    pass


class ExternalEvidenceAuthorityStoreConflictError(
    ExternalEvidenceAuthorityStoreError,
):
    pass


class ExternalEvidenceAuthorityStoreIntegrityError(
    ExternalEvidenceAuthorityStoreError,
):
    pass


class ExternalEvidenceAuthorityStoreLimitError(
    ExternalEvidenceAuthorityStoreError,
):
    pass


class ExternalEvidenceAuthorityStoreInjectedCrash(
    ExternalEvidenceAuthorityStoreError,
):
    pass


class ExternalEvidenceAuthorityStoreCodec(StrEnum):
    ADMISSION_REPORT = "admission-report"
    EVIDENCE_INDEX = "evidence-index"
    OBSERVATION_SUMMARY = "observation-summary"
    PACKAGE_MANIFEST = "package-manifest"


class ExternalEvidenceAuthorityStoreFaultPoint(StrEnum):
    AFTER_CAS_WRITE = "after_cas_write"
    AFTER_ENVELOPE_WRITE = "after_envelope_write"


class ExternalEvidenceAuthorityStoreFaultInjector(Protocol):
    def maybe_raise(
        self,
        point: ExternalEvidenceAuthorityStoreFaultPoint,
        *,
        object_ref: ObjectRef,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class StaticExternalEvidenceAuthorityStoreFaultInjector:
    crash_points: frozenset[ExternalEvidenceAuthorityStoreFaultPoint] = frozenset()

    def maybe_raise(
        self,
        point: ExternalEvidenceAuthorityStoreFaultPoint,
        *,
        object_ref: ObjectRef,
    ) -> None:
        del object_ref
        if point in self.crash_points:
            raise ExternalEvidenceAuthorityStoreInjectedCrash(
                f"injected external evidence authority store crash at {point.value}"
            )


@dataclass(frozen=True, slots=True)
class ExternalEvidenceAuthorityStoreWrite:
    object_ref: ObjectRef
    written: bool
    content_blob_written: bool
    canonical_size_bytes: int


class _ExternalEvidenceAuthorityEnvelopeV1(ContractModelV2):
    schema_version: Literal["eval-factory/external-evidence-authority-envelope/private-v1"] = (
        "eval-factory/external-evidence-authority-envelope/private-v1"
    )
    object_ref: ObjectRef
    codec: ExternalEvidenceAuthorityStoreCodec
    content_blob_ref: ObjectRef
    canonical_size_bytes: int = Field(ge=2)

    @model_validator(mode="after")
    def validate_envelope(self) -> Self:
        expected_id = (
            "external-evidence-authority-content://"
            f"{self.codec.value}/sha256/"
            f"{self.content_blob_ref.object_sha256}"
        )
        if (
            self.content_blob_ref.object_type != "external-evidence-authority-content"
            or self.content_blob_ref.object_version != "private-v1"
            or self.content_blob_ref.object_id != expected_id
        ):
            raise ValueError("external evidence authority content ref is invalid")
        return self


_Authority = (
    ExternalEvidenceAdmissionReportV2
    | ExternalEvidencePackageManifestV2
    | ExternalLabelObservationSummaryV2
    | ExternalSc010Sc011EvidenceIndexV2
)


class ExternalEvidenceAuthorityStore:
    def __init__(
        self,
        root: Path,
        *,
        max_authority_bytes: int,
        fault_injector: (ExternalEvidenceAuthorityStoreFaultInjector | None) = None,
    ) -> None:
        if max_authority_bytes < 2 or max_authority_bytes > 100_000_000:
            raise ExternalEvidenceAuthorityStoreLimitError(
                "external evidence authority byte limit is invalid"
            )
        candidate = root.expanduser()
        if candidate.is_symlink() or (candidate.exists() and not candidate.is_dir()):
            raise ExternalEvidenceAuthorityStoreTypeError(
                "external evidence authority root must be a non-symlink directory"
            )
        self.max_authority_bytes = max_authority_bytes
        self.fault_injector = fault_injector
        resolved = candidate.resolve()
        self._cas = ContentAddressedByteStore(resolved / "cas")
        self._envelopes = resolved / "envelopes"

    def put(
        self,
        value: _Authority,
    ) -> ExternalEvidenceAuthorityStoreWrite:
        ref = value.to_ref()
        codec = _codec_for_value(value)
        payload = value.canonical_json()
        if len(payload) > self.max_authority_bytes:
            raise ExternalEvidenceAuthorityStoreLimitError("external evidence authority exceeds byte limit")
        existing = self._existing(ref, codec)
        if existing is not None:
            return existing
        digest = hashlib.sha256(payload).hexdigest()
        content_ref = ObjectRef(
            object_type="external-evidence-authority-content",
            object_id=(f"external-evidence-authority-content://{codec.value}/sha256/{digest}"),
            object_version="private-v1",
            object_sha256=digest,
        )
        envelope = _ExternalEvidenceAuthorityEnvelopeV1(
            object_ref=ref,
            codec=codec,
            content_blob_ref=content_ref,
            canonical_size_bytes=len(payload),
        )
        try:
            content_written = self._cas.write(
                object_id=content_ref.object_id,
                digest=digest,
                value=payload,
            )
        except ContentAddressedByteStoreConflictError as exc:
            raise ExternalEvidenceAuthorityStoreConflictError(
                "external evidence authority CAS write conflicted"
            ) from exc
        self._fault(
            ExternalEvidenceAuthorityStoreFaultPoint.AFTER_CAS_WRITE,
            ref,
        )
        written = self._write_envelope(envelope)
        self._fault(
            ExternalEvidenceAuthorityStoreFaultPoint.AFTER_ENVELOPE_WRITE,
            ref,
        )
        if not written:
            existing = self._existing(ref, codec)
            if existing is None:
                raise ExternalEvidenceAuthorityStoreIntegrityError(
                    "external evidence authority envelope disappeared"
                )
            return existing
        self.get(ref)
        return ExternalEvidenceAuthorityStoreWrite(
            object_ref=ref,
            written=True,
            content_blob_written=content_written,
            canonical_size_bytes=len(payload),
        )

    def get(
        self,
        ref: ObjectRef,
    ) -> _Authority:
        codec = _codec_for_ref(ref)
        envelope = self._read_envelope(ref, codec)
        try:
            payload = self._cas.read(
                object_id=envelope.content_blob_ref.object_id,
                digest=envelope.content_blob_ref.object_sha256,
            )
        except ContentAddressedByteStoreCorruptionError as exc:
            raise ExternalEvidenceAuthorityStoreIntegrityError(
                "external evidence authority content is missing or corrupt"
            ) from exc
        if len(payload) > self.max_authority_bytes or len(payload) != envelope.canonical_size_bytes:
            raise ExternalEvidenceAuthorityStoreIntegrityError(
                "stored external evidence authority byte size is invalid"
            )
        try:
            value = _decode(codec, payload)
        except (ValidationError, ValueError) as exc:
            raise ExternalEvidenceAuthorityStoreIntegrityError(
                "stored external evidence authority is invalid"
            ) from exc
        if value.to_ref() != ref or value.canonical_json() != payload:
            raise ExternalEvidenceAuthorityStoreIntegrityError(
                "stored external evidence authority differs from its ref"
            )
        return value

    def verify(self, ref: ObjectRef) -> None:
        self.get(ref)

    def _existing(
        self,
        ref: ObjectRef,
        codec: ExternalEvidenceAuthorityStoreCodec,
    ) -> ExternalEvidenceAuthorityStoreWrite | None:
        path = self._envelope_path(ref, codec)
        if not path.exists():
            return None
        envelope = self._read_envelope(ref, codec)
        self.get(ref)
        return ExternalEvidenceAuthorityStoreWrite(
            object_ref=ref,
            written=False,
            content_blob_written=False,
            canonical_size_bytes=envelope.canonical_size_bytes,
        )

    def _write_envelope(
        self,
        envelope: _ExternalEvidenceAuthorityEnvelopeV1,
    ) -> bool:
        path = self._envelope_path(
            envelope.object_ref,
            envelope.codec,
        )
        if path.exists():
            self._read_envelope(
                envelope.object_ref,
                envelope.codec,
            )
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as handle:
                handle.write(envelope.canonical_json() + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                self._read_envelope(
                    envelope.object_ref,
                    envelope.codec,
                )
                return False
        finally:
            temporary.unlink(missing_ok=True)
        return True

    def _read_envelope(
        self,
        ref: ObjectRef,
        codec: ExternalEvidenceAuthorityStoreCodec,
    ) -> _ExternalEvidenceAuthorityEnvelopeV1:
        path = self._envelope_path(ref, codec)
        if not path.exists():
            raise ExternalEvidenceAuthorityStoreIntegrityError(
                "external evidence authority envelope is missing"
            )
        try:
            encoded = path.read_bytes()
            envelope = _ExternalEvidenceAuthorityEnvelopeV1.model_validate_json(encoded)
        except (OSError, ValidationError, ValueError) as exc:
            raise ExternalEvidenceAuthorityStoreIntegrityError(
                "external evidence authority envelope is invalid"
            ) from exc
        if encoded != envelope.canonical_json() + b"\n":
            raise ExternalEvidenceAuthorityStoreIntegrityError(
                "external evidence authority envelope is not canonical"
            )
        if envelope.object_ref != ref or envelope.codec is not codec:
            raise ExternalEvidenceAuthorityStoreIntegrityError("external evidence authority envelope differs")
        return envelope

    def _envelope_path(
        self,
        ref: ObjectRef,
        codec: ExternalEvidenceAuthorityStoreCodec,
    ) -> Path:
        return self._envelopes / codec.value / "sha256" / ref.object_sha256[:2] / f"{ref.object_sha256}.json"

    def _fault(
        self,
        point: ExternalEvidenceAuthorityStoreFaultPoint,
        ref: ObjectRef,
    ) -> None:
        if self.fault_injector is not None:
            self.fault_injector.maybe_raise(point, object_ref=ref)


def _codec_for_value(
    value: _Authority,
) -> ExternalEvidenceAuthorityStoreCodec:
    if isinstance(value, ExternalEvidenceAdmissionReportV2):
        return ExternalEvidenceAuthorityStoreCodec.ADMISSION_REPORT
    if isinstance(value, ExternalSc010Sc011EvidenceIndexV2):
        return ExternalEvidenceAuthorityStoreCodec.EVIDENCE_INDEX
    if isinstance(value, ExternalLabelObservationSummaryV2):
        return ExternalEvidenceAuthorityStoreCodec.OBSERVATION_SUMMARY
    return ExternalEvidenceAuthorityStoreCodec.PACKAGE_MANIFEST


def _codec_for_ref(
    ref: ObjectRef,
) -> ExternalEvidenceAuthorityStoreCodec:
    expected = {
        "external-evidence-admission-report": (ExternalEvidenceAuthorityStoreCodec.ADMISSION_REPORT),
        "external-sc010-sc011-evidence-index": (ExternalEvidenceAuthorityStoreCodec.EVIDENCE_INDEX),
        "external-label-observation-summary": (ExternalEvidenceAuthorityStoreCodec.OBSERVATION_SUMMARY),
        "external-evidence-package-manifest": (ExternalEvidenceAuthorityStoreCodec.PACKAGE_MANIFEST),
    }
    codec = expected.get(ref.object_type)
    if codec is None or ref.object_version != "v2":
        raise ExternalEvidenceAuthorityStoreTypeError("external evidence authority ref has wrong type")
    return codec


def _decode(
    codec: ExternalEvidenceAuthorityStoreCodec,
    payload: bytes,
) -> _Authority:
    if codec is ExternalEvidenceAuthorityStoreCodec.ADMISSION_REPORT:
        return ExternalEvidenceAdmissionReportV2.model_validate_json(payload)
    if codec is ExternalEvidenceAuthorityStoreCodec.EVIDENCE_INDEX:
        return ExternalSc010Sc011EvidenceIndexV2.model_validate_json(payload)
    if codec is ExternalEvidenceAuthorityStoreCodec.OBSERVATION_SUMMARY:
        return ExternalLabelObservationSummaryV2.model_validate_json(payload)
    return ExternalEvidencePackageManifestV2.model_validate_json(payload)


__all__ = [
    "ExternalEvidenceAuthorityStore",
    "ExternalEvidenceAuthorityStoreCodec",
    "ExternalEvidenceAuthorityStoreConflictError",
    "ExternalEvidenceAuthorityStoreError",
    "ExternalEvidenceAuthorityStoreFaultInjector",
    "ExternalEvidenceAuthorityStoreFaultPoint",
    "ExternalEvidenceAuthorityStoreInjectedCrash",
    "ExternalEvidenceAuthorityStoreIntegrityError",
    "ExternalEvidenceAuthorityStoreLimitError",
    "ExternalEvidenceAuthorityStoreTypeError",
    "ExternalEvidenceAuthorityStoreWrite",
    "StaticExternalEvidenceAuthorityStoreFaultInjector",
]
