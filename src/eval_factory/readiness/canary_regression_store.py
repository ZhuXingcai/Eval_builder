from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal, Protocol, Self
from uuid import uuid4

from pydantic import Field, ValidationError, model_validator

from eval_factory.contracts.canary_regression_v2 import (
    CanaryRegressionReportV2,
    canary_regression_report_v2_ref,
    validate_canary_regression_report_v2_identity,
)
from eval_factory.contracts.core import ObjectRef
from eval_factory.contracts.core_v2 import ContractModelV2
from eval_factory.storage import (
    ContentAddressedByteStore,
    ContentAddressedByteStoreConflictError,
    ContentAddressedByteStoreCorruptionError,
)


class CanaryRegressionReportStoreError(RuntimeError):
    pass


class CanaryRegressionReportTypeError(CanaryRegressionReportStoreError):
    pass


class CanaryRegressionReportConflictError(CanaryRegressionReportStoreError):
    pass


class CanaryRegressionReportIntegrityError(CanaryRegressionReportStoreError):
    pass


class CanaryRegressionReportLimitError(CanaryRegressionReportStoreError):
    pass


class CanaryRegressionReportInjectedCrash(CanaryRegressionReportStoreError):
    pass


class CanaryRegressionReportStoreFaultPoint(StrEnum):
    AFTER_CAS_WRITE = "after_cas_write"
    AFTER_ENVELOPE_WRITE = "after_envelope_write"


class CanaryRegressionReportStoreFaultInjector(Protocol):
    def maybe_raise(
        self,
        point: CanaryRegressionReportStoreFaultPoint,
        *,
        object_ref: ObjectRef,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class StaticCanaryRegressionReportStoreFaultInjector:
    crash_points: frozenset[CanaryRegressionReportStoreFaultPoint] = frozenset()

    def maybe_raise(
        self,
        point: CanaryRegressionReportStoreFaultPoint,
        *,
        object_ref: ObjectRef,
    ) -> None:
        del object_ref
        if point in self.crash_points:
            raise CanaryRegressionReportInjectedCrash(
                f"injected canary regression report crash at {point.value}"
            )


@dataclass(frozen=True, slots=True)
class CanaryRegressionReportWrite:
    object_ref: ObjectRef
    written: bool
    content_blob_written: bool
    canonical_size_bytes: int


class _CanaryRegressionReportEnvelopeV1(ContractModelV2):
    schema_version: Literal["eval-factory/canary-regression-report-envelope/private-v1"] = (
        "eval-factory/canary-regression-report-envelope/private-v1"
    )
    object_ref: ObjectRef
    content_blob_ref: ObjectRef
    canonical_size_bytes: int = Field(ge=2)

    @model_validator(mode="after")
    def validate_envelope(self) -> Self:
        if (
            self.object_ref.object_type != "canary-regression-report"
            or self.object_ref.object_version != "v2"
            or self.content_blob_ref.object_type != "canary-regression-report-content"
            or self.content_blob_ref.object_version != "private-v1"
            or self.content_blob_ref.object_id
            != (f"canary-regression-report-content://sha256/{self.content_blob_ref.object_sha256}")
        ):
            raise ValueError("canary regression report envelope refs are invalid")
        return self


class CanaryRegressionReportStore:
    def __init__(
        self,
        root: Path,
        *,
        max_report_bytes: int,
        max_case_refs: int,
        fault_injector: CanaryRegressionReportStoreFaultInjector | None = None,
    ) -> None:
        if max_report_bytes < 2 or max_case_refs < 24:
            raise CanaryRegressionReportLimitError("canary regression report limits are invalid")
        candidate = root.expanduser()
        if candidate.is_symlink() or (candidate.exists() and not candidate.is_dir()):
            raise CanaryRegressionReportTypeError(
                "canary regression report root must be a non-symlink directory"
            )
        resolved = candidate.resolve()
        self.max_report_bytes = max_report_bytes
        self.max_case_refs = max_case_refs
        self.fault_injector = fault_injector
        self._cas = ContentAddressedByteStore(resolved / "cas")
        self._envelopes = resolved / "envelopes"

    def put(
        self,
        value: CanaryRegressionReportV2,
    ) -> CanaryRegressionReportWrite:
        try:
            validate_canary_regression_report_v2_identity(value)
        except ValueError as exc:
            raise CanaryRegressionReportConflictError("canary regression report identity is stale") from exc
        ref = canary_regression_report_v2_ref(value)
        payload = value.canonical_json()
        self._require_limits(value, payload)
        existing = self._existing_write(ref)
        if existing is not None:
            return existing
        digest = hashlib.sha256(payload).hexdigest()
        content_ref = ObjectRef(
            object_type="canary-regression-report-content",
            object_id=(f"canary-regression-report-content://sha256/{digest}"),
            object_version="private-v1",
            object_sha256=digest,
        )
        envelope = _CanaryRegressionReportEnvelopeV1(
            object_ref=ref,
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
            raise CanaryRegressionReportConflictError(
                "canary regression report CAS write conflicted"
            ) from exc
        self._maybe_raise(
            CanaryRegressionReportStoreFaultPoint.AFTER_CAS_WRITE,
            ref,
        )
        written = self._write_envelope(envelope)
        self._maybe_raise(
            CanaryRegressionReportStoreFaultPoint.AFTER_ENVELOPE_WRITE,
            ref,
        )
        if not written:
            existing = self._existing_write(ref)
            if existing is None:
                raise CanaryRegressionReportIntegrityError("canary regression report envelope disappeared")
            return existing
        self.get(ref)
        return CanaryRegressionReportWrite(
            object_ref=ref,
            written=True,
            content_blob_written=content_written,
            canonical_size_bytes=len(payload),
        )

    def get(
        self,
        ref: ObjectRef,
    ) -> CanaryRegressionReportV2:
        self._require_ref(ref)
        envelope = self._read_envelope(ref)
        payload = self._read_payload(envelope)
        try:
            value = CanaryRegressionReportV2.model_validate_json(payload)
            validate_canary_regression_report_v2_identity(value)
            observed = canary_regression_report_v2_ref(value)
        except (ValidationError, ValueError) as exc:
            raise CanaryRegressionReportIntegrityError("stored canary regression report is invalid") from exc
        self._require_limits(value, payload)
        if observed != ref or value.canonical_json() != payload:
            raise CanaryRegressionReportIntegrityError(
                "stored canary regression report differs from its reference"
            )
        return value

    def verify(self, ref: ObjectRef) -> None:
        self.get(ref)

    def _existing_write(
        self,
        ref: ObjectRef,
    ) -> CanaryRegressionReportWrite | None:
        path = self._envelope_path(ref)
        if not path.exists():
            return None
        envelope = self._read_envelope(ref)
        self.get(ref)
        return CanaryRegressionReportWrite(
            object_ref=ref,
            written=False,
            content_blob_written=False,
            canonical_size_bytes=envelope.canonical_size_bytes,
        )

    def _read_payload(
        self,
        envelope: _CanaryRegressionReportEnvelopeV1,
    ) -> bytes:
        try:
            payload = self._cas.read(
                object_id=envelope.content_blob_ref.object_id,
                digest=envelope.content_blob_ref.object_sha256,
            )
        except ContentAddressedByteStoreCorruptionError as exc:
            raise CanaryRegressionReportIntegrityError(
                "canary regression report content is missing or corrupt"
            ) from exc
        if len(payload) > self.max_report_bytes or len(payload) != envelope.canonical_size_bytes:
            raise CanaryRegressionReportLimitError("stored canary regression report byte size is invalid")
        return payload

    def _write_envelope(
        self,
        envelope: _CanaryRegressionReportEnvelopeV1,
    ) -> bool:
        path = self._envelope_path(envelope.object_ref)
        if path.exists():
            self._read_envelope(envelope.object_ref)
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
                self._read_envelope(envelope.object_ref)
                return False
        finally:
            temporary.unlink(missing_ok=True)
        return True

    def _read_envelope(
        self,
        ref: ObjectRef,
    ) -> _CanaryRegressionReportEnvelopeV1:
        path = self._envelope_path(ref)
        if not path.exists():
            raise CanaryRegressionReportIntegrityError("canary regression report envelope is missing")
        try:
            envelope = _CanaryRegressionReportEnvelopeV1.model_validate_json(path.read_bytes())
        except (OSError, ValidationError, ValueError) as exc:
            raise CanaryRegressionReportIntegrityError(
                "canary regression report envelope is invalid"
            ) from exc
        if envelope.object_ref != ref:
            raise CanaryRegressionReportIntegrityError(
                "canary regression report envelope differs from its reference"
            )
        return envelope

    def _envelope_path(self, ref: ObjectRef) -> Path:
        return self._envelopes / "sha256" / ref.object_sha256[:2] / f"{ref.object_sha256}.json"

    def _require_limits(
        self,
        value: CanaryRegressionReportV2,
        payload: bytes,
    ) -> None:
        if len(payload) > self.max_report_bytes:
            raise CanaryRegressionReportLimitError("canary regression report exceeds byte limit")
        if len(value.case_result_refs) > self.max_case_refs:
            raise CanaryRegressionReportLimitError("canary regression report exceeds case-ref limit")

    @staticmethod
    def _require_ref(ref: ObjectRef) -> None:
        if ref.object_type != "canary-regression-report" or ref.object_version != "v2":
            raise CanaryRegressionReportTypeError("canary regression report reference has the wrong type")

    def _maybe_raise(
        self,
        point: CanaryRegressionReportStoreFaultPoint,
        ref: ObjectRef,
    ) -> None:
        if self.fault_injector is not None:
            self.fault_injector.maybe_raise(point, object_ref=ref)


__all__ = [
    "CanaryRegressionReportConflictError",
    "CanaryRegressionReportInjectedCrash",
    "CanaryRegressionReportIntegrityError",
    "CanaryRegressionReportLimitError",
    "CanaryRegressionReportStore",
    "CanaryRegressionReportStoreError",
    "CanaryRegressionReportStoreFaultPoint",
    "CanaryRegressionReportTypeError",
    "CanaryRegressionReportWrite",
    "StaticCanaryRegressionReportStoreFaultInjector",
]
