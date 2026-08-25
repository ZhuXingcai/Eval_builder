from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import shutil
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import BinaryIO, ClassVar, Literal, Self
from uuid import uuid4

from pydantic import Field, ValidationError, model_validator

from eval_factory.contracts.core import (
    ContractAudit,
    Identifier,
    ObjectRef,
    Sha256,
)
from eval_factory.contracts.core_v2 import ContractModelV2
from eval_factory.harness.artifacts import (
    ArtifactEnvelopeV1,
    ArtifactModalityV1,
)
from eval_factory.harness.contracts import (
    HarnessObjectV1,
    require_ref,
    require_sorted_unique,
    require_sorted_unique_refs,
    sorted_refs,
)
from eval_factory.trace import RawTrajV1Adapter, TraceProbeStatus

SOURCE_MANIFEST_COLUMNS = (
    "instance_id",
    "sid",
    "p_date",
    "business",
    "category",
    "pool_id",
    "pool_category",
)
_CHUNK_BYTES = 64 * 1024


class HarnessSourceAdmissionError(RuntimeError):
    code: str = "SOURCE_ADMISSION_INVALID"


class HarnessSourceAdmissionValidationError(HarnessSourceAdmissionError):
    pass


class HarnessSourceAdmissionMultipartError(
    HarnessSourceAdmissionValidationError,
):
    code = "SOURCE_ADMISSION_MULTIPART_INVALID"


class HarnessSourceAdmissionUnsafeNameError(
    HarnessSourceAdmissionValidationError,
):
    code = "SOURCE_ADMISSION_UNSAFE_NAME"


class HarnessSourceAdmissionManifestError(
    HarnessSourceAdmissionValidationError,
):
    code = "SOURCE_ADMISSION_MANIFEST_INVALID"


class HarnessSourceAdmissionInventoryError(
    HarnessSourceAdmissionValidationError,
):
    code = "SOURCE_ADMISSION_INVENTORY_MISMATCH"


class HarnessSourceAdmissionHashError(
    HarnessSourceAdmissionValidationError,
):
    code = "SOURCE_ADMISSION_HASH_MISMATCH"


class HarnessSourceAdmissionSourceError(
    HarnessSourceAdmissionValidationError,
):
    code = "SOURCE_ADMISSION_SOURCE_INVALID"


class HarnessSourceAdmissionMetadataError(
    HarnessSourceAdmissionValidationError,
):
    code = "SOURCE_ADMISSION_SOURCE_METADATA_MISMATCH"


class HarnessSourceAdmissionLimitError(HarnessSourceAdmissionError):
    code = "SOURCE_ADMISSION_LIMIT_EXCEEDED"


class HarnessSourceAdmissionConflictError(HarnessSourceAdmissionError):
    code = "SOURCE_ADMISSION_CONFLICT"


class HarnessSourceAdmissionIntegrityError(HarnessSourceAdmissionError):
    code = "SOURCE_ADMISSION_INTEGRITY"


class HarnessSourceAdmissionNotFoundError(HarnessSourceAdmissionError):
    code = "SOURCE_ADMISSION_NOT_FOUND"


@dataclass(frozen=True, slots=True)
class HarnessSourceAdmissionLimits:
    max_source_files: int = 500
    max_manifest_bytes: int = 1024 * 1024
    max_source_bytes: int = 16 * 1024 * 1024
    max_total_source_bytes: int = 256 * 1024 * 1024
    max_request_bytes: int = 260 * 1024 * 1024

    def __post_init__(self) -> None:
        if (
            self.max_source_files < 1
            or self.max_source_files > 500
            or self.max_manifest_bytes < 1
            or self.max_manifest_bytes > 16_777_216
            or self.max_source_bytes < 1
            or self.max_source_bytes > 1_000_000_000
            or self.max_total_source_bytes < self.max_source_bytes
            or self.max_total_source_bytes > 1_000_000_000
            or self.max_request_bytes < self.max_manifest_bytes + self.max_total_source_bytes
            or self.max_request_bytes > 2_000_000_000
        ):
            raise HarnessSourceAdmissionLimitError(
                "source admission limits are invalid",
            )


@dataclass(frozen=True, slots=True)
class SourceUploadClaim:
    relative_name: str
    expected_sha256: str
    expected_size_bytes: int


@dataclass(frozen=True, slots=True)
class SourceUploadPart:
    filename: str
    stream: BinaryIO


class HarnessSourceAdmissionMemberV1(ContractModelV2):
    schema_version: Literal["eval-harness/source-admission-member/v1"] = (
        "eval-harness/source-admission-member/v1"
    )
    relative_name: str = Field(min_length=1, max_length=255)
    media_type: str = Field(min_length=3, max_length=255)
    raw_sha256: Sha256
    size_bytes: int = Field(ge=1, le=1_000_000_000)
    record_count: int = Field(ge=1, le=1_000_000)
    outer_fields: tuple[Identifier, ...] = Field(min_length=1, max_length=128)
    source_ref: ObjectRef
    content_ref: ObjectRef
    artifact_envelope_ref: ObjectRef

    @model_validator(mode="after")
    def validate_member(self) -> Self:
        if not _is_safe_upload_name(self.relative_name):
            raise ValueError("source admission member name is unsafe")
        if (
            self.media_type != "application/x-ndjson"
            or not self.relative_name.endswith(".jsonl")
            or self.source_ref.object_type != "trace-source"
            or self.source_ref.object_version != "v2"
            or self.source_ref.object_sha256 != self.raw_sha256
            or self.content_ref.object_type != "harness-source-content"
            or self.content_ref.object_version != "private-v1"
            or self.content_ref.object_sha256 != self.raw_sha256
            or self.artifact_envelope_ref.object_type != "artifact-envelope"
            or self.artifact_envelope_ref.object_version != "v1"
        ):
            raise ValueError("source admission member references are invalid")
        require_sorted_unique(self.outer_fields, "outer_fields")
        return self


class HarnessSourceAdmissionV1(HarnessObjectV1):
    schema_version: Literal["eval-harness/source-admission/v1"] = "eval-harness/source-admission/v1"
    OBJECT_TYPE: ClassVar[str] = "harness-source-admission"

    session_ref: ObjectRef
    session_id: Identifier
    session_version: int = Field(ge=1, le=1_000_000_000)
    manifest_sha256: Sha256
    manifest_size_bytes: int = Field(ge=1, le=16_777_216)
    manifest_ref: ObjectRef
    manifest_content_ref: ObjectRef
    members: tuple[HarnessSourceAdmissionMemberV1, ...] = Field(
        min_length=1,
        max_length=500,
    )
    artifact_envelope_refs: tuple[ObjectRef, ...] = Field(
        min_length=1,
        max_length=500,
    )
    total_source_bytes: int = Field(ge=1, le=1_000_000_000)

    @model_validator(mode="after")
    def validate_admission(self) -> Self:
        require_ref(self.session_ref, "harness-session", "session_ref")
        if (
            self.manifest_ref.object_type != "source-admission-manifest"
            or self.manifest_ref.object_version != "v1"
            or self.manifest_ref.object_sha256 != self.manifest_sha256
            or self.manifest_ref.object_id != (f"source-admission-manifest://sha256/{self.manifest_sha256}")
            or self.manifest_content_ref.object_type != "harness-source-content"
            or self.manifest_content_ref.object_version != "private-v1"
            or self.manifest_content_ref.object_sha256 != self.manifest_sha256
            or self.manifest_content_ref.object_id
            != f"harness-source-content://sha256/{self.manifest_sha256}"
        ):
            raise ValueError("source admission manifest references are invalid")
        names = tuple(item.relative_name for item in self.members)
        if (
            names != tuple(sorted(names))
            or len(names) != len(set(names))
            or len({name.casefold() for name in names}) != len(names)
            or len({item.raw_sha256 for item in self.members}) != len(self.members)
            or sum(item.size_bytes for item in self.members) != self.total_source_bytes
        ):
            raise ValueError("source admission members are inconsistent")
        expected_refs = sorted_refs(item.artifact_envelope_ref for item in self.members)
        require_sorted_unique_refs(
            self.artifact_envelope_refs,
            "artifact_envelope_refs",
        )
        if self.artifact_envelope_refs != expected_refs:
            raise ValueError("source admission artifact refs are incomplete")
        return self


@dataclass(frozen=True, slots=True)
class HarnessSourceAdmissionWrite:
    admission: HarnessSourceAdmissionV1
    artifact_envelopes: tuple[ArtifactEnvelopeV1, ...]
    replayed: bool


@dataclass(frozen=True, slots=True)
class HarnessSourceExecutionView:
    admission_ref: ObjectRef
    raw_root: Path
    manifest_path: Path
    trace_paths: tuple[Path, ...]


class _ManifestRow(ContractModelV2):
    instance_id: Identifier
    sid: Identifier
    p_date: str
    business: str
    category: str
    pool_id: str
    pool_category: str

    @property
    def relative_name(self) -> str:
        return f"{self.instance_id}_{self.sid}.jsonl"


@dataclass(frozen=True, slots=True)
class _StagedFile:
    relative_name: str
    path: Path
    sha256: str
    size_bytes: int


class HarnessSourceAdmissionStore:
    def __init__(
        self,
        root: Path,
        *,
        limits: HarnessSourceAdmissionLimits | None = None,
        fault_injector: Callable[[str], None] | None = None,
    ) -> None:
        candidate = root.expanduser().absolute()
        if candidate.is_symlink() or (candidate.exists() and not candidate.is_dir()):
            raise HarnessSourceAdmissionIntegrityError(
                "source admission root must be a non-symlink directory",
            )
        candidate.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.root = candidate.resolve()
        self.path = self.root / "authority.sqlite3"
        self.staging = self.root / ".staging"
        self.cas = self.root / "cas"
        self.execution = self.root / "execution"
        self.staging.mkdir(mode=0o700, exist_ok=True)
        self.cas.mkdir(mode=0o700, exist_ok=True)
        self.execution.mkdir(mode=0o700, exist_ok=True)
        self.limits = limits or HarnessSourceAdmissionLimits()
        self._fault = fault_injector or (lambda _: None)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialize(self) -> None:
        connection = self._connect()
        try:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS harness_source_admissions (
                    admission_object_id TEXT PRIMARY KEY,
                    admission_sha256 TEXT NOT NULL,
                    session_id TEXT NOT NULL UNIQUE,
                    session_object_id TEXT NOT NULL,
                    session_version INTEGER NOT NULL,
                    manifest_sha256 TEXT NOT NULL,
                    source_count INTEGER NOT NULL,
                    total_source_bytes INTEGER NOT NULL,
                    admission_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS harness_source_admission_members (
                    admission_object_id TEXT NOT NULL
                        REFERENCES harness_source_admissions(admission_object_id),
                    relative_name TEXT NOT NULL,
                    raw_sha256 TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    source_object_id TEXT NOT NULL,
                    envelope_object_id TEXT NOT NULL,
                    member_json TEXT NOT NULL,
                    envelope_json TEXT NOT NULL,
                    PRIMARY KEY(admission_object_id, relative_name),
                    UNIQUE(admission_object_id, raw_sha256),
                    UNIQUE(admission_object_id, source_object_id),
                    UNIQUE(admission_object_id, envelope_object_id)
                );

                CREATE TABLE IF NOT EXISTS harness_source_admission_idempotency (
                    scope TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    request_sha256 TEXT NOT NULL,
                    response_id TEXT NOT NULL,
                    PRIMARY KEY(scope, idempotency_key)
                );
                """
            )
        finally:
            connection.close()

    def admit(
        self,
        *,
        session_ref: ObjectRef,
        session_id: str,
        session_version: int,
        manifest_claim: SourceUploadClaim,
        trace_claims: tuple[SourceUploadClaim, ...],
        manifest_upload: SourceUploadPart,
        trace_uploads: tuple[SourceUploadPart, ...],
        trace_schema_ref: ObjectRef,
        producer_capability_ref: ObjectRef,
        audit: ContractAudit,
        idempotency_key: str,
    ) -> HarnessSourceAdmissionWrite:
        self._validate_inputs(
            manifest_claim=manifest_claim,
            trace_claims=trace_claims,
            manifest_upload=manifest_upload,
            trace_uploads=trace_uploads,
        )
        stage = self.staging / uuid4().hex
        stage.mkdir(mode=0o700)
        try:
            manifest = self._stage_file(
                stage=stage,
                claim=manifest_claim,
                upload=manifest_upload,
                byte_limit=self.limits.max_manifest_bytes,
            )
            staged_traces: list[_StagedFile] = []
            total_source_bytes = 0
            for claim, upload in zip(
                trace_claims,
                sorted(
                    trace_uploads,
                    key=lambda item: item.filename,
                ),
                strict=True,
            ):
                remaining = self.limits.max_total_source_bytes - total_source_bytes
                if remaining < 1:
                    raise HarnessSourceAdmissionLimitError(
                        "source upload exceeds the aggregate byte limit",
                    )
                trace = self._stage_file(
                    stage=stage,
                    claim=claim,
                    upload=upload,
                    byte_limit=min(
                        self.limits.max_source_bytes,
                        remaining,
                    ),
                )
                staged_traces.append(trace)
                total_source_bytes += trace.size_bytes
            traces = tuple(staged_traces)
            self._fault("after_source_admission_stage")
            rows = self._parse_manifest(manifest.path)
            self._validate_inventory(rows, traces)
            members, envelopes = self._compile_members(
                rows=rows,
                traces=traces,
                manifest=manifest,
                trace_schema_ref=trace_schema_ref,
                producer_capability_ref=producer_capability_ref,
                audit=audit,
            )
            manifest_ref = _content_ref(
                object_type="source-admission-manifest",
                object_version="v1",
                sha256=manifest.sha256,
            )
            admission = HarnessSourceAdmissionV1.create(
                audit=audit,
                session_ref=session_ref,
                session_id=session_id,
                session_version=session_version,
                manifest_sha256=manifest.sha256,
                manifest_size_bytes=manifest.size_bytes,
                manifest_ref=manifest_ref,
                manifest_content_ref=_content_ref(
                    object_type="harness-source-content",
                    object_version="private-v1",
                    sha256=manifest.sha256,
                ),
                members=members,
                artifact_envelope_refs=sorted_refs(envelope.to_ref() for envelope in envelopes),
                total_source_bytes=total_source_bytes,
            )
            request_sha256 = _request_sha256(
                session_ref=session_ref,
                session_version=session_version,
                manifest_claim=manifest_claim,
                trace_claims=trace_claims,
                created_by=audit.created_by,
                trace_schema_ref=trace_schema_ref,
                producer_capability_ref=producer_capability_ref,
            )
            self._publish(manifest, kind="manifest")
            for trace in traces:
                self._publish(trace, kind="trace")
            self._fault("after_source_admission_cas")
            return self._commit(
                admission=admission,
                envelopes=envelopes,
                request_sha256=request_sha256,
                idempotency_key=idempotency_key,
            )
        finally:
            shutil.rmtree(stage, ignore_errors=True)

    def get_for_session(
        self,
        session_id: str,
    ) -> HarnessSourceAdmissionWrite:
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            row = connection.execute(
                """
                SELECT * FROM harness_source_admissions
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
            if row is None:
                raise HarnessSourceAdmissionNotFoundError(
                    "source admission was not found",
                )
            result = self._load_write(connection, row, replayed=True)
            connection.rollback()
            return result
        finally:
            connection.close()

    def get(
        self,
        reference: ObjectRef,
    ) -> HarnessSourceAdmissionWrite:
        if reference.object_type != "harness-source-admission" or reference.object_version != "v1":
            raise HarnessSourceAdmissionIntegrityError(
                "source admission ref has the wrong type or version",
            )
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            row = connection.execute(
                """
                SELECT * FROM harness_source_admissions
                WHERE admission_object_id = ?
                """,
                (reference.object_id,),
            ).fetchone()
            if row is None:
                raise HarnessSourceAdmissionNotFoundError(
                    "source admission was not found",
                )
            result = self._load_write(connection, row, replayed=True)
            if result.admission.to_ref() != reference:
                raise HarnessSourceAdmissionIntegrityError(
                    "source admission differs from its reference",
                )
            connection.rollback()
            return result
        finally:
            connection.close()

    def source_path(self, source_ref: ObjectRef) -> Path:
        if source_ref.object_type != "trace-source" or source_ref.object_version != "v2":
            raise HarnessSourceAdmissionIntegrityError(
                "trace source ref has the wrong type or version",
            )
        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT raw_sha256, size_bytes, source_object_id
                FROM harness_source_admission_members
                WHERE source_object_id = ?
                """,
                (source_ref.object_id,),
            ).fetchall()
            if not rows:
                raise HarnessSourceAdmissionNotFoundError(
                    "trace source was not found",
                )
            if any(
                str(row["raw_sha256"]) != source_ref.object_sha256
                or str(row["source_object_id"]) != source_ref.object_id
                or int(row["size_bytes"]) != int(rows[0]["size_bytes"])
                for row in rows
            ):
                raise HarnessSourceAdmissionIntegrityError(
                    "trace source columns drifted",
                )
            path = self._cas_path(source_ref.object_sha256, kind="trace")
            self._verify_cas(
                path,
                expected_sha256=source_ref.object_sha256,
                expected_size=int(rows[0]["size_bytes"]),
            )
            return path
        finally:
            connection.close()

    def materialize_for_execution(
        self,
        session_id: str,
    ) -> HarnessSourceExecutionView:
        write = self.get_for_session(session_id)
        admission = write.admission
        target = self.execution / admission.object_sha256
        if target.exists():
            return self._execution_view(target, admission)
        stage = self.staging / f"execution-{uuid4().hex}"
        stage.mkdir(mode=0o700)
        try:
            manifest_source = self._cas_path(
                admission.manifest_sha256,
                kind="manifest",
            )
            self._verify_cas(
                manifest_source,
                expected_sha256=admission.manifest_sha256,
                expected_size=admission.manifest_size_bytes,
            )
            os.link(manifest_source, stage / "manifest.csv")
            for member in admission.members:
                source = self.source_path(member.source_ref)
                os.link(source, stage / member.relative_name)
            _fsync_directory(stage)
            try:
                os.rename(stage, target)
                _fsync_directory(self.execution)
            except OSError as exc:
                if not target.is_dir() or target.is_symlink():
                    raise HarnessSourceAdmissionIntegrityError(
                        "source execution view publication failed",
                    ) from exc
            return self._execution_view(target, admission)
        except HarnessSourceAdmissionError:
            raise
        except OSError as exc:
            raise HarnessSourceAdmissionIntegrityError(
                "source execution view materialization failed",
            ) from exc
        finally:
            if stage.exists():
                shutil.rmtree(stage, ignore_errors=True)

    def _execution_view(
        self,
        root: Path,
        admission: HarnessSourceAdmissionV1,
    ) -> HarnessSourceExecutionView:
        if root.is_symlink() or not root.is_dir():
            raise HarnessSourceAdmissionIntegrityError(
                "source execution view is missing or unsafe",
            )
        expected_names = {
            "manifest.csv",
            *(member.relative_name for member in admission.members),
        }
        observed = tuple(root.iterdir())
        if {path.name for path in observed} != expected_names or any(
            path.is_symlink() or not path.is_file() for path in observed
        ):
            raise HarnessSourceAdmissionIntegrityError(
                "source execution view inventory drifted",
            )
        manifest_path = root / "manifest.csv"
        self._verify_cas(
            manifest_path,
            expected_sha256=admission.manifest_sha256,
            expected_size=admission.manifest_size_bytes,
        )
        trace_paths: list[Path] = []
        for member in admission.members:
            path = root / member.relative_name
            self._verify_cas(
                path,
                expected_sha256=member.raw_sha256,
                expected_size=member.size_bytes,
            )
            trace_paths.append(path)
        return HarnessSourceExecutionView(
            admission_ref=admission.to_ref(),
            raw_root=root,
            manifest_path=manifest_path,
            trace_paths=tuple(trace_paths),
        )

    def recover_staging(self) -> int:
        removed = 0
        for path in tuple(self.staging.iterdir()):
            if path.is_symlink():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            removed += 1
        return removed

    def _commit(
        self,
        *,
        admission: HarnessSourceAdmissionV1,
        envelopes: tuple[ArtifactEnvelopeV1, ...],
        request_sha256: str,
        idempotency_key: str,
    ) -> HarnessSourceAdmissionWrite:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            scope = f"source-admission:{admission.session_id}"
            replay = connection.execute(
                """
                SELECT request_sha256, response_id
                FROM harness_source_admission_idempotency
                WHERE scope = ? AND idempotency_key = ?
                """,
                (scope, idempotency_key),
            ).fetchone()
            if replay is not None:
                if str(replay["request_sha256"]) != request_sha256:
                    raise HarnessSourceAdmissionConflictError(
                        "source admission idempotency key changed input",
                    )
                row = connection.execute(
                    """
                    SELECT * FROM harness_source_admissions
                    WHERE admission_object_id = ?
                    """,
                    (str(replay["response_id"]),),
                ).fetchone()
                if row is None:
                    raise HarnessSourceAdmissionIntegrityError(
                        "source admission replay target is missing",
                    )
                result = self._load_write(
                    connection,
                    row,
                    replayed=True,
                )
                connection.rollback()
                return result
            existing = connection.execute(
                """
                SELECT * FROM harness_source_admissions
                WHERE session_id = ?
                """,
                (admission.session_id,),
            ).fetchone()
            if existing is not None:
                stored = self._load_write(
                    connection,
                    existing,
                    replayed=True,
                )
                if (
                    stored.admission.to_ref() != admission.to_ref()
                    or stored.admission.audit.created_by != admission.audit.created_by
                ):
                    raise HarnessSourceAdmissionConflictError(
                        "session already has another source admission",
                    )
                self._insert_idempotency(
                    connection,
                    scope=scope,
                    idempotency_key=idempotency_key,
                    request_sha256=request_sha256,
                    response_id=stored.admission.object_id,
                )
                connection.commit()
                return stored
            connection.execute(
                """
                INSERT INTO harness_source_admissions (
                    admission_object_id, admission_sha256, session_id,
                    session_object_id, session_version, manifest_sha256,
                    source_count, total_source_bytes, admission_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    admission.object_id,
                    admission.object_sha256,
                    admission.session_id,
                    admission.session_ref.object_id,
                    admission.session_version,
                    admission.manifest_sha256,
                    len(admission.members),
                    admission.total_source_bytes,
                    admission.canonical_json().decode(),
                ),
            )
            envelopes_by_ref = {envelope.to_ref(): envelope for envelope in envelopes}
            for member in admission.members:
                envelope = envelopes_by_ref[member.artifact_envelope_ref]
                connection.execute(
                    """
                    INSERT INTO harness_source_admission_members (
                        admission_object_id, relative_name, raw_sha256,
                        size_bytes, source_object_id, envelope_object_id,
                        member_json, envelope_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        admission.object_id,
                        member.relative_name,
                        member.raw_sha256,
                        member.size_bytes,
                        member.source_ref.object_id,
                        envelope.object_id,
                        member.canonical_json().decode(),
                        envelope.canonical_json().decode(),
                    ),
                )
            self._insert_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_id=admission.object_id,
            )
            self._fault("before_source_admission_commit")
            connection.commit()
            return HarnessSourceAdmissionWrite(
                admission=admission,
                artifact_envelopes=envelopes,
                replayed=False,
            )
        except sqlite3.IntegrityError as exc:
            connection.rollback()
            raise HarnessSourceAdmissionConflictError(
                "source admission authority already exists",
            ) from exc
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _load_write(
        self,
        connection: sqlite3.Connection,
        row: sqlite3.Row,
        *,
        replayed: bool,
    ) -> HarnessSourceAdmissionWrite:
        admission = _parse_model(
            HarnessSourceAdmissionV1,
            str(row["admission_json"]),
            "source admission",
        )
        if (
            admission.object_id != str(row["admission_object_id"])
            or admission.object_sha256 != str(row["admission_sha256"])
            or admission.session_id != str(row["session_id"])
            or admission.session_ref.object_id != str(row["session_object_id"])
            or admission.session_version != int(row["session_version"])
            or admission.manifest_sha256 != str(row["manifest_sha256"])
            or len(admission.members) != int(row["source_count"])
            or admission.total_source_bytes != int(row["total_source_bytes"])
        ):
            raise HarnessSourceAdmissionIntegrityError(
                "source admission columns drifted",
            )
        member_rows = connection.execute(
            """
            SELECT * FROM harness_source_admission_members
            WHERE admission_object_id = ?
            ORDER BY relative_name
            """,
            (admission.object_id,),
        ).fetchall()
        members = tuple(
            _parse_model(
                HarnessSourceAdmissionMemberV1,
                str(member["member_json"]),
                "source admission member",
            )
            for member in member_rows
        )
        envelopes = tuple(
            _parse_model(
                ArtifactEnvelopeV1,
                str(member["envelope_json"]),
                "source artifact envelope",
            )
            for member in member_rows
        )
        if (
            members != admission.members
            or sorted_refs(envelope.to_ref() for envelope in envelopes) != admission.artifact_envelope_refs
        ):
            raise HarnessSourceAdmissionIntegrityError(
                "source admission member inventory drifted",
            )
        for member, member_row, envelope in zip(
            members,
            member_rows,
            envelopes,
            strict=True,
        ):
            if (
                member.relative_name != str(member_row["relative_name"])
                or member.raw_sha256 != str(member_row["raw_sha256"])
                or member.size_bytes != int(member_row["size_bytes"])
                or member.source_ref.object_id != str(member_row["source_object_id"])
                or envelope.object_id != str(member_row["envelope_object_id"])
                or envelope.to_ref() != member.artifact_envelope_ref
                or envelope.subject_ref != member.source_ref
                or envelope.content_ref != member.content_ref
            ):
                raise HarnessSourceAdmissionIntegrityError(
                    "source admission member columns drifted",
                )
            self._verify_cas(
                self._cas_path(member.raw_sha256, kind="trace"),
                expected_sha256=member.raw_sha256,
                expected_size=member.size_bytes,
            )
        self._verify_cas(
            self._cas_path(admission.manifest_sha256, kind="manifest"),
            expected_sha256=admission.manifest_sha256,
            expected_size=admission.manifest_size_bytes,
        )
        return HarnessSourceAdmissionWrite(
            admission=admission,
            artifact_envelopes=envelopes,
            replayed=replayed,
        )

    def _compile_members(
        self,
        *,
        rows: tuple[_ManifestRow, ...],
        traces: tuple[_StagedFile, ...],
        manifest: _StagedFile,
        trace_schema_ref: ObjectRef,
        producer_capability_ref: ObjectRef,
        audit: ContractAudit,
    ) -> tuple[
        tuple[HarnessSourceAdmissionMemberV1, ...],
        tuple[ArtifactEnvelopeV1, ...],
    ]:
        traces_by_name = {item.relative_name: item for item in traces}
        manifest_ref = _content_ref(
            object_type="source-admission-manifest",
            object_version="v1",
            sha256=manifest.sha256,
        )
        members: list[HarnessSourceAdmissionMemberV1] = []
        envelopes: list[ArtifactEnvelopeV1] = []
        adapter = RawTrajV1Adapter()
        for row in rows:
            trace = traces_by_name[row.relative_name]
            probe = adapter.probe(trace.path)
            if (
                probe.status is not TraceProbeStatus.SUPPORTED
                or probe.raw_sha256 != trace.sha256
                or probe.size_bytes != trace.size_bytes
                or probe.record_count < 1
            ):
                raise HarnessSourceAdmissionSourceError(
                    "trace upload is not a supported raw_traj_v1 source",
                )
            self._validate_trace_metadata(trace.path, row)
            source_ref = _trace_source_ref(row, trace.sha256)
            content_ref = _content_ref(
                object_type="harness-source-content",
                object_version="private-v1",
                sha256=trace.sha256,
            )
            envelope = ArtifactEnvelopeV1.create(
                artifact_id=(f"source-artifact://{source_ref.object_id.rsplit('/', 1)[-1]}"),
                subject_ref=source_ref,
                schema_ref=trace_schema_ref,
                content_ref=content_ref,
                media_type="application/x-ndjson",
                modality=ArtifactModalityV1.TEXT,
                domain_tags=("generic-agent-trace", "raw-traj-v1"),
                semantic_role="trace-source",
                purpose="evaluation-data-production",
                classification="RESTRICTED_TRACE_RAW",
                lineage_refs=(manifest_ref,),
                producer_capability_ref=producer_capability_ref,
                producer_task_ref=None,
                validation_refs=(),
                revision=1,
                predecessor_envelope_ref=None,
                audit=audit,
            )
            members.append(
                HarnessSourceAdmissionMemberV1(
                    relative_name=row.relative_name,
                    media_type="application/x-ndjson",
                    raw_sha256=trace.sha256,
                    size_bytes=trace.size_bytes,
                    record_count=probe.record_count,
                    outer_fields=tuple(sorted(probe.outer_fields)),
                    source_ref=source_ref,
                    content_ref=content_ref,
                    artifact_envelope_ref=envelope.to_ref(),
                )
            )
            envelopes.append(envelope)
        ordered_members = tuple(
            sorted(members, key=lambda item: item.relative_name),
        )
        envelopes_by_ref = {envelope.to_ref(): envelope for envelope in envelopes}
        ordered_envelopes = tuple(
            envelopes_by_ref[member.artifact_envelope_ref] for member in ordered_members
        )
        return ordered_members, ordered_envelopes

    def _validate_inputs(
        self,
        *,
        manifest_claim: SourceUploadClaim,
        trace_claims: tuple[SourceUploadClaim, ...],
        manifest_upload: SourceUploadPart,
        trace_uploads: tuple[SourceUploadPart, ...],
    ) -> None:
        if not trace_claims or len(trace_claims) > self.limits.max_source_files:
            raise HarnessSourceAdmissionLimitError(
                "source upload exceeds the file-count limit",
            )
        if (
            manifest_claim.expected_size_bytes > self.limits.max_manifest_bytes
            or any(claim.expected_size_bytes > self.limits.max_source_bytes for claim in trace_claims)
            or sum(claim.expected_size_bytes for claim in trace_claims) > self.limits.max_total_source_bytes
        ):
            raise HarnessSourceAdmissionLimitError(
                "source upload claims exceed configured byte limits",
            )
        _validate_upload_name(manifest_claim.relative_name)
        _validate_upload_name(manifest_upload.filename)
        if manifest_claim.relative_name != "manifest.csv" or manifest_upload.filename != "manifest.csv":
            raise HarnessSourceAdmissionInventoryError(
                "source upload requires exactly one manifest.csv",
            )
        claim_names = tuple(item.relative_name for item in trace_claims)
        upload_names = tuple(item.filename for item in trace_uploads)
        for name in (*claim_names, *upload_names):
            _validate_upload_name(name)
            if not name.endswith(".jsonl"):
                raise HarnessSourceAdmissionUnsafeNameError(
                    "trace upload name must use the .jsonl extension",
                )
        if (
            claim_names != tuple(sorted(claim_names))
            or len(claim_names) != len(set(claim_names))
            or len({name.casefold() for name in claim_names}) != len(claim_names)
            or len(upload_names) != len(set(upload_names))
            or len({name.casefold() for name in upload_names}) != len(upload_names)
            or set(claim_names) != set(upload_names)
        ):
            raise HarnessSourceAdmissionInventoryError(
                "trace upload claims and files differ",
            )

    def _stage_file(
        self,
        *,
        stage: Path,
        claim: SourceUploadClaim,
        upload: SourceUploadPart,
        byte_limit: int,
    ) -> _StagedFile:
        target = stage / claim.relative_name
        hasher = hashlib.sha256()
        size = 0
        try:
            with target.open("xb") as handle:
                while True:
                    chunk = upload.stream.read(_CHUNK_BYTES)
                    if not chunk:
                        break
                    if not isinstance(chunk, bytes):
                        raise HarnessSourceAdmissionValidationError(
                            "source upload stream must return bytes",
                        )
                    size += len(chunk)
                    if size > byte_limit:
                        raise HarnessSourceAdmissionLimitError(
                            "source upload exceeds its byte limit",
                        )
                    hasher.update(chunk)
                    handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            raise HarnessSourceAdmissionIntegrityError(
                "source upload staging failed",
            ) from exc
        digest = hasher.hexdigest()
        if size != claim.expected_size_bytes or digest != claim.expected_sha256:
            raise HarnessSourceAdmissionHashError(
                "source upload differs from its declared bytes",
            )
        return _StagedFile(
            relative_name=claim.relative_name,
            path=target,
            sha256=digest,
            size_bytes=size,
        )

    def _parse_manifest(
        self,
        path: Path,
    ) -> tuple[_ManifestRow, ...]:
        try:
            payload = path.read_bytes()
            text = payload.decode("utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise HarnessSourceAdmissionManifestError(
                "source manifest is not readable UTF-8",
            ) from exc
        try:
            reader = csv.DictReader(
                io.StringIO(text, newline=""),
                strict=True,
            )
            if tuple(reader.fieldnames or ()) != SOURCE_MANIFEST_COLUMNS or len(
                reader.fieldnames or ()
            ) != len(set(reader.fieldnames or ())):
                raise HarnessSourceAdmissionManifestError(
                    "source manifest header is not exact",
                )
            rows: list[_ManifestRow] = []
            for raw in reader:
                if len(rows) >= self.limits.max_source_files:
                    raise HarnessSourceAdmissionLimitError(
                        "source manifest exceeds the row limit",
                    )
                if (
                    None in raw
                    or set(raw) != set(SOURCE_MANIFEST_COLUMNS)
                    or any(not isinstance(value, str) for value in raw.values())
                ):
                    raise HarnessSourceAdmissionManifestError(
                        "source manifest row is invalid",
                    )
                values = {key: raw[key] for key in SOURCE_MANIFEST_COLUMNS}
                if any(
                    not value
                    or value != value.strip()
                    or len(value) > 256
                    or any(ord(character) < 32 for character in value)
                    for value in values.values()
                ):
                    raise HarnessSourceAdmissionManifestError(
                        "source manifest row value is invalid",
                    )
                try:
                    date.fromisoformat(values["p_date"])
                except ValueError as exc:
                    raise HarnessSourceAdmissionManifestError(
                        "source manifest date is invalid",
                    ) from exc
                try:
                    row = _ManifestRow(
                        instance_id=values["instance_id"],
                        sid=values["sid"],
                        p_date=values["p_date"],
                        business=values["business"],
                        category=values["category"],
                        pool_id=values["pool_id"],
                        pool_category=values["pool_category"],
                    )
                except ValidationError as exc:
                    raise HarnessSourceAdmissionManifestError(
                        "source manifest identity is invalid",
                    ) from exc
                _validate_upload_name(row.relative_name)
                rows.append(row)
        except csv.Error as exc:
            raise HarnessSourceAdmissionManifestError(
                "source manifest CSV is malformed",
            ) from exc
        if not rows:
            raise HarnessSourceAdmissionManifestError(
                "source manifest is empty",
            )
        for identities, label in (
            ((row.instance_id for row in rows), "instance"),
            ((row.sid for row in rows), "SID"),
            ((row.pool_id for row in rows), "pool"),
            ((row.relative_name.casefold() for row in rows), "file"),
        ):
            observed = tuple(identities)
            if len(observed) != len(set(observed)):
                raise HarnessSourceAdmissionManifestError(
                    f"source manifest contains duplicate {label}",
                )
        return tuple(sorted(rows, key=lambda row: row.relative_name))

    @staticmethod
    def _validate_inventory(
        rows: tuple[_ManifestRow, ...],
        traces: tuple[_StagedFile, ...],
    ) -> None:
        expected = {row.relative_name for row in rows}
        actual = {trace.relative_name for trace in traces}
        if expected != actual or len(expected) != len(traces):
            raise HarnessSourceAdmissionInventoryError(
                "source manifest and uploaded trace inventory differ",
            )
        hashes = tuple(trace.sha256 for trace in traces)
        if len(hashes) != len(set(hashes)):
            raise HarnessSourceAdmissionInventoryError(
                "source upload contains aliased trace bytes",
            )

    @staticmethod
    def _validate_trace_metadata(
        path: Path,
        row: _ManifestRow,
    ) -> None:
        try:
            with path.open("rb") as handle:
                for raw_line in handle:
                    if not raw_line.strip():
                        continue
                    value = json.loads(
                        raw_line,
                        parse_constant=_reject_json_constant,
                    )
                    if (
                        not isinstance(value, dict)
                        or value.get("sid") != row.sid
                        or value.get("p_date") != row.p_date
                        or value.get("business") != row.business
                    ):
                        raise HarnessSourceAdmissionMetadataError(
                            "trace metadata differs from the source manifest",
                        )
        except HarnessSourceAdmissionMetadataError:
            raise
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise HarnessSourceAdmissionSourceError(
                "trace upload cannot be validated",
            ) from exc

    def _publish(
        self,
        staged: _StagedFile,
        *,
        kind: Literal["manifest", "trace"],
    ) -> None:
        target = self._cas_path(staged.sha256, kind=kind)
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if target.exists():
            self._verify_cas(
                target,
                expected_sha256=staged.sha256,
                expected_size=staged.size_bytes,
            )
            return
        try:
            os.link(staged.path, target)
            _fsync_directory(target.parent)
        except FileExistsError:
            self._verify_cas(
                target,
                expected_sha256=staged.sha256,
                expected_size=staged.size_bytes,
            )
        except OSError as exc:
            raise HarnessSourceAdmissionIntegrityError(
                "source CAS publication failed",
            ) from exc

    def _cas_path(
        self,
        sha256: str,
        *,
        kind: Literal["manifest", "trace"],
    ) -> Path:
        suffix = ".csv" if kind == "manifest" else ".jsonl"
        return self.cas / kind / "sha256" / sha256[:2] / f"{sha256}{suffix}"

    @staticmethod
    def _verify_cas(
        path: Path,
        *,
        expected_sha256: str,
        expected_size: int | None,
    ) -> None:
        if not path.is_file() or path.is_symlink():
            raise HarnessSourceAdmissionIntegrityError(
                "source CAS object is missing or unsafe",
            )
        hasher = hashlib.sha256()
        size = 0
        try:
            with path.open("rb") as handle:
                while chunk := handle.read(_CHUNK_BYTES):
                    size += len(chunk)
                    hasher.update(chunk)
        except OSError as exc:
            raise HarnessSourceAdmissionIntegrityError(
                "source CAS object is unreadable",
            ) from exc
        if hasher.hexdigest() != expected_sha256 or (expected_size is not None and size != expected_size):
            raise HarnessSourceAdmissionIntegrityError(
                "source CAS object failed integrity validation",
            )

    @staticmethod
    def _insert_idempotency(
        connection: sqlite3.Connection,
        *,
        scope: str,
        idempotency_key: str,
        request_sha256: str,
        response_id: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO harness_source_admission_idempotency (
                scope, idempotency_key, request_sha256, response_id
            ) VALUES (?, ?, ?, ?)
            """,
            (scope, idempotency_key, request_sha256, response_id),
        )


def _validate_upload_name(value: str) -> None:
    if not _is_safe_upload_name(value):
        raise HarnessSourceAdmissionUnsafeNameError(
            "source upload name is unsafe",
        )


def _is_safe_upload_name(value: str) -> bool:
    return not (
        not value
        or len(value) > 255
        or value[0] not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
        or any(
            character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-"
            for character in value
        )
        or "/" in value
        or "\\" in value
        or ".." in value
        or value in {".", ".."}
    )


def _trace_source_ref(row: _ManifestRow, raw_sha256: str) -> ObjectRef:
    identity = json.dumps(
        {
            "instance_id": row.instance_id,
            "sid": row.sid,
            "raw_sha256": raw_sha256,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    digest = hashlib.sha256(identity).hexdigest()
    return ObjectRef(
        object_type="trace-source",
        object_id=f"source-trace://agent-upload/{digest}",
        object_version="v2",
        object_sha256=raw_sha256,
    )


def _content_ref(
    *,
    object_type: str,
    object_version: str,
    sha256: str,
) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://sha256/{sha256}",
        object_version=object_version,
        object_sha256=sha256,
    )


def _request_sha256(
    *,
    session_ref: ObjectRef,
    session_version: int,
    manifest_claim: SourceUploadClaim,
    trace_claims: tuple[SourceUploadClaim, ...],
    created_by: str,
    trace_schema_ref: ObjectRef,
    producer_capability_ref: ObjectRef,
) -> str:
    payload = {
        "session_ref": session_ref.model_dump(mode="json"),
        "session_version": session_version,
        "manifest_claim": _claim_value(manifest_claim),
        "trace_claims": [_claim_value(claim) for claim in trace_claims],
        "created_by": created_by,
        "trace_schema_ref": trace_schema_ref.model_dump(mode="json"),
        "producer_capability_ref": producer_capability_ref.model_dump(
            mode="json",
        ),
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _claim_value(value: SourceUploadClaim) -> dict[str, object]:
    return {
        "relative_name": value.relative_name,
        "expected_sha256": value.expected_sha256,
        "expected_size_bytes": value.expected_size_bytes,
    }


def _parse_model[ModelT: ContractModelV2](
    model_type: type[ModelT],
    payload: str,
    label: str,
) -> ModelT:
    try:
        value = model_type.model_validate_json(payload)
    except (ValidationError, ValueError) as exc:
        raise HarnessSourceAdmissionIntegrityError(
            f"{label} is invalid",
        ) from exc
    if value.canonical_json().decode() != payload:
        raise HarnessSourceAdmissionIntegrityError(
            f"{label} is not canonical",
        )
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "SOURCE_MANIFEST_COLUMNS",
    "HarnessSourceAdmissionConflictError",
    "HarnessSourceAdmissionError",
    "HarnessSourceAdmissionHashError",
    "HarnessSourceAdmissionIntegrityError",
    "HarnessSourceAdmissionInventoryError",
    "HarnessSourceAdmissionLimitError",
    "HarnessSourceAdmissionLimits",
    "HarnessSourceAdmissionManifestError",
    "HarnessSourceAdmissionMemberV1",
    "HarnessSourceAdmissionMetadataError",
    "HarnessSourceAdmissionMultipartError",
    "HarnessSourceAdmissionNotFoundError",
    "HarnessSourceAdmissionSourceError",
    "HarnessSourceAdmissionStore",
    "HarnessSourceAdmissionUnsafeNameError",
    "HarnessSourceAdmissionV1",
    "HarnessSourceAdmissionValidationError",
    "HarnessSourceAdmissionWrite",
    "HarnessSourceExecutionView",
    "SourceUploadClaim",
    "SourceUploadPart",
]
