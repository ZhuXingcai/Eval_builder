from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal, Protocol, Self

from pydantic import Field, ValidationError, model_validator

from eval_factory.attachment_planning.dependency_models import (
    FakePromptOnlyDependencyFixture,
)
from eval_factory.contracts.canary_execution_v2 import (
    R6CanaryObjectCodecV2,
    R6CanarySeedObjectV2,
)
from eval_factory.contracts.core import ContractModel, ObjectRef
from eval_factory.contracts.core_v2 import (
    ContractModelV2,
    canonical_value_v2,
)
from eval_factory.contracts.labeling_v2 import LabelSpecV2
from eval_factory.contracts.review_v2 import SemanticReviewPolicyV2
from eval_factory.labeling.semantic import (
    FakeSemanticResidualFixture,
)
from eval_factory.labeling.structured import StructuredFactSet
from eval_factory.storage import (
    ContentAddressedByteStore,
    ContentAddressedByteStoreConflictError,
    ContentAddressedByteStoreCorruptionError,
)
from eval_factory.task_authoring.draft_models import (
    FakeTaskDraftAuthoringFixture,
)
from eval_factory.task_authoring.models import (
    FakeTaskEpisodeGroupingFixture,
)
from eval_factory.task_authoring.prompt_safety_models import (
    FakeTaskPromptSafetyFixture,
)
from eval_factory.task_authoring.rubric_models import (
    FakeRubricGenerationFixture,
)

STAGE_OBJECT_ENVELOPE_VERSION: Literal["eval-factory/stage-object-envelope/v2"] = (
    "eval-factory/stage-object-envelope/v2"
)


class StageObjectStoreError(RuntimeError):
    pass


class StageObjectCodecError(StageObjectStoreError):
    pass


class StageObjectConflictError(StageObjectStoreError):
    pass


class StageObjectCorruptionError(StageObjectStoreError):
    pass


class StageObjectProjectionError(StageObjectStoreError):
    pass


class StageObjectNotFoundError(StageObjectStoreError):
    pass


class StageObjectInjectedCrash(StageObjectStoreError):
    pass


class StageObjectStoreFaultPoint(StrEnum):
    AFTER_CAS_WRITE = "after_cas_write"
    AFTER_ENVELOPE_APPEND = "after_envelope_append"
    AFTER_PROJECTION_REBUILD = "after_projection_rebuild"


class StageObjectStoreFaultInjector(Protocol):
    def maybe_raise(
        self,
        point: StageObjectStoreFaultPoint,
        *,
        object_ref: ObjectRef,
    ) -> None: ...


@dataclass(frozen=True)
class StaticStageObjectStoreFaultInjector:
    crash_points: frozenset[StageObjectStoreFaultPoint] = frozenset()

    def maybe_raise(
        self,
        point: StageObjectStoreFaultPoint,
        *,
        object_ref: ObjectRef,
    ) -> None:
        del object_ref
        if point in self.crash_points:
            raise StageObjectInjectedCrash(f"injected stage object crash at {point.value}")


class StageObjectEnvelopeV2(ContractModelV2):
    schema_version: Literal["eval-factory/stage-object-envelope/v2"] = STAGE_OBJECT_ENVELOPE_VERSION
    object_ref: ObjectRef
    codec: R6CanaryObjectCodecV2
    content_blob_ref: ObjectRef
    canonical_size_bytes: int = Field(ge=2)

    @model_validator(mode="after")
    def validate_envelope(self) -> Self:
        if (
            self.content_blob_ref.object_type != "stage-object-content"
            or self.content_blob_ref.object_version != "v1"
            or self.content_blob_ref.object_id
            != (f"stage-object-content://sha256/{self.content_blob_ref.object_sha256}")
        ):
            raise ValueError("stage object content ref is invalid")
        return self

    @property
    def object_key(self) -> tuple[str, str, str]:
        return (
            self.object_ref.object_type,
            self.object_ref.object_id,
            self.object_ref.object_version,
        )

    def jsonl_bytes(self) -> bytes:
        return self.canonical_json() + b"\n"


@dataclass(frozen=True)
class StageObjectCodecDefinition:
    codec: R6CanaryObjectCodecV2
    model_type: type[ContractModel]
    object_type: str
    fixed_object_version: str | None = None
    object_version_field: str | None = None
    object_id_field: str | None = None
    object_sha256_field: str | None = None
    exclude_audit_from_reference_hash: bool = False

    def reference(self, value: ContractModel) -> ObjectRef:
        if type(value) is not self.model_type:
            raise StageObjectCodecError(f"{self.codec.value} received the wrong model type")
        digest = (
            _string_field(value, self.object_sha256_field)
            if self.object_sha256_field is not None
            else _reference_sha256(
                value,
                exclude_audit=(self.exclude_audit_from_reference_hash),
            )
        )
        object_id = (
            _string_field(value, self.object_id_field)
            if self.object_id_field is not None
            else f"{self.object_type}://sha256/{digest}"
        )
        object_version = (
            self.fixed_object_version
            if self.fixed_object_version is not None
            else _string_field(value, self.object_version_field)
        )
        return ObjectRef(
            object_type=self.object_type,
            object_id=object_id,
            object_version=object_version,
            object_sha256=digest,
        )


class StageObjectCodecRegistry:
    def __init__(
        self,
        definitions: tuple[StageObjectCodecDefinition, ...],
    ) -> None:
        by_codec: dict[
            R6CanaryObjectCodecV2,
            StageObjectCodecDefinition,
        ] = {}
        for definition in definitions:
            if definition.codec in by_codec:
                raise StageObjectCodecError(f"duplicate stage object codec {definition.codec.value}")
            by_codec[definition.codec] = definition
        self._definitions = by_codec

    @property
    def definitions(
        self,
    ) -> tuple[StageObjectCodecDefinition, ...]:
        return tuple(self._definitions.values())

    def extend(
        self,
        definitions: tuple[StageObjectCodecDefinition, ...],
    ) -> StageObjectCodecRegistry:
        return StageObjectCodecRegistry((*self.definitions, *definitions))

    def definition(
        self,
        codec: R6CanaryObjectCodecV2,
    ) -> StageObjectCodecDefinition:
        definition = self._definitions.get(codec)
        if definition is None:
            raise StageObjectCodecError(f"unsupported stage object codec {codec.value}")
        return definition

    def reference(
        self,
        codec: R6CanaryObjectCodecV2,
        value: ContractModel,
    ) -> ObjectRef:
        return self.definition(codec).reference(value)

    def encode(
        self,
        codec: R6CanaryObjectCodecV2,
        value: ContractModel,
    ) -> bytes:
        self.definition(codec).reference(value)
        return value.canonical_json()

    def decode(
        self,
        codec: R6CanaryObjectCodecV2,
        payload: bytes,
    ) -> ContractModel:
        definition = self.definition(codec)
        try:
            value = definition.model_type.model_validate_json(payload)
        except ValidationError as exc:
            raise StageObjectCodecError(f"{codec.value} payload is invalid") from exc
        if value.canonical_json() != payload:
            raise StageObjectCodecError(f"{codec.value} payload is not canonical")
        return value


def canary_stage_object_codec_registry() -> StageObjectCodecRegistry:
    return StageObjectCodecRegistry(
        (
            StageObjectCodecDefinition(
                codec=R6CanaryObjectCodecV2.STRUCTURED_FACT_SET,
                model_type=StructuredFactSet,
                object_type="structured-fact-set",
                fixed_object_version="r3-02",
                exclude_audit_from_reference_hash=True,
            ),
            StageObjectCodecDefinition(
                codec=R6CanaryObjectCodecV2.LABEL_SPEC,
                model_type=LabelSpecV2,
                object_type="label-spec",
                object_version_field="label_version",
                object_id_field="label_spec_id",
                object_sha256_field="label_spec_sha256",
            ),
            StageObjectCodecDefinition(
                codec=(R6CanaryObjectCodecV2.SEMANTIC_RESIDUAL_FIXTURE),
                model_type=FakeSemanticResidualFixture,
                object_type="fake-semantic-residual-fixture",
                fixed_object_version="r3-03",
                object_id_field="fixture_id",
            ),
            StageObjectCodecDefinition(
                codec=R6CanaryObjectCodecV2.TASK_EPISODE_FIXTURE,
                model_type=FakeTaskEpisodeGroupingFixture,
                object_type=("fake-task-episode-grouping-fixture"),
                fixed_object_version="r4-01",
                object_id_field="fixture_id",
            ),
            StageObjectCodecDefinition(
                codec=R6CanaryObjectCodecV2.TASK_DRAFT_FIXTURE,
                model_type=FakeTaskDraftAuthoringFixture,
                object_type="fake-task-draft-authoring-fixture",
                fixed_object_version="r4-03",
                object_id_field="fixture_id",
            ),
            StageObjectCodecDefinition(
                codec=(R6CanaryObjectCodecV2.TASK_PROMPT_SAFETY_FIXTURE),
                model_type=FakeTaskPromptSafetyFixture,
                object_type="fake-task-prompt-safety-fixture",
                fixed_object_version="r4-04",
                object_id_field="fixture_id",
            ),
            StageObjectCodecDefinition(
                codec=(R6CanaryObjectCodecV2.RUBRIC_GENERATION_FIXTURE),
                model_type=FakeRubricGenerationFixture,
                object_type="fake-rubric-generation-fixture",
                fixed_object_version="r4-05",
                object_id_field="fixture_id",
            ),
            StageObjectCodecDefinition(
                codec=(R6CanaryObjectCodecV2.PROMPT_ONLY_DEPENDENCY_FIXTURE),
                model_type=FakePromptOnlyDependencyFixture,
                object_type=("fake-prompt-only-dependency-fixture"),
                fixed_object_version="r5-03",
                object_id_field="fixture_id",
            ),
            StageObjectCodecDefinition(
                codec=R6CanaryObjectCodecV2.SEMANTIC_REVIEW_POLICY,
                model_type=SemanticReviewPolicyV2,
                object_type="semantic-review-policy",
                fixed_object_version="v2",
                object_id_field="semantic_review_policy_id",
                object_sha256_field="policy_sha256",
            ),
        )
    )


@dataclass(frozen=True)
class StageObjectWriteResult:
    object_ref: ObjectRef
    envelope: StageObjectEnvelopeV2
    content_blob_written: bool
    envelope_appended: bool


@dataclass(frozen=True)
class StageObjectProjectionRebuild:
    object_count: int
    projection_sha256: str


class StageObjectStore:
    def __init__(
        self,
        root: Path,
        *,
        registry: StageObjectCodecRegistry | None = None,
        fault_injector: StageObjectStoreFaultInjector | None = None,
    ) -> None:
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.object_log_path = self.root / "objects.jsonl"
        self.registry = registry or canary_stage_object_codec_registry()
        self.cas = ContentAddressedByteStore(self.root / "cas")
        self.projection = _StageObjectProjection(self.root / "projection.sqlite3")
        self.fault_injector = fault_injector

    def admit_seed(
        self,
        seed: R6CanarySeedObjectV2,
    ) -> StageObjectWriteResult:
        value = self.registry.decode(
            seed.codec,
            seed.payload_json.encode(),
        )
        observed_ref = self.registry.reference(
            seed.codec,
            value,
        )
        if observed_ref != seed.object_ref:
            raise StageObjectCodecError("seed object reference does not match its codec payload")
        return self._put(
            seed.codec,
            value,
            object_ref=observed_ref,
        )

    def put(
        self,
        codec: R6CanaryObjectCodecV2,
        value: ContractModel,
    ) -> StageObjectWriteResult:
        object_ref = self.registry.reference(codec, value)
        return self._put(codec, value, object_ref=object_ref)

    def get(
        self,
        object_ref: ObjectRef,
        *,
        codec: R6CanaryObjectCodecV2,
    ) -> ContractModel:
        envelopes = self._read_envelopes()
        self.projection.ensure_consistent(envelopes)
        envelope = next(
            (
                item
                for item in envelopes
                if item.object_key
                == (
                    object_ref.object_type,
                    object_ref.object_id,
                    object_ref.object_version,
                )
            ),
            None,
        )
        if envelope is None:
            raise StageObjectNotFoundError("stage object reference was not found")
        if envelope.object_ref != object_ref:
            raise StageObjectConflictError("stage object reference differs from immutable storage")
        if envelope.codec is not codec:
            raise StageObjectCodecError("stage object codec differs from immutable storage")
        payload = self._read_payload(envelope)
        value = self.registry.decode(codec, payload)
        if self.registry.reference(codec, value) != object_ref:
            raise StageObjectCorruptionError("stored stage object payload reference is stale")
        return value

    def get_as[T: ContractModel](
        self,
        object_ref: ObjectRef,
        *,
        codec: R6CanaryObjectCodecV2,
        model_type: type[T],
    ) -> T:
        value = self.get(object_ref, codec=codec)
        if type(value) is not model_type:
            raise StageObjectCodecError("stored stage object has the wrong model type")
        return value

    def list_envelopes(
        self,
    ) -> tuple[StageObjectEnvelopeV2, ...]:
        envelopes = self._read_envelopes()
        self.projection.ensure_consistent(envelopes)
        return envelopes

    def rebuild_projection(
        self,
    ) -> StageObjectProjectionRebuild:
        return self.projection.rebuild(self._read_envelopes())

    def _put(
        self,
        codec: R6CanaryObjectCodecV2,
        value: ContractModel,
        *,
        object_ref: ObjectRef,
    ) -> StageObjectWriteResult:
        definition = self.registry.definition(codec)
        if object_ref.object_type != definition.object_type:
            raise StageObjectCodecError("stage object reference type does not match codec")
        payload = self.registry.encode(codec, value)
        content_digest = hashlib.sha256(payload).hexdigest()
        envelope = StageObjectEnvelopeV2(
            object_ref=object_ref,
            codec=codec,
            content_blob_ref=ObjectRef(
                object_type="stage-object-content",
                object_id=(f"stage-object-content://sha256/{content_digest}"),
                object_version="v1",
                object_sha256=content_digest,
            ),
            canonical_size_bytes=len(payload),
        )
        existing = self._read_envelopes()
        prior = next(
            (item for item in existing if item.object_key == envelope.object_key),
            None,
        )
        if prior is not None:
            if prior != envelope:
                raise StageObjectConflictError(
                    "immutable stage object already has different content or codec"
                )
            self.projection.ensure_consistent(existing)
            stored_value = self.get(object_ref, codec=codec)
            if stored_value != value:
                raise StageObjectCorruptionError("stored stage object differs from exact replay")
            return StageObjectWriteResult(
                object_ref=object_ref,
                envelope=prior,
                content_blob_written=False,
                envelope_appended=False,
            )
        try:
            blob_written = self.cas.write(
                object_id=envelope.content_blob_ref.object_id,
                digest=content_digest,
                value=payload,
            )
        except ContentAddressedByteStoreConflictError as exc:
            raise StageObjectConflictError("stage object CAS write conflicted") from exc
        self._maybe_raise(
            StageObjectStoreFaultPoint.AFTER_CAS_WRITE,
            object_ref,
        )
        self._append_envelope(envelope)
        self._maybe_raise(
            StageObjectStoreFaultPoint.AFTER_ENVELOPE_APPEND,
            object_ref,
        )
        self.projection.rebuild((*existing, envelope))
        self._maybe_raise(
            StageObjectStoreFaultPoint.AFTER_PROJECTION_REBUILD,
            object_ref,
        )
        return StageObjectWriteResult(
            object_ref=object_ref,
            envelope=envelope,
            content_blob_written=blob_written,
            envelope_appended=True,
        )

    def _read_envelopes(
        self,
    ) -> tuple[StageObjectEnvelopeV2, ...]:
        if not self.object_log_path.exists():
            return ()
        envelopes: list[StageObjectEnvelopeV2] = []
        by_key: dict[
            tuple[str, str, str],
            StageObjectEnvelopeV2,
        ] = {}
        with self.object_log_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    envelope = StageObjectEnvelopeV2.model_validate_json(line)
                except (ValidationError, ValueError) as exc:
                    raise StageObjectCorruptionError(
                        f"invalid stage object JSONL at line {line_number}"
                    ) from exc
                definition = self.registry.definition(envelope.codec)
                if envelope.object_ref.object_type != definition.object_type:
                    raise StageObjectCorruptionError("stage object envelope codec/ref mismatch")
                prior = by_key.get(envelope.object_key)
                if prior is not None:
                    if prior != envelope:
                        raise StageObjectConflictError("conflicting stage object envelopes")
                    raise StageObjectCorruptionError("duplicate stage object envelope")
                by_key[envelope.object_key] = envelope
                envelopes.append(envelope)
        return tuple(envelopes)

    def _append_envelope(
        self,
        envelope: StageObjectEnvelopeV2,
    ) -> None:
        with self.object_log_path.open("ab") as handle:
            handle.write(envelope.jsonl_bytes())
            handle.flush()
            os.fsync(handle.fileno())

    def _read_payload(
        self,
        envelope: StageObjectEnvelopeV2,
    ) -> bytes:
        try:
            payload = self.cas.read(
                object_id=envelope.content_blob_ref.object_id,
                digest=(envelope.content_blob_ref.object_sha256),
            )
        except ContentAddressedByteStoreCorruptionError as exc:
            raise StageObjectCorruptionError("stored stage object CAS payload is corrupt") from exc
        if len(payload) != envelope.canonical_size_bytes:
            raise StageObjectCorruptionError("stored stage object CAS payload size differs")
        return payload

    def _maybe_raise(
        self,
        point: StageObjectStoreFaultPoint,
        object_ref: ObjectRef,
    ) -> None:
        if self.fault_injector is not None:
            self.fault_injector.maybe_raise(
                point,
                object_ref=object_ref,
            )


class _StageObjectProjection:
    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self._connect() as connection:
                connection.execute("PRAGMA journal_mode = WAL")
                self._create_schema(connection)
        except sqlite3.Error as exc:
            raise StageObjectProjectionError("failed to initialize stage object projection") from exc

    def rebuild(
        self,
        envelopes: tuple[StageObjectEnvelopeV2, ...],
    ) -> StageObjectProjectionRebuild:
        try:
            with self._transaction() as connection:
                connection.execute("DROP TABLE IF EXISTS stage_objects")
                self._create_schema(connection)
                for sequence, envelope in enumerate(envelopes):
                    self._insert(
                        connection,
                        sequence=sequence,
                        envelope=envelope,
                    )
                digest = self._digest(connection)
        except sqlite3.Error as exc:
            raise StageObjectProjectionError("failed to rebuild stage object projection") from exc
        return StageObjectProjectionRebuild(
            object_count=len(envelopes),
            projection_sha256=digest,
        )

    def ensure_consistent(
        self,
        envelopes: tuple[StageObjectEnvelopeV2, ...],
    ) -> None:
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    """
                    SELECT
                        object_type,
                        object_id,
                        object_version,
                        object_sha256,
                        codec,
                        content_blob_id,
                        content_blob_sha256,
                        canonical_size_bytes,
                        log_sequence
                    FROM stage_objects
                    ORDER BY log_sequence
                    """
                ).fetchall()
        except sqlite3.Error as exc:
            raise StageObjectProjectionError("failed to read stage object projection") from exc
        observed = tuple(
            (
                str(row["object_type"]),
                str(row["object_id"]),
                str(row["object_version"]),
                str(row["object_sha256"]),
                str(row["codec"]),
                str(row["content_blob_id"]),
                str(row["content_blob_sha256"]),
                int(row["canonical_size_bytes"]),
                int(row["log_sequence"]),
            )
            for row in rows
        )
        expected = tuple(_projection_row(sequence, envelope) for sequence, envelope in enumerate(envelopes))
        if observed != expected:
            raise StageObjectCorruptionError("stage object projection differs from object log")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    @contextmanager
    def _transaction(
        self,
    ) -> Generator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _create_schema(
        self,
        connection: sqlite3.Connection,
    ) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS stage_objects (
                object_type TEXT NOT NULL,
                object_id TEXT NOT NULL,
                object_version TEXT NOT NULL,
                object_sha256 TEXT NOT NULL,
                codec TEXT NOT NULL,
                content_blob_id TEXT NOT NULL,
                content_blob_sha256 TEXT NOT NULL,
                canonical_size_bytes INTEGER NOT NULL,
                log_sequence INTEGER NOT NULL UNIQUE,
                PRIMARY KEY (
                    object_type,
                    object_id,
                    object_version
                )
            )
            """
        )

    def _insert(
        self,
        connection: sqlite3.Connection,
        *,
        sequence: int,
        envelope: StageObjectEnvelopeV2,
    ) -> None:
        connection.execute(
            """
            INSERT INTO stage_objects (
                object_type,
                object_id,
                object_version,
                object_sha256,
                codec,
                content_blob_id,
                content_blob_sha256,
                canonical_size_bytes,
                log_sequence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            _projection_row(sequence, envelope),
        )

    def _digest(self, connection: sqlite3.Connection) -> str:
        rows = [
            dict(row)
            for row in connection.execute(
                """
                SELECT
                    object_type,
                    object_id,
                    object_version,
                    object_sha256,
                    codec,
                    content_blob_id,
                    content_blob_sha256,
                    canonical_size_bytes,
                    log_sequence
                FROM stage_objects
                ORDER BY log_sequence
                """
            )
        ]
        encoded = json.dumps(
            rows,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


def _projection_row(
    sequence: int,
    envelope: StageObjectEnvelopeV2,
) -> tuple[str, str, str, str, str, str, str, int, int]:
    return (
        envelope.object_ref.object_type,
        envelope.object_ref.object_id,
        envelope.object_ref.object_version,
        envelope.object_ref.object_sha256,
        envelope.codec.value,
        envelope.content_blob_ref.object_id,
        envelope.content_blob_ref.object_sha256,
        envelope.canonical_size_bytes,
        sequence,
    )


def _string_field(
    value: ContractModel,
    field_name: str | None,
) -> str:
    if field_name is None:
        raise StageObjectCodecError("stage object codec has no object version source")
    observed = getattr(value, field_name, None)
    if not isinstance(observed, str) or not observed:
        raise StageObjectCodecError(f"stage object codec field {field_name} is invalid")
    return observed


def _reference_sha256(
    value: ContractModel,
    *,
    exclude_audit: bool,
) -> str:
    payload = value.model_dump(
        mode="python",
        exclude={"audit"} if exclude_audit else None,
    )
    encoded = json.dumps(
        canonical_value_v2(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
