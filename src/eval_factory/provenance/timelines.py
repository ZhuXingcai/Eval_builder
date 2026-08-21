from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

from pydantic import Field, field_validator

from eval_factory.contracts.core import (
    ContractAudit,
    ContractModel,
    Identifier,
    ObjectRef,
    RelativePath,
    Sha256,
)
from eval_factory.contracts.safety import OriginClass
from eval_factory.contracts.trace import Completeness, FileObservation, FileOperation

FILE_VERSION_TIMELINE_POLICY_VERSION: Literal["file-version-timeline/r2-02-v1"] = (
    "file-version-timeline/r2-02-v1"
)


class FileVersionOrigin(StrEnum):
    PREEXISTING_WORKSPACE_INPUT = "PREEXISTING_WORKSPACE_INPUT"
    UNKNOWN_PREEXISTING = "UNKNOWN_PREEXISTING"
    AGENT_GENERATED = "AGENT_GENERATED"
    UNKNOWN_BOUNDARY = "UNKNOWN_BOUNDARY"


class FileVersionUncertainty(StrEnum):
    NO_PRE_MUTATION_READ = "NO_PRE_MUTATION_READ"
    UNKNOWN_MUTATION_BOUNDARY = "UNKNOWN_MUTATION_BOUNDARY"
    PARTIAL_OBSERVATION = "PARTIAL_OBSERVATION"
    UNKNOWN_COMPLETENESS = "UNKNOWN_COMPLETENESS"
    TRUNCATED_OBSERVATION = "TRUNCATED_OBSERVATION"
    TRUNCATION_UNKNOWN = "TRUNCATION_UNKNOWN"
    CONTENT_NOT_OBSERVED = "CONTENT_NOT_OBSERVED"
    DUPLICATE_OBSERVATION_ID = "DUPLICATE_OBSERVATION_ID"


class FileVersionObservationRef(ContractModel):
    schema_version: Literal["eval-factory/file-version-observation-ref/r2-02"] = (
        "eval-factory/file-version-observation-ref/r2-02"
    )
    observation_ref: ObjectRef
    operation: FileOperation
    sequence: int = Field(ge=0)
    observed_start: int | None = Field(default=None, ge=0)
    observed_end: int | None = Field(default=None, ge=0)
    completeness: Completeness
    truncated: bool | None
    content_ref: ObjectRef | None
    content_sha256: Sha256 | None
    source_event_refs: tuple[ObjectRef, ...] = Field(min_length=1)

    @field_validator("operation", mode="before")
    @classmethod
    def parse_operation(cls, value: object) -> FileOperation:
        if isinstance(value, FileOperation):
            return value
        if isinstance(value, str):
            return FileOperation(value)
        raise TypeError("operation must be a FileOperation")

    @field_validator("completeness", mode="before")
    @classmethod
    def parse_completeness(cls, value: object) -> Completeness:
        if isinstance(value, Completeness):
            return value
        if isinstance(value, str):
            return Completeness(value)
        raise TypeError("completeness must be a Completeness")


class FileVersionRecord(ContractModel):
    schema_version: Literal["eval-factory/file-version-record/r2-02"] = (
        "eval-factory/file-version-record/r2-02"
    )
    version_id: Identifier
    trace_ir_version_id: Identifier
    logical_path: RelativePath
    version_index: int = Field(ge=0)
    origin: FileVersionOrigin
    origin_class: OriginClass
    operation: FileOperation | None = None
    creating_observation_ref: ObjectRef | None = None
    observation_refs: tuple[FileVersionObservationRef, ...] = ()
    read_observation_refs: tuple[ObjectRef, ...] = ()
    content_refs: tuple[ObjectRef, ...] = ()
    content_sha256s: tuple[Sha256, ...] = ()
    completeness: Completeness
    truncated: bool | None
    uncertainties: tuple[FileVersionUncertainty, ...] = ()
    source_event_refs: tuple[ObjectRef, ...] = ()
    policy_version: Literal["file-version-timeline/r2-02-v1"] = FILE_VERSION_TIMELINE_POLICY_VERSION
    audit: ContractAudit

    @field_validator("origin", mode="before")
    @classmethod
    def parse_origin(cls, value: object) -> FileVersionOrigin:
        if isinstance(value, FileVersionOrigin):
            return value
        if isinstance(value, str):
            return FileVersionOrigin(value)
        raise TypeError("origin must be a FileVersionOrigin")

    @field_validator("origin_class", mode="before")
    @classmethod
    def parse_origin_class(cls, value: object) -> OriginClass:
        if isinstance(value, OriginClass):
            return value
        if isinstance(value, str):
            return OriginClass(value)
        raise TypeError("origin_class must be an OriginClass")

    @field_validator("operation", mode="before")
    @classmethod
    def parse_operation(cls, value: object) -> FileOperation | None:
        if value is None or isinstance(value, FileOperation):
            return value
        if isinstance(value, str):
            return FileOperation(value)
        raise TypeError("operation must be a FileOperation or None")

    @field_validator("completeness", mode="before")
    @classmethod
    def parse_completeness(cls, value: object) -> Completeness:
        if isinstance(value, Completeness):
            return value
        if isinstance(value, str):
            return Completeness(value)
        raise TypeError("completeness must be a Completeness")

    @field_validator("uncertainties", mode="before")
    @classmethod
    def parse_uncertainties(cls, value: object) -> tuple[FileVersionUncertainty, ...]:
        if isinstance(value, tuple):
            return tuple(
                item if isinstance(item, FileVersionUncertainty) else FileVersionUncertainty(item)
                for item in value
            )
        if isinstance(value, (list, set, frozenset)):
            return tuple(FileVersionUncertainty(item) for item in value)
        raise TypeError("uncertainties must be a collection")


class FileVersionTimeline(ContractModel):
    schema_version: Literal["eval-factory/file-version-timeline/r2-02"] = (
        "eval-factory/file-version-timeline/r2-02"
    )
    timeline_id: Identifier
    trace_ir_version_id: Identifier
    logical_path: RelativePath
    versions: tuple[FileVersionRecord, ...] = Field(min_length=1)
    policy_version: Literal["file-version-timeline/r2-02-v1"] = FILE_VERSION_TIMELINE_POLICY_VERSION
    audit: ContractAudit


class FileVersionTimelineBuilder:
    policy_version = FILE_VERSION_TIMELINE_POLICY_VERSION

    def build(
        self,
        *,
        trace_ir_version_id: str,
        file_observations: tuple[FileObservation, ...],
        audit: ContractAudit,
    ) -> tuple[FileVersionTimeline, ...]:
        grouped: dict[str, list[FileObservation]] = defaultdict(list)
        for observation in file_observations:
            if observation.trace_ir_version_id != trace_ir_version_id:
                continue
            grouped[observation.logical_path].append(observation)

        timelines = [
            self._build_path_timeline(
                trace_ir_version_id=trace_ir_version_id,
                logical_path=logical_path,
                observations=tuple(sorted(items, key=lambda item: (item.sequence, item.observation_id))),
                audit=audit,
            )
            for logical_path, items in sorted(grouped.items())
        ]
        return tuple(timelines)

    def _build_path_timeline(
        self,
        *,
        trace_ir_version_id: str,
        logical_path: str,
        observations: tuple[FileObservation, ...],
        audit: ContractAudit,
    ) -> FileVersionTimeline:
        duplicate_observation_ids = _duplicate_observation_ids(observations)
        drafts: list[_VersionDraft] = [
            _VersionDraft(
                version_index=0,
                origin=FileVersionOrigin.UNKNOWN_PREEXISTING,
                origin_class=OriginClass.UNKNOWN,
                operation=None,
            )
        ]
        current = drafts[0]

        for observation in observations:
            summary = _observation_summary(observation)
            observation_uncertainties = _observation_uncertainties(
                observation,
                duplicate_observation_ids=duplicate_observation_ids,
            )
            if _is_read_like(observation.operation):
                if current.version_index == 0 and current.origin is FileVersionOrigin.UNKNOWN_PREEXISTING:
                    current.origin = FileVersionOrigin.PREEXISTING_WORKSPACE_INPUT
                    current.origin_class = OriginClass.PREEXISTING_WORKSPACE_INPUT
                current.add_observation(
                    summary,
                    is_read=True,
                    uncertainties=observation_uncertainties,
                )
                continue

            if _is_agent_generated_mutation(observation.operation):
                current = _VersionDraft(
                    version_index=len(drafts),
                    origin=FileVersionOrigin.AGENT_GENERATED,
                    origin_class=OriginClass.AGENT_GENERATED_INTERMEDIATE,
                    operation=observation.operation,
                    creating_observation_ref=summary.observation_ref,
                )
                current.add_observation(
                    summary,
                    is_read=False,
                    uncertainties=observation_uncertainties,
                )
                drafts.append(current)
                continue

            current = _VersionDraft(
                version_index=len(drafts),
                origin=FileVersionOrigin.UNKNOWN_BOUNDARY,
                origin_class=OriginClass.UNKNOWN,
                operation=observation.operation,
                creating_observation_ref=summary.observation_ref,
                uncertainties=[FileVersionUncertainty.UNKNOWN_MUTATION_BOUNDARY],
            )
            current.add_observation(
                summary,
                is_read=False,
                uncertainties=observation_uncertainties,
            )
            drafts.append(current)

        versions = tuple(
            _build_record(
                trace_ir_version_id=trace_ir_version_id,
                logical_path=logical_path,
                draft=draft,
                policy_version=self.policy_version,
                audit=audit,
            )
            for draft in drafts
        )
        timeline_id = _stable_id(
            "file-version-timeline",
            {
                "trace_ir_version_id": trace_ir_version_id,
                "logical_path": logical_path,
                "policy_version": self.policy_version,
                "version_ids": [version.version_id for version in versions],
            },
        )
        return FileVersionTimeline(
            timeline_id=timeline_id,
            trace_ir_version_id=trace_ir_version_id,
            logical_path=logical_path,
            versions=versions,
            policy_version=self.policy_version,
            audit=audit,
        )


@dataclass
class _VersionDraft:
    version_index: int
    origin: FileVersionOrigin
    origin_class: OriginClass
    operation: FileOperation | None
    creating_observation_ref: ObjectRef | None = None
    observations: list[FileVersionObservationRef] = field(default_factory=list)
    read_observation_refs: list[ObjectRef] = field(default_factory=list)
    uncertainties: list[FileVersionUncertainty] = field(default_factory=list)

    def add_observation(
        self,
        observation: FileVersionObservationRef,
        *,
        is_read: bool,
        uncertainties: tuple[FileVersionUncertainty, ...],
    ) -> None:
        self.observations.append(observation)
        if is_read:
            self.read_observation_refs.append(observation.observation_ref)
        self.uncertainties.extend(uncertainties)


def _build_record(
    *,
    trace_ir_version_id: str,
    logical_path: str,
    draft: _VersionDraft,
    policy_version: Literal["file-version-timeline/r2-02-v1"],
    audit: ContractAudit,
) -> FileVersionRecord:
    observations = tuple(draft.observations)
    uncertainties = _record_uncertainties(draft)
    content_refs = _content_refs(observations)
    content_sha256s = _content_sha256s(observations)
    source_event_refs = _source_event_refs(observations)
    completeness = _summarize_completeness(observations)
    truncated = _summarize_truncated(observations)
    version_payload = {
        "trace_ir_version_id": trace_ir_version_id,
        "logical_path": logical_path,
        "version_index": draft.version_index,
        "origin": draft.origin.value,
        "origin_class": draft.origin_class.value,
        "operation": draft.operation.value if draft.operation is not None else None,
        "creating_observation_ref": (
            draft.creating_observation_ref.model_dump(mode="json", exclude_none=False)
            if draft.creating_observation_ref is not None
            else None
        ),
        "observation_refs": [item.model_dump(mode="json", exclude_none=False) for item in observations],
        "read_observation_refs": [
            item.model_dump(mode="json", exclude_none=False) for item in draft.read_observation_refs
        ],
        "content_refs": [item.model_dump(mode="json", exclude_none=False) for item in content_refs],
        "content_sha256s": list(content_sha256s),
        "completeness": completeness.value,
        "truncated": truncated,
        "uncertainties": [item.value for item in uncertainties],
        "source_event_refs": [item.model_dump(mode="json", exclude_none=False) for item in source_event_refs],
        "policy_version": policy_version,
    }
    return FileVersionRecord(
        version_id=_stable_id("file-version", version_payload),
        trace_ir_version_id=trace_ir_version_id,
        logical_path=logical_path,
        version_index=draft.version_index,
        origin=draft.origin,
        origin_class=draft.origin_class,
        operation=draft.operation,
        creating_observation_ref=draft.creating_observation_ref,
        observation_refs=observations,
        read_observation_refs=tuple(draft.read_observation_refs),
        content_refs=content_refs,
        content_sha256s=content_sha256s,
        completeness=completeness,
        truncated=truncated,
        uncertainties=uncertainties,
        source_event_refs=source_event_refs,
        policy_version=policy_version,
        audit=audit,
    )


def _observation_summary(observation: FileObservation) -> FileVersionObservationRef:
    return FileVersionObservationRef(
        observation_ref=_observation_ref(observation),
        operation=observation.operation,
        sequence=observation.sequence,
        observed_start=observation.observed_start,
        observed_end=observation.observed_end,
        completeness=observation.completeness,
        truncated=observation.truncated,
        content_ref=observation.content_ref,
        content_sha256=observation.content_sha256,
        source_event_refs=observation.source_event_refs,
    )


def _observation_ref(observation: FileObservation) -> ObjectRef:
    return ObjectRef(
        object_type="file-observation",
        object_id=observation.observation_id,
        object_version="v1",
        object_sha256=observation.canonical_sha256(),
    )


def _observation_uncertainties(
    observation: FileObservation,
    *,
    duplicate_observation_ids: frozenset[str],
) -> tuple[FileVersionUncertainty, ...]:
    uncertainties: list[FileVersionUncertainty] = []
    if observation.completeness is Completeness.PARTIAL:
        uncertainties.append(FileVersionUncertainty.PARTIAL_OBSERVATION)
    elif observation.completeness is Completeness.UNKNOWN:
        uncertainties.append(FileVersionUncertainty.UNKNOWN_COMPLETENESS)
    if observation.truncated is True:
        uncertainties.append(FileVersionUncertainty.TRUNCATED_OBSERVATION)
    elif observation.truncated is None:
        uncertainties.append(FileVersionUncertainty.TRUNCATION_UNKNOWN)
    if observation.content_ref is None or observation.content_sha256 is None:
        uncertainties.append(FileVersionUncertainty.CONTENT_NOT_OBSERVED)
    if observation.observation_id in duplicate_observation_ids:
        uncertainties.append(FileVersionUncertainty.DUPLICATE_OBSERVATION_ID)
    return _unique_uncertainties(uncertainties)


def _record_uncertainties(draft: _VersionDraft) -> tuple[FileVersionUncertainty, ...]:
    uncertainties = list(draft.uncertainties)
    if draft.version_index == 0 and not draft.read_observation_refs:
        uncertainties.append(FileVersionUncertainty.NO_PRE_MUTATION_READ)
    return _unique_uncertainties(uncertainties)


def _is_read_like(operation: FileOperation) -> bool:
    return operation in {FileOperation.READ, FileOperation.LIST}


def _is_agent_generated_mutation(operation: FileOperation) -> bool:
    return operation in {FileOperation.WRITE, FileOperation.EDIT}


def _content_refs(observations: tuple[FileVersionObservationRef, ...]) -> tuple[ObjectRef, ...]:
    return _unique_object_refs(
        tuple(item.content_ref for item in observations if item.content_ref is not None)
    )


def _content_sha256s(observations: tuple[FileVersionObservationRef, ...]) -> tuple[str, ...]:
    values: list[str] = []
    seen: set[str] = set()
    for observation in observations:
        if observation.content_sha256 is None or observation.content_sha256 in seen:
            continue
        seen.add(observation.content_sha256)
        values.append(observation.content_sha256)
    return tuple(values)


def _source_event_refs(observations: tuple[FileVersionObservationRef, ...]) -> tuple[ObjectRef, ...]:
    return _unique_object_refs(tuple(ref for item in observations for ref in item.source_event_refs))


def _unique_object_refs(values: tuple[ObjectRef | None, ...]) -> tuple[ObjectRef, ...]:
    refs: list[ObjectRef] = []
    seen: set[tuple[str, str, str, str]] = set()
    for value in values:
        if value is None:
            continue
        key = (value.object_type, value.object_id, value.object_version, value.object_sha256)
        if key in seen:
            continue
        seen.add(key)
        refs.append(value)
    return tuple(refs)


def _unique_uncertainties(
    values: list[FileVersionUncertainty] | tuple[FileVersionUncertainty, ...],
) -> tuple[FileVersionUncertainty, ...]:
    return tuple(
        FileVersionUncertainty(item)
        for item in sorted({FileVersionUncertainty(item).value for item in values})
    )


def _summarize_completeness(observations: tuple[FileVersionObservationRef, ...]) -> Completeness:
    if not observations:
        return Completeness.UNKNOWN
    values = {item.completeness for item in observations}
    if Completeness.UNKNOWN in values:
        return Completeness.UNKNOWN
    if Completeness.PARTIAL in values:
        return Completeness.PARTIAL
    return Completeness.COMPLETE


def _summarize_truncated(observations: tuple[FileVersionObservationRef, ...]) -> bool | None:
    if not observations:
        return None
    values = {item.truncated for item in observations}
    if True in values:
        return True
    if None in values:
        return None
    return False


def _duplicate_observation_ids(observations: tuple[FileObservation, ...]) -> frozenset[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for observation in observations:
        if observation.observation_id in seen:
            duplicates.add(observation.observation_id)
        seen.add(observation.observation_id)
    return frozenset(duplicates)


def _stable_id(kind: str, payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return f"{kind}://sha256/{hashlib.sha256(encoded).hexdigest()}"
