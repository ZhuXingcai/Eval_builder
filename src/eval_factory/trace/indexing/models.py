from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from eval_factory.contracts.core import ContractAudit, ObjectRef, TypedAttribute
from eval_factory.contracts.trace import FileObservation, InteractionSegment
from eval_factory.trace.normalization import TraceNormalizationResult
from eval_factory.trace.normalization.models import object_ref_for_event

FILE_OBSERVATION_POLICY_VERSION = "raw-traj-file-observation/v1"
SEGMENTATION_POLICY_VERSION = "deterministic-interaction-segments/v1"


@dataclass(frozen=True)
class IndexDiagnostic:
    code: str
    event_ref: ObjectRef | None = None
    tool_call_record_ref: ObjectRef | None = None
    attributes: tuple[TypedAttribute, ...] = ()


@dataclass(frozen=True)
class TraceIndexResult:
    normalization_result: TraceNormalizationResult
    file_observation_policy_version: str
    segmentation_policy_version: str
    file_observations: tuple[FileObservation, ...]
    interaction_segments: tuple[InteractionSegment, ...]
    diagnostics: tuple[IndexDiagnostic, ...]
    audit: ContractAudit

    def __post_init__(self) -> None:
        if self.file_observation_policy_version != FILE_OBSERVATION_POLICY_VERSION:
            raise ValueError("unexpected file-observation policy version")
        if self.segmentation_policy_version != SEGMENTATION_POLICY_VERSION:
            raise ValueError("unexpected segmentation policy version")
        _require_unique(
            "file observation",
            (item.observation_id for item in self.file_observations),
        )
        _require_unique(
            "interaction segment",
            (item.segment_id for item in self.interaction_segments),
        )
        events = {event.event_id: object_ref_for_event(event) for event in self.normalization_result.events}
        content = {
            blob.content_ref.object_id: blob.content_ref for blob in self.normalization_result.content_blobs
        }
        for observation in self.file_observations:
            for ref in observation.source_event_refs:
                if events.get(ref.object_id) != ref:
                    raise ValueError(f"file observation references unresolved event: {ref.object_id}")
            if content.get(observation.raw_path_ref.object_id) != observation.raw_path_ref:
                raise ValueError("file observation references unresolved raw path blob")
            if observation.content_ref is not None:
                if content.get(observation.content_ref.object_id) != observation.content_ref:
                    raise ValueError("file observation references unresolved content blob")
                if observation.content_sha256 != observation.content_ref.object_sha256:
                    raise ValueError("file observation content hash does not match content ref")
            elif observation.content_sha256 is not None:
                raise ValueError("file observation content hash requires content ref")
        for segment in self.interaction_segments:
            for ref in segment.member_event_refs:
                if events.get(ref.object_id) != ref:
                    raise ValueError(f"segment references unresolved event: {ref.object_id}")


def object_ref_for_observation(observation: FileObservation) -> ObjectRef:
    return ObjectRef(
        object_type="file-observation",
        object_id=observation.observation_id,
        object_version="v1",
        object_sha256=observation.canonical_sha256(),
    )


def object_ref_for_segment(segment: InteractionSegment) -> ObjectRef:
    return ObjectRef(
        object_type="interaction-segment",
        object_id=segment.segment_id,
        object_version=SEGMENTATION_POLICY_VERSION,
        object_sha256=segment.canonical_sha256(),
    )


def _require_unique(label: str, values: Iterable[str]) -> None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            raise ValueError(f"duplicate {label} id: {value}")
        seen.add(value)
