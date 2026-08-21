from __future__ import annotations

from enum import StrEnum
from typing import Annotated, ClassVar, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from eval_factory.contracts.core import (
    ContractAudit,
    Identifier,
    ObjectRef,
)
from eval_factory.harness.contracts import (
    HarnessObjectV1,
    require_ref,
    require_sorted_unique,
    require_sorted_unique_refs,
    sorted_refs,
)

MediaType = Annotated[
    str,
    StringConstraints(
        pattern=r"^[A-Za-z0-9][A-Za-z0-9!#$&^_.+-]*/[A-Za-z0-9][A-Za-z0-9!#$&^_.+-]*$",
        max_length=255,
    ),
]


class ArtifactModalityV1(StrEnum):
    TEXT = "TEXT"
    TABLE = "TABLE"
    DOCUMENT = "DOCUMENT"
    IMAGE = "IMAGE"
    AUDIO = "AUDIO"
    VIDEO = "VIDEO"
    THREE_DIMENSIONAL = "THREE_DIMENSIONAL"
    SIMULATION = "SIMULATION"
    BINARY = "BINARY"
    MIXED = "MIXED"


class ArtifactEnvelopeV1(HarnessObjectV1):
    schema_version: Literal["eval-harness/artifact-envelope/v1"] = "eval-harness/artifact-envelope/v1"
    OBJECT_TYPE: ClassVar[str] = "artifact-envelope"

    artifact_id: Identifier
    subject_ref: ObjectRef
    schema_ref: ObjectRef
    content_ref: ObjectRef
    media_type: MediaType
    modality: ArtifactModalityV1
    domain_tags: tuple[Identifier, ...] = Field(default=(), max_length=64)
    semantic_role: Identifier
    purpose: Identifier
    classification: Identifier
    lineage_refs: tuple[ObjectRef, ...] = Field(default=(), max_length=10_000)
    producer_capability_ref: ObjectRef
    producer_task_ref: ObjectRef | None = None
    validation_refs: tuple[ObjectRef, ...] = Field(default=(), max_length=10_000)
    revision: int = Field(ge=1, le=1_000_000_000)
    predecessor_envelope_ref: ObjectRef | None = None

    @classmethod
    def create(  # type: ignore[override]
        cls,
        *,
        artifact_id: str,
        subject_ref: ObjectRef,
        schema_ref: ObjectRef,
        content_ref: ObjectRef,
        media_type: str,
        modality: ArtifactModalityV1,
        domain_tags: tuple[str, ...],
        semantic_role: str,
        purpose: str,
        classification: str,
        lineage_refs: tuple[ObjectRef, ...],
        producer_capability_ref: ObjectRef,
        producer_task_ref: ObjectRef | None,
        validation_refs: tuple[ObjectRef, ...],
        revision: int,
        predecessor_envelope_ref: ObjectRef | None,
        audit: ContractAudit,
    ) -> ArtifactEnvelopeV1:
        return super().create(
            audit=audit,
            artifact_id=artifact_id,
            subject_ref=subject_ref,
            schema_ref=schema_ref,
            content_ref=content_ref,
            media_type=media_type,
            modality=modality,
            domain_tags=tuple(sorted(domain_tags)),
            semantic_role=semantic_role,
            purpose=purpose,
            classification=classification,
            lineage_refs=sorted_refs(lineage_refs),
            producer_capability_ref=producer_capability_ref,
            producer_task_ref=producer_task_ref,
            validation_refs=sorted_refs(validation_refs),
            revision=revision,
            predecessor_envelope_ref=predecessor_envelope_ref,
        )

    @model_validator(mode="after")
    def validate_envelope(self) -> Self:
        require_ref(
            self.schema_ref,
            "json-schema",
            "schema_ref",
            object_version=self.schema_ref.object_version,
        )
        require_ref(
            self.producer_capability_ref,
            "harness-capability-definition",
            "producer_capability_ref",
        )
        require_sorted_unique(self.domain_tags, "domain_tags")
        require_sorted_unique_refs(self.lineage_refs, "lineage_refs")
        require_sorted_unique_refs(self.validation_refs, "validation_refs")
        if self.producer_task_ref is not None and self.producer_task_ref.object_type not in {
            "agent-task",
            "team-task",
        }:
            raise ValueError("producer_task_ref must reference Agent or Team task authority")
        if self.revision == 1:
            if self.predecessor_envelope_ref is not None:
                raise ValueError("first artifact revision cannot have a predecessor")
        else:
            if self.predecessor_envelope_ref is None:
                raise ValueError("artifact successor requires its predecessor envelope")
            require_ref(
                self.predecessor_envelope_ref,
                "artifact-envelope",
                "predecessor_envelope_ref",
            )
        return self


class ArtifactHeadV1(HarnessObjectV1):
    schema_version: Literal["eval-harness/artifact-head/v1"] = "eval-harness/artifact-head/v1"
    OBJECT_TYPE: ClassVar[str] = "artifact-head"

    head_id: Identifier
    team_ref: ObjectRef
    semantic_role: Identifier
    revision: int = Field(ge=1, le=1_000_000_000)
    envelope_ref: ObjectRef
    predecessor_head_ref: ObjectRef | None = None

    @model_validator(mode="after")
    def validate_head(self) -> Self:
        require_ref(self.team_ref, "agent-team", "team_ref")
        require_ref(self.envelope_ref, "artifact-envelope", "envelope_ref")
        if self.revision == 1:
            if self.predecessor_head_ref is not None:
                raise ValueError("first artifact head cannot have a predecessor")
        elif self.predecessor_head_ref is None:
            raise ValueError("artifact head successor requires its predecessor")
        else:
            require_ref(
                self.predecessor_head_ref,
                "artifact-head",
                "predecessor_head_ref",
            )
        return self


__all__ = [
    "ArtifactEnvelopeV1",
    "ArtifactHeadV1",
    "ArtifactModalityV1",
    "MediaType",
]
