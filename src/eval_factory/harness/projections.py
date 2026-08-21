from __future__ import annotations

from enum import StrEnum
from typing import Annotated, ClassVar, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from eval_factory.contracts.core import ContractAudit, Identifier, ObjectRef
from eval_factory.harness.contracts import (
    HarnessObjectV1,
    require_sorted_unique,
)

ProjectionVersion = Annotated[
    str,
    StringConstraints(min_length=1, max_length=128),
]


class ProjectionVisibilityV1(StrEnum):
    INTERNAL = "INTERNAL"
    USER_SAFE = "USER_SAFE"
    MEMBER_SCOPED = "MEMBER_SCOPED"


class ProjectionDefinitionV1(HarnessObjectV1):
    schema_version: Literal["eval-harness/projection-definition/v1"] = "eval-harness/projection-definition/v1"
    OBJECT_TYPE: ClassVar[str] = "harness-projection-definition"

    projection_id: Identifier
    projection_version: ProjectionVersion
    reducer_id: Identifier
    source_event_families: tuple[Identifier, ...] = Field(
        default=(),
        max_length=128,
    )
    source_artifact_roles: tuple[Identifier, ...] = Field(
        default=(),
        max_length=256,
    )
    output_schema_ref: ObjectRef
    visibility: ProjectionVisibilityV1
    server_owned: Literal[True] = True
    content_safe: bool

    @classmethod
    def create(  # type: ignore[override]
        cls,
        *,
        projection_id: str,
        projection_version: str,
        reducer_id: str,
        source_event_families: tuple[str, ...],
        source_artifact_roles: tuple[str, ...],
        output_schema_ref: ObjectRef,
        visibility: ProjectionVisibilityV1,
        content_safe: bool,
        audit: ContractAudit,
    ) -> ProjectionDefinitionV1:
        return super().create(
            audit=audit,
            projection_id=projection_id,
            projection_version=projection_version,
            reducer_id=reducer_id,
            source_event_families=tuple(sorted(source_event_families)),
            source_artifact_roles=tuple(sorted(source_artifact_roles)),
            output_schema_ref=output_schema_ref,
            visibility=visibility,
            content_safe=content_safe,
        )

    @model_validator(mode="after")
    def validate_definition(self) -> Self:
        if not self.source_event_families and not self.source_artifact_roles:
            raise ValueError("projection requires an event or artifact source")
        require_sorted_unique(
            self.source_event_families,
            "source_event_families",
        )
        require_sorted_unique(
            self.source_artifact_roles,
            "source_artifact_roles",
        )
        if self.output_schema_ref.object_type != "json-schema":
            raise ValueError("output_schema_ref must reference a JSON schema")
        if self.visibility is ProjectionVisibilityV1.USER_SAFE and not self.content_safe:
            raise ValueError("user-safe projection must be content safe")
        return self


__all__ = [
    "ProjectionDefinitionV1",
    "ProjectionVersion",
    "ProjectionVisibilityV1",
]
