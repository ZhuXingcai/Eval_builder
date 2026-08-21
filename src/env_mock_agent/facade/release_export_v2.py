from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from pathlib import Path
from typing import Literal, Protocol, Self

from pydantic import Field, field_validator, model_validator

from env_mock_agent.facade.contracts import (
    FacadeModel,
    FacadeObjectRef,
    Identifier,
    RelativePath,
    Sha256,
)

LH_RELEASE_EXPORT_POLICY_VERSION: Literal["lh-release-export/r7-09-v1"] = "lh-release-export/r7-09-v1"


class LHWorkspaceExportMemberTypeV2(StrEnum):
    DIRECTORY = "DIRECTORY"
    FILE = "FILE"
    NESTED_MEMBER = "NESTED_MEMBER"


class LHWorkspaceExportOutcomeV2(StrEnum):
    EXPORTED = "EXPORTED"
    BLOCKED = "BLOCKED"


class LHWorkspaceExportFailureCodeV2(StrEnum):
    MATERIAL_NOT_FOUND = "MATERIAL_NOT_FOUND"
    MATERIAL_MISMATCH = "MATERIAL_MISMATCH"
    SOURCE_PATH_VIOLATION = "SOURCE_PATH_VIOLATION"
    DESTINATION_PATH_VIOLATION = "DESTINATION_PATH_VIOLATION"
    SYMLINK_UNSUPPORTED = "SYMLINK_UNSUPPORTED"
    SPECIAL_FILE_UNSUPPORTED = "SPECIAL_FILE_UNSUPPORTED"
    SOURCE_HASH_MISMATCH = "SOURCE_HASH_MISMATCH"
    COPIED_HASH_MISMATCH = "COPIED_HASH_MISMATCH"
    INVENTORY_MISMATCH = "INVENTORY_MISMATCH"
    EXPORT_LIMIT_EXCEEDED = "EXPORT_LIMIT_EXCEEDED"
    CONTAINER_UNSUPPORTED = "CONTAINER_UNSUPPORTED"


class LHWorkspaceExportMemberV2(FacadeModel):
    schema_version: Literal["env-mock-agent/lh-workspace-export-member/v2"] = (
        "env-mock-agent/lh-workspace-export-member/v2"
    )
    normalized_path: RelativePath
    member_type: LHWorkspaceExportMemberTypeV2
    media_type: str | None = Field(
        default=None,
        min_length=3,
        max_length=255,
    )
    size_bytes: int = Field(ge=0, le=5_000_000_000)
    content_sha256: Sha256 | None = None
    container_ref: FacadeObjectRef | None = None
    output_ref: FacadeObjectRef

    @field_validator("member_type", mode="before")
    @classmethod
    def parse_member_type(
        cls,
        value: object,
    ) -> LHWorkspaceExportMemberTypeV2:
        return _parse_enum(
            value,
            LHWorkspaceExportMemberTypeV2,
            "member_type",
        )

    @model_validator(mode="after")
    def validate_member(self) -> Self:
        _require_ref(
            self.output_ref,
            "attachment-output",
            "v2",
            "output_ref",
        )
        if self.member_type is LHWorkspaceExportMemberTypeV2.DIRECTORY:
            if self.size_bytes or self.content_sha256 is not None or self.media_type is not None:
                raise ValueError("DIRECTORY member cannot carry content")
            if self.container_ref is not None:
                _require_ref(
                    self.container_ref,
                    "attachment-output",
                    "v2",
                    "container_ref",
                )
        elif self.member_type is LHWorkspaceExportMemberTypeV2.FILE:
            if self.content_sha256 is None:
                raise ValueError("file members require a content hash")
            if self.container_ref is not None:
                raise ValueError("FILE member cannot carry a container ref")
        else:
            if self.content_sha256 is None or self.container_ref is None:
                raise ValueError("nested member requires a content hash and container ref")
            _require_ref(
                self.container_ref,
                "attachment-output",
                "v2",
                "container_ref",
            )
        return self

    @classmethod
    def create(
        cls,
        *,
        normalized_path: str,
        member_type: LHWorkspaceExportMemberTypeV2,
        media_type: str | None,
        size_bytes: int,
        content_sha256: str | None,
        container_ref: FacadeObjectRef | None,
        output_ref: FacadeObjectRef,
    ) -> LHWorkspaceExportMemberV2:
        return cls(
            normalized_path=normalized_path,
            member_type=member_type,
            media_type=media_type,
            size_bytes=size_bytes,
            content_sha256=content_sha256,
            container_ref=container_ref,
            output_ref=output_ref,
        )


class LHWorkspaceExportRequestV2(FacadeModel):
    schema_version: Literal["env-mock-agent/lh-workspace-export-request/v2"] = (
        "env-mock-agent/lh-workspace-export-request/v2"
    )
    request_id: Identifier
    item_id: Identifier
    final_package_manifest_ref: FacadeObjectRef
    package_sha256: Sha256
    output_refs: tuple[FacadeObjectRef, ...]
    members: tuple[LHWorkspaceExportMemberV2, ...]
    max_member_count: int = Field(ge=1, le=100_000)
    max_total_bytes: int = Field(ge=1, le=5_000_000_000)
    max_file_bytes: int = Field(ge=1, le=5_000_000_000)
    max_nested_depth: int = Field(ge=1, le=32)
    max_compression_ratio: int = Field(ge=1, le=10_000)
    policy_version: Literal["lh-release-export/r7-09-v1"] = LH_RELEASE_EXPORT_POLICY_VERSION
    request_sha256: Sha256

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        _require_ref(
            self.final_package_manifest_ref,
            "final-package-manifest",
            "v2",
            "final_package_manifest_ref",
        )
        _require_sorted_unique_refs(
            "workspace export output refs",
            self.output_refs,
        )
        for ref in self.output_refs:
            _require_ref(ref, "attachment-output", "v2", "output_refs")
        _require_sorted_unique_members(self.members)
        observed_outputs = _sorted_refs(tuple(member.output_ref for member in self.members))
        if observed_outputs != self.output_refs:
            raise ValueError("workspace members must exactly cover output refs")
        if (
            len(self.members) > self.max_member_count
            or sum(member.size_bytes for member in self.members) > self.max_total_bytes
            or self.max_file_bytes > self.max_total_bytes
        ):
            raise ValueError("workspace export budget exceeded")
        _validate_identity(
            object_id=self.request_id,
            object_sha256=self.request_sha256,
            prefix="lh-workspace-export-request",
            observed=lh_workspace_export_request_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        item_id: str,
        final_package_manifest_ref: FacadeObjectRef,
        package_sha256: str,
        output_refs: tuple[FacadeObjectRef, ...],
        members: tuple[LHWorkspaceExportMemberV2, ...],
        max_member_count: int,
        max_total_bytes: int,
        max_file_bytes: int | None = None,
        max_nested_depth: int = 8,
        max_compression_ratio: int = 100,
    ) -> LHWorkspaceExportRequestV2:
        value = cls(
            request_id="lh-workspace-export-request://pending",
            item_id=item_id,
            final_package_manifest_ref=final_package_manifest_ref,
            package_sha256=package_sha256,
            output_refs=_sorted_refs(output_refs),
            members=_sorted_members(members),
            max_member_count=max_member_count,
            max_total_bytes=max_total_bytes,
            max_file_bytes=(max_total_bytes if max_file_bytes is None else max_file_bytes),
            max_nested_depth=max_nested_depth,
            max_compression_ratio=max_compression_ratio,
            request_sha256="0" * 64,
        )
        digest = lh_workspace_export_request_carried_sha256(value)
        return cls.model_validate(
            value.model_copy(
                update={
                    "request_id": (f"lh-workspace-export-request://sha256/{digest}"),
                    "request_sha256": digest,
                }
            ).model_dump(mode="python")
        )


class LHWorkspaceExportResultV2(FacadeModel):
    schema_version: Literal["env-mock-agent/lh-workspace-export-result/v2"] = (
        "env-mock-agent/lh-workspace-export-result/v2"
    )
    result_id: Identifier
    request_ref: FacadeObjectRef
    outcome: LHWorkspaceExportOutcomeV2
    observed_members: tuple[LHWorkspaceExportMemberV2, ...] = ()
    copied_output_refs: tuple[FacadeObjectRef, ...] = ()
    workspace_file_count: int = Field(default=0, ge=0, le=100_000)
    workspace_total_bytes: int = Field(
        default=0,
        ge=0,
        le=5_000_000_000,
    )
    workspace_sha256: Sha256 | None = None
    failure_code: LHWorkspaceExportFailureCodeV2 | None = None
    policy_version: Literal["lh-release-export/r7-09-v1"] = LH_RELEASE_EXPORT_POLICY_VERSION
    result_sha256: Sha256

    @field_validator("outcome", mode="before")
    @classmethod
    def parse_outcome(
        cls,
        value: object,
    ) -> LHWorkspaceExportOutcomeV2:
        return _parse_enum(
            value,
            LHWorkspaceExportOutcomeV2,
            "outcome",
        )

    @field_validator("failure_code", mode="before")
    @classmethod
    def parse_failure_code(
        cls,
        value: object,
    ) -> LHWorkspaceExportFailureCodeV2 | None:
        if value is None:
            return None
        return _parse_enum(
            value,
            LHWorkspaceExportFailureCodeV2,
            "failure_code",
        )

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        _require_ref(
            self.request_ref,
            "lh-workspace-export-request",
            "v2",
            "request_ref",
        )
        _require_sorted_unique_members(self.observed_members)
        _require_sorted_unique_refs(
            "copied output refs",
            self.copied_output_refs,
        )
        if self.outcome is LHWorkspaceExportOutcomeV2.EXPORTED:
            expected_count = sum(
                member.member_type is not LHWorkspaceExportMemberTypeV2.DIRECTORY
                for member in self.observed_members
            )
            expected_bytes = sum(
                member.size_bytes
                for member in self.observed_members
                if member.member_type is not LHWorkspaceExportMemberTypeV2.DIRECTORY
            )
            if (
                self.workspace_file_count != expected_count
                or self.workspace_total_bytes != expected_bytes
                or self.workspace_sha256 != _workspace_sha256(self.observed_members)
                or self.failure_code is not None
            ):
                raise ValueError("EXPORTED result requires exact members, outputs, and hash")
        elif (
            self.observed_members
            or self.copied_output_refs
            or self.workspace_file_count
            or self.workspace_total_bytes
            or self.workspace_sha256 is not None
            or self.failure_code is None
        ):
            raise ValueError("BLOCKED result carries only a failure code")
        _validate_identity(
            object_id=self.result_id,
            object_sha256=self.result_sha256,
            prefix="lh-workspace-export-result",
            observed=lh_workspace_export_result_carried_sha256(self),
        )
        return self

    @classmethod
    def exported(
        cls,
        *,
        request: LHWorkspaceExportRequestV2,
        observed_members: tuple[LHWorkspaceExportMemberV2, ...],
        copied_output_refs: tuple[FacadeObjectRef, ...],
    ) -> LHWorkspaceExportResultV2:
        members = _sorted_members(observed_members)
        if members != request.members or _sorted_refs(copied_output_refs) != request.output_refs:
            raise ValueError("exported result must exactly match the request")
        value = cls(
            result_id="lh-workspace-export-result://pending",
            request_ref=lh_workspace_export_request_ref(request),
            outcome=LHWorkspaceExportOutcomeV2.EXPORTED,
            observed_members=members,
            copied_output_refs=_sorted_refs(copied_output_refs),
            workspace_file_count=sum(
                member.member_type is not LHWorkspaceExportMemberTypeV2.DIRECTORY for member in members
            ),
            workspace_total_bytes=sum(
                member.size_bytes
                for member in members
                if member.member_type is not LHWorkspaceExportMemberTypeV2.DIRECTORY
            ),
            workspace_sha256=_workspace_sha256(members),
            failure_code=None,
            result_sha256="0" * 64,
        )
        return _finalize_result(value)

    @classmethod
    def blocked(
        cls,
        *,
        request: LHWorkspaceExportRequestV2,
        failure_code: LHWorkspaceExportFailureCodeV2,
    ) -> LHWorkspaceExportResultV2:
        value = cls(
            result_id="lh-workspace-export-result://pending",
            request_ref=lh_workspace_export_request_ref(request),
            outcome=LHWorkspaceExportOutcomeV2.BLOCKED,
            failure_code=failure_code,
            result_sha256="0" * 64,
        )
        return _finalize_result(value)


class LHWorkspaceExportFacade(Protocol):
    def export(
        self,
        request: LHWorkspaceExportRequestV2,
        *,
        destination: Path,
    ) -> LHWorkspaceExportResultV2: ...


def lh_workspace_export_request_carried_sha256(
    value: LHWorkspaceExportRequestV2,
) -> str:
    return _payload_sha256(
        value.model_dump(
            mode="json",
            exclude={"request_id", "request_sha256"},
            exclude_none=False,
        )
    )


def lh_workspace_export_request_ref(
    value: LHWorkspaceExportRequestV2,
) -> FacadeObjectRef:
    validate_lh_workspace_export_request_identity(value)
    return FacadeObjectRef(
        object_type="lh-workspace-export-request",
        object_id=value.request_id,
        object_version="v2",
        object_sha256=value.request_sha256,
    )


def lh_workspace_export_result_carried_sha256(
    value: LHWorkspaceExportResultV2,
) -> str:
    return _payload_sha256(
        value.model_dump(
            mode="json",
            exclude={"result_id", "result_sha256"},
            exclude_none=False,
        )
    )


def lh_workspace_export_result_ref(
    value: LHWorkspaceExportResultV2,
) -> FacadeObjectRef:
    validate_lh_workspace_export_result_identity(value)
    return FacadeObjectRef(
        object_type="lh-workspace-export-result",
        object_id=value.result_id,
        object_version="v2",
        object_sha256=value.result_sha256,
    )


def validate_lh_workspace_export_request_identity(
    value: LHWorkspaceExportRequestV2,
) -> None:
    _validate_identity(
        object_id=value.request_id,
        object_sha256=value.request_sha256,
        prefix="lh-workspace-export-request",
        observed=lh_workspace_export_request_carried_sha256(value),
        allow_pending=False,
    )


def validate_lh_workspace_export_result_identity(
    value: LHWorkspaceExportResultV2,
) -> None:
    _validate_identity(
        object_id=value.result_id,
        object_sha256=value.result_sha256,
        prefix="lh-workspace-export-result",
        observed=lh_workspace_export_result_carried_sha256(value),
        allow_pending=False,
    )


def _finalize_result(
    value: LHWorkspaceExportResultV2,
) -> LHWorkspaceExportResultV2:
    digest = lh_workspace_export_result_carried_sha256(value)
    return LHWorkspaceExportResultV2.model_validate(
        value.model_copy(
            update={
                "result_id": f"lh-workspace-export-result://sha256/{digest}",
                "result_sha256": digest,
            }
        ).model_dump(mode="python")
    )


def _workspace_sha256(
    members: tuple[LHWorkspaceExportMemberV2, ...],
) -> str:
    return _payload_sha256([member.model_dump(mode="json", exclude_none=False) for member in members])


def _member_key(
    value: LHWorkspaceExportMemberV2,
) -> tuple[str, str, str, str]:
    return (
        value.container_ref.object_id if value.container_ref is not None else "",
        value.normalized_path,
        value.member_type.value,
        value.output_ref.object_id,
    )


def _sorted_members(
    values: tuple[LHWorkspaceExportMemberV2, ...],
) -> tuple[LHWorkspaceExportMemberV2, ...]:
    return tuple(sorted(values, key=_member_key))


def _require_sorted_unique_members(
    values: tuple[LHWorkspaceExportMemberV2, ...],
) -> None:
    if values != _sorted_members(values) or len({_member_key(value) for value in values}) != len(values):
        raise ValueError("workspace export members must be sorted and unique")


def _ref_key(
    value: FacadeObjectRef,
) -> tuple[str, str, str, str]:
    return (
        value.object_type,
        value.object_id,
        value.object_version,
        value.object_sha256,
    )


def _sorted_refs(
    values: tuple[FacadeObjectRef, ...],
) -> tuple[FacadeObjectRef, ...]:
    by_key = {_ref_key(value): value for value in values}
    return tuple(by_key[key] for key in sorted(by_key))


def _require_sorted_unique_refs(
    label: str,
    values: tuple[FacadeObjectRef, ...],
) -> None:
    if values != _sorted_refs(values) or len(values) != len(set(values)):
        raise ValueError(f"{label} must be sorted and unique")


def _require_ref(
    value: FacadeObjectRef,
    object_type: str,
    object_version: str,
    field_name: str,
) -> None:
    if value.object_type != object_type or value.object_version != object_version:
        raise ValueError(f"{field_name} must reference {object_type} {object_version}")


def _validate_identity(
    *,
    object_id: str,
    object_sha256: str,
    prefix: str,
    observed: str,
    allow_pending: bool = True,
) -> None:
    if allow_pending and object_id == f"{prefix}://pending" and object_sha256 == "0" * 64:
        return
    if object_sha256 != observed or object_id != f"{prefix}://sha256/{observed}":
        raise ValueError(f"{prefix} identity is stale")


def _payload_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _parse_enum[EnumT: StrEnum](
    value: object,
    enum_type: type[EnumT],
    field_name: str,
) -> EnumT:
    if isinstance(value, enum_type):
        return value
    if isinstance(value, str):
        return enum_type(value)
    raise TypeError(f"{field_name} must be a {enum_type.__name__}")


__all__ = [
    "LH_RELEASE_EXPORT_POLICY_VERSION",
    "LHWorkspaceExportFacade",
    "LHWorkspaceExportFailureCodeV2",
    "LHWorkspaceExportMemberTypeV2",
    "LHWorkspaceExportMemberV2",
    "LHWorkspaceExportOutcomeV2",
    "LHWorkspaceExportRequestV2",
    "LHWorkspaceExportResultV2",
    "lh_workspace_export_request_carried_sha256",
    "lh_workspace_export_request_ref",
    "lh_workspace_export_result_carried_sha256",
    "lh_workspace_export_result_ref",
    "validate_lh_workspace_export_request_identity",
    "validate_lh_workspace_export_result_identity",
]
