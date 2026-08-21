from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError

from env_mock_agent.facade.contracts import FacadeObjectRef
from env_mock_agent.facade.release_export_v2 import (
    LHWorkspaceExportFailureCodeV2,
    LHWorkspaceExportMemberTypeV2,
    LHWorkspaceExportMemberV2,
    LHWorkspaceExportOutcomeV2,
    LHWorkspaceExportRequestV2,
    LHWorkspaceExportResultV2,
    lh_workspace_export_request_ref,
    validate_lh_workspace_export_result_identity,
)


def _ref(object_type: str, suffix: str) -> FacadeObjectRef:
    digest = hashlib.sha256(f"{object_type}:{suffix}".encode()).hexdigest()
    return FacadeObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://r7-09/{suffix}",
        object_version="v2",
        object_sha256=digest,
    )


def _member() -> LHWorkspaceExportMemberV2:
    output = _ref("attachment-output", "input")
    return LHWorkspaceExportMemberV2.create(
        normalized_path="inputs/source.txt",
        member_type=LHWorkspaceExportMemberTypeV2.FILE,
        media_type="text/plain",
        size_bytes=12,
        content_sha256=output.object_sha256,
        container_ref=None,
        output_ref=output,
    )


def _request() -> LHWorkspaceExportRequestV2:
    member = _member()
    return LHWorkspaceExportRequestV2.create(
        item_id="item://r7-09/a",
        final_package_manifest_ref=_ref(
            "final-package-manifest",
            "a",
        ),
        package_sha256=hashlib.sha256(b"package").hexdigest(),
        output_refs=(member.output_ref,),
        members=(member,),
        max_member_count=32,
        max_total_bytes=1024,
    )


def test_release_export_contracts_are_strict_frozen_and_content_addressed() -> None:
    request = _request()
    result = LHWorkspaceExportResultV2.exported(
        request=request,
        observed_members=request.members,
        copied_output_refs=request.output_refs,
    )

    assert result.outcome is LHWorkspaceExportOutcomeV2.EXPORTED
    assert result.request_ref == lh_workspace_export_request_ref(request)
    assert result.workspace_file_count == 1
    assert result.workspace_total_bytes == 12
    validate_lh_workspace_export_result_identity(result)

    with pytest.raises(ValidationError):
        LHWorkspaceExportRequestV2.model_validate(
            {
                **request.model_dump(mode="python"),
                "unknown": True,
            }
        )
    with pytest.raises(ValidationError):
        request.max_member_count = 99  # type: ignore[misc]


def test_release_export_result_matrix_and_path_safety_fail_closed() -> None:
    request = _request()
    blocked = LHWorkspaceExportResultV2.blocked(
        request=request,
        failure_code=(LHWorkspaceExportFailureCodeV2.SOURCE_HASH_MISMATCH),
    )
    assert blocked.outcome is LHWorkspaceExportOutcomeV2.BLOCKED
    assert blocked.observed_members == ()
    assert blocked.workspace_sha256 is None

    with pytest.raises(ValidationError, match="relative"):
        LHWorkspaceExportMemberV2.create(
            normalized_path="../escape.txt",
            member_type=LHWorkspaceExportMemberTypeV2.FILE,
            media_type="text/plain",
            size_bytes=1,
            content_sha256="a" * 64,
            container_ref=None,
            output_ref=_ref("attachment-output", "escape"),
        )
    with pytest.raises(ValidationError, match="budget"):
        LHWorkspaceExportRequestV2.create(
            item_id="item://r7-09/a",
            final_package_manifest_ref=_ref(
                "final-package-manifest",
                "a",
            ),
            package_sha256="a" * 64,
            output_refs=(_member().output_ref,),
            members=(_member(),),
            max_member_count=1,
            max_total_bytes=1,
        )


def test_export_result_rejects_partial_or_stale_success() -> None:
    request = _request()
    with pytest.raises((ValidationError, ValueError), match="exact"):
        LHWorkspaceExportResultV2.exported(
            request=request,
            observed_members=(),
            copied_output_refs=request.output_refs,
        )

    result = LHWorkspaceExportResultV2.exported(
        request=request,
        observed_members=request.members,
        copied_output_refs=request.output_refs,
    )
    stale = result.model_copy(update={"workspace_total_bytes": 13})
    with pytest.raises(ValueError, match="identity"):
        validate_lh_workspace_export_result_identity(stale)


def test_empty_workspace_is_an_exact_export() -> None:
    request = LHWorkspaceExportRequestV2.create(
        item_id="item://r7-09/empty",
        final_package_manifest_ref=_ref(
            "final-package-manifest",
            "empty",
        ),
        package_sha256=hashlib.sha256(b"[]").hexdigest(),
        output_refs=(),
        members=(),
        max_member_count=1,
        max_total_bytes=1,
    )

    result = LHWorkspaceExportResultV2.exported(
        request=request,
        observed_members=(),
        copied_output_refs=(),
    )

    assert result.workspace_file_count == 0
    assert result.workspace_total_bytes == 0
    assert result.workspace_sha256 is not None
    validate_lh_workspace_export_result_identity(result)


def test_directory_children_and_nested_directories_have_exact_shapes() -> None:
    output = _ref("attachment-output", "directory")
    child = LHWorkspaceExportMemberV2.create(
        normalized_path="inputs/project/readme.txt",
        member_type=LHWorkspaceExportMemberTypeV2.FILE,
        media_type="text/plain",
        size_bytes=8,
        content_sha256=hashlib.sha256(b"read me\n").hexdigest(),
        container_ref=None,
        output_ref=output,
    )
    nested_directory = LHWorkspaceExportMemberV2.create(
        normalized_path="nested/",
        member_type=LHWorkspaceExportMemberTypeV2.DIRECTORY,
        media_type=None,
        size_bytes=0,
        content_sha256=None,
        container_ref=output,
        output_ref=output,
    )

    assert child.content_sha256 != child.output_ref.object_sha256
    assert nested_directory.container_ref == output
