from __future__ import annotations

import hashlib
import io
import os
import zipfile
from pathlib import Path

import pytest

from env_mock_agent.facade.contracts import FacadeObjectRef
from env_mock_agent.facade.release_export_adapter import (
    LHWorkspaceExportMaterial,
    MappingLHWorkspaceExportMaterialResolver,
    RegistryLHWorkspaceExportFacade,
)
from env_mock_agent.facade.release_export_v2 import (
    LHWorkspaceExportFailureCodeV2,
    LHWorkspaceExportMemberTypeV2,
    LHWorkspaceExportMemberV2,
    LHWorkspaceExportOutcomeV2,
    LHWorkspaceExportRequestV2,
)
from env_mock_agent.providers.helpers import sha256_path


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _ref(
    object_type: str,
    suffix: str,
    *,
    digest: str,
) -> FacadeObjectRef:
    return FacadeObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://r7-09/{suffix}",
        object_version="v2",
        object_sha256=digest,
    )


def _member(
    *,
    path: str,
    member_type: LHWorkspaceExportMemberTypeV2,
    output_ref: FacadeObjectRef,
    size: int = 0,
    digest: str | None = None,
    media_type: str | None = None,
    container_ref: FacadeObjectRef | None = None,
) -> LHWorkspaceExportMemberV2:
    return LHWorkspaceExportMemberV2.create(
        normalized_path=path,
        member_type=member_type,
        media_type=media_type,
        size_bytes=size,
        content_sha256=digest,
        container_ref=container_ref,
        output_ref=output_ref,
    )


def _request(
    *,
    output_refs: tuple[FacadeObjectRef, ...],
    members: tuple[LHWorkspaceExportMemberV2, ...],
    max_nested_depth: int = 8,
    max_file_bytes: int = 500_000,
    max_compression_ratio: int = 100,
) -> LHWorkspaceExportRequestV2:
    return LHWorkspaceExportRequestV2.create(
        item_id="item://r7-09/export",
        final_package_manifest_ref=_ref(
            "final-package-manifest",
            "export",
            digest=_digest(b"manifest"),
        ),
        package_sha256=_digest(b"package"),
        output_refs=output_refs,
        members=members,
        max_member_count=100,
        max_total_bytes=1_000_000,
        max_file_bytes=max_file_bytes,
        max_nested_depth=max_nested_depth,
        max_compression_ratio=max_compression_ratio,
    )


def _facade(
    *,
    output_ref: FacadeObjectRef,
    source: Path,
    approved_root: Path | None = None,
) -> RegistryLHWorkspaceExportFacade:
    return RegistryLHWorkspaceExportFacade(
        resolver=MappingLHWorkspaceExportMaterialResolver(
            outputs={output_ref.object_id: source},
            approved_roots=({output_ref.object_id: approved_root} if approved_root is not None else None),
        )
    )


def test_exports_exact_regular_file(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("safe input\n", encoding="utf-8")
    output_ref = _ref(
        "attachment-output",
        "file",
        digest=sha256_path(source),
    )
    member = _member(
        path="inputs/source.txt",
        member_type=LHWorkspaceExportMemberTypeV2.FILE,
        output_ref=output_ref,
        size=source.stat().st_size,
        digest=output_ref.object_sha256,
        media_type="text/plain",
    )
    request = _request(output_refs=(output_ref,), members=(member,))

    result = _facade(output_ref=output_ref, source=source).export(
        request,
        destination=tmp_path / "workspace",
    )

    assert result.outcome is LHWorkspaceExportOutcomeV2.EXPORTED
    assert result.observed_members == request.members
    assert (tmp_path / "workspace/inputs/source.txt").read_text() == "safe input\n"


def test_exports_directory_and_preserves_nested_zip_inventory(
    tmp_path: Path,
) -> None:
    source = tmp_path / "project"
    source.mkdir()
    readme = source / "readme.txt"
    readme.write_text("input readme\n", encoding="utf-8")
    archive_path = source / "nested.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("folder/", b"")
        archive.writestr("folder/inner.txt", b"nested input\n")
    output_ref = _ref(
        "attachment-output",
        "directory",
        digest=sha256_path(source),
    )
    members = (
        _member(
            path="inputs/project",
            member_type=LHWorkspaceExportMemberTypeV2.DIRECTORY,
            output_ref=output_ref,
        ),
        _member(
            path="inputs/project/readme.txt",
            member_type=LHWorkspaceExportMemberTypeV2.FILE,
            output_ref=output_ref,
            size=readme.stat().st_size,
            digest=sha256_path(readme),
            media_type="text/plain",
        ),
        _member(
            path="inputs/project/nested.zip",
            member_type=LHWorkspaceExportMemberTypeV2.FILE,
            output_ref=output_ref,
            size=archive_path.stat().st_size,
            digest=sha256_path(archive_path),
            media_type="application/zip",
        ),
        _member(
            path="folder/",
            member_type=LHWorkspaceExportMemberTypeV2.DIRECTORY,
            output_ref=output_ref,
            container_ref=output_ref,
        ),
        _member(
            path="folder/inner.txt",
            member_type=LHWorkspaceExportMemberTypeV2.NESTED_MEMBER,
            output_ref=output_ref,
            size=len(b"nested input\n"),
            digest=_digest(b"nested input\n"),
            media_type="text/plain",
            container_ref=output_ref,
        ),
    )
    request = _request(output_refs=(output_ref,), members=members)

    result = _facade(output_ref=output_ref, source=source).export(
        request,
        destination=tmp_path / "workspace",
    )

    assert result.outcome is LHWorkspaceExportOutcomeV2.EXPORTED
    copied = tmp_path / "workspace/inputs/project"
    assert sha256_path(copied) == output_ref.object_sha256
    assert result.workspace_file_count == 3


def test_exports_exact_empty_workspace(tmp_path: Path) -> None:
    request = _request(output_refs=(), members=())
    facade = RegistryLHWorkspaceExportFacade(resolver=MappingLHWorkspaceExportMaterialResolver(outputs={}))

    result = facade.export(
        request,
        destination=tmp_path / "workspace",
    )

    assert result.outcome is LHWorkspaceExportOutcomeV2.EXPORTED
    assert list((tmp_path / "workspace").iterdir()) == []


@pytest.mark.parametrize(
    ("mutation", "failure"),
    [
        ("source-hash", LHWorkspaceExportFailureCodeV2.SOURCE_HASH_MISMATCH),
        ("inventory", LHWorkspaceExportFailureCodeV2.INVENTORY_MISMATCH),
        ("destination", LHWorkspaceExportFailureCodeV2.DESTINATION_PATH_VIOLATION),
        ("source-root", LHWorkspaceExportFailureCodeV2.SOURCE_PATH_VIOLATION),
        ("symlink", LHWorkspaceExportFailureCodeV2.SYMLINK_UNSUPPORTED),
        ("special", LHWorkspaceExportFailureCodeV2.SPECIAL_FILE_UNSUPPORTED),
        ("container", LHWorkspaceExportFailureCodeV2.CONTAINER_UNSUPPORTED),
    ],
)
def test_export_failures_are_closed_and_content_free(
    tmp_path: Path,
    mutation: str,
    failure: LHWorkspaceExportFailureCodeV2,
) -> None:
    actual = tmp_path / "actual.txt"
    actual.write_text("safe input\n", encoding="utf-8")
    source = actual
    approved_root: Path | None = None
    output_digest = sha256_path(actual)
    member_size = actual.stat().st_size
    destination = tmp_path / "workspace"
    suffix = "file"

    if mutation == "source-hash":
        output_digest = _digest(b"different")
    elif mutation == "inventory":
        member_size += 1
    elif mutation == "destination":
        destination.mkdir()
        (destination / "occupied").write_text("x", encoding="utf-8")
    elif mutation == "source-root":
        approved_root = tmp_path / "approved"
        approved_root.mkdir()
    elif mutation == "symlink":
        source = tmp_path / "link.txt"
        source.symlink_to(actual)
    elif mutation == "special":
        source = tmp_path / "pipe"
        os.mkfifo(source)
    elif mutation == "container":
        source = tmp_path / "input.tar"
        source.write_text("not an allowed container", encoding="utf-8")
        output_digest = sha256_path(source)
        member_size = source.stat().st_size
        suffix = "container"

    output_ref = _ref(
        "attachment-output",
        suffix,
        digest=output_digest,
    )
    member = _member(
        path=f"inputs/{source.name}",
        member_type=LHWorkspaceExportMemberTypeV2.FILE,
        output_ref=output_ref,
        size=member_size,
        digest=sha256_path(actual) if mutation == "source-hash" else output_digest,
        media_type="text/plain",
    )
    request = _request(output_refs=(output_ref,), members=(member,))

    result = _facade(
        output_ref=output_ref,
        source=source,
        approved_root=approved_root,
    ).export(
        request,
        destination=destination,
    )

    assert result.outcome is LHWorkspaceExportOutcomeV2.BLOCKED
    assert result.failure_code is failure
    serialized = result.model_dump_json()
    assert str(source) not in serialized
    assert "safe input" not in serialized


def test_nested_container_depth_limit_blocks_before_partial_success(
    tmp_path: Path,
) -> None:
    inner_bytes = _zip_bytes({"inner.txt": b"safe input\n"})
    outer = tmp_path / "outer.zip"
    with zipfile.ZipFile(outer, "w") as archive:
        archive.writestr("inner.zip", inner_bytes)
    output_ref = _ref(
        "attachment-output",
        "nested-depth",
        digest=sha256_path(outer),
    )
    member = _member(
        path="inputs/outer.zip",
        member_type=LHWorkspaceExportMemberTypeV2.FILE,
        output_ref=output_ref,
        size=outer.stat().st_size,
        digest=output_ref.object_sha256,
        media_type="application/zip",
    )
    request = _request(
        output_refs=(output_ref,),
        members=(member,),
        max_nested_depth=1,
    )

    result = _facade(output_ref=output_ref, source=outer).export(
        request,
        destination=tmp_path / "workspace",
    )

    assert result.outcome is LHWorkspaceExportOutcomeV2.BLOCKED
    assert result.failure_code is LHWorkspaceExportFailureCodeV2.EXPORT_LIMIT_EXCEEDED
    assert result.observed_members == ()
    assert result.copied_output_refs == ()


def test_missing_and_mismatched_material_are_closed(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.txt"
    source.write_text("safe input\n", encoding="utf-8")
    output_ref = _ref(
        "attachment-output",
        "material",
        digest=sha256_path(source),
    )
    member = _member(
        path="inputs/source.txt",
        member_type=LHWorkspaceExportMemberTypeV2.FILE,
        output_ref=output_ref,
        size=source.stat().st_size,
        digest=output_ref.object_sha256,
        media_type="text/plain",
    )
    request = _request(output_refs=(output_ref,), members=(member,))
    missing = RegistryLHWorkspaceExportFacade(
        resolver=MappingLHWorkspaceExportMaterialResolver(outputs={})
    ).export(
        request,
        destination=tmp_path / "missing-workspace",
    )
    assert missing.failure_code is LHWorkspaceExportFailureCodeV2.MATERIAL_NOT_FOUND

    class _MismatchedResolver:
        def resolve_output(
            self,
            request: LHWorkspaceExportRequestV2,
            selected: FacadeObjectRef,
        ) -> LHWorkspaceExportMaterial:
            del request, selected
            return LHWorkspaceExportMaterial(
                output_ref=_ref(
                    "attachment-output",
                    "other",
                    digest=output_ref.object_sha256,
                ),
                source_path=source,
                approved_root=tmp_path,
            )

    mismatched = RegistryLHWorkspaceExportFacade(resolver=_MismatchedResolver()).export(
        request,
        destination=tmp_path / "mismatched-workspace",
    )
    assert mismatched.failure_code is LHWorkspaceExportFailureCodeV2.MATERIAL_MISMATCH


def test_file_size_and_zip_safety_limits_block(
    tmp_path: Path,
) -> None:
    source = tmp_path / "large.txt"
    source.write_bytes(b"0123456789")
    output_ref = _ref(
        "attachment-output",
        "large",
        digest=sha256_path(source),
    )
    member = _member(
        path="inputs/large.txt",
        member_type=LHWorkspaceExportMemberTypeV2.FILE,
        output_ref=output_ref,
        size=source.stat().st_size,
        digest=output_ref.object_sha256,
        media_type="text/plain",
    )
    limited = _request(
        output_refs=(output_ref,),
        members=(member,),
        max_file_bytes=5,
    )
    result = _facade(output_ref=output_ref, source=source).export(
        limited,
        destination=tmp_path / "large-workspace",
    )
    assert result.failure_code is LHWorkspaceExportFailureCodeV2.EXPORT_LIMIT_EXCEEDED

    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(
        archive_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        archive.writestr("../escape.txt", b"escape")
    archive_ref = _ref(
        "attachment-output",
        "unsafe-zip",
        digest=sha256_path(archive_path),
    )
    archive_member = _member(
        path="inputs/unsafe.zip",
        member_type=LHWorkspaceExportMemberTypeV2.FILE,
        output_ref=archive_ref,
        size=archive_path.stat().st_size,
        digest=archive_ref.object_sha256,
        media_type="application/zip",
    )
    unsafe_request = _request(
        output_refs=(archive_ref,),
        members=(archive_member,),
    )
    unsafe = _facade(
        output_ref=archive_ref,
        source=archive_path,
    ).export(
        unsafe_request,
        destination=tmp_path / "unsafe-workspace",
    )
    assert unsafe.failure_code is LHWorkspaceExportFailureCodeV2.SOURCE_PATH_VIOLATION


def test_zip_compression_ratio_limit_blocks(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "compressed.zip"
    with zipfile.ZipFile(
        archive_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        archive.writestr("repeated.txt", b"a" * 10_000)
    output_ref = _ref(
        "attachment-output",
        "compressed",
        digest=sha256_path(archive_path),
    )
    member = _member(
        path="inputs/compressed.zip",
        member_type=LHWorkspaceExportMemberTypeV2.FILE,
        output_ref=output_ref,
        size=archive_path.stat().st_size,
        digest=output_ref.object_sha256,
        media_type="application/zip",
    )
    request = _request(
        output_refs=(output_ref,),
        members=(member,),
        max_compression_ratio=2,
    )

    result = _facade(output_ref=output_ref, source=archive_path).export(
        request,
        destination=tmp_path / "workspace",
    )

    assert result.failure_code is LHWorkspaceExportFailureCodeV2.EXPORT_LIMIT_EXCEEDED


def _zip_bytes(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()
