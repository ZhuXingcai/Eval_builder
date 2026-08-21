from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pytest

from env_mock_agent.facade import FacadeObjectRef
from env_mock_agent.facade.validation_adapter import (
    MappingAttachmentValidationMaterialResolver,
    ProviderValidationMaterial,
    RegistryAttachmentValidationFacade,
)
from env_mock_agent.facade.validation_v2 import (
    ATTACHMENT_VALIDATION_POLICY_VERSION,
    AttachmentInventoryMemberTypeV2,
    AttachmentValidationFailureCodeV2,
    AttachmentValidationRequestV2,
    AttachmentValidationStatusV2,
    attachment_validation_request_carried_sha256,
)
from env_mock_agent.schemas import ArtifactPlan

HASH = "a" * 64


def _ref(
    object_type: str,
    suffix: str,
    *,
    digest: str = HASH,
) -> FacadeObjectRef:
    return FacadeObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://{suffix}",
        object_version="v2",
        object_sha256=digest,
    )


def _request(
    *,
    output_ref: FacadeObjectRef,
    logical_path: str,
    media_type: str,
    max_inventory_members: int = 20_000,
) -> AttachmentValidationRequestV2:
    request = AttachmentValidationRequestV2(
        validation_request_id="attachment-validation-request://pending",
        artifact_id="artifact://input",
        artifact_result_ref=_ref("artifact-build-result", "input"),
        execution_request_ref=_ref("attachment-execution-request", "input"),
        execution_result_ref=_ref("attachment-execution-result", "input"),
        build_spec_ref=_ref("artifact-build-spec", "input"),
        producer_task_view_ref=_ref("producer-task-view", "current"),
        output_ref=output_ref,
        output_sha256=output_ref.object_sha256,
        logical_path=logical_path,
        media_type=media_type,
        declared_validator_ids=("text-validator",),
        configured_pii_rules=(),
        leakage_reference_set_ref=_ref(
            "prompt-leakage-reference-set",
            "current",
        ),
        leakage_fingerprints=(),
        complete_leakage_categories=(),
        forbidden_output_values=("original final answer",),
        scan_limits={"max_inventory_members": max_inventory_members},
        policy_version=ATTACHMENT_VALIDATION_POLICY_VERSION,
        idempotency_key="attachment-validation-idempotency://input",
        validation_request_sha256=HASH,
    )
    digest = attachment_validation_request_carried_sha256(request)
    return request.model_copy(
        update={
            "validation_request_id": (f"attachment-validation-request://sha256/{digest}"),
            "validation_request_sha256": digest,
        }
    )


def _material(logical_path: str) -> ProviderValidationMaterial:
    return ProviderValidationMaterial(
        plan=ArtifactPlan(
            artifact_id="artifact://input",
            dependency_id="attachment-dependency://input",
            relative_path=logical_path,
            asset_type="txt",
            content_contract={"min_characters": 1},
            render_contract={},
            validators=["text-validator"],
        )
    )


async def _validate(
    path: Path,
    *,
    logical_path: str,
    media_type: str,
):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    output_ref = _ref("attachment-output", "input", digest=digest)
    request = _request(
        output_ref=output_ref,
        logical_path=logical_path,
        media_type=media_type,
    )
    resolver = MappingAttachmentValidationMaterialResolver(
        outputs={output_ref.object_id: path},
        materials={request.build_spec_ref.object_id: _material(logical_path)},
    )
    result = await RegistryAttachmentValidationFacade(
        resolver=resolver,
    ).validate(request)
    return request, result


@pytest.mark.asyncio
async def test_file_inventory_binds_exact_logical_path_size_and_hash(
    tmp_path: Path,
) -> None:
    path = tmp_path / "source.txt"
    path.write_text("Neutral input-state facts.", encoding="utf-8")

    request, result = await _validate(
        path,
        logical_path="inputs/source.txt",
        media_type="text/plain",
    )

    assert result.status is AttachmentValidationStatusV2.PASSED
    assert len(result.inventory_members) == 1
    member = result.inventory_members[0]
    assert member.member_type is AttachmentInventoryMemberTypeV2.FILE
    assert member.normalized_path == "inputs/source.txt"
    assert member.size_bytes == path.stat().st_size
    assert member.content_sha256 == request.output_sha256


@pytest.mark.asyncio
async def test_symlink_output_blocks_without_exposing_physical_path(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.txt"
    target.write_text("Neutral.", encoding="utf-8")
    linked = tmp_path / "linked.txt"
    linked.symlink_to(target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    output_ref = _ref("attachment-output", "linked", digest=digest)
    request = _request(
        output_ref=output_ref,
        logical_path="inputs/linked.txt",
        media_type="text/plain",
    )
    resolver = MappingAttachmentValidationMaterialResolver(
        outputs={output_ref.object_id: linked},
        materials={request.build_spec_ref.object_id: _material("inputs/linked.txt")},
    )

    result = await RegistryAttachmentValidationFacade(
        resolver=resolver,
    ).validate(request)

    assert result.status is AttachmentValidationStatusV2.BLOCKED
    assert result.failure_code is AttachmentValidationFailureCodeV2.SYMLINK_UNSUPPORTED
    assert str(tmp_path) not in str(result.model_dump(mode="json"))


@pytest.mark.asyncio
async def test_zip_inventory_rejects_nested_parent_traversal(
    tmp_path: Path,
) -> None:
    path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("../escape.txt", "not allowed")

    _, result = await _validate(
        path,
        logical_path="inputs/unsafe.zip",
        media_type="application/zip",
    )

    assert result.status is AttachmentValidationStatusV2.BLOCKED
    assert result.failure_code is (AttachmentValidationFailureCodeV2.PACKAGE_PATH_INVALID)


@pytest.mark.asyncio
async def test_encrypted_zip_member_blocks_inventory(
    tmp_path: Path,
) -> None:
    path = tmp_path / "encrypted-flag.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("safe.txt", "neutral")
    data = bytearray(path.read_bytes())
    for signature in (b"PK\x03\x04", b"PK\x01\x02"):
        offset = 0
        while True:
            index = data.find(signature, offset)
            if index < 0:
                break
            flag_offset = index + 6 if signature == b"PK\x03\x04" else index + 8
            flags = int.from_bytes(data[flag_offset : flag_offset + 2], "little")
            data[flag_offset : flag_offset + 2] = (flags | 1).to_bytes(2, "little")
            offset = index + 4
    path.write_bytes(data)

    _, result = await _validate(
        path,
        logical_path="inputs/encrypted-flag.zip",
        media_type="application/zip",
    )

    assert result.status is AttachmentValidationStatusV2.BLOCKED
    assert result.failure_code is (AttachmentValidationFailureCodeV2.CONTAINER_ENCRYPTED)


@pytest.mark.asyncio
async def test_truncated_zip_blocks_inventory(
    tmp_path: Path,
) -> None:
    path = tmp_path / "truncated.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("safe.txt", "neutral")
    path.write_bytes(path.read_bytes()[:-8])

    _, result = await _validate(
        path,
        logical_path="inputs/truncated.zip",
        media_type="application/zip",
    )

    assert result.status is AttachmentValidationStatusV2.BLOCKED
    assert result.failure_code is (AttachmentValidationFailureCodeV2.CONTAINER_TRUNCATED)


@pytest.mark.asyncio
async def test_inventory_member_limit_blocks_without_partial_success(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "a.txt").write_text("a", encoding="utf-8")
    (root / "b.txt").write_text("b", encoding="utf-8")
    digest = hashlib.sha256()
    for child in sorted(root.iterdir()):
        digest.update(child.name.encode())
        digest.update(child.read_bytes())
    output_ref = _ref(
        "attachment-output",
        "project",
        digest=digest.hexdigest(),
    )
    request = _request(
        output_ref=output_ref,
        logical_path="inputs/project",
        media_type="application/x-directory",
        max_inventory_members=2,
    )
    material = _material("inputs/project")
    material.plan.asset_type = "project"

    result = await RegistryAttachmentValidationFacade(
        resolver=MappingAttachmentValidationMaterialResolver(
            outputs={output_ref.object_id: root},
            materials={request.build_spec_ref.object_id: material},
        )
    ).validate(request)

    assert result.status is AttachmentValidationStatusV2.BLOCKED
    assert result.failure_code is (AttachmentValidationFailureCodeV2.SCAN_LIMIT_EXCEEDED)
    assert result.inventory_members == ()


@pytest.mark.asyncio
async def test_valid_zip_enumerates_nested_members(
    tmp_path: Path,
) -> None:
    path = tmp_path / "safe.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("folder/", "")
        archive.writestr("folder/safe.txt", "neutral input")

    _, result = await _validate(
        path,
        logical_path="inputs/safe.zip",
        media_type="application/zip",
    )

    assert result.status is not AttachmentValidationStatusV2.BLOCKED
    assert [(item.member_type.value, item.normalized_path) for item in result.inventory_members] == [
        ("FILE", "inputs/safe.zip"),
        ("DIRECTORY", "folder/"),
        ("NESTED_MEMBER", "folder/safe.txt"),
    ]
    assert all(item.container_ref is not None for item in result.inventory_members[1:])


@pytest.mark.asyncio
async def test_duplicate_zip_member_blocks_inventory(
    tmp_path: Path,
) -> None:
    path = tmp_path / "duplicate.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("same.txt", "first")
        archive.writestr("same.txt", "second")

    _, result = await _validate(
        path,
        logical_path="inputs/duplicate.zip",
        media_type="application/zip",
    )

    assert result.status is AttachmentValidationStatusV2.BLOCKED
    assert result.failure_code is (AttachmentValidationFailureCodeV2.PACKAGE_PATH_COLLISION)


@pytest.mark.asyncio
async def test_unsupported_container_blocks_inventory(
    tmp_path: Path,
) -> None:
    path = tmp_path / "archive.tar"
    path.write_bytes(b"not-a-supported-container")

    _, result = await _validate(
        path,
        logical_path="inputs/archive.tar",
        media_type="application/x-tar",
    )

    assert result.status is AttachmentValidationStatusV2.BLOCKED
    assert result.failure_code is (AttachmentValidationFailureCodeV2.CONTAINER_UNSUPPORTED)


@pytest.mark.asyncio
async def test_unscannable_binary_blocks_inventory(
    tmp_path: Path,
) -> None:
    path = tmp_path / "opaque.bin"
    path.write_bytes(b"\x00\xff\x00\xfe")

    _, result = await _validate(
        path,
        logical_path="inputs/opaque.bin",
        media_type="application/octet-stream",
    )

    assert result.status is AttachmentValidationStatusV2.BLOCKED
    assert result.failure_code is (AttachmentValidationFailureCodeV2.CONTENT_UNSCANNABLE)


@pytest.mark.asyncio
async def test_hard_link_alias_blocks_directory_inventory(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    source = root / "source.txt"
    source.write_text("neutral", encoding="utf-8")
    (root / "alias.txt").hardlink_to(source)
    digest = hashlib.sha256()
    for child in sorted(root.iterdir()):
        digest.update(child.name.encode())
        digest.update(child.read_bytes())
    output_ref = _ref(
        "attachment-output",
        "hard-link-project",
        digest=digest.hexdigest(),
    )
    request = _request(
        output_ref=output_ref,
        logical_path="inputs/project",
        media_type="application/x-directory",
    )
    material = _material("inputs/project")
    material.plan.asset_type = "project"

    result = await RegistryAttachmentValidationFacade(
        resolver=MappingAttachmentValidationMaterialResolver(
            outputs={output_ref.object_id: root},
            materials={request.build_spec_ref.object_id: material},
        )
    ).validate(request)

    assert result.status is AttachmentValidationStatusV2.BLOCKED
    assert result.failure_code is (AttachmentValidationFailureCodeV2.PACKAGE_PATH_COLLISION)
