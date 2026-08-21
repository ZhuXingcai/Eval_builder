from __future__ import annotations

import hashlib
import io
import mimetypes
import os
import stat
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Protocol

from env_mock_agent.facade.contracts import FacadeObjectRef
from env_mock_agent.facade.release_export_v2 import (
    LHWorkspaceExportFailureCodeV2,
    LHWorkspaceExportMemberTypeV2,
    LHWorkspaceExportMemberV2,
    LHWorkspaceExportRequestV2,
    LHWorkspaceExportResultV2,
    validate_lh_workspace_export_request_identity,
)
from env_mock_agent.providers.helpers import sha256_path
from env_mock_agent.runtimes.security import is_path_inside

_UNSUPPORTED_CONTAINER_SUFFIXES = frozenset(
    {
        ".7z",
        ".bz2",
        ".gz",
        ".rar",
        ".tar",
        ".tgz",
        ".xz",
    }
)


class LHWorkspaceExportMaterialError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        code: LHWorkspaceExportFailureCodeV2,
    ) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class LHWorkspaceExportMaterial:
    output_ref: FacadeObjectRef
    source_path: Path
    approved_root: Path


class LHWorkspaceExportMaterialResolver(Protocol):
    def resolve_output(
        self,
        request: LHWorkspaceExportRequestV2,
        output_ref: FacadeObjectRef,
    ) -> LHWorkspaceExportMaterial: ...


class MappingLHWorkspaceExportMaterialResolver:
    def __init__(
        self,
        *,
        outputs: Mapping[str, Path],
        approved_roots: Mapping[str, Path] | None = None,
    ) -> None:
        self._outputs = dict(outputs)
        self._approved_roots = dict(approved_roots or {})

    def resolve_output(
        self,
        request: LHWorkspaceExportRequestV2,
        output_ref: FacadeObjectRef,
    ) -> LHWorkspaceExportMaterial:
        del request
        try:
            path = self._outputs[output_ref.object_id].expanduser()
        except KeyError as exc:
            raise LHWorkspaceExportMaterialError(
                "LH workspace output is not registered",
                code=LHWorkspaceExportFailureCodeV2.MATERIAL_NOT_FOUND,
            ) from exc
        root = self._approved_roots.get(output_ref.object_id, path.parent).expanduser()
        return LHWorkspaceExportMaterial(
            output_ref=output_ref,
            source_path=path,
            approved_root=root,
        )


class RegistryLHWorkspaceExportFacade:
    def __init__(
        self,
        *,
        resolver: LHWorkspaceExportMaterialResolver,
    ) -> None:
        self._resolver = resolver

    def export(
        self,
        request: LHWorkspaceExportRequestV2,
        *,
        destination: Path,
    ) -> LHWorkspaceExportResultV2:
        validate_lh_workspace_export_request_identity(request)
        try:
            workspace = _prepare_destination(destination)
            observed: list[LHWorkspaceExportMemberV2] = []
            for output_ref in request.output_refs:
                material = self._resolver.resolve_output(request, output_ref)
                _validate_material(material, output_ref)
                expected = tuple(member for member in request.members if member.output_ref == output_ref)
                source_root_path = _source_root_path(
                    material.source_path,
                    expected,
                )
                source_members = _scan_material(
                    material.source_path,
                    root_normalized_path=source_root_path,
                    root_media_type=_root_media_type(expected),
                    output_ref=output_ref,
                    request=request,
                )
                if source_members != expected:
                    raise _ExportBlocked(LHWorkspaceExportFailureCodeV2.INVENTORY_MISMATCH)
                source_sha256 = _safe_sha256_path(material.source_path)
                if source_sha256 != output_ref.object_sha256:
                    raise _ExportBlocked(LHWorkspaceExportFailureCodeV2.SOURCE_HASH_MISMATCH)
                copied_root = _copy_material(
                    source=material.source_path,
                    destination=workspace,
                    normalized_path=source_root_path,
                    max_file_bytes=request.max_file_bytes,
                )
                copied_members = _scan_material(
                    copied_root,
                    root_normalized_path=source_root_path,
                    root_media_type=_root_media_type(expected),
                    output_ref=output_ref,
                    request=request,
                )
                if copied_members != expected:
                    raise _ExportBlocked(LHWorkspaceExportFailureCodeV2.INVENTORY_MISMATCH)
                if _safe_sha256_path(copied_root) != output_ref.object_sha256:
                    raise _ExportBlocked(LHWorkspaceExportFailureCodeV2.COPIED_HASH_MISMATCH)
                if _safe_sha256_path(material.source_path) != source_sha256:
                    raise _ExportBlocked(LHWorkspaceExportFailureCodeV2.SOURCE_HASH_MISMATCH)
                observed.extend(copied_members)
            observed_members = _sorted_members(tuple(observed))
            if observed_members != request.members:
                raise _ExportBlocked(LHWorkspaceExportFailureCodeV2.INVENTORY_MISMATCH)
            return LHWorkspaceExportResultV2.exported(
                request=request,
                observed_members=observed_members,
                copied_output_refs=request.output_refs,
            )
        except LHWorkspaceExportMaterialError as exc:
            return LHWorkspaceExportResultV2.blocked(
                request=request,
                failure_code=exc.code,
            )
        except _ExportBlocked as exc:
            return LHWorkspaceExportResultV2.blocked(
                request=request,
                failure_code=exc.code,
            )
        except OSError:
            return LHWorkspaceExportResultV2.blocked(
                request=request,
                failure_code=LHWorkspaceExportFailureCodeV2.MATERIAL_NOT_FOUND,
            )


class _ExportBlocked(RuntimeError):
    def __init__(
        self,
        code: LHWorkspaceExportFailureCodeV2,
    ) -> None:
        super().__init__(code.value)
        self.code = code


@dataclass(slots=True)
class _ScanState:
    request: LHWorkspaceExportRequestV2
    members: list[LHWorkspaceExportMemberV2] = field(default_factory=list)
    total_bytes: int = 0
    file_identities: set[tuple[int, int]] = field(default_factory=set)

    def add(self, member: LHWorkspaceExportMemberV2) -> None:
        self.members.append(member)
        if member.member_type is not LHWorkspaceExportMemberTypeV2.DIRECTORY:
            self.total_bytes += member.size_bytes
        if (
            len(self.members) > self.request.max_member_count
            or self.total_bytes > self.request.max_total_bytes
        ):
            raise _ExportBlocked(LHWorkspaceExportFailureCodeV2.EXPORT_LIMIT_EXCEEDED)


def _prepare_destination(destination: Path) -> Path:
    candidate = destination.expanduser()
    parent = candidate.parent
    if not parent.exists() or parent.is_symlink() or not parent.is_dir() or candidate.is_symlink():
        raise _ExportBlocked(LHWorkspaceExportFailureCodeV2.DESTINATION_PATH_VIOLATION)
    parent_resolved = parent.resolve(strict=True)
    if candidate.exists():
        if not candidate.is_dir() or any(candidate.iterdir()):
            raise _ExportBlocked(LHWorkspaceExportFailureCodeV2.DESTINATION_PATH_VIOLATION)
    else:
        candidate.mkdir(mode=0o700)
    resolved = candidate.resolve(strict=True)
    if not is_path_inside(parent_resolved, resolved):
        raise _ExportBlocked(LHWorkspaceExportFailureCodeV2.DESTINATION_PATH_VIOLATION)
    return resolved


def _validate_material(
    material: LHWorkspaceExportMaterial,
    output_ref: FacadeObjectRef,
) -> None:
    if material.output_ref != output_ref:
        raise LHWorkspaceExportMaterialError(
            "LH workspace material does not match the requested output",
            code=LHWorkspaceExportFailureCodeV2.MATERIAL_MISMATCH,
        )
    source = material.source_path.expanduser()
    root = material.approved_root.expanduser()
    if source.is_symlink():
        raise _ExportBlocked(LHWorkspaceExportFailureCodeV2.SYMLINK_UNSUPPORTED)
    if not source.exists() or not root.exists() or not root.is_dir():
        raise LHWorkspaceExportMaterialError(
            "LH workspace material is not available",
            code=LHWorkspaceExportFailureCodeV2.MATERIAL_NOT_FOUND,
        )
    root_resolved = root.resolve(strict=True)
    source_resolved = source.resolve(strict=True)
    if not is_path_inside(root_resolved, source_resolved):
        raise _ExportBlocked(LHWorkspaceExportFailureCodeV2.SOURCE_PATH_VIOLATION)
    mode = source.lstat().st_mode
    if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
        raise _ExportBlocked(LHWorkspaceExportFailureCodeV2.SPECIAL_FILE_UNSUPPORTED)


def _source_root_path(
    source: Path,
    expected: tuple[LHWorkspaceExportMemberV2, ...],
) -> str:
    top_level = tuple(member for member in expected if member.container_ref is None)
    if source.is_file():
        roots = tuple(
            member for member in top_level if member.member_type is LHWorkspaceExportMemberTypeV2.FILE
        )
        if len(roots) != 1:
            raise _ExportBlocked(LHWorkspaceExportFailureCodeV2.INVENTORY_MISMATCH)
        return roots[0].normalized_path
    directory_roots = tuple(
        member
        for member in top_level
        if member.member_type is LHWorkspaceExportMemberTypeV2.DIRECTORY
        and all(
            _is_same_or_descendant(
                member.normalized_path,
                candidate.normalized_path,
            )
            for candidate in top_level
        )
    )
    if len(directory_roots) != 1:
        raise _ExportBlocked(LHWorkspaceExportFailureCodeV2.INVENTORY_MISMATCH)
    return directory_roots[0].normalized_path.rstrip("/")


def _root_media_type(
    expected: tuple[LHWorkspaceExportMemberV2, ...],
) -> str | None:
    top_files = tuple(
        member
        for member in expected
        if member.container_ref is None and member.member_type is LHWorkspaceExportMemberTypeV2.FILE
    )
    return top_files[0].media_type if len(top_files) == 1 else None


def _scan_material(
    source: Path,
    *,
    root_normalized_path: str,
    root_media_type: str | None,
    output_ref: FacadeObjectRef,
    request: LHWorkspaceExportRequestV2,
) -> tuple[LHWorkspaceExportMemberV2, ...]:
    state = _ScanState(request=request)
    _scan_node(
        state,
        source,
        normalized_path=root_normalized_path,
        output_ref=output_ref,
        media_type=root_media_type,
        is_root=True,
    )
    return _sorted_members(tuple(state.members))


def _scan_node(
    state: _ScanState,
    path: Path,
    *,
    normalized_path: str,
    output_ref: FacadeObjectRef,
    media_type: str | None,
    is_root: bool,
) -> None:
    if path.is_symlink():
        raise _ExportBlocked(LHWorkspaceExportFailureCodeV2.SYMLINK_UNSUPPORTED)
    mode = path.lstat().st_mode
    if stat.S_ISDIR(mode):
        state.add(
            LHWorkspaceExportMemberV2.create(
                normalized_path=normalized_path,
                member_type=LHWorkspaceExportMemberTypeV2.DIRECTORY,
                media_type=None,
                size_bytes=0,
                content_sha256=None,
                container_ref=None,
                output_ref=output_ref,
            )
        )
        for child in sorted(path.iterdir(), key=lambda value: value.name):
            _scan_node(
                state,
                child,
                normalized_path=_join_relative(
                    normalized_path,
                    child.name,
                ),
                output_ref=output_ref,
                media_type=_media_type(child),
                is_root=False,
            )
        return
    if not stat.S_ISREG(mode):
        raise _ExportBlocked(LHWorkspaceExportFailureCodeV2.SPECIAL_FILE_UNSUPPORTED)
    file_stat = path.stat()
    identity = (file_stat.st_dev, file_stat.st_ino)
    if identity in state.file_identities:
        raise _ExportBlocked(LHWorkspaceExportFailureCodeV2.INVENTORY_MISMATCH)
    state.file_identities.add(identity)
    if file_stat.st_size > state.request.max_file_bytes:
        raise _ExportBlocked(LHWorkspaceExportFailureCodeV2.EXPORT_LIMIT_EXCEEDED)
    digest = _sha256_file(path)
    state.add(
        LHWorkspaceExportMemberV2.create(
            normalized_path=normalized_path,
            member_type=LHWorkspaceExportMemberTypeV2.FILE,
            media_type=media_type if is_root else _media_type(path),
            size_bytes=file_stat.st_size,
            content_sha256=digest,
            container_ref=None,
            output_ref=output_ref,
        )
    )
    suffix = path.suffix.casefold()
    if suffix in _UNSUPPORTED_CONTAINER_SUFFIXES:
        raise _ExportBlocked(LHWorkspaceExportFailureCodeV2.CONTAINER_UNSUPPORTED)
    if zipfile.is_zipfile(path):
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise _ExportBlocked(LHWorkspaceExportFailureCodeV2.MATERIAL_NOT_FOUND) from exc
        _scan_zip(
            state,
            data,
            output_ref=output_ref,
            container_ref=output_ref,
            depth=1,
        )


def _scan_zip(
    state: _ScanState,
    data: bytes,
    *,
    output_ref: FacadeObjectRef,
    container_ref: FacadeObjectRef,
    depth: int,
) -> None:
    if depth > state.request.max_nested_depth:
        raise _ExportBlocked(LHWorkspaceExportFailureCodeV2.EXPORT_LIMIT_EXCEEDED)
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise _ExportBlocked(LHWorkspaceExportFailureCodeV2.CONTAINER_UNSUPPORTED) from exc
    seen: set[str] = set()
    try:
        for info in sorted(archive.infolist(), key=lambda value: value.filename):
            normalized = _normalize_archive_path(info.filename)
            if normalized in seen:
                raise _ExportBlocked(LHWorkspaceExportFailureCodeV2.INVENTORY_MISMATCH)
            seen.add(normalized)
            if info.flag_bits & 0x1:
                raise _ExportBlocked(LHWorkspaceExportFailureCodeV2.CONTAINER_UNSUPPORTED)
            file_type = stat.S_IFMT(info.external_attr >> 16)
            if file_type == stat.S_IFLNK:
                raise _ExportBlocked(LHWorkspaceExportFailureCodeV2.SYMLINK_UNSUPPORTED)
            if info.is_dir():
                state.add(
                    LHWorkspaceExportMemberV2.create(
                        normalized_path=f"{normalized}/",
                        member_type=(LHWorkspaceExportMemberTypeV2.DIRECTORY),
                        media_type=None,
                        size_bytes=0,
                        content_sha256=None,
                        container_ref=container_ref,
                        output_ref=output_ref,
                    )
                )
                continue
            if file_type not in {0, stat.S_IFREG}:
                raise _ExportBlocked(LHWorkspaceExportFailureCodeV2.SPECIAL_FILE_UNSUPPORTED)
            if info.file_size > state.request.max_file_bytes:
                raise _ExportBlocked(LHWorkspaceExportFailureCodeV2.EXPORT_LIMIT_EXCEEDED)
            ratio = info.file_size / max(info.compress_size, 1)
            if ratio > state.request.max_compression_ratio:
                raise _ExportBlocked(LHWorkspaceExportFailureCodeV2.EXPORT_LIMIT_EXCEEDED)
            try:
                content = archive.read(info)
            except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
                raise _ExportBlocked(LHWorkspaceExportFailureCodeV2.CONTAINER_UNSUPPORTED) from exc
            digest = hashlib.sha256(content).hexdigest()
            state.add(
                LHWorkspaceExportMemberV2.create(
                    normalized_path=normalized,
                    member_type=(LHWorkspaceExportMemberTypeV2.NESTED_MEMBER),
                    media_type=_media_type(Path(normalized)),
                    size_bytes=len(content),
                    content_sha256=digest,
                    container_ref=container_ref,
                    output_ref=output_ref,
                )
            )
            suffix = PurePosixPath(normalized).suffix.casefold()
            if suffix in _UNSUPPORTED_CONTAINER_SUFFIXES:
                raise _ExportBlocked(LHWorkspaceExportFailureCodeV2.CONTAINER_UNSUPPORTED)
            if zipfile.is_zipfile(io.BytesIO(content)):
                _scan_zip(
                    state,
                    content,
                    output_ref=output_ref,
                    container_ref=_container_ref(
                        parent=container_ref,
                        normalized_path=normalized,
                        digest=digest,
                    ),
                    depth=depth + 1,
                )
    finally:
        archive.close()


def _copy_material(
    *,
    source: Path,
    destination: Path,
    normalized_path: str,
    max_file_bytes: int,
) -> Path:
    target = _destination_path(destination, normalized_path)
    mode = source.lstat().st_mode
    if stat.S_ISREG(mode):
        _copy_regular_file(
            source,
            target,
            destination=destination,
            max_file_bytes=max_file_bytes,
        )
        return target
    if not stat.S_ISDIR(mode):
        raise _ExportBlocked(LHWorkspaceExportFailureCodeV2.SPECIAL_FILE_UNSUPPORTED)
    _ensure_directory(destination, target)
    for child in sorted(source.rglob("*"), key=lambda value: value.relative_to(source).as_posix()):
        if child.is_symlink():
            raise _ExportBlocked(LHWorkspaceExportFailureCodeV2.SYMLINK_UNSUPPORTED)
        relative = child.relative_to(source).as_posix()
        child_target = _destination_path(
            destination,
            _join_relative(normalized_path, relative),
        )
        child_mode = child.lstat().st_mode
        if stat.S_ISDIR(child_mode):
            _ensure_directory(destination, child_target)
        elif stat.S_ISREG(child_mode):
            _copy_regular_file(
                child,
                child_target,
                destination=destination,
                max_file_bytes=max_file_bytes,
            )
        else:
            raise _ExportBlocked(LHWorkspaceExportFailureCodeV2.SPECIAL_FILE_UNSUPPORTED)
    return target


def _copy_regular_file(
    source: Path,
    target: Path,
    *,
    destination: Path,
    max_file_bytes: int,
) -> None:
    _ensure_directory(destination, target.parent)
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    source_fd = os.open(source, os.O_RDONLY | no_follow)
    try:
        source_stat = os.fstat(source_fd)
        if not stat.S_ISREG(source_stat.st_mode) or source_stat.st_size > max_file_bytes:
            raise _ExportBlocked(LHWorkspaceExportFailureCodeV2.SPECIAL_FILE_UNSUPPORTED)
        target_fd = os.open(
            target,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | no_follow,
            0o600,
        )
        try:
            remaining = source_stat.st_size
            while remaining:
                chunk = os.read(source_fd, min(1024 * 1024, remaining))
                if not chunk:
                    raise _ExportBlocked(LHWorkspaceExportFailureCodeV2.SOURCE_HASH_MISMATCH)
                view = memoryview(chunk)
                while view:
                    written = os.write(target_fd, view)
                    view = view[written:]
                remaining -= len(chunk)
            if os.read(source_fd, 1):
                raise _ExportBlocked(LHWorkspaceExportFailureCodeV2.SOURCE_HASH_MISMATCH)
            os.fsync(target_fd)
        finally:
            os.close(target_fd)
    finally:
        os.close(source_fd)


def _ensure_directory(root: Path, path: Path) -> None:
    if not is_path_inside(root, path):
        raise _ExportBlocked(LHWorkspaceExportFailureCodeV2.DESTINATION_PATH_VIOLATION)
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    current = path
    while current != root:
        if current.is_symlink() or not current.is_dir():
            raise _ExportBlocked(LHWorkspaceExportFailureCodeV2.DESTINATION_PATH_VIOLATION)
        current = current.parent


def _destination_path(root: Path, normalized_path: str) -> Path:
    target = root.joinpath(*PurePosixPath(normalized_path).parts)
    if not is_path_inside(root, target):
        raise _ExportBlocked(LHWorkspaceExportFailureCodeV2.DESTINATION_PATH_VIOLATION)
    return target


def _normalize_archive_path(value: str) -> str:
    normalized = value.replace("\\", "/")
    if not normalized or normalized.startswith("/") or any(ord(character) < 32 for character in normalized):
        raise _ExportBlocked(LHWorkspaceExportFailureCodeV2.SOURCE_PATH_VIOLATION)
    path = PurePosixPath(normalized)
    if any(part in {"", ".", ".."} for part in path.parts) or (path.parts and path.parts[0].endswith(":")):
        raise _ExportBlocked(LHWorkspaceExportFailureCodeV2.SOURCE_PATH_VIOLATION)
    result = path.as_posix().rstrip("/")
    if not result or len(result) > 512:
        raise _ExportBlocked(LHWorkspaceExportFailureCodeV2.SOURCE_PATH_VIOLATION)
    return result


def _container_ref(
    *,
    parent: FacadeObjectRef,
    normalized_path: str,
    digest: str,
) -> FacadeObjectRef:
    identity = hashlib.sha256(f"{parent.object_id}|{normalized_path}".encode()).hexdigest()
    return FacadeObjectRef(
        object_type="attachment-output",
        object_id=f"attachment-output://container/{identity}",
        object_version="v2",
        object_sha256=digest,
    )


def _join_relative(root: str, child: str) -> str:
    value = f"{root.rstrip('/')}/{child.lstrip('/')}"
    if len(value) > 512:
        raise _ExportBlocked(LHWorkspaceExportFailureCodeV2.SOURCE_PATH_VIOLATION)
    return value


def _is_same_or_descendant(root: str, candidate: str) -> bool:
    normalized_root = root.rstrip("/")
    normalized_candidate = candidate.rstrip("/")
    return normalized_candidate == normalized_root or normalized_candidate.startswith(f"{normalized_root}/")


def _safe_sha256_path(path: Path) -> str:
    try:
        return sha256_path(path)
    except (OSError, ValueError) as exc:
        raise _ExportBlocked(LHWorkspaceExportFailureCodeV2.SOURCE_HASH_MISMATCH) from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, os.O_RDONLY | no_follow)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise _ExportBlocked(LHWorkspaceExportFailureCodeV2.SPECIAL_FILE_UNSUPPORTED)
        for chunk in iter(lambda: os.read(descriptor, 1024 * 1024), b""):
            digest.update(chunk)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _media_type(path: Path) -> str | None:
    return mimetypes.guess_type(path.name)[0]


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


__all__ = [
    "LHWorkspaceExportMaterial",
    "LHWorkspaceExportMaterialError",
    "LHWorkspaceExportMaterialResolver",
    "MappingLHWorkspaceExportMaterialResolver",
    "RegistryLHWorkspaceExportFacade",
]
