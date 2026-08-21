from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from pydantic import BaseModel

from env_mock_agent.facade import (
    LHWorkspaceExportFacade,
    LHWorkspaceExportMemberTypeV2,
    LHWorkspaceExportOutcomeV2,
    LHWorkspaceExportRequestV2,
    LHWorkspaceExportResultV2,
    validate_lh_workspace_export_result_identity,
)
from eval_factory.contracts.core_v2 import canonical_value_v2
from eval_factory.dataset.export import ReleaseBundleFacts

ReleaseBundleFaultInjector = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class ReleaseBundleItemPlan:
    item_id: str
    query_yaml_bytes: bytes
    rubrics_json_bytes: bytes
    manifest_bytes: bytes
    workspace_request: LHWorkspaceExportRequestV2


@dataclass(frozen=True, slots=True)
class ReleaseBundleStorageWrite:
    item_id: str
    bundle_path: Path
    workspace_export_result: LHWorkspaceExportResultV2
    facts: ReleaseBundleFacts
    reused: bool


@dataclass(frozen=True, slots=True)
class _BundleFile:
    relative_path: str
    size_bytes: int
    content_sha256: str


class ImmutableReleaseBundleStorage:
    def __init__(
        self,
        root: Path,
        *,
        policy_error: type[Exception],
        export_error: type[Exception],
        integrity_error: type[Exception],
        fault_injector: ReleaseBundleFaultInjector | None = None,
    ) -> None:
        self.root = root
        self._policy_error = policy_error
        self._export_error = export_error
        self._integrity_error = integrity_error
        self._fault_injector = fault_injector
        self._cas = root / "cas/sha256"
        self._staging = root / ".staging"
        self._cas.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._staging.mkdir(mode=0o700, parents=True, exist_ok=True)

    def build(
        self,
        plan: ReleaseBundleItemPlan,
        *,
        facade: LHWorkspaceExportFacade,
    ) -> ReleaseBundleStorageWrite:
        expected_files = self._expected_files(plan)
        expected_facts = _facts(expected_files)
        expected_directories = _expected_directories(
            expected_files,
            plan.workspace_request,
        )
        final_path = self._bundle_path(expected_facts.bundle_sha256)
        if final_path.exists():
            self._verify_bundle(
                final_path,
                expected_files,
                expected_directories,
                expected_facts,
            )
            return ReleaseBundleStorageWrite(
                item_id=plan.item_id,
                bundle_path=final_path,
                workspace_export_result=_replayed_workspace(plan),
                facts=expected_facts,
                reused=True,
            )

        temporary = Path(tempfile.mkdtemp(prefix="bundle-", dir=self._staging))
        renamed = False
        try:
            self._fault("after_temporary_directory")
            self._write_bytes(
                temporary,
                "query.yaml",
                plan.query_yaml_bytes,
            )
            self._write_bytes(
                temporary,
                ".eval/rubrics.json",
                plan.rubrics_json_bytes,
            )
            self._write_bytes(
                temporary,
                "release-manifest.json",
                plan.manifest_bytes,
            )
            self._fault("after_static_files")
            workspace_result = facade.export(
                plan.workspace_request,
                destination=temporary / "workspace",
            )
            if workspace_result.outcome is not LHWorkspaceExportOutcomeV2.EXPORTED:
                code = (
                    workspace_result.failure_code.value
                    if workspace_result.failure_code is not None
                    else "BLOCKED"
                )
                raise self._export_error(f"LH workspace export blocked: {code}")
            validate_lh_workspace_export_result_identity(workspace_result)
            self._fault("after_workspace_export")
            self._verify_bundle(
                temporary,
                expected_files,
                expected_directories,
                expected_facts,
            )
            _fsync_tree(temporary, integrity_error=self._integrity_error)
            self._fault("before_rename")
            final_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            if final_path.exists():
                self._verify_bundle(
                    final_path,
                    expected_files,
                    expected_directories,
                    expected_facts,
                )
                shutil.rmtree(temporary)
                return ReleaseBundleStorageWrite(
                    item_id=plan.item_id,
                    bundle_path=final_path,
                    workspace_export_result=workspace_result,
                    facts=expected_facts,
                    reused=True,
                )
            try:
                os.rename(temporary, final_path)
            except OSError:
                if not final_path.exists():
                    raise
                self._verify_bundle(
                    final_path,
                    expected_files,
                    expected_directories,
                    expected_facts,
                )
                shutil.rmtree(temporary)
                return ReleaseBundleStorageWrite(
                    item_id=plan.item_id,
                    bundle_path=final_path,
                    workspace_export_result=workspace_result,
                    facts=expected_facts,
                    reused=True,
                )
            renamed = True
            _fsync_directory(final_path.parent)
            self._fault("after_rename")
            return ReleaseBundleStorageWrite(
                item_id=plan.item_id,
                bundle_path=final_path,
                workspace_export_result=workspace_result,
                facts=expected_facts,
                reused=False,
            )
        finally:
            if not renamed and temporary.exists():
                shutil.rmtree(temporary)

    def facts_for_plan(
        self,
        plan: ReleaseBundleItemPlan,
    ) -> ReleaseBundleFacts:
        return _facts(self._expected_files(plan))

    def verify_plan(
        self,
        plan: ReleaseBundleItemPlan,
        facts: ReleaseBundleFacts,
    ) -> Path:
        expected_files = self._expected_files(plan)
        expected_facts = _facts(expected_files)
        if facts != expected_facts:
            raise self._integrity_error("bundle facts differ from release manifest")
        expected_directories = _expected_directories(
            expected_files,
            plan.workspace_request,
        )
        path = self._bundle_path(facts.bundle_sha256)
        self._verify_bundle(
            path,
            expected_files,
            expected_directories,
            expected_facts,
        )
        return path

    def verify_facts(
        self,
        facts: ReleaseBundleFacts,
        workspace_result: LHWorkspaceExportResultV2,
    ) -> Path:
        validate_lh_workspace_export_result_identity(workspace_result)
        if workspace_result.outcome is not LHWorkspaceExportOutcomeV2.EXPORTED:
            raise self._integrity_error("release receipt does not contain an exported workspace")
        path = self._bundle_path(facts.bundle_sha256)
        files, directories = self._read_bundle(path)
        workspace_files = tuple(value for value in files if value.relative_path.startswith("workspace/"))
        expected_workspace_files = _workspace_files(workspace_result)
        if (
            {".eval/rubrics.json", "query.yaml", "release-manifest.json"}
            - {value.relative_path for value in files}
            or any(
                value.relative_path
                not in {
                    ".eval/rubrics.json",
                    "query.yaml",
                    "release-manifest.json",
                }
                and not value.relative_path.startswith("workspace/")
                for value in files
            )
            or workspace_files != expected_workspace_files
            or directories != _workspace_directories(files, workspace_result)
            or _facts(files) != facts
        ):
            raise self._integrity_error("release bundle differs from receipt facts")
        return path

    def _expected_files(
        self,
        plan: ReleaseBundleItemPlan,
    ) -> tuple[_BundleFile, ...]:
        files = [
            _file("query.yaml", plan.query_yaml_bytes),
            _file(".eval/rubrics.json", plan.rubrics_json_bytes),
            _file("release-manifest.json", plan.manifest_bytes),
        ]
        for member in plan.workspace_request.members:
            if member.member_type is LHWorkspaceExportMemberTypeV2.FILE:
                if member.content_sha256 is None:
                    raise self._policy_error("bundle file member has no content hash")
                files.append(
                    _BundleFile(
                        relative_path=(f"workspace/{member.normalized_path}"),
                        size_bytes=member.size_bytes,
                        content_sha256=member.content_sha256,
                    )
                )
        ordered = tuple(sorted(files, key=lambda value: value.relative_path))
        if len({value.relative_path for value in ordered}) != len(ordered):
            raise self._policy_error("bundle manifest contains duplicate physical file paths")
        return ordered

    def _verify_bundle(
        self,
        path: Path,
        expected_files: tuple[_BundleFile, ...],
        expected_directories: set[str],
        expected_facts: ReleaseBundleFacts,
    ) -> None:
        ordered, actual_directories = self._read_bundle(path)
        if ordered != expected_files or _facts(ordered) != expected_facts:
            raise self._integrity_error("release bundle files differ from immutable facts")
        if actual_directories != expected_directories:
            raise self._integrity_error("release bundle directories differ from immutable facts")

    def _read_bundle(
        self,
        path: Path,
    ) -> tuple[tuple[_BundleFile, ...], set[str]]:
        if path.is_symlink() or not path.is_dir():
            raise self._integrity_error("release bundle directory is missing or unsafe")
        actual_files: list[_BundleFile] = []
        actual_directories: set[str] = set()
        for child in sorted(path.rglob("*")):
            if child.is_symlink():
                raise self._integrity_error("release bundle contains a symlink")
            mode = child.lstat().st_mode
            relative = child.relative_to(path).as_posix()
            if stat.S_ISDIR(mode):
                actual_directories.add(relative)
            elif stat.S_ISREG(mode):
                actual_files.append(
                    _BundleFile(
                        relative_path=relative,
                        size_bytes=child.stat().st_size,
                        content_sha256=_sha256_file(
                            child,
                            integrity_error=self._integrity_error,
                        ),
                    )
                )
            else:
                raise self._integrity_error("release bundle contains a special file")
        return (
            tuple(
                sorted(
                    actual_files,
                    key=lambda value: value.relative_path,
                )
            ),
            actual_directories,
        )

    def _write_bytes(
        self,
        root: Path,
        relative_path: str,
        content: bytes,
    ) -> None:
        relative = _safe_relative(
            relative_path,
            policy_error=self._policy_error,
        )
        target = root.joinpath(*relative.parts)
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if target.is_symlink():
            raise self._integrity_error("release bundle static path is a symlink")
        descriptor = os.open(
            target,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            view = memoryview(content)
            while view:
                written = os.write(descriptor, view)
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _bundle_path(self, digest: str) -> Path:
        return self._cas / digest[:2] / digest

    def _fault(self, point: str) -> None:
        if self._fault_injector is not None:
            self._fault_injector(point)


def prepare_bundle_root(
    path: Path,
    *,
    policy_error: type[Exception],
    forbidden_names: frozenset[str] = frozenset(),
) -> Path:
    expanded = path.expanduser()
    if any(part.casefold() in forbidden_names for part in expanded.parts):
        raise policy_error("non-production release root cannot use a production namespace")
    absolute = expanded.absolute()
    if any(value.is_symlink() for value in (absolute, *absolute.parents)) or (
        expanded.exists() and not expanded.is_dir()
    ):
        raise policy_error("release bundle root must be a real directory")
    expanded.mkdir(mode=0o700, parents=True, exist_ok=True)
    return expanded.resolve(strict=True)


def roots_overlap(left: Path, right: Path) -> bool:
    return left == right or _is_same_or_descendant(left, right) or _is_same_or_descendant(right, left)


def manifest_bytes_without_audit(value: BaseModel) -> bytes:
    payload = value.model_dump(mode="python", exclude_none=False)
    return (
        json.dumps(
            canonical_value_v2(_without_audit(payload)),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        + b"\n"
    )


def _replayed_workspace(
    plan: ReleaseBundleItemPlan,
) -> LHWorkspaceExportResultV2:
    return LHWorkspaceExportResultV2.exported(
        request=plan.workspace_request,
        observed_members=plan.workspace_request.members,
        copied_output_refs=plan.workspace_request.output_refs,
    )


def _file(relative_path: str, content: bytes) -> _BundleFile:
    return _BundleFile(
        relative_path=relative_path,
        size_bytes=len(content),
        content_sha256=_sha256(content),
    )


def _facts(files: tuple[_BundleFile, ...]) -> ReleaseBundleFacts:
    payload = [
        {
            "relative_path": value.relative_path,
            "size_bytes": value.size_bytes,
            "content_sha256": value.content_sha256,
        }
        for value in files
    ]
    return ReleaseBundleFacts(
        bundle_sha256=_sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ),
        bundle_file_count=len(files),
        bundle_total_bytes=sum(value.size_bytes for value in files),
    )


def _expected_directories(
    files: tuple[_BundleFile, ...],
    request: LHWorkspaceExportRequestV2,
) -> set[str]:
    directories = {".eval", "workspace"}
    for value in files:
        parent = PurePosixPath(value.relative_path).parent
        while str(parent) not in {"", "."}:
            directories.add(parent.as_posix())
            parent = parent.parent
    for member in request.members:
        if member.member_type is LHWorkspaceExportMemberTypeV2.DIRECTORY and member.container_ref is None:
            parent = PurePosixPath(f"workspace/{member.normalized_path.rstrip('/')}")
            while str(parent) not in {"", "."}:
                directories.add(parent.as_posix())
                parent = parent.parent
    return directories


def _workspace_files(
    result: LHWorkspaceExportResultV2,
) -> tuple[_BundleFile, ...]:
    files = tuple(
        _BundleFile(
            relative_path=f"workspace/{member.normalized_path}",
            size_bytes=member.size_bytes,
            content_sha256=member.content_sha256 or "",
        )
        for member in result.observed_members
        if member.member_type is LHWorkspaceExportMemberTypeV2.FILE
    )
    return tuple(sorted(files, key=lambda value: value.relative_path))


def _workspace_directories(
    files: tuple[_BundleFile, ...],
    result: LHWorkspaceExportResultV2,
) -> set[str]:
    directories = {".eval", "workspace"}
    for value in files:
        parent = PurePosixPath(value.relative_path).parent
        while str(parent) not in {"", "."}:
            directories.add(parent.as_posix())
            parent = parent.parent
    for member in result.observed_members:
        if member.member_type is LHWorkspaceExportMemberTypeV2.DIRECTORY and member.container_ref is None:
            parent = PurePosixPath(f"workspace/{member.normalized_path.rstrip('/')}")
            while str(parent) not in {"", "."}:
                directories.add(parent.as_posix())
                parent = parent.parent
    return directories


def _safe_relative(
    value: str,
    *,
    policy_error: type[Exception],
) -> PurePosixPath:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or any(part in {"", "."} for part in path.parts):
        raise policy_error("release bundle path is unsafe")
    return path


def _without_audit(value: object) -> object:
    if isinstance(value, dict):
        return {key: _without_audit(item) for key, item in value.items() if key != "audit"}
    if isinstance(value, (list, tuple)):
        return [_without_audit(item) for item in value]
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(
    root: Path,
    *,
    integrity_error: type[Exception],
) -> None:
    directories: list[Path] = [root]
    for child in root.rglob("*"):
        if child.is_symlink():
            raise integrity_error("release bundle contains a symlink before rename")
        mode = child.lstat().st_mode
        if stat.S_ISDIR(mode):
            directories.append(child)
        elif stat.S_ISREG(mode):
            descriptor = os.open(
                child,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        else:
            raise integrity_error("release bundle contains a special file before rename")
    for directory in sorted(
        directories,
        key=lambda value: len(value.parts),
        reverse=True,
    ):
        _fsync_directory(directory)


def _is_same_or_descendant(root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(
    path: Path,
    *,
    integrity_error: type[Exception],
) -> str:
    digest = hashlib.sha256()
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise integrity_error("release bundle member is not a regular file")
        for chunk in iter(lambda: os.read(descriptor, 1024 * 1024), b""):
            digest.update(chunk)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


__all__ = [
    "ImmutableReleaseBundleStorage",
    "ReleaseBundleFaultInjector",
    "ReleaseBundleItemPlan",
    "ReleaseBundleStorageWrite",
    "manifest_bytes_without_audit",
    "prepare_bundle_root",
    "roots_overlap",
]
