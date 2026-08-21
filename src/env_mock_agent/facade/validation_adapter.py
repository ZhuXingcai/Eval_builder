from __future__ import annotations

import hashlib
import io
import mimetypes
import re
import stat
import unicodedata
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Protocol

import pymupdf

from env_mock_agent.facade.contracts import FacadeObjectRef
from env_mock_agent.facade.execution_v2 import (
    AttachmentExecutionRequestV2,
    attachment_execution_request_ref,
    validate_attachment_execution_request_identity,
)
from env_mock_agent.facade.semantic_review_adapter import (
    ProviderAttachmentRepairMaterial,
)
from env_mock_agent.facade.semantic_review_v2 import AttachmentRepairRequestV2
from env_mock_agent.facade.validation_v2 import (
    ATTACHMENT_VALIDATION_POLICY_VERSION,
    PROMPT_LEAKAGE_NORMALIZATION_VERSION,
    AttachmentInventoryMemberTypeV2,
    AttachmentInventoryMemberV2,
    AttachmentValidationFacade,
    AttachmentValidationFailureCodeV2,
    AttachmentValidationFindingCodeV2,
    AttachmentValidationFindingV2,
    AttachmentValidationFingerprintMatchKindV2,
    AttachmentValidationFingerprintV2,
    AttachmentValidationRequestV2,
    AttachmentValidationResultV2,
    AttachmentValidationStatusV2,
    attachment_inventory_member_carried_sha256,
    attachment_validation_finding_carried_sha256,
    attachment_validation_finding_policy,
    attachment_validation_request_ref,
    attachment_validation_result_carried_sha256,
    validate_attachment_validation_request_identity,
)
from env_mock_agent.providers.helpers import sha256_path
from env_mock_agent.runtimes.security import is_path_inside
from env_mock_agent.schemas import (
    ArtifactPlan,
    ArtifactResult,
    ArtifactStatus,
    FindingCategory,
    ForbiddenOutputSpec,
    Severity,
    ValidationFinding,
)
from env_mock_agent.validators import ValidationRequest, ValidatorRegistry
from env_mock_agent.validators.extract import TEXT_SUFFIXES, extract_text
from env_mock_agent.validators.secrets import SECRET_PATTERNS

_TOKEN_PATTERN = re.compile(r"[\w]+(?:[._/-][\w]+)*", flags=re.UNICODE)
_TEXT_SUFFIXES = TEXT_SUFFIXES | {
    ".cfg",
    ".conf",
    ".env",
    ".ini",
    ".log",
    ".properties",
    ".sh",
}
_ZIP_SUFFIXES = {
    ".docm",
    ".docx",
    ".jar",
    ".ods",
    ".odt",
    ".pptm",
    ".pptx",
    ".whl",
    ".xlsm",
    ".xlsx",
    ".zip",
}
_UNSUPPORTED_CONTAINER_SUFFIXES = {
    ".7z",
    ".bz2",
    ".gz",
    ".rar",
    ".tar",
    ".tgz",
    ".xz",
}
_MANDATORY_VALIDATOR_IDS = frozenset(
    {
        "common",
        "configured-pii",
        "metadata",
        "package-inventory",
        "restricted-fingerprint",
        "secrets",
    }
)
_VALIDATOR_ALIASES = {
    "code-project-validator": "code_project",
    "code_project": "code_project",
    "common": "common",
    "common-validator": "common",
    "docx": "docx",
    "docx-validator": "docx",
    "leakage": "restricted-fingerprint",
    "leakage-validator": "restricted-fingerprint",
    "metadata": "metadata",
    "metadata-validator": "metadata",
    "pdf": "pdf",
    "pdf-validator": "pdf",
    "secret-validator": "secrets",
    "secrets": "secrets",
    "structured": "structured",
    "structured-validator": "structured",
    "text": "text",
    "text-validator": "text",
    "xlsx": "xlsx",
    "xlsx-validator": "xlsx",
}


class AttachmentValidationMaterialError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        code: AttachmentValidationFailureCodeV2,
    ) -> None:
        super().__init__(message)
        self.code = code


class _ValidationBlocked(RuntimeError):
    def __init__(
        self,
        code: AttachmentValidationFailureCodeV2,
    ) -> None:
        super().__init__(code.value)
        self.code = code


@dataclass(frozen=True, slots=True)
class ProviderValidationMaterial:
    plan: ArtifactPlan


class AttachmentValidationMaterialResolver(Protocol):
    def resolve_output(
        self,
        request: AttachmentValidationRequestV2,
    ) -> Path: ...

    def resolve_validation_material(
        self,
        request: AttachmentValidationRequestV2,
    ) -> ProviderValidationMaterial: ...


class MappingAttachmentValidationMaterialResolver:
    def __init__(
        self,
        *,
        outputs: Mapping[str, Path],
        materials: Mapping[str, ProviderValidationMaterial],
    ) -> None:
        self._outputs = dict(outputs)
        self._materials = dict(materials)

    def resolve_output(
        self,
        request: AttachmentValidationRequestV2,
    ) -> Path:
        try:
            return self._outputs[request.output_ref.object_id].expanduser()
        except KeyError as exc:
            raise AttachmentValidationMaterialError(
                "attachment output is not registered",
                code=AttachmentValidationFailureCodeV2.MATERIAL_NOT_FOUND,
            ) from exc

    def resolve_validation_material(
        self,
        request: AttachmentValidationRequestV2,
    ) -> ProviderValidationMaterial:
        try:
            return self._materials[request.build_spec_ref.object_id]
        except KeyError as exc:
            raise AttachmentValidationMaterialError(
                "attachment validation material is not registered",
                code=AttachmentValidationFailureCodeV2.MATERIAL_NOT_FOUND,
            ) from exc


class StagingAttachmentValidationMaterialResolver:
    def __init__(
        self,
        *,
        staging_root: Path,
        execution_requests: Mapping[str, AttachmentExecutionRequestV2],
        materials: Mapping[str, ProviderValidationMaterial],
    ) -> None:
        self._staging_root = staging_root.expanduser().resolve()
        self._execution_requests = dict(execution_requests)
        self._materials = dict(materials)
        self._repaired_outputs: dict[str, _RegisteredRepairedOutput] = {}

    def resolve_output(
        self,
        request: AttachmentValidationRequestV2,
    ) -> Path:
        repaired = self._repaired_outputs.get(request.output_ref.object_id)
        if repaired is not None:
            if (
                repaired.output_ref != request.output_ref
                or repaired.artifact_id != request.artifact_id
                or repaired.build_spec_ref != request.build_spec_ref
                or repaired.execution_result_ref != request.execution_result_ref
                or repaired.logical_path != request.logical_path
                or repaired.media_type != request.media_type
            ):
                raise AttachmentValidationMaterialError(
                    "registered repaired output does not match validation request",
                    code=AttachmentValidationFailureCodeV2.MATERIAL_MISMATCH,
                )
            return repaired.path
        try:
            execution = self._execution_requests[request.execution_request_ref.object_id]
        except KeyError as exc:
            raise AttachmentValidationMaterialError(
                "attachment execution request is not registered",
                code=AttachmentValidationFailureCodeV2.MATERIAL_NOT_FOUND,
            ) from exc
        validate_attachment_execution_request_identity(execution)
        if (
            attachment_execution_request_ref(execution) != request.execution_request_ref
            or execution.artifact_id != request.artifact_id
            or execution.relative_path != request.logical_path
            or execution.media_type != request.media_type
            or execution.build_spec_ref != request.build_spec_ref
            or execution.producer_task_view_ref != request.producer_task_view_ref
        ):
            raise AttachmentValidationMaterialError(
                "attachment execution material does not match validation request",
                code=AttachmentValidationFailureCodeV2.MATERIAL_MISMATCH,
            )
        digest = hashlib.sha256(execution.execution_request_id.encode()).hexdigest()
        staging = (self._staging_root / digest).resolve()
        target = staging / execution.relative_path
        if not is_path_inside(staging, target):
            raise AttachmentValidationMaterialError(
                "attachment output path escapes staging",
                code=AttachmentValidationFailureCodeV2.OUTPUT_PATH_VIOLATION,
            )
        return target

    def resolve_validation_material(
        self,
        request: AttachmentValidationRequestV2,
    ) -> ProviderValidationMaterial:
        try:
            return self._materials[request.build_spec_ref.object_id]
        except KeyError as exc:
            raise AttachmentValidationMaterialError(
                "attachment validation material is not registered",
                code=AttachmentValidationFailureCodeV2.MATERIAL_NOT_FOUND,
            ) from exc

    def register_repaired_output(
        self,
        *,
        request: AttachmentRepairRequestV2,
        material: ProviderAttachmentRepairMaterial,
        output_ref: FacadeObjectRef,
        output_bytes: bytes,
    ) -> None:
        if (
            output_ref.object_sha256 != hashlib.sha256(output_bytes).hexdigest()
            or material.source_output_ref != request.output_ref
            or material.logical_path.startswith("/")
        ):
            raise AttachmentValidationMaterialError(
                "repaired output registration does not match repair material",
                code=AttachmentValidationFailureCodeV2.MATERIAL_MISMATCH,
            )
        repair_root = (
            self._staging_root
            / "semantic-repairs"
            / hashlib.sha256(request.repair_request_id.encode()).hexdigest()
        ).resolve()
        target = (repair_root / material.logical_path).resolve()
        if not is_path_inside(repair_root, target):
            raise AttachmentValidationMaterialError(
                "repaired output registration escapes staging",
                code=AttachmentValidationFailureCodeV2.OUTPUT_PATH_VIOLATION,
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            raise AttachmentValidationMaterialError(
                "repaired output registration cannot replace a symlink",
                code=AttachmentValidationFailureCodeV2.SYMLINK_UNSUPPORTED,
            )
        target.write_bytes(output_bytes)
        if sha256_path(target) != output_ref.object_sha256:
            raise AttachmentValidationMaterialError(
                "registered repaired output hash changed",
                code=AttachmentValidationFailureCodeV2.OUTPUT_HASH_MISMATCH,
            )
        self._repaired_outputs[output_ref.object_id] = _RegisteredRepairedOutput(
            output_ref=output_ref,
            artifact_id=request.artifact_id,
            build_spec_ref=request.build_spec_ref,
            execution_result_ref=request.execution_result_ref,
            logical_path=material.logical_path,
            media_type=material.media_type,
            path=target,
        )


@dataclass(frozen=True, slots=True)
class _RegisteredRepairedOutput:
    output_ref: FacadeObjectRef
    artifact_id: str
    build_spec_ref: FacadeObjectRef
    execution_result_ref: FacadeObjectRef
    logical_path: str
    media_type: str
    path: Path


@dataclass(frozen=True, slots=True)
class _TextProjection:
    inventory_member_id: str | None
    normalized_path: str
    text: str


@dataclass(slots=True)
class _ScanState:
    request: AttachmentValidationRequestV2
    members: list[AttachmentInventoryMemberV2] = field(default_factory=list)
    projections: list[_TextProjection] = field(default_factory=list)
    expanded_bytes: int = 0
    extracted_characters: int = 0

    def add_member(self, member: AttachmentInventoryMemberV2) -> None:
        if len(self.members) >= self.request.scan_limits.max_inventory_members:
            raise _ValidationBlocked(AttachmentValidationFailureCodeV2.SCAN_LIMIT_EXCEEDED)
        self.members.append(member)

    def add_projection(
        self,
        *,
        inventory_member_id: str | None,
        normalized_path: str,
        text: str,
    ) -> None:
        observed = self.extracted_characters + len(text)
        if observed > self.request.scan_limits.max_extracted_characters:
            raise _ValidationBlocked(AttachmentValidationFailureCodeV2.SCAN_LIMIT_EXCEEDED)
        self.extracted_characters = observed
        self.projections.append(
            _TextProjection(
                inventory_member_id=inventory_member_id,
                normalized_path=normalized_path,
                text=text,
            )
        )


class RegistryAttachmentValidationFacade(AttachmentValidationFacade):
    def __init__(
        self,
        *,
        resolver: AttachmentValidationMaterialResolver,
        validators: ValidatorRegistry | None = None,
    ) -> None:
        self._resolver = resolver
        self._validators = validators or ValidatorRegistry.default()
        self._cache: dict[
            str,
            tuple[str, AttachmentValidationResultV2],
        ] = {}

    async def validate(
        self,
        request: AttachmentValidationRequestV2,
    ) -> AttachmentValidationResultV2:
        validate_attachment_validation_request_identity(request)
        cached = self._cache.get(request.idempotency_key)
        if cached is not None:
            request_sha256, result = cached
            if request_sha256 != request.validation_request_sha256:
                raise ValueError("attachment validation idempotency conflict")
            return result

        required_validator_ids = _required_validator_ids(request)
        if any(validator_id not in _VALIDATOR_ALIASES for validator_id in request.declared_validator_ids):
            return self._cache_result(
                request,
                _blocked_result(
                    request,
                    required_validator_ids=required_validator_ids,
                    code=(AttachmentValidationFailureCodeV2.VALIDATOR_UNAVAILABLE),
                ),
            )
        try:
            path = self._resolver.resolve_output(request)
            material = self._resolver.resolve_validation_material(request)
            _validate_material(request, material)
            _validate_output(path, request)
            _validate_fingerprint_policy(request)
            state = _build_inventory(path, request)
            compiled_pii = _compile_pii_rules(request)
            legacy_findings = _run_legacy_validators(
                registry=self._validators,
                request=request,
                material=material,
                path=path,
                validator_ids=required_validator_ids,
            )
            findings = [
                *_sanitize_legacy_findings(
                    request=request,
                    findings=legacy_findings,
                ),
                *_scan_secret_findings(request=request, state=state),
                *_scan_pii_findings(
                    request=request,
                    state=state,
                    compiled_rules=compiled_pii,
                ),
                *_scan_forbidden_output_findings(
                    request=request,
                    state=state,
                ),
                *_scan_fingerprint_findings(
                    request=request,
                    state=state,
                ),
            ]
            _validate_output(path, request)
        except AttachmentValidationMaterialError as exc:
            return self._cache_result(
                request,
                _blocked_result(
                    request,
                    required_validator_ids=required_validator_ids,
                    code=exc.code,
                ),
            )
        except _ValidationBlocked as exc:
            return self._cache_result(
                request,
                _blocked_result(
                    request,
                    required_validator_ids=required_validator_ids,
                    code=exc.code,
                ),
            )
        except Exception:
            return self._cache_result(
                request,
                _blocked_result(
                    request,
                    required_validator_ids=required_validator_ids,
                    code=AttachmentValidationFailureCodeV2.VALIDATOR_FAILED,
                ),
            )

        ordered_findings = tuple(
            sorted(
                _deduplicate_findings(findings),
                key=lambda item: item.finding_id,
            )
        )
        ordered_members = tuple(
            sorted(
                state.members,
                key=_inventory_member_key,
            )
        )
        status = (
            AttachmentValidationStatusV2.FAILED if ordered_findings else AttachmentValidationStatusV2.PASSED
        )
        result = AttachmentValidationResultV2(
            validation_result_id="attachment-validation-result://pending",
            validation_request_ref=attachment_validation_request_ref(request),
            artifact_id=request.artifact_id,
            output_ref=request.output_ref,
            output_sha256=request.output_sha256,
            status=status,
            executed_validator_ids=required_validator_ids,
            required_validator_ids=required_validator_ids,
            complete_leakage_categories=request.complete_leakage_categories,
            findings=ordered_findings,
            inventory_members=ordered_members,
            inventory_complete=True,
            scan_complete=True,
            failure_code=None,
            policy_version=ATTACHMENT_VALIDATION_POLICY_VERSION,
            validation_result_sha256="0" * 64,
        )
        return self._cache_result(request, _finalize_result(result))

    def _cache_result(
        self,
        request: AttachmentValidationRequestV2,
        result: AttachmentValidationResultV2,
    ) -> AttachmentValidationResultV2:
        self._cache[request.idempotency_key] = (
            request.validation_request_sha256,
            result,
        )
        return result


def _required_validator_ids(
    request: AttachmentValidationRequestV2,
) -> tuple[str, ...]:
    values = set(_MANDATORY_VALIDATOR_IDS)
    for validator_id in request.declared_validator_ids:
        values.add(_VALIDATOR_ALIASES.get(validator_id, validator_id))
    return tuple(sorted(values))


def _validate_material(
    request: AttachmentValidationRequestV2,
    material: ProviderValidationMaterial,
) -> None:
    plan = material.plan
    expected_declared = tuple(sorted(plan.validators))
    if (
        plan.artifact_id != request.artifact_id
        or plan.relative_path != request.logical_path
        or expected_declared != request.declared_validator_ids
    ):
        raise AttachmentValidationMaterialError(
            "attachment validation material is mismatched",
            code=AttachmentValidationFailureCodeV2.MATERIAL_MISMATCH,
        )
    if plan.render_contract:
        raise AttachmentValidationMaterialError(
            "render contract validation capability is unavailable",
            code=AttachmentValidationFailureCodeV2.VALIDATOR_UNAVAILABLE,
        )


def _validate_output(
    path: Path,
    request: AttachmentValidationRequestV2,
) -> None:
    if path.is_symlink():
        raise _ValidationBlocked(AttachmentValidationFailureCodeV2.SYMLINK_UNSUPPORTED)
    if not path.exists():
        raise _ValidationBlocked(AttachmentValidationFailureCodeV2.OUTPUT_MISSING)
    if path.is_dir():
        file_identities: set[tuple[int, int]] = set()
        for child in path.rglob("*"):
            if child.is_symlink():
                raise _ValidationBlocked(AttachmentValidationFailureCodeV2.SYMLINK_UNSUPPORTED)
            if child.is_file():
                child_stat = child.stat()
                identity = (child_stat.st_dev, child_stat.st_ino)
                if identity in file_identities:
                    raise _ValidationBlocked(AttachmentValidationFailureCodeV2.PACKAGE_PATH_COLLISION)
                file_identities.add(identity)
    if sha256_path(path) != request.output_sha256:
        raise _ValidationBlocked(AttachmentValidationFailureCodeV2.OUTPUT_HASH_MISMATCH)


def _build_inventory(
    path: Path,
    request: AttachmentValidationRequestV2,
) -> _ScanState:
    state = _ScanState(request=request)
    if path.is_file():
        _add_regular_file(
            state=state,
            path=path,
            normalized_path=request.logical_path,
            output_ref=request.output_ref,
            media_type=request.media_type,
        )
        return state

    _add_member(
        state,
        artifact_id=request.artifact_id,
        output_ref=request.output_ref,
        normalized_path=request.logical_path,
        member_type=AttachmentInventoryMemberTypeV2.DIRECTORY,
        media_type=None,
        size_bytes=0,
        content_sha256=None,
        container_ref=None,
    )
    root = path
    for child in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if child.is_symlink():
            raise _ValidationBlocked(AttachmentValidationFailureCodeV2.SYMLINK_UNSUPPORTED)
        relative = child.relative_to(root).as_posix()
        normalized = _join_relative(request.logical_path, relative)
        if child.is_dir():
            _add_member(
                state,
                artifact_id=request.artifact_id,
                output_ref=request.output_ref,
                normalized_path=normalized,
                member_type=AttachmentInventoryMemberTypeV2.DIRECTORY,
                media_type=None,
                size_bytes=0,
                content_sha256=None,
                container_ref=None,
            )
        elif child.is_file():
            _add_regular_file(
                state=state,
                path=child,
                normalized_path=normalized,
                output_ref=request.output_ref,
                media_type=_media_type(child),
            )
        else:
            raise _ValidationBlocked(AttachmentValidationFailureCodeV2.CONTENT_UNSCANNABLE)
    return state


def _add_regular_file(
    *,
    state: _ScanState,
    path: Path,
    normalized_path: str,
    output_ref: FacadeObjectRef,
    media_type: str | None,
) -> None:
    size = path.stat().st_size
    if size > state.request.scan_limits.max_file_bytes:
        raise _ValidationBlocked(AttachmentValidationFailureCodeV2.SCAN_LIMIT_EXCEEDED)
    digest = _sha256_file(path)
    member = _add_member(
        state,
        artifact_id=state.request.artifact_id,
        output_ref=state.request.output_ref,
        normalized_path=normalized_path,
        member_type=AttachmentInventoryMemberTypeV2.FILE,
        media_type=media_type,
        size_bytes=size,
        content_sha256=digest,
        container_ref=None,
    )
    _add_path_projection(state, member)
    suffix = path.suffix.lower()
    if suffix in _UNSUPPORTED_CONTAINER_SUFFIXES:
        raise _ValidationBlocked(AttachmentValidationFailureCodeV2.CONTAINER_UNSUPPORTED)
    if suffix in _ZIP_SUFFIXES or zipfile.is_zipfile(path):
        if suffix in {".docm", ".docx", ".xlsm", ".xlsx"}:
            try:
                extracted = extract_text(
                    path,
                    max_characters=(state.request.scan_limits.max_extracted_characters + 1),
                )
            except Exception as exc:
                raise _ValidationBlocked(AttachmentValidationFailureCodeV2.CONTENT_UNSCANNABLE) from exc
            state.add_projection(
                inventory_member_id=member.inventory_member_id,
                normalized_path=normalized_path,
                text=extracted,
            )
        try:
            with path.open("rb") as handle:
                data = handle.read(state.request.scan_limits.max_file_bytes + 1)
        except OSError as exc:
            raise _ValidationBlocked(AttachmentValidationFailureCodeV2.CONTENT_UNSCANNABLE) from exc
        if len(data) > state.request.scan_limits.max_file_bytes:
            raise _ValidationBlocked(AttachmentValidationFailureCodeV2.SCAN_LIMIT_EXCEEDED)
        _scan_zip(
            state=state,
            data=data,
            container_ref=output_ref,
            depth=1,
        )
        return
    text = _text_from_file(path)
    if text is not None:
        state.add_projection(
            inventory_member_id=member.inventory_member_id,
            normalized_path=normalized_path,
            text=text,
        )


def _scan_zip(
    *,
    state: _ScanState,
    data: bytes,
    container_ref: FacadeObjectRef,
    depth: int,
) -> None:
    if depth > state.request.scan_limits.max_nested_depth:
        raise _ValidationBlocked(AttachmentValidationFailureCodeV2.SCAN_LIMIT_EXCEEDED)
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise _ValidationBlocked(AttachmentValidationFailureCodeV2.CONTAINER_TRUNCATED) from exc
    seen: set[str] = set()
    try:
        for info in sorted(archive.infolist(), key=lambda item: item.filename):
            normalized = _normalize_archive_path(info.filename)
            if normalized in seen:
                raise _ValidationBlocked(AttachmentValidationFailureCodeV2.PACKAGE_PATH_COLLISION)
            seen.add(normalized)
            if info.flag_bits & 0x1:
                raise _ValidationBlocked(AttachmentValidationFailureCodeV2.CONTAINER_ENCRYPTED)
            mode = info.external_attr >> 16
            if stat.S_IFMT(mode) == stat.S_IFLNK:
                raise _ValidationBlocked(AttachmentValidationFailureCodeV2.SYMLINK_UNSUPPORTED)
            if info.is_dir():
                _add_member(
                    state,
                    artifact_id=state.request.artifact_id,
                    output_ref=state.request.output_ref,
                    normalized_path=f"{normalized}/",
                    member_type=AttachmentInventoryMemberTypeV2.DIRECTORY,
                    media_type=None,
                    size_bytes=0,
                    content_sha256=None,
                    container_ref=container_ref,
                )
                continue
            if info.file_size > state.request.scan_limits.max_file_bytes:
                raise _ValidationBlocked(AttachmentValidationFailureCodeV2.SCAN_LIMIT_EXCEEDED)
            ratio = info.file_size / max(info.compress_size, 1)
            if ratio > state.request.scan_limits.max_compression_ratio:
                raise _ValidationBlocked(AttachmentValidationFailureCodeV2.SCAN_LIMIT_EXCEEDED)
            state.expanded_bytes += info.file_size
            if state.expanded_bytes > state.request.scan_limits.max_expanded_bytes:
                raise _ValidationBlocked(AttachmentValidationFailureCodeV2.SCAN_LIMIT_EXCEEDED)
            try:
                content = archive.read(info)
            except RuntimeError as exc:
                raise _ValidationBlocked(AttachmentValidationFailureCodeV2.CONTAINER_ENCRYPTED) from exc
            except (OSError, zipfile.BadZipFile) as exc:
                raise _ValidationBlocked(AttachmentValidationFailureCodeV2.CONTAINER_TRUNCATED) from exc
            digest = hashlib.sha256(content).hexdigest()
            member = _add_member(
                state,
                artifact_id=state.request.artifact_id,
                output_ref=state.request.output_ref,
                normalized_path=normalized,
                member_type=AttachmentInventoryMemberTypeV2.NESTED_MEMBER,
                media_type=_media_type(Path(normalized)),
                size_bytes=len(content),
                content_sha256=digest,
                container_ref=container_ref,
            )
            _add_path_projection(state, member)
            suffix = PurePosixPath(normalized).suffix.lower()
            text = _text_from_bytes(content, suffix=suffix)
            if text is not None:
                state.add_projection(
                    inventory_member_id=member.inventory_member_id,
                    normalized_path=normalized,
                    text=text,
                )
            if zipfile.is_zipfile(io.BytesIO(content)):
                child_ref = _container_ref(
                    parent=container_ref,
                    normalized_path=normalized,
                    digest=digest,
                )
                _scan_zip(
                    state=state,
                    data=content,
                    container_ref=child_ref,
                    depth=depth + 1,
                )
    finally:
        archive.close()


def _add_path_projection(
    state: _ScanState,
    member: AttachmentInventoryMemberV2,
) -> None:
    state.add_projection(
        inventory_member_id=member.inventory_member_id,
        normalized_path=member.normalized_path,
        text="",
    )


def _add_member(
    state: _ScanState,
    *,
    artifact_id: str,
    output_ref: FacadeObjectRef,
    normalized_path: str,
    member_type: AttachmentInventoryMemberTypeV2,
    media_type: str | None,
    size_bytes: int,
    content_sha256: str | None,
    container_ref: FacadeObjectRef | None,
) -> AttachmentInventoryMemberV2:
    member = AttachmentInventoryMemberV2(
        inventory_member_id="attachment-inventory-member://pending",
        artifact_id=artifact_id,
        output_ref=output_ref,
        normalized_path=normalized_path,
        member_type=member_type,
        media_type=media_type,
        size_bytes=size_bytes,
        content_sha256=content_sha256,
        container_ref=container_ref,
        inventory_member_sha256="0" * 64,
    )
    digest = attachment_inventory_member_carried_sha256(member)
    member = member.model_copy(
        update={
            "inventory_member_id": (f"attachment-inventory-member://sha256/{digest}"),
            "inventory_member_sha256": digest,
        }
    )
    state.add_member(member)
    return member


def _compile_pii_rules(
    request: AttachmentValidationRequestV2,
) -> tuple[tuple[str, re.Pattern[str]], ...]:
    compiled: list[tuple[str, re.Pattern[str]]] = []
    for rule in request.configured_pii_rules:
        try:
            compiled.append((rule.rule_id, re.compile(rule.pattern)))
        except re.error as exc:
            raise _ValidationBlocked(AttachmentValidationFailureCodeV2.INVALID_CONFIGURED_PII_RULE) from exc
    return tuple(compiled)


def _validate_fingerprint_policy(
    request: AttachmentValidationRequestV2,
) -> None:
    if any(
        item.normalization_version != PROMPT_LEAKAGE_NORMALIZATION_VERSION
        for item in request.leakage_fingerprints
    ):
        raise _ValidationBlocked(AttachmentValidationFailureCodeV2.UNKNOWN_FINGERPRINT_NORMALIZATION)


def _run_legacy_validators(
    *,
    registry: ValidatorRegistry,
    request: AttachmentValidationRequestV2,
    material: ProviderValidationMaterial,
    path: Path,
    validator_ids: tuple[str, ...],
) -> list[ValidationFinding]:
    names = tuple(
        item
        for item in validator_ids
        if item
        in {
            "code_project",
            "common",
            "docx",
            "metadata",
            "pdf",
            "structured",
            "text",
            "xlsx",
        }
    )
    result = ArtifactResult(
        artifact_id=request.artifact_id,
        status=ArtifactStatus.BUILT,
        path=str(path),
        sha256=request.output_sha256,
        bytes_count=(
            path.stat().st_size
            if path.is_file()
            else sum(
                item.stat().st_size for item in path.rglob("*") if item.is_file() and not item.is_symlink()
            )
        ),
        attempts=1,
    )
    validation_request = ValidationRequest(
        plan=material.plan.model_copy(deep=True),
        result=result,
        forbidden_outputs=[
            ForbiddenOutputSpec(
                forbidden_id=(f"forbidden-output-{hashlib.sha256(value.encode()).hexdigest()[:16]}"),
                description="producer-safe forbidden output",
                semantic_patterns=[value],
            )
            for value in request.forbidden_output_values
        ],
    )
    try:
        return registry.validate(validation_request, list(names))
    except KeyError as exc:
        raise _ValidationBlocked(AttachmentValidationFailureCodeV2.VALIDATOR_UNAVAILABLE) from exc
    except Exception as exc:
        raise _ValidationBlocked(AttachmentValidationFailureCodeV2.VALIDATOR_FAILED) from exc


def _sanitize_legacy_findings(
    *,
    request: AttachmentValidationRequestV2,
    findings: list[ValidationFinding],
) -> tuple[AttachmentValidationFindingV2, ...]:
    values: list[AttachmentValidationFindingV2] = []
    for finding in findings:
        code = _legacy_finding_code(finding)
        if code is None:
            raise _ValidationBlocked(AttachmentValidationFailureCodeV2.LEGACY_FINDING_UNMAPPED)
        values.append(
            _finding(
                request=request,
                code=code,
                rule_ids=(
                    f"legacy-validator/{_safe_identifier_part(finding.finding_id.split('-', 1)[0])}/{finding.category.value}",
                ),
            )
        )
    return tuple(values)


def _legacy_finding_code(
    finding: ValidationFinding,
) -> AttachmentValidationFindingCodeV2 | None:
    if finding.category is FindingCategory.SECURITY:
        return AttachmentValidationFindingCodeV2.SECRET_DETECTED
    if finding.category is FindingCategory.ANSWER_LEAKAGE:
        return AttachmentValidationFindingCodeV2.FORBIDDEN_OUTPUT_MATCH
    if finding.category is FindingCategory.METADATA:
        return AttachmentValidationFindingCodeV2.METADATA_POLICY_VIOLATION
    if finding.category is FindingCategory.TRACEABILITY:
        return AttachmentValidationFindingCodeV2.LINEAGE_INCOMPLETE
    if finding.category is FindingCategory.STRUCTURE:
        if finding.finding_id.startswith("common-") and finding.severity is Severity.P0:
            return AttachmentValidationFindingCodeV2.OUTPUT_EMPTY
        return (
            AttachmentValidationFindingCodeV2.FORMAT_INVALID
            if finding.severity in {Severity.P0, Severity.P1}
            else AttachmentValidationFindingCodeV2.RENDER_CONTRACT_VIOLATION
        )
    if finding.category in {
        FindingCategory.COVERAGE,
        FindingCategory.EXECUTABILITY,
    }:
        return AttachmentValidationFindingCodeV2.CONTENT_CONTRACT_VIOLATION
    return None


def _scan_secret_findings(
    *,
    request: AttachmentValidationRequestV2,
    state: _ScanState,
) -> tuple[AttachmentValidationFindingV2, ...]:
    values: list[AttachmentValidationFindingV2] = []
    for projection in state.projections:
        if not projection.text:
            continue
        for name, pattern in SECRET_PATTERNS.items():
            if pattern.search(projection.text):
                values.append(
                    _finding(
                        request=request,
                        code=AttachmentValidationFindingCodeV2.SECRET_DETECTED,
                        inventory_member_id=projection.inventory_member_id,
                        rule_ids=(f"builtin-secret/{name.replace('_', '-')}/v1",),
                    )
                )
    return tuple(values)


def _scan_pii_findings(
    *,
    request: AttachmentValidationRequestV2,
    state: _ScanState,
    compiled_rules: tuple[tuple[str, re.Pattern[str]], ...],
) -> tuple[AttachmentValidationFindingV2, ...]:
    values: list[AttachmentValidationFindingV2] = []
    for projection in state.projections:
        if not projection.text:
            continue
        for rule_id, pattern in compiled_rules:
            if pattern.search(projection.text):
                values.append(
                    _finding(
                        request=request,
                        code=(AttachmentValidationFindingCodeV2.CONFIGURED_PII_DETECTED),
                        inventory_member_id=projection.inventory_member_id,
                        rule_ids=(rule_id,),
                    )
                )
    return tuple(values)


def _scan_forbidden_output_findings(
    *,
    request: AttachmentValidationRequestV2,
    state: _ScanState,
) -> tuple[AttachmentValidationFindingV2, ...]:
    values: list[AttachmentValidationFindingV2] = []
    for forbidden in request.forbidden_output_values:
        normalized = unicodedata.normalize("NFKC", forbidden).casefold()
        if not normalized:
            continue
        rule_id = f"forbidden-output/sha256/{hashlib.sha256(normalized.encode()).hexdigest()}"
        for projection in state.projections:
            path_match = (
                normalized
                in unicodedata.normalize(
                    "NFKC",
                    projection.normalized_path,
                ).casefold()
            )
            text_match = (
                normalized
                in unicodedata.normalize(
                    "NFKC",
                    projection.text,
                ).casefold()
            )
            if path_match or text_match:
                values.append(
                    _finding(
                        request=request,
                        code=AttachmentValidationFindingCodeV2.FORBIDDEN_OUTPUT_MATCH,
                        inventory_member_id=projection.inventory_member_id,
                        rule_ids=(rule_id,),
                    )
                )
    return tuple(values)


def _scan_fingerprint_findings(
    *,
    request: AttachmentValidationRequestV2,
    state: _ScanState,
) -> tuple[AttachmentValidationFindingV2, ...]:
    matched: dict[
        tuple[str | None, str],
        set[str],
    ] = {}
    for projection in state.projections:
        text_tokens = _tokenize(projection.text)
        path_tokens = _tokenize(projection.normalized_path)
        for fingerprint in request.leakage_fingerprints:
            tokens = (
                path_tokens
                if fingerprint.match_kind is AttachmentValidationFingerprintMatchKindV2.PATH_COMPONENT
                else text_tokens
            )
            if _fingerprint_matches(tokens, fingerprint):
                key = (
                    projection.inventory_member_id,
                    fingerprint.category.value,
                )
                matched.setdefault(key, set()).add(fingerprint.fingerprint_id)
    return tuple(
        _finding(
            request=request,
            code=AttachmentValidationFindingCodeV2.RESTRICTED_FINGERPRINT_MATCH,
            inventory_member_id=member_id,
            rule_ids=("attachment-validation/restricted-fingerprint/r5-08-v1",),
            matched_fingerprint_ids=tuple(sorted(fingerprint_ids)),
        )
        for (member_id, _), fingerprint_ids in sorted(
            matched.items(),
            key=lambda item: (item[0][0] or "", item[0][1]),
        )
    )


def _fingerprint_matches(
    tokens: tuple[str, ...],
    fingerprint: AttachmentValidationFingerprintV2,
) -> bool:
    count = fingerprint.token_count
    if count > len(tokens):
        return False
    return any(
        _digest_tokens(tokens[start : start + count]) == fingerprint.digest_sha256
        for start in range(0, len(tokens) - count + 1)
    )


def _finding(
    *,
    request: AttachmentValidationRequestV2,
    code: AttachmentValidationFindingCodeV2,
    inventory_member_id: str | None = None,
    rule_ids: tuple[str, ...] = (),
    matched_fingerprint_ids: tuple[str, ...] = (),
) -> AttachmentValidationFindingV2:
    severity, category, non_waivable = attachment_validation_finding_policy(code)
    finding = AttachmentValidationFindingV2(
        finding_id="attachment-validation-finding://pending",
        artifact_id=request.artifact_id,
        subject_output_ref=request.output_ref,
        inventory_member_id=inventory_member_id,
        severity=severity,
        category=category,
        code=code,
        rule_ids=tuple(sorted(set(rule_ids))),
        matched_fingerprint_ids=tuple(sorted(set(matched_fingerprint_ids))),
        non_waivable=non_waivable,
        finding_sha256="0" * 64,
    )
    digest = attachment_validation_finding_carried_sha256(finding)
    return finding.model_copy(
        update={
            "finding_id": f"attachment-validation-finding://sha256/{digest}",
            "finding_sha256": digest,
        }
    )


def _deduplicate_findings(
    findings: list[AttachmentValidationFindingV2],
) -> tuple[AttachmentValidationFindingV2, ...]:
    by_id = {item.finding_id: item for item in findings}
    return tuple(by_id[key] for key in sorted(by_id))


def _blocked_result(
    request: AttachmentValidationRequestV2,
    *,
    required_validator_ids: tuple[str, ...],
    code: AttachmentValidationFailureCodeV2,
) -> AttachmentValidationResultV2:
    result = AttachmentValidationResultV2(
        validation_result_id="attachment-validation-result://pending",
        validation_request_ref=attachment_validation_request_ref(request),
        artifact_id=request.artifact_id,
        output_ref=request.output_ref,
        output_sha256=request.output_sha256,
        status=AttachmentValidationStatusV2.BLOCKED,
        executed_validator_ids=(),
        required_validator_ids=required_validator_ids,
        complete_leakage_categories=request.complete_leakage_categories,
        findings=(),
        inventory_members=(),
        inventory_complete=False,
        scan_complete=False,
        failure_code=code,
        policy_version=ATTACHMENT_VALIDATION_POLICY_VERSION,
        validation_result_sha256="0" * 64,
    )
    return _finalize_result(result)


def _finalize_result(
    result: AttachmentValidationResultV2,
) -> AttachmentValidationResultV2:
    digest = attachment_validation_result_carried_sha256(result)
    return result.model_copy(
        update={
            "validation_result_id": (f"attachment-validation-result://sha256/{digest}"),
            "validation_result_sha256": digest,
        }
    )


def _text_from_file(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            with pymupdf.open(path) as document:  # type: ignore[no-untyped-call]
                values = [str(value or "") for value in document.metadata.values()]
                values.extend(document[index].get_text() for index in range(document.page_count))
                return "\n".join(values)
        except Exception as exc:
            raise _ValidationBlocked(AttachmentValidationFailureCodeV2.CONTENT_UNSCANNABLE) from exc
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise _ValidationBlocked(AttachmentValidationFailureCodeV2.CONTENT_UNSCANNABLE) from exc
    return _text_from_bytes(data, suffix=suffix)


def _text_from_bytes(
    data: bytes,
    *,
    suffix: str,
) -> str | None:
    if suffix not in _TEXT_SUFFIXES:
        if b"\x00" in data:
            raise _ValidationBlocked(AttachmentValidationFailureCodeV2.CONTENT_UNSCANNABLE)
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise _ValidationBlocked(AttachmentValidationFailureCodeV2.CONTENT_UNSCANNABLE) from exc
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise _ValidationBlocked(AttachmentValidationFailureCodeV2.CONTENT_UNSCANNABLE) from exc


def _normalize_archive_path(value: str) -> str:
    normalized = value.replace("\\", "/")
    if not normalized or normalized.startswith("/") or any(ord(character) < 32 for character in normalized):
        raise _ValidationBlocked(AttachmentValidationFailureCodeV2.PACKAGE_PATH_INVALID)
    path = PurePosixPath(normalized)
    if any(part in {"", ".", ".."} for part in path.parts) or (path.parts and path.parts[0].endswith(":")):
        raise _ValidationBlocked(AttachmentValidationFailureCodeV2.PACKAGE_PATH_INVALID)
    result = path.as_posix().rstrip("/")
    if not result or len(result) > 512:
        raise _ValidationBlocked(AttachmentValidationFailureCodeV2.PACKAGE_PATH_INVALID)
    return result


def _join_relative(root: str, child: str) -> str:
    value = f"{root.rstrip('/')}/{child.lstrip('/')}"
    if len(value) > 512:
        raise _ValidationBlocked(AttachmentValidationFailureCodeV2.PACKAGE_PATH_INVALID)
    return value


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _media_type(path: Path) -> str | None:
    return mimetypes.guess_type(path.name)[0]


def _tokenize(text: str) -> tuple[str, ...]:
    return tuple(
        normalized
        for match in _TOKEN_PATTERN.finditer(text)
        if (
            normalized := unicodedata.normalize(
                "NFKC",
                match.group(),
            ).casefold()
        )
    )


def _digest_tokens(tokens: tuple[str, ...]) -> str:
    return hashlib.sha256(" ".join(tokens).encode()).hexdigest()


def _safe_identifier_part(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._/-]+", "-", value).strip("-")
    return normalized or "validator"


def _inventory_member_key(
    member: AttachmentInventoryMemberV2,
) -> tuple[str, str, str]:
    return (
        member.container_ref.object_id if member.container_ref is not None else "",
        member.normalized_path,
        member.member_type.value,
    )
