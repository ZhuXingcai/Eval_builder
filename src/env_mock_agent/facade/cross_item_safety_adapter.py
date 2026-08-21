from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

from env_mock_agent.facade.cross_item_safety_v2 import (
    ATTACHMENT_CROSS_ITEM_SAFETY_POLICY_VERSION,
    AttachmentCrossItemSafetyScanFacade,
    AttachmentCrossItemSafetyScanFailureCodeV2,
    AttachmentCrossItemSafetyScanRequestV2,
    AttachmentCrossItemSafetyScanResultV2,
    AttachmentCrossItemSafetyScanStatusV2,
    attachment_cross_item_safety_scan_request_ref,
    attachment_cross_item_safety_scan_result_carried_sha256,
    validate_attachment_cross_item_safety_scan_request_identity,
)
from env_mock_agent.facade.duplicate_adapter import (
    AttachmentTextExtractionError,
    AttachmentTextExtractionFailureReason,
    extract_attachment_text_projections,
    validate_attachment_output_material,
)
from env_mock_agent.facade.duplicate_v2 import (
    AttachmentDuplicateFingerprintFailureCodeV2,
)
from env_mock_agent.facade.validation_v2 import (
    PROMPT_LEAKAGE_NORMALIZATION_VERSION,
    AttachmentValidationFingerprintMatchKindV2,
)
from env_mock_agent.runtimes.security import is_path_inside

ATTACHMENT_CROSS_ITEM_SAFETY_EXTRACTOR_VERSION = "attachment-cross-item-safety-extractor/r7-02-v1"

_TOKEN_PATTERN = re.compile(
    r"[\w]+(?:[._/-][\w]+)*",
    flags=re.UNICODE,
)


class _ScanBlocked(RuntimeError):
    def __init__(
        self,
        code: AttachmentCrossItemSafetyScanFailureCodeV2,
    ) -> None:
        super().__init__(code.value)
        self.code = code


class AttachmentCrossItemSafetyMaterialResolver(Protocol):
    def resolve_output(
        self,
        request: AttachmentCrossItemSafetyScanRequestV2,
    ) -> Path: ...


class MappingAttachmentCrossItemSafetyMaterialResolver:
    def __init__(
        self,
        *,
        outputs: Mapping[str, Path],
    ) -> None:
        self._outputs = dict(outputs)

    def resolve_output(
        self,
        request: AttachmentCrossItemSafetyScanRequestV2,
    ) -> Path:
        try:
            return self._outputs[request.output_ref.object_id].expanduser()
        except KeyError as exc:
            raise _ScanBlocked(AttachmentCrossItemSafetyScanFailureCodeV2.MATERIAL_NOT_FOUND) from exc


class StagingAttachmentCrossItemSafetyMaterialResolver:
    def __init__(
        self,
        *,
        staging_root: Path,
        outputs: Mapping[str, Path],
    ) -> None:
        self._staging_root = staging_root.expanduser().resolve()
        self._outputs = dict(outputs)

    def resolve_output(
        self,
        request: AttachmentCrossItemSafetyScanRequestV2,
    ) -> Path:
        try:
            target = self._outputs[request.output_ref.object_id].expanduser()
        except KeyError as exc:
            raise _ScanBlocked(AttachmentCrossItemSafetyScanFailureCodeV2.MATERIAL_NOT_FOUND) from exc
        resolved = target.resolve(strict=False)
        if not is_path_inside(self._staging_root, resolved):
            raise _ScanBlocked(AttachmentCrossItemSafetyScanFailureCodeV2.OUTPUT_PATH_VIOLATION)
        return target


class RegistryAttachmentCrossItemSafetyScanFacade(AttachmentCrossItemSafetyScanFacade):
    def __init__(
        self,
        *,
        resolver: AttachmentCrossItemSafetyMaterialResolver,
    ) -> None:
        self._resolver = resolver
        self._cache: dict[
            str,
            tuple[str, AttachmentCrossItemSafetyScanResultV2],
        ] = {}

    async def scan(
        self,
        request: AttachmentCrossItemSafetyScanRequestV2,
    ) -> AttachmentCrossItemSafetyScanResultV2:
        validate_attachment_cross_item_safety_scan_request_identity(request)
        cached = self._cache.get(request.idempotency_key)
        if cached is not None:
            request_sha256, result = cached
            if request_sha256 == request.request_sha256:
                return result
            return _result(
                request,
                status=AttachmentCrossItemSafetyScanStatusV2.BLOCKED,
                failure_code=(AttachmentCrossItemSafetyScanFailureCodeV2.IDEMPOTENCY_CONFLICT),
            )

        try:
            _validate_fingerprint_policy(request)
            path = self._resolver.resolve_output(request)
            validate_attachment_output_material(path, request)
            text_projections, member_paths, member_count = extract_attachment_text_projections(
                path,
                request,
                require_all_members_scannable=True,
            )
            validate_attachment_output_material(path, request)
            matched_ids = _matched_fingerprint_ids(
                request,
                text_projections=text_projections,
                member_paths=(
                    request.logical_path,
                    *member_paths,
                ),
            )
            if len(matched_ids) > request.limits.max_match_count:
                raise _ScanBlocked(AttachmentCrossItemSafetyScanFailureCodeV2.SCAN_LIMIT_EXCEEDED)
            result = _result(
                request,
                status=AttachmentCrossItemSafetyScanStatusV2.PASSED,
                matched_fingerprint_ids=matched_ids,
                scanned_member_count=member_count,
                extractor_version=(ATTACHMENT_CROSS_ITEM_SAFETY_EXTRACTOR_VERSION),
            )
        except _ScanBlocked as exc:
            result = _result(
                request,
                status=AttachmentCrossItemSafetyScanStatusV2.BLOCKED,
                failure_code=exc.code,
            )
        except AttachmentTextExtractionError as exc:
            result = _result(
                request,
                status=AttachmentCrossItemSafetyScanStatusV2.BLOCKED,
                failure_code=_map_extraction_failure(exc),
            )
        except Exception:
            result = _result(
                request,
                status=AttachmentCrossItemSafetyScanStatusV2.BLOCKED,
                failure_code=(AttachmentCrossItemSafetyScanFailureCodeV2.CONTENT_UNSCANNABLE),
            )
        self._cache[request.idempotency_key] = (
            request.request_sha256,
            result,
        )
        return result


def _validate_fingerprint_policy(
    request: AttachmentCrossItemSafetyScanRequestV2,
) -> None:
    if request.normalization_version != PROMPT_LEAKAGE_NORMALIZATION_VERSION or any(
        value.normalization_version != PROMPT_LEAKAGE_NORMALIZATION_VERSION
        for value in request.foreign_fingerprints
    ):
        raise _ScanBlocked(AttachmentCrossItemSafetyScanFailureCodeV2.UNKNOWN_FINGERPRINT_NORMALIZATION)


def _matched_fingerprint_ids(
    request: AttachmentCrossItemSafetyScanRequestV2,
    *,
    text_projections: tuple[str, ...],
    member_paths: tuple[str, ...],
) -> tuple[str, ...]:
    text_token_sets = tuple(_tokenize(value) for value in text_projections)
    path_token_sets = tuple(_tokenize(value) for value in member_paths)
    indexes: dict[
        AttachmentValidationFingerprintMatchKindV2,
        dict[tuple[int, str], set[str]],
    ] = {kind: {} for kind in AttachmentValidationFingerprintMatchKindV2}
    for fingerprint in request.foreign_fingerprints:
        indexes[fingerprint.match_kind].setdefault(
            (
                fingerprint.token_count,
                fingerprint.digest_sha256,
            ),
            set(),
        ).add(fingerprint.fingerprint_id)
    matched: set[str] = set()
    for kind, index in indexes.items():
        projections = (
            path_token_sets
            if kind is AttachmentValidationFingerprintMatchKindV2.PATH_COMPONENT
            else text_token_sets
        )
        for tokens in projections:
            for token_count in sorted({key[0] for key in index}):
                if token_count > len(tokens):
                    continue
                for start in range(0, len(tokens) - token_count + 1):
                    digest = _digest_tokens(tokens[start : start + token_count])
                    matched.update(index.get((token_count, digest), ()))
                    if len(matched) > request.limits.max_match_count:
                        raise _ScanBlocked(AttachmentCrossItemSafetyScanFailureCodeV2.SCAN_LIMIT_EXCEEDED)
    return tuple(sorted(matched))


def _tokenize(text: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return tuple(match.group(0) for match in _TOKEN_PATTERN.finditer(normalized))


def _digest_tokens(tokens: tuple[str, ...]) -> str:
    return hashlib.sha256(" ".join(tokens).encode("utf-8")).hexdigest()


def _map_extraction_failure(
    error: AttachmentTextExtractionError,
) -> AttachmentCrossItemSafetyScanFailureCodeV2:
    if error.reason is AttachmentTextExtractionFailureReason.CONTAINER_ENCRYPTED:
        return AttachmentCrossItemSafetyScanFailureCodeV2.CONTAINER_ENCRYPTED
    if error.reason is AttachmentTextExtractionFailureReason.CONTAINER_TRUNCATED:
        return AttachmentCrossItemSafetyScanFailureCodeV2.CONTAINER_TRUNCATED
    code = error.code
    direct = {
        AttachmentDuplicateFingerprintFailureCodeV2.MATERIAL_NOT_FOUND: (
            AttachmentCrossItemSafetyScanFailureCodeV2.MATERIAL_NOT_FOUND
        ),
        AttachmentDuplicateFingerprintFailureCodeV2.MATERIAL_MISMATCH: (
            AttachmentCrossItemSafetyScanFailureCodeV2.MATERIAL_MISMATCH
        ),
        AttachmentDuplicateFingerprintFailureCodeV2.OUTPUT_HASH_MISMATCH: (
            AttachmentCrossItemSafetyScanFailureCodeV2.OUTPUT_HASH_MISMATCH
        ),
        AttachmentDuplicateFingerprintFailureCodeV2.OUTPUT_PATH_VIOLATION: (
            AttachmentCrossItemSafetyScanFailureCodeV2.OUTPUT_PATH_VIOLATION
        ),
        AttachmentDuplicateFingerprintFailureCodeV2.SYMLINK_UNSUPPORTED: (
            AttachmentCrossItemSafetyScanFailureCodeV2.SYMLINK_UNSUPPORTED
        ),
        AttachmentDuplicateFingerprintFailureCodeV2.SCAN_LIMIT_EXCEEDED: (
            AttachmentCrossItemSafetyScanFailureCodeV2.SCAN_LIMIT_EXCEEDED
        ),
    }
    if code in {
        AttachmentDuplicateFingerprintFailureCodeV2.MEDIA_UNSUPPORTED,
        AttachmentDuplicateFingerprintFailureCodeV2.CONTAINER_UNSUPPORTED,
    }:
        return AttachmentCrossItemSafetyScanFailureCodeV2.CONTAINER_UNSUPPORTED
    return direct.get(
        code,
        AttachmentCrossItemSafetyScanFailureCodeV2.CONTENT_UNSCANNABLE,
    )


def _result(
    request: AttachmentCrossItemSafetyScanRequestV2,
    *,
    status: AttachmentCrossItemSafetyScanStatusV2,
    matched_fingerprint_ids: tuple[str, ...] = (),
    scanned_member_count: int = 0,
    failure_code: AttachmentCrossItemSafetyScanFailureCodeV2 | None = None,
    extractor_version: str | None = None,
) -> AttachmentCrossItemSafetyScanResultV2:
    pending = AttachmentCrossItemSafetyScanResultV2(
        scan_result_id="attachment-cross-item-safety-scan-result://pending",
        scan_request_ref=attachment_cross_item_safety_scan_request_ref(request),
        environment_artifact_ref=request.environment_artifact_ref,
        output_ref=request.output_ref,
        output_sha256=request.content_sha256,
        status=status,
        matched_fingerprint_ids=matched_fingerprint_ids,
        scanned_member_count=scanned_member_count,
        scan_complete=status is AttachmentCrossItemSafetyScanStatusV2.PASSED,
        failure_code=failure_code,
        extractor_version=extractor_version,
        normalization_version=PROMPT_LEAKAGE_NORMALIZATION_VERSION,
        policy_version=ATTACHMENT_CROSS_ITEM_SAFETY_POLICY_VERSION,
        result_sha256="0" * 64,
    )
    digest = attachment_cross_item_safety_scan_result_carried_sha256(pending)
    return pending.model_copy(
        update={
            "scan_result_id": (f"attachment-cross-item-safety-scan-result://sha256/{digest}"),
            "result_sha256": digest,
        }
    )
