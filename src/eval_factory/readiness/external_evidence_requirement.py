from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from eval_factory.contracts.agent_system_v2 import (
    EvaluationRequirementSpecV2,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef


class ExternalEvidenceRequirementError(RuntimeError):
    pass


class ExternalEvidenceRequirementLimitError(ExternalEvidenceRequirementError):
    pass


@dataclass(frozen=True, slots=True)
class ExternalRequirementCompilation:
    source_ref: ObjectRef
    source_sha256: str
    source_size_bytes: int
    media_type: str
    requirement: EvaluationRequirementSpecV2


class ExternalRequirementCompiler:
    def compile(
        self,
        *,
        source: Path,
        run_id: str,
        requirement_spec_id: str,
        goals: tuple[str, ...],
        constraints: tuple[str, ...],
        assumptions: tuple[str, ...],
        open_questions: tuple[str, ...],
        requirement_version: int,
        audit: ContractAudit,
        max_source_bytes: int,
    ) -> ExternalRequirementCompilation:
        if max_source_bytes < 1 or max_source_bytes > 10_000_000:
            raise ExternalEvidenceRequirementLimitError("external requirement byte limit is invalid")
        path = source.expanduser().absolute()
        if path.is_symlink():
            raise ExternalEvidenceRequirementError("external requirement source cannot be a symlink")
        if not path.is_file():
            raise ExternalEvidenceRequirementError("external requirement source must be a regular file")
        if path.suffix.casefold() not in {".md", ".markdown"}:
            raise ExternalEvidenceRequirementError("external requirement source must be Markdown")
        payload = _read_stable(path)
        if len(payload) > max_source_bytes:
            raise ExternalEvidenceRequirementLimitError("external requirement source exceeds the byte limit")
        try:
            payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ExternalEvidenceRequirementError("external requirement source is not UTF-8") from exc
        source_sha256 = hashlib.sha256(payload).hexdigest()
        source_ref = ObjectRef(
            object_type="evaluation-requirement-source",
            object_id=(f"evaluation-requirement-source://sha256/{source_sha256}"),
            object_version="v2",
            object_sha256=source_sha256,
        )
        try:
            requirement = EvaluationRequirementSpecV2.create(
                requirement_spec_id=requirement_spec_id,
                run_id=run_id,
                source_ref=source_ref,
                goals=goals,
                constraints=constraints,
                assumptions=assumptions,
                open_questions=open_questions,
                requirement_version=requirement_version,
                audit=audit,
            )
        except (ValidationError, ValueError) as exc:
            raise ExternalEvidenceRequirementError("external requirement summaries are invalid") from exc
        return ExternalRequirementCompilation(
            source_ref=source_ref,
            source_sha256=source_sha256,
            source_size_bytes=len(payload),
            media_type="text/markdown; charset=utf-8",
            requirement=requirement,
        )


def _read_stable(path: Path) -> bytes:
    try:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            payload = handle.read()
            after = os.fstat(handle.fileno())
        current = path.stat()
    except OSError as exc:
        raise ExternalEvidenceRequirementError("external requirement source is unreadable") from exc
    identity = _stat_identity(before)
    if (
        identity != _stat_identity(after)
        or identity != _stat_identity(current)
        or before.st_size != len(payload)
    ):
        raise ExternalEvidenceRequirementError("external requirement source changed during admission")
    return payload


def _stat_identity(
    value: os.stat_result,
) -> tuple[int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
    )


__all__ = [
    "ExternalEvidenceRequirementError",
    "ExternalEvidenceRequirementLimitError",
    "ExternalRequirementCompilation",
    "ExternalRequirementCompiler",
]
