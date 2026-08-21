from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from eval_factory.trace.models import (
    TraceProbeDiagnostic,
    TraceProbeResult,
    TraceProbeStatus,
)

RAW_TRAJ_V1_REQUIRED_FIELDS = frozenset(
    {
        "account",
        "api_type",
        "business",
        "endpoint",
        "event_time",
        "extra",
        "mm_urls",
        "model",
        "p_date",
        "request",
        "response",
        "sid",
        "source",
    }
)
RAW_TRAJ_V1_STRING_FIELDS = RAW_TRAJ_V1_REQUIRED_FIELDS - {"source"}
MAX_DIAGNOSTICS = 20


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def _source_uri(path: Path) -> str:
    try:
        return path.expanduser().resolve(strict=False).as_uri()
    except ValueError:
        return f"file://{path.expanduser().absolute()}"


def _diagnostic(
    code: str,
    message: str,
    line_number: int | None = None,
) -> TraceProbeDiagnostic:
    return TraceProbeDiagnostic(
        code=code,
        message=message,
        line_number=line_number,
    )


class RawTrajV1Adapter:
    name = "raw_traj_v1"
    version = "1.0.0"

    def probe(self, source: Path) -> TraceProbeResult:
        path = source.expanduser().absolute()
        source_uri = _source_uri(path)
        preflight = self._preflight(path, source_uri)
        if preflight is not None:
            return preflight

        try:
            before = path.stat()
            raw_sha256, size_bytes, record_count, fields, diagnostics = self._scan(path)
            after = path.stat()
        except (OSError, PermissionError) as exc:
            return TraceProbeResult(
                source_uri=source_uri,
                status=TraceProbeStatus.UNREADABLE,
                record_count=0,
                diagnostics=(
                    _diagnostic(
                        "raw-traj-read-error",
                        f"cannot read trace source: {type(exc).__name__}",
                    ),
                ),
            )

        if self._changed_during_probe(before, after, size_bytes):
            return TraceProbeResult(
                source_uri=source_uri,
                status=TraceProbeStatus.UNREADABLE,
                raw_sha256=raw_sha256,
                size_bytes=size_bytes,
                record_count=record_count,
                outer_fields=tuple(sorted(fields)),
                diagnostics=(
                    _diagnostic(
                        "raw-traj-source-changed",
                        "source metadata changed while it was being probed",
                    ),
                ),
            )
        if record_count == 0:
            diagnostics.append(
                _diagnostic(
                    "raw-traj-empty",
                    "trace source contains no non-empty JSONL records",
                )
            )
        status = TraceProbeStatus.INVALID if diagnostics else TraceProbeStatus.SUPPORTED
        return TraceProbeResult(
            source_uri=source_uri,
            status=status,
            raw_sha256=raw_sha256,
            size_bytes=size_bytes,
            record_count=record_count,
            outer_fields=tuple(sorted(fields)),
            diagnostics=tuple(diagnostics),
        )

    def _preflight(self, path: Path, source_uri: str) -> TraceProbeResult | None:
        if not path.exists():
            return TraceProbeResult(
                source_uri=source_uri,
                status=TraceProbeStatus.UNREADABLE,
                record_count=0,
                diagnostics=(_diagnostic("raw-traj-source-missing", "trace source does not exist"),),
            )
        if path.is_symlink():
            return TraceProbeResult(
                source_uri=source_uri,
                status=TraceProbeStatus.UNSUPPORTED,
                record_count=0,
                diagnostics=(
                    _diagnostic(
                        "raw-traj-symlink-unsupported",
                        "trace source must be a regular file, not a symbolic link",
                    ),
                ),
            )
        if not path.is_file():
            return TraceProbeResult(
                source_uri=source_uri,
                status=TraceProbeStatus.UNSUPPORTED,
                record_count=0,
                diagnostics=(
                    _diagnostic(
                        "raw-traj-not-regular-file",
                        "trace source must be a regular file",
                    ),
                ),
            )
        if path.suffix.casefold() != ".jsonl":
            return TraceProbeResult(
                source_uri=source_uri,
                status=TraceProbeStatus.UNSUPPORTED,
                record_count=0,
                diagnostics=(
                    _diagnostic(
                        "raw-traj-extension-mismatch",
                        "raw_traj_v1 sources must use the .jsonl extension",
                    ),
                ),
            )
        return None

    def _scan(
        self,
        path: Path,
    ) -> tuple[str, int, int, set[str], list[TraceProbeDiagnostic]]:
        hasher = hashlib.sha256()
        size_bytes = 0
        record_count = 0
        observed_fields: set[str] = set()
        diagnostics: list[TraceProbeDiagnostic] = []
        with path.open("rb") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                hasher.update(raw_line)
                size_bytes += len(raw_line)
                if not raw_line.strip():
                    continue
                record_count += 1
                if len(diagnostics) >= MAX_DIAGNOSTICS:
                    continue
                diagnostic, fields = self._validate_outer_record(raw_line, line_number)
                observed_fields.update(fields)
                if diagnostic is not None:
                    diagnostics.append(diagnostic)
        return hasher.hexdigest(), size_bytes, record_count, observed_fields, diagnostics

    @staticmethod
    def _validate_outer_record(
        raw_line: bytes,
        line_number: int,
    ) -> tuple[TraceProbeDiagnostic | None, set[str]]:
        try:
            value = json.loads(
                raw_line,
                parse_constant=_reject_json_constant,
            )
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
            return (
                _diagnostic(
                    "raw-traj-outer-json-invalid",
                    f"outer JSON record is invalid: {type(exc).__name__}",
                    line_number,
                ),
                set(),
            )
        if not isinstance(value, dict):
            return (
                _diagnostic(
                    "raw-traj-outer-not-object",
                    "outer JSONL record must be an object",
                    line_number,
                ),
                set(),
            )
        fields = {str(key) for key in value}
        missing = sorted(RAW_TRAJ_V1_REQUIRED_FIELDS - fields)
        if missing:
            return (
                _diagnostic(
                    "raw-traj-envelope-fields-missing",
                    f"outer record is missing required fields: {', '.join(missing)}",
                    line_number,
                ),
                fields,
            )
        invalid_types = sorted(
            field for field in RAW_TRAJ_V1_STRING_FIELDS if not isinstance(value.get(field), str)
        )
        if not (value.get("source") is None or isinstance(value.get("source"), str)):
            invalid_types.append("source")
        if invalid_types:
            return (
                _diagnostic(
                    "raw-traj-envelope-field-type-invalid",
                    f"outer record fields have invalid types: {', '.join(invalid_types)}",
                    line_number,
                ),
                fields,
            )
        return None, fields

    @staticmethod
    def _changed_during_probe(
        before: os.stat_result,
        after: os.stat_result,
        observed_size: int,
    ) -> bool:
        return (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or after.st_size != observed_size
        )
