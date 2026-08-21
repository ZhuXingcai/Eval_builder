from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "eval-factory-canary-manifest/v4"
SELECTOR_VERSION = "4.0.0"
DEFAULT_TARGET_COUNT = 24
MIN_TARGET_COUNT = 20
MAX_TARGET_COUNT = 30

TOOL_USE_RE = re.compile(
    r'"type"\s*:\s*"tool_use".{0,400}?"name"\s*:\s*"([^"]+)"',
    re.DOTALL,
)
ERROR_RESULT_RE = re.compile(
    r'"type"\s*:\s*"tool_result".{0,2000}?"is_error"\s*:\s*true',
    re.DOTALL,
)
CONTINUATION_TERMS = (
    "continuation from",
    "continue from",
    "context window",
    "继续上次",
    "接着之前",
)
COMPLETION_TERMS = (
    "final output",
    "final deliverable",
    "completed the",
    "已完成",
    "最终交付",
    "交付完成",
)

VERIFIED_TARGETS: dict[str, int] = {
    "parse.strict_request": 10,
    "parse.invalid_escape": 6,
    "parse.invalid_unicode_escape": 3,
    "parse.invalid_response": 1,
    "size.large": 3,
    "size.small": 2,
    "tool.file_read": 6,
    "tool.file_write": 6,
    "tool.file_edit": 6,
    "tool.search": 3,
    "tool.fetch": 2,
    "tool.powershell": 1,
    "tool.error_result": 5,
    "flow.pre_mutation_read": 3,
    "flow.post_mutation_read": 3,
}

ANNOTATION_TARGETS: dict[str, int] = {
    "candidate.truncation_signal": 3,
    "candidate.continuation_signal": 3,
    "candidate.final_output_claim": 2,
    "candidate.no_attachment_read": 3,
}


@dataclass(frozen=True)
class ParseDiagnostic:
    status: str
    message: str | None = None
    position: int | None = None


@dataclass(frozen=True)
class ScanResult:
    instance_id: str
    raw_sha256: str
    size_bytes: int
    metadata: dict[str, str]
    parse: dict[str, ParseDiagnostic]
    verified_traits: frozenset[str]
    annotation_candidates: frozenset[str]
    signal_quality: str
    metrics: dict[str, Any]


def _strict_json(value: object) -> tuple[object | None, ParseDiagnostic]:
    if not isinstance(value, str):
        return None, ParseDiagnostic(status="missing_or_non_string")
    try:
        return json.loads(value), ParseDiagnostic(status="strict")
    except json.JSONDecodeError as exc:
        message = exc.msg
        if message == "Invalid \\escape":
            status = "invalid_escape"
        elif message == "Invalid \\uXXXX escape":
            status = "invalid_unicode_escape"
        else:
            status = "invalid_json"
        return None, ParseDiagnostic(status=status, message=message, position=exc.pos)


def _visible_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        value = block.get("text")
        if isinstance(value, str):
            parts.append(value)
    return "\n".join(parts)


def _normalize_tool(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def _tool_family(name: str) -> str | None:
    normalized = _normalize_tool(name)
    if normalized in {"read", "read_file"}:
        return "file_read"
    if normalized in {"write", "write_file"}:
        return "file_write"
    if normalized in {"edit", "edit_file", "multi_edit"}:
        return "file_edit"
    if normalized in {"websearch", "web_search", "search"}:
        return "search"
    if normalized in {"webfetch", "web_fetch", "fetch"}:
        return "fetch"
    if normalized == "powershell":
        return "powershell"
    return None


def _input_path(tool_input: object) -> str | None:
    if not isinstance(tool_input, dict):
        return None
    for key in ("path", "file_path", "filename"):
        value = tool_input.get(key)
        if isinstance(value, str) and value:
            return value.replace("\\", "/").casefold()
    return None


def _scan_strict_messages(request: dict[str, Any]) -> tuple[Counter[str], dict[str, int]]:
    tool_counts: Counter[str] = Counter()
    file_reads: set[str] = set()
    mutated_paths: set[str] = set()
    metrics = {
        "messages": 0,
        "error_results": 0,
        "truncation_signals": 0,
        "pre_mutation_reads": 0,
        "post_mutation_reads": 0,
        "continuation_signals": 0,
        "completion_claim_signals": 0,
    }
    messages = request.get("messages")
    if not isinstance(messages, list):
        return tool_counts, metrics
    metrics["messages"] = len(messages)
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        text = _visible_text(content).casefold()
        metrics["continuation_signals"] += sum(term in text for term in CONTINUATION_TERMS)
        if message.get("role") == "assistant" and text:
            metrics["completion_claim_signals"] += sum(term in text for term in COMPLETION_TERMS)
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "tool_use":
                name = str(block.get("name") or "")
                family = _tool_family(name)
                if family:
                    tool_counts[family] += 1
                path = _input_path(block.get("input"))
                if family == "file_read" and path:
                    if path in mutated_paths:
                        metrics["post_mutation_reads"] += 1
                    file_reads.add(path)
                elif family in {"file_write", "file_edit"} and path:
                    if path in file_reads:
                        metrics["pre_mutation_reads"] += 1
                    mutated_paths.add(path)
            elif block_type == "tool_result":
                if block.get("is_error") is True:
                    metrics["error_results"] += 1
                result_text = str(block.get("content") or "").casefold()
                if "truncat" in result_text:
                    metrics["truncation_signals"] += 1
    return tool_counts, metrics


def _scan_heuristic_messages(request_text: str) -> tuple[Counter[str], dict[str, int]]:
    message_region = request_text.split(',"tools":', 1)[0]
    tool_counts: Counter[str] = Counter()
    for name in TOOL_USE_RE.findall(message_region):
        family = _tool_family(name)
        if family:
            tool_counts[family] += 1
    folded = message_region.casefold()
    metrics = {
        "messages": 0,
        "error_results": len(ERROR_RESULT_RE.findall(message_region)),
        "truncation_signals": folded.count("truncat"),
        "pre_mutation_reads": 0,
        "post_mutation_reads": 0,
        "continuation_signals": sum(folded.count(term) for term in CONTINUATION_TERMS),
        "completion_claim_signals": sum(folded.count(term) for term in COMPLETION_TERMS),
    }
    return tool_counts, metrics


def _instance_id(path: Path) -> str:
    match = re.match(r"(LH_\d+)_", path.name)
    if not match:
        raise ValueError(f"cannot derive instance id from {path.name}")
    return match.group(1)


def _load_metadata(manifest_path: Path) -> dict[str, dict[str, str]]:
    if not manifest_path.exists():
        return {}
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        return {row["instance_id"]: row for row in csv.DictReader(handle)}


def scan_trace(path: Path, metadata: dict[str, str]) -> ScanResult:
    raw = path.read_bytes()
    outer = json.loads(raw)
    request, request_diagnostic = _strict_json(outer.get("request"))
    _, response_diagnostic = _strict_json(outer.get("response"))
    _, extra_diagnostic = _strict_json(outer.get("extra"))
    request_text = outer.get("request") if isinstance(outer.get("request"), str) else ""
    if isinstance(request, dict):
        tool_counts, metrics = _scan_strict_messages(request)
        signal_quality = "strict_events"
    else:
        tool_counts, metrics = _scan_heuristic_messages(request_text)
        signal_quality = "heuristic_requires_annotation"

    verified_traits: set[str] = {f"parse.request_{request_diagnostic.status}"}
    annotation_candidates: set[str] = set()
    if request_diagnostic.status == "strict":
        verified_traits.add("parse.strict_request")
    if request_diagnostic.status == "invalid_escape":
        verified_traits.add("parse.invalid_escape")
    if request_diagnostic.status == "invalid_unicode_escape":
        verified_traits.add("parse.invalid_unicode_escape")
    if response_diagnostic.status != "strict":
        verified_traits.add("parse.invalid_response")
    if signal_quality == "strict_events":
        for family, count in tool_counts.items():
            if count:
                verified_traits.add(f"tool.{family}")
        if metrics["error_results"]:
            verified_traits.add("tool.error_result")
        if metrics["pre_mutation_reads"]:
            verified_traits.add("flow.pre_mutation_read")
        if metrics["post_mutation_reads"]:
            verified_traits.add("flow.post_mutation_read")
    if metrics["truncation_signals"]:
        annotation_candidates.add("candidate.truncation_signal")
    if metrics["continuation_signals"]:
        annotation_candidates.add("candidate.continuation_signal")
    if metrics["completion_claim_signals"]:
        annotation_candidates.add("candidate.final_output_claim")
    if signal_quality != "strict_events" or tool_counts["file_read"] == 0:
        annotation_candidates.add("candidate.no_attachment_read")

    return ScanResult(
        instance_id=_instance_id(path),
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        size_bytes=len(raw),
        metadata={
            key: metadata[key]
            for key in ("business", "category", "pool_id", "pool_category")
            if metadata.get(key)
        },
        parse={
            "outer": ParseDiagnostic(status="strict"),
            "request": request_diagnostic,
            "response": response_diagnostic,
            "extra": extra_diagnostic,
        },
        verified_traits=frozenset(verified_traits),
        annotation_candidates=frozenset(annotation_candidates),
        signal_quality=signal_quality,
        metrics={"tool_counts": dict(sorted(tool_counts.items())), **metrics},
    )


def add_size_traits(scans: list[ScanResult]) -> list[ScanResult]:
    ordered = sorted(scan.size_bytes for scan in scans)
    small_limit = ordered[(len(ordered) - 1) // 4]
    large_limit = ordered[((len(ordered) - 1) * 3) // 4]
    updated: list[ScanResult] = []
    for scan in scans:
        verified_traits = set(scan.verified_traits)
        if scan.size_bytes <= small_limit:
            verified_traits.add("size.small")
        if scan.size_bytes >= large_limit:
            verified_traits.add("size.large")
        updated.append(
            ScanResult(
                instance_id=scan.instance_id,
                raw_sha256=scan.raw_sha256,
                size_bytes=scan.size_bytes,
                metadata=scan.metadata,
                parse=scan.parse,
                verified_traits=frozenset(verified_traits),
                annotation_candidates=scan.annotation_candidates,
                signal_quality=scan.signal_quality,
                metrics=scan.metrics,
            )
        )
    return updated


def select_canaries(scans: list[ScanResult], target_count: int) -> list[ScanResult]:
    if not MIN_TARGET_COUNT <= target_count <= MAX_TARGET_COUNT:
        raise ValueError(f"target_count must be between {MIN_TARGET_COUNT} and {MAX_TARGET_COUNT}")
    if target_count > len(scans):
        raise ValueError("target_count exceeds source count")
    verified_coverage: Counter[str] = Counter()
    candidate_coverage: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    selected: list[ScanResult] = []
    remaining = {scan.instance_id: scan for scan in scans}

    while len(selected) < target_count:

        def score(scan: ScanResult) -> tuple[int, int, int, int, str]:
            target_gain = sum(
                max(0, minimum - verified_coverage[trait])
                for trait, minimum in VERIFIED_TARGETS.items()
                if trait in scan.verified_traits
            )
            candidate_gain = sum(
                max(0, minimum - candidate_coverage[trait])
                for trait, minimum in ANNOTATION_TARGETS.items()
                if trait in scan.annotation_candidates
            )
            category = scan.metadata.get("category") or scan.metadata.get("pool_category") or ""
            category_gain = 3 if category and categories[category] == 0 else 0
            exact_gain = 2 if scan.signal_quality == "strict_events" else 0
            rare_gain = sum(1 for trait in scan.verified_traits if verified_coverage[trait] == 0)
            return (
                target_gain,
                candidate_gain,
                category_gain,
                exact_gain + rare_gain,
                scan.instance_id,
            )

        candidate = max(remaining.values(), key=score)
        selected.append(candidate)
        remaining.pop(candidate.instance_id)
        verified_coverage.update(candidate.verified_traits)
        candidate_coverage.update(candidate.annotation_candidates)
        category = candidate.metadata.get("category") or candidate.metadata.get("pool_category")
        if category:
            categories[category] += 1
    return sorted(selected, key=lambda item: item.instance_id)


def _diagnostic_dict(value: ParseDiagnostic) -> dict[str, object]:
    return {
        key: item
        for key, item in {
            "status": value.status,
            "message": value.message,
            "position": value.position,
        }.items()
        if item is not None
    }


def _manifest_item(scan: ScanResult) -> dict[str, object]:
    return {
        "instance_id": scan.instance_id,
        "source_ref": f"raw_traj://{scan.instance_id}",
        "raw_sha256": scan.raw_sha256,
        "size_bytes": scan.size_bytes,
        "metadata": scan.metadata,
        "parse": {key: _diagnostic_dict(value) for key, value in scan.parse.items()},
        "verified_traits": sorted(scan.verified_traits),
        "annotation_candidates": sorted(scan.annotation_candidates),
        "signal_quality": scan.signal_quality,
        "requires_annotation": True,
        "metrics": scan.metrics,
    }


def build_manifest(source_root: Path, target_count: int) -> dict[str, object]:
    metadata = _load_metadata(source_root / "manifest.csv")
    scans = [
        scan_trace(path, metadata.get(_instance_id(path), {})) for path in sorted(source_root.glob("*.jsonl"))
    ]
    scans = add_size_traits(scans)
    selected = select_canaries(scans, target_count)
    source_categories = Counter(
        category
        for scan in scans
        if (category := scan.metadata.get("category") or scan.metadata.get("pool_category"))
    )
    selected_categories = Counter(
        category
        for scan in selected
        if (category := scan.metadata.get("category") or scan.metadata.get("pool_category"))
    )
    uncovered_categories = sorted(set(source_categories) - set(selected_categories))
    if uncovered_categories:
        raise RuntimeError(f"selection did not cover source categories: {uncovered_categories}")
    verified_coverage = Counter(trait for scan in selected for trait in scan.verified_traits)
    candidate_coverage = Counter(trait for scan in selected for trait in scan.annotation_candidates)
    unmet_verified = {
        trait: {"required": minimum, "observed": verified_coverage[trait]}
        for trait, minimum in VERIFIED_TARGETS.items()
        if verified_coverage[trait] < minimum
    }
    unmet_candidates = {
        trait: {"required": minimum, "observed": candidate_coverage[trait]}
        for trait, minimum in ANNOTATION_TARGETS.items()
        if candidate_coverage[trait] < minimum
    }
    if unmet_verified or unmet_candidates:
        raise RuntimeError(
            "selection did not meet coverage targets: "
            f"verified={unmet_verified}, annotation_candidates={unmet_candidates}"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "selector_version": SELECTOR_VERSION,
        "source_count": len(scans),
        "target_count": target_count,
        "source_reference": "raw_traj://approved-local-manifest",
        "processing_class": "RESTRICTED_TRACE_RAW",
        "lifecycle_profile": "R0_MANIFEST_ONLY_V1",
        "selection_policy": {
            "verified_targets": VERIFIED_TARGETS,
            "annotation_candidate_targets": ANNOTATION_TARGETS,
            "behavioral_signal_rule": (
                "Only verified_traits satisfy deterministic coverage. Annotation candidates, "
                "including all malformed-request behavioral signals and semantic text signals, "
                "are selection-only and cannot be used as label or safety truth."
            ),
        },
        "verified_coverage": dict(sorted(verified_coverage.items())),
        "annotation_candidate_coverage": dict(sorted(candidate_coverage.items())),
        "source_category_counts": dict(sorted(source_categories.items())),
        "selected_category_counts": dict(sorted(selected_categories.items())),
        "items": [_manifest_item(scan) for scan in selected],
    }


def render_report(manifest: dict[str, object]) -> str:
    verified_coverage = manifest["verified_coverage"]
    assert isinstance(verified_coverage, dict)
    candidate_coverage = manifest["annotation_candidate_coverage"]
    assert isinstance(candidate_coverage, dict)
    source_categories = manifest["source_category_counts"]
    assert isinstance(source_categories, dict)
    selected_categories = manifest["selected_category_counts"]
    assert isinstance(selected_categories, dict)
    items = manifest["items"]
    assert isinstance(items, list)
    lines = [
        "# Eval Dataset Factory Development Canary Selection",
        "",
        f"- Schema: `{manifest['schema_version']}`",
        f"- Selector: `{manifest['selector_version']}`",
        f"- Source traces: {manifest['source_count']}",
        f"- Selected traces: {manifest['target_count']}",
        "- Raw SID, account, prompt, response, and file content are not emitted.",
        "- Every selected item requires R0-08 annotation.",
        "- Only verified traits count as deterministic coverage.",
        "- Semantic and malformed-request signals are annotation candidates, not truth.",
        "",
        "## Verified Coverage",
        "",
        "| Trait | Required | Observed |",
        "|---|---:|---:|",
    ]
    selection_policy = manifest["selection_policy"]
    assert isinstance(selection_policy, dict)
    targets = selection_policy["verified_targets"]
    assert isinstance(targets, dict)
    for trait, required in sorted(targets.items()):
        lines.append(f"| `{trait}` | {required} | {verified_coverage.get(trait, 0)} |")
    candidate_targets = selection_policy["annotation_candidate_targets"]
    assert isinstance(candidate_targets, dict)
    lines.extend(
        [
            "",
            "## Annotation Candidate Coverage",
            "",
            "| Candidate | Required | Observed |",
            "|---|---:|---:|",
        ]
    )
    for trait, required in sorted(candidate_targets.items()):
        lines.append(f"| `{trait}` | {required} | {candidate_coverage.get(trait, 0)} |")
    lines.extend(
        [
            "",
            "## Selected Items",
            "",
            "| Instance | Request parse | Signal quality | Verified traits | Annotation candidates |",
            "|---|---|---|---|---|",
        ]
    )
    for item in items:
        assert isinstance(item, dict)
        parse = item["parse"]
        assert isinstance(parse, dict)
        request = parse["request"]
        assert isinstance(request, dict)
        trait_values = item["verified_traits"]
        assert isinstance(trait_values, list)
        traits = ", ".join(f"`{value}`" for value in trait_values)
        candidate_values = item["annotation_candidates"]
        assert isinstance(candidate_values, list)
        candidates = ", ".join(f"`{value}`" for value in candidate_values)
        lines.append(
            f"| `{item['instance_id']}` | `{request['status']}` | "
            f"`{item['signal_quality']}` | {traits} | {candidates} |"
        )
    lines.extend(
        [
            "",
            "## Business Category Coverage",
            "",
            "| Category | Source | Selected |",
            "|---|---:|---:|",
        ]
    )
    for category, source_count in sorted(source_categories.items()):
        lines.append(f"| {category} | {source_count} | {selected_categories.get(category, 0)} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This is a development-canary manifest, not a quality benchmark or gold label set. "
            "R0-08 must adjudicate every item and add annotation truth. In particular, "
            "`candidate.no_attachment_read` means no normalized file-read tool event was observed; "
            "it does not prove the task had no attachment read requirement. R8 uses separate frozen "
            "independent datasets for statistical claims.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--target-count", type=int, default=DEFAULT_TARGET_COUNT)
    args = parser.parse_args()
    manifest = build_manifest(args.source_root, args.target_count)
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    report_text = render_report(manifest)
    _write_immutable(args.output, manifest_text)
    _write_immutable(args.report, report_text)


def _write_immutable(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing == content:
            return
        raise FileExistsError(f"refusing to overwrite immutable output {path}; use a versioned path")
    path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
