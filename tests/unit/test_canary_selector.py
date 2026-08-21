from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts/select_eval_factory_canaries.py"
MODULE_SPEC = importlib.util.spec_from_file_location("select_eval_factory_canaries", MODULE_PATH)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
selector = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = selector
MODULE_SPEC.loader.exec_module(selector)

ParseDiagnostic = selector.ParseDiagnostic
ScanResult = selector.ScanResult
add_size_traits = selector.add_size_traits
scan_trace = selector.scan_trace
select_canaries = selector.select_canaries
build_manifest = selector.build_manifest
manifest_item = selector._manifest_item
write_immutable = selector._write_immutable


def _write_outer(path: Path, request: str, response: str = "{}") -> None:
    path.write_text(
        json.dumps(
            {
                "sid": "must-not-be-exported",
                "request": request,
                "response": response,
                "extra": "{}",
            }
        ),
        encoding="utf-8",
    )


def test_scan_strict_request_uses_structured_events(tmp_path: Path) -> None:
    path = tmp_path / "LH_001_test.jsonl"
    request = {
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "name": "Read",
                        "input": {"file_path": "C:\\work\\input.txt"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "is_error": True,
                        "content": "result truncated",
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "name": "Read",
                        "input": {"file_path": "C:\\work\\input.txt"},
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "name": "Edit",
                        "input": {"file_path": "C:\\work\\input.txt"},
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "name": "Read",
                        "input": {"file_path": "C:\\work\\input.txt"},
                    }
                ],
            },
        ]
    }
    _write_outer(path, json.dumps(request))

    result = scan_trace(path, {"category": "test"})

    assert result.instance_id == "LH_001"
    assert result.signal_quality == "strict_events"
    assert result.parse["request"].status == "strict"
    assert "tool.file_read" in result.verified_traits
    assert "tool.file_edit" in result.verified_traits
    assert "tool.error_result" in result.verified_traits
    assert "candidate.truncation_signal" in result.annotation_candidates
    assert "flow.pre_mutation_read" in result.verified_traits
    assert "flow.post_mutation_read" in result.verified_traits
    assert result.metrics["pre_mutation_reads"] == 1
    assert result.metrics["post_mutation_reads"] == 1


def test_scan_malformed_request_marks_behavior_as_heuristic(tmp_path: Path) -> None:
    path = tmp_path / "LH_002_test.jsonl"
    request = (
        r'{"messages":[{"role":"assistant","content":[{"type":"tool_use",'
        r'"name":"WebSearch","input":{"query":"C:\oops"}}]}]}'
    )
    _write_outer(path, request)

    result = scan_trace(path, {})

    assert result.parse["request"].status == "invalid_escape"
    assert result.signal_quality == "heuristic_requires_annotation"
    assert "parse.invalid_escape" in result.verified_traits
    assert "tool.search" not in result.verified_traits
    assert "candidate.no_attachment_read" in result.annotation_candidates


def test_selection_is_deterministic_and_size_traits_are_stable() -> None:
    scans = [
        ScanResult(
            instance_id=f"LH_{index:03d}",
            raw_sha256=str(index) * 64,
            size_bytes=100 + index,
            metadata={"category": f"category-{index % 2}"},
            parse={"request": ParseDiagnostic(status="strict")},
            verified_traits=frozenset({"parse.strict_request", f"trait.{index}"}),
            annotation_candidates=frozenset(),
            signal_quality="strict_events",
            metrics={},
        )
        for index in range(24)
    ]
    sized = add_size_traits(scans)

    first = [item.instance_id for item in select_canaries(sized, 20)]
    second = [item.instance_id for item in select_canaries(sized, 20)]

    assert first == second
    assert len(first) == 20
    assert sum("size.small" in item.verified_traits for item in sized) == 6
    assert sum("size.large" in item.verified_traits for item in sized) == 7


def test_tool_definition_does_not_count_as_tool_call(tmp_path: Path) -> None:
    path = tmp_path / "LH_003_test.jsonl"
    request = {
        "messages": [{"role": "assistant", "content": [{"type": "text", "text": "done"}]}],
        "tools": [{"name": "WebSearch", "description": "tool definition only"}],
    }
    _write_outer(path, json.dumps(request))

    result = scan_trace(path, {})

    assert "tool.search" not in result.verified_traits
    assert "candidate.no_attachment_read" in result.annotation_candidates


def test_completion_claim_signal_survives_later_assistant_messages(tmp_path: Path) -> None:
    path = tmp_path / "LH_103_test.jsonl"
    request = {
        "messages": [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "The final deliverable is ready."}],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "A later progress update."}],
            },
        ]
    }
    _write_outer(path, json.dumps(request))

    result = scan_trace(path, {})

    assert result.metrics["completion_claim_signals"] == 1
    assert "candidate.final_output_claim" in result.annotation_candidates


def test_manifest_item_omits_sensitive_outer_fields_and_verifies_hash(tmp_path: Path) -> None:
    path = tmp_path / "LH_004_test.jsonl"
    request = {"messages": []}
    _write_outer(path, json.dumps(request))

    result = scan_trace(path, {})
    item_text = json.dumps(manifest_item(result))

    assert result.raw_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert "must-not-be-exported" not in item_text
    assert '"sid"' not in item_text
    assert '"account"' not in item_text
    assert '"request"' in item_text  # parse diagnostics only
    assert json.dumps(request) not in item_text


def test_target_count_boundaries() -> None:
    assert selector.MIN_TARGET_COUNT == 20
    assert selector.MAX_TARGET_COUNT == 30
    try:
        select_canaries([], 19)
    except ValueError as exc:
        assert "between 20 and 30" in str(exc)
    else:
        raise AssertionError("target count below R0 range should fail")


def test_insufficient_coverage_fails_hard(tmp_path: Path) -> None:
    for index in range(20):
        _write_outer(
            tmp_path / f"LH_{index:03d}_test.jsonl",
            json.dumps({"messages": []}),
        )

    try:
        build_manifest(tmp_path, 20)
    except RuntimeError as exc:
        assert "did not meet coverage targets" in str(exc)
    else:
        raise AssertionError("insufficient coverage should fail")


def test_uncovered_source_category_fails_hard(tmp_path: Path) -> None:
    manifest_rows = ["instance_id,category"]
    for index in range(21):
        instance_id = f"LH_{index:03d}"
        manifest_rows.append(f"{instance_id},category-{index}")
        _write_outer(
            tmp_path / f"{instance_id}_test.jsonl",
            json.dumps({"messages": []}),
        )
    (tmp_path / "manifest.csv").write_text(
        "\n".join(manifest_rows) + "\n",
        encoding="utf-8",
    )

    try:
        build_manifest(tmp_path, 20)
    except RuntimeError as exc:
        assert "did not cover source categories" in str(exc)
    else:
        raise AssertionError("uncovered source category should fail")


def test_immutable_output_is_noop_for_same_content_and_rejects_change(
    tmp_path: Path,
) -> None:
    output = tmp_path / "manifest.json"

    write_immutable(output, "same\n")
    write_immutable(output, "same\n")

    try:
        write_immutable(output, "different\n")
    except FileExistsError as exc:
        assert "refusing to overwrite immutable output" in str(exc)
    else:
        raise AssertionError("immutable output change should fail")
