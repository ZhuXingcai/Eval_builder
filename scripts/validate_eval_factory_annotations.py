from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from jsonschema import FormatChecker, validators  # type: ignore[import-untyped]
from referencing import Registry, Resource

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA_DIR = REPO_ROOT / "evals/annotation/v1/schemas"
DEFAULT_CANARY_MANIFEST = REPO_ROOT / "evals/golden/eval_factory/canary_manifest.v4.json"
DEFAULT_CONTRACT_MANIFEST = REPO_ROOT / "evals/annotation/v1/contract-manifest.json"
FROZEN_CANARY_PATH = "evals/golden/eval_factory/canary_manifest.v4.json"
FROZEN_CANARY_SHA256 = "1a409bbb92ef0fc075d8dd2144f36cfd5829976ab6718c20db3a8e9f2d61a6a6"

SCHEMA_FILES = {
    "PARSER": "parser-annotation.schema.json",
    "SAFETY": "safety-annotation.schema.json",
    "LABEL": "label-annotation.schema.json",
    "TASK": "task-annotation.schema.json",
    "ATTACHMENT": "attachment-annotation.schema.json",
    "RELEASE": "release-annotation.schema.json",
}
REQUIRED_CONTRACT_FILES = {
    "evals/annotation/v1/annotation-guide.md",
    "scripts/validate_eval_factory_annotations.py",
    *{
        f"evals/annotation/v1/schemas/{file_name}"
        for file_name in {"common.schema.json", *SCHEMA_FILES.values()}
    },
}

NON_WAIVABLE_TAINTS = {
    "FINAL_OUTPUT_DERIVED",
    "PRIVATE_REFERENCE_DERIVED",
    "GRADER_RULE_DERIVED",
}
NON_WAIVABLE_RISKS = {
    "ANSWER_BEARING",
    "HIDDEN_PASS_CONDITION",
    "SECRET",
}
FAIL_CLOSED_TAINTS = {"UNKNOWN_DERIVATION"}
FAIL_CLOSED_RISKS = {"UNSCANNABLE_CONTENT"}
ALLOW_DISPOSITIONS = {
    "ALLOW_INPUT_EVIDENCE",
    "ALLOW_STRUCTURE_ONLY",
    "ALLOW_EXTERNAL_LEAD_ONLY",
}


def load_schema_registry(schema_dir: Path) -> tuple[Registry, dict[str, dict[str, Any]]]:
    registry = Registry()
    documents: dict[str, dict[str, Any]] = {}
    for path in sorted(schema_dir.glob("*.schema.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        schema_id = document.get("$id")
        if not isinstance(schema_id, str):
            raise ValueError(f"{path}: missing string $id")
        validator_class = validators.validator_for(document)
        validator_class.check_schema(document)
        registry = registry.with_resource(schema_id, Resource.from_contents(document))
        documents[path.name] = document
    missing = sorted(set(SCHEMA_FILES.values()) - set(documents))
    if missing:
        raise ValueError(f"missing annotation schemas: {missing}")
    return registry, documents


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: annotation must be a JSON object")
        records.append(value)
    return records


def load_canary_index(path: Path) -> dict[str, dict[str, Any]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    items = manifest.get("items")
    if not isinstance(items, list):
        raise ValueError(f"{path}: items must be an array")
    return {
        str(item["instance_id"]): item
        for item in items
        if isinstance(item, dict) and isinstance(item.get("instance_id"), str)
    }


def load_contract_manifest(path: Path) -> str:
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if not isinstance(manifest, dict):
        raise ValueError(f"{path}: contract manifest must be an object")
    expected_keys = {
        "schema_version",
        "contract_version",
        "guide_version",
        "files",
        "frozen_canary_manifest",
    }
    if set(manifest) != expected_keys:
        raise ValueError(f"{path}: contract manifest fields must be exactly {sorted(expected_keys)}")
    if manifest.get("schema_version") != "eval-factory-annotation-contract-manifest/v1":
        raise ValueError(f"{path}: unsupported annotation contract manifest")
    if manifest.get("contract_version") != "eval-factory-annotation/v1":
        raise ValueError(f"{path}: unsupported annotation contract version")
    if manifest.get("guide_version") != "eval-factory-annotation-guide/v1":
        raise ValueError(f"{path}: unsupported annotation guide version")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError(f"{path}: files must be a non-empty array")
    seen_paths: set[str] = set()
    for item in files:
        if not isinstance(item, dict):
            raise ValueError(f"{path}: contract file entry must be an object")
        relative_path = item.get("path")
        expected_hash = item.get("sha256")
        if not isinstance(relative_path, str) or not isinstance(expected_hash, str):
            raise ValueError(f"{path}: contract file entry requires path and sha256")
        if relative_path in seen_paths:
            raise ValueError(f"{path}: duplicate contract file path {relative_path}")
        seen_paths.add(relative_path)
        file_path = (REPO_ROOT / relative_path).resolve()
        try:
            file_path.relative_to(REPO_ROOT.resolve())
        except ValueError as exc:
            raise ValueError(f"{path}: contract path escapes repository: {relative_path}") from exc
        actual_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(
                f"{path}: contract drift for {relative_path}: "
                f"expected {expected_hash}, observed {actual_hash}"
            )
    if seen_paths != REQUIRED_CONTRACT_FILES:
        missing = sorted(REQUIRED_CONTRACT_FILES - seen_paths)
        extra = sorted(seen_paths - REQUIRED_CONTRACT_FILES)
        raise ValueError(f"{path}: contract file set mismatch: missing={missing}, extra={extra}")
    frozen_canary = manifest.get("frozen_canary_manifest")
    if not isinstance(frozen_canary, dict):
        raise ValueError(f"{path}: frozen_canary_manifest must be an object")
    if frozen_canary.get("path") != FROZEN_CANARY_PATH:
        raise ValueError(f"{path}: frozen canary path does not match v1 contract")
    if frozen_canary.get("sha256") != FROZEN_CANARY_SHA256:
        raise ValueError(f"{path}: frozen canary SHA-256 does not match v1 contract")
    actual_canary_hash = hashlib.sha256(DEFAULT_CANARY_MANIFEST.read_bytes()).hexdigest()
    if actual_canary_hash != FROZEN_CANARY_SHA256:
        raise ValueError(
            f"{path}: frozen canary manifest drift: "
            f"expected {FROZEN_CANARY_SHA256}, observed {actual_canary_hash}"
        )
    return hashlib.sha256(raw).hexdigest()


def _json_path(parts: Iterable[object]) -> str:
    rendered = "$"
    for part in parts:
        if isinstance(part, int):
            rendered += f"[{part}]"
        else:
            rendered += f".{part}"
    return rendered


def _payload_evidence_refs(value: object) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key.endswith("_evidence_ref_ids") and isinstance(child, list):
                found.update(item for item in child if isinstance(item, str))
            else:
                found.update(_payload_evidence_refs(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_payload_evidence_refs(child))
    return found


def _record_prefix(record: dict[str, Any], index: int) -> str:
    annotation_id = record.get("annotation_id")
    return str(annotation_id) if isinstance(annotation_id, str) else f"record[{index}]"


def _validate_schema(
    record: dict[str, Any],
    prefix: str,
    registry: Registry,
    documents: dict[str, dict[str, Any]],
) -> list[str]:
    annotation_type = record.get("annotation_type")
    if annotation_type not in SCHEMA_FILES:
        return [f"{prefix}: unknown annotation_type {annotation_type!r}"]
    schema = documents[SCHEMA_FILES[str(annotation_type)]]
    validator_class = validators.validator_for(schema)
    validator = validator_class(
        schema,
        registry=registry,
        format_checker=FormatChecker(),
    )
    return [
        f"{prefix}:{_json_path(error.absolute_path)}: {error.message}"
        for error in sorted(validator.iter_errors(record), key=lambda item: list(item.absolute_path))
    ]


def _validate_canary_binding(
    record: dict[str, Any],
    prefix: str,
    canaries: dict[str, dict[str, Any]],
) -> list[str]:
    instance_id = record.get("canary_instance_id")
    canary = canaries.get(str(instance_id))
    if canary is None:
        return [f"{prefix}: canary_instance_id {instance_id!r} is not in the frozen manifest"]
    errors: list[str] = []
    if record.get("source_ref") != canary.get("source_ref"):
        errors.append(f"{prefix}: source_ref does not match frozen canary manifest")
    if record.get("source_sha256") != canary.get("raw_sha256"):
        errors.append(f"{prefix}: source_sha256 does not match frozen canary manifest")
    return errors


def _validate_contract_binding(
    record: dict[str, Any],
    prefix: str,
    contract_manifest_sha256: str,
) -> list[str]:
    if record.get("contract_manifest_sha256") != contract_manifest_sha256:
        return [f"{prefix}: contract_manifest_sha256 does not match the frozen annotation contract"]
    return []


def _validate_evidence(record: dict[str, Any], prefix: str) -> list[str]:
    evidence = record.get("evidence")
    if not isinstance(evidence, list):
        return []
    evidence_by_id = {
        item.get("evidence_ref_id"): item
        for item in evidence
        if isinstance(item, dict) and isinstance(item.get("evidence_ref_id"), str)
    }
    errors: list[str] = []
    if len(evidence_by_id) != len(evidence):
        errors.append(f"{prefix}: evidence_ref_id values must be unique")
    referenced = _payload_evidence_refs(record.get("payload"))
    missing = sorted(referenced - set(evidence_by_id))
    if missing:
        errors.append(f"{prefix}: payload references missing evidence IDs {missing}")
    if (
        record.get("annotation_type") in {"SAFETY", "RELEASE"}
        and evidence
        and all(item.get("approximate") is True for item in evidence if isinstance(item, dict))
    ):
        errors.append(f"{prefix}: high-risk conclusion cannot rely only on approximate evidence")
    return errors


def _validate_adjudication(
    record: dict[str, Any],
    prefix: str,
    records_by_id: dict[str, dict[str, Any]],
) -> list[str]:
    if record.get("status") != "ADJUDICATED":
        return []
    adjudication = record.get("adjudication")
    if not isinstance(adjudication, dict):
        return []
    errors: list[str] = []
    if adjudication.get("decision") not in {"ACCEPT", "REVISE"}:
        errors.append(f"{prefix}: ADJUDICATED record requires ACCEPT or REVISE decision")
    adjudicator_id = adjudication.get("adjudicator_id")
    if adjudicator_id == record.get("annotator_id"):
        errors.append(f"{prefix}: adjudicator must differ from the record annotator")
    input_ids = adjudication.get("input_annotation_ids")
    if not isinstance(input_ids, list):
        return errors
    required_inputs = 2 if record.get("annotation_type") in {"SAFETY", "RELEASE"} else 1
    if len(input_ids) < required_inputs:
        errors.append(
            f"{prefix}: {record.get('annotation_type')} adjudication requires "
            f"at least {required_inputs} input annotation(s)"
        )
    input_annotators: set[str] = set()
    for input_id in input_ids:
        source = records_by_id.get(str(input_id))
        if source is None:
            errors.append(f"{prefix}: adjudication input {input_id!r} is not in the JSONL bundle")
            continue
        if source is record:
            errors.append(f"{prefix}: adjudication cannot reference itself")
            continue
        if source.get("status") != "SUBMITTED" or source.get("adjudication") is not None:
            errors.append(
                f"{prefix}: adjudication input {input_id!r} must be an independent SUBMITTED record"
            )
        for field in ("annotation_type", "canary_instance_id", "subject_ref", "subject_sha256"):
            if source.get(field) != record.get(field):
                errors.append(f"{prefix}: adjudication input {input_id!r} has mismatched {field}")
        source_annotator = source.get("annotator_id")
        if isinstance(source_annotator, str):
            input_annotators.add(source_annotator)
    if adjudicator_id in input_annotators:
        errors.append(f"{prefix}: adjudicator must differ from every input annotator")
    if len(input_annotators) != len(input_ids):
        errors.append(f"{prefix}: adjudication inputs must come from distinct annotators")
    return errors


def _validate_label(record: dict[str, Any], prefix: str) -> list[str]:
    if record.get("annotation_type") != "LABEL":
        return []
    payload = record.get("payload")
    evidence = record.get("evidence")
    if not isinstance(payload, dict) or not isinstance(evidence, list):
        return []
    evidence_by_id = {item.get("evidence_ref_id"): item for item in evidence if isinstance(item, dict)}
    decision = payload.get("expected_decision")
    if decision == "MATCH" and not payload.get("positive_evidence_ref_ids"):
        return [f"{prefix}: MATCH requires positive evidence"]
    if decision != "NO_MATCH":
        return []
    errors: list[str] = []
    negative_ids = payload.get("negative_evidence_ref_ids")
    if not payload.get("structured_capability_complete"):
        errors.append(f"{prefix}: NO_MATCH requires complete structured capability")
    if not isinstance(negative_ids, list) or not negative_ids:
        errors.append(f"{prefix}: NO_MATCH requires negative evidence")
        return errors
    for evidence_id in negative_ids:
        item = evidence_by_id.get(evidence_id)
        if not isinstance(item, dict):
            continue
        if item.get("supports") != "NEGATIVE" or item.get("capability_complete") is not True:
            errors.append(
                f"{prefix}: negative evidence {evidence_id!r} must be NEGATIVE and capability-complete"
            )
    return errors


def _validate_parser(record: dict[str, Any], prefix: str) -> list[str]:
    if record.get("annotation_type") != "PARSER":
        return []
    payload = record.get("payload")
    if not isinstance(payload, dict):
        return []
    capability_truth = payload.get("capability_truth")
    predicates = payload.get("negative_predicate_capabilities")
    if not isinstance(capability_truth, dict) or not isinstance(predicates, list):
        return []
    incomplete = sorted(
        capability for capability in predicates if capability_truth.get(capability) != "COMPLETE"
    )
    if incomplete:
        return [
            f"{prefix}: negative predicates require COMPLETE parser capabilities; incomplete={incomplete}"
        ]
    return []


def _validate_safety(record: dict[str, Any], prefix: str) -> list[str]:
    if record.get("annotation_type") != "SAFETY":
        return []
    payload = record.get("payload")
    if not isinstance(payload, dict):
        return []
    errors: list[str] = []
    decisions = payload.get("subject_decisions")
    if not isinstance(decisions, list):
        return errors
    has_output_contamination = False
    has_answer_leakage = False
    for index, decision in enumerate(decisions):
        if not isinstance(decision, dict):
            continue
        taints = set(decision.get("taint_labels") or [])
        risks = set(decision.get("content_risk_labels") or [])
        has_output_contamination |= bool(
            taints & {"FINAL_OUTPUT_DERIVED", "PRIVATE_REFERENCE_DERIVED", "GRADER_RULE_DERIVED"}
        )
        has_answer_leakage |= bool(
            risks & {"ANSWER_BEARING", "HIDDEN_PASS_CONDITION"}
            or taints & {"FINAL_OUTPUT_DERIVED", "PRIVATE_REFERENCE_DERIVED", "GRADER_RULE_DERIVED"}
        )
        must_block = bool((taints & NON_WAIVABLE_TAINTS) or (risks & NON_WAIVABLE_RISKS))
        must_fail_closed = bool((taints & FAIL_CLOSED_TAINTS) or (risks & FAIL_CLOSED_RISKS))
        if must_block and decision.get("non_waivable") is not True:
            errors.append(f"{prefix}: subject_decisions[{index}] must be marked non_waivable")
        if must_block and decision.get("disposition") in ALLOW_DISPOSITIONS:
            errors.append(f"{prefix}: subject_decisions[{index}] cannot allow non-waivable content")
        if must_fail_closed and decision.get("disposition") in ALLOW_DISPOSITIONS:
            errors.append(
                f"{prefix}: subject_decisions[{index}] must fail closed for unknown or unscannable content"
            )
    if has_output_contamination and payload.get("known_output_contamination") != "PRESENT":
        errors.append(f"{prefix}: output-derived subject requires contamination=PRESENT")
    if has_answer_leakage and payload.get("answer_leakage") != "PRESENT":
        errors.append(f"{prefix}: answer-bearing subject requires answer_leakage=PRESENT")
    if payload.get("expected_package_disposition") == "ALLOW" and (
        has_output_contamination or has_answer_leakage or bool(payload.get("non_waivable_blockers"))
    ):
        errors.append(f"{prefix}: package cannot ALLOW known leakage or non-waivable blockers")
    return errors


def _validate_task(record: dict[str, Any], prefix: str) -> list[str]:
    if record.get("annotation_type") != "TASK":
        return []
    payload = record.get("payload")
    if not isinstance(payload, dict) or payload.get("task_status") != "USABLE":
        return []
    firewall = payload.get("selection_firewall")
    if not isinstance(firewall, dict) or not all(value is True for value in firewall.values()):
        return [f"{prefix}: USABLE task requires every selection-firewall check to pass"]
    lineage = payload.get("requirement_lineage")
    if isinstance(lineage, list) and any(
        isinstance(item, dict) and item.get("conflict_status") == "UNRESOLVED" for item in lineage
    ):
        return [f"{prefix}: USABLE task cannot retain unresolved requirement conflicts"]
    return []


def _validate_attachment(record: dict[str, Any], prefix: str) -> list[str]:
    if record.get("annotation_type") != "ATTACHMENT":
        return []
    payload = record.get("payload")
    if not isinstance(payload, dict):
        return []
    errors: list[str] = []
    artifacts = payload.get("artifacts")
    if isinstance(artifacts, list):
        for index, artifact in enumerate(artifacts):
            if not isinstance(artifact, dict):
                continue
            if (
                artifact.get("criticality") == "CRITICAL"
                and artifact.get("expected_outcome") == "OMIT_OPTIONAL"
            ):
                errors.append(f"{prefix}: critical artifacts cannot be omitted as optional")
            if (
                artifact.get("criticality") == "CRITICAL"
                and artifact.get("blocking_uncertainties")
                and artifact.get("expected_outcome") == "BUILD"
            ):
                errors.append(f"{prefix}: artifacts[{index}] cannot BUILD through blocking uncertainties")
    if payload.get("expected_package_disposition") == "BUILD":
        if payload.get("input_state_only") is not True:
            errors.append(f"{prefix}: BUILD package must be input-state-only")
        if payload.get("required_artifact_set_complete") is not True:
            errors.append(f"{prefix}: BUILD package requires a complete required artifact set")
    return errors


def _validate_release(record: dict[str, Any], prefix: str) -> list[str]:
    if record.get("annotation_type") != "RELEASE":
        return []
    payload = record.get("payload")
    if not isinstance(payload, dict) or payload.get("release_allowed") is not True:
        return []
    errors: list[str] = []
    if payload.get("package_provenance_exact_set") is not True:
        errors.append(f"{prefix}: release requires exact package/provenance set equality")
    if payload.get("review_records_current") is not True:
        errors.append(f"{prefix}: release requires current review records")
    if payload.get("registry_isolated") is not True:
        errors.append(f"{prefix}: release requires channel-appropriate registry isolation")
    findings = payload.get("hard_gate_findings")
    if isinstance(findings, list):
        for finding in findings:
            if not isinstance(finding, dict) or finding.get("status") != "OPEN":
                continue
            if finding.get("severity") in {"P0", "P1"} or finding.get("non_waivable") is True:
                errors.append(f"{prefix}: release cannot retain open blocking findings")
                break
    if payload.get("channel") == "PRODUCTION" and payload.get("production_attestation") is None:
        errors.append(f"{prefix}: production release requires an attestation")
    if payload.get("expected_action") not in {"APPROVE", "PUBLISH"}:
        errors.append(f"{prefix}: release_allowed requires APPROVE or PUBLISH action")
    if payload.get("expected_state") not in {"APPROVED", "RELEASED"}:
        errors.append(f"{prefix}: release_allowed requires APPROVED or RELEASED state")
    return errors


def validate_records(
    records: list[dict[str, Any]],
    *,
    require_adjudicated: bool = False,
) -> list[str]:
    registry, documents = load_schema_registry(DEFAULT_SCHEMA_DIR)
    canaries = load_canary_index(DEFAULT_CANARY_MANIFEST)
    contract_manifest_sha256 = load_contract_manifest(DEFAULT_CONTRACT_MANIFEST)
    records_by_id: dict[str, dict[str, Any]] = {}
    adjudication_input_ids: set[str] = set()
    for record in records:
        adjudication = record.get("adjudication")
        if record.get("status") != "ADJUDICATED" or not isinstance(adjudication, dict):
            continue
        input_ids = adjudication.get("input_annotation_ids")
        if isinstance(input_ids, list):
            adjudication_input_ids.update(str(input_id) for input_id in input_ids)
    errors: list[str] = []
    if require_adjudicated and not records:
        errors.append("bundle: gold bundle cannot be empty")
    for index, record in enumerate(records):
        prefix = _record_prefix(record, index)
        annotation_id = record.get("annotation_id")
        if isinstance(annotation_id, str):
            if annotation_id in records_by_id:
                errors.append(f"{prefix}: duplicate annotation_id")
            records_by_id[annotation_id] = record
        errors.extend(_validate_schema(record, prefix, registry, documents))
        errors.extend(_validate_canary_binding(record, prefix, canaries))
        errors.extend(_validate_contract_binding(record, prefix, contract_manifest_sha256))
        errors.extend(_validate_evidence(record, prefix))
        if (
            require_adjudicated
            and record.get("status") not in {"ADJUDICATED", "SUPERSEDED"}
            and annotation_id not in adjudication_input_ids
        ):
            errors.append(f"{prefix}: gold bundle requires ADJUDICATED or SUPERSEDED status")
    for index, record in enumerate(records):
        prefix = _record_prefix(record, index)
        errors.extend(_validate_adjudication(record, prefix, records_by_id))
        errors.extend(_validate_parser(record, prefix))
        errors.extend(_validate_label(record, prefix))
        errors.extend(_validate_safety(record, prefix))
        errors.extend(_validate_task(record, prefix))
        errors.extend(_validate_attachment(record, prefix))
        errors.extend(_validate_release(record, prefix))
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Eval Factory annotation JSONL.")
    parser.add_argument("annotations", type=Path)
    parser.add_argument("--require-adjudicated", action="store_true")
    args = parser.parse_args()

    try:
        records = load_jsonl(args.annotations)
        errors = validate_records(
            records,
            require_adjudicated=args.require_adjudicated,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"validated {len(records)} annotation record(s)")


if __name__ == "__main__":
    main()
