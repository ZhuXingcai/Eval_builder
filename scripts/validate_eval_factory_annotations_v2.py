from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from jsonschema import FormatChecker, validators  # type: ignore[import-untyped]
from referencing import Registry, Resource

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = REPO_ROOT / "evals/annotation/v2/schemas"
CANARY_MANIFEST = REPO_ROOT / "evals/golden/eval_factory/canary_manifest.v4.json"
CONTRACT_MANIFEST = REPO_ROOT / "evals/annotation/v2/contract-manifest.json"
CROSS_STAGE_MANIFEST = REPO_ROOT / "specs/002-eval-dataset-factory/contracts/v2/manifest.json"

SCHEMA_FILES = {
    "PARSER": "parser-annotation.schema.json",
    "SAFETY": "safety-annotation.schema.json",
    "LABEL": "label-annotation.schema.json",
    "TASK": "task-annotation.schema.json",
    "ATTACHMENT": "attachment-annotation.schema.json",
    "RELEASE": "release-annotation.schema.json",
}
REQUIRED_CONTRACT_FILES = {
    "evals/annotation/v2/annotation-guide.md",
    "scripts/export_eval_factory_annotations_v2.py",
    "scripts/validate_eval_factory_annotations_v2.py",
    *{
        f"evals/annotation/v2/schemas/{file_name}"
        for file_name in {"common.schema.json", *SCHEMA_FILES.values()}
    },
}
HISTORICAL_WORKFLOW_FIELDS = {
    "adjudication",
    "adjudicator_id",
    "annotator_role",
    "required_review_roles",
    "review_records_current",
}

V1_VALIDATOR_PATH = REPO_ROOT / "scripts/validate_eval_factory_annotations.py"
V1_SPEC = importlib.util.spec_from_file_location(
    "validate_eval_factory_annotations_v1_base",
    V1_VALIDATOR_PATH,
)
if V1_SPEC is None or V1_SPEC.loader is None:
    raise RuntimeError(f"cannot load v1 annotation validator from {V1_VALIDATOR_PATH}")
v1 = importlib.util.module_from_spec(V1_SPEC)
sys.modules[V1_SPEC.name] = v1
V1_SPEC.loader.exec_module(v1)


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
    missing = sorted({"common.schema.json", *SCHEMA_FILES.values()} - set(documents))
    if missing:
        raise ValueError(f"missing annotation v2 schemas: {missing}")
    return registry, documents


def load_contract_manifest(path: Path) -> str:
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if not isinstance(manifest, dict):
        raise ValueError(f"{path}: contract manifest must be an object")
    expected_fields = {
        "schema_version",
        "contract_version",
        "guide_version",
        "architecture",
        "base_annotation_contract",
        "active_cross_stage_contract",
        "files",
        "frozen_canary_manifest",
        "historical_only_fields",
    }
    if set(manifest) != expected_fields:
        raise ValueError(f"{path}: annotation v2 manifest fields do not match contract")
    if manifest["schema_version"] != "eval-factory-annotation-contract-manifest/v2":
        raise ValueError(f"{path}: unsupported annotation contract manifest")
    if manifest["contract_version"] != "eval-factory-annotation/v2":
        raise ValueError(f"{path}: unsupported annotation contract version")
    if manifest["guide_version"] != "eval-factory-annotation-guide/v2":
        raise ValueError(f"{path}: unsupported annotation guide version")
    if set(manifest["historical_only_fields"]) != HISTORICAL_WORKFLOW_FIELDS:
        raise ValueError(f"{path}: historical-only field set drift")

    seen_paths: set[str] = set()
    for item in manifest["files"]:
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
        observed = hashlib.sha256(file_path.read_bytes()).hexdigest()
        if observed != expected_hash:
            raise ValueError(
                f"{path}: contract drift for {relative_path}: expected {expected_hash}, observed {observed}"
            )
    if seen_paths != REQUIRED_CONTRACT_FILES:
        raise ValueError(f"{path}: annotation v2 contract file set mismatch")

    cross_stage = manifest["active_cross_stage_contract"]
    observed_cross_stage = hashlib.sha256(CROSS_STAGE_MANIFEST.read_bytes()).hexdigest()
    if cross_stage != {
        "path": str(CROSS_STAGE_MANIFEST.relative_to(REPO_ROOT)),
        "sha256": observed_cross_stage,
    }:
        raise ValueError(f"{path}: active cross-stage contract binding drift")

    canary = manifest["frozen_canary_manifest"]
    observed_canary = hashlib.sha256(CANARY_MANIFEST.read_bytes()).hexdigest()
    if canary != {
        "path": str(CANARY_MANIFEST.relative_to(REPO_ROOT)),
        "sha256": observed_canary,
    }:
        raise ValueError(f"{path}: frozen canary binding drift")
    return hashlib.sha256(raw).hexdigest()


def _json_path(parts: Iterable[object]) -> str:
    rendered = "$"
    for part in parts:
        rendered += f"[{part}]" if isinstance(part, int) else f".{part}"
    return rendered


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


def _find_historical_fields(value: object, path: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key in HISTORICAL_WORKFLOW_FIELDS:
                found.append(child_path)
            found.extend(_find_historical_fields(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_find_historical_fields(child, f"{path}[{index}]"))
    return found


def _validate_v2_binding(
    record: dict[str, Any],
    prefix: str,
    contract_manifest_sha256: str,
) -> list[str]:
    errors: list[str] = []
    if record.get("contract_manifest_sha256") != contract_manifest_sha256:
        errors.append(f"{prefix}: contract_manifest_sha256 does not match annotation v2")
    governing = record.get("governing_versions")
    expected_contract_hash = hashlib.sha256(CROSS_STAGE_MANIFEST.read_bytes()).hexdigest()
    if (
        not isinstance(governing, dict)
        or governing.get("active_contract_manifest_sha256") != expected_contract_hash
    ):
        errors.append(f"{prefix}: active_contract_manifest_sha256 does not match v2")
    historical = _find_historical_fields(record)
    if historical:
        errors.append(f"{prefix}: historical independent-review fields are forbidden: {historical}")
    return errors


def _validate_release_v2(record: dict[str, Any], prefix: str) -> list[str]:
    if record.get("annotation_type") != "RELEASE":
        return []
    payload = record.get("payload")
    if not isinstance(payload, dict):
        return []
    errors: list[str] = []
    required = set(payload.get("required_checkpoints") or [])
    bindings = payload.get("checkpoint_decisions")
    if not isinstance(bindings, list):
        bindings = []
    bound_checkpoints = [
        binding.get("checkpoint")
        for binding in bindings
        if isinstance(binding, dict) and isinstance(binding.get("checkpoint"), str)
    ]
    satisfied = set(bound_checkpoints)
    if len(satisfied) != len(bound_checkpoints):
        errors.append(f"{prefix}: a checkpoint can bind at most one current user decision")
    if not required and bindings:
        errors.append(f"{prefix}: disabled or unsatisfied checkpoints cannot create synthetic decisions")
    if not satisfied.issubset(required):
        errors.append(f"{prefix}: checkpoint decisions must be required by policy")

    release_allowed = payload.get("release_allowed") is True
    if not release_allowed:
        return errors
    if payload.get("automated_quality_passed") is not True:
        errors.append(f"{prefix}: release requires passing automated quality gates")
    if required != satisfied:
        errors.append(f"{prefix}: release requires every configured user checkpoint")
    if payload.get("package_provenance_exact_set") is not True:
        errors.append(f"{prefix}: release requires exact package/provenance set equality")
    if payload.get("registry_isolated") is not True:
        errors.append(f"{prefix}: release requires channel-appropriate registry isolation")
    findings = payload.get("hard_gate_findings")
    if isinstance(findings, list) and any(
        isinstance(finding, dict)
        and finding.get("status") == "OPEN"
        and (finding.get("severity") in {"P0", "P1"} or finding.get("non_waivable") is True)
        for finding in findings
    ):
        errors.append(f"{prefix}: release cannot retain open blocking findings")
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
    require_reference_ready: bool = False,
) -> list[str]:
    registry, documents = load_schema_registry(SCHEMA_DIR)
    canaries = v1.load_canary_index(CANARY_MANIFEST)
    contract_manifest_sha256 = load_contract_manifest(CONTRACT_MANIFEST)
    errors: list[str] = []
    if require_reference_ready and not records:
        errors.append("bundle: reference bundle cannot be empty")
    seen_ids: set[str] = set()
    for index, record in enumerate(records):
        prefix = _record_prefix(record, index)
        annotation_id = record.get("annotation_id")
        if isinstance(annotation_id, str):
            if annotation_id in seen_ids:
                errors.append(f"{prefix}: duplicate annotation_id")
            seen_ids.add(annotation_id)
        errors.extend(_validate_schema(record, prefix, registry, documents))
        errors.extend(v1._validate_canary_binding(record, prefix, canaries))
        errors.extend(_validate_v2_binding(record, prefix, contract_manifest_sha256))
        errors.extend(v1._validate_evidence(record, prefix))
        if require_reference_ready and record.get("status") not in {
            "REFERENCE_READY",
            "SUPERSEDED",
        }:
            errors.append(f"{prefix}: reference bundle requires REFERENCE_READY or SUPERSEDED status")
        errors.extend(v1._validate_parser(record, prefix))
        errors.extend(v1._validate_label(record, prefix))
        errors.extend(v1._validate_safety(record, prefix))
        errors.extend(v1._validate_task(record, prefix))
        errors.extend(v1._validate_attachment(record, prefix))
        errors.extend(_validate_release_v2(record, prefix))
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Eval Factory annotation v2 JSONL.")
    parser.add_argument("annotations", type=Path)
    parser.add_argument("--require-reference-ready", action="store_true")
    args = parser.parse_args()

    try:
        records = v1.load_jsonl(args.annotations)
        errors = validate_records(
            records,
            require_reference_ready=args.require_reference_ready,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"validated {len(records)} annotation v2 record(s)")


if __name__ == "__main__":
    main()
