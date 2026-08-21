from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, cast

REPO_ROOT = Path(__file__).resolve().parents[1]
V1_ROOT = REPO_ROOT / "evals/annotation/v1"
OUTPUT_ROOT = REPO_ROOT / "evals/annotation/v2"
SCHEMA_ROOT = OUTPUT_ROOT / "schemas"
MANIFEST_PATH = OUTPUT_ROOT / "contract-manifest.json"
GUIDE_PATH = OUTPUT_ROOT / "annotation-guide.md"
VALIDATOR_PATH = REPO_ROOT / "scripts/validate_eval_factory_annotations_v2.py"
BASE_MANIFEST_PATH = V1_ROOT / "contract-manifest.json"
CANARY_PATH = REPO_ROOT / "evals/golden/eval_factory/canary_manifest.v4.json"
CROSS_STAGE_MANIFEST = REPO_ROOT / "specs/002-eval-dataset-factory/contracts/v2/manifest.json"

DOMAIN_FILES = (
    "parser-annotation.schema.json",
    "safety-annotation.schema.json",
    "label-annotation.schema.json",
    "task-annotation.schema.json",
    "attachment-annotation.schema.json",
    "release-annotation.schema.json",
)
V1_URI = "https://schemas.eval-factory.local/annotation/v1/"
V2_URI = "https://schemas.eval-factory.local/annotation/v2/"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"


def _replace_uri(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _replace_uri(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_replace_uri(child) for child in value]
    if isinstance(value, str):
        return value.replace(V1_URI, V2_URI)
    return value


def _build_common() -> dict[str, Any]:
    common = json.loads((V1_ROOT / "schemas/common.schema.json").read_text(encoding="utf-8"))
    common["$id"] = f"{V2_URI}common.schema.json"
    common["title"] = "Eval Factory Annotation Common Definitions v2"
    definitions = common["$defs"]
    definitions.pop("adjudication")

    governing = definitions["governingVersions"]
    governing["required"].remove("human_review_policy")
    governing["required"].extend(("user_approval_policy", "active_contract_manifest_sha256"))
    governing["properties"].pop("human_review_policy")
    governing["properties"]["spec"]["const"] = "002-eval-dataset-factory/approved-v2-2026-07-20"
    governing["properties"]["user_approval_policy"] = {"const": "eval-factory-user-approval/v2"}
    governing["properties"]["active_contract_manifest_sha256"] = {"$ref": "#/$defs/sha256"}

    record = definitions["annotationRecord"]
    required = record["required"]
    for field in ("annotator_id", "annotator_role", "adjudication"):
        required.remove(field)
    required.extend(("author_id", "author_kind", "checkpoint_preview_refs"))
    properties = record["properties"]
    for field in ("annotator_id", "annotator_role", "adjudication"):
        properties.pop(field)
    properties["schema_version"]["const"] = "eval-factory-annotation/v2"
    properties["guide_version"]["const"] = "eval-factory-annotation-guide/v2"
    properties["author_id"] = {"type": "string", "minLength": 3, "maxLength": 255}
    properties["author_kind"] = {"enum": ["DETERMINISTIC_SERVICE", "SEMANTIC_AGENT", "USER"]}
    properties["checkpoint_preview_refs"] = {
        "type": "array",
        "items": {"$ref": "#/$defs/identifier"},
        "uniqueItems": True,
    }
    properties["status"]["enum"] = [
        "DRAFT",
        "REFERENCE_READY",
        "ABSTAINED",
        "SUPERSEDED",
    ]
    record["allOf"] = [
        rule for rule in record["allOf"] if rule["if"]["properties"]["status"]["const"] != "ADJUDICATED"
    ]
    return cast(dict[str, Any], common)


def _build_release(schema: dict[str, Any]) -> dict[str, Any]:
    payload = schema["$defs"]["payload"]
    required = payload["required"]
    required.remove("required_review_roles")
    required.remove("review_records_current")
    required.extend(
        (
            "automated_quality_passed",
            "user_approval_policy_ref",
            "required_checkpoints",
            "checkpoint_decisions",
        )
    )
    properties = payload["properties"]
    properties.pop("required_review_roles")
    properties.pop("review_records_current")
    properties["automated_quality_passed"] = {"type": "boolean"}
    properties["user_approval_policy_ref"] = {"$ref": "#/$defs/component"}
    checkpoint = {
        "enum": [
            "LABEL_PLAN",
            "TASK_REWRITE_PLAN",
            "ENVIRONMENT_STRATEGY",
            "FINAL_DATASET_REVIEW",
        ]
    }
    properties["required_checkpoints"] = {
        "type": "array",
        "items": checkpoint,
        "uniqueItems": True,
    }
    properties["checkpoint_decisions"] = {
        "type": "array",
        "items": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "checkpoint",
                "request_ref",
                "decision_record_ref",
            ],
            "properties": {
                "checkpoint": copy.deepcopy(checkpoint),
                "request_ref": {"$ref": f"{V2_URI}common.schema.json#/$defs/identifier"},
                "decision_record_ref": {"$ref": f"{V2_URI}common.schema.json#/$defs/identifier"},
            },
        },
        "uniqueItems": True,
    }
    properties["expected_action"]["enum"] = [
        "REQUEST_RELEASE",
        "APPROVE",
        "REJECT",
        "PUBLISH",
        "REVOKE",
        "BLOCK",
    ]
    properties["expected_state"]["enum"] = [
        "CANDIDATE",
        "REJECTED",
        "APPROVED",
        "RELEASED",
        "REVOKED",
    ]
    return schema


def build_artifacts() -> tuple[dict[str, bytes], dict[str, Any]]:
    artifacts: dict[str, bytes] = {
        "schemas/common.schema.json": _json_bytes(_build_common()),
    }
    for file_name in DOMAIN_FILES:
        source = json.loads((V1_ROOT / "schemas" / file_name).read_text(encoding="utf-8"))
        schema = _replace_uri(source)
        schema["title"] = str(schema["title"]).removesuffix(" v1") + " v2"
        if file_name == "release-annotation.schema.json":
            schema = _build_release(schema)
        artifacts[f"schemas/{file_name}"] = _json_bytes(schema)

    file_paths = [
        GUIDE_PATH,
        Path(__file__).resolve(),
        VALIDATOR_PATH,
    ]
    files = [
        {
            "path": str(path.relative_to(REPO_ROOT)),
            "sha256": _sha256(path.read_bytes()),
        }
        for path in file_paths
    ]
    files.extend(
        {
            "path": f"evals/annotation/v2/{relative_path}",
            "sha256": _sha256(content),
        }
        for relative_path, content in sorted(artifacts.items())
    )
    manifest = {
        "schema_version": "eval-factory-annotation-contract-manifest/v2",
        "contract_version": "eval-factory-annotation/v2",
        "guide_version": "eval-factory-annotation-guide/v2",
        "architecture": "v1-domain-base-plus-v2-workflow-overlay",
        "base_annotation_contract": {
            "path": str(BASE_MANIFEST_PATH.relative_to(REPO_ROOT)),
            "sha256": _sha256(BASE_MANIFEST_PATH.read_bytes()),
        },
        "active_cross_stage_contract": {
            "path": str(CROSS_STAGE_MANIFEST.relative_to(REPO_ROOT)),
            "sha256": _sha256(CROSS_STAGE_MANIFEST.read_bytes()),
        },
        "files": sorted(files, key=lambda item: item["path"]),
        "frozen_canary_manifest": {
            "path": str(CANARY_PATH.relative_to(REPO_ROOT)),
            "sha256": _sha256(CANARY_PATH.read_bytes()),
        },
        "historical_only_fields": [
            "adjudication",
            "adjudicator_id",
            "annotator_role",
            "required_review_roles",
            "review_records_current",
        ],
    }
    return artifacts, manifest


def write_artifacts(artifacts: dict[str, bytes], manifest: dict[str, Any]) -> None:
    for relative_path, content in artifacts.items():
        path = OUTPUT_ROOT / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    MANIFEST_PATH.write_bytes(_json_bytes(manifest))


def check_artifacts(artifacts: dict[str, bytes], manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    expected_paths = {OUTPUT_ROOT / path for path in artifacts}
    existing_paths = set(SCHEMA_ROOT.glob("*.schema.json"))
    if expected_paths != existing_paths:
        errors.append("annotation v2 schema file set drift")
    for relative_path, expected in artifacts.items():
        path = OUTPUT_ROOT / relative_path
        if not path.exists() or path.read_bytes() != expected:
            errors.append(f"annotation v2 schema drift: {path}")
    expected_manifest = _json_bytes(manifest)
    if not MANIFEST_PATH.exists() or MANIFEST_PATH.read_bytes() != expected_manifest:
        errors.append(f"annotation v2 manifest drift: {MANIFEST_PATH}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Eval Factory annotation v2 overlay.")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true")
    action.add_argument("--check", action="store_true")
    args = parser.parse_args()

    artifacts, manifest = build_artifacts()
    if args.write:
        write_artifacts(artifacts, manifest)
        print(f"wrote {len(artifacts)} annotation v2 schemas and {MANIFEST_PATH}")
        return
    errors = check_artifacts(artifacts, manifest)
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"validated {len(artifacts)} annotation v2 schemas")


if __name__ == "__main__":
    main()
