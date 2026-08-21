from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "specs/002-eval-dataset-factory/contracts/v2"
MANIFEST_PATH = OUTPUT_ROOT / "manifest.json"
POLICY_PATH = REPO_ROOT / "specs/002-eval-dataset-factory/contracts/compatibility-policy-v2.md"
BASE_MANIFEST_PATH = REPO_ROOT / "specs/002-eval-dataset-factory/contracts/v1/manifest.json"

OVERLAY_MODULES = (
    ("factory", "eval_factory.contracts.core_v2"),
    ("factory", "eval_factory.contracts.agent_system_v2"),
    ("factory", "eval_factory.contracts.dataset_runtime_v2"),
    ("factory", "eval_factory.contracts.ai_gateway_v2"),
    ("factory", "eval_factory.contracts.approval"),
    ("factory", "eval_factory.contracts.approval_v2"),
    ("factory", "eval_factory.contracts.approval_decision_v2"),
    ("factory", "eval_factory.contracts.approval_application_v2"),
    ("factory", "eval_factory.contracts.attachment_v2"),
    ("factory", "eval_factory.contracts.batch_quality_v2"),
    ("factory", "eval_factory.contracts.canary_execution_v2"),
    ("factory", "eval_factory.contracts.canary_regression_v2"),
    ("factory", "eval_factory.contracts.real_trace_stability_v2"),
    ("factory", "eval_factory.contracts.scheduler_load_v2"),
    ("factory", "eval_factory.contracts.checkpoint_interaction_v2"),
    ("factory", "eval_factory.contracts.cli_v2"),
    ("factory", "eval_factory.contracts.concurrency_experiment_v2"),
    ("factory", "eval_factory.contracts.cross_item_safety_v2"),
    ("factory", "eval_factory.contracts.duplicate_v2"),
    ("factory", "eval_factory.contracts.external_evidence_v2"),
    ("factory", "eval_factory.contracts.external_stability_v2"),
    ("factory", "eval_factory.contracts.label_quality_v2"),
    ("factory", "eval_factory.contracts.labeling_v2"),
    ("factory", "eval_factory.contracts.model_control_v2"),
    ("factory", "eval_factory.contracts.observability_v2"),
    ("factory", "eval_factory.contracts.orchestration_v2"),
    ("factory", "eval_factory.contracts.production_attestation_v2"),
    ("factory", "eval_factory.contracts.production_release_v2"),
    ("factory", "eval_factory.contracts.production_readiness_review_v2"),
    ("factory", "eval_factory.contracts.quality_v2"),
    ("factory", "eval_factory.contracts.release_projection_v2"),
    ("factory", "eval_factory.contracts.release_publication_v2"),
    ("factory", "eval_factory.contracts.release_v2"),
    ("factory", "eval_factory.contracts.resource_v2"),
    ("factory", "eval_factory.contracts.review_v2"),
    ("factory", "eval_factory.contracts.statistics_v2"),
    ("factory", "eval_factory.contracts.task_v2"),
    ("factory", "eval_factory.contracts.validation_v2"),
    ("facade", "env_mock_agent.facade.cross_item_safety_v2"),
    ("facade", "env_mock_agent.facade.duplicate_v2"),
    ("facade", "env_mock_agent.facade.execution_v2"),
    ("facade", "env_mock_agent.facade.model_control_v2"),
    ("facade", "env_mock_agent.facade.release_export_v2"),
    ("facade", "env_mock_agent.facade.retrieval_v2"),
    ("facade", "env_mock_agent.facade.resource_v2"),
    ("facade", "env_mock_agent.facade.routing_v2"),
    ("facade", "env_mock_agent.facade.semantic_review_v2"),
    ("facade", "env_mock_agent.facade.telemetry_v2"),
    ("facade", "env_mock_agent.facade.validation_v2"),
)
EXCLUDED_MODELS = {
    "_FactoryObjectV2",
    "_GatewayObjectV2",
    "_DatasetRuntimeObjectV2",
    "_ExternalEvidenceObjectV2",
    "_ExternalStabilityObjectV2",
    "ConcurrencyExperimentFaultCountV2",
    "ContractModel",
    "ContractModelV2",
}


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _snake_case(value: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", value).lower()


def discover_models() -> list[tuple[str, type[BaseModel]]]:
    discovered: list[tuple[str, type[BaseModel]]] = []
    for owner, module_name in OVERLAY_MODULES:
        module = importlib.import_module(module_name)
        for name, model in inspect.getmembers(module, inspect.isclass):
            if (
                name not in EXCLUDED_MODELS
                and issubclass(model, BaseModel)
                and model.__module__ == module_name
            ):
                discovered.append((owner, model))
    return sorted(
        discovered,
        key=lambda item: (item[1].__name__, item[0]),
    )


def _schema_version(model: type[BaseModel]) -> str | None:
    field = model.model_fields.get("schema_version")
    if field is None or not isinstance(field.default, str):
        return None
    return field.default


def render_schema(
    owner: str,
    model: type[BaseModel],
) -> tuple[str, bytes]:
    relative_path = f"schemas/overlay/{_snake_case(model.__name__)}.schema.json"
    schema = model.model_json_schema(mode="validation")
    schema["$id"] = (
        f"https://schemas.eval-factory.local/contracts/v2/overlay/{_snake_case(model.__name__)}.schema.json"
    )
    schema["x-contract-owner"] = owner
    schema["x-contract-layer"] = "v2-overlay"
    schema["x-python-type"] = f"{model.__module__}.{model.__qualname__}"
    version = _schema_version(model)
    if version is not None:
        schema["x-schema-version"] = version
    content = json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"
    return relative_path, content


def build_artifacts() -> tuple[dict[str, bytes], dict[str, Any]]:
    artifacts: dict[str, bytes] = {}
    contracts: list[dict[str, Any]] = []
    for owner, model in discover_models():
        relative_path, content = render_schema(owner, model)
        if relative_path in artifacts:
            raise ValueError(f"duplicate generated schema path: {relative_path}")
        artifacts[relative_path] = content
        contracts.append(
            {
                "owner": owner,
                "layer": "v2-overlay",
                "python_type": f"{model.__module__}.{model.__qualname__}",
                "schema_version": _schema_version(model),
                "schema_path": relative_path,
                "schema_sha256": _sha256(content),
            }
        )

    source_paths: list[Path] = []
    for _, module_name in OVERLAY_MODULES:
        module_path = importlib.import_module(module_name).__file__
        if module_path is None:
            raise ValueError(f"contract module has no source path: {module_name}")
        source_paths.append(Path(module_path).resolve())
    source_files = [
        {
            "path": str(path.relative_to(REPO_ROOT)),
            "sha256": _sha256(path.read_bytes()),
        }
        for path in sorted(set(source_paths))
    ]
    manifest = {
        "schema_version": "eval-factory-contract-manifest/v2",
        "contract_set_version": "v2",
        "architecture": "v1-base-plus-v2-overlay",
        "generator": "scripts/export_eval_factory_contracts_v2.py",
        "generator_version": "2.0.0",
        "pydantic_version": importlib.import_module("pydantic").__version__,
        "base_contract_set": {
            "path": str(BASE_MANIFEST_PATH.relative_to(REPO_ROOT)),
            "sha256": _sha256(BASE_MANIFEST_PATH.read_bytes()),
        },
        "compatibility_policy": {
            "path": str(POLICY_PATH.relative_to(REPO_ROOT)),
            "sha256": _sha256(POLICY_PATH.read_bytes()),
        },
        "supersedes_for_v2_jobs": [
            "eval-factory/artifact-build-spec/v1",
            "eval-factory/artifact-evidence-matrix/v1",
            "eval-factory/artifact-evidence-row/v1",
            "eval-factory/dataset-job-spec/v1",
            "eval-factory/evaluation-item/v1",
            "eval-factory/evaluator-spec/v1",
            "eval-factory/environment-spec/v1",
            "eval-factory/producer-task-view/v1",
            "eval-factory/provenance-manifest/v1",
            "eval-factory/quality-report/v1",
            "eval-factory/reference-policy/v1",
            "eval-factory/release-decision/v1",
            "eval-factory/rubric-criterion/v1",
            "eval-factory/rubric-set/v1",
            "eval-factory/source-evidence/v1",
            "eval-factory/task-draft/v1",
            "eval-factory/task-episode/v1",
            "eval-factory/tool-policy/v1",
            "eval-factory/tool-rule/v1",
        ],
        "historical_only_for_v2_jobs": [
            "eval-factory/human-review-record/v1",
            "eval-factory/review-record-invalidation/v1",
            "eval-factory/review-quorum/v1",
        ],
        "source_files": source_files,
        "contracts": contracts,
    }
    return artifacts, manifest


def _manifest_bytes(manifest: dict[str, Any]) -> bytes:
    return json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"


def write_artifacts(artifacts: dict[str, bytes], manifest: dict[str, Any]) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for relative_path, content in artifacts.items():
        path = OUTPUT_ROOT / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    MANIFEST_PATH.write_bytes(_manifest_bytes(manifest))


def check_artifacts(artifacts: dict[str, bytes], manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    expected_paths = {OUTPUT_ROOT / path for path in artifacts}
    existing_paths = set((OUTPUT_ROOT / "schemas").glob("*/*.schema.json"))
    missing = sorted(expected_paths - existing_paths)
    extra = sorted(existing_paths - expected_paths)
    if missing:
        errors.append(f"missing generated schemas: {[str(path) for path in missing]}")
    if extra:
        errors.append(f"unexpected generated schemas: {[str(path) for path in extra]}")
    for relative_path, expected in artifacts.items():
        path = OUTPUT_ROOT / relative_path
        if path.exists() and path.read_bytes() != expected:
            errors.append(f"schema drift: {path}")
    expected_manifest = _manifest_bytes(manifest)
    if not MANIFEST_PATH.exists():
        errors.append(f"missing contract manifest: {MANIFEST_PATH}")
    elif MANIFEST_PATH.read_bytes() != expected_manifest:
        errors.append(f"contract manifest drift: {MANIFEST_PATH}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Export frozen Eval Factory v2 overlay schemas.")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true")
    action.add_argument("--check", action="store_true")
    args = parser.parse_args()

    artifacts, manifest = build_artifacts()
    if args.write:
        write_artifacts(artifacts, manifest)
        print(f"wrote {len(artifacts)} schemas and {MANIFEST_PATH}")
        return
    errors = check_artifacts(artifacts, manifest)
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"validated {len(artifacts)} frozen v2 overlay schemas")


if __name__ == "__main__":
    main()
