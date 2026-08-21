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
OUTPUT_ROOT = REPO_ROOT / "specs/002-eval-dataset-factory/contracts/v1"
MANIFEST_PATH = OUTPUT_ROOT / "manifest.json"
POLICY_PATH = REPO_ROOT / "specs/002-eval-dataset-factory/contracts/compatibility-policy-v1.md"

FACTORY_MODULES = (
    "eval_factory.contracts.core",
    "eval_factory.contracts.orchestration",
    "eval_factory.contracts.trace",
    "eval_factory.contracts.safety",
    "eval_factory.contracts.labeling",
    "eval_factory.contracts.task",
    "eval_factory.contracts.attachment",
    "eval_factory.contracts.quality",
    "eval_factory.contracts.release",
)
FACADE_MODULES = ("env_mock_agent.facade.contracts",)
EXCLUDED_MODELS = {"ContractModel", "FacadeModel"}


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _snake_case(value: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", value).lower()


def discover_models() -> list[tuple[str, type[BaseModel]]]:
    discovered: list[tuple[str, type[BaseModel]]] = []
    for namespace, module_names in (
        ("factory", FACTORY_MODULES),
        ("facade", FACADE_MODULES),
    ):
        for module_name in module_names:
            module = importlib.import_module(module_name)
            for name, model in inspect.getmembers(module, inspect.isclass):
                if (
                    name not in EXCLUDED_MODELS
                    and issubclass(model, BaseModel)
                    and model.__module__ == module_name
                ):
                    discovered.append((namespace, model))
    return sorted(discovered, key=lambda item: (item[0], item[1].__name__))


def _schema_version(model: type[BaseModel]) -> str | None:
    field = model.model_fields.get("schema_version")
    if field is None or not isinstance(field.default, str):
        return None
    return field.default


def render_schema(namespace: str, model: type[BaseModel]) -> tuple[str, bytes]:
    relative_path = f"schemas/{namespace}/{_snake_case(model.__name__)}.schema.json"
    schema = model.model_json_schema(mode="validation")
    schema["$id"] = (
        "https://schemas.eval-factory.local/contracts/v1/"
        f"{namespace}/{_snake_case(model.__name__)}.schema.json"
    )
    schema["x-contract-owner"] = namespace
    schema["x-python-type"] = f"{model.__module__}.{model.__qualname__}"
    version = _schema_version(model)
    if version is not None:
        schema["x-schema-version"] = version
    content = json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"
    return relative_path, content


def build_artifacts() -> tuple[dict[str, bytes], dict[str, Any]]:
    artifacts: dict[str, bytes] = {}
    contracts: list[dict[str, Any]] = []
    for namespace, model in discover_models():
        relative_path, content = render_schema(namespace, model)
        if relative_path in artifacts:
            raise ValueError(f"duplicate generated schema path: {relative_path}")
        artifacts[relative_path] = content
        contracts.append(
            {
                "owner": namespace,
                "python_type": f"{model.__module__}.{model.__qualname__}",
                "schema_version": _schema_version(model),
                "schema_path": relative_path,
                "schema_sha256": _sha256(content),
            }
        )

    source_paths: list[Path] = []
    for module_name in (*FACTORY_MODULES, *FACADE_MODULES):
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
        "schema_version": "eval-factory-contract-manifest/v1",
        "contract_set_version": "v1",
        "generator": "scripts/export_eval_factory_contracts.py",
        "generator_version": "1.0.0",
        "pydantic_version": importlib.import_module("pydantic").__version__,
        "compatibility_policy": {
            "path": str(POLICY_PATH.relative_to(REPO_ROOT)),
            "sha256": _sha256(POLICY_PATH.read_bytes()),
        },
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
    parser = argparse.ArgumentParser(description="Export frozen Eval Factory v1 contract schemas.")
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
    print(f"validated {len(artifacts)} frozen contract schemas")


if __name__ == "__main__":
    main()
