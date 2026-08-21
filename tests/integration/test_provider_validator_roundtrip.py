from __future__ import annotations

import zipfile
from pathlib import Path
from xml.etree import ElementTree

import pytest

from env_mock_agent.providers import ProviderRegistry, ProviderRequest
from env_mock_agent.schemas import ArtifactPlan
from env_mock_agent.validators import ValidationRequest, ValidatorRegistry


@pytest.mark.parametrize(
    ("asset_type", "relative_path", "contract", "validator"),
    [
        (
            "txt",
            "notes/input.txt",
            {"content": "Neutral input facts for the task.", "min_characters": 10},
            "text",
        ),
        (
            "json",
            "data/input.json",
            {"data": {"records": [1, 2]}, "required_keys": ["records"]},
            "structured",
        ),
        (
            "docx",
            "documents/input.docx",
            {
                "title": "Input Memorandum",
                "paragraphs": ["Neutral source fact.", "Additional source context."],
                "tables": [{"headers": ["Field", "Value"], "rows": [["Status", "Open"]]}],
                "min_paragraphs": 2,
                "min_tables": 1,
            },
            "docx",
        ),
        (
            "xlsx",
            "data/input.xlsx",
            {
                "sheets": [
                    {
                        "name": "Survey",
                        "rows": [["Question", "Response"], ["Q1", "A1"]],
                    }
                ],
                "required_sheets": ["Survey"],
            },
            "xlsx",
        ),
        (
            "pdf",
            "documents/input.pdf",
            {
                "title": "Reference Policy",
                "paragraphs": ["This file contains neutral input material."],
                "min_pages": 1,
                "min_characters": 10,
            },
            "pdf",
        ),
        (
            "project",
            "project",
            {
                "files": {
                    "README.md": "# Input Project\n",
                    "src/main.py": "def value() -> int:\n    return 1\n",
                },
                "required_files": ["README.md", "src/main.py"],
            },
            "code_project",
        ),
    ],
)
def test_provider_output_round_trips_through_validator(
    tmp_path: Path,
    asset_type: str,
    relative_path: str,
    contract: dict[str, object],
    validator: str,
) -> None:
    plan = ArtifactPlan(
        artifact_id=f"artifact-{asset_type}",
        dependency_id=f"dependency-{asset_type}",
        relative_path=relative_path,
        asset_type=asset_type,
        content_contract=contract,
    )
    providers = ProviderRegistry.default()
    provider = providers.for_asset_type(asset_type)
    assert provider is not None
    result = provider.generate(ProviderRequest(plan=plan, staging_root=tmp_path))

    request = ValidationRequest(plan=plan, result=result)
    findings = ValidatorRegistry.default().validate(
        request,
        ["common", validator, "metadata", "secrets", "leakage"],
    )
    assert findings == []
    assert result.sha256
    assert result.bytes_count > 0


@pytest.mark.parametrize("asset_type", ["docx", "xlsx", "pdf"])
def test_binary_provider_is_reproducible_for_same_seed(
    tmp_path: Path,
    asset_type: str,
) -> None:
    contracts: dict[str, dict[str, object]] = {
        "docx": {"title": "Input", "paragraphs": ["Neutral fact."]},
        "xlsx": {"sheets": [{"name": "Data", "rows": [["Field"], ["Value"]]}]},
        "pdf": {"title": "Input", "paragraphs": ["Neutral fact."]},
    }
    plan = ArtifactPlan(
        artifact_id=f"artifact-{asset_type}",
        dependency_id=f"dependency-{asset_type}",
        relative_path=f"input.{asset_type}",
        asset_type=asset_type,
        content_contract=contracts[asset_type],
        seed=17,
    )
    provider = ProviderRegistry.default().for_asset_type(asset_type)
    assert provider is not None
    first = provider.generate(ProviderRequest(plan=plan, staging_root=tmp_path / "first"))
    second = provider.generate(ProviderRequest(plan=plan, staging_root=tmp_path / "second"))
    assert first.sha256 == second.sha256
    if asset_type in {"docx", "xlsx"}:
        assert first.path is not None
        with zipfile.ZipFile(first.path) as package:
            core = ElementTree.fromstring(package.read("docProps/core.xml"))
        namespace = "http://purl.org/dc/terms/"
        created = core.find(f"{{{namespace}}}created")
        modified = core.find(f"{{{namespace}}}modified")
        assert created is not None and created.text == "2020-01-18T00:00:00Z"
        assert modified is not None and modified.text == "2020-01-18T00:00:00Z"
