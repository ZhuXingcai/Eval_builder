from __future__ import annotations

from pathlib import Path

from env_mock_agent.schemas import (
    ArtifactPlan,
    ArtifactResult,
    ArtifactStatus,
    Criticality,
    DependencySpec,
    ForbiddenOutputSpec,
    ReconstructionStrategy,
)
from env_mock_agent.validators import ValidationRequest, ValidatorRegistry


def make_request(tmp_path: Path, content: str) -> ValidationRequest:
    path = tmp_path / "input.txt"
    path.write_text(content, encoding="utf-8")
    plan = ArtifactPlan(
        artifact_id="A-001",
        dependency_id="D-001",
        relative_path="input.txt",
        asset_type="txt",
    )
    result = ArtifactResult(
        artifact_id=plan.artifact_id,
        status=ArtifactStatus.BUILT,
        path=str(path),
    )
    return ValidationRequest(plan=plan, result=result)


def test_secret_validator_emits_release_blocker(tmp_path: Path) -> None:
    request = make_request(tmp_path, "api_key = abcdefghijklmnopqrstuvwxyz\n")
    findings = ValidatorRegistry.default().validate(request, ["secrets"])
    assert len(findings) == 1
    assert findings[0].severity.value == "P0"
    assert findings[0].category.value == "security"


def test_leakage_validator_emits_release_blocker(tmp_path: Path) -> None:
    request = make_request(tmp_path, "Final recommendation: approve the transaction.\n")
    request.forbidden_outputs = [
        ForbiddenOutputSpec(
            forbidden_id="F-001",
            description="final recommendation",
            semantic_patterns=["Final recommendation"],
        )
    ]
    findings = ValidatorRegistry.default().validate(request, ["leakage"])
    assert len(findings) == 1
    assert findings[0].severity.value == "P0"
    assert findings[0].category.value == "answer_leakage"


def test_critical_grounded_dependency_requires_source_evidence(tmp_path: Path) -> None:
    request = make_request(tmp_path, "Neutral facts.\n")
    request.dependency = DependencySpec(
        dependency_id="D-001",
        path="input.txt",
        asset_type="txt",
        criticality=Criticality.CRITICAL,
        reconstruction_strategy=ReconstructionStrategy.SEARCH_DOWNLOAD,
    )
    findings = ValidatorRegistry.default().validate(request, ["traceability"])
    assert len(findings) == 1
    assert findings[0].category.value == "traceability"
    assert findings[0].severity.value == "P1"
