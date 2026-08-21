from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest

from eval_factory.contracts.core import ContractAudit, VersionBinding
from eval_factory.readiness.external_evidence_requirement import (
    ExternalEvidenceRequirementError,
    ExternalEvidenceRequirementLimitError,
    ExternalRequirementCompilation,
    ExternalRequirementCompiler,
)


def _audit(
    *,
    created_at: datetime = datetime(2026, 8, 9, tzinfo=UTC),
) -> ContractAudit:
    return ContractAudit(
        created_at=created_at,
        created_by="external-requirement-test",
        governing_versions=(
            VersionBinding(
                component="external-requirement-intake",
                version="external-evidence/r8-10-v1",
            ),
        ),
    )


def _compile(
    path: Path,
    *,
    audit: ContractAudit | None = None,
) -> ExternalRequirementCompilation:
    return ExternalRequirementCompiler().compile(
        source=path,
        run_id="factory-run://r8-10/external-evidence",
        requirement_spec_id=("evaluation-requirement-spec://r8-10/external-evidence"),
        goals=("Interpret the requirement and select grounded trace candidates.",),
        constraints=("Keep original final outputs and private references unavailable.",),
        assumptions=("The approved source is untrusted Markdown data.",),
        open_questions=(),
        requirement_version=1,
        audit=audit or _audit(),
        max_source_bytes=100_000,
    )


def test_requirement_compiler_binds_exact_source_without_body_leakage(
    tmp_path: Path,
) -> None:
    source = tmp_path / "requirement.md"
    body = "# Private requirement\n\nDO_NOT_COPY_THIS_BODY\n"
    source.write_text(body, encoding="utf-8")

    compiled = _compile(source)

    expected = hashlib.sha256(body.encode()).hexdigest()
    assert compiled.source_sha256 == expected
    assert compiled.source_size_bytes == len(body.encode())
    assert compiled.media_type == "text/markdown; charset=utf-8"
    assert compiled.source_ref.object_type == "evaluation-requirement-source"
    assert compiled.source_ref.object_sha256 == expected
    assert compiled.requirement.source_ref == compiled.source_ref
    assert compiled.requirement.audit.input_refs == (compiled.source_ref,)
    assert "DO_NOT_COPY_THIS_BODY" not in compiled.requirement.model_dump_json()


def test_requirement_identity_is_stable_across_audit_time(
    tmp_path: Path,
) -> None:
    source = tmp_path / "requirement.md"
    source.write_text("# Requirement\n", encoding="utf-8")

    first = _compile(source)
    second = _compile(
        source,
        audit=_audit(created_at=datetime(2026, 8, 10, tzinfo=UTC)),
    )

    assert second.source_ref == first.source_ref
    assert second.requirement.object_id == first.requirement.object_id
    assert second.requirement.object_sha256 == first.requirement.object_sha256
    assert second.requirement.canonical_sha256() != first.requirement.canonical_sha256()


def test_requirement_compiler_rejects_symlink_extension_and_limit(
    tmp_path: Path,
) -> None:
    source = tmp_path / "requirement.md"
    source.write_text("# Requirement\n", encoding="utf-8")
    linked = tmp_path / "linked.md"
    linked.symlink_to(source)
    wrong = tmp_path / "requirement.txt"
    wrong.write_text("requirement", encoding="utf-8")

    with pytest.raises(ExternalEvidenceRequirementError, match="symlink"):
        _compile(linked)
    with pytest.raises(ExternalEvidenceRequirementError, match="Markdown"):
        _compile(wrong)
    with pytest.raises(ExternalEvidenceRequirementLimitError):
        ExternalRequirementCompiler().compile(
            source=source,
            run_id="factory-run://r8-10/external-evidence",
            requirement_spec_id=("evaluation-requirement-spec://r8-10/external-evidence"),
            goals=("Goal",),
            constraints=(),
            assumptions=(),
            open_questions=(),
            requirement_version=1,
            audit=_audit(),
            max_source_bytes=2,
        )
