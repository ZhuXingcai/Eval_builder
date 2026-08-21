from __future__ import annotations

import pytest
from pydantic import ValidationError

from env_mock_agent.schemas import DependencySpec, Problem, ProblemLedger, Severity


def test_dependency_rejects_path_traversal() -> None:
    with pytest.raises(ValidationError):
        DependencySpec(dependency_id="D-001", path="../outside.txt")


def test_problem_ledger_detects_release_blocker() -> None:
    ledger = ProblemLedger(
        problems=[
            Problem(
                problem_id="P0-001",
                severity=Severity.P0,
                category="answer_leakage",
                description="final answer present",
                introduced_round=1,
            )
        ]
    )
    assert ledger.has_release_blocker is True
