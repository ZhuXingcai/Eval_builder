from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from eval_factory.attachment_planning.execution_models import (
    ArtifactConsistencyDefinition,
    ArtifactExecutionPlanningOutcome,
    ArtifactExecutionPlanningResult,
    ArtifactExecutionPreparation,
)
from eval_factory.contracts.core import ContractAudit, VersionBinding


def _audit() -> ContractAudit:
    return ContractAudit(
        created_at=datetime(2026, 7, 28, tzinfo=UTC),
        created_by="execution-model-test",
        governing_versions=(
            VersionBinding(
                component="artifact-execution",
                version="r5-06",
            ),
        ),
    )


def test_consistency_definition_accepts_only_sorted_unique_control_facts() -> None:
    definition = ArtifactConsistencyDefinition(
        artifact_id="artifact://input",
        dependency_artifact_ids=(
            "artifact://base",
            "artifact://schema",
        ),
        locked_fact_ids=(
            "world-fact://organization",
            "world-fact://reporting-period",
        ),
    )

    assert definition.artifact_id == "artifact://input"
    assert definition.dependency_artifact_ids == (
        "artifact://base",
        "artifact://schema",
    )

    with pytest.raises(ValidationError, match="sorted"):
        ArtifactConsistencyDefinition(
            artifact_id="artifact://input",
            dependency_artifact_ids=(
                "artifact://schema",
                "artifact://base",
            ),
            locked_fact_ids=(),
        )

    with pytest.raises(ValidationError, match="unique"):
        ArtifactConsistencyDefinition(
            artifact_id="artifact://input",
            dependency_artifact_ids=(),
            locked_fact_ids=(
                "world-fact://organization",
                "world-fact://organization",
            ),
        )

    with pytest.raises(ValidationError, match="itself"):
        ArtifactConsistencyDefinition(
            artifact_id="artifact://input",
            dependency_artifact_ids=("artifact://input",),
            locked_fact_ids=(),
        )


@pytest.mark.parametrize(
    "field",
    [
        "relative_path",
        "selected_provider",
        "selected_runtime",
        "evidence_refs",
        "world_fact_values",
    ],
)
def test_consistency_definition_rejects_execution_authority_fields(
    field: str,
) -> None:
    values: dict[str, object] = {
        "artifact_id": "artifact://input",
        "dependency_artifact_ids": (),
        "locked_fact_ids": (),
        field: (),
    }

    with pytest.raises(ValidationError, match="extra"):
        ArtifactConsistencyDefinition.model_validate(values)


def test_execution_planning_models_reject_incomplete_outcome_shapes() -> None:
    definition = ArtifactConsistencyDefinition(
        artifact_id="artifact://input",
        dependency_artifact_ids=(),
        locked_fact_ids=(),
    )

    with pytest.raises(ValidationError, match="requires request"):
        ArtifactExecutionPreparation(
            outcome=ArtifactExecutionPlanningOutcome.PLANNED,
            snapshot_request=None,
            definitions=(definition,),
            audit=_audit(),
        )
    with pytest.raises(ValidationError, match="cannot contain work"):
        ArtifactExecutionPreparation(
            outcome=ArtifactExecutionPlanningOutcome.NOT_REQUIRED,
            snapshot_request=None,
            definitions=(definition,),
            audit=_audit(),
        )
    with pytest.raises(ValidationError, match="requires snapshot"):
        ArtifactExecutionPlanningResult(
            outcome=ArtifactExecutionPlanningOutcome.PLANNED,
            world_ledger_snapshot=None,
            execution_plan=None,
            audit=_audit(),
        )
