from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import model_validator

from env_mock_agent.facade import WorldLedgerSnapshotRequestV2, WorldLedgerSnapshotV2
from eval_factory.contracts.attachment_v2 import (
    ArtifactExecutionBatchV2,
    ArtifactExecutionPlanV2,
)
from eval_factory.contracts.core import ContractAudit, ContractModel, Identifier

ARTIFACT_EXECUTION_POLICY_VERSION: Literal["artifact-execution/r5-06-v1"] = "artifact-execution/r5-06-v1"


class ArtifactExecutionPolicyError(RuntimeError):
    pass


class ArtifactExecutionPlanningOutcome(StrEnum):
    PLANNED = "PLANNED"
    NOT_REQUIRED = "NOT_REQUIRED"
    NO_ROUTED_ARTIFACTS = "NO_ROUTED_ARTIFACTS"


class ArtifactConsistencyDefinition(ContractModel):
    schema_version: Literal["eval-factory/artifact-consistency-definition/r5-06"] = (
        "eval-factory/artifact-consistency-definition/r5-06"
    )
    artifact_id: Identifier
    dependency_artifact_ids: tuple[Identifier, ...]
    locked_fact_ids: tuple[Identifier, ...]

    @model_validator(mode="after")
    def validate_definition(self) -> ArtifactConsistencyDefinition:
        _require_sorted_unique(
            "dependency artifact IDs",
            self.dependency_artifact_ids,
        )
        _require_sorted_unique("locked fact IDs", self.locked_fact_ids)
        if self.artifact_id in self.dependency_artifact_ids:
            raise ValueError("artifact cannot depend on itself")
        return self


class ArtifactExecutionPreparation(ContractModel):
    schema_version: Literal["eval-factory/artifact-execution-preparation/r5-06"] = (
        "eval-factory/artifact-execution-preparation/r5-06"
    )
    outcome: ArtifactExecutionPlanningOutcome
    snapshot_request: WorldLedgerSnapshotRequestV2 | None
    definitions: tuple[ArtifactConsistencyDefinition, ...]
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_preparation(self) -> ArtifactExecutionPreparation:
        if self.outcome is ArtifactExecutionPlanningOutcome.PLANNED:
            if self.snapshot_request is None or not self.definitions:
                raise ValueError("planned execution requires request and definitions")
        elif self.snapshot_request is not None or self.definitions:
            raise ValueError("non-planned execution cannot contain work")
        return self


class ArtifactExecutionPlanningResult(ContractModel):
    schema_version: Literal["eval-factory/artifact-execution-planning-result/r5-06"] = (
        "eval-factory/artifact-execution-planning-result/r5-06"
    )
    outcome: ArtifactExecutionPlanningOutcome
    world_ledger_snapshot: WorldLedgerSnapshotV2 | None
    execution_plan: ArtifactExecutionPlanV2 | None
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_result(self) -> ArtifactExecutionPlanningResult:
        if self.outcome is ArtifactExecutionPlanningOutcome.PLANNED:
            if self.world_ledger_snapshot is None or self.execution_plan is None:
                raise ValueError("planned result requires snapshot and execution plan")
        elif self.world_ledger_snapshot is not None or self.execution_plan is not None:
            raise ValueError("non-planned result cannot contain execution objects")
        return self


class ArtifactExecutionRunResult(ContractModel):
    schema_version: Literal["eval-factory/artifact-execution-run-result/r5-06"] = (
        "eval-factory/artifact-execution-run-result/r5-06"
    )
    execution_plan: ArtifactExecutionPlanV2
    execution_batch: ArtifactExecutionBatchV2


def _require_sorted_unique(
    label: str,
    values: tuple[str, ...],
) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must be unique")
    if values != tuple(sorted(values)):
        raise ValueError(f"{label} must be sorted")
