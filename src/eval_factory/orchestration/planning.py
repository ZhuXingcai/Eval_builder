from __future__ import annotations

from typing import Final

from pydantic import ValidationError

from eval_factory.contracts.approval import ApprovalCheckpoint
from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.contracts.orchestration_v2 import (
    CHECKPOINT_STAGES,
    DatasetJobSpecV2,
    ResolvedDatasetJobPlanV2,
    ResolvedStageNodeV2,
    StageNameV2,
    dataset_job_spec_v2_ref,
)

R6_STAGE_PLAN_POLICY_VERSION: Final = "dataset-job-stage-policy/r6-01-v1"
_POLICY_COMPONENT: Final = "dataset-job-stage-policy"
_CHECKPOINT_BY_STAGE: Final = {stage: checkpoint for checkpoint, stage in CHECKPOINT_STAGES.items()}


class JobPlanningPolicyError(RuntimeError):
    pass


class DatasetJobPlanCompiler:
    def compile(
        self,
        *,
        job_spec: DatasetJobSpecV2,
        audit: ContractAudit,
    ) -> ResolvedDatasetJobPlanV2:
        canonical_spec = DatasetJobSpecV2.model_validate(job_spec.model_dump(mode="python"))
        active_chain = _active_chain(canonical_spec.enabled_checkpoints)
        requested = canonical_spec.requested_stages
        active_positions = {stage: index for index, stage in enumerate(active_chain)}
        try:
            highest_position = max(active_positions[stage] for stage in requested)
        except (KeyError, ValueError) as exc:
            raise JobPlanningPolicyError(
                "requested stages are not admitted by the current stage policy"
            ) from exc

        resolved = active_chain[: highest_position + 1]
        requested_set = set(requested)
        auto_added = tuple(stage for stage in resolved if stage not in requested_set)
        nodes = tuple(
            ResolvedStageNodeV2(
                stage=stage,
                depends_on=() if index == 0 else (resolved[index - 1],),
            )
            for index, stage in enumerate(resolved)
        )
        spec_ref = dataset_job_spec_v2_ref(canonical_spec)
        return ResolvedDatasetJobPlanV2.create(
            dataset_job_spec_ref=spec_ref,
            requested_stages=requested,
            resolved_stages=resolved,
            auto_added_stages=auto_added,
            nodes=nodes,
            policy_version=R6_STAGE_PLAN_POLICY_VERSION,
            audit=_plan_audit(audit, spec_ref=spec_ref),
        )

    def validate_current(
        self,
        plan: ResolvedDatasetJobPlanV2,
        *,
        job_spec: DatasetJobSpecV2,
    ) -> None:
        try:
            canonical_spec = DatasetJobSpecV2.model_validate(job_spec.model_dump(mode="python"))
        except ValidationError as exc:
            raise JobPlanningPolicyError("dataset job spec is invalid") from exc

        if plan.policy_version != R6_STAGE_PLAN_POLICY_VERSION:
            raise JobPlanningPolicyError("resolved plan policy is stale")
        expected_spec_ref = dataset_job_spec_v2_ref(canonical_spec)
        if plan.dataset_job_spec_ref != expected_spec_ref:
            raise JobPlanningPolicyError("resolved plan spec binding is stale")

        try:
            canonical_plan = ResolvedDatasetJobPlanV2.model_validate(plan.model_dump(mode="python"))
        except (ValidationError, ValueError) as exc:
            raise JobPlanningPolicyError("resolved plan graph or identity is invalid") from exc

        expected = self.compile(job_spec=canonical_spec, audit=canonical_plan.audit)
        if canonical_plan != expected:
            raise JobPlanningPolicyError("resolved plan does not match the current canonical stage policy")


def _active_chain(
    enabled_checkpoints: frozenset[ApprovalCheckpoint],
) -> tuple[StageNameV2, ...]:
    return tuple(
        stage
        for stage in StageNameV2
        if stage not in _CHECKPOINT_BY_STAGE or _CHECKPOINT_BY_STAGE[stage] in enabled_checkpoints
    )


def _plan_audit(
    audit: ContractAudit,
    *,
    spec_ref: ObjectRef,
) -> ContractAudit:
    versions = tuple(
        sorted(
            (
                *(binding for binding in audit.governing_versions if binding.component != _POLICY_COMPONENT),
                VersionBinding(
                    component=_POLICY_COMPONENT,
                    version=R6_STAGE_PLAN_POLICY_VERSION,
                ),
            ),
            key=lambda binding: (
                binding.component,
                binding.version,
                binding.sha256 or "",
            ),
        )
    )
    return ContractAudit(
        created_at=audit.created_at,
        created_by=audit.created_by,
        governing_versions=versions,
        input_refs=(spec_ref,),
    )
