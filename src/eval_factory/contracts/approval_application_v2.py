from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from eval_factory.contracts.approval import (
    ApprovalCheckpoint,
    InvalidationScope,
    TypedAdjustment,
)
from eval_factory.contracts.core import (
    ContractAudit,
    Identifier,
    ObjectRef,
    Sha256,
    VersionBinding,
)
from eval_factory.contracts.core_v2 import ContractModelV2, canonical_value_v2
from eval_factory.contracts.orchestration_v2 import StageNameV2

USER_PLAN_APPLICATION_POLICY_VERSION: Literal["user-plan-application/r7-06-v1"] = (
    "user-plan-application/r7-06-v1"
)


class RevalidationWorkScopeV2(StrEnum):
    JOB = "JOB"
    ITEM = "ITEM"


class RevalidationWorkOutcomeV2(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    BLOCKED_POLICY = "BLOCKED_POLICY"
    BLOCKED_CAPABILITY = "BLOCKED_CAPABILITY"
    TERMINAL_FAILURE = "TERMINAL_FAILURE"


class DirectedRevalidationReportOutcomeV2(StrEnum):
    COMPLETE = "COMPLETE"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"


class UserPlanApplicationPolicyV2(ContractModelV2):
    schema_version: Literal["eval-factory/user-plan-application-policy/v2"] = (
        "eval-factory/user-plan-application-policy/v2"
    )
    policy_id: Identifier
    max_adjustments: int = Field(ge=1, le=10_000)
    max_invalidated_refs: int = Field(ge=1, le=100_000)
    max_preserved_refs: int = Field(ge=1, le=100_000)
    max_affected_items: int = Field(ge=1, le=100_000)
    max_work_items: int = Field(ge=1, le=100_000)
    max_outputs_per_work: int = Field(ge=1, le=10_000)
    max_revalidation_attempts: int = Field(ge=1, le=100)
    max_failure_code_characters: int = Field(ge=1, le=256)
    policy_version: Literal["user-plan-application/r7-06-v1"] = USER_PLAN_APPLICATION_POLICY_VERSION
    policy_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_policy(self) -> Self:
        _validate_audit(self.audit, (), "user plan application policy")
        _validate_identity(
            object_id=self.policy_id,
            object_sha256=self.policy_sha256,
            expected_prefix="user-plan-application-policy",
            observed=user_plan_application_policy_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        max_adjustments: int,
        max_invalidated_refs: int,
        max_preserved_refs: int,
        max_affected_items: int,
        max_work_items: int,
        max_outputs_per_work: int,
        max_revalidation_attempts: int,
        max_failure_code_characters: int,
        audit: ContractAudit,
    ) -> UserPlanApplicationPolicyV2:
        value = cls(
            policy_id="user-plan-application-policy://pending",
            max_adjustments=max_adjustments,
            max_invalidated_refs=max_invalidated_refs,
            max_preserved_refs=max_preserved_refs,
            max_affected_items=max_affected_items,
            max_work_items=max_work_items,
            max_outputs_per_work=max_outputs_per_work,
            max_revalidation_attempts=max_revalidation_attempts,
            max_failure_code_characters=max_failure_code_characters,
            policy_sha256="0" * 64,
            audit=_safe_audit(audit, ()),
        )
        return _finalize(
            value,
            id_field="policy_id",
            hash_field="policy_sha256",
            prefix="user-plan-application-policy",
            digest=user_plan_application_policy_v2_carried_sha256(value),
        )


class LabelPlanAdjustmentResultV2(ContractModelV2):
    schema_version: Literal["eval-factory/label-plan-adjustment-result/v2"] = (
        "eval-factory/label-plan-adjustment-result/v2"
    )
    result_id: Identifier
    source_request_ref: ObjectRef
    source_label_spec_ref: ObjectRef
    source_label_plan_ref: ObjectRef
    adjustments: tuple[TypedAdjustment, ...] = Field(min_length=1)
    replacement_label_spec_ref: ObjectRef
    replacement_label_plan_ref: ObjectRef
    invalidation_scope: InvalidationScope
    hard_gate_revalidation_required: Literal[True] = True
    policy_version: Literal["user-plan-application/r7-06-v1"] = USER_PLAN_APPLICATION_POLICY_VERSION
    result_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        _require_ref(self.source_request_ref, "user-approval-request", "v2", "source_request_ref")
        _require_ref(self.source_label_spec_ref, "label-spec", "v2", "source_label_spec_ref")
        _require_ref(self.source_label_plan_ref, "label-plan", "v2", "source_label_plan_ref")
        _require_ref(
            self.replacement_label_spec_ref,
            "label-spec",
            "v2",
            "replacement_label_spec_ref",
        )
        _require_ref(
            self.replacement_label_plan_ref,
            "label-plan",
            "v2",
            "replacement_label_plan_ref",
        )
        if self.source_label_spec_ref == self.replacement_label_spec_ref:
            raise ValueError("label adjustment requires a replacement LabelSpec")
        if self.source_label_plan_ref == self.replacement_label_plan_ref:
            raise ValueError("label adjustment requires a replacement LabelPlan")
        _validate_adjustments(self.adjustments)
        _validate_invalidation(self.invalidation_scope)
        refs = _label_adjustment_refs(self)
        _validate_audit(self.audit, refs, "label plan adjustment result")
        _validate_identity(
            object_id=self.result_id,
            object_sha256=self.result_sha256,
            expected_prefix="label-plan-adjustment-result",
            observed=label_plan_adjustment_result_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        source_request_ref: ObjectRef,
        source_label_spec_ref: ObjectRef,
        source_label_plan_ref: ObjectRef,
        adjustments: tuple[TypedAdjustment, ...],
        replacement_label_spec_ref: ObjectRef,
        replacement_label_plan_ref: ObjectRef,
        invalidation_scope: InvalidationScope,
        audit: ContractAudit,
    ) -> LabelPlanAdjustmentResultV2:
        normalized_adjustments = _sorted_adjustments(adjustments)
        normalized_invalidation = _normalized_invalidation(invalidation_scope)
        refs = (
            source_request_ref,
            source_label_spec_ref,
            source_label_plan_ref,
            replacement_label_spec_ref,
            replacement_label_plan_ref,
            *normalized_invalidation.object_refs,
        )
        value = cls(
            result_id="label-plan-adjustment-result://pending",
            source_request_ref=source_request_ref,
            source_label_spec_ref=source_label_spec_ref,
            source_label_plan_ref=source_label_plan_ref,
            adjustments=normalized_adjustments,
            replacement_label_spec_ref=replacement_label_spec_ref,
            replacement_label_plan_ref=replacement_label_plan_ref,
            invalidation_scope=normalized_invalidation,
            result_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        return _finalize(
            value,
            id_field="result_id",
            hash_field="result_sha256",
            prefix="label-plan-adjustment-result",
            digest=label_plan_adjustment_result_v2_carried_sha256(value),
        )


class EnvironmentStrategyAdjustmentResultV2(ContractModelV2):
    schema_version: Literal["eval-factory/environment-strategy-adjustment-result/v2"] = (
        "eval-factory/environment-strategy-adjustment-result/v2"
    )
    result_id: Identifier
    source_request_ref: ObjectRef
    source_strategy_ref: ObjectRef
    adjustments: tuple[TypedAdjustment, ...] = Field(min_length=1)
    replacement_strategy_ref: ObjectRef
    invalidation_scope: InvalidationScope
    hard_gate_revalidation_required: Literal[True] = True
    policy_version: Literal["user-plan-application/r7-06-v1"] = USER_PLAN_APPLICATION_POLICY_VERSION
    result_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        _require_ref(self.source_request_ref, "user-approval-request", "v2", "source_request_ref")
        _require_ref(
            self.source_strategy_ref,
            "environment-strategy",
            "v2",
            "source_strategy_ref",
        )
        _require_ref(
            self.replacement_strategy_ref,
            "environment-strategy",
            "v2",
            "replacement_strategy_ref",
        )
        if self.source_strategy_ref == self.replacement_strategy_ref:
            raise ValueError("environment adjustment requires a replacement strategy")
        _validate_adjustments(self.adjustments)
        _validate_invalidation(self.invalidation_scope)
        refs = _environment_adjustment_refs(self)
        _validate_audit(self.audit, refs, "environment strategy adjustment result")
        _validate_identity(
            object_id=self.result_id,
            object_sha256=self.result_sha256,
            expected_prefix="environment-strategy-adjustment-result",
            observed=environment_strategy_adjustment_result_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        source_request_ref: ObjectRef,
        source_strategy_ref: ObjectRef,
        adjustments: tuple[TypedAdjustment, ...],
        replacement_strategy_ref: ObjectRef,
        invalidation_scope: InvalidationScope,
        audit: ContractAudit,
    ) -> EnvironmentStrategyAdjustmentResultV2:
        normalized_adjustments = _sorted_adjustments(adjustments)
        normalized_invalidation = _normalized_invalidation(invalidation_scope)
        refs = (
            source_request_ref,
            source_strategy_ref,
            replacement_strategy_ref,
            *normalized_invalidation.object_refs,
        )
        value = cls(
            result_id="environment-strategy-adjustment-result://pending",
            source_request_ref=source_request_ref,
            source_strategy_ref=source_strategy_ref,
            adjustments=normalized_adjustments,
            replacement_strategy_ref=replacement_strategy_ref,
            invalidation_scope=normalized_invalidation,
            result_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        return _finalize(
            value,
            id_field="result_id",
            hash_field="result_sha256",
            prefix="environment-strategy-adjustment-result",
            digest=environment_strategy_adjustment_result_v2_carried_sha256(value),
        )


_FINAL_RESTART_STAGES = frozenset(
    {
        StageNameV2.TASK_AUTHORING,
        StageNameV2.ATTACHMENT,
        StageNameV2.ITEM_QUALITY,
    }
)


class FinalDatasetAdjustmentResultV2(ContractModelV2):
    schema_version: Literal["eval-factory/final-dataset-adjustment-result/v2"] = (
        "eval-factory/final-dataset-adjustment-result/v2"
    )
    result_id: Identifier
    source_request_ref: ObjectRef
    source_preview_ref: ObjectRef
    adjustments: tuple[TypedAdjustment, ...] = Field(min_length=1)
    replacement_preview_ref: ObjectRef
    target_item_ids: tuple[Identifier, ...] = Field(min_length=1)
    restart_stages: tuple[StageNameV2, ...] = Field(min_length=1)
    invalidation_scope: InvalidationScope
    hard_gate_revalidation_required: Literal[True] = True
    policy_version: Literal["user-plan-application/r7-06-v1"] = USER_PLAN_APPLICATION_POLICY_VERSION
    result_sha256: Sha256
    audit: ContractAudit

    @field_validator("restart_stages", mode="before")
    @classmethod
    def parse_restart_stages(cls, value: object) -> tuple[StageNameV2, ...]:
        return _parse_enum_tuple(value, StageNameV2, "restart_stages")

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        _require_ref(self.source_request_ref, "user-approval-request", "v2", "source_request_ref")
        _require_ref(
            self.source_preview_ref,
            "final-dataset-review-preview",
            "v2",
            "source_preview_ref",
        )
        _require_ref(
            self.replacement_preview_ref,
            "final-dataset-review-preview",
            "v2",
            "replacement_preview_ref",
        )
        if self.source_preview_ref == self.replacement_preview_ref:
            raise ValueError("final adjustment requires a replacement preview")
        _validate_adjustments(self.adjustments)
        _require_sorted_unique_values("final adjustment target Item IDs", self.target_item_ids)
        _require_canonical_stages("final adjustment restart stages", self.restart_stages)
        if any(stage not in _FINAL_RESTART_STAGES for stage in self.restart_stages):
            raise ValueError("final adjustment restart stage is not allowed")
        _validate_invalidation(self.invalidation_scope)
        refs = _final_adjustment_refs(self)
        _validate_audit(self.audit, refs, "final dataset adjustment result")
        _validate_identity(
            object_id=self.result_id,
            object_sha256=self.result_sha256,
            expected_prefix="final-dataset-adjustment-result",
            observed=final_dataset_adjustment_result_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        source_request_ref: ObjectRef,
        source_preview_ref: ObjectRef,
        adjustments: tuple[TypedAdjustment, ...],
        replacement_preview_ref: ObjectRef,
        target_item_ids: tuple[str, ...],
        restart_stages: tuple[StageNameV2, ...],
        invalidation_scope: InvalidationScope,
        audit: ContractAudit,
    ) -> FinalDatasetAdjustmentResultV2:
        normalized_adjustments = _sorted_adjustments(adjustments)
        normalized_invalidation = _normalized_invalidation(invalidation_scope)
        normalized_stages = _sorted_stages(restart_stages)
        targets = tuple(sorted(set(target_item_ids)))
        refs = (
            source_request_ref,
            source_preview_ref,
            replacement_preview_ref,
            *normalized_invalidation.object_refs,
        )
        value = cls(
            result_id="final-dataset-adjustment-result://pending",
            source_request_ref=source_request_ref,
            source_preview_ref=source_preview_ref,
            adjustments=normalized_adjustments,
            replacement_preview_ref=replacement_preview_ref,
            target_item_ids=targets,
            restart_stages=normalized_stages,
            invalidation_scope=normalized_invalidation,
            result_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        return _finalize(
            value,
            id_field="result_id",
            hash_field="result_sha256",
            prefix="final-dataset-adjustment-result",
            digest=final_dataset_adjustment_result_v2_carried_sha256(value),
        )


class UserPlanApplicationV2(ContractModelV2):
    schema_version: Literal["eval-factory/user-plan-application/v2"] = "eval-factory/user-plan-application/v2"
    application_id: Identifier
    job_id: Identifier
    decision_commit_ref: ObjectRef
    decision_record_ref: ObjectRef
    adjustment_effect_ref: ObjectRef
    checkpoint: ApprovalCheckpoint
    source_root_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    replacement_root_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    invalidated_object_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    preserved_object_refs: tuple[ObjectRef, ...] = ()
    affected_item_ids: tuple[Identifier, ...] = Field(min_length=1)
    excluded_item_ids: tuple[Identifier, ...] = ()
    restart_stages: tuple[StageNameV2, ...] = Field(min_length=1)
    batch_revalidation_required: Literal[True] = True
    required_checkpoint_reapprovals: tuple[ApprovalCheckpoint, ...] = ()
    final_checkpoint_reapproval_required: bool
    policy_ref: ObjectRef
    policy_version: Literal["user-plan-application/r7-06-v1"] = USER_PLAN_APPLICATION_POLICY_VERSION
    application_sha256: Sha256
    audit: ContractAudit

    @field_validator("checkpoint", mode="before")
    @classmethod
    def parse_checkpoint(cls, value: object) -> ApprovalCheckpoint:
        return _parse_enum(value, ApprovalCheckpoint, "checkpoint")

    @field_validator("restart_stages", mode="before")
    @classmethod
    def parse_restart_stages(cls, value: object) -> tuple[StageNameV2, ...]:
        return _parse_enum_tuple(value, StageNameV2, "restart_stages")

    @field_validator("required_checkpoint_reapprovals", mode="before")
    @classmethod
    def parse_reapprovals(
        cls,
        value: object,
    ) -> tuple[ApprovalCheckpoint, ...]:
        return _parse_enum_tuple(
            value,
            ApprovalCheckpoint,
            "required_checkpoint_reapprovals",
        )

    @model_validator(mode="after")
    def validate_application(self) -> Self:
        _require_ref(
            self.decision_commit_ref,
            "user-decision-commit",
            "v2",
            "decision_commit_ref",
        )
        _require_ref(
            self.decision_record_ref,
            "user-decision-record",
            "v2",
            "decision_record_ref",
        )
        _require_ref(
            self.adjustment_effect_ref,
            "user-decision-adjustment-effect",
            "v2",
            "adjustment_effect_ref",
        )
        _require_ref(
            self.policy_ref,
            "user-plan-application-policy",
            "v2",
            "policy_ref",
        )
        for label, refs in (
            ("application source root refs", self.source_root_refs),
            ("application replacement root refs", self.replacement_root_refs),
            ("application invalidated refs", self.invalidated_object_refs),
            ("application preserved refs", self.preserved_object_refs),
        ):
            _require_sorted_unique_refs(label, refs)
            _require_safe_refs(refs, label)
        if set(self.source_root_refs).intersection(self.replacement_root_refs):
            raise ValueError("application source and replacement roots must differ")
        if set(self.invalidated_object_refs).intersection(self.preserved_object_refs):
            raise ValueError("application invalidated and preserved refs must be disjoint")
        _require_sorted_unique_values("application affected Item IDs", self.affected_item_ids)
        _require_sorted_unique_values("application excluded Item IDs", self.excluded_item_ids)
        if not set(self.excluded_item_ids).issubset(self.affected_item_ids):
            raise ValueError("application excluded Items must be affected Items")
        _require_canonical_stages("application restart stages", self.restart_stages)
        if self.restart_stages[-1] is not StageNameV2.RELEASE:
            raise ValueError("application restart stages must invalidate release")
        _require_canonical_checkpoints(
            "application checkpoint reapprovals",
            self.required_checkpoint_reapprovals,
        )
        if self.final_checkpoint_reapproval_required != (
            ApprovalCheckpoint.FINAL_DATASET_REVIEW in self.required_checkpoint_reapprovals
        ):
            raise ValueError("final checkpoint reapproval flag must match checkpoint inventory")
        _validate_audit(
            self.audit,
            _application_refs(self),
            "user plan application",
        )
        _validate_identity(
            object_id=self.application_id,
            object_sha256=self.application_sha256,
            expected_prefix="user-plan-application",
            observed=user_plan_application_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        job_id: str,
        decision_commit_ref: ObjectRef,
        decision_record_ref: ObjectRef,
        adjustment_effect_ref: ObjectRef,
        checkpoint: ApprovalCheckpoint,
        source_root_refs: tuple[ObjectRef, ...],
        replacement_root_refs: tuple[ObjectRef, ...],
        invalidated_object_refs: tuple[ObjectRef, ...],
        preserved_object_refs: tuple[ObjectRef, ...],
        affected_item_ids: tuple[str, ...],
        excluded_item_ids: tuple[str, ...],
        restart_stages: tuple[StageNameV2, ...],
        required_checkpoint_reapprovals: tuple[ApprovalCheckpoint, ...],
        policy_ref: ObjectRef,
        audit: ContractAudit,
    ) -> UserPlanApplicationV2:
        sources = _sorted_refs(source_root_refs)
        replacements = _sorted_refs(replacement_root_refs)
        invalidated = _sorted_refs(invalidated_object_refs)
        preserved = _sorted_refs(preserved_object_refs)
        affected = tuple(sorted(set(affected_item_ids)))
        excluded = tuple(sorted(set(excluded_item_ids)))
        stages = _sorted_stages(restart_stages)
        reapprovals = _sorted_checkpoints(required_checkpoint_reapprovals)
        refs = (
            decision_commit_ref,
            decision_record_ref,
            adjustment_effect_ref,
            *sources,
            *replacements,
            *invalidated,
            *preserved,
            policy_ref,
        )
        value = cls(
            application_id="user-plan-application://pending",
            job_id=job_id,
            decision_commit_ref=decision_commit_ref,
            decision_record_ref=decision_record_ref,
            adjustment_effect_ref=adjustment_effect_ref,
            checkpoint=checkpoint,
            source_root_refs=sources,
            replacement_root_refs=replacements,
            invalidated_object_refs=invalidated,
            preserved_object_refs=preserved,
            affected_item_ids=affected,
            excluded_item_ids=excluded,
            restart_stages=stages,
            required_checkpoint_reapprovals=reapprovals,
            final_checkpoint_reapproval_required=(ApprovalCheckpoint.FINAL_DATASET_REVIEW in reapprovals),
            policy_ref=policy_ref,
            application_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        return _finalize(
            value,
            id_field="application_id",
            hash_field="application_sha256",
            prefix="user-plan-application",
            digest=user_plan_application_v2_carried_sha256(value),
        )


class RevalidationWorkItemV2(ContractModelV2):
    schema_version: Literal["eval-factory/revalidation-work-item/v2"] = (
        "eval-factory/revalidation-work-item/v2"
    )
    work_item_id: Identifier
    application_ref: ObjectRef
    scope: RevalidationWorkScopeV2
    item_id: Identifier | None = None
    stage: StageNameV2
    invalidated_input_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    replacement_input_refs: tuple[ObjectRef, ...] = ()
    depends_on_work_item_refs: tuple[ObjectRef, ...] = ()
    required_output_types: tuple[Identifier, ...] = Field(min_length=1)
    hard_gate_required: bool
    policy_version: Literal["user-plan-application/r7-06-v1"] = USER_PLAN_APPLICATION_POLICY_VERSION
    work_sha256: Sha256

    @field_validator("scope", mode="before")
    @classmethod
    def parse_scope(cls, value: object) -> RevalidationWorkScopeV2:
        return _parse_enum(value, RevalidationWorkScopeV2, "scope")

    @field_validator("stage", mode="before")
    @classmethod
    def parse_stage(cls, value: object) -> StageNameV2:
        return _parse_enum(value, StageNameV2, "stage")

    @model_validator(mode="after")
    def validate_work(self) -> Self:
        _require_ref(
            self.application_ref,
            "user-plan-application",
            "v2",
            "application_ref",
        )
        if self.scope is RevalidationWorkScopeV2.ITEM:
            if self.item_id is None:
                raise ValueError("ITEM revalidation work requires item_id")
        elif self.item_id is not None:
            raise ValueError("JOB revalidation work cannot carry item_id")
        if self.stage is StageNameV2.RELEASE:
            raise ValueError("R7-06 cannot execute release work")
        for label, refs in (
            ("work invalidated input refs", self.invalidated_input_refs),
            ("work replacement input refs", self.replacement_input_refs),
            ("work dependency refs", self.depends_on_work_item_refs),
        ):
            _require_sorted_unique_refs(label, refs)
            _require_safe_refs(refs, label)
        for ref in self.depends_on_work_item_refs:
            _require_ref(ref, "revalidation-work-item", "v2", "depends_on_work_item_refs")
        if set(self.invalidated_input_refs).intersection(self.replacement_input_refs):
            raise ValueError("work invalidated and replacement inputs must be disjoint")
        _require_sorted_unique_values(
            "work required output types",
            self.required_output_types,
        )
        _validate_identity(
            object_id=self.work_item_id,
            object_sha256=self.work_sha256,
            expected_prefix="revalidation-work-item",
            observed=revalidation_work_item_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        application_ref: ObjectRef,
        scope: RevalidationWorkScopeV2,
        item_id: str | None,
        stage: StageNameV2,
        invalidated_input_refs: tuple[ObjectRef, ...],
        replacement_input_refs: tuple[ObjectRef, ...],
        depends_on_work_item_refs: tuple[ObjectRef, ...],
        required_output_types: tuple[str, ...],
        hard_gate_required: bool,
    ) -> RevalidationWorkItemV2:
        value = cls(
            work_item_id="revalidation-work-item://pending",
            application_ref=application_ref,
            scope=scope,
            item_id=item_id,
            stage=stage,
            invalidated_input_refs=_sorted_refs(invalidated_input_refs),
            replacement_input_refs=_sorted_refs(replacement_input_refs),
            depends_on_work_item_refs=_sorted_refs(depends_on_work_item_refs),
            required_output_types=tuple(sorted(set(required_output_types))),
            hard_gate_required=hard_gate_required,
            work_sha256="0" * 64,
        )
        return _finalize(
            value,
            id_field="work_item_id",
            hash_field="work_sha256",
            prefix="revalidation-work-item",
            digest=revalidation_work_item_v2_carried_sha256(value),
        )


class DirectedRevalidationPlanV2(ContractModelV2):
    schema_version: Literal["eval-factory/directed-revalidation-plan/v2"] = (
        "eval-factory/directed-revalidation-plan/v2"
    )
    plan_id: Identifier
    application_ref: ObjectRef
    resolved_job_work_graph_ref: ObjectRef
    work_items: tuple[RevalidationWorkItemV2, ...] = Field(min_length=1)
    work_item_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    preserved_item_ids: tuple[Identifier, ...] = ()
    excluded_item_ids: tuple[Identifier, ...] = ()
    work_item_count: int = Field(ge=1)
    policy_version: Literal["user-plan-application/r7-06-v1"] = USER_PLAN_APPLICATION_POLICY_VERSION
    plan_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_plan(self) -> Self:
        _require_ref(
            self.application_ref,
            "user-plan-application",
            "v2",
            "application_ref",
        )
        _require_ref(
            self.resolved_job_work_graph_ref,
            "resolved-job-work-graph",
            "v2",
            "resolved_job_work_graph_ref",
        )
        expected_refs = tuple(revalidation_work_item_v2_ref(value) for value in self.work_items)
        if self.work_item_refs != expected_refs:
            raise ValueError("revalidation plan work refs do not match nested work")
        if self.work_item_count != len(self.work_items):
            raise ValueError("revalidation plan work count is not exact")
        seen: set[ObjectRef] = set()
        prior_stage_index = -1
        prior_key: tuple[str, str] | None = None
        for work in self.work_items:
            if work.application_ref != self.application_ref:
                raise ValueError("revalidation work belongs to another application")
            stage_index = list(StageNameV2).index(work.stage)
            key = (work.stage.value, work.item_id or "")
            if stage_index < prior_stage_index or (
                stage_index == prior_stage_index and prior_key is not None and key < prior_key
            ):
                raise ValueError("revalidation work must use canonical stage and Item order")
            for dependency in work.depends_on_work_item_refs:
                if dependency not in seen:
                    raise ValueError("revalidation dependency must reference preceding work")
            seen.add(revalidation_work_item_v2_ref(work))
            prior_stage_index = stage_index
            prior_key = key
        _require_sorted_unique_values("plan preserved Item IDs", self.preserved_item_ids)
        _require_sorted_unique_values("plan excluded Item IDs", self.excluded_item_ids)
        if set(self.preserved_item_ids).intersection(self.excluded_item_ids):
            raise ValueError("plan preserved and excluded Items must be disjoint")
        _validate_audit(
            self.audit,
            (
                self.application_ref,
                self.resolved_job_work_graph_ref,
                *self.work_item_refs,
            ),
            "directed revalidation plan",
        )
        _validate_identity(
            object_id=self.plan_id,
            object_sha256=self.plan_sha256,
            expected_prefix="directed-revalidation-plan",
            observed=directed_revalidation_plan_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        application_ref: ObjectRef,
        resolved_job_work_graph_ref: ObjectRef,
        work_items: tuple[RevalidationWorkItemV2, ...],
        preserved_item_ids: tuple[str, ...],
        excluded_item_ids: tuple[str, ...],
        audit: ContractAudit,
    ) -> DirectedRevalidationPlanV2:
        refs = tuple(revalidation_work_item_v2_ref(value) for value in work_items)
        value = cls(
            plan_id="directed-revalidation-plan://pending",
            application_ref=application_ref,
            resolved_job_work_graph_ref=resolved_job_work_graph_ref,
            work_items=work_items,
            work_item_refs=refs,
            preserved_item_ids=tuple(sorted(set(preserved_item_ids))),
            excluded_item_ids=tuple(sorted(set(excluded_item_ids))),
            work_item_count=len(work_items),
            plan_sha256="0" * 64,
            audit=_safe_audit(
                audit,
                (application_ref, resolved_job_work_graph_ref, *refs),
            ),
        )
        return _finalize(
            value,
            id_field="plan_id",
            hash_field="plan_sha256",
            prefix="directed-revalidation-plan",
            digest=directed_revalidation_plan_v2_carried_sha256(value),
        )


class RevalidationWorkResultV2(ContractModelV2):
    schema_version: Literal["eval-factory/revalidation-work-result/v2"] = (
        "eval-factory/revalidation-work-result/v2"
    )
    result_id: Identifier
    work_item_ref: ObjectRef
    stage_result_ref: ObjectRef
    outcome: RevalidationWorkOutcomeV2
    output_refs: tuple[ObjectRef, ...] = ()
    failure_code: Identifier | None = None
    policy_version: Literal["user-plan-application/r7-06-v1"] = USER_PLAN_APPLICATION_POLICY_VERSION
    result_sha256: Sha256
    audit: ContractAudit

    @field_validator("outcome", mode="before")
    @classmethod
    def parse_outcome(cls, value: object) -> RevalidationWorkOutcomeV2:
        return _parse_enum(value, RevalidationWorkOutcomeV2, "outcome")

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        _require_ref(
            self.work_item_ref,
            "revalidation-work-item",
            "v2",
            "work_item_ref",
        )
        _require_ref(
            self.stage_result_ref,
            "stage-result",
            "record/v1",
            "stage_result_ref",
        )
        _require_sorted_unique_refs("revalidation output refs", self.output_refs)
        _require_safe_refs(self.output_refs, "revalidation output refs")
        if self.outcome is RevalidationWorkOutcomeV2.SUCCEEDED:
            if not self.output_refs or self.failure_code is not None:
                raise ValueError("successful revalidation requires outputs and no failure")
        elif self.output_refs or self.failure_code is None:
            raise ValueError("non-success revalidation requires one failure code and no outputs")
        _validate_audit(
            self.audit,
            (self.work_item_ref, self.stage_result_ref, *self.output_refs),
            "revalidation work result",
        )
        _validate_identity(
            object_id=self.result_id,
            object_sha256=self.result_sha256,
            expected_prefix="revalidation-work-result",
            observed=revalidation_work_result_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        work_item_ref: ObjectRef,
        stage_result_ref: ObjectRef,
        outcome: RevalidationWorkOutcomeV2,
        output_refs: tuple[ObjectRef, ...],
        failure_code: str | None,
        audit: ContractAudit,
    ) -> RevalidationWorkResultV2:
        outputs = _sorted_refs(output_refs)
        value = cls(
            result_id="revalidation-work-result://pending",
            work_item_ref=work_item_ref,
            stage_result_ref=stage_result_ref,
            outcome=outcome,
            output_refs=outputs,
            failure_code=failure_code,
            result_sha256="0" * 64,
            audit=_safe_audit(audit, (work_item_ref, stage_result_ref, *outputs)),
        )
        return _finalize(
            value,
            id_field="result_id",
            hash_field="result_sha256",
            prefix="revalidation-work-result",
            digest=revalidation_work_result_v2_carried_sha256(value),
        )


class DirectedRevalidationReportV2(ContractModelV2):
    schema_version: Literal["eval-factory/directed-revalidation-report/v2"] = (
        "eval-factory/directed-revalidation-report/v2"
    )
    report_id: Identifier
    application_ref: ObjectRef
    plan_ref: ObjectRef
    work_results: tuple[RevalidationWorkResultV2, ...] = Field(min_length=1)
    work_result_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    replacement_current_refs: tuple[ObjectRef, ...] = ()
    invalidated_prior_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    excluded_item_ids: tuple[Identifier, ...] = ()
    required_checkpoint_reapprovals: tuple[ApprovalCheckpoint, ...] = ()
    final_checkpoint_reapproval_required: bool
    outcome: DirectedRevalidationReportOutcomeV2
    policy_version: Literal["user-plan-application/r7-06-v1"] = USER_PLAN_APPLICATION_POLICY_VERSION
    report_sha256: Sha256
    audit: ContractAudit

    @field_validator("outcome", mode="before")
    @classmethod
    def parse_outcome(
        cls,
        value: object,
    ) -> DirectedRevalidationReportOutcomeV2:
        return _parse_enum(value, DirectedRevalidationReportOutcomeV2, "outcome")

    @field_validator("required_checkpoint_reapprovals", mode="before")
    @classmethod
    def parse_reapprovals(
        cls,
        value: object,
    ) -> tuple[ApprovalCheckpoint, ...]:
        return _parse_enum_tuple(
            value,
            ApprovalCheckpoint,
            "required_checkpoint_reapprovals",
        )

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        _require_ref(
            self.application_ref,
            "user-plan-application",
            "v2",
            "application_ref",
        )
        _require_ref(
            self.plan_ref,
            "directed-revalidation-plan",
            "v2",
            "plan_ref",
        )
        expected_refs = tuple(revalidation_work_result_v2_ref(value) for value in self.work_results)
        if self.work_result_refs != expected_refs:
            raise ValueError("revalidation report result refs do not match nested results")
        for label, refs in (
            ("report replacement current refs", self.replacement_current_refs),
            ("report invalidated prior refs", self.invalidated_prior_refs),
        ):
            _require_sorted_unique_refs(label, refs)
            _require_safe_refs(refs, label)
        if set(self.replacement_current_refs).intersection(self.invalidated_prior_refs):
            raise ValueError("report replacement and invalidated refs must be disjoint")
        _require_sorted_unique_values("report excluded Item IDs", self.excluded_item_ids)
        _require_canonical_checkpoints(
            "report checkpoint reapprovals",
            self.required_checkpoint_reapprovals,
        )
        if self.final_checkpoint_reapproval_required != (
            ApprovalCheckpoint.FINAL_DATASET_REVIEW in self.required_checkpoint_reapprovals
        ):
            raise ValueError("report final reapproval flag must match checkpoint inventory")
        outcomes = {value.outcome for value in self.work_results}
        if self.outcome is DirectedRevalidationReportOutcomeV2.COMPLETE:
            if outcomes != {RevalidationWorkOutcomeV2.SUCCEEDED}:
                raise ValueError("complete report requires every work result to succeed")
            if not self.replacement_current_refs:
                raise ValueError("complete report requires replacement current refs")
        else:
            if self.replacement_current_refs:
                raise ValueError("non-complete report cannot publish replacement current refs")
            if self.outcome is DirectedRevalidationReportOutcomeV2.BLOCKED and not outcomes.intersection(
                {
                    RevalidationWorkOutcomeV2.BLOCKED_POLICY,
                    RevalidationWorkOutcomeV2.BLOCKED_CAPABILITY,
                }
            ):
                raise ValueError("blocked report requires a blocked work result")
            if (
                self.outcome is DirectedRevalidationReportOutcomeV2.FAILED
                and RevalidationWorkOutcomeV2.TERMINAL_FAILURE not in outcomes
            ):
                raise ValueError("failed report requires a terminal failure")
        _validate_audit(
            self.audit,
            _report_refs(self),
            "directed revalidation report",
        )
        _validate_identity(
            object_id=self.report_id,
            object_sha256=self.report_sha256,
            expected_prefix="directed-revalidation-report",
            observed=directed_revalidation_report_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        application_ref: ObjectRef,
        plan_ref: ObjectRef,
        work_results: tuple[RevalidationWorkResultV2, ...],
        replacement_current_refs: tuple[ObjectRef, ...],
        invalidated_prior_refs: tuple[ObjectRef, ...],
        excluded_item_ids: tuple[str, ...],
        required_checkpoint_reapprovals: tuple[ApprovalCheckpoint, ...],
        outcome: DirectedRevalidationReportOutcomeV2,
        audit: ContractAudit,
    ) -> DirectedRevalidationReportV2:
        result_refs = tuple(revalidation_work_result_v2_ref(value) for value in work_results)
        replacements = _sorted_refs(replacement_current_refs)
        invalidated = _sorted_refs(invalidated_prior_refs)
        reapprovals = _sorted_checkpoints(required_checkpoint_reapprovals)
        value = cls(
            report_id="directed-revalidation-report://pending",
            application_ref=application_ref,
            plan_ref=plan_ref,
            work_results=work_results,
            work_result_refs=result_refs,
            replacement_current_refs=replacements,
            invalidated_prior_refs=invalidated,
            excluded_item_ids=tuple(sorted(set(excluded_item_ids))),
            required_checkpoint_reapprovals=reapprovals,
            final_checkpoint_reapproval_required=(ApprovalCheckpoint.FINAL_DATASET_REVIEW in reapprovals),
            outcome=outcome,
            report_sha256="0" * 64,
            audit=_safe_audit(
                audit,
                (
                    application_ref,
                    plan_ref,
                    *result_refs,
                    *replacements,
                    *invalidated,
                ),
            ),
        )
        return _finalize(
            value,
            id_field="report_id",
            hash_field="report_sha256",
            prefix="directed-revalidation-report",
            digest=directed_revalidation_report_v2_carried_sha256(value),
        )


def user_plan_application_policy_v2_carried_sha256(
    value: UserPlanApplicationPolicyV2,
) -> str:
    return _carried_sha256(value, {"policy_id", "policy_sha256", "audit"})


def label_plan_adjustment_result_v2_carried_sha256(
    value: LabelPlanAdjustmentResultV2,
) -> str:
    return _carried_sha256(value, {"result_id", "result_sha256", "audit"})


def environment_strategy_adjustment_result_v2_carried_sha256(
    value: EnvironmentStrategyAdjustmentResultV2,
) -> str:
    return _carried_sha256(value, {"result_id", "result_sha256", "audit"})


def final_dataset_adjustment_result_v2_carried_sha256(
    value: FinalDatasetAdjustmentResultV2,
) -> str:
    return _carried_sha256(value, {"result_id", "result_sha256", "audit"})


def user_plan_application_v2_carried_sha256(
    value: UserPlanApplicationV2,
) -> str:
    return _carried_sha256(value, {"application_id", "application_sha256", "audit"})


def revalidation_work_item_v2_carried_sha256(
    value: RevalidationWorkItemV2,
) -> str:
    return _carried_sha256(value, {"work_item_id", "work_sha256"})


def directed_revalidation_plan_v2_carried_sha256(
    value: DirectedRevalidationPlanV2,
) -> str:
    return _carried_sha256(value, {"plan_id", "plan_sha256", "audit"})


def revalidation_work_result_v2_carried_sha256(
    value: RevalidationWorkResultV2,
) -> str:
    return _carried_sha256(value, {"result_id", "result_sha256", "audit"})


def directed_revalidation_report_v2_carried_sha256(
    value: DirectedRevalidationReportV2,
) -> str:
    return _carried_sha256(value, {"report_id", "report_sha256", "audit"})


def user_plan_application_policy_v2_ref(
    value: UserPlanApplicationPolicyV2,
) -> ObjectRef:
    validate_user_plan_application_policy_v2_identity(value)
    return _object_ref(
        "user-plan-application-policy",
        value.policy_id,
        value.policy_sha256,
    )


def label_plan_adjustment_result_v2_ref(
    value: LabelPlanAdjustmentResultV2,
) -> ObjectRef:
    validate_label_plan_adjustment_result_v2_identity(value)
    return _object_ref(
        "label-plan-adjustment-result",
        value.result_id,
        value.result_sha256,
    )


def environment_strategy_adjustment_result_v2_ref(
    value: EnvironmentStrategyAdjustmentResultV2,
) -> ObjectRef:
    validate_environment_strategy_adjustment_result_v2_identity(value)
    return _object_ref(
        "environment-strategy-adjustment-result",
        value.result_id,
        value.result_sha256,
    )


def final_dataset_adjustment_result_v2_ref(
    value: FinalDatasetAdjustmentResultV2,
) -> ObjectRef:
    validate_final_dataset_adjustment_result_v2_identity(value)
    return _object_ref(
        "final-dataset-adjustment-result",
        value.result_id,
        value.result_sha256,
    )


def user_plan_application_v2_ref(value: UserPlanApplicationV2) -> ObjectRef:
    validate_user_plan_application_v2_identity(value)
    return _object_ref(
        "user-plan-application",
        value.application_id,
        value.application_sha256,
    )


def revalidation_work_item_v2_ref(value: RevalidationWorkItemV2) -> ObjectRef:
    validate_revalidation_work_item_v2_identity(value)
    return _object_ref(
        "revalidation-work-item",
        value.work_item_id,
        value.work_sha256,
    )


def directed_revalidation_plan_v2_ref(
    value: DirectedRevalidationPlanV2,
) -> ObjectRef:
    validate_directed_revalidation_plan_v2_identity(value)
    return _object_ref(
        "directed-revalidation-plan",
        value.plan_id,
        value.plan_sha256,
    )


def revalidation_work_result_v2_ref(
    value: RevalidationWorkResultV2,
) -> ObjectRef:
    validate_revalidation_work_result_v2_identity(value)
    return _object_ref(
        "revalidation-work-result",
        value.result_id,
        value.result_sha256,
    )


def directed_revalidation_report_v2_ref(
    value: DirectedRevalidationReportV2,
) -> ObjectRef:
    validate_directed_revalidation_report_v2_identity(value)
    return _object_ref(
        "directed-revalidation-report",
        value.report_id,
        value.report_sha256,
    )


def validate_user_plan_application_policy_v2_identity(
    value: UserPlanApplicationPolicyV2,
) -> None:
    _validate_current(
        value.policy_id,
        value.policy_sha256,
        "user-plan-application-policy",
        user_plan_application_policy_v2_carried_sha256(value),
    )


def validate_label_plan_adjustment_result_v2_identity(
    value: LabelPlanAdjustmentResultV2,
) -> None:
    _validate_audit(value.audit, _label_adjustment_refs(value), "label plan adjustment result")
    _validate_current(
        value.result_id,
        value.result_sha256,
        "label-plan-adjustment-result",
        label_plan_adjustment_result_v2_carried_sha256(value),
    )


def validate_environment_strategy_adjustment_result_v2_identity(
    value: EnvironmentStrategyAdjustmentResultV2,
) -> None:
    _validate_audit(
        value.audit,
        _environment_adjustment_refs(value),
        "environment strategy adjustment result",
    )
    _validate_current(
        value.result_id,
        value.result_sha256,
        "environment-strategy-adjustment-result",
        environment_strategy_adjustment_result_v2_carried_sha256(value),
    )


def validate_final_dataset_adjustment_result_v2_identity(
    value: FinalDatasetAdjustmentResultV2,
) -> None:
    _validate_audit(value.audit, _final_adjustment_refs(value), "final dataset adjustment result")
    _validate_current(
        value.result_id,
        value.result_sha256,
        "final-dataset-adjustment-result",
        final_dataset_adjustment_result_v2_carried_sha256(value),
    )


def validate_user_plan_application_v2_identity(
    value: UserPlanApplicationV2,
) -> None:
    _validate_audit(value.audit, _application_refs(value), "user plan application")
    _validate_current(
        value.application_id,
        value.application_sha256,
        "user-plan-application",
        user_plan_application_v2_carried_sha256(value),
    )


def validate_revalidation_work_item_v2_identity(
    value: RevalidationWorkItemV2,
) -> None:
    _validate_current(
        value.work_item_id,
        value.work_sha256,
        "revalidation-work-item",
        revalidation_work_item_v2_carried_sha256(value),
    )


def validate_directed_revalidation_plan_v2_identity(
    value: DirectedRevalidationPlanV2,
) -> None:
    _validate_audit(
        value.audit,
        (
            value.application_ref,
            value.resolved_job_work_graph_ref,
            *value.work_item_refs,
        ),
        "directed revalidation plan",
    )
    _validate_current(
        value.plan_id,
        value.plan_sha256,
        "directed-revalidation-plan",
        directed_revalidation_plan_v2_carried_sha256(value),
    )


def validate_revalidation_work_result_v2_identity(
    value: RevalidationWorkResultV2,
) -> None:
    _validate_audit(
        value.audit,
        (value.work_item_ref, value.stage_result_ref, *value.output_refs),
        "revalidation work result",
    )
    _validate_current(
        value.result_id,
        value.result_sha256,
        "revalidation-work-result",
        revalidation_work_result_v2_carried_sha256(value),
    )


def validate_directed_revalidation_report_v2_identity(
    value: DirectedRevalidationReportV2,
) -> None:
    _validate_audit(value.audit, _report_refs(value), "directed revalidation report")
    _validate_current(
        value.report_id,
        value.report_sha256,
        "directed-revalidation-report",
        directed_revalidation_report_v2_carried_sha256(value),
    )


def _label_adjustment_refs(value: LabelPlanAdjustmentResultV2) -> tuple[ObjectRef, ...]:
    return (
        value.source_request_ref,
        value.source_label_spec_ref,
        value.source_label_plan_ref,
        value.replacement_label_spec_ref,
        value.replacement_label_plan_ref,
        *value.invalidation_scope.object_refs,
    )


def _environment_adjustment_refs(
    value: EnvironmentStrategyAdjustmentResultV2,
) -> tuple[ObjectRef, ...]:
    return (
        value.source_request_ref,
        value.source_strategy_ref,
        value.replacement_strategy_ref,
        *value.invalidation_scope.object_refs,
    )


def _final_adjustment_refs(
    value: FinalDatasetAdjustmentResultV2,
) -> tuple[ObjectRef, ...]:
    return (
        value.source_request_ref,
        value.source_preview_ref,
        value.replacement_preview_ref,
        *value.invalidation_scope.object_refs,
    )


def _application_refs(value: UserPlanApplicationV2) -> tuple[ObjectRef, ...]:
    return (
        value.decision_commit_ref,
        value.decision_record_ref,
        value.adjustment_effect_ref,
        *value.source_root_refs,
        *value.replacement_root_refs,
        *value.invalidated_object_refs,
        *value.preserved_object_refs,
        value.policy_ref,
    )


def _report_refs(value: DirectedRevalidationReportV2) -> tuple[ObjectRef, ...]:
    return (
        value.application_ref,
        value.plan_ref,
        *value.work_result_refs,
        *value.replacement_current_refs,
        *value.invalidated_prior_refs,
    )


def _validate_adjustments(values: tuple[TypedAdjustment, ...]) -> None:
    paths = tuple(value.target_path for value in values)
    if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
        raise ValueError("adjustments must be sorted and unique by target path")


def _sorted_adjustments(
    values: tuple[TypedAdjustment, ...],
) -> tuple[TypedAdjustment, ...]:
    return tuple(sorted(values, key=lambda value: value.target_path))


def _validate_invalidation(value: InvalidationScope) -> None:
    _require_sorted_unique_refs("invalidation object refs", value.object_refs)
    _require_safe_refs(value.object_refs, "invalidation object refs")
    if value.stages != tuple(sorted(value.stages)) or len(value.stages) != len(set(value.stages)):
        raise ValueError("invalidation stages must be sorted and unique")


def _normalized_invalidation(value: InvalidationScope) -> InvalidationScope:
    return InvalidationScope(
        object_refs=_sorted_refs(value.object_refs),
        stages=tuple(sorted(set(value.stages))),
    )


def _carried_sha256(
    value: ContractModelV2,
    exclude: set[str],
) -> str:
    return _payload_sha256(
        value.model_dump(
            mode="json",
            exclude=exclude,
            exclude_none=False,
        )
    )


def _payload_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            canonical_value_v2(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _object_ref(
    object_type: str,
    object_id: str,
    object_sha256: str,
) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=object_id,
        object_version="v2",
        object_sha256=object_sha256,
    )


def _ref_key(value: ObjectRef) -> tuple[str, str, str, str]:
    return (
        value.object_type,
        value.object_id,
        value.object_version,
        value.object_sha256,
    )


def _sorted_refs(values: tuple[ObjectRef, ...]) -> tuple[ObjectRef, ...]:
    unique = {_ref_key(value): value for value in values}
    return tuple(unique[key] for key in sorted(unique))


def _require_sorted_unique_refs(
    label: str,
    values: tuple[ObjectRef, ...],
) -> None:
    keys = tuple(_ref_key(value) for value in values)
    if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
        raise ValueError(f"{label} must be sorted and unique")


def _require_sorted_unique_values(
    label: str,
    values: tuple[str, ...],
) -> None:
    if values != tuple(sorted(values)) or len(values) != len(set(values)):
        raise ValueError(f"{label} must be sorted and unique")


_DENIED_REF_MARKERS = frozenset(
    {
        "credential",
        "final-answer",
        "final-output",
        "grader-rule",
        "hidden-pass-condition",
        "hidden-selection-signal",
        "private-reference",
        "quarantine",
        "raw-trace",
        "raw-traj",
        "restricted-trace-span",
        "secret",
    }
)


def _require_safe_refs(
    values: tuple[ObjectRef, ...],
    label: str,
) -> None:
    if any(
        any(marker in f"{ref.object_type}:{ref.object_id}".casefold() for marker in _DENIED_REF_MARKERS)
        for ref in values
    ):
        raise ValueError(f"{label} contain a restricted reference")


def _require_ref(
    ref: ObjectRef,
    expected_type: str,
    expected_version: str,
    field_name: str,
) -> None:
    if ref.object_type != expected_type or ref.object_version != expected_version:
        raise ValueError(f"{field_name} must reference {expected_type} {expected_version}")


def _stage_key(value: StageNameV2) -> int:
    return list(StageNameV2).index(value)


def _sorted_stages(
    values: tuple[StageNameV2, ...],
) -> tuple[StageNameV2, ...]:
    return tuple(sorted(set(values), key=_stage_key))


def _require_canonical_stages(
    label: str,
    values: tuple[StageNameV2, ...],
) -> None:
    if values != _sorted_stages(values):
        raise ValueError(f"{label} must be canonical and unique")


def _checkpoint_key(value: ApprovalCheckpoint) -> int:
    return list(ApprovalCheckpoint).index(value)


def _sorted_checkpoints(
    values: tuple[ApprovalCheckpoint, ...],
) -> tuple[ApprovalCheckpoint, ...]:
    return tuple(sorted(set(values), key=_checkpoint_key))


def _require_canonical_checkpoints(
    label: str,
    values: tuple[ApprovalCheckpoint, ...],
) -> None:
    if values != _sorted_checkpoints(values):
        raise ValueError(f"{label} must be canonical and unique")


def _validate_audit(
    audit: ContractAudit,
    expected_refs: tuple[ObjectRef, ...],
    label: str,
) -> None:
    if audit.input_refs != _sorted_refs(expected_refs):
        raise ValueError(f"{label} audit input refs are not exact")
    bindings = tuple(
        value for value in audit.governing_versions if value.component == "user-plan-application"
    )
    if len(bindings) != 1 or bindings[0].version != USER_PLAN_APPLICATION_POLICY_VERSION:
        raise ValueError(f"{label} audit is missing the current policy")


def _safe_audit(
    audit: ContractAudit,
    refs: tuple[ObjectRef, ...],
) -> ContractAudit:
    versions = (
        *(value for value in audit.governing_versions if value.component != "user-plan-application"),
        VersionBinding(
            component="user-plan-application",
            version=USER_PLAN_APPLICATION_POLICY_VERSION,
        ),
    )
    return audit.model_copy(
        update={
            "governing_versions": tuple(
                sorted(
                    versions,
                    key=lambda value: (
                        value.component,
                        value.version,
                        value.sha256 or "",
                    ),
                )
            ),
            "input_refs": _sorted_refs(refs),
        }
    )


def _validate_identity(
    *,
    object_id: str,
    object_sha256: str,
    expected_prefix: str,
    observed: str,
) -> None:
    if object_id.endswith("://pending") or object_sha256 == "0" * 64:
        return
    if object_sha256 != observed or object_id != f"{expected_prefix}://sha256/{observed}":
        raise ValueError(f"{expected_prefix} identity is stale")


def _validate_current(
    object_id: str,
    object_sha256: str,
    expected_prefix: str,
    observed: str,
) -> None:
    if object_id.endswith("://pending") or object_sha256 == "0" * 64:
        raise ValueError(f"{expected_prefix} identity is pending")
    _validate_identity(
        object_id=object_id,
        object_sha256=object_sha256,
        expected_prefix=expected_prefix,
        observed=observed,
    )


def _finalize[ModelT: ContractModelV2](
    value: ModelT,
    *,
    id_field: str,
    hash_field: str,
    prefix: str,
    digest: str,
) -> ModelT:
    return value.model_copy(
        update={
            id_field: f"{prefix}://sha256/{digest}",
            hash_field: digest,
        }
    )


def _parse_enum[EnumT: StrEnum](
    value: object,
    enum_type: type[EnumT],
    field_name: str,
) -> EnumT:
    if isinstance(value, enum_type):
        return value
    if isinstance(value, str):
        return enum_type(value)
    raise TypeError(f"{field_name} must be a {enum_type.__name__}")


def _parse_enum_tuple[EnumT: StrEnum](
    value: object,
    enum_type: type[EnumT],
    field_name: str,
) -> tuple[EnumT, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{field_name} must be a collection")
    return tuple(item if isinstance(item, enum_type) else enum_type(item) for item in value)


__all__ = [
    "USER_PLAN_APPLICATION_POLICY_VERSION",
    "DirectedRevalidationPlanV2",
    "DirectedRevalidationReportOutcomeV2",
    "DirectedRevalidationReportV2",
    "EnvironmentStrategyAdjustmentResultV2",
    "FinalDatasetAdjustmentResultV2",
    "LabelPlanAdjustmentResultV2",
    "RevalidationWorkItemV2",
    "RevalidationWorkOutcomeV2",
    "RevalidationWorkResultV2",
    "RevalidationWorkScopeV2",
    "UserPlanApplicationPolicyV2",
    "UserPlanApplicationV2",
    "directed_revalidation_plan_v2_carried_sha256",
    "directed_revalidation_plan_v2_ref",
    "directed_revalidation_report_v2_carried_sha256",
    "directed_revalidation_report_v2_ref",
    "environment_strategy_adjustment_result_v2_carried_sha256",
    "environment_strategy_adjustment_result_v2_ref",
    "final_dataset_adjustment_result_v2_carried_sha256",
    "final_dataset_adjustment_result_v2_ref",
    "label_plan_adjustment_result_v2_carried_sha256",
    "label_plan_adjustment_result_v2_ref",
    "revalidation_work_item_v2_carried_sha256",
    "revalidation_work_item_v2_ref",
    "revalidation_work_result_v2_carried_sha256",
    "revalidation_work_result_v2_ref",
    "user_plan_application_policy_v2_carried_sha256",
    "user_plan_application_policy_v2_ref",
    "user_plan_application_v2_carried_sha256",
    "user_plan_application_v2_ref",
    "validate_directed_revalidation_plan_v2_identity",
    "validate_directed_revalidation_report_v2_identity",
    "validate_environment_strategy_adjustment_result_v2_identity",
    "validate_final_dataset_adjustment_result_v2_identity",
    "validate_label_plan_adjustment_result_v2_identity",
    "validate_revalidation_work_item_v2_identity",
    "validate_revalidation_work_result_v2_identity",
    "validate_user_plan_application_policy_v2_identity",
    "validate_user_plan_application_v2_identity",
]
