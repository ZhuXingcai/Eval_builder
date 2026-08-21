from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from pydantic import ValidationError

from eval_factory.contracts.approval import (
    ApprovalCheckpoint,
    EnvironmentAlternative,
    EnvironmentStrategy,
    EnvironmentStrategyChoice,
    InvalidationScope,
    LabelPlan,
    TypedAdjustment,
    UserApprovalRequest,
)
from eval_factory.contracts.approval_application_v2 import (
    USER_PLAN_APPLICATION_POLICY_VERSION,
    EnvironmentStrategyAdjustmentResultV2,
    FinalDatasetAdjustmentResultV2,
    LabelPlanAdjustmentResultV2,
    environment_strategy_adjustment_result_v2_ref,
    final_dataset_adjustment_result_v2_ref,
    label_plan_adjustment_result_v2_ref,
)
from eval_factory.contracts.approval_decision_v2 import (
    UserDecisionAdjustmentEffectV2,
)
from eval_factory.contracts.approval_v2 import (
    FinalDatasetReviewPreviewV2,
    environment_strategy_carried_sha256,
    environment_strategy_ref,
    final_dataset_review_preview_v2_ref,
    label_plan_carried_sha256,
    label_plan_ref,
    user_approval_request_ref,
    validate_user_approval_request_identity,
)
from eval_factory.contracts.core import (
    ContractAudit,
    ObjectRef,
    VersionBinding,
)
from eval_factory.contracts.core_v2 import canonical_value_v2
from eval_factory.contracts.labeling_v2 import LabelSpecV2
from eval_factory.contracts.orchestration_v2 import StageNameV2


class UserPlanAdjustmentPolicyError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class LabelPlanAdjustmentCompilation:
    result: LabelPlanAdjustmentResultV2
    replacement_label_spec: LabelSpecV2
    replacement_label_plan: LabelPlan
    effect: UserDecisionAdjustmentEffectV2


@dataclass(frozen=True, slots=True)
class EnvironmentStrategyAdjustmentCompilation:
    result: EnvironmentStrategyAdjustmentResultV2
    replacement_strategy: EnvironmentStrategy
    effect: UserDecisionAdjustmentEffectV2


@dataclass(frozen=True, slots=True)
class FinalDatasetAdjustmentCompilation:
    result: FinalDatasetAdjustmentResultV2
    replacement_preview: FinalDatasetReviewPreviewV2
    effect: UserDecisionAdjustmentEffectV2


_SELECTOR = r"([a-f0-9]{64})"
_LABEL_EXAMPLE_PATH = re.compile(rf"label_plan\.examples\[{_SELECTOR}\]\.(input_summary|expected_treatment)")
_LABEL_PREDICATE_PATH = re.compile(rf"label_spec\.predicates\[{_SELECTOR}\]\.expected_value")
_ENVIRONMENT_REQUIREMENT_PATH = re.compile(rf"requirements\[{_SELECTOR}\]\.description")
_ENVIRONMENT_ALTERNATIVE_PATH = re.compile(
    rf"requirements\[{_SELECTOR}\]\.alternatives\.([A-Z_]+)\.capability_impact"
)
_FINAL_ITEM_PATH = re.compile(rf"items\[{_SELECTOR}\]\.restart_stage")

_FINAL_DIRECTIVES = {
    "REBUILD_FROM_TASK_AUTHORING": StageNameV2.TASK_AUTHORING,
    "REBUILD_FROM_ATTACHMENT": StageNameV2.ATTACHMENT,
    "REBUILD_FROM_ITEM_QUALITY": StageNameV2.ITEM_QUALITY,
    "EXCLUDE_ITEM": StageNameV2.ITEM_QUALITY,
}


class LabelPlanAdjustmentCompiler:
    def compile(
        self,
        *,
        request: UserApprovalRequest,
        source_label_spec: LabelSpecV2,
        source_label_plan: LabelPlan,
        adjustments: tuple[TypedAdjustment, ...],
        audit: ContractAudit,
    ) -> LabelPlanAdjustmentCompilation:
        current_request = _current_request(
            request,
            ApprovalCheckpoint.LABEL_PLAN,
        )
        source_spec = LabelSpecV2.model_validate(source_label_spec.model_dump(mode="python"))
        source_plan = LabelPlan.model_validate(source_label_plan.model_dump(mode="python"))
        source_spec_ref = _label_spec_ref(source_spec)
        try:
            source_plan_ref = label_plan_ref(source_plan)
        except ValueError as exc:
            raise UserPlanAdjustmentPolicyError("label adjustment source is stale or malformed") from exc
        if (
            current_request.subject_refs != (source_spec_ref,)
            or current_request.plan_ref != source_plan_ref
            or current_request.preview_refs != (source_plan_ref,)
            or source_plan.label_spec_ref != source_spec_ref
        ):
            raise UserPlanAdjustmentPolicyError("label adjustment source does not match the approval request")
        normalized = _normalize_adjustments(adjustments)
        spec, plan = _apply_label_adjustments(
            source_spec,
            source_plan,
            normalized,
            audit=audit,
        )
        replacement_spec_ref = _label_spec_ref(spec)
        replacement_plan_ref = label_plan_ref(plan)
        invalidation = InvalidationScope(
            object_refs=(source_plan_ref, source_spec_ref),
            stages=tuple(
                sorted(
                    {
                        StageNameV2.LABEL.value,
                        StageNameV2.TASK_AUTHORING.value,
                        StageNameV2.ATTACHMENT.value,
                        StageNameV2.ITEM_QUALITY.value,
                        StageNameV2.BATCH_QUALITY.value,
                        StageNameV2.RELEASE.value,
                    }
                )
            ),
        )
        result = LabelPlanAdjustmentResultV2.create(
            source_request_ref=user_approval_request_ref(current_request),
            source_label_spec_ref=source_spec_ref,
            source_label_plan_ref=source_plan_ref,
            adjustments=normalized,
            replacement_label_spec_ref=replacement_spec_ref,
            replacement_label_plan_ref=replacement_plan_ref,
            invalidation_scope=invalidation,
            audit=audit,
        )
        effect = UserDecisionAdjustmentEffectV2.create(
            source_request_ref=user_approval_request_ref(current_request),
            checkpoint=current_request.checkpoint,
            source_subject_refs=current_request.subject_refs,
            source_plan_ref=current_request.plan_ref,
            adjustments=normalized,
            producer_result_ref=label_plan_adjustment_result_v2_ref(result),
            resulting_object_ref=replacement_plan_ref,
            related_result_refs=(replacement_spec_ref,),
            invalidation_scope=invalidation,
            audit=audit,
        )
        return LabelPlanAdjustmentCompilation(
            result=result,
            replacement_label_spec=spec,
            replacement_label_plan=plan,
            effect=effect,
        )


class EnvironmentStrategyAdjustmentCompiler:
    def compile(
        self,
        *,
        request: UserApprovalRequest,
        source_strategy: EnvironmentStrategy,
        adjustments: tuple[TypedAdjustment, ...],
        audit: ContractAudit,
    ) -> EnvironmentStrategyAdjustmentCompilation:
        current_request = _current_request(
            request,
            ApprovalCheckpoint.ENVIRONMENT_STRATEGY,
        )
        strategy = EnvironmentStrategy.model_validate(source_strategy.model_dump(mode="python"))
        try:
            source_ref = environment_strategy_ref(strategy)
        except ValueError as exc:
            raise UserPlanAdjustmentPolicyError(
                "environment adjustment source is stale or malformed"
            ) from exc
        affected = tuple(
            sorted(
                {ref for requirement in strategy.requirements for ref in requirement.affected_subject_refs},
                key=_ref_key,
            )
        )
        if (
            current_request.subject_refs != affected
            or current_request.plan_ref != source_ref
            or current_request.preview_refs != (source_ref,)
        ):
            raise UserPlanAdjustmentPolicyError(
                "environment adjustment source does not match the approval request"
            )
        normalized = _normalize_adjustments(adjustments)
        replacement = _apply_environment_adjustments(
            strategy,
            normalized,
            audit=audit,
        )
        replacement_ref = environment_strategy_ref(replacement)
        invalidation = InvalidationScope(
            object_refs=(source_ref,),
            stages=tuple(
                sorted(
                    {
                        StageNameV2.ENVIRONMENT_STRATEGY.value,
                        StageNameV2.TASK_AUTHORING.value,
                        StageNameV2.ATTACHMENT.value,
                        StageNameV2.ITEM_QUALITY.value,
                        StageNameV2.BATCH_QUALITY.value,
                        StageNameV2.RELEASE.value,
                    }
                )
            ),
        )
        result = EnvironmentStrategyAdjustmentResultV2.create(
            source_request_ref=user_approval_request_ref(current_request),
            source_strategy_ref=source_ref,
            adjustments=normalized,
            replacement_strategy_ref=replacement_ref,
            invalidation_scope=invalidation,
            audit=audit,
        )
        effect = UserDecisionAdjustmentEffectV2.create(
            source_request_ref=user_approval_request_ref(current_request),
            checkpoint=current_request.checkpoint,
            source_subject_refs=current_request.subject_refs,
            source_plan_ref=current_request.plan_ref,
            adjustments=normalized,
            producer_result_ref=environment_strategy_adjustment_result_v2_ref(result),
            resulting_object_ref=replacement_ref,
            related_result_refs=(),
            invalidation_scope=invalidation,
            audit=audit,
        )
        return EnvironmentStrategyAdjustmentCompilation(
            result=result,
            replacement_strategy=replacement,
            effect=effect,
        )


class FinalDatasetAdjustmentCompiler:
    def compile(
        self,
        *,
        request: UserApprovalRequest,
        source_preview: FinalDatasetReviewPreviewV2,
        adjustments: tuple[TypedAdjustment, ...],
        audit: ContractAudit,
    ) -> FinalDatasetAdjustmentCompilation:
        current_request = _current_request(
            request,
            ApprovalCheckpoint.FINAL_DATASET_REVIEW,
        )
        preview = FinalDatasetReviewPreviewV2.model_validate(source_preview.model_dump(mode="python"))
        try:
            source_ref = final_dataset_review_preview_v2_ref(preview)
        except ValueError as exc:
            raise UserPlanAdjustmentPolicyError("final adjustment source is stale or malformed") from exc
        if (
            current_request.subject_refs != preview.subject_refs
            or current_request.plan_ref is not None
            or current_request.preview_refs != (source_ref,)
        ):
            raise UserPlanAdjustmentPolicyError("final adjustment source does not match the approval request")
        normalized = _normalize_adjustments(adjustments)
        target_refs, restart_stages = _parse_final_directives(
            preview,
            normalized,
        )
        adjustment_digest = _payload_sha256(
            [value.model_dump(mode="json", exclude_none=False) for value in normalized]
        )
        replacement = FinalDatasetReviewPreviewV2.create(
            scope=preview.scope,
            subject_refs=preview.subject_refs,
            prompt_projection_refs=preview.prompt_projection_refs,
            selected_item_projection_refs=preview.selected_item_projection_refs,
            dataset_projection_ref=preview.dataset_projection_ref,
            quality_summary_refs=preview.quality_summary_refs,
            open_low_severity_finding_refs=(preview.open_low_severity_finding_refs),
            sample_navigation_refs=preview.sample_navigation_refs,
            projection_policy_version=(f"final-review-adjustment/r7-06-v1/{adjustment_digest[:16]}"),
            audit=_producer_audit(audit, (source_ref, *target_refs)),
        )
        replacement_ref = final_dataset_review_preview_v2_ref(replacement)
        invalidation_stages = {
            StageNameV2.BATCH_QUALITY,
            StageNameV2.FINAL_DATASET_REVIEW,
            StageNameV2.RELEASE,
            *restart_stages,
        }
        if StageNameV2.TASK_AUTHORING in restart_stages:
            invalidation_stages.update(
                {
                    StageNameV2.ATTACHMENT,
                    StageNameV2.ITEM_QUALITY,
                }
            )
        if StageNameV2.ATTACHMENT in restart_stages:
            invalidation_stages.add(StageNameV2.ITEM_QUALITY)
        invalidation = InvalidationScope(
            object_refs=target_refs,
            stages=tuple(sorted(stage.value for stage in invalidation_stages)),
        )
        result = FinalDatasetAdjustmentResultV2.create(
            source_request_ref=user_approval_request_ref(current_request),
            source_preview_ref=source_ref,
            adjustments=normalized,
            replacement_preview_ref=replacement_ref,
            target_item_ids=tuple(ref.object_id for ref in target_refs),
            restart_stages=restart_stages,
            invalidation_scope=invalidation,
            audit=audit,
        )
        effect = UserDecisionAdjustmentEffectV2.create(
            source_request_ref=user_approval_request_ref(current_request),
            checkpoint=current_request.checkpoint,
            source_subject_refs=current_request.subject_refs,
            source_plan_ref=None,
            adjustments=normalized,
            producer_result_ref=final_dataset_adjustment_result_v2_ref(result),
            resulting_object_ref=replacement_ref,
            related_result_refs=(),
            invalidation_scope=invalidation,
            audit=audit,
        )
        return FinalDatasetAdjustmentCompilation(
            result=result,
            replacement_preview=replacement,
            effect=effect,
        )


def _apply_label_adjustments(
    source_spec: LabelSpecV2,
    source_plan: LabelPlan,
    adjustments: tuple[TypedAdjustment, ...],
    *,
    audit: ContractAudit,
) -> tuple[LabelSpecV2, LabelPlan]:
    spec_updates: dict[str, object] = {}
    plan_updates: dict[str, object] = {}
    examples = list(source_plan.examples)
    predicates = {
        _selector(value.predicate_id): (group_name, index, value)
        for group_name in (
            "prerequisite_predicates",
            "positive_predicates",
            "negative_predicates",
        )
        for index, value in enumerate(getattr(source_spec, group_name))
    }
    predicate_groups = {
        group_name: list(getattr(source_spec, group_name))
        for group_name in (
            "prerequisite_predicates",
            "positive_predicates",
            "negative_predicates",
        )
    }
    abstain_rules = list(source_plan.abstain_rules)
    blind_spots = list(source_plan.blind_spots)
    for adjustment in adjustments:
        path = adjustment.target_path
        if path == "label_spec.requirement":
            spec_updates["requirement"] = _changed_text(
                adjustment,
                source_spec.requirement,
                path,
            )
            continue
        if path in {
            "label_spec.decision_threshold",
            "label_spec.review_threshold",
        }:
            current = getattr(source_spec, path.rsplit(".", 1)[1])
            value = adjustment.value
            if (
                adjustment.operation not in {"SET", "REPLACE"}
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0 <= float(value) <= 1
            ):
                raise UserPlanAdjustmentPolicyError(f"{path} requires a numeric SET/REPLACE value")
            if float(value) == current:
                raise UserPlanAdjustmentPolicyError(f"{path} adjustment is a no-op")
            spec_updates[path.rsplit(".", 1)[1]] = float(value)
            continue
        match = _LABEL_PREDICATE_PATH.fullmatch(path)
        if match is not None:
            selector = match.group(1)
            selected = predicates.get(selector)
            if selected is None:
                raise UserPlanAdjustmentPolicyError("label predicate selector is unknown")
            if adjustment.operation not in {"SET", "REPLACE"}:
                raise UserPlanAdjustmentPolicyError("label predicate expected_value requires SET/REPLACE")
            group_name, index, predicate = selected
            if predicate.expected_value == adjustment.value:
                raise UserPlanAdjustmentPolicyError("label predicate adjustment is a no-op")
            predicate_groups[group_name][index] = predicate.model_copy(
                update={"expected_value": adjustment.value}
            )
            continue
        if path in {"label_plan.intent", "label_plan.boundary"}:
            field = path.rsplit(".", 1)[1]
            plan_updates[field] = _changed_text(
                adjustment,
                getattr(source_plan, field),
                path,
            )
            continue
        if path in {"label_plan.abstain_rules", "label_plan.blind_spots"}:
            if adjustment.operation != "ADD":
                raise UserPlanAdjustmentPolicyError(f"{path} supports ADD only")
            value = _nonempty_string(adjustment.value, path)
            values = abstain_rules if path.endswith("abstain_rules") else blind_spots
            if value in values:
                raise UserPlanAdjustmentPolicyError(f"{path} adjustment is a no-op")
            values.append(value)
            continue
        match = _LABEL_EXAMPLE_PATH.fullmatch(path)
        if match is not None:
            selector, field = match.groups()
            example_index = next(
                (
                    position
                    for position, example in enumerate(examples)
                    if _selector(example.example_id) == selector
                ),
                None,
            )
            if example_index is None:
                raise UserPlanAdjustmentPolicyError("label example selector is unknown")
            value = _changed_text(
                adjustment,
                getattr(examples[example_index], field),
                path,
            )
            examples[example_index] = examples[example_index].model_copy(update={field: value})
            continue
        raise UserPlanAdjustmentPolicyError(f"label adjustment target path is not allowed: {path}")

    spec_updates.update(
        {
            key: tuple(values)
            for key, values in predicate_groups.items()
            if tuple(values) != getattr(source_spec, key)
        }
    )
    try:
        pending_spec = LabelSpecV2.model_validate(
            source_spec.model_copy(
                update={
                    **spec_updates,
                    "label_spec_id": "label-spec://pending",
                    "label_plan_ref": None,
                    "label_spec_sha256": "0" * 64,
                    "audit": _producer_audit(
                        audit,
                        (_label_spec_ref(source_spec), label_plan_ref(source_plan)),
                    ),
                }
            ).model_dump(mode="python")
        )
    except ValidationError as exc:
        raise UserPlanAdjustmentPolicyError("replacement label specification is invalid") from exc
    spec_digest = _label_spec_carried_sha256(pending_spec)
    replacement_spec = pending_spec.model_copy(
        update={
            "label_spec_id": f"label-spec://sha256/{spec_digest}",
            "label_spec_sha256": spec_digest,
        }
    )
    replacement_spec_ref = _label_spec_ref(replacement_spec)
    plan_updates.update(
        {
            "examples": tuple(examples),
            "abstain_rules": tuple(sorted(abstain_rules)),
            "blind_spots": tuple(sorted(blind_spots)),
            "label_plan_id": "label-plan://pending",
            "label_spec_ref": replacement_spec_ref,
            "audit": _producer_audit(audit, (replacement_spec_ref,)),
        }
    )
    try:
        pending_plan = LabelPlan.model_validate(
            source_plan.model_copy(update=plan_updates).model_dump(mode="python")
        )
    except ValidationError as exc:
        raise UserPlanAdjustmentPolicyError("replacement label plan is invalid") from exc
    plan_digest = label_plan_carried_sha256(pending_plan)
    replacement_plan = pending_plan.model_copy(update={"label_plan_id": f"label-plan://sha256/{plan_digest}"})
    try:
        replacement_plan = LabelPlan.model_validate(replacement_plan.model_dump(mode="python"))
        label_plan_ref(replacement_plan)
    except (ValidationError, ValueError) as exc:
        raise UserPlanAdjustmentPolicyError("replacement label plan is invalid") from exc
    return replacement_spec, replacement_plan


def _apply_environment_adjustments(
    source: EnvironmentStrategy,
    adjustments: tuple[TypedAdjustment, ...],
    *,
    audit: ContractAudit,
) -> EnvironmentStrategy:
    requirements = list(source.requirements)
    strategy_updates: dict[str, object] = {}
    requirement_by_selector = {
        _selector(value.requirement_id): index for index, value in enumerate(requirements)
    }
    for adjustment in adjustments:
        path = adjustment.target_path
        if path == "recommended_strategy":
            if adjustment.operation not in {"SET", "REPLACE"}:
                raise UserPlanAdjustmentPolicyError("recommended_strategy requires SET/REPLACE")
            if not isinstance(adjustment.value, str):
                raise UserPlanAdjustmentPolicyError("recommended_strategy is invalid")
            try:
                value = EnvironmentStrategyChoice(adjustment.value)
            except (TypeError, ValueError) as exc:
                raise UserPlanAdjustmentPolicyError("recommended_strategy is invalid") from exc
            if value is source.recommended_strategy:
                raise UserPlanAdjustmentPolicyError("recommended_strategy adjustment is a no-op")
            strategy_updates["recommended_strategy"] = value
            continue
        if path == "recommendation_reason":
            strategy_updates["recommendation_reason"] = _changed_text(
                adjustment,
                source.recommendation_reason,
                path,
            )
            continue
        match = _ENVIRONMENT_REQUIREMENT_PATH.fullmatch(path)
        if match is not None:
            index = requirement_by_selector.get(match.group(1))
            if index is None:
                raise UserPlanAdjustmentPolicyError("environment requirement selector is unknown")
            requirement = requirements[index]
            requirements[index] = requirement.model_copy(
                update={
                    "description": _changed_text(
                        adjustment,
                        requirement.description,
                        path,
                    )
                }
            )
            continue
        match = _ENVIRONMENT_ALTERNATIVE_PATH.fullmatch(path)
        if match is not None:
            index = requirement_by_selector.get(match.group(1))
            if index is None:
                raise UserPlanAdjustmentPolicyError("environment requirement selector is unknown")
            try:
                choice = EnvironmentStrategyChoice(match.group(2))
            except ValueError as exc:
                raise UserPlanAdjustmentPolicyError("environment alternative selector is unknown") from exc
            requirement = requirements[index]
            alternatives = list(requirement.alternatives)
            alternative_index = next(
                (position for position, value in enumerate(alternatives) if value.strategy is choice),
                None,
            )
            if alternative_index is None:
                raise UserPlanAdjustmentPolicyError("environment alternative selector is unknown")
            alternative = alternatives[alternative_index]
            alternatives[alternative_index] = EnvironmentAlternative(
                strategy=alternative.strategy,
                feasible=alternative.feasible,
                capability_impact=_changed_text(
                    adjustment,
                    alternative.capability_impact,
                    path,
                ),
                blocking_reason=alternative.blocking_reason,
            )
            requirements[index] = requirement.model_copy(update={"alternatives": tuple(alternatives)})
            continue
        raise UserPlanAdjustmentPolicyError(f"environment adjustment target path is not allowed: {path}")
    if (
        "recommended_strategy" in strategy_updates
        and source.recommendation_reason is None
        and "recommendation_reason" not in strategy_updates
    ):
        raise UserPlanAdjustmentPolicyError("recommended strategy requires a recommendation reason")
    affected_refs = tuple(
        sorted(
            {ref for requirement in requirements for ref in requirement.affected_subject_refs},
            key=_ref_key,
        )
    )
    try:
        pending = EnvironmentStrategy.model_validate(
            source.model_copy(
                update={
                    **strategy_updates,
                    "requirements": tuple(requirements),
                    "environment_strategy_id": "environment-strategy://pending",
                    "audit": _producer_audit(audit, affected_refs),
                }
            ).model_dump(mode="python")
        )
    except ValidationError as exc:
        raise UserPlanAdjustmentPolicyError("replacement environment strategy is invalid") from exc
    digest = environment_strategy_carried_sha256(pending)
    replacement = pending.model_copy(
        update={"environment_strategy_id": (f"environment-strategy://sha256/{digest}")}
    )
    try:
        replacement = EnvironmentStrategy.model_validate(replacement.model_dump(mode="python"))
        environment_strategy_ref(replacement)
    except (ValidationError, ValueError) as exc:
        raise UserPlanAdjustmentPolicyError("replacement environment strategy is invalid") from exc
    return replacement


def _parse_final_directives(
    preview: FinalDatasetReviewPreviewV2,
    adjustments: tuple[TypedAdjustment, ...],
) -> tuple[tuple[ObjectRef, ...], tuple[StageNameV2, ...]]:
    by_selector = {_selector(ref.object_id): ref for ref in preview.subject_refs}
    targets: list[ObjectRef] = []
    stages: list[StageNameV2] = []
    for adjustment in adjustments:
        match = _FINAL_ITEM_PATH.fullmatch(adjustment.target_path)
        if match is None:
            raise UserPlanAdjustmentPolicyError("final adjustment target path is not allowed")
        if adjustment.operation not in {"SET", "REPLACE"}:
            raise UserPlanAdjustmentPolicyError("final restart directive requires SET/REPLACE")
        target = by_selector.get(match.group(1))
        if target is None:
            raise UserPlanAdjustmentPolicyError("final adjustment Item selector is unknown")
        if not isinstance(adjustment.value, str):
            raise UserPlanAdjustmentPolicyError("final restart directive must be a closed string")
        stage = _FINAL_DIRECTIVES.get(adjustment.value)
        if stage is None:
            raise UserPlanAdjustmentPolicyError("final restart directive is unsupported")
        targets.append(target)
        stages.append(stage)
    ordered_targets = tuple(sorted(set(targets), key=_ref_key))
    ordered_stages = tuple(sorted(set(stages), key=lambda value: list(StageNameV2).index(value)))
    return ordered_targets, ordered_stages


def _current_request(
    value: UserApprovalRequest,
    checkpoint: ApprovalCheckpoint,
) -> UserApprovalRequest:
    try:
        request = UserApprovalRequest.model_validate(value.model_dump(mode="python"))
        validate_user_approval_request_identity(request)
    except (ValidationError, ValueError) as exc:
        raise UserPlanAdjustmentPolicyError("approval request is stale or malformed") from exc
    if request.checkpoint is not checkpoint:
        raise UserPlanAdjustmentPolicyError(
            "approval request checkpoint does not match the adjustment producer"
        )
    return request


def _normalize_adjustments(
    values: tuple[TypedAdjustment, ...],
) -> tuple[TypedAdjustment, ...]:
    if not values:
        raise UserPlanAdjustmentPolicyError("adjustment producer requires typed adjustments")
    ordered = tuple(sorted(values, key=lambda value: value.target_path))
    paths = tuple(value.target_path for value in ordered)
    if len(paths) != len(set(paths)):
        raise UserPlanAdjustmentPolicyError("adjustment target paths must be unique")
    return ordered


def _changed_text(
    adjustment: TypedAdjustment,
    current: str | None,
    path: str,
) -> str:
    if adjustment.operation not in {"SET", "REPLACE"}:
        raise UserPlanAdjustmentPolicyError(f"{path} requires SET/REPLACE")
    value = _nonempty_string(adjustment.value, path)
    if value == current:
        raise UserPlanAdjustmentPolicyError(f"{path} adjustment is a no-op")
    return value


def _nonempty_string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise UserPlanAdjustmentPolicyError(f"{path} requires a non-empty string")
    return value.strip()


def _label_spec_ref(value: LabelSpecV2) -> ObjectRef:
    return ObjectRef(
        object_type="label-spec",
        object_id=value.label_spec_id,
        object_version=value.label_version,
        object_sha256=value.label_spec_sha256,
    )


def _label_spec_carried_sha256(value: LabelSpecV2) -> str:
    return _payload_sha256(
        value.model_dump(
            mode="json",
            exclude={"label_spec_id", "label_spec_sha256", "audit"},
            exclude_none=False,
        )
    )


def _selector(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _ref_key(value: ObjectRef) -> tuple[str, str, str, str]:
    return (
        value.object_type,
        value.object_id,
        value.object_version,
        value.object_sha256,
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


def _producer_audit(
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
            "input_refs": tuple(sorted(set(refs), key=_ref_key)),
        }
    )


__all__ = [
    "EnvironmentStrategyAdjustmentCompilation",
    "EnvironmentStrategyAdjustmentCompiler",
    "FinalDatasetAdjustmentCompilation",
    "FinalDatasetAdjustmentCompiler",
    "LabelPlanAdjustmentCompilation",
    "LabelPlanAdjustmentCompiler",
    "UserPlanAdjustmentPolicyError",
]
