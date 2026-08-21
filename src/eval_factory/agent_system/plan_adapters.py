from __future__ import annotations

from typing import Any, Protocol, cast

from pydantic import ValidationError

from eval_factory.agent_system.attachment_planning import (
    AttachmentGenerationPlanCompiler,
    AttachmentGenerationPlanCompilerError,
)
from eval_factory.agent_system.criteria_planning import (
    CriteriaRubricPlanCompiler,
    CriteriaRubricPlanCompilerError,
)
from eval_factory.agent_system.delivery_planning import (
    DatasetDeliveryPlanCompiler,
    DatasetDeliveryPlanCompilerError,
)
from eval_factory.agent_system.grading_planning import (
    GradingDesignPlanCompiler,
    GradingDesignPlanCompilerError,
)
from eval_factory.agent_system.planner import (
    DatasetBuildPlanCompiler,
    DatasetBuildPlanCompilerError,
)
from eval_factory.contracts.agent_system_v2 import (
    AttachmentGenerationPlanV2,
    AttachmentMockWorkV2,
    CompiledAttachmentGenerationPlanV2,
    CompiledCriteriaRubricPlanV2,
    CompiledDatasetBuildPlanV2,
    CompiledGradingDesignPlanV2,
    CriteriaRubricPlanV2,
    DatasetBuildPlanV2,
    FactoryRunPolicyV2,
    GradingDesignPlanV2,
    PlanKindV2,
    PlanReviewPresentationV2,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.core_v2 import ContractModelV2
from eval_factory.contracts.dataset_runtime_v2 import (
    CompiledDatasetDeliveryPlanV2,
    DatasetDeliveryPlanV2,
)

type ReviewablePlanV2 = (
    DatasetBuildPlanV2
    | AttachmentGenerationPlanV2
    | CriteriaRubricPlanV2
    | GradingDesignPlanV2
    | DatasetDeliveryPlanV2
)
type CompiledReviewablePlanV2 = (
    CompiledDatasetBuildPlanV2
    | CompiledAttachmentGenerationPlanV2
    | CompiledCriteriaRubricPlanV2
    | CompiledGradingDesignPlanV2
    | CompiledDatasetDeliveryPlanV2
)


class ReviewablePlanAdapterError(RuntimeError):
    pass


class ReviewablePlanAdapter[
    PlanT: ContractModelV2,
    CompiledT: ContractModelV2,
](Protocol):
    plan_kind: PlanKindV2
    object_type: str
    compiled_object_type: str

    def parse(self, payload: bytes) -> PlanT: ...

    def parse_compiled(self, payload: bytes) -> CompiledT: ...

    def compile(
        self,
        plan: PlanT,
        *,
        policy: FactoryRunPolicyV2,
        audit: ContractAudit,
    ) -> CompiledT: ...

    def validate_successor(self, base: PlanT, successor: PlanT) -> None: ...

    def safe_presentation(
        self,
        plan: PlanT,
    ) -> PlanReviewPresentationV2: ...

    def invalidated_refs(
        self,
        base: PlanT,
        successor: PlanT,
    ) -> tuple[ObjectRef, ...]: ...


type _ErasedPlanAdapter = ReviewablePlanAdapter[Any, Any]


class ReviewablePlanAdapterRegistry:
    def __init__(
        self,
        adapters: tuple[_ErasedPlanAdapter, ...],
    ) -> None:
        self._adapters: dict[
            tuple[PlanKindV2, str],
            _ErasedPlanAdapter,
        ] = {}
        for adapter in adapters:
            key = (adapter.plan_kind, adapter.object_type)
            if key in self._adapters:
                raise ReviewablePlanAdapterError("duplicate reviewable plan adapter")
            self._adapters[key] = adapter

    @property
    def adapters(self) -> tuple[_ErasedPlanAdapter, ...]:
        return tuple(
            self._adapters[key]
            for key in sorted(
                self._adapters,
                key=lambda value: (value[0].value, value[1]),
            )
        )

    def resolve(
        self,
        plan_kind: PlanKindV2,
        object_type: str,
    ) -> _ErasedPlanAdapter:
        adapter = self._adapters.get((plan_kind, object_type))
        if adapter is None:
            raise ReviewablePlanAdapterError("unknown reviewable plan kind or object type")
        return adapter


class GlobalBuildPlanAdapter:
    plan_kind = PlanKindV2.GLOBAL_BUILD
    object_type = "dataset-build-plan"
    compiled_object_type = "compiled-dataset-build-plan"

    def __init__(
        self,
        compiler: DatasetBuildPlanCompiler | None,
    ) -> None:
        self.compiler = compiler

    def parse(self, payload: bytes) -> DatasetBuildPlanV2:
        try:
            return DatasetBuildPlanV2.model_validate_json(payload)
        except ValidationError as exc:
            raise ReviewablePlanAdapterError("global build plan material is invalid") from exc

    def parse_compiled(
        self,
        payload: bytes,
    ) -> CompiledDatasetBuildPlanV2:
        try:
            return CompiledDatasetBuildPlanV2.model_validate_json(payload)
        except ValidationError as exc:
            raise ReviewablePlanAdapterError("compiled global build plan material is invalid") from exc

    def compile(
        self,
        plan: DatasetBuildPlanV2,
        *,
        policy: FactoryRunPolicyV2,
        audit: ContractAudit,
    ) -> CompiledDatasetBuildPlanV2:
        if self.compiler is None:
            raise ReviewablePlanAdapterError("global plan edit requires a deterministic compiler")
        try:
            return self.compiler.compile(
                plan=plan,
                policy=policy,
                audit=audit,
            )
        except DatasetBuildPlanCompilerError as exc:
            raise ReviewablePlanAdapterError("global plan failed deterministic compilation") from exc

    def validate_successor(
        self,
        base: DatasetBuildPlanV2,
        successor: DatasetBuildPlanV2,
    ) -> None:
        if (
            successor.plan_id != base.plan_id
            or successor.plan_version != base.plan_version + 1
            or successor.predecessor_plan_ref != base.to_ref()
        ):
            raise ReviewablePlanAdapterError("global plan successor is not contiguous")

    def safe_presentation(
        self,
        plan: DatasetBuildPlanV2,
    ) -> PlanReviewPresentationV2:
        return PlanReviewPresentationV2.create(
            presentation_id=(f"plan-review-presentation://global/{_slug(plan.plan_id)}/{plan.plan_version}"),
            plan_ref=plan.to_ref(),
            plan_kind=self.plan_kind,
            title="Review the global dataset build plan",
            summary_lines=(
                f"{len(plan.tasks)} planned Agent tasks.",
                f"{plan.total_model_requests} maximum model requests.",
            ),
            editable_paths=(
                "assumptions",
                "goals",
                "required_review_kinds",
                "stage_order",
                "tasks",
                "unresolved_questions",
                "user_constraints",
            ),
            warning_codes=("CORE_VERTICAL_ONLY",),
            audit=plan.audit,
        )

    def invalidated_refs(
        self,
        base: DatasetBuildPlanV2,
        successor: DatasetBuildPlanV2,
    ) -> tuple[ObjectRef, ...]:
        base_tasks = {task.task_key: task for task in base.tasks}
        successor_tasks = {task.task_key: task for task in successor.tasks}
        changed = {
            task_key
            for task_key in set(base_tasks) | set(successor_tasks)
            if base_tasks.get(task_key) != successor_tasks.get(task_key)
        }
        refs = {
            reference
            for task_key in changed
            for task in (
                base_tasks.get(task_key),
                successor_tasks.get(task_key),
            )
            if task is not None
            for reference in task.acceptance_check_refs
        }
        return tuple(sorted(refs, key=_ref_key))


class AttachmentGenerationPlanAdapter:
    plan_kind = PlanKindV2.ATTACHMENT_GENERATION
    object_type = "attachment-generation-plan"
    compiled_object_type = "compiled-attachment-generation-plan"

    def __init__(
        self,
        compiler: AttachmentGenerationPlanCompiler | None,
    ) -> None:
        self.compiler = compiler

    def parse(self, payload: bytes) -> AttachmentGenerationPlanV2:
        try:
            return AttachmentGenerationPlanV2.model_validate_json(payload)
        except ValidationError as exc:
            raise ReviewablePlanAdapterError("attachment generation plan material is invalid") from exc

    def parse_compiled(
        self,
        payload: bytes,
    ) -> CompiledAttachmentGenerationPlanV2:
        try:
            return CompiledAttachmentGenerationPlanV2.model_validate_json(payload)
        except ValidationError as exc:
            raise ReviewablePlanAdapterError("compiled attachment plan material is invalid") from exc

    def compile(
        self,
        plan: AttachmentGenerationPlanV2,
        *,
        policy: FactoryRunPolicyV2,
        audit: ContractAudit,
    ) -> CompiledAttachmentGenerationPlanV2:
        if self.compiler is None:
            raise ReviewablePlanAdapterError("attachment plan edit requires a deterministic compiler")
        try:
            return self.compiler.compile(
                plan=plan,
                policy=policy,
                audit=audit,
            )
        except AttachmentGenerationPlanCompilerError as exc:
            raise ReviewablePlanAdapterError("attachment plan failed deterministic compilation") from exc

    def validate_successor(
        self,
        base: AttachmentGenerationPlanV2,
        successor: AttachmentGenerationPlanV2,
    ) -> None:
        if (
            successor.plan_id != base.plan_id
            or successor.plan_version != base.plan_version + 1
            or successor.predecessor_plan_ref != base.to_ref()
        ):
            raise ReviewablePlanAdapterError("attachment plan successor is not contiguous")
        if (
            successor.producer_task_view_ref != base.producer_task_view_ref
            or successor.evidence_bundle_ref != base.evidence_bundle_ref
            or successor.attachment_planning_context_ref != base.attachment_planning_context_ref
            or successor.quality_policy_ref != base.quality_policy_ref
            or successor.solvability_policy_ref != base.solvability_policy_ref
        ):
            raise ReviewablePlanAdapterError("attachment plan successor widens source or policy scope")

        base_by_group = {work.artifact_group_ref: work for work in base.works}
        successor_by_group = {work.artifact_group_ref: work for work in successor.works}
        if set(base_by_group) != set(successor_by_group):
            raise ReviewablePlanAdapterError("attachment plan successor changes artifact groups")
        for group_ref, successor_work in successor_by_group.items():
            base_work = base_by_group[group_ref]
            self._validate_work_scope(base_work, successor_work)
        if successor.max_parallel_groups > base.max_parallel_groups:
            raise ReviewablePlanAdapterError("attachment plan successor widens parallelism")
        if (
            successor.total_model_requests > base.total_model_requests
            or successor.total_model_tokens > base.total_model_tokens
            or successor.total_cost_micro_usd > base.total_cost_micro_usd
        ):
            raise ReviewablePlanAdapterError("attachment plan successor widens model budget")

    def safe_presentation(
        self,
        plan: AttachmentGenerationPlanV2,
    ) -> PlanReviewPresentationV2:
        return PlanReviewPresentationV2.create(
            presentation_id=(
                f"plan-review-presentation://attachment/{_slug(plan.plan_id)}/{plan.plan_version}"
            ),
            plan_ref=plan.to_ref(),
            plan_kind=self.plan_kind,
            title="Review the attachment generation plan",
            summary_lines=(
                f"{len(plan.works)} isolated artifact groups.",
                f"Parallel group limit is {plan.max_parallel_groups}.",
            ),
            editable_paths=("max_parallel_groups", "works"),
            warning_codes=("PRODUCTION_RELEASE_BLOCKED",),
            audit=plan.audit,
        )

    def invalidated_refs(
        self,
        base: AttachmentGenerationPlanV2,
        successor: AttachmentGenerationPlanV2,
    ) -> tuple[ObjectRef, ...]:
        successor_by_group = {work.artifact_group_ref: work for work in successor.works}
        invalidated = tuple(
            work.artifact_group_ref
            for work in base.works
            if successor_by_group.get(work.artifact_group_ref) != work
        )
        return tuple(sorted(invalidated, key=_ref_key))

    @staticmethod
    def _validate_work_scope(
        base: AttachmentMockWorkV2,
        successor: AttachmentMockWorkV2,
    ) -> None:
        if (
            successor.artifact_ids != base.artifact_ids
            or successor.agent_role != base.agent_role
            or successor.workspace_policy_ref != base.workspace_policy_ref
            or successor.acceptance_check_refs != base.acceptance_check_refs
        ):
            raise ReviewablePlanAdapterError("attachment work successor changes owned artifact scope")
        for successor_values, base_values, label in (
            (
                successor.input_object_types,
                base.input_object_types,
                "input schema",
            ),
            (
                successor.output_object_types,
                base.output_object_types,
                "output schema",
            ),
            (
                successor.required_capability_ids,
                base.required_capability_ids,
                "capability",
            ),
            (
                successor.allowed_tool_ids,
                base.allowed_tool_ids,
                "tool",
            ),
            (
                successor.data_purposes,
                base.data_purposes,
                "data purpose",
            ),
            (
                successor.data_classifications,
                base.data_classifications,
                "data classification",
            ),
        ):
            if not set(successor_values).issubset(base_values):
                raise ReviewablePlanAdapterError(f"attachment work successor widens {label} scope")
        if (
            successor.max_attempts > base.max_attempts
            or successor.max_model_requests > base.max_model_requests
            or successor.max_model_tokens > base.max_model_tokens
            or successor.max_cost_micro_usd > base.max_cost_micro_usd
        ):
            raise ReviewablePlanAdapterError("attachment work successor widens model budget")


class CriteriaRubricPlanAdapter:
    plan_kind = PlanKindV2.CRITERIA_RUBRIC
    object_type = "criteria-rubric-plan"
    compiled_object_type = "compiled-criteria-rubric-plan"

    def __init__(
        self,
        compiler: CriteriaRubricPlanCompiler | None,
    ) -> None:
        self.compiler = compiler

    def parse(
        self,
        payload: bytes,
    ) -> CriteriaRubricPlanV2:
        try:
            return CriteriaRubricPlanV2.model_validate_json(payload)
        except ValidationError as exc:
            raise ReviewablePlanAdapterError("criteria/rubric plan material is invalid") from exc

    def parse_compiled(
        self,
        payload: bytes,
    ) -> CompiledCriteriaRubricPlanV2:
        try:
            return CompiledCriteriaRubricPlanV2.model_validate_json(payload)
        except ValidationError as exc:
            raise ReviewablePlanAdapterError("compiled criteria/rubric plan material is invalid") from exc

    def compile(
        self,
        plan: CriteriaRubricPlanV2,
        *,
        policy: FactoryRunPolicyV2,
        audit: ContractAudit,
    ) -> CompiledCriteriaRubricPlanV2:
        if self.compiler is None:
            raise ReviewablePlanAdapterError("criteria/rubric plan edit requires a deterministic compiler")
        try:
            return self.compiler.compile(
                plan=plan,
                policy=policy,
                audit=audit,
            )
        except CriteriaRubricPlanCompilerError as exc:
            raise ReviewablePlanAdapterError("criteria/rubric plan failed deterministic compilation") from exc

    def validate_successor(
        self,
        base: CriteriaRubricPlanV2,
        successor: CriteriaRubricPlanV2,
    ) -> None:
        if (
            successor.plan_id != base.plan_id
            or successor.plan_version != base.plan_version + 1
            or successor.predecessor_plan_ref != base.to_ref()
        ):
            raise ReviewablePlanAdapterError("criteria/rubric plan successor is not contiguous")
        for field_name in (
            "task_draft_ref",
            "attachment_quality_ref",
            "solvability_ref",
            "allowed_prompt_requirement_ids",
            "required_prompt_requirement_ids",
            "allowed_attachment_dependency_ids",
            "required_attachment_dependency_ids",
            "allowed_task_tool_ids",
            "required_task_tool_ids",
            "tool_catalog_ref",
            "agent_role",
            "required_capability_ids",
            "specialist_tool_ids",
            "data_purpose",
            "data_classifications",
            "prompt_template_ref",
            "model_policy_ref",
        ):
            if getattr(successor, field_name) != getattr(base, field_name):
                raise ReviewablePlanAdapterError(
                    "criteria/rubric plan successor widens source or Agent scope"
                )
        for successor_values, base_values, label in (
            (
                successor.allowed_evaluator_binding_ids,
                base.allowed_evaluator_binding_ids,
                "evaluator binding",
            ),
            (
                successor.allowed_reference_modes,
                base.allowed_reference_modes,
                "reference mode",
            ),
            (
                successor.acceptance_check_refs,
                base.acceptance_check_refs,
                "validator",
            ),
        ):
            if not set(successor_values).issubset(base_values):
                raise ReviewablePlanAdapterError(f"criteria/rubric plan successor widens {label} scope")
        if (
            successor.max_attempts > base.max_attempts
            or successor.max_model_requests > base.max_model_requests
            or successor.max_model_tokens > base.max_model_tokens
            or successor.max_cost_micro_usd > base.max_cost_micro_usd
        ):
            raise ReviewablePlanAdapterError("criteria/rubric plan successor widens execution budget")

    def safe_presentation(
        self,
        plan: CriteriaRubricPlanV2,
    ) -> PlanReviewPresentationV2:
        return PlanReviewPresentationV2.create(
            presentation_id=(
                f"plan-review-presentation://criteria-rubric/{_slug(plan.plan_id)}/{plan.plan_version}"
            ),
            plan_ref=plan.to_ref(),
            plan_kind=self.plan_kind,
            title="Review the criteria and rubric plan",
            summary_lines=(
                f"{len(plan.criterion_goals)} criterion goals.",
                (f"Reference mode is {plan.selected_reference_mode.value}."),
                (f"{plan.max_model_requests} maximum model requests."),
            ),
            editable_paths=(
                "acceptance_check_refs",
                "allowed_evaluator_binding_ids",
                "allowed_reference_modes",
                "criterion_goals",
                "max_attempts",
                "max_cost_micro_usd",
                "max_model_requests",
                "max_model_tokens",
                "selected_reference_mode",
            ),
            warning_codes=("PRODUCTION_RELEASE_BLOCKED",),
            audit=plan.audit,
        )

    def invalidated_refs(
        self,
        base: CriteriaRubricPlanV2,
        successor: CriteriaRubricPlanV2,
    ) -> tuple[ObjectRef, ...]:
        del successor
        return (base.to_ref(),)


class GradingDesignPlanAdapter:
    plan_kind = PlanKindV2.GRADING_DESIGN
    object_type = "grading-design-plan"
    compiled_object_type = "compiled-grading-design-plan"

    def __init__(
        self,
        compiler: GradingDesignPlanCompiler | None,
    ) -> None:
        self.compiler = compiler

    def parse(
        self,
        payload: bytes,
    ) -> GradingDesignPlanV2:
        try:
            return GradingDesignPlanV2.model_validate_json(payload)
        except ValidationError as exc:
            raise ReviewablePlanAdapterError("grading design plan material is invalid") from exc

    def parse_compiled(
        self,
        payload: bytes,
    ) -> CompiledGradingDesignPlanV2:
        try:
            return CompiledGradingDesignPlanV2.model_validate_json(payload)
        except ValidationError as exc:
            raise ReviewablePlanAdapterError("compiled grading design plan material is invalid") from exc

    def compile(
        self,
        plan: GradingDesignPlanV2,
        *,
        policy: FactoryRunPolicyV2,
        audit: ContractAudit,
    ) -> CompiledGradingDesignPlanV2:
        if self.compiler is None:
            raise ReviewablePlanAdapterError("grading design edit requires a deterministic compiler")
        try:
            return self.compiler.compile(
                plan=plan,
                policy=policy,
                audit=audit,
            )
        except GradingDesignPlanCompilerError as exc:
            raise ReviewablePlanAdapterError("grading design plan failed deterministic compilation") from exc

    def validate_successor(
        self,
        base: GradingDesignPlanV2,
        successor: GradingDesignPlanV2,
    ) -> None:
        if (
            successor.plan_id != base.plan_id
            or successor.plan_version != base.plan_version + 1
            or successor.predecessor_plan_ref != base.to_ref()
        ):
            raise ReviewablePlanAdapterError("grading design successor is not contiguous")
        for field_name in (
            "criteria_rubric_result_ref",
            "rubric_set_ref",
            "evaluator_spec_ref",
            "reference_policy_ref",
            "tool_policy_ref",
            "generator_model_profile_ref",
            "judge_input_schema_ref",
            "judge_output_schema_ref",
            "required_output_fields",
            "judge_prompt_template_ref",
            "agent_role",
            "required_capability_ids",
            "specialist_tool_ids",
            "data_purpose",
            "data_classifications",
            "model_policy_ref",
        ):
            if getattr(successor, field_name) != getattr(
                base,
                field_name,
            ):
                raise ReviewablePlanAdapterError("grading design successor widens source or Agent scope")
        for successor_values, base_values, label in (
            (
                successor.allowed_judge_model_profile_refs,
                base.allowed_judge_model_profile_refs,
                "judge model",
            ),
            (
                successor.acceptance_check_refs,
                base.acceptance_check_refs,
                "validator",
            ),
        ):
            if not set(successor_values).issubset(base_values):
                raise ReviewablePlanAdapterError(f"grading design successor widens {label} scope")
        if (
            successor.max_attempts > base.max_attempts
            or successor.max_model_requests > base.max_model_requests
            or successor.max_model_tokens > base.max_model_tokens
            or successor.max_cost_micro_usd > base.max_cost_micro_usd
        ):
            raise ReviewablePlanAdapterError("grading design successor widens execution budget")

    def safe_presentation(
        self,
        plan: GradingDesignPlanV2,
    ) -> PlanReviewPresentationV2:
        return PlanReviewPresentationV2.create(
            presentation_id=(
                f"plan-review-presentation://grading-design/{_slug(plan.plan_id)}/{plan.plan_version}"
            ),
            plan_ref=plan.to_ref(),
            plan_kind=self.plan_kind,
            title="Review the grading design plan",
            summary_lines=(
                f"{len(plan.judge_tasks)} judge tasks.",
                (f"Aggregation mode is {plan.aggregation_mode.value}."),
                (f"Passing score is {plan.passing_score_basis_points} basis points."),
            ),
            editable_paths=(
                "acceptance_check_refs",
                "aggregation_mode",
                "allowed_judge_model_profile_refs",
                "escalate_on_reference_unavailable",
                "judge_tasks",
                "max_attempts",
                "max_cost_micro_usd",
                "max_model_requests",
                "max_model_tokens",
                "minimum_confidence_basis_points",
                "passing_score_basis_points",
            ),
            warning_codes=("PRODUCTION_RELEASE_BLOCKED",),
            audit=plan.audit,
        )

    def invalidated_refs(
        self,
        base: GradingDesignPlanV2,
        successor: GradingDesignPlanV2,
    ) -> tuple[ObjectRef, ...]:
        del successor
        return (base.to_ref(),)


class DatasetDeliveryPlanAdapter:
    plan_kind = PlanKindV2.FINAL_DELIVERY
    object_type = "dataset-delivery-plan"
    compiled_object_type = "compiled-dataset-delivery-plan"

    def __init__(
        self,
        compiler: DatasetDeliveryPlanCompiler | None,
    ) -> None:
        self.compiler = compiler

    def parse(self, payload: bytes) -> DatasetDeliveryPlanV2:
        try:
            return DatasetDeliveryPlanV2.model_validate_json(payload)
        except ValidationError as exc:
            raise ReviewablePlanAdapterError("dataset delivery plan material is invalid") from exc

    def parse_compiled(
        self,
        payload: bytes,
    ) -> CompiledDatasetDeliveryPlanV2:
        try:
            return CompiledDatasetDeliveryPlanV2.model_validate_json(payload)
        except ValidationError as exc:
            raise ReviewablePlanAdapterError("compiled dataset delivery plan is invalid") from exc

    def compile(
        self,
        plan: DatasetDeliveryPlanV2,
        *,
        policy: FactoryRunPolicyV2,
        audit: ContractAudit,
    ) -> CompiledDatasetDeliveryPlanV2:
        if self.compiler is None:
            raise ReviewablePlanAdapterError("delivery plan edit requires a deterministic compiler")
        try:
            return self.compiler.compile(
                plan=plan,
                policy=policy,
                audit=audit,
            )
        except DatasetDeliveryPlanCompilerError as exc:
            raise ReviewablePlanAdapterError("delivery plan failed deterministic compilation") from exc

    def validate_successor(
        self,
        base: DatasetDeliveryPlanV2,
        successor: DatasetDeliveryPlanV2,
    ) -> None:
        if (
            successor.plan_id != base.plan_id
            or successor.plan_version != base.plan_version + 1
            or successor.predecessor_plan_ref != base.to_ref()
        ):
            raise ReviewablePlanAdapterError("delivery plan successor is not contiguous")
        for field_name in (
            "aggregate_result_ref",
            "candidate_item_refs",
            "candidate_projection_refs",
            "rejected_binding_refs",
            "blocked_binding_refs",
            "output_target_ref",
            "production_release_allowed",
        ):
            if getattr(successor, field_name) != getattr(
                base,
                field_name,
            ):
                raise ReviewablePlanAdapterError("delivery plan successor changes fixed authority")
        if successor.max_files > base.max_files or successor.max_total_bytes > base.max_total_bytes:
            raise ReviewablePlanAdapterError("delivery plan successor widens output budget")

    def safe_presentation(
        self,
        plan: DatasetDeliveryPlanV2,
    ) -> PlanReviewPresentationV2:
        return PlanReviewPresentationV2.create(
            presentation_id=(
                f"plan-review-presentation://final-delivery/{_slug(plan.plan_id)}/{plan.plan_version}"
            ),
            plan_ref=plan.to_ref(),
            plan_kind=self.plan_kind,
            title="Review the candidate dataset delivery plan",
            summary_lines=(
                (f"{len(plan.candidate_item_refs)} candidate items."),
                (
                    f"{len(plan.rejected_binding_refs)} rejected "
                    "and "
                    f"{len(plan.blocked_binding_refs)} blocked "
                    "items."
                ),
                (f"Output limit is {plan.max_files} files and {plan.max_total_bytes} bytes."),
            ),
            editable_paths=(
                "max_files",
                "max_total_bytes",
            ),
            warning_codes=("PRODUCTION_RELEASE_BLOCKED",),
            audit=plan.audit,
        )

    def invalidated_refs(
        self,
        base: DatasetDeliveryPlanV2,
        successor: DatasetDeliveryPlanV2,
    ) -> tuple[ObjectRef, ...]:
        del successor
        return (base.to_ref(),)


def _ref_key(value: ObjectRef) -> tuple[str, str, str, str]:
    return (
        value.object_type,
        value.object_id,
        value.object_version,
        value.object_sha256,
    )


def _slug(value: str) -> str:
    return value.rsplit("://", 1)[-1].replace("/", "-")


def erased_adapter[
    PlanT: ContractModelV2,
    CompiledT: ContractModelV2,
](
    adapter: ReviewablePlanAdapter[PlanT, CompiledT],
) -> _ErasedPlanAdapter:
    return cast(_ErasedPlanAdapter, adapter)


__all__ = [
    "AttachmentGenerationPlanAdapter",
    "CompiledReviewablePlanV2",
    "CriteriaRubricPlanAdapter",
    "DatasetDeliveryPlanAdapter",
    "GlobalBuildPlanAdapter",
    "GradingDesignPlanAdapter",
    "ReviewablePlanAdapter",
    "ReviewablePlanAdapterError",
    "ReviewablePlanAdapterRegistry",
    "ReviewablePlanV2",
    "erased_adapter",
]
