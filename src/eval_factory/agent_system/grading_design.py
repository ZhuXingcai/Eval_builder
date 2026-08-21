from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, model_validator

from eval_factory.contracts.agent_system_v2 import (
    CriteriaRubricOutcomeV2,
    CriteriaRubricResultV2,
    GradingDesignOutcomeV2,
    GradingDesignPlanV2,
    JudgeDesignSpecV2,
    JudgeDesignValidationOutcomeV2,
    JudgeDesignValidationV2,
)
from eval_factory.contracts.ai_gateway_v2 import (
    GatewayInvocationResultV2,
    ModelRouteDecisionV2,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.core_v2 import (
    ContractModelV2,
    canonical_value_v2,
)
from eval_factory.contracts.task_v2 import (
    EvaluatorReferenceGrantV2,
    EvaluatorSpecV2,
    ReferencePolicyV2,
    RubricSetV2,
    ToolPolicyV2,
    evaluator_reference_grant_ref,
    evaluator_spec_ref,
    reference_policy_ref,
    rubric_set_ref,
    tool_policy_ref,
)
from eval_factory.task_authoring import (
    REFERENCE_ACCESS_POLICY_VERSION,
    EvaluatorAccessPrincipalType,
    EvaluatorReferenceAccessGate,
    EvaluatorReferenceAccessOutcome,
    EvaluatorReferenceAccessRequest,
    EvaluatorReferenceAccessResult,
    ProducerTaskViewPolicyError,
    VerifiedModelDomainAuthorization,
    evaluator_reference_access_request_sha256,
)
from eval_factory.task_authoring.producer_view import (
    validate_evaluator_spec_identity,
    validate_reference_policy_identity,
    validate_rubric_set_identity,
    validate_tool_policy_identity,
)


class GradingDesignPolicyError(RuntimeError):
    pass


class JudgeDesignProposalOutcomeV1(StrEnum):
    PROPOSED = "PROPOSED"
    ABSTAIN = "ABSTAIN"
    ESCALATE_REFERENCE_REQUIRED = "ESCALATE_REFERENCE_REQUIRED"


class JudgeInstructionSelectionV1(ContractModelV2):
    schema_version: Literal["eval-factory/private-judge-instruction-selection/v1"] = (
        "eval-factory/private-judge-instruction-selection/v1"
    )
    task_key: str = Field(min_length=3, max_length=256)
    criterion_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=10_000,
    )
    evaluator_binding_id: str = Field(
        min_length=3,
        max_length=512,
    )
    instruction: str = Field(min_length=1, max_length=16_000)

    @model_validator(mode="after")
    def validate_selection(self) -> Self:
        if tuple(sorted(set(self.criterion_ids))) != self.criterion_ids:
            raise ValueError("judge instruction criterion IDs must be sorted and unique")
        return self


class JudgeDesignProposalV1(ContractModelV2):
    schema_version: Literal["eval-factory/private-judge-design-proposal/v1"] = (
        "eval-factory/private-judge-design-proposal/v1"
    )
    outcome: JudgeDesignProposalOutcomeV1
    instructions: tuple[JudgeInstructionSelectionV1, ...] = Field(default=(), max_length=10_000)
    output_fields: tuple[str, ...] = Field(
        default=(),
        max_length=16,
    )
    confidence_basis_points: int = Field(ge=0, le=10_000)
    reason_codes: tuple[str, ...] = Field(
        default=(),
        max_length=256,
    )

    @model_validator(mode="after")
    def validate_proposal(self) -> Self:
        task_keys = tuple(value.task_key for value in self.instructions)
        if (
            tuple(sorted(set(task_keys))) != task_keys
            or tuple(sorted(set(self.output_fields))) != self.output_fields
            or tuple(sorted(set(self.reason_codes))) != self.reason_codes
        ):
            raise ValueError("judge proposal inventories must be sorted and unique")
        if self.outcome is JudgeDesignProposalOutcomeV1.PROPOSED:
            if not self.instructions or not self.output_fields or self.reason_codes:
                raise ValueError("proposed judge design requires instructions and output fields only")
        elif self.instructions or self.output_fields or not self.reason_codes:
            raise ValueError("non-proposed judge design requires reasons only")
        return self


@dataclass(frozen=True, slots=True)
class JudgeDesignCandidate:
    plan: GradingDesignPlanV2
    criteria_result: CriteriaRubricResultV2
    rubric_set: RubricSetV2
    evaluator_spec: EvaluatorSpecV2
    reference_policy: ReferencePolicyV2
    tool_policy: ToolPolicyV2
    proposal: JudgeDesignProposalV1
    prompt_rendering_ref: ObjectRef
    route: ModelRouteDecisionV2
    gateway_result: GatewayInvocationResultV2


@dataclass(frozen=True, slots=True)
class JudgeDesignCompilation:
    design_spec: JudgeDesignSpecV2 | None
    validation: JudgeDesignValidationV2
    outcome: GradingDesignOutcomeV2
    reason_codes: tuple[str, ...]
    reference_grants: tuple[EvaluatorReferenceGrantV2, ...]


class JudgeDesignCompiler:
    def compile(
        self,
        *,
        plan: GradingDesignPlanV2,
        criteria_result: CriteriaRubricResultV2,
        criteria_route: ModelRouteDecisionV2,
        rubric_set: RubricSetV2,
        evaluator_spec: EvaluatorSpecV2,
        reference_policy: ReferencePolicyV2,
        tool_policy: ToolPolicyV2,
        proposal: JudgeDesignProposalV1,
        prompt_rendering_ref: ObjectRef,
        route: ModelRouteDecisionV2,
        gateway_result: GatewayInvocationResultV2,
    ) -> JudgeDesignCandidate:
        self.validate_sources(
            plan=plan,
            criteria_result=criteria_result,
            criteria_route=criteria_route,
            rubric_set=rubric_set,
            evaluator_spec=evaluator_spec,
            reference_policy=reference_policy,
            tool_policy=tool_policy,
        )
        if route.selected_model_profile_ref == plan.generator_model_profile_ref:
            raise GradingDesignPolicyError("judge route collides with generator model")
        if (
            route.selected_model_profile_ref not in plan.allowed_judge_model_profile_refs
            or gateway_result.route_decision_ref != route.to_ref()
        ):
            raise GradingDesignPolicyError("judge route differs from reviewed authority")
        if proposal.outcome is not (JudgeDesignProposalOutcomeV1.PROPOSED):
            raise GradingDesignPolicyError("non-proposed output cannot compile a judge design")
        mappings = {value.task_key: value for value in plan.judge_tasks}
        instructions = {value.task_key: value for value in proposal.instructions}
        if set(mappings) != set(instructions):
            raise GradingDesignPolicyError("judge proposal does not cover reviewed tasks")
        for task_key, mapping in mappings.items():
            instruction = instructions[task_key]
            if (
                instruction.criterion_ids != mapping.criterion_ids
                or instruction.evaluator_binding_id != mapping.evaluator_binding_id
            ):
                raise GradingDesignPolicyError("judge proposal widens reviewed criterion authority")
        if proposal.output_fields != plan.required_output_fields:
            raise GradingDesignPolicyError("judge proposal output schema mapping is incomplete")
        return JudgeDesignCandidate(
            plan=plan,
            criteria_result=criteria_result,
            rubric_set=rubric_set,
            evaluator_spec=evaluator_spec,
            reference_policy=reference_policy,
            tool_policy=tool_policy,
            proposal=proposal,
            prompt_rendering_ref=prompt_rendering_ref,
            route=route,
            gateway_result=gateway_result,
        )

    @staticmethod
    def validate_sources(
        *,
        plan: GradingDesignPlanV2,
        criteria_result: CriteriaRubricResultV2,
        criteria_route: ModelRouteDecisionV2,
        rubric_set: RubricSetV2,
        evaluator_spec: EvaluatorSpecV2,
        reference_policy: ReferencePolicyV2,
        tool_policy: ToolPolicyV2,
    ) -> None:
        if (
            criteria_result.outcome is not CriteriaRubricOutcomeV2.SUCCEEDED
            or plan.criteria_rubric_result_ref != criteria_result.to_ref()
            or criteria_result.route_decision_ref != criteria_route.to_ref()
            or criteria_route.selected_model_profile_ref != plan.generator_model_profile_ref
            or criteria_result.rubric_set_ref != rubric_set_ref(rubric_set)
            or criteria_result.evaluator_spec_ref != evaluator_spec_ref(evaluator_spec)
            or criteria_result.reference_policy_ref != reference_policy_ref(reference_policy)
            or criteria_result.tool_policy_ref != tool_policy_ref(tool_policy)
            or plan.rubric_set_ref != rubric_set_ref(rubric_set)
            or plan.evaluator_spec_ref != evaluator_spec_ref(evaluator_spec)
            or plan.reference_policy_ref != reference_policy_ref(reference_policy)
            or plan.tool_policy_ref != tool_policy_ref(tool_policy)
        ):
            raise GradingDesignPolicyError("grading design source authority is stale")
        try:
            validate_rubric_set_identity(rubric_set)
            validate_evaluator_spec_identity(evaluator_spec)
            validate_reference_policy_identity(
                reference_policy,
                evaluator_spec,
            )
            validate_tool_policy_identity(tool_policy)
        except ProducerTaskViewPolicyError as exc:
            raise GradingDesignPolicyError("grading design R4 authority is stale") from exc
        if (
            evaluator_spec.rubric_set_ref != rubric_set_ref(rubric_set)
            or reference_policy.evaluator_spec_ref != evaluator_spec_ref(evaluator_spec)
            or tool_policy.rubric_set_ref != rubric_set_ref(rubric_set)
            or tool_policy.evaluator_spec_ref != evaluator_spec_ref(evaluator_spec)
        ):
            raise GradingDesignPolicyError("grading design R4 chain is inconsistent")
        criteria = {criterion.criterion_id: criterion for criterion in rubric_set.criteria}
        planned = {criterion_id for task in plan.judge_tasks for criterion_id in task.criterion_ids}
        if planned != set(criteria):
            raise GradingDesignPolicyError("grading plan does not exactly partition criteria")
        binding_criteria = {
            binding.evaluator_binding_id: set(binding.criterion_ids) for binding in evaluator_spec.bindings
        }
        for task in plan.judge_tasks:
            if set(task.criterion_ids) - binding_criteria.get(
                task.evaluator_binding_id,
                set(),
            ) or any(
                criteria[criterion_id].evaluator_binding != task.evaluator_binding_id
                for criterion_id in task.criterion_ids
            ):
                raise GradingDesignPolicyError("grading task differs from evaluator binding authority")


class JudgeDesignValidator:
    def validate(
        self,
        *,
        candidate: JudgeDesignCandidate,
        model_authorizations: tuple[VerifiedModelDomainAuthorization, ...],
        evaluated_at: datetime,
        audit: ContractAudit,
    ) -> JudgeDesignCompilation:
        if candidate.proposal.confidence_basis_points < candidate.plan.minimum_confidence_basis_points:
            return self._non_valid(
                candidate=candidate,
                access_results=(),
                outcome=(JudgeDesignValidationOutcomeV2.ABSTAIN_UNJUDGEABLE),
                reason_codes=("JUDGE_CONFIDENCE_INSUFFICIENT",),
                audit=audit,
            )
        access_results = []
        grants = []
        authorizations = {value.model_profile_ref: value for value in model_authorizations}
        bindings = {value.evaluator_binding_id: value for value in candidate.evaluator_spec.bindings}
        for task in candidate.plan.judge_tasks:
            binding = bindings[task.evaluator_binding_id]
            request = _access_request(
                binding_id=task.evaluator_binding_id,
                principal_id=(binding.evaluator_principal_id or "principal://grading-design/human-only"),
                model_profile_ref=binding.model_profile_ref,
                audit=audit,
            )
            access = EvaluatorReferenceAccessGate().authorize(
                evaluator_spec=candidate.evaluator_spec,
                reference_policy=candidate.reference_policy,
                request=request,
                model_authorization=(
                    authorizations.get(binding.model_profile_ref)
                    if binding.model_profile_ref is not None
                    else None
                ),
                evaluated_at=evaluated_at,
                audit=audit,
            )
            access_results.append(access)
            if access.outcome is (EvaluatorReferenceAccessOutcome.GRANTED):
                assert access.grant is not None
                grants.append(access.grant)
            elif access.outcome is (EvaluatorReferenceAccessOutcome.NO_REFERENCE_REQUIRED):
                continue
            elif access.outcome is (EvaluatorReferenceAccessOutcome.HUMAN_ONLY):
                return self._non_valid(
                    candidate=candidate,
                    access_results=tuple(access_results),
                    outcome=(JudgeDesignValidationOutcomeV2.ABSTAIN_UNJUDGEABLE),
                    reason_codes=("HUMAN_ONLY_EVALUATOR",),
                    audit=audit,
                )
            elif candidate.plan.escalate_on_reference_unavailable:
                return self._non_valid(
                    candidate=candidate,
                    access_results=tuple(access_results),
                    outcome=(JudgeDesignValidationOutcomeV2.ESCALATE_REFERENCE_REQUIRED),
                    reason_codes=("REFERENCE_ACCESS_REQUIRED",),
                    audit=audit,
                )
            else:
                return self._non_valid(
                    candidate=candidate,
                    access_results=tuple(access_results),
                    outcome=(JudgeDesignValidationOutcomeV2.BLOCKED_POLICY),
                    reason_codes=("REFERENCE_ACCESS_DENIED",),
                    audit=audit,
                )
        access_refs = tuple(
            sorted(
                (_access_result_ref(value) for value in access_results),
                key=_ref_key,
            )
        )
        grant_refs = tuple(
            sorted(
                (evaluator_reference_grant_ref(value) for value in grants),
                key=_ref_key,
            )
        )
        inventory_sha256 = _inventory_sha256(
            candidate.prompt_rendering_ref,
            candidate.gateway_result.output_ref,
            access_refs,
            grant_refs,
        )
        spec = JudgeDesignSpecV2.create(
            design_spec_id=(f"judge-design-spec://{candidate.plan.object_sha256}"),
            plan_ref=candidate.plan.to_ref(),
            criteria_rubric_result_ref=(candidate.criteria_result.to_ref()),
            rubric_set_ref=rubric_set_ref(candidate.rubric_set),
            evaluator_spec_ref=evaluator_spec_ref(candidate.evaluator_spec),
            reference_policy_ref=reference_policy_ref(candidate.reference_policy),
            tool_policy_ref=tool_policy_ref(candidate.tool_policy),
            prompt_template_ref=(candidate.plan.judge_prompt_template_ref),
            prompt_rendering_ref=candidate.prompt_rendering_ref,
            route_decision_ref=candidate.route.to_ref(),
            gateway_receipt_ref=(candidate.gateway_result.receipt_ref),
            gateway_invocation_result_ref=(candidate.gateway_result.to_ref()),
            proposal_ref=(candidate.gateway_result.output_ref or _missing_proposal_ref()),
            selected_judge_model_profile_ref=(candidate.route.selected_model_profile_ref),
            generator_model_profile_ref=(candidate.plan.generator_model_profile_ref),
            judge_tasks=candidate.plan.judge_tasks,
            aggregation_mode=candidate.plan.aggregation_mode,
            passing_score_basis_points=(candidate.plan.passing_score_basis_points),
            minimum_confidence_basis_points=(candidate.plan.minimum_confidence_basis_points),
            escalate_on_reference_unavailable=(candidate.plan.escalate_on_reference_unavailable),
            judge_input_schema_ref=(candidate.plan.judge_input_schema_ref),
            judge_output_schema_ref=(candidate.plan.judge_output_schema_ref),
            required_output_fields=(candidate.plan.required_output_fields),
            reference_grant_refs=grant_refs,
            private_material_inventory_sha256=(inventory_sha256),
            audit=audit,
        )
        criterion_ids = tuple(sorted(criterion.criterion_id for criterion in candidate.rubric_set.criteria))
        validation = JudgeDesignValidationV2.create(
            validation_id=(f"judge-design-validation://{spec.object_sha256}"),
            plan_ref=candidate.plan.to_ref(),
            design_spec_ref=spec.to_ref(),
            criterion_ids=criterion_ids,
            validated_output_fields=(candidate.plan.required_output_fields),
            reference_access_result_refs=access_refs,
            outcome=JudgeDesignValidationOutcomeV2.VALID,
            reason_codes=(),
            audit=audit,
        )
        return JudgeDesignCompilation(
            design_spec=spec,
            validation=validation,
            outcome=GradingDesignOutcomeV2.SUCCEEDED,
            reason_codes=(),
            reference_grants=tuple(grants),
        )

    @staticmethod
    def _non_valid(
        *,
        candidate: JudgeDesignCandidate,
        access_results: tuple[EvaluatorReferenceAccessResult, ...],
        outcome: JudgeDesignValidationOutcomeV2,
        reason_codes: tuple[str, ...],
        audit: ContractAudit,
    ) -> JudgeDesignCompilation:
        validation = JudgeDesignValidationV2.create(
            validation_id=(f"judge-design-validation://{candidate.plan.object_sha256}/{outcome.value}"),
            plan_ref=candidate.plan.to_ref(),
            design_spec_ref=None,
            criterion_ids=(),
            validated_output_fields=(),
            reference_access_result_refs=tuple(
                sorted(
                    (_access_result_ref(value) for value in access_results),
                    key=_ref_key,
                )
            ),
            outcome=outcome,
            reason_codes=reason_codes,
            audit=audit,
        )
        result_outcome = {
            JudgeDesignValidationOutcomeV2.ABSTAIN_UNJUDGEABLE: (GradingDesignOutcomeV2.ABSTAINED),
            JudgeDesignValidationOutcomeV2.ESCALATE_REFERENCE_REQUIRED: (GradingDesignOutcomeV2.ESCALATED),
            JudgeDesignValidationOutcomeV2.BLOCKED_CAPABILITY: (GradingDesignOutcomeV2.BLOCKED_CAPABILITY),
            JudgeDesignValidationOutcomeV2.BLOCKED_POLICY: (GradingDesignOutcomeV2.BLOCKED_POLICY),
            JudgeDesignValidationOutcomeV2.INVALID_DESIGN: (GradingDesignOutcomeV2.INVALID_DESIGN),
        }[outcome]
        return JudgeDesignCompilation(
            design_spec=None,
            validation=validation,
            outcome=result_outcome,
            reason_codes=reason_codes,
            reference_grants=(),
        )


def _access_request(
    *,
    binding_id: str,
    principal_id: str,
    model_profile_ref: ObjectRef | None,
    audit: ContractAudit,
) -> EvaluatorReferenceAccessRequest:
    value = EvaluatorReferenceAccessRequest(
        request_id="evaluator-reference-access-request://pending",
        principal_type=EvaluatorAccessPrincipalType.EVALUATOR,
        principal_id=principal_id,
        purpose="EVALUATION",
        evaluator_binding_id=binding_id,
        model_profile_ref=model_profile_ref,
        policy_version=REFERENCE_ACCESS_POLICY_VERSION,
        request_sha256="0" * 64,
        audit=audit,
    )
    digest = evaluator_reference_access_request_sha256(value)
    return value.model_copy(
        update={
            "request_id": (f"evaluator-reference-access-request://sha256/{digest}"),
            "request_sha256": digest,
        }
    )


def _access_result_ref(
    value: EvaluatorReferenceAccessResult,
) -> ObjectRef:
    payload = value.model_dump(mode="python")
    return ObjectRef(
        object_type="evaluator-reference-access-result",
        object_id=str(payload["access_result_id"]),
        object_version="r4-06",
        object_sha256=str(payload["result_sha256"]),
    )


def _inventory_sha256(
    prompt_rendering_ref: ObjectRef,
    proposal_ref: ObjectRef | None,
    access_refs: tuple[ObjectRef, ...],
    grant_refs: tuple[ObjectRef, ...],
) -> str:
    encoded = json.dumps(
        canonical_value_v2(
            {
                "prompt_rendering_ref": prompt_rendering_ref,
                "proposal_ref": proposal_ref,
                "access_refs": access_refs,
                "grant_refs": grant_refs,
            }
        ),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _missing_proposal_ref() -> ObjectRef:
    raise GradingDesignPolicyError("successful Gateway result lacks proposal ref")


def _ref_key(
    value: ObjectRef,
) -> tuple[str, str, str, str]:
    return (
        value.object_type,
        value.object_id,
        value.object_version,
        value.object_sha256,
    )


__all__ = [
    "GradingDesignPolicyError",
    "JudgeDesignCandidate",
    "JudgeDesignCompilation",
    "JudgeDesignCompiler",
    "JudgeDesignProposalOutcomeV1",
    "JudgeDesignProposalV1",
    "JudgeDesignValidator",
    "JudgeInstructionSelectionV1",
]
