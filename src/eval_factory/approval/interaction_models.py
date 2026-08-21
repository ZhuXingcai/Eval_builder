from __future__ import annotations

import hashlib
import json
from typing import Literal, Self

from pydantic import model_validator

from eval_factory.approval.decisions import AuthenticatedUserContext
from eval_factory.approval.requests import (
    ApprovalCheckpointSource,
    EnvironmentStrategyApprovalSource,
    FinalDatasetReviewApprovalSource,
    LabelPlanApprovalSource,
    TaskRewriteApprovalSource,
    UserApprovalRequestCompiler,
)
from eval_factory.contracts.approval import (
    ApprovalCheckpoint,
    EnvironmentStrategy,
    LabelPlan,
    UserApprovalPolicy,
)
from eval_factory.contracts.approval_decision_v2 import (
    UserDecisionHandlingPolicyV2,
    validate_user_decision_handling_policy_v2_identity,
)
from eval_factory.contracts.approval_v2 import (
    ApprovalRequestGenerationPolicyV2,
    FinalDatasetReviewPreviewV2,
    UserApprovalRequestCompilationResultV2,
    environment_strategy_ref,
    final_dataset_review_preview_v2_ref,
    label_plan_ref,
    validate_approval_request_generation_policy_v2_identity,
    validate_user_approval_policy_identity,
)
from eval_factory.contracts.checkpoint_interaction_v2 import (
    UserCheckpointPresentationV2,
)
from eval_factory.contracts.core import Identifier, ObjectRef, Sha256
from eval_factory.contracts.core_v2 import (
    ContractModelV2,
    canonical_value_v2,
)
from eval_factory.contracts.labeling_v2 import LabelSpecV2
from eval_factory.contracts.orchestration_v2 import DatasetJobSpecV2
from eval_factory.contracts.task_v2 import (
    TaskRewritePlanPreviewV2,
    TaskRewritePlanVersionV2,
    TaskRewritePreviewSafetyGateV2,
    task_rewrite_plan_preview_ref,
)


class UserCheckpointSourceContextV2(ContractModelV2):
    schema_version: Literal["eval-factory/user-checkpoint-source-context/private-v1"] = (
        "eval-factory/user-checkpoint-source-context/private-v1"
    )
    context_id: Identifier
    job_spec: DatasetJobSpecV2
    approval_policy: UserApprovalPolicy
    generation_policy: ApprovalRequestGenerationPolicyV2
    handling_policy: UserDecisionHandlingPolicyV2
    request_compilation: UserApprovalRequestCompilationResultV2
    requested_by: Identifier
    label_specs: tuple[LabelSpecV2, ...] = ()
    label_plans: tuple[LabelPlan, ...] = ()
    task_rewrite_plan_versions: tuple[TaskRewritePlanVersionV2, ...] = ()
    task_rewrite_preview_safety_gates: tuple[TaskRewritePreviewSafetyGateV2, ...] = ()
    task_rewrite_previews: tuple[TaskRewritePlanPreviewV2, ...] = ()
    environment_strategies: tuple[EnvironmentStrategy, ...] = ()
    final_dataset_review_previews: tuple[FinalDatasetReviewPreviewV2, ...] = ()
    context_sha256: Sha256

    @model_validator(mode="after")
    def validate_context(self) -> Self:
        validate_user_approval_policy_identity(self.approval_policy)
        validate_approval_request_generation_policy_v2_identity(self.generation_policy)
        validate_user_decision_handling_policy_v2_identity(self.handling_policy)
        if (
            self.job_spec.job_id != self.request_compilation.job_id
            or self.job_spec.approval_policy_ref != self.request_compilation.approval_policy_ref
            or self.approval_policy.policy_id != self.job_spec.approval_policy_ref.object_id
        ):
            raise ValueError("checkpoint source context Job/policy binding is stale")
        requests = self.request_compilation.requests
        if requests and {request.requested_by for request in requests} != {self.requested_by}:
            raise ValueError("checkpoint source context requester is stale")
        sources = self.sources()
        UserApprovalRequestCompiler().validate_current(
            self.request_compilation,
            job_spec=self.job_spec,
            approval_policy=self.approval_policy,
            generation_policy=self.generation_policy,
            sources=sources,
            requested_by=self.requested_by,
        )
        _validate_variant_shape(self)
        observed = user_checkpoint_source_context_v2_carried_sha256(self)
        if (
            self.context_sha256 != observed
            or self.context_id != f"user-checkpoint-source-context://sha256/{observed}"
        ) and not (
            self.context_id == "user-checkpoint-source-context://pending" and self.context_sha256 == "0" * 64
        ):
            raise ValueError("checkpoint source context identity is stale")
        return self

    def sources(self) -> tuple[ApprovalCheckpointSource, ...]:
        checkpoint = self.request_compilation.checkpoint
        if checkpoint is ApprovalCheckpoint.LABEL_PLAN:
            return tuple(
                LabelPlanApprovalSource(
                    label_spec=label_spec,
                    label_plan=label_plan,
                )
                for label_spec, label_plan in zip(
                    self.label_specs,
                    self.label_plans,
                    strict=True,
                )
            )
        if checkpoint is ApprovalCheckpoint.TASK_REWRITE_PLAN:
            return tuple(
                TaskRewriteApprovalSource(
                    plan_version=plan_version,
                    preview_safety_gate=gate,
                    preview=preview,
                )
                for plan_version, gate, preview in zip(
                    self.task_rewrite_plan_versions,
                    self.task_rewrite_preview_safety_gates,
                    self.task_rewrite_previews,
                    strict=True,
                )
            )
        if checkpoint is ApprovalCheckpoint.ENVIRONMENT_STRATEGY:
            return tuple(
                EnvironmentStrategyApprovalSource(strategy=value) for value in self.environment_strategies
            )
        return tuple(
            FinalDatasetReviewApprovalSource(preview=value) for value in self.final_dataset_review_previews
        )

    def presentations(
        self,
    ) -> tuple[UserCheckpointPresentationV2, ...]:
        bodies: dict[ObjectRef, object] = {}
        for label_plan in self.label_plans:
            bodies[label_plan_ref(label_plan)] = label_plan
        for rewrite_preview in self.task_rewrite_previews:
            bodies[task_rewrite_plan_preview_ref(rewrite_preview)] = rewrite_preview
        for environment_strategy in self.environment_strategies:
            bodies[environment_strategy_ref(environment_strategy)] = environment_strategy
        for final_preview in self.final_dataset_review_previews:
            bodies[final_dataset_review_preview_v2_ref(final_preview)] = final_preview

        presentations: list[UserCheckpointPresentationV2] = []
        for request in self.request_compilation.requests:
            if len(request.preview_refs) != 1:
                raise ValueError("checkpoint interaction requires one presentation per request")
            body = bodies.get(request.preview_refs[0])
            if body is None:
                raise ValueError("checkpoint presentation body is missing")
            presentations.append(
                UserCheckpointPresentationV2.create(
                    job_id=self.job_spec.job_id,
                    request=request,
                    label_plan=body if isinstance(body, LabelPlan) else None,
                    task_rewrite_preview=(body if isinstance(body, TaskRewritePlanPreviewV2) else None),
                    environment_strategy=(body if isinstance(body, EnvironmentStrategy) else None),
                    final_dataset_review_preview=(
                        body if isinstance(body, FinalDatasetReviewPreviewV2) else None
                    ),
                    audit=request.audit,
                )
            )
        return tuple(
            sorted(
                presentations,
                key=lambda value: _ref_key(value.request_ref),
            )
        )

    @classmethod
    def create(
        cls,
        *,
        job_spec: DatasetJobSpecV2,
        approval_policy: UserApprovalPolicy,
        generation_policy: ApprovalRequestGenerationPolicyV2,
        handling_policy: UserDecisionHandlingPolicyV2,
        request_compilation: UserApprovalRequestCompilationResultV2,
        requested_by: str,
        sources: tuple[ApprovalCheckpointSource, ...],
    ) -> UserCheckpointSourceContextV2:
        label_specs = tuple(
            value.label_spec for value in sources if isinstance(value, LabelPlanApprovalSource)
        )
        label_plans = tuple(
            value.label_plan for value in sources if isinstance(value, LabelPlanApprovalSource)
        )
        task_rewrite_plan_versions = tuple(
            value.plan_version for value in sources if isinstance(value, TaskRewriteApprovalSource)
        )
        task_rewrite_preview_safety_gates = tuple(
            value.preview_safety_gate for value in sources if isinstance(value, TaskRewriteApprovalSource)
        )
        task_rewrite_previews = tuple(
            value.preview for value in sources if isinstance(value, TaskRewriteApprovalSource)
        )
        environment_strategies = tuple(
            value.strategy for value in sources if isinstance(value, EnvironmentStrategyApprovalSource)
        )
        final_dataset_review_previews = tuple(
            value.preview for value in sources if isinstance(value, FinalDatasetReviewApprovalSource)
        )
        value = cls(
            context_id="user-checkpoint-source-context://pending",
            job_spec=job_spec,
            approval_policy=approval_policy,
            generation_policy=generation_policy,
            handling_policy=handling_policy,
            request_compilation=request_compilation,
            requested_by=requested_by,
            label_specs=label_specs,
            label_plans=label_plans,
            task_rewrite_plan_versions=task_rewrite_plan_versions,
            task_rewrite_preview_safety_gates=(task_rewrite_preview_safety_gates),
            task_rewrite_previews=task_rewrite_previews,
            environment_strategies=environment_strategies,
            final_dataset_review_previews=final_dataset_review_previews,
            context_sha256="0" * 64,
        )
        digest = user_checkpoint_source_context_v2_carried_sha256(value)
        return cls.model_validate(
            value.model_copy(
                update={
                    "context_id": (f"user-checkpoint-source-context://sha256/{digest}"),
                    "context_sha256": digest,
                }
            ).model_dump(mode="python")
        )


class TrustedAuthenticatedUserContextV2(ContractModelV2):
    schema_version: Literal["eval-factory/trusted-authenticated-user-context/private-v1"] = (
        "eval-factory/trusted-authenticated-user-context/private-v1"
    )
    authenticated_user: Identifier
    authentication_context_ref: ObjectRef

    @model_validator(mode="after")
    def validate_context(self) -> Self:
        if (
            self.authentication_context_ref.object_type != "authenticated-user-context"
            or self.authentication_context_ref.object_version != "v2"
        ):
            raise ValueError("authentication_context_ref must reference authenticated-user-context v2")
        return self

    def to_domain(self) -> AuthenticatedUserContext:
        return AuthenticatedUserContext(
            authenticated_user=self.authenticated_user,
            authentication_context_ref=self.authentication_context_ref,
        )


def user_checkpoint_source_context_v2_carried_sha256(
    value: UserCheckpointSourceContextV2,
) -> str:
    payload = value.model_dump(
        mode="python",
        exclude={"context_id", "context_sha256"},
        exclude_none=False,
    )
    return hashlib.sha256(
        json.dumps(
            canonical_value_v2(_without_audit_actor_time(payload)),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def user_checkpoint_source_context_v2_ref(
    value: UserCheckpointSourceContextV2,
) -> ObjectRef:
    observed = user_checkpoint_source_context_v2_carried_sha256(value)
    if (
        value.context_sha256 != observed
        or value.context_id != f"user-checkpoint-source-context://sha256/{observed}"
    ):
        raise ValueError("checkpoint source context identity is stale")
    return ObjectRef(
        object_type="user-checkpoint-source-context",
        object_id=value.context_id,
        object_version="private-v1",
        object_sha256=value.context_sha256,
    )


def _validate_variant_shape(value: UserCheckpointSourceContextV2) -> None:
    checkpoint = value.request_compilation.checkpoint
    counts = {
        ApprovalCheckpoint.LABEL_PLAN: (
            len(value.label_specs),
            len(value.label_plans),
        ),
        ApprovalCheckpoint.TASK_REWRITE_PLAN: (
            len(value.task_rewrite_plan_versions),
            len(value.task_rewrite_preview_safety_gates),
            len(value.task_rewrite_previews),
        ),
        ApprovalCheckpoint.ENVIRONMENT_STRATEGY: (len(value.environment_strategies),),
        ApprovalCheckpoint.FINAL_DATASET_REVIEW: (len(value.final_dataset_review_previews),),
    }
    active = counts[checkpoint]
    if len(set(active)) != 1:
        raise ValueError("checkpoint source variant arrays do not align")
    expected = value.request_compilation.request_count
    if active[0] != expected:
        empty_outcome = expected == 0 and active[0] == 0
        if not empty_outcome:
            raise ValueError("checkpoint source count differs from request count")
    all_values = (
        *value.label_specs,
        *value.label_plans,
        *value.task_rewrite_plan_versions,
        *value.task_rewrite_preview_safety_gates,
        *value.task_rewrite_previews,
        *value.environment_strategies,
        *value.final_dataset_review_previews,
    )
    if len(all_values) != sum(active):
        raise ValueError("checkpoint source context mixes source variants")


def _ref_key(value: ObjectRef) -> tuple[str, str, str, str]:
    return (
        value.object_type,
        value.object_id,
        value.object_version,
        value.object_sha256,
    )


def _without_audit_actor_time(value: object) -> object:
    if isinstance(value, dict):
        normalized: dict[object, object] = {}
        for key, nested in value.items():
            if (
                key == "audit"
                and isinstance(nested, dict)
                and {
                    "created_at",
                    "created_by",
                    "governing_versions",
                    "input_refs",
                }.issubset(nested)
            ):
                normalized[key] = {
                    "governing_versions": _without_audit_actor_time(nested["governing_versions"]),
                    "input_refs": _without_audit_actor_time(nested["input_refs"]),
                }
            else:
                normalized[key] = _without_audit_actor_time(nested)
        return normalized
    if isinstance(value, (list, tuple)):
        return tuple(_without_audit_actor_time(item) for item in value)
    return value


__all__ = [
    "TrustedAuthenticatedUserContextV2",
    "UserCheckpointSourceContextV2",
    "user_checkpoint_source_context_v2_ref",
]
