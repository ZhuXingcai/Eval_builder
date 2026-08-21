from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field, ValidationError, model_validator

from eval_factory.contracts.core import ContractAudit, Identifier, ObjectRef
from eval_factory.contracts.core_v2 import ContractModelV2
from eval_factory.harness.artifacts import (
    ArtifactEnvelopeV1,
    ArtifactModalityV1,
)
from eval_factory.harness.capability import (
    CapabilityCallV1,
    CapabilityConsumerBindingV1,
    CapabilityConsumerKindV1,
    CapabilityDefinitionV1,
    CapabilityInvocationContextV1,
    CapabilityInvocationOutcomeV1,
    CapabilityProviderBindingV1,
    CapabilityProviderKindV1,
    CapabilityResultV1,
)
from eval_factory.harness.composition import StaticPackRegistrationV1
from eval_factory.harness.contracts import (
    ref_key,
    require_sorted_unique_refs,
    sorted_refs,
    static_object_ref,
)
from eval_factory.harness.interaction import (
    BalancedAutonomyEvaluator,
    BalancedAutonomyPolicyV1,
    ExecutionAuthorityV1,
    GraphMutationKindV1,
    PermissionActionV1,
    PermissionDecisionV1,
    PermissionOutcomeV1,
)


class CapabilityRuntimeError(RuntimeError):
    """Base error for closed capability-runtime structural failures."""


class CapabilityRuntimeRegistrationError(CapabilityRuntimeError):
    """Raised when installed implementation authority differs from the Pack."""


class CapabilityRuntimeResolutionError(CapabilityRuntimeError):
    """Raised when an invocation cannot resolve one exact installed provider."""


class CapabilityRuntimeInputError(CapabilityRuntimeError):
    """Raised before provider work when invocation authority is inconsistent."""


class CapabilityPermissionScopeV1(ContractModelV2):
    schema_version: Literal["eval-harness/capability-permission-scope/v1"] = (
        "eval-harness/capability-permission-scope/v1"
    )

    member_id: Identifier
    task_id: Identifier
    data_scope_refs: tuple[ObjectRef, ...] = Field(default=(), max_length=10_000)
    graph_mutation: GraphMutationKindV1 = GraphMutationKindV1.NONE
    local_execution: bool
    deterministic: bool
    source_admission: bool = False
    external_execution: bool = False
    destructive: bool = False
    release: bool = False
    model_requests_delta: int = Field(default=0, ge=0, le=10_000_000)
    model_tokens_delta: int = Field(default=0, ge=0, le=10_000_000_000)
    cost_micro_usd_delta: int = Field(
        default=0,
        ge=0,
        le=10_000_000_000_000,
    )

    @model_validator(mode="after")
    def validate_scope(self) -> CapabilityPermissionScopeV1:
        require_sorted_unique_refs(self.data_scope_refs, "data_scope_refs")
        if self.local_execution == self.external_execution:
            raise ValueError("capability execution must be exactly local or external")
        return self


@dataclass(frozen=True, slots=True)
class CapabilityProviderExecution:
    outcome: CapabilityInvocationOutcomeV1
    canonical_result_ref: ObjectRef | None = None
    content_ref: ObjectRef | None = None
    validation_refs: tuple[ObjectRef, ...] = ()
    failure_code: str | None = None
    media_type: str = "application/json"
    modality: ArtifactModalityV1 = ArtifactModalityV1.DOCUMENT
    domain_tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.outcome is CapabilityInvocationOutcomeV1.SUCCEEDED:
            if self.canonical_result_ref is None or self.content_ref is None or self.failure_code is not None:
                raise ValueError(
                    "successful provider execution requires result and content authority",
                )
        elif (
            self.canonical_result_ref is not None or self.content_ref is not None or self.failure_code is None
        ):
            raise ValueError(
                "non-success provider execution requires only a closed failure code",
            )
        if self.validation_refs != sorted_refs(self.validation_refs):
            raise ValueError("provider validation refs must be sorted")
        if tuple(sorted(set(self.domain_tags))) != self.domain_tags:
            raise ValueError("provider domain tags must be sorted and unique")


@runtime_checkable
class CapabilityRuntimeProvider(Protocol):
    @property
    def provider_binding_ref(self) -> ObjectRef: ...

    @property
    def implementation_id(self) -> str: ...

    @property
    def request_model(self) -> type[BaseModel]: ...

    @property
    def request_schema_ref(self) -> ObjectRef: ...

    @property
    def result_model(self) -> type[BaseModel]: ...

    @property
    def result_schema_ref(self) -> ObjectRef: ...

    async def invoke(
        self,
        request: BaseModel,
        *,
        call: CapabilityCallV1,
        audit: ContractAudit,
    ) -> CapabilityProviderExecution: ...


@dataclass(frozen=True, slots=True)
class ResolvedCapabilityProvider:
    definition: CapabilityDefinitionV1
    binding: CapabilityProviderBindingV1
    implementation: CapabilityRuntimeProvider


@dataclass(frozen=True, slots=True)
class CapabilityRuntimeInvocation:
    call: CapabilityCallV1
    permission_action: PermissionActionV1
    permission_decision: PermissionDecisionV1
    result: CapabilityResultV1
    output_artifacts: tuple[ArtifactEnvelopeV1, ...]


class CapabilityRuntimeRegistry:
    """Immutable exact-composition registry for executable capability providers."""

    def __init__(
        self,
        *,
        registration: StaticPackRegistrationV1,
        providers: tuple[CapabilityRuntimeProvider, ...],
    ) -> None:
        self.registration = StaticPackRegistrationV1.model_validate(
            registration,
        )
        self._definitions = {
            ref_key(value.to_ref()): value for value in self.registration.capability_definitions
        }
        self._bindings = {ref_key(value.to_ref()): value for value in self.registration.provider_bindings}
        self._providers: dict[
            tuple[str, str, str, str],
            CapabilityRuntimeProvider,
        ] = {}
        for implementation in providers:
            key = ref_key(implementation.provider_binding_ref)
            if key in self._providers:
                raise CapabilityRuntimeRegistrationError(
                    "duplicate capability provider implementation",
                )
            binding = self._bindings.get(key)
            if binding is None:
                raise CapabilityRuntimeRegistrationError(
                    "capability implementation is not pinned by the static Pack",
                )
            definition = self._definitions.get(
                ref_key(binding.capability_definition_ref),
            )
            if definition is None:
                raise CapabilityRuntimeRegistrationError(
                    "capability provider definition is not installed",
                )
            if (
                implementation.implementation_id != binding.implementation_id
                or _schema_ref_for_model(
                    implementation.request_model,
                    implementation.request_schema_ref,
                )
                != definition.request_schema_ref
                or _schema_ref_for_model(
                    implementation.result_model,
                    implementation.result_schema_ref,
                )
                != definition.result_schema_ref
            ):
                raise CapabilityRuntimeRegistrationError(
                    "capability implementation authority drifted",
                )
            self._providers[key] = implementation
        if set(self._providers) != set(self._bindings):
            raise CapabilityRuntimeRegistrationError(
                "static Pack requires one installed implementation per provider",
            )

    @property
    def providers(self) -> tuple[CapabilityRuntimeProvider, ...]:
        return tuple(self._providers[key] for key in sorted(self._providers))

    def bind_consumer(
        self,
        *,
        capability_id: str,
        consumer_id: str,
        consumer_version: str,
        consumer_kind: CapabilityConsumerKindV1,
        projection_ref: ObjectRef | None,
        audit: ContractAudit,
    ) -> CapabilityConsumerBindingV1:
        matches = tuple(
            value
            for value in self.registration.capability_definitions
            if value.capability_id == capability_id
        )
        if len(matches) != 1:
            raise CapabilityRuntimeResolutionError(
                "capability ID is not uniquely installed",
            )
        definition = matches[0]
        binding = self._provider_binding(definition.to_ref())
        if (
            consumer_kind not in definition.supported_consumers
            or consumer_kind not in binding.supported_consumers
        ):
            raise CapabilityRuntimeResolutionError(
                "capability consumer kind is not supported",
            )
        return CapabilityConsumerBindingV1.create(
            consumer_id=consumer_id,
            consumer_version=consumer_version,
            consumer_kind=consumer_kind,
            capability_definition_ref=definition.to_ref(),
            provider_binding_ref=binding.to_ref(),
            projection_ref=projection_ref,
            audit=audit,
        )

    def resolve(
        self,
        consumer: CapabilityConsumerBindingV1,
    ) -> ResolvedCapabilityProvider:
        definition = self._definitions.get(
            ref_key(consumer.capability_definition_ref),
        )
        binding = self._bindings.get(ref_key(consumer.provider_binding_ref))
        implementation = self._providers.get(
            ref_key(consumer.provider_binding_ref),
        )
        if definition is None or binding is None or implementation is None:
            raise CapabilityRuntimeResolutionError(
                "capability consumer authority is not installed",
            )
        if binding.capability_definition_ref != definition.to_ref():
            raise CapabilityRuntimeResolutionError(
                "capability provider binding references another definition",
            )
        if (
            consumer.consumer_kind not in definition.supported_consumers
            or consumer.consumer_kind not in binding.supported_consumers
        ):
            raise CapabilityRuntimeResolutionError(
                "capability consumer kind is not supported",
            )
        return ResolvedCapabilityProvider(
            definition=definition,
            binding=binding,
            implementation=implementation,
        )

    def permission_policy(
        self,
        policy_ref: ObjectRef,
    ) -> BalancedAutonomyPolicyV1:
        matches = tuple(
            policy for policy in self.registration.permission_policies if policy.to_ref() == policy_ref
        )
        if len(matches) != 1:
            raise CapabilityRuntimeResolutionError(
                "execution permission policy is not installed",
            )
        return matches[0]

    def _provider_binding(
        self,
        definition_ref: ObjectRef,
    ) -> CapabilityProviderBindingV1:
        matches = tuple(
            value
            for value in self.registration.provider_bindings
            if value.capability_definition_ref == definition_ref
        )
        if len(matches) != 1:
            raise CapabilityRuntimeResolutionError(
                "capability provider is not uniquely installed",
            )
        return matches[0]


class HarnessCapabilityRuntime:
    def __init__(
        self,
        registry: CapabilityRuntimeRegistry,
        *,
        evaluator: BalancedAutonomyEvaluator | None = None,
    ) -> None:
        self.registry = registry
        self.evaluator = evaluator or BalancedAutonomyEvaluator()

    async def invoke(
        self,
        *,
        consumer: CapabilityConsumerBindingV1,
        context: CapabilityInvocationContextV1,
        permission_scope: CapabilityPermissionScopeV1,
        authority: ExecutionAuthorityV1,
        request: BaseModel,
        input_artifacts: tuple[ArtifactEnvelopeV1, ...],
        audit: ContractAudit,
    ) -> CapabilityRuntimeInvocation:
        resolved = self.registry.resolve(consumer)
        self._validate_invocation(
            resolved=resolved,
            context=context,
            permission_scope=permission_scope,
            authority=authority,
            request=request,
            input_artifacts=input_artifacts,
        )
        request_ref = _request_ref(request)
        input_refs = sorted_refs(value.to_ref() for value in input_artifacts)
        call = CapabilityCallV1.create(
            call_id=f"capability-call.{context.idempotency_key}",
            capability_definition_ref=resolved.definition.to_ref(),
            provider_binding_ref=resolved.binding.to_ref(),
            context=context,
            request_ref=request_ref,
            input_artifact_refs=input_refs,
            audit=audit,
        )
        policy = self.registry.permission_policy(
            authority.permission_policy_ref,
        )
        action = PermissionActionV1.create(
            action_id=f"capability-action.{call.object_sha256}",
            authority_ref=authority.to_ref(),
            permission_policy_ref=policy.to_ref(),
            member_id=permission_scope.member_id,
            principal_ref=context.principal_ref,
            capability_definition_ref=resolved.definition.to_ref(),
            provider_binding_ref=resolved.binding.to_ref(),
            task_id=permission_scope.task_id,
            data_scope_refs=permission_scope.data_scope_refs,
            data_purpose=context.data_purpose,
            data_classification=context.data_classification,
            side_effect=resolved.definition.side_effect,
            graph_mutation=permission_scope.graph_mutation,
            local_execution=permission_scope.local_execution,
            deterministic=permission_scope.deterministic,
            source_admission=permission_scope.source_admission,
            external_execution=permission_scope.external_execution,
            destructive=permission_scope.destructive,
            release=permission_scope.release,
            model_requests_delta=permission_scope.model_requests_delta,
            model_tokens_delta=permission_scope.model_tokens_delta,
            cost_micro_usd_delta=permission_scope.cost_micro_usd_delta,
            audit=audit,
        )
        decision = self.evaluator.evaluate(
            policy=policy,
            authority=authority,
            action=action,
            audit=audit,
        )
        if decision.outcome is not PermissionOutcomeV1.ALLOW:
            failure_code = (
                "CAPABILITY_APPROVAL_REQUIRED"
                if decision.outcome is PermissionOutcomeV1.REQUIRE_APPROVAL
                else "CAPABILITY_PERMISSION_DENIED"
            )
            result = CapabilityResultV1.create(
                result_id=f"capability-result.{call.object_sha256}",
                call_ref=call.to_ref(),
                outcome=CapabilityInvocationOutcomeV1.BLOCKED_POLICY,
                canonical_result_ref=None,
                output_artifact_refs=(),
                validation_refs=(decision.to_ref(),),
                failure_code=failure_code,
                audit=audit,
            )
            return CapabilityRuntimeInvocation(
                call=call,
                permission_action=action,
                permission_decision=decision,
                result=result,
                output_artifacts=(),
            )

        execution = await resolved.implementation.invoke(
            request,
            call=call,
            audit=audit,
        )
        validation_refs = sorted_refs(
            (decision.to_ref(), *execution.validation_refs),
        )
        outputs: tuple[ArtifactEnvelopeV1, ...]
        output_refs: tuple[ObjectRef, ...]
        if execution.outcome is CapabilityInvocationOutcomeV1.SUCCEEDED:
            assert execution.canonical_result_ref is not None
            assert execution.content_ref is not None
            output = ArtifactEnvelopeV1.create(
                artifact_id=(f"artifact.{call.object_sha256}.{resolved.definition.output_artifact_roles[0]}"),
                subject_ref=execution.canonical_result_ref,
                schema_ref=resolved.definition.result_schema_ref,
                content_ref=execution.content_ref,
                media_type=execution.media_type,
                modality=execution.modality,
                domain_tags=execution.domain_tags,
                semantic_role=resolved.definition.output_artifact_roles[0],
                purpose=context.data_purpose,
                classification=context.data_classification,
                lineage_refs=sorted_refs((request_ref, *input_refs)),
                producer_capability_ref=resolved.definition.to_ref(),
                producer_task_ref=context.task_ref,
                validation_refs=validation_refs,
                revision=1,
                predecessor_envelope_ref=None,
                audit=audit,
            )
            outputs = (output,)
            output_refs = (output.to_ref(),)
        else:
            outputs = ()
            output_refs = ()
        result = CapabilityResultV1.create(
            result_id=f"capability-result.{call.object_sha256}",
            call_ref=call.to_ref(),
            outcome=execution.outcome,
            canonical_result_ref=execution.canonical_result_ref,
            output_artifact_refs=output_refs,
            validation_refs=validation_refs,
            failure_code=execution.failure_code,
            audit=audit,
        )
        return CapabilityRuntimeInvocation(
            call=call,
            permission_action=action,
            permission_decision=decision,
            result=result,
            output_artifacts=outputs,
        )

    @staticmethod
    def _validate_invocation(
        *,
        resolved: ResolvedCapabilityProvider,
        context: CapabilityInvocationContextV1,
        permission_scope: CapabilityPermissionScopeV1,
        authority: ExecutionAuthorityV1,
        request: BaseModel,
        input_artifacts: tuple[ArtifactEnvelopeV1, ...],
    ) -> None:
        if context.authority_ref != authority.to_ref():
            raise CapabilityRuntimeInputError(
                "capability context uses stale execution authority",
            )
        if authority.permission_policy_ref not in resolved.binding.policy_refs:
            raise CapabilityRuntimeInputError(
                "capability provider is not governed by the execution policy",
            )
        if type(request) is not resolved.implementation.request_model:
            raise CapabilityRuntimeInputError(
                "capability request model differs from the provider schema",
            )
        try:
            canonical_request = resolved.implementation.request_model.model_validate_json(
                request.model_dump_json(),
            )
        except ValidationError as exc:
            raise CapabilityRuntimeInputError(
                "capability request authority is invalid",
            ) from exc
        if canonical_request != request or _request_ref(canonical_request) != _request_ref(request):
            raise CapabilityRuntimeInputError(
                "capability request authority drifted",
            )
        if context.data_purpose not in resolved.definition.data_purposes:
            raise CapabilityRuntimeInputError(
                "capability data purpose is unsupported",
            )
        if context.data_classification not in resolved.definition.data_classifications:
            raise CapabilityRuntimeInputError(
                "capability data classification is unsupported",
            )
        roles = tuple(sorted(value.semantic_role for value in input_artifacts))
        if set(roles) != set(resolved.definition.input_artifact_roles):
            raise CapabilityRuntimeInputError(
                "capability input artifact roles differ from the definition",
            )
        if any(
            value.purpose != context.data_purpose or value.classification != context.data_classification
            for value in input_artifacts
        ):
            raise CapabilityRuntimeInputError(
                "capability input artifact scope differs from invocation context",
            )
        input_refs = sorted_refs(value.to_ref() for value in input_artifacts)
        if permission_scope.data_scope_refs != input_refs:
            raise CapabilityRuntimeInputError(
                "capability permission scope differs from input artifacts",
            )
        local_provider = resolved.binding.provider_kind in {
            CapabilityProviderKindV1.LOCAL_DETERMINISTIC,
            CapabilityProviderKindV1.LOCAL_MODEL_ASSISTED,
        }
        if local_provider != permission_scope.local_execution:
            raise CapabilityRuntimeInputError(
                "capability execution locality differs from provider binding",
            )


@dataclass(frozen=True, slots=True)
class CapabilityConsumerFacade:
    runtime: HarnessCapabilityRuntime
    binding: CapabilityConsumerBindingV1

    async def invoke(
        self,
        *,
        context: CapabilityInvocationContextV1,
        permission_scope: CapabilityPermissionScopeV1,
        authority: ExecutionAuthorityV1,
        request: BaseModel,
        input_artifacts: tuple[ArtifactEnvelopeV1, ...],
        audit: ContractAudit,
    ) -> CapabilityRuntimeInvocation:
        return await self.runtime.invoke(
            consumer=self.binding,
            context=context,
            permission_scope=permission_scope,
            authority=authority,
            request=request,
            input_artifacts=input_artifacts,
            audit=audit,
        )


def _request_ref(request: BaseModel) -> ObjectRef:
    to_ref = getattr(request, "to_ref", None)
    if not callable(to_ref):
        raise CapabilityRuntimeInputError(
            "capability request lacks immutable reference authority",
        )
    reference = to_ref()
    if not isinstance(reference, ObjectRef):
        raise CapabilityRuntimeInputError(
            "capability request reference authority is invalid",
        )
    return reference


def _schema_ref_for_model(
    model_type: type[BaseModel],
    claimed_ref: ObjectRef,
) -> ObjectRef:
    if claimed_ref.object_type != "json-schema":
        raise CapabilityRuntimeRegistrationError(
            "capability implementation schema authority is invalid",
        )
    return static_object_ref(
        object_type="json-schema",
        object_id=claimed_ref.object_id,
        object_version=claimed_ref.object_version,
        payload=model_type.model_json_schema(),
    )


__all__ = [
    "CapabilityConsumerFacade",
    "CapabilityPermissionScopeV1",
    "CapabilityProviderExecution",
    "CapabilityRuntimeError",
    "CapabilityRuntimeInputError",
    "CapabilityRuntimeInvocation",
    "CapabilityRuntimeProvider",
    "CapabilityRuntimeRegistrationError",
    "CapabilityRuntimeRegistry",
    "CapabilityRuntimeResolutionError",
    "HarnessCapabilityRuntime",
    "ResolvedCapabilityProvider",
]
