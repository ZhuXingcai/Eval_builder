from __future__ import annotations

from typing import ClassVar, Literal, Self

from pydantic import Field, model_validator

from eval_factory.contracts.core import ContractAudit, Identifier, ObjectRef
from eval_factory.contracts.core_v2 import ContractModelV2
from eval_factory.harness.capability import (
    CapabilityDefinitionV1,
    CapabilityProviderBindingV1,
)
from eval_factory.harness.contracts import (
    HarnessObjectV1,
    ref_key,
    require_ref,
    require_sorted_unique_refs,
    sorted_refs,
)
from eval_factory.harness.interaction import BalancedAutonomyPolicyV1
from eval_factory.harness.projections import ProjectionDefinitionV1


class PackRegistrationError(ValueError):
    pass


class PackResolutionError(LookupError):
    pass


class PackAgentProfileV1(HarnessObjectV1):
    schema_version: Literal["eval-harness/pack-agent-profile/v1"] = "eval-harness/pack-agent-profile/v1"
    OBJECT_TYPE: ClassVar[str] = "pack-agent-profile"

    profile_id: Identifier
    profile_version: str = Field(min_length=1, max_length=128)
    role: Identifier
    capability_definition_refs: tuple[ObjectRef, ...] = Field(
        min_length=1,
        max_length=256,
    )
    context_projection_ref: ObjectRef
    model_policy_ref: ObjectRef
    coordinator: bool = False

    @model_validator(mode="after")
    def validate_profile(self) -> Self:
        require_sorted_unique_refs(
            self.capability_definition_refs,
            "capability_definition_refs",
        )
        if any(ref.object_type != "harness-capability-definition" for ref in self.capability_definition_refs):
            raise ValueError("Agent profile capabilities must reference Harness definitions")
        require_ref(
            self.context_projection_ref,
            "harness-projection-definition",
            "context_projection_ref",
        )
        require_ref(
            self.model_policy_ref,
            "model-routing-policy",
            "model_policy_ref",
            object_version=self.model_policy_ref.object_version,
        )
        return self


class EvalPackManifestV1(HarnessObjectV1):
    schema_version: Literal["eval-harness/eval-pack-manifest/v1"] = "eval-harness/eval-pack-manifest/v1"
    OBJECT_TYPE: ClassVar[str] = "eval-pack-manifest"

    pack_id: Identifier
    pack_version: str = Field(min_length=1, max_length=128)
    harness_api_version: Literal["eval-harness/v1"] = "eval-harness/v1"
    artifact_schema_refs: tuple[ObjectRef, ...] = Field(min_length=1, max_length=1_000)
    capability_definition_refs: tuple[ObjectRef, ...] = Field(
        min_length=1,
        max_length=1_000,
    )
    provider_binding_refs: tuple[ObjectRef, ...] = Field(
        min_length=1,
        max_length=1_000,
    )
    agent_profile_refs: tuple[ObjectRef, ...] = Field(
        min_length=1,
        max_length=1_000,
    )
    graph_template_ref: ObjectRef
    blueprint_template_ref: ObjectRef
    permission_policy_refs: tuple[ObjectRef, ...] = Field(
        min_length=1,
        max_length=64,
    )
    projection_refs: tuple[ObjectRef, ...] = Field(min_length=1, max_length=256)
    delivery_adapter_refs: tuple[ObjectRef, ...] = Field(
        min_length=1,
        max_length=64,
    )
    certification_refs: tuple[ObjectRef, ...] = Field(min_length=1, max_length=256)

    @classmethod
    def create(  # type: ignore[override]
        cls,
        *,
        pack_id: str,
        pack_version: str,
        artifact_schema_refs: tuple[ObjectRef, ...],
        capability_definition_refs: tuple[ObjectRef, ...],
        provider_binding_refs: tuple[ObjectRef, ...],
        agent_profile_refs: tuple[ObjectRef, ...],
        graph_template_ref: ObjectRef,
        blueprint_template_ref: ObjectRef,
        permission_policy_refs: tuple[ObjectRef, ...],
        projection_refs: tuple[ObjectRef, ...],
        delivery_adapter_refs: tuple[ObjectRef, ...],
        certification_refs: tuple[ObjectRef, ...],
        audit: ContractAudit,
    ) -> EvalPackManifestV1:
        return super().create(
            audit=audit,
            pack_id=pack_id,
            pack_version=pack_version,
            artifact_schema_refs=sorted_refs(artifact_schema_refs),
            capability_definition_refs=sorted_refs(capability_definition_refs),
            provider_binding_refs=sorted_refs(provider_binding_refs),
            agent_profile_refs=sorted_refs(agent_profile_refs),
            graph_template_ref=graph_template_ref,
            blueprint_template_ref=blueprint_template_ref,
            permission_policy_refs=sorted_refs(permission_policy_refs),
            projection_refs=sorted_refs(projection_refs),
            delivery_adapter_refs=sorted_refs(delivery_adapter_refs),
            certification_refs=sorted_refs(certification_refs),
        )

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        for values, label, object_type in (
            (self.artifact_schema_refs, "artifact_schema_refs", "json-schema"),
            (
                self.capability_definition_refs,
                "capability_definition_refs",
                "harness-capability-definition",
            ),
            (
                self.provider_binding_refs,
                "provider_binding_refs",
                "harness-capability-provider",
            ),
            (
                self.agent_profile_refs,
                "agent_profile_refs",
                "pack-agent-profile",
            ),
            (
                self.permission_policy_refs,
                "permission_policy_refs",
                "balanced-autonomy-policy",
            ),
            (
                self.projection_refs,
                "projection_refs",
                "harness-projection-definition",
            ),
        ):
            require_sorted_unique_refs(values, label)
            if any(ref.object_type != object_type for ref in values):
                raise ValueError(f"{label} contains an invalid object type")
        require_ref(
            self.graph_template_ref,
            "graph-template",
            "graph_template_ref",
        )
        require_ref(
            self.blueprint_template_ref,
            "blueprint-template",
            "blueprint_template_ref",
        )
        require_sorted_unique_refs(
            self.delivery_adapter_refs,
            "delivery_adapter_refs",
        )
        if any(ref.object_type != "delivery-adapter" for ref in self.delivery_adapter_refs):
            raise ValueError("delivery adapters contain an invalid object type")
        require_sorted_unique_refs(self.certification_refs, "certification_refs")
        return self


class HarnessCompositionV1(HarnessObjectV1):
    schema_version: Literal["eval-harness/composition/v1"] = "eval-harness/composition/v1"
    OBJECT_TYPE: ClassVar[str] = "harness-composition"

    profile_id: Identifier
    profile_version: str = Field(min_length=1, max_length=128)
    harness_api_version: Literal["eval-harness/v1"] = "eval-harness/v1"
    pack_manifest_refs: tuple[ObjectRef, ...] = Field(min_length=1, max_length=64)
    capability_definition_refs: tuple[ObjectRef, ...] = Field(
        min_length=1,
        max_length=1_000,
    )
    provider_binding_refs: tuple[ObjectRef, ...] = Field(
        min_length=1,
        max_length=1_000,
    )
    agent_profile_refs: tuple[ObjectRef, ...] = Field(
        min_length=1,
        max_length=1_000,
    )
    permission_policy_refs: tuple[ObjectRef, ...] = Field(
        min_length=1,
        max_length=64,
    )
    projection_refs: tuple[ObjectRef, ...] = Field(min_length=1, max_length=256)
    blueprint_template_ref: ObjectRef

    @classmethod
    def create(  # type: ignore[override]
        cls,
        *,
        profile_id: str,
        profile_version: str,
        pack_manifest_refs: tuple[ObjectRef, ...],
        capability_definition_refs: tuple[ObjectRef, ...],
        provider_binding_refs: tuple[ObjectRef, ...],
        agent_profile_refs: tuple[ObjectRef, ...],
        permission_policy_refs: tuple[ObjectRef, ...],
        projection_refs: tuple[ObjectRef, ...],
        blueprint_template_ref: ObjectRef,
        audit: ContractAudit,
    ) -> HarnessCompositionV1:
        return super().create(
            audit=audit,
            profile_id=profile_id,
            profile_version=profile_version,
            pack_manifest_refs=sorted_refs(pack_manifest_refs),
            capability_definition_refs=sorted_refs(capability_definition_refs),
            provider_binding_refs=sorted_refs(provider_binding_refs),
            agent_profile_refs=sorted_refs(agent_profile_refs),
            permission_policy_refs=sorted_refs(permission_policy_refs),
            projection_refs=sorted_refs(projection_refs),
            blueprint_template_ref=blueprint_template_ref,
        )

    @model_validator(mode="after")
    def validate_composition(self) -> Self:
        for values, label in (
            (self.pack_manifest_refs, "pack_manifest_refs"),
            (self.capability_definition_refs, "capability_definition_refs"),
            (self.provider_binding_refs, "provider_binding_refs"),
            (self.agent_profile_refs, "agent_profile_refs"),
            (self.permission_policy_refs, "permission_policy_refs"),
            (self.projection_refs, "projection_refs"),
        ):
            require_sorted_unique_refs(values, label)
        require_ref(
            self.blueprint_template_ref,
            "blueprint-template",
            "blueprint_template_ref",
        )
        return self


class StaticPackRegistrationV1(ContractModelV2):
    schema_version: Literal["eval-harness/static-pack-registration/v1"] = (
        "eval-harness/static-pack-registration/v1"
    )
    manifest: EvalPackManifestV1
    capability_definitions: tuple[CapabilityDefinitionV1, ...] = Field(
        min_length=1,
        max_length=1_000,
    )
    provider_bindings: tuple[CapabilityProviderBindingV1, ...] = Field(
        min_length=1,
        max_length=1_000,
    )
    agent_profiles: tuple[PackAgentProfileV1, ...] = Field(
        min_length=1,
        max_length=1_000,
    )
    permission_policies: tuple[BalancedAutonomyPolicyV1, ...] = Field(
        min_length=1,
        max_length=64,
    )
    projections: tuple[ProjectionDefinitionV1, ...] = Field(
        min_length=1,
        max_length=256,
    )
    composition: HarnessCompositionV1

    @model_validator(mode="after")
    def validate_registration(self) -> Self:
        _validate_registration(self)
        return self


class StaticPackRegistry:
    def __init__(
        self,
        registrations: tuple[StaticPackRegistrationV1, ...],
    ) -> None:
        self._by_version: dict[
            tuple[str, str],
            StaticPackRegistrationV1,
        ] = {}
        self._by_manifest: dict[
            tuple[str, str, str, str],
            StaticPackRegistrationV1,
        ] = {}
        for raw_registration in registrations:
            registration = StaticPackRegistrationV1.model_validate(
                raw_registration,
            )
            key = (
                registration.manifest.pack_id,
                registration.manifest.pack_version,
            )
            manifest_key = ref_key(registration.manifest.to_ref())
            if key in self._by_version or manifest_key in self._by_manifest:
                raise PackRegistrationError("duplicate static Pack authority")
            self._by_version[key] = registration
            self._by_manifest[manifest_key] = registration

    @property
    def registrations(self) -> tuple[StaticPackRegistrationV1, ...]:
        return tuple(self._by_version[key] for key in sorted(self._by_version))

    def resolve(
        self,
        *,
        pack_id: str,
        pack_version: str,
        expected_manifest_ref: ObjectRef | None = None,
    ) -> StaticPackRegistrationV1:
        registration = self._by_version.get((pack_id, pack_version))
        if registration is None:
            raise PackResolutionError("static Pack version is not registered")
        if expected_manifest_ref is not None and registration.manifest.to_ref() != expected_manifest_ref:
            raise PackResolutionError("static Pack manifest authority drifted")
        return registration

    def resolve_manifest(
        self,
        manifest_ref: ObjectRef,
    ) -> StaticPackRegistrationV1:
        registration = self._by_manifest.get(ref_key(manifest_ref))
        if registration is None:
            raise PackResolutionError("static Pack manifest is not registered")
        return registration


def _validate_registration(registration: StaticPackRegistrationV1) -> None:
    manifest = registration.manifest
    expected_sets = (
        (
            manifest.capability_definition_refs,
            tuple(value.to_ref() for value in registration.capability_definitions),
            "capability definitions",
        ),
        (
            manifest.provider_binding_refs,
            tuple(value.to_ref() for value in registration.provider_bindings),
            "provider bindings",
        ),
        (
            manifest.agent_profile_refs,
            tuple(value.to_ref() for value in registration.agent_profiles),
            "Agent profiles",
        ),
        (
            manifest.permission_policy_refs,
            tuple(value.to_ref() for value in registration.permission_policies),
            "permission policies",
        ),
        (
            manifest.projection_refs,
            tuple(value.to_ref() for value in registration.projections),
            "projections",
        ),
    )
    for manifest_refs, actual_refs, label in expected_sets:
        if manifest_refs != sorted_refs(actual_refs):
            raise ValueError(f"Pack manifest {label} differ from registration")

    definitions = {definition.to_ref(): definition for definition in registration.capability_definitions}
    provider_counts = {definition_ref: 0 for definition_ref in definitions}
    for provider in registration.provider_bindings:
        definition = definitions.get(provider.capability_definition_ref)
        if definition is None:
            raise ValueError("Pack provider references an unregistered capability")
        if not set(provider.supported_consumers).issubset(
            definition.supported_consumers,
        ):
            raise ValueError("Pack provider widens capability consumer support")
        provider_counts[provider.capability_definition_ref] += 1
    if any(count != 1 for count in provider_counts.values()):
        raise ValueError("V1 static composition requires exactly one provider per capability")

    definition_refs = set(definitions)
    for profile in registration.agent_profiles:
        if not set(profile.capability_definition_refs).issubset(definition_refs):
            raise ValueError("Agent profile references an unregistered capability")
    if sum(profile.coordinator for profile in registration.agent_profiles) != 1:
        raise ValueError("static Pack requires exactly one coordinator profile")

    required_schema_refs = {
        schema_ref
        for definition in registration.capability_definitions
        for schema_ref in (
            definition.request_schema_ref,
            definition.result_schema_ref,
        )
    }
    if not required_schema_refs.issubset(manifest.artifact_schema_refs):
        raise ValueError("Pack manifest omits a capability request/result schema")

    composition = registration.composition
    if composition.pack_manifest_refs != (manifest.to_ref(),):
        raise ValueError("static composition must pin the registered Pack manifest")
    for composition_refs, manifest_refs, label in (
        (
            composition.capability_definition_refs,
            manifest.capability_definition_refs,
            "capabilities",
        ),
        (
            composition.provider_binding_refs,
            manifest.provider_binding_refs,
            "providers",
        ),
        (
            composition.agent_profile_refs,
            manifest.agent_profile_refs,
            "Agent profiles",
        ),
        (
            composition.permission_policy_refs,
            manifest.permission_policy_refs,
            "permission policies",
        ),
        (
            composition.projection_refs,
            manifest.projection_refs,
            "projections",
        ),
    ):
        if composition_refs != manifest_refs:
            raise ValueError(f"static composition {label} differ from Pack manifest")
    if composition.blueprint_template_ref != manifest.blueprint_template_ref:
        raise ValueError("static composition Blueprint template differs from Pack")


__all__ = [
    "EvalPackManifestV1",
    "HarnessCompositionV1",
    "PackAgentProfileV1",
    "PackRegistrationError",
    "PackResolutionError",
    "StaticPackRegistrationV1",
    "StaticPackRegistry",
]
