from __future__ import annotations

import subprocess
import sys
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from eval_factory.blueprints import (
    BlueprintCapabilityBindingV1,
    BlueprintDataFlowEdgeV1,
    EvaluationBlueprintV1,
)
from eval_factory.contracts.core import (
    ContractAudit,
    ObjectRef,
    VersionBinding,
)
from eval_factory.harness import (
    CapabilityProviderBindingV1,
    PackResolutionError,
    StaticPackRegistrationV1,
    StaticPackRegistry,
)
from eval_factory.packs import (
    GENERIC_AGENT_TRACE_PACK_EXECUTION_VERSION,
    GENERIC_AGENT_TRACE_PACK_ID,
    GENERIC_AGENT_TRACE_PACK_LEGACY_VERSION,
    GENERIC_AGENT_TRACE_PACK_VERSION,
    build_generic_agent_evaluation_blueprint,
    build_generic_agent_trace_execution_pack,
    build_generic_agent_trace_pack,
    build_generic_agent_trace_pack_v1_0,
)

HASH = "a" * 64


def _ref(
    object_type: str,
    suffix: str = "example",
    *,
    version: str = "v1",
) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://{suffix}/{version}",
        object_version=version,
        object_sha256=HASH,
    )


def _audit() -> ContractAudit:
    return ContractAudit(
        created_at=datetime(2026, 8, 17, tzinfo=UTC),
        created_by="static-pack-test",
        governing_versions=(
            VersionBinding(
                component="eval-harness-stage0",
                version="v1",
                sha256=HASH,
            ),
        ),
    )


def _registration() -> StaticPackRegistrationV1:
    return build_generic_agent_trace_pack(audit=_audit())


def test_pack_root_remains_a_cold_import() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import eval_factory.packs; "
                "blocked = [name for name in sys.modules "
                "if name.endswith('.generic_agent_trace.capabilities') "
                "or name.endswith('.agent_system.candidate_output')]; "
                "print('\\n'.join(blocked))"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""


def test_static_pack_registration_is_exact_and_resolvable() -> None:
    registration = _registration()
    registry = StaticPackRegistry((registration,))

    resolved = registry.resolve(
        pack_id=GENERIC_AGENT_TRACE_PACK_ID,
        pack_version=GENERIC_AGENT_TRACE_PACK_VERSION,
        expected_manifest_ref=registration.manifest.to_ref(),
    )
    assert resolved == registration
    assert registry.resolve_manifest(registration.manifest.to_ref()) == registration
    assert len(resolved.capability_definitions) == 10
    assert len(resolved.provider_bindings) == 10
    assert {value.capability_id for value in resolved.capability_definitions} >= {"capability.batch-quality"}
    assert len(resolved.agent_profiles) == 5
    assert sum(profile.coordinator for profile in resolved.agent_profiles) == 1


def test_legacy_static_pack_remains_exactly_resolvable() -> None:
    legacy = build_generic_agent_trace_pack_v1_0(audit=_audit())
    current = _registration()
    registry = StaticPackRegistry((legacy, current))

    assert (
        registry.resolve(
            pack_id=GENERIC_AGENT_TRACE_PACK_ID,
            pack_version=GENERIC_AGENT_TRACE_PACK_LEGACY_VERSION,
            expected_manifest_ref=legacy.manifest.to_ref(),
        )
        == legacy
    )
    assert len(legacy.capability_definitions) == 9
    assert "capability.batch-quality" not in {value.capability_id for value in legacy.capability_definitions}
    with pytest.raises(PackResolutionError, match="drifted"):
        registry.resolve(
            pack_id=GENERIC_AGENT_TRACE_PACK_ID,
            pack_version=GENERIC_AGENT_TRACE_PACK_LEGACY_VERSION,
            expected_manifest_ref=current.manifest.to_ref(),
        )


def test_execution_pack_preserves_frozen_versions() -> None:
    legacy = build_generic_agent_trace_pack_v1_0(audit=_audit())
    current = _registration()
    execution = build_generic_agent_trace_execution_pack(audit=_audit())
    registry = StaticPackRegistry((legacy, current, execution))

    assert (
        registry.resolve(
            pack_id=GENERIC_AGENT_TRACE_PACK_ID,
            pack_version=GENERIC_AGENT_TRACE_PACK_EXECUTION_VERSION,
            expected_manifest_ref=execution.manifest.to_ref(),
        )
        == execution
    )
    assert (
        registry.resolve(
            pack_id=GENERIC_AGENT_TRACE_PACK_ID,
            pack_version=GENERIC_AGENT_TRACE_PACK_VERSION,
            expected_manifest_ref=current.manifest.to_ref(),
        )
        == current
    )
    quality = next(
        value
        for value in execution.capability_definitions
        if value.capability_id == "capability.quality-review"
    )
    current_quality = next(
        value
        for value in current.capability_definitions
        if value.capability_id == "capability.quality-review"
    )
    assert quality.request_schema_ref != current_quality.request_schema_ref


def test_static_pack_resolution_fails_closed_on_version_or_manifest_drift() -> None:
    registration = _registration()
    registry = StaticPackRegistry((registration,))

    with pytest.raises(PackResolutionError, match="version"):
        registry.resolve(
            pack_id=GENERIC_AGENT_TRACE_PACK_ID,
            pack_version="2.0.0",
        )
    with pytest.raises(PackResolutionError, match="drifted"):
        registry.resolve(
            pack_id=GENERIC_AGENT_TRACE_PACK_ID,
            pack_version=GENERIC_AGENT_TRACE_PACK_VERSION,
            expected_manifest_ref=_ref("eval-pack-manifest", "drifted"),
        )
    with pytest.raises(PackResolutionError, match="not registered"):
        registry.resolve_manifest(_ref("eval-pack-manifest", "missing"))


def test_static_registration_rejects_manifest_or_provider_drift() -> None:
    registration = _registration()
    source = registration.manifest
    manifest = type(source).create(
        pack_id=source.pack_id,
        pack_version=source.pack_version,
        artifact_schema_refs=source.artifact_schema_refs,
        capability_definition_refs=source.capability_definition_refs[1:],
        provider_binding_refs=source.provider_binding_refs,
        agent_profile_refs=source.agent_profile_refs,
        graph_template_ref=source.graph_template_ref,
        blueprint_template_ref=source.blueprint_template_ref,
        permission_policy_refs=source.permission_policy_refs,
        projection_refs=source.projection_refs,
        delivery_adapter_refs=source.delivery_adapter_refs,
        certification_refs=source.certification_refs,
        audit=_audit(),
    )
    with pytest.raises(ValidationError, match="capability definitions differ"):
        StaticPackRegistrationV1(
            manifest=manifest,
            capability_definitions=registration.capability_definitions,
            provider_bindings=registration.provider_bindings,
            agent_profiles=registration.agent_profiles,
            permission_policies=registration.permission_policies,
            projections=registration.projections,
            composition=registration.composition,
        )

    source_provider = registration.provider_bindings[0]
    duplicate_provider = CapabilityProviderBindingV1.create(
        provider_id=f"{source_provider.provider_id}.second",
        provider_version=source_provider.provider_version,
        capability_definition_ref=source_provider.capability_definition_ref,
        implementation_id=f"{source_provider.implementation_id}.second",
        provider_kind=source_provider.provider_kind,
        supported_consumers=source_provider.supported_consumers,
        policy_refs=source_provider.policy_refs,
        audit=_audit(),
    )
    provider_manifest = type(source).create(
        pack_id=source.pack_id,
        pack_version=source.pack_version,
        artifact_schema_refs=source.artifact_schema_refs,
        capability_definition_refs=source.capability_definition_refs,
        provider_binding_refs=(
            *source.provider_binding_refs,
            duplicate_provider.to_ref(),
        ),
        agent_profile_refs=source.agent_profile_refs,
        graph_template_ref=source.graph_template_ref,
        blueprint_template_ref=source.blueprint_template_ref,
        permission_policy_refs=source.permission_policy_refs,
        projection_refs=source.projection_refs,
        delivery_adapter_refs=source.delivery_adapter_refs,
        certification_refs=source.certification_refs,
        audit=_audit(),
    )
    with pytest.raises(ValidationError, match="exactly one provider"):
        StaticPackRegistrationV1(
            manifest=provider_manifest,
            capability_definitions=registration.capability_definitions,
            provider_bindings=(
                *registration.provider_bindings,
                duplicate_provider,
            ),
            agent_profiles=registration.agent_profiles,
            permission_policies=registration.permission_policies,
            projections=registration.projections,
            composition=registration.composition,
        )


def test_fixed_blueprint_binds_pack_team_authority_and_connected_data_flow() -> None:
    registration = _registration()
    blueprint = build_generic_agent_evaluation_blueprint(
        registration=registration,
        requirement_ref=_ref("evaluation-requirement-spec", version="v2"),
        team_ref=_ref("agent-team"),
        roster_ref=_ref("team-roster"),
        task_graph_ref=_ref("team-task-graph"),
        authority_ref=_ref("execution-authority"),
        audit=_audit(),
    )

    assert blueprint.composition_ref == registration.composition.to_ref()
    assert blueprint.pack_manifest_refs == (registration.manifest.to_ref(),)
    assert len(blueprint.capability_bindings) == 10
    assert len(blueprint.data_flow_edges) == 19
    assert blueprint.delivery_profile_ref.object_type == "delivery-profile"
    assert blueprint.to_ref().object_type == "evaluation-blueprint"


def test_blueprint_rejects_schema_disconnect_and_cycles() -> None:
    registration = _registration()
    blueprint = build_generic_agent_evaluation_blueprint(
        registration=registration,
        requirement_ref=_ref("evaluation-requirement-spec", version="v2"),
        team_ref=_ref("agent-team"),
        roster_ref=_ref("team-roster"),
        task_graph_ref=_ref("team-task-graph"),
        authority_ref=_ref("execution-authority"),
        audit=_audit(),
    )
    edge = blueprint.data_flow_edges[0]
    disconnected = edge.model_copy(update={"artifact_schema_ref": _ref("json-schema", "unknown")})
    with pytest.raises(ValidationError, match="not produced"):
        EvaluationBlueprintV1.create(
            blueprint_id="evaluation-blueprint.invalid-schema",
            blueprint_version=1,
            requirement_ref=blueprint.requirement_ref,
            composition_ref=blueprint.composition_ref,
            pack_manifest_refs=blueprint.pack_manifest_refs,
            artifact_schema_bindings=blueprint.artifact_schema_bindings,
            capability_bindings=blueprint.capability_bindings,
            data_flow_edges=(disconnected, *blueprint.data_flow_edges[1:]),
            team_ref=blueprint.team_ref,
            roster_ref=blueprint.roster_ref,
            task_graph_ref=blueprint.task_graph_ref,
            authority_ref=blueprint.authority_ref,
            graph_template_ref=blueprint.graph_template_ref,
            execution_provider_refs=blueprint.execution_provider_refs,
            evaluator_refs=blueprint.evaluator_refs,
            review_gate_refs=blueprint.review_gate_refs,
            budget=blueprint.budget,
            delivery_profile_ref=blueprint.delivery_profile_ref,
            audit=_audit(),
        )

    requirement = next(
        binding
        for binding in blueprint.capability_bindings
        if binding.capability_id == "capability.requirement-planning"
    )
    delivery = next(
        binding for binding in blueprint.capability_bindings if binding.capability_id == "capability.delivery"
    )
    cyclic_requirement = BlueprintCapabilityBindingV1(
        capability_id=requirement.capability_id,
        capability_definition_ref=requirement.capability_definition_ref,
        provider_binding_ref=requirement.provider_binding_ref,
        input_schema_refs=(delivery.output_schema_refs[0],),
        output_schema_refs=requirement.output_schema_refs,
    )
    cycle_edge = BlueprintDataFlowEdgeV1(
        edge_id="edge.delivery.requirement",
        producer_capability_id=delivery.capability_id,
        consumer_capability_id=requirement.capability_id,
        artifact_schema_ref=delivery.output_schema_refs[0],
    )
    bindings = tuple(
        cyclic_requirement if value.capability_id == requirement.capability_id else value
        for value in blueprint.capability_bindings
    )
    with pytest.raises(ValidationError, match="acyclic"):
        EvaluationBlueprintV1.create(
            blueprint_id="evaluation-blueprint.invalid-cycle",
            blueprint_version=1,
            requirement_ref=blueprint.requirement_ref,
            composition_ref=blueprint.composition_ref,
            pack_manifest_refs=blueprint.pack_manifest_refs,
            artifact_schema_bindings=blueprint.artifact_schema_bindings,
            capability_bindings=bindings,
            data_flow_edges=(*blueprint.data_flow_edges, cycle_edge),
            team_ref=blueprint.team_ref,
            roster_ref=blueprint.roster_ref,
            task_graph_ref=blueprint.task_graph_ref,
            authority_ref=blueprint.authority_ref,
            graph_template_ref=blueprint.graph_template_ref,
            execution_provider_refs=blueprint.execution_provider_refs,
            evaluator_refs=blueprint.evaluator_refs,
            review_gate_refs=blueprint.review_gate_refs,
            budget=blueprint.budget,
            delivery_profile_ref=blueprint.delivery_profile_ref,
            audit=_audit(),
        )


def test_harness_kernel_contract_fields_do_not_encode_trace_vertical() -> None:
    from eval_factory import blueprints, harness, team

    modules = (harness, team, blueprints)
    contract_types = {
        value
        for module in modules
        for value in vars(module).values()
        if isinstance(value, type)
        and hasattr(value, "model_fields")
        and value.__module__.startswith("eval_factory")
    }
    field_names = {
        field_name.casefold() for contract_type in contract_types for field_name in contract_type.model_fields
    }
    assert not any("trace" in field_name for field_name in field_names)
