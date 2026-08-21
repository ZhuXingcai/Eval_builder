from __future__ import annotations

from typing import ClassVar, Literal, Self

from pydantic import Field, model_validator

from eval_factory.contracts.core import ContractAudit, Identifier, ObjectRef
from eval_factory.contracts.core_v2 import ContractModelV2
from eval_factory.harness.artifacts import ArtifactModalityV1
from eval_factory.harness.contracts import (
    HarnessObjectV1,
    require_ref,
    require_sorted_unique,
    require_sorted_unique_refs,
    sorted_refs,
)


class ArtifactSchemaBindingV1(ContractModelV2):
    schema_version: Literal["eval-harness/artifact-schema-binding/v1"] = (
        "eval-harness/artifact-schema-binding/v1"
    )
    semantic_role: Identifier
    schema_ref: ObjectRef
    modality: ArtifactModalityV1
    media_types: tuple[str, ...] = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def validate_binding(self) -> Self:
        if self.schema_ref.object_type != "json-schema":
            raise ValueError("artifact schema binding requires a JSON schema ref")
        require_sorted_unique(self.media_types, "media_types")
        return self


class BlueprintCapabilityBindingV1(ContractModelV2):
    schema_version: Literal["eval-harness/blueprint-capability-binding/v1"] = (
        "eval-harness/blueprint-capability-binding/v1"
    )
    capability_id: Identifier
    capability_definition_ref: ObjectRef
    provider_binding_ref: ObjectRef
    input_schema_refs: tuple[ObjectRef, ...] = Field(default=(), max_length=256)
    output_schema_refs: tuple[ObjectRef, ...] = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def validate_binding(self) -> Self:
        require_ref(
            self.capability_definition_ref,
            "harness-capability-definition",
            "capability_definition_ref",
        )
        require_ref(
            self.provider_binding_ref,
            "harness-capability-provider",
            "provider_binding_ref",
        )
        require_sorted_unique_refs(self.input_schema_refs, "input_schema_refs")
        require_sorted_unique_refs(self.output_schema_refs, "output_schema_refs")
        if any(
            ref.object_type != "json-schema" for ref in (*self.input_schema_refs, *self.output_schema_refs)
        ):
            raise ValueError("capability schema bindings must reference JSON schemas")
        return self


class BlueprintDataFlowEdgeV1(ContractModelV2):
    schema_version: Literal["eval-harness/blueprint-data-flow-edge/v1"] = (
        "eval-harness/blueprint-data-flow-edge/v1"
    )
    edge_id: Identifier
    producer_capability_id: Identifier
    consumer_capability_id: Identifier
    artifact_schema_ref: ObjectRef

    @model_validator(mode="after")
    def validate_edge(self) -> Self:
        if self.producer_capability_id == self.consumer_capability_id:
            raise ValueError("Blueprint data-flow edge cannot be self-referential")
        if self.artifact_schema_ref.object_type != "json-schema":
            raise ValueError("data-flow edge requires a JSON schema ref")
        return self


class BlueprintBudgetV1(ContractModelV2):
    schema_version: Literal["eval-harness/blueprint-budget/v1"] = "eval-harness/blueprint-budget/v1"
    max_model_requests: int = Field(ge=0, le=10_000_000)
    max_model_tokens: int = Field(ge=0, le=10_000_000_000)
    max_cost_micro_usd: int = Field(ge=0, le=10_000_000_000_000)
    max_parallel_tasks: int = Field(ge=1, le=1_000)


class EvaluationBlueprintV1(HarnessObjectV1):
    schema_version: Literal["eval-harness/evaluation-blueprint/v1"] = "eval-harness/evaluation-blueprint/v1"
    OBJECT_TYPE: ClassVar[str] = "evaluation-blueprint"

    blueprint_id: Identifier
    blueprint_version: int = Field(ge=1, le=1_000_000_000)
    requirement_ref: ObjectRef
    composition_ref: ObjectRef
    pack_manifest_refs: tuple[ObjectRef, ...] = Field(min_length=1, max_length=64)
    artifact_schema_bindings: tuple[ArtifactSchemaBindingV1, ...] = Field(
        min_length=1,
        max_length=1_000,
    )
    capability_bindings: tuple[BlueprintCapabilityBindingV1, ...] = Field(
        min_length=1,
        max_length=1_000,
    )
    data_flow_edges: tuple[BlueprintDataFlowEdgeV1, ...] = Field(
        default=(),
        max_length=10_000,
    )
    team_ref: ObjectRef
    roster_ref: ObjectRef
    task_graph_ref: ObjectRef
    authority_ref: ObjectRef
    graph_template_ref: ObjectRef
    execution_provider_refs: tuple[ObjectRef, ...] = Field(
        default=(),
        max_length=256,
    )
    evaluator_refs: tuple[ObjectRef, ...] = Field(default=(), max_length=256)
    review_gate_refs: tuple[ObjectRef, ...] = Field(default=(), max_length=256)
    budget: BlueprintBudgetV1
    delivery_profile_ref: ObjectRef

    @classmethod
    def create(  # type: ignore[override]
        cls,
        *,
        blueprint_id: str,
        blueprint_version: int,
        requirement_ref: ObjectRef,
        composition_ref: ObjectRef,
        pack_manifest_refs: tuple[ObjectRef, ...],
        artifact_schema_bindings: tuple[ArtifactSchemaBindingV1, ...],
        capability_bindings: tuple[BlueprintCapabilityBindingV1, ...],
        data_flow_edges: tuple[BlueprintDataFlowEdgeV1, ...],
        team_ref: ObjectRef,
        roster_ref: ObjectRef,
        task_graph_ref: ObjectRef,
        authority_ref: ObjectRef,
        graph_template_ref: ObjectRef,
        execution_provider_refs: tuple[ObjectRef, ...],
        evaluator_refs: tuple[ObjectRef, ...],
        review_gate_refs: tuple[ObjectRef, ...],
        budget: BlueprintBudgetV1,
        delivery_profile_ref: ObjectRef,
        audit: ContractAudit,
    ) -> EvaluationBlueprintV1:
        return super().create(
            audit=audit,
            blueprint_id=blueprint_id,
            blueprint_version=blueprint_version,
            requirement_ref=requirement_ref,
            composition_ref=composition_ref,
            pack_manifest_refs=sorted_refs(pack_manifest_refs),
            artifact_schema_bindings=tuple(
                sorted(
                    artifact_schema_bindings,
                    key=lambda value: value.semantic_role,
                )
            ),
            capability_bindings=tuple(
                sorted(
                    capability_bindings,
                    key=lambda value: value.capability_id,
                )
            ),
            data_flow_edges=tuple(sorted(data_flow_edges, key=lambda value: value.edge_id)),
            team_ref=team_ref,
            roster_ref=roster_ref,
            task_graph_ref=task_graph_ref,
            authority_ref=authority_ref,
            graph_template_ref=graph_template_ref,
            execution_provider_refs=sorted_refs(execution_provider_refs),
            evaluator_refs=sorted_refs(evaluator_refs),
            review_gate_refs=sorted_refs(review_gate_refs),
            budget=budget,
            delivery_profile_ref=delivery_profile_ref,
        )

    @model_validator(mode="after")
    def validate_blueprint(self) -> Self:
        require_ref(
            self.composition_ref,
            "harness-composition",
            "composition_ref",
        )
        require_sorted_unique_refs(
            self.pack_manifest_refs,
            "pack_manifest_refs",
        )
        if any(ref.object_type != "eval-pack-manifest" for ref in self.pack_manifest_refs):
            raise ValueError("Blueprint Packs must reference Eval Pack manifests")
        roles = tuple(binding.semantic_role for binding in self.artifact_schema_bindings)
        require_sorted_unique(roles, "artifact semantic roles")
        capability_ids = tuple(binding.capability_id for binding in self.capability_bindings)
        require_sorted_unique(capability_ids, "capability IDs")
        edge_ids = tuple(edge.edge_id for edge in self.data_flow_edges)
        require_sorted_unique(edge_ids, "data-flow edge IDs")
        _validate_data_flow(self.capability_bindings, self.data_flow_edges)
        require_ref(self.team_ref, "agent-team", "team_ref")
        require_ref(self.roster_ref, "team-roster", "roster_ref")
        require_ref(self.task_graph_ref, "team-task-graph", "task_graph_ref")
        require_ref(self.authority_ref, "execution-authority", "authority_ref")
        require_ref(
            self.graph_template_ref,
            "graph-template",
            "graph_template_ref",
        )
        for values, label in (
            (self.execution_provider_refs, "execution_provider_refs"),
            (self.evaluator_refs, "evaluator_refs"),
            (self.review_gate_refs, "review_gate_refs"),
        ):
            require_sorted_unique_refs(values, label)
        require_ref(
            self.delivery_profile_ref,
            "delivery-profile",
            "delivery_profile_ref",
        )
        return self


def _validate_data_flow(
    bindings: tuple[BlueprintCapabilityBindingV1, ...],
    edges: tuple[BlueprintDataFlowEdgeV1, ...],
) -> None:
    by_id = {binding.capability_id: binding for binding in bindings}
    dependencies: dict[str, set[str]] = {binding.capability_id: set() for binding in bindings}
    for edge in edges:
        producer = by_id.get(edge.producer_capability_id)
        consumer = by_id.get(edge.consumer_capability_id)
        if producer is None or consumer is None:
            raise ValueError("data-flow edge references an unknown capability")
        if edge.artifact_schema_ref not in producer.output_schema_refs:
            raise ValueError("data-flow schema is not produced by the source capability")
        if edge.artifact_schema_ref not in consumer.input_schema_refs:
            raise ValueError("data-flow schema is not consumed by the target capability")
        dependencies[consumer.capability_id].add(producer.capability_id)
    ready = sorted(capability_id for capability_id, values in dependencies.items() if not values)
    visited: list[str] = []
    while ready:
        capability_id = ready.pop(0)
        visited.append(capability_id)
        for candidate_id in sorted(dependencies):
            if capability_id in dependencies[candidate_id]:
                dependencies[candidate_id].remove(capability_id)
                if not dependencies[candidate_id] and candidate_id not in visited:
                    ready.append(candidate_id)
                    ready.sort()
    if len(visited) != len(bindings):
        raise ValueError("Blueprint capability data flow must be acyclic")


__all__ = [
    "ArtifactSchemaBindingV1",
    "BlueprintBudgetV1",
    "BlueprintCapabilityBindingV1",
    "BlueprintDataFlowEdgeV1",
    "EvaluationBlueprintV1",
]
