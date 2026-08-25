from __future__ import annotations

import hashlib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Literal, Self

from pydantic import model_validator

from eval_factory.agent_system.candidate_output import (
    CandidateDatasetOutputAssembler,
)
from eval_factory.agent_system.dataset_release_fixture import (
    FactoryDatasetReleaseRuntimeConfigV1,
)
from eval_factory.agent_system.dataset_runtime_fixture import (
    FactoryDatasetAttachmentRuntimeConfigV1,
    FactoryDatasetCoreRuntimeConfigV1,
    FactoryDatasetPlanningFixtureV1,
    FactoryDatasetR4RuntimeConfigV1,
    FactoryDatasetRuntimeConfigV1,
    build_fixture_dataset_components,
)
from eval_factory.agent_system.dataset_specialist_fixture import (
    FactoryDatasetSpecialistRuntimeConfigV1,
)
from eval_factory.agent_system.graph_journal import (
    FactoryGraphJournalStore,
)
from eval_factory.agent_system.plan_review import PlanReviewService
from eval_factory.console_api.agent_service import AgentShellService
from eval_factory.console_api.composition_service import (
    AgentShellCompositionService,
)
from eval_factory.console_api.source_admission import (
    HarnessSourceAdmissionService,
)
from eval_factory.contracts.core import ObjectRef
from eval_factory.contracts.core_v2 import ContractModelV2
from eval_factory.harness.requirement_fixture import (
    FixtureRequirementAgentConfigV1,
    build_fixture_requirement_session_service,
)
from eval_factory.harness.source_admission import (
    HarnessSourceAdmissionStore,
)
from eval_factory.packs.generic_agent_trace.manifest import (
    build_generic_agent_trace_execution_pack,
)
from eval_factory.packs.generic_agent_trace.product_runtime import (
    GenericAgentGraphProductPaths,
)
from eval_factory.packs.generic_agent_trace.shell_composition import (
    GenericAgentFixtureShellGraphStarter,
    GenericAgentShellCompositionConfigV1,
)
from eval_factory.team import TeamStore


class AgentShellFixtureHostConfigV1(ContractModelV2):
    schema_version: Literal["eval-factory/agent-shell-fixture-host-config/v1"] = (
        "eval-factory/agent-shell-fixture-host-config/v1"
    )
    claim_scope: Literal["DEVELOPMENT_FIXTURE_ONLY"] = "DEVELOPMENT_FIXTURE_ONLY"
    requirement_agent: FixtureRequirementAgentConfigV1
    dataset_runtime: FactoryDatasetRuntimeConfigV1
    planning_fixture: FactoryDatasetPlanningFixtureV1
    core_runtime: FactoryDatasetCoreRuntimeConfigV1
    r4_runtime: FactoryDatasetR4RuntimeConfigV1
    attachment_runtime: FactoryDatasetAttachmentRuntimeConfigV1
    specialist_runtime: FactoryDatasetSpecialistRuntimeConfigV1
    release_runtime: FactoryDatasetReleaseRuntimeConfigV1
    shell_composition: GenericAgentShellCompositionConfigV1
    trace_schema_ref: ObjectRef

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        if self.trace_schema_ref.object_type != "json-schema" or self.trace_schema_ref.object_version != "v1":
            raise ValueError(
                "Agent Shell fixture trace schema ref is invalid",
            )
        if self.release_runtime.output_target_ref != self.shell_composition.output_target_ref:
            raise ValueError(
                "Agent Shell fixture output target authority differs",
            )
        return self


@dataclass(frozen=True, slots=True)
class AgentShellFixtureHostPaths:
    factory_store: Path
    private_store: Path
    gateway_store: Path
    harness_store: Path
    source_store: Path
    team_store: Path
    capability_request_store: Path
    graph_journal: Path
    graph_checkpoint: Path
    core_workspace: Path
    attachment_workspace: Path
    job_store: Path
    specialist_workspace: Path
    candidate_output: Path

    def resolved(self) -> AgentShellFixtureHostPaths:
        return AgentShellFixtureHostPaths(
            **{field.name: getattr(self, field.name).expanduser().resolve() for field in fields(self)}
        )

    def validate(self) -> AgentShellFixtureHostPaths:
        resolved = self.resolved()
        values = tuple(getattr(resolved, field.name) for field in fields(resolved))
        if any(
            left == right or left in right.parents or right in left.parents
            for index, left in enumerate(values)
            for right in values[index + 1 :]
        ):
            raise ValueError(
                "Agent Shell fixture authorities and workspaces must not overlap",
            )
        return resolved


@dataclass(frozen=True, slots=True)
class AgentShellFixtureHost:
    shell: AgentShellService
    sources: HarnessSourceAdmissionService
    reviews: PlanReviewService


def build_fixture_agent_shell_host(
    *,
    paths: AgentShellFixtureHostPaths,
    config: AgentShellFixtureHostConfigV1,
    requested_by: str,
) -> AgentShellFixtureHost:
    resolved = paths.validate()
    sessions = build_fixture_requirement_session_service(
        harness_store_path=resolved.harness_store,
        private_store_path=resolved.private_store,
        gateway_store_path=resolved.gateway_store,
        config=config.requirement_agent,
    )
    source_store = HarnessSourceAdmissionStore(
        resolved.source_store,
    )
    components = build_fixture_dataset_components(
        factory_store_path=resolved.factory_store,
        private_store_path=resolved.private_store,
        gateway_store_path=resolved.gateway_store,
        config=config.dataset_runtime,
        fixture=config.planning_fixture,
        requested_by=requested_by,
        core_config=config.core_runtime,
        r4_config=config.r4_runtime,
        core_workspace_path=resolved.core_workspace,
        attachment_config=config.attachment_runtime,
        attachment_workspace_path=resolved.attachment_workspace,
        specialist_config=config.specialist_runtime,
        job_store_path=resolved.job_store,
        specialist_workspace_path=resolved.specialist_workspace,
        release_config=config.release_runtime,
        candidate_output_path=resolved.candidate_output,
    )
    graph_paths = GenericAgentGraphProductPaths(
        team_store=resolved.team_store,
        request_store=resolved.capability_request_store,
        graph_journal=resolved.graph_journal,
        graph_checkpoint=resolved.graph_checkpoint,
    )
    registration = build_generic_agent_trace_execution_pack(
        audit=config.requirement_agent.prompt.audit,
    )
    trace_provider_ref = next(
        value.to_ref()
        for value in registration.capability_definitions
        if value.capability_id == "capability.trace-ingestion"
    )
    source_service = HarnessSourceAdmissionService(
        sessions=sessions,
        store=source_store,
        trace_schema_ref=config.trace_schema_ref,
        producer_capability_ref=trace_provider_ref,
        governing_versions=(config.requirement_agent.prompt.audit.governing_versions),
    )
    graph_starter = GenericAgentFixtureShellGraphStarter(
        components=components,
        session_store=sessions.store,
        source_store=source_store,
        config=config.shell_composition,
        paths=graph_paths,
        candidate_output_root=resolved.candidate_output,
    )
    composition = AgentShellCompositionService(
        sessions=sessions,
        sources=source_store,
        team_store=TeamStore(resolved.team_store),
        factory_store=components.runtime.store,
        graph_journal=FactoryGraphJournalStore(
            resolved.graph_journal,
        ),
        plan_reviews=components.runtime.plan_reviews,
        candidate_output=CandidateDatasetOutputAssembler(
            store=components.runtime.store,
            root=resolved.candidate_output,
        ),
        graph_starter=graph_starter,
    )
    return AgentShellFixtureHost(
        shell=AgentShellService(
            sessions=sessions,
            composition_ref=registration.composition.to_ref(),
            principal_resolver=_principal_ref,
            governing_versions=(config.requirement_agent.prompt.audit.governing_versions),
            composition=composition,
        ),
        sources=source_service,
        reviews=components.runtime.plan_reviews,
    )


def _principal_ref(principal: str) -> ObjectRef:
    digest = hashlib.sha256(principal.encode()).hexdigest()
    return ObjectRef(
        object_type="principal",
        object_id=f"principal://sha256/{digest}",
        object_version="v1",
        object_sha256=digest,
    )


__all__ = [
    "AgentShellFixtureHost",
    "AgentShellFixtureHostConfigV1",
    "AgentShellFixtureHostPaths",
    "build_fixture_agent_shell_host",
]
