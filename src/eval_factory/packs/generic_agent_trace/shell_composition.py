from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from eval_factory.agent_system.dataset_runtime import (
    FactoryDatasetCoreInput,
)
from eval_factory.agent_system.dataset_runtime_fixture import (
    FactoryDatasetFixtureComponents,
)
from eval_factory.console_api.composition_types import (
    AgentShellCompositionBlockedError,
    AgentShellCompositionConflictError,
    AgentShellCompositionPreflight,
)
from eval_factory.contracts.core import ContractAudit, Identifier, ObjectRef
from eval_factory.contracts.core_v2 import ContractModelV2
from eval_factory.contracts.dataset_runtime_v2 import (
    FactoryDatasetRunViewV2,
)
from eval_factory.harness.contracts import (
    require_sorted_unique,
    require_sorted_unique_refs,
    sorted_refs,
    static_object_ref,
)
from eval_factory.harness.graph_models import (
    HarnessGraphExecutionBindingV1,
)
from eval_factory.harness.runtime_models import (
    HarnessSessionProjectionV1,
    HarnessSessionStatusV1,
    HarnessTurnOutcomeV1,
    HarnessTurnResultV1,
)
from eval_factory.harness.session_store import (
    HarnessSessionNotFoundError,
    HarnessSessionStore,
)
from eval_factory.harness.source_admission import (
    HarnessSourceAdmissionNotFoundError,
    HarnessSourceAdmissionStore,
    HarnessSourceAdmissionWrite,
)
from eval_factory.packs.generic_agent_trace.product_builder import (
    GenericAgentFirstPartyGraphBuilder,
)
from eval_factory.packs.generic_agent_trace.product_runtime import (
    GenericAgentGraphProductPaths,
    GenericAgentGraphRuntimeConfigV1,
)
from eval_factory.packs.generic_agent_trace.requirement_bridge import (
    GenericAgentRequirementBridge,
    GenericAgentRequirementBridgeInputV1,
    GenericAgentRequirementBridgeResultV1,
    RequirementBridgeOutcomeV1,
)


class GenericAgentShellCompositionConfigV1(ContractModelV2):
    schema_version: Literal["generic-agent-trace/shell-composition-config/v1"] = (
        "generic-agent-trace/shell-composition-config/v1"
    )
    claim_scope: Literal["DEVELOPMENT_FIXTURE_ONLY"] = "DEVELOPMENT_FIXTURE_ONLY"
    allowed_task_kinds: tuple[Identifier, ...] = Field(
        min_length=1,
        max_length=256,
    )
    pipeline_policy_refs: tuple[ObjectRef, ...] = Field(
        min_length=1,
        max_length=128,
    )
    gateway_registry_refs: tuple[ObjectRef, ...] = Field(
        min_length=1,
        max_length=32,
    )
    output_target_ref: ObjectRef
    max_transitions: int = Field(ge=1, le=100_000)
    max_plan_revisions: int = Field(ge=0, le=1_000)
    max_agent_attempts: int = Field(ge=1, le=100)

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        require_sorted_unique(
            self.allowed_task_kinds,
            "allowed_task_kinds",
        )
        require_sorted_unique_refs(
            self.pipeline_policy_refs,
            "pipeline_policy_refs",
        )
        require_sorted_unique_refs(
            self.gateway_registry_refs,
            "gateway_registry_refs",
        )
        if any(
            not value.object_type.endswith("-policy") or value.object_version != "v2"
            for value in self.pipeline_policy_refs
        ):
            raise ValueError("shell pipeline policy refs are invalid")
        if any(
            value.object_type
            not in {
                "agent-registry",
                "model-catalog",
                "prompt-registry",
                "rag-registry",
            }
            or value.object_version != "v2"
            for value in self.gateway_registry_refs
        ):
            raise ValueError("shell gateway registry refs are invalid")
        if (
            self.output_target_ref.object_type != "candidate-output-target"
            or self.output_target_ref.object_version != "v2"
        ):
            raise ValueError("shell output target ref is invalid")
        return self


class GenericAgentFixtureShellGraphStarter:
    """Builds the exact Pack 1.2.0 fixture Graph from Harness authority."""

    def __init__(
        self,
        *,
        components: FactoryDatasetFixtureComponents,
        session_store: HarnessSessionStore,
        source_store: HarnessSourceAdmissionStore,
        config: GenericAgentShellCompositionConfigV1,
        paths: GenericAgentGraphProductPaths,
        candidate_output_root: Path,
    ) -> None:
        self.components = components
        self.session_store = session_store
        self.source_store = source_store
        self.config = config
        self.paths = paths.resolved()
        self.candidate_output_root = candidate_output_root.expanduser().resolve()

    def preflight(
        self,
        session_id: str,
    ) -> AgentShellCompositionPreflight:
        try:
            bridge, _source = self._bridge(
                session_id,
                audit=None,
            )
        except HarnessSessionNotFoundError:
            return AgentShellCompositionPreflight(
                ready=False,
                reason_codes=("REQUIREMENT_NOT_READY",),
            )
        except HarnessSourceAdmissionNotFoundError:
            return AgentShellCompositionPreflight(
                ready=False,
                reason_codes=("SOURCE_NOT_ADMITTED",),
            )
        except AgentShellCompositionBlockedError as exc:
            return AgentShellCompositionPreflight(
                ready=False,
                reason_codes=exc.reason_codes,
            )
        if bridge.outcome is not RequirementBridgeOutcomeV1.SUCCEEDED:
            return AgentShellCompositionPreflight(
                ready=False,
                reason_codes=tuple(value.value for value in bridge.failure_codes),
            )
        return AgentShellCompositionPreflight(
            ready=True,
            reason_codes=(),
        )

    async def start(
        self,
        session_id: str,
        *,
        turn: HarnessTurnResultV1,
        principal: str,
        audit: ContractAudit,
    ) -> FactoryDatasetRunViewV2:
        bridge, source = self._bridge(
            session_id,
            audit=audit,
        )
        if bridge.outcome is not RequirementBridgeOutcomeV1.SUCCEEDED:
            raise AgentShellCompositionBlockedError(
                tuple(value.value for value in bridge.failure_codes),
            )
        assert bridge.requirement is not None
        assert bridge.factory_policy is not None
        assert bridge.run_request is not None
        current = self.session_store.get_projection(session_id)
        if (
            current.session.status is not HarnessSessionStatusV1.ACTIVE
            or turn.outcome is not HarnessTurnOutcomeV1.READY
            or turn.interpretation_ref != current.session.current_interpretation_ref
            or turn.requirement_policy_ref != current.session.current_requirement_policy_ref
            or self.components.runtime.requested_by != principal
        ):
            raise AgentShellCompositionConflictError(
                "shell Graph start uses stale session or principal authority",
            )
        return await self._advance_graph(
            bridge=bridge,
            source=source,
            current=current,
            principal=principal,
            audit=audit,
        )

    async def advance(
        self,
        session_id: str,
        *,
        binding: HarnessGraphExecutionBindingV1,
        audit: ContractAudit,
    ) -> FactoryDatasetRunViewV2:
        self.current_view(
            session_id,
            binding=binding,
        )
        stable_audit = binding.audit
        bridge, source = self._bridge(
            session_id,
            audit=stable_audit,
        )
        if (
            bridge.outcome is not RequirementBridgeOutcomeV1.SUCCEEDED
            or bridge.requirement is None
            or bridge.factory_policy is None
            or bridge.run_request is None
        ):
            raise AgentShellCompositionConflictError(
                "shell Graph advance uses stale compiled authority",
            )
        current = self.session_store.get_projection(session_id)
        return await self._advance_graph(
            bridge=bridge,
            source=source,
            current=current,
            principal=self.components.runtime.requested_by,
            audit=stable_audit,
        )

    async def _advance_graph(
        self,
        *,
        bridge: GenericAgentRequirementBridgeResultV1,
        source: HarnessSourceAdmissionWrite,
        current: HarnessSessionProjectionV1,
        principal: str,
        audit: ContractAudit,
    ) -> FactoryDatasetRunViewV2:
        assert bridge.requirement is not None
        assert bridge.factory_policy is not None
        assert bridge.run_request is not None
        graph_config = self._graph_config(current.session.session_ref)
        execution = self.source_store.materialize_for_execution(
            current.session.session_id,
        )
        self._ensure_member_sessions(
            graph_config=graph_config,
            main=current,
            principal=principal,
            audit=audit,
        )
        core_input = FactoryDatasetCoreInput(
            manifest_path=execution.manifest_path,
            raw_root=execution.raw_root,
            expected_manifest_sha256=source.admission.manifest_sha256,
        )
        product = GenericAgentFirstPartyGraphBuilder.build(
            components=self.components,
            session_store=self.session_store,
            config=graph_config,
            paths=self.paths,
            core_input=core_input,
            request=bridge.run_request,
            policy=bridge.factory_policy,
            requirement=bridge.requirement,
            candidate_output_root=self.candidate_output_root,
            audit=audit,
            source_ref_overrides={
                member.relative_name: member.source_ref for member in source.admission.members
            },
        )
        return await product.advance()

    def current_view(
        self,
        session_id: str,
        *,
        binding: HarnessGraphExecutionBindingV1,
    ) -> FactoryDatasetRunViewV2:
        projection = self.session_store.get_projection(session_id)
        if projection.session.session_ref != binding.session_ref:
            raise AgentShellCompositionConflictError(
                "shell Graph binding belongs to another session",
            )
        try:
            bridge, _source = self._bridge(
                session_id,
                audit=None,
            )
        except (
            AgentShellCompositionBlockedError,
            HarnessSessionNotFoundError,
            HarnessSourceAdmissionNotFoundError,
        ) as exc:
            raise AgentShellCompositionConflictError(
                "shell Graph binding uses stale Harness source authority",
            ) from exc
        if (
            bridge.outcome is not RequirementBridgeOutcomeV1.SUCCEEDED
            or bridge.requirement is None
            or bridge.factory_policy is None
            or bridge.run_request is None
            or binding.requirement_ref != bridge.requirement.to_ref()
            or binding.factory_policy_ref != bridge.factory_policy.to_ref()
            or binding.factory_request_ref != bridge.run_request.to_ref()
        ):
            raise AgentShellCompositionConflictError(
                "shell Graph binding uses stale compiled authority",
            )
        request = self.components.runtime.store.get_dataset_request_by_ref(
            binding.factory_request_ref,
        )
        return self.components.runtime.current_view(
            request=request,
            audit=request.audit,
        )

    def _bridge(
        self,
        session_id: str,
        *,
        audit: ContractAudit | None,
    ) -> tuple[
        GenericAgentRequirementBridgeResultV1,
        HarnessSourceAdmissionWrite,
    ]:
        projection = self.session_store.get_projection(session_id)
        interpretation, policy = self.session_store.get_current_requirement(
            session_id,
        )
        source = self.source_store.get_for_session(session_id)
        message = self.session_store.get_message(
            interpretation.user_message_ref,
        )
        if not set(source.admission.artifact_envelope_refs).issubset(
            message.artifact_envelope_refs,
        ):
            raise AgentShellCompositionBlockedError(
                ("SOURCE_CONFIRMATION_REQUIRED",),
            )
        effective_audit = audit or interpretation.audit
        session_digest = projection.session.session_ref.object_sha256
        requirement_source_ref = static_object_ref(
            object_type="evaluation-requirement-source",
            object_id=(f"evaluation-requirement-source://harness/{session_digest}"),
            object_version="v2",
            payload={
                "interpretation_ref": interpretation.to_ref(),
                "user_message_ref": interpretation.user_message_ref,
            },
        )
        manifest_ref = ObjectRef(
            object_type="trace-manifest",
            object_id=(f"trace-manifest://sha256/{source.admission.manifest_sha256}"),
            object_version="v2",
            object_sha256=source.admission.manifest_sha256,
        )
        source_authorization_ref = static_object_ref(
            object_type="trace-source-authorization",
            object_id=(f"trace-source-authorization://harness/{source.admission.object_sha256}"),
            object_version="v2",
            payload={
                "admission_ref": source.admission.to_ref(),
                "source_refs": sorted_refs(member.source_ref for member in source.admission.members),
            },
        )
        bridge_input = GenericAgentRequirementBridgeInputV1.create(
            audit=effective_audit,
            interpretation=interpretation,
            requirement_policy=policy,
            dataset_run_id=f"dataset-run.shell.{session_digest[:32]}",
            requirement_source_ref=requirement_source_ref,
            manifest_ref=manifest_ref,
            source_authorization_ref=source_authorization_ref,
            allowed_task_kinds=self.config.allowed_task_kinds,
            pipeline_policy_refs=self.config.pipeline_policy_refs,
            gateway_registry_refs=self.config.gateway_registry_refs,
            output_target_ref=self.config.output_target_ref,
            max_transitions=self.config.max_transitions,
            max_plan_revisions=self.config.max_plan_revisions,
            max_agent_attempts=self.config.max_agent_attempts,
            idempotency_key=f"shell-bridge.{session_digest[:32]}",
        )
        return (
            GenericAgentRequirementBridge().compile(
                bridge_input,
                audit=effective_audit,
            ),
            source,
        )

    def _graph_config(
        self,
        session_ref: ObjectRef,
    ) -> GenericAgentGraphRuntimeConfigV1:
        suffix = session_ref.object_sha256[:24]
        return GenericAgentGraphRuntimeConfigV1(
            main_session_id=self.session_store.get_projection_by_ref(
                session_ref,
            ).session.session_id,
            control_session_id=f"shell-{suffix}-control",
            coordinator_session_id=f"shell-{suffix}-coordinator",
            quality_session_id=f"shell-{suffix}-quality",
            requirement_session_id=f"shell-{suffix}-requirement",
            task_session_id=f"shell-{suffix}-task",
            trace_session_id=f"shell-{suffix}-trace",
            team_id=f"team-shell-{suffix}",
            team_incarnation_id=f"team-shell-{suffix}-incarnation",
            graph_binding_id=f"graph-shell-{suffix}",
            thread_id=f"thread-shell-{suffix}",
        )

    def _ensure_member_sessions(
        self,
        *,
        graph_config: GenericAgentGraphRuntimeConfigV1,
        main: HarnessSessionProjectionV1,
        principal: str,
        audit: ContractAudit,
    ) -> None:
        from eval_factory.packs.generic_agent_trace.manifest import (
            build_generic_agent_trace_execution_pack,
        )

        registration = build_generic_agent_trace_execution_pack(
            audit=audit,
        )
        if main.session.composition_ref != registration.composition.to_ref():
            raise AgentShellCompositionConflictError(
                "shell session is not bound to Pack 1.2.0",
            )
        stable_audit = ContractAudit(
            created_at=main.session.created_at,
            created_by=principal,
            governing_versions=audit.governing_versions,
            input_refs=(registration.composition.to_ref(),),
        )
        for role, member_session_id in graph_config.member_session_ids().items():
            self.session_store.create_session(
                session_id=member_session_id,
                incarnation_id=f"{member_session_id}-incarnation",
                composition_ref=registration.composition.to_ref(),
                created_by=principal,
                idempotency_key=f"create-shell-member-{role}-{member_session_id}",
                audit=stable_audit,
            )


__all__ = [
    "GenericAgentFixtureShellGraphStarter",
    "GenericAgentShellCompositionConfigV1",
]
