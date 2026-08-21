from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from env_mock_agent.facade.contracts import FacadeObjectRef
from env_mock_agent.facade.execution_adapter import (
    MappingAttachmentExecutionMaterialResolver,
    ProviderExecutionMaterial,
    RegistryAttachmentExecutionFacade,
    world_ledger_object_ref,
)
from env_mock_agent.facade.execution_v2 import (
    ARTIFACT_EXECUTION_POLICY_VERSION,
    AttachmentExecutionRequestV2,
    AttachmentExecutionResultV2,
    AttachmentExecutionRouteKindV2,
    AttachmentExecutionStatusV2,
    WorldLedgerSnapshotRequestV2,
    WorldLedgerSnapshotV2,
    attachment_execution_request_ref,
    attachment_execution_result_carried_sha256,
    validate_attachment_execution_request_identity,
)
from env_mock_agent.facade.resource_v2 import (
    AttachmentResourceGrantV2,
    AttachmentResourceTerminationRequestV2,
    AttachmentResourceTerminationResultV2,
    AttachmentResourceUsageV2,
    ResourceBoundAttachmentExecutionResultV2,
    attachment_resource_grant_ref,
    validate_attachment_resource_grant_identity,
)
from env_mock_agent.facade.routing_v2 import (
    RegistryAttachmentRoutingFacade,
)
from env_mock_agent.facade.telemetry_v2 import ExecutionTelemetryV2
from env_mock_agent.facade.validation_adapter import (
    ProviderValidationMaterial,
    RegistryAttachmentValidationFacade,
    StagingAttachmentValidationMaterialResolver,
)
from env_mock_agent.facade.validation_v2 import (
    AttachmentValidationFacade,
)
from env_mock_agent.providers import ProviderRegistry, TextProvider
from env_mock_agent.runtimes import RuntimeRegistry
from env_mock_agent.runtimes.security import is_path_inside
from env_mock_agent.schemas import ArtifactPlan, WorldLedger


@dataclass(frozen=True)
class CanaryTextExecutionFixture:
    artifact_id: str
    dependency_id: str
    relative_path: str
    content: str
    validators: tuple[str, ...]
    producer_task_view_ref: FacadeObjectRef
    build_spec_ref: FacadeObjectRef
    content_contract_ref: FacadeObjectRef
    render_contract_ref: FacadeObjectRef
    provider_payload_ref: FacadeObjectRef
    authorized_evidence_ref_ids: tuple[str, ...]


class _CanaryTextExecutionFacade:
    def __init__(
        self,
        *,
        delegate: RegistryAttachmentExecutionFacade,
        fixture: CanaryTextExecutionFixture,
        staging_root: Path,
        clock: Callable[[], datetime],
    ) -> None:
        self._delegate = delegate
        self._fixture = fixture
        self._staging_root = staging_root.expanduser().resolve()
        self._clock = clock

    async def snapshot_world(
        self,
        request: WorldLedgerSnapshotRequestV2,
    ) -> WorldLedgerSnapshotV2:
        return await self._delegate.snapshot_world(request)

    async def execute(
        self,
        request: AttachmentExecutionRequestV2,
    ) -> AttachmentExecutionResultV2:
        return await self._delegate.execute(request)

    async def execute_with_resources(
        self,
        request: AttachmentExecutionRequestV2,
        grant: AttachmentResourceGrantV2,
    ) -> ResourceBoundAttachmentExecutionResultV2:
        return await self._delegate.execute_with_resources(
            request,
            grant,
        )

    async def terminate_resources(
        self,
        request: AttachmentResourceTerminationRequestV2,
    ) -> AttachmentResourceTerminationResultV2:
        return await self._delegate.terminate_resources(request)

    def recover_existing_output(
        self,
        request: AttachmentExecutionRequestV2,
        grant: AttachmentResourceGrantV2,
    ) -> ResourceBoundAttachmentExecutionResultV2 | None:
        validate_attachment_execution_request_identity(request)
        validate_attachment_resource_grant_identity(grant)
        request_ref = attachment_execution_request_ref(request)
        implementation = f"{TextProvider.__module__}.{TextProvider.__qualname__}"
        fixture = self._fixture
        if (
            grant.operation_ref != request_ref
            or request.selected_route_kind is not AttachmentExecutionRouteKindV2.PROVIDER
            or request.selected_route_id != TextProvider.name
            or request.selected_route_version != implementation
            or request.artifact_id != fixture.artifact_id
            or request.relative_path != fixture.relative_path
            or request.producer_task_view_ref != fixture.producer_task_view_ref
            or request.build_spec_ref != fixture.build_spec_ref
            or request.content_contract_ref != fixture.content_contract_ref
            or request.render_contract_ref != fixture.render_contract_ref
            or request.provider_payload_ref != fixture.provider_payload_ref
        ):
            raise ValueError("existing canary output request binding is invalid")
        staging = (
            self._staging_root / hashlib.sha256(request.execution_request_id.encode("utf-8")).hexdigest()
        ).resolve()
        candidate = staging / request.relative_path
        target = candidate.resolve()
        if not is_path_inside(staging, target):
            raise ValueError("existing canary output path is invalid")
        if not target.exists():
            return None
        expected = (fixture.content.rstrip() + "\n").encode("utf-8")
        if candidate.is_symlink() or not target.is_file():
            raise ValueError("existing canary output differs from the exact fixture")
        with target.open("rb") as handle:
            observed = handle.read(len(expected) + 1)
        if observed != expected:
            raise ValueError("existing canary output differs from the exact fixture")
        digest = hashlib.sha256(expected).hexdigest()
        telemetry = ExecutionTelemetryV2.unavailable(
            observed_at=self._clock(),
        )
        result = AttachmentExecutionResultV2(
            execution_result_id="attachment-execution-result://pending",
            execution_request_ref=request_ref,
            artifact_id=request.artifact_id,
            attempt=request.attempt,
            status=AttachmentExecutionStatusV2.SUCCEEDED,
            world_ledger_snapshot_ref=request.world_ledger_snapshot_ref,
            selected_route_kind=request.selected_route_kind,
            selected_route_id=request.selected_route_id,
            worker_version=implementation,
            output_ref=FacadeObjectRef(
                object_type="attachment-output",
                object_id=f"attachment-output://sha256/{digest}",
                object_version="v2",
                object_sha256=digest,
            ),
            output_sha256=digest,
            retryable=False,
            failure_code=None,
            telemetry=telemetry,
            policy_version=ARTIFACT_EXECUTION_POLICY_VERSION,
            execution_result_sha256="0" * 64,
        )
        result_digest = attachment_execution_result_carried_sha256(result)
        result = result.model_copy(
            update={
                "execution_result_id": (f"attachment-execution-result://sha256/{result_digest}"),
                "execution_result_sha256": result_digest,
            }
        )
        usage = AttachmentResourceUsageV2(
            process_starts=0,
            renderer_operations=0,
            network_requests=0,
            retained_storage_bytes=len(expected),
        )
        grant.validate_usage(usage)
        return ResourceBoundAttachmentExecutionResultV2(
            resource_grant_ref=attachment_resource_grant_ref(grant),
            execution_result=result,
            usage=usage,
        )


class CanaryTextAttachmentFacades:
    """Provider-owned physical resources for one deterministic text canary."""

    def __init__(
        self,
        *,
        fixture: CanaryTextExecutionFixture,
        staging_root: Path,
        clock: Callable[[], datetime],
    ) -> None:
        plan = ArtifactPlan(
            artifact_id=fixture.artifact_id,
            dependency_id=fixture.dependency_id,
            relative_path=fixture.relative_path,
            asset_type="txt",
            provider="text",
            content_contract={"content": fixture.content},
            render_contract={},
            validators=list(fixture.validators),
            source_evidence_ids=list(fixture.authorized_evidence_ref_ids),
        )
        ledger = WorldLedger()
        self.world_ledger_ref = world_ledger_object_ref(
            "world-ledger://r6-canary/text",
            ledger,
        )
        providers = ProviderRegistry()
        providers.register(TextProvider())
        self._plan = plan
        self._staging_root = staging_root
        execution = RegistryAttachmentExecutionFacade(
            providers,
            RuntimeRegistry(),
            resolver=MappingAttachmentExecutionMaterialResolver(
                world_ledgers={self.world_ledger_ref.object_id: ledger},
                provider_materials={
                    fixture.build_spec_ref.object_id: (
                        ProviderExecutionMaterial(
                            producer_task_view_ref=(fixture.producer_task_view_ref),
                            content_contract_ref=(fixture.content_contract_ref),
                            render_contract_ref=(fixture.render_contract_ref),
                            provider_payload_ref=(fixture.provider_payload_ref),
                            authorized_evidence_ref_ids=(fixture.authorized_evidence_ref_ids),
                            plan=plan,
                        )
                    )
                },
            ),
            staging_root=staging_root,
            clock=clock,
        )
        self.execution = _CanaryTextExecutionFacade(
            delegate=execution,
            fixture=fixture,
            staging_root=staging_root,
            clock=clock,
        )

    def validation_facade(
        self,
        requests: tuple[AttachmentExecutionRequestV2, ...],
    ) -> AttachmentValidationFacade:
        return RegistryAttachmentValidationFacade(
            resolver=StagingAttachmentValidationMaterialResolver(
                staging_root=self._staging_root,
                execution_requests={request.execution_request_id: request for request in requests},
                materials={
                    request.build_spec_ref.object_id: (ProviderValidationMaterial(plan=self._plan))
                    for request in requests
                },
            )
        )


def create_canary_text_routing_facade(
    *,
    clock: Callable[[], datetime],
) -> RegistryAttachmentRoutingFacade:
    providers = ProviderRegistry()
    providers.register(TextProvider())
    return RegistryAttachmentRoutingFacade(
        providers,
        RuntimeRegistry(),
        clock=clock,
    )
