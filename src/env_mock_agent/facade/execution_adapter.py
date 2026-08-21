from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

from env_mock_agent.facade.contracts import FacadeObjectRef
from env_mock_agent.facade.execution_v2 import (
    ARTIFACT_EXECUTION_POLICY_VERSION,
    AttachmentExecutionFailureCodeV2,
    AttachmentExecutionRequestV2,
    AttachmentExecutionResultV2,
    AttachmentExecutionRouteKindV2,
    AttachmentExecutionStatusV2,
    WorldLedgerFactLockV2,
    WorldLedgerSnapshotRequestV2,
    WorldLedgerSnapshotV2,
    attachment_execution_request_ref,
    attachment_execution_result_carried_sha256,
    attachment_execution_result_ref,
    validate_attachment_execution_request_identity,
    validate_world_ledger_snapshot_request_identity,
    world_ledger_fact_id,
    world_ledger_snapshot_carried_sha256,
    world_ledger_snapshot_ref,
    world_ledger_snapshot_request_ref,
)
from env_mock_agent.facade.model_control_v2 import (
    ModelControlGrantV2,
    ModelControlReceiptOutcomeV2,
    ModelControlReceiptV2,
    ModelControlUsageSourceV2,
    ModelControlUsageV2,
    ModelDemandModeFacadeV2,
    ModelInvocationCancellationOutcomeV2,
    ModelInvocationCancellationRequestV2,
    ModelInvocationCancellationResultV2,
    ModelInvocationDescriptorV2,
    model_control_grant_ref,
    model_invocation_cancellation_request_ref,
    model_invocation_descriptor_ref,
    validate_model_control_grant_identity,
    validate_model_invocation_cancellation_request_identity,
    validate_model_invocation_descriptor_identity,
)
from env_mock_agent.facade.resource_v2 import (
    AttachmentResourceGrantV2,
    AttachmentResourceTerminationOutcomeV2,
    AttachmentResourceTerminationRequestV2,
    AttachmentResourceTerminationResultV2,
    AttachmentResourceUsageV2,
    ResourceBoundAttachmentExecutionResultV2,
    attachment_resource_grant_ref,
    attachment_resource_termination_request_ref,
    validate_attachment_resource_grant_identity,
    validate_attachment_resource_termination_request_identity,
)
from env_mock_agent.facade.telemetry_v2 import (
    ExecutionTelemetryV2,
    TelemetryAvailabilityV2,
    normalize_reported_cost_usd,
    normalize_tool_family_counts,
)
from env_mock_agent.providers import ProviderRegistry
from env_mock_agent.providers.base import ProviderRequest
from env_mock_agent.providers.helpers import sha256_path
from env_mock_agent.runtimes import RuntimeRegistry
from env_mock_agent.runtimes.base import AgentRuntime
from env_mock_agent.runtimes.security import is_path_inside
from env_mock_agent.schemas import (
    ArtifactPlan,
    ArtifactStatus,
    RuntimeErrorCode,
    RuntimeEvent,
    RuntimeEventType,
    RuntimeName,
    RuntimeRequest,
    SourceEvidence,
    WorldLedger,
)


class AttachmentExecutionMaterialError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        code: AttachmentExecutionFailureCodeV2,
    ) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class ProviderExecutionMaterial:
    producer_task_view_ref: FacadeObjectRef
    content_contract_ref: FacadeObjectRef
    render_contract_ref: FacadeObjectRef
    provider_payload_ref: FacadeObjectRef
    authorized_evidence_ref_ids: tuple[str, ...]
    plan: ArtifactPlan
    evidence: tuple[SourceEvidence, ...] = ()


@dataclass(frozen=True, slots=True)
class RuntimeExecutionMaterial:
    producer_task_view_ref: FacadeObjectRef
    content_contract_ref: FacadeObjectRef
    render_contract_ref: FacadeObjectRef
    model_profile_ref: FacadeObjectRef
    authorized_evidence_ref_ids: tuple[str, ...]
    request: RuntimeRequest


class AttachmentExecutionMaterialResolver(Protocol):
    def resolve_world_ledger(
        self,
        ref: FacadeObjectRef,
    ) -> WorldLedger: ...

    def resolve_provider_request(
        self,
        request: AttachmentExecutionRequestV2,
        *,
        staging_root: Path,
        world_ledger: WorldLedger,
    ) -> ProviderRequest: ...

    def resolve_runtime_request(
        self,
        request: AttachmentExecutionRequestV2,
        *,
        staging_root: Path,
        world_ledger: WorldLedger,
    ) -> RuntimeRequest: ...


class MappingAttachmentExecutionMaterialResolver:
    def __init__(
        self,
        *,
        world_ledgers: Mapping[str, WorldLedger],
        provider_materials: Mapping[str, ProviderExecutionMaterial] | None = None,
        runtime_materials: Mapping[str, RuntimeExecutionMaterial] | None = None,
    ) -> None:
        self._world_ledgers = dict(world_ledgers)
        self._provider_materials = dict(provider_materials or {})
        self._runtime_materials = dict(runtime_materials or {})

    def resolve_world_ledger(
        self,
        ref: FacadeObjectRef,
    ) -> WorldLedger:
        try:
            return self._world_ledgers[ref.object_id]
        except KeyError as exc:
            raise AttachmentExecutionMaterialError(
                "world ledger material is not registered",
                code=AttachmentExecutionFailureCodeV2.MATERIAL_NOT_FOUND,
            ) from exc

    def resolve_provider_request(
        self,
        request: AttachmentExecutionRequestV2,
        *,
        staging_root: Path,
        world_ledger: WorldLedger,
    ) -> ProviderRequest:
        try:
            material = self._provider_materials[request.build_spec_ref.object_id]
        except KeyError as exc:
            raise AttachmentExecutionMaterialError(
                "provider material is not registered",
                code=AttachmentExecutionFailureCodeV2.MATERIAL_NOT_FOUND,
            ) from exc
        expected_refs = (
            material.producer_task_view_ref,
            material.content_contract_ref,
            material.render_contract_ref,
            material.provider_payload_ref,
        )
        observed_refs = (
            request.producer_task_view_ref,
            request.content_contract_ref,
            request.render_contract_ref,
            request.provider_payload_ref,
        )
        if expected_refs != observed_refs or material.authorized_evidence_ref_ids != tuple(
            item.evidence_ref_id for item in request.evidence_grants
        ):
            raise AttachmentExecutionMaterialError(
                "provider material does not match execution request",
                code=AttachmentExecutionFailureCodeV2.MATERIAL_MISMATCH,
            )
        return ProviderRequest(
            plan=material.plan.model_copy(deep=True),
            staging_root=staging_root,
            evidence=[item.model_copy(deep=True) for item in material.evidence],
            world_ledger=world_ledger,
        )

    def resolve_runtime_request(
        self,
        request: AttachmentExecutionRequestV2,
        *,
        staging_root: Path,
        world_ledger: WorldLedger,
    ) -> RuntimeRequest:
        del world_ledger
        try:
            material = self._runtime_materials[request.build_spec_ref.object_id]
        except KeyError as exc:
            raise AttachmentExecutionMaterialError(
                "runtime material is not registered",
                code=AttachmentExecutionFailureCodeV2.MATERIAL_NOT_FOUND,
            ) from exc
        expected_refs = (
            material.producer_task_view_ref,
            material.content_contract_ref,
            material.render_contract_ref,
            material.model_profile_ref,
        )
        observed_refs = (
            request.producer_task_view_ref,
            request.content_contract_ref,
            request.render_contract_ref,
            request.model_profile_ref,
        )
        if expected_refs != observed_refs or material.authorized_evidence_ref_ids != tuple(
            item.evidence_ref_id for item in request.evidence_grants
        ):
            raise AttachmentExecutionMaterialError(
                "runtime material does not match execution request",
                code=AttachmentExecutionFailureCodeV2.MATERIAL_MISMATCH,
            )
        return material.request.model_copy(
            deep=True,
            update={
                "artifact_id": request.artifact_id,
                "workspace": str(staging_root),
            },
        )


@dataclass(frozen=True, slots=True)
class _SnapshotMaterial:
    world_ledger_ref: FacadeObjectRef
    ledger_sha256: str


class RegistryAttachmentExecutionFacade:
    def __init__(
        self,
        providers: ProviderRegistry,
        runtimes: RuntimeRegistry,
        *,
        resolver: AttachmentExecutionMaterialResolver,
        staging_root: Path,
        clock: Callable[[], datetime],
    ) -> None:
        self._providers = providers
        self._runtimes = runtimes
        self._resolver = resolver
        self._staging_root = staging_root.expanduser().resolve()
        self._clock = clock
        self._snapshot_cache: dict[
            str,
            tuple[str, WorldLedgerSnapshotV2],
        ] = {}
        self._snapshot_materials: dict[
            tuple[str, str, str, str],
            _SnapshotMaterial,
        ] = {}
        self._execution_cache: dict[
            str,
            tuple[str, AttachmentExecutionResultV2],
        ] = {}
        self._resource_execution_cache: dict[
            str,
            tuple[str, str, ResourceBoundAttachmentExecutionResultV2],
        ] = {}
        self._resource_handles: dict[
            str,
            tuple[AgentRuntime, str, str, int],
        ] = {}
        self._resource_stopped_handles: set[str] = set()
        self._model_execution_cache: dict[
            str,
            tuple[
                str,
                str,
                str,
                ResourceBoundAttachmentExecutionResultV2,
                ModelControlReceiptV2,
            ],
        ] = {}
        self._model_handles: dict[
            str,
            tuple[AgentRuntime, str, str, int],
        ] = {}
        self._model_stopped_handles: set[str] = set()
        self._model_cancellations: dict[
            str,
            ModelInvocationCancellationResultV2,
        ] = {}

    async def snapshot_world(
        self,
        request: WorldLedgerSnapshotRequestV2,
    ) -> WorldLedgerSnapshotV2:
        validate_world_ledger_snapshot_request_identity(request)
        cached = self._snapshot_cache.get(request.idempotency_key)
        if cached is not None:
            request_sha256, snapshot = cached
            if request_sha256 != request.snapshot_request_sha256:
                raise ValueError("world ledger snapshot idempotency conflict")
            return snapshot
        ledger = self._resolver.resolve_world_ledger(request.world_ledger_ref)
        if ledger.allowed_conflicts:
            raise ValueError("WorldLedger allowed conflicts are not supported")
        ledger_sha256 = _world_ledger_sha256(ledger)
        if request.world_ledger_ref.object_sha256 != ledger_sha256:
            raise ValueError("WorldLedger reference is stale or mismatched")
        facts_by_id: dict[str, tuple[str, object]] = {}
        for key, value in ledger.locked_facts.items():
            fact_id = world_ledger_fact_id(key)
            if fact_id in facts_by_id:
                raise ValueError("duplicate canonical WorldLedger fact key")
            facts_by_id[fact_id] = (key, value)
        locks: list[WorldLedgerFactLockV2] = []
        for fact_id in request.required_fact_ids:
            material = facts_by_id.get(fact_id)
            if material is None:
                raise ValueError(f"missing locked fact: {fact_id}")
            _, value = material
            value_sha256 = _value_sha256(value)
            locks.append(
                WorldLedgerFactLockV2(
                    fact_id=fact_id,
                    value_ref=FacadeObjectRef(
                        object_type="world-fact-value",
                        object_id=f"world-fact-value://sha256/{value_sha256}",
                        object_version="v2",
                        object_sha256=value_sha256,
                    ),
                    value_sha256=value_sha256,
                    source_refs=(),
                )
            )
        snapshot = WorldLedgerSnapshotV2(
            world_ledger_snapshot_id="world-ledger-snapshot://pending",
            snapshot_request_ref=world_ledger_snapshot_request_ref(request),
            world_ledger_ref=request.world_ledger_ref,
            fact_locks=tuple(locks),
            ledger_sha256=ledger_sha256,
            policy_version=ARTIFACT_EXECUTION_POLICY_VERSION,
            snapshotted_at=self._clock(),
            world_ledger_snapshot_sha256="0" * 64,
        )
        digest = world_ledger_snapshot_carried_sha256(snapshot)
        snapshot = snapshot.model_copy(
            update={
                "world_ledger_snapshot_id": f"world-ledger-snapshot://sha256/{digest}",
                "world_ledger_snapshot_sha256": digest,
            }
        )
        self._snapshot_cache[request.idempotency_key] = (
            request.snapshot_request_sha256,
            snapshot,
        )
        self._snapshot_materials[_ref_key(world_ledger_snapshot_ref(snapshot))] = _SnapshotMaterial(
            world_ledger_ref=request.world_ledger_ref,
            ledger_sha256=ledger_sha256,
        )
        return snapshot

    async def execute(
        self,
        request: AttachmentExecutionRequestV2,
    ) -> AttachmentExecutionResultV2:
        return await self._execute(request)

    async def _execute(
        self,
        request: AttachmentExecutionRequestV2,
        *,
        resource_grant: AttachmentResourceGrantV2 | None = None,
        resource_usage: dict[str, int] | None = None,
        model_grant: ModelControlGrantV2 | None = None,
        model_observation: dict[str, object] | None = None,
    ) -> AttachmentExecutionResultV2:
        validate_attachment_execution_request_identity(request)
        cached = self._execution_cache.get(request.idempotency_key)
        if cached is not None:
            request_sha256, result = cached
            if request_sha256 != request.execution_request_sha256:
                raise ValueError("attachment execution idempotency conflict")
            return result
        snapshot_material = self._snapshot_materials.get(_ref_key(request.world_ledger_snapshot_ref))
        if snapshot_material is None:
            return self._cache_result(
                request,
                self._failure_result(
                    request,
                    status=AttachmentExecutionStatusV2.BLOCKED_POLICY,
                    code=AttachmentExecutionFailureCodeV2.WORLD_LEDGER_STALE,
                ),
            )
        try:
            current_ledger = self._resolver.resolve_world_ledger(snapshot_material.world_ledger_ref)
        except AttachmentExecutionMaterialError as exc:
            return self._cache_result(
                request,
                self._failure_result(
                    request,
                    status=AttachmentExecutionStatusV2.BLOCKED_POLICY,
                    code=exc.code,
                ),
            )
        if _world_ledger_sha256(current_ledger) != snapshot_material.ledger_sha256:
            return self._cache_result(
                request,
                self._failure_result(
                    request,
                    status=AttachmentExecutionStatusV2.BLOCKED_POLICY,
                    code=AttachmentExecutionFailureCodeV2.WORLD_LEDGER_STALE,
                ),
            )
        ledger_copy = current_ledger.model_copy(deep=True)
        ledger_before = _world_ledger_sha256(ledger_copy)
        staging = self._staging_path(request)
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True, exist_ok=True)
        try:
            if request.selected_route_kind is AttachmentExecutionRouteKindV2.PROVIDER:
                result = await self._execute_provider(
                    request,
                    staging=staging,
                    world_ledger=ledger_copy,
                )
            else:
                result = await self._execute_runtime(
                    request,
                    staging=staging,
                    world_ledger=ledger_copy,
                    resource_grant=resource_grant,
                    resource_usage=resource_usage,
                    model_grant=model_grant,
                    model_observation=model_observation,
                )
        except AttachmentExecutionMaterialError as exc:
            result = self._failure_result(
                request,
                status=AttachmentExecutionStatusV2.BLOCKED_POLICY,
                code=exc.code,
            )
        if _world_ledger_sha256(ledger_copy) != ledger_before:
            result = self._failure_result(
                request,
                status=AttachmentExecutionStatusV2.BLOCKED_POLICY,
                code=(AttachmentExecutionFailureCodeV2.WORLD_LEDGER_MUTATION_ATTEMPT),
            )
        return self._cache_result(request, result)

    async def execute_with_resources(
        self,
        request: AttachmentExecutionRequestV2,
        grant: AttachmentResourceGrantV2,
    ) -> ResourceBoundAttachmentExecutionResultV2:
        return await self._execute_with_resources(request, grant)

    async def _execute_with_resources(
        self,
        request: AttachmentExecutionRequestV2,
        grant: AttachmentResourceGrantV2,
        *,
        model_grant: ModelControlGrantV2 | None = None,
        model_observation: dict[str, object] | None = None,
    ) -> ResourceBoundAttachmentExecutionResultV2:
        validate_attachment_resource_grant_identity(grant)
        cached = self._resource_execution_cache.get(request.idempotency_key)
        if cached is not None:
            request_sha256, grant_sha256, cached_result = cached
            if (
                request_sha256 != request.execution_request_sha256
                or grant_sha256 != grant.resource_grant_sha256
            ):
                raise ValueError("resource-bound execution idempotency conflict")
            return cached_result
        if request.idempotency_key in self._execution_cache:
            raise ValueError("resource-bound execution cannot reuse an unmetered result")
        usage_values = {
            "process_starts": 0,
            "renderer_operations": 0,
            "network_requests": 0,
            "retained_storage_bytes": 0,
        }
        if grant.operation_ref != attachment_execution_request_ref(request):
            blocked_result = self._failure_result(
                request,
                status=AttachmentExecutionStatusV2.BLOCKED_POLICY,
                code=AttachmentExecutionFailureCodeV2.RESOURCE_GRANT_INVALID,
            )
            return self._cache_resource_execution(
                request,
                grant,
                ResourceBoundAttachmentExecutionResultV2(
                    resource_grant_ref=attachment_resource_grant_ref(grant),
                    execution_result=blocked_result,
                    usage=AttachmentResourceUsageV2(
                        process_starts=0,
                        renderer_operations=0,
                        network_requests=0,
                        retained_storage_bytes=0,
                    ),
                ),
            )
        runtime_route = request.selected_route_kind is AttachmentExecutionRouteKindV2.RUNTIME
        network_tools = {
            "webfetch",
            "websearch",
            "fetch",
            "search",
        }
        requires_network = any(tool.casefold() in network_tools for tool in request.required_runtime_tools)
        if (runtime_route and grant.process_limit < 1) or (
            requires_network and grant.network_request_limit < 1
        ):
            blocked_result = self._failure_result(
                request,
                status=AttachmentExecutionStatusV2.BLOCKED_POLICY,
                code=AttachmentExecutionFailureCodeV2.RESOURCE_LIMIT_EXCEEDED,
            )
            wrapped_block = ResourceBoundAttachmentExecutionResultV2(
                resource_grant_ref=attachment_resource_grant_ref(grant),
                execution_result=blocked_result,
                usage=AttachmentResourceUsageV2(
                    process_starts=0,
                    renderer_operations=0,
                    network_requests=0,
                    retained_storage_bytes=0,
                ),
            )
            return self._cache_resource_execution(
                request,
                grant,
                wrapped_block,
            )
        renderer_media = {
            "application/pdf",
            "image/jpeg",
            "image/png",
            "image/svg+xml",
        }
        if request.media_type in renderer_media:
            if grant.renderer_limit < 1:
                blocked_result = self._failure_result(
                    request,
                    status=AttachmentExecutionStatusV2.BLOCKED_CAPABILITY,
                    code=AttachmentExecutionFailureCodeV2.RESOURCE_RENDERER_UNAVAILABLE,
                )
                wrapped_block = ResourceBoundAttachmentExecutionResultV2(
                    resource_grant_ref=attachment_resource_grant_ref(grant),
                    execution_result=blocked_result,
                    usage=AttachmentResourceUsageV2(
                        process_starts=usage_values["process_starts"],
                        renderer_operations=usage_values["renderer_operations"],
                        network_requests=usage_values["network_requests"],
                        retained_storage_bytes=usage_values["retained_storage_bytes"],
                    ),
                )
                return self._cache_resource_execution(
                    request,
                    grant,
                    wrapped_block,
                )
            usage_values["renderer_operations"] = 1
        execution_result = await self._execute(
            request,
            resource_grant=grant,
            resource_usage=usage_values,
            model_grant=model_grant,
            model_observation=model_observation,
        )
        if execution_result.status is AttachmentExecutionStatusV2.SUCCEEDED:
            target = (self._staging_path(request) / request.relative_path).resolve()
            usage_values["retained_storage_bytes"] = _path_size(target)
        usage = AttachmentResourceUsageV2(
            process_starts=usage_values["process_starts"],
            renderer_operations=usage_values["renderer_operations"],
            network_requests=usage_values["network_requests"],
            retained_storage_bytes=usage_values["retained_storage_bytes"],
        )
        try:
            grant.validate_usage(usage)
        except ValueError:
            shutil.rmtree(self._staging_path(request), ignore_errors=True)
            execution_result = self._failure_result(
                request,
                status=AttachmentExecutionStatusV2.BLOCKED_POLICY,
                code=AttachmentExecutionFailureCodeV2.RESOURCE_LIMIT_EXCEEDED,
                telemetry=execution_result.telemetry,
            )
        wrapped = ResourceBoundAttachmentExecutionResultV2(
            resource_grant_ref=attachment_resource_grant_ref(grant),
            execution_result=execution_result,
            usage=usage,
        )
        return self._cache_resource_execution(request, grant, wrapped)

    async def execute_with_controls(
        self,
        request: AttachmentExecutionRequestV2,
        resource_grant: AttachmentResourceGrantV2,
        model_descriptor: ModelInvocationDescriptorV2,
        model_grant: ModelControlGrantV2,
    ) -> tuple[
        ResourceBoundAttachmentExecutionResultV2,
        ModelControlReceiptV2,
    ]:
        validate_attachment_execution_request_identity(request)
        validate_model_invocation_descriptor_identity(model_descriptor)
        validate_model_control_grant_identity(model_grant)
        cache_key = request.idempotency_key
        cached = self._model_execution_cache.get(cache_key)
        if cached is not None:
            request_sha, resource_sha, model_sha, resource_result, receipt = cached
            if (
                request_sha != request.execution_request_sha256
                or resource_sha != resource_grant.resource_grant_sha256
                or model_sha != model_grant.model_control_grant_sha256
            ):
                raise ValueError("combined control execution idempotency conflict")
            return resource_result, receipt
        if cache_key in self._resource_execution_cache:
            raise ValueError("model-controlled execution cannot reuse a resource-only result")
        invalid_code: AttachmentExecutionFailureCodeV2 | None = None
        try:
            model_grant.validate_descriptor(model_descriptor)
        except ValueError:
            invalid_code = AttachmentExecutionFailureCodeV2.MODEL_GRANT_INVALID
        if model_descriptor.operation_ref != attachment_execution_request_ref(request):
            invalid_code = AttachmentExecutionFailureCodeV2.MODEL_GRANT_INVALID
        if request.model_profile_ref != model_descriptor.model_profile_ref:
            invalid_code = AttachmentExecutionFailureCodeV2.MODEL_PROFILE_MISMATCH
        if request.selected_route_kind is not AttachmentExecutionRouteKindV2.RUNTIME:
            invalid_code = AttachmentExecutionFailureCodeV2.MODEL_GRANT_INVALID
        if model_descriptor.mode is not ModelDemandModeFacadeV2.OPAQUE_RUNTIME_ENVELOPE:
            invalid_code = AttachmentExecutionFailureCodeV2.MODEL_GRANT_INVALID
        if invalid_code is not None:
            execution_result = self._failure_result(
                request,
                status=AttachmentExecutionStatusV2.BLOCKED_POLICY,
                code=invalid_code,
            )
            resource_result = ResourceBoundAttachmentExecutionResultV2(
                resource_grant_ref=attachment_resource_grant_ref(resource_grant),
                execution_result=execution_result,
                usage=AttachmentResourceUsageV2(
                    process_starts=0,
                    renderer_operations=0,
                    network_requests=0,
                    retained_storage_bytes=0,
                ),
            )
            receipt = ModelControlReceiptV2.create(
                descriptor_ref=model_invocation_descriptor_ref(model_descriptor),
                grant_ref=model_control_grant_ref(model_grant),
                outcome=ModelControlReceiptOutcomeV2.BLOCKED_POLICY,
                result_ref=None,
                usage=_zero_model_usage(),
                backpressure_ref=None,
                failure_code=invalid_code.value,
                completed_at=self._clock(),
                telemetry=execution_result.telemetry,
            )
            return self._cache_model_execution(
                request,
                resource_grant,
                model_grant,
                resource_result,
                receipt,
            )
        observation: dict[str, object] = {
            "requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }
        resource_result = await self._execute_with_resources(
            request,
            resource_grant,
            model_grant=model_grant,
            model_observation=observation,
        )
        malformed_usage = False
        if observation.get("usage_reported") is True:
            try:
                usage = _model_usage_from_observation(observation)
            except ValueError:
                malformed_usage = True
                usage = ModelControlUsageV2.conservative(
                    requests=model_grant.request_allowance,
                    tokens=model_grant.token_allowance,
                )
        else:
            usage = ModelControlUsageV2.conservative(
                requests=model_grant.request_allowance,
                tokens=model_grant.token_allowance,
            )
        try:
            if malformed_usage:
                raise ValueError("runtime model usage is malformed")
            model_grant.validate_usage(usage)
        except ValueError:
            shutil.rmtree(self._staging_path(request), ignore_errors=True)
            execution_result = self._failure_result(
                request,
                status=AttachmentExecutionStatusV2.BLOCKED_POLICY,
                code=AttachmentExecutionFailureCodeV2.MODEL_LIMIT_EXCEEDED,
                telemetry=resource_result.execution_result.telemetry,
            )
            resource_result = ResourceBoundAttachmentExecutionResultV2(
                resource_grant_ref=resource_result.resource_grant_ref,
                execution_result=execution_result,
                usage=resource_result.usage,
            )
            receipt = ModelControlReceiptV2.create(
                descriptor_ref=model_invocation_descriptor_ref(model_descriptor),
                grant_ref=model_control_grant_ref(model_grant),
                outcome=ModelControlReceiptOutcomeV2.BLOCKED_POLICY,
                result_ref=None,
                usage=usage,
                backpressure_ref=None,
                failure_code=AttachmentExecutionFailureCodeV2.MODEL_LIMIT_EXCEEDED.value,
                completed_at=self._clock(),
                telemetry=execution_result.telemetry,
            )
        else:
            execution_result = resource_result.execution_result
            if execution_result.status is AttachmentExecutionStatusV2.SUCCEEDED:
                receipt = ModelControlReceiptV2.create(
                    descriptor_ref=model_invocation_descriptor_ref(model_descriptor),
                    grant_ref=model_control_grant_ref(model_grant),
                    outcome=ModelControlReceiptOutcomeV2.SUCCEEDED,
                    result_ref=attachment_execution_result_ref(execution_result),
                    usage=usage,
                    backpressure_ref=None,
                    failure_code=None,
                    completed_at=self._clock(),
                    telemetry=execution_result.telemetry,
                )
            else:
                receipt = ModelControlReceiptV2.create(
                    descriptor_ref=model_invocation_descriptor_ref(model_descriptor),
                    grant_ref=model_control_grant_ref(model_grant),
                    outcome=ModelControlReceiptOutcomeV2.FAILED,
                    result_ref=None,
                    usage=usage,
                    backpressure_ref=None,
                    failure_code=(
                        execution_result.failure_code
                        or AttachmentExecutionFailureCodeV2.RUNTIME_EXECUTION_FAILED
                    ),
                    completed_at=self._clock(),
                    telemetry=execution_result.telemetry,
                )
        return self._cache_model_execution(
            request,
            resource_grant,
            model_grant,
            resource_result,
            receipt,
        )

    async def cancel_model_invocation(
        self,
        request: ModelInvocationCancellationRequestV2,
    ) -> ModelInvocationCancellationResultV2:
        validate_model_invocation_cancellation_request_identity(request)
        cached = self._model_cancellations.get(request.cancellation_request_id)
        if cached is not None:
            return cached
        entry = self._model_handles.get(request.invocation_handle_ref.object_id)
        if entry is None:
            outcome = (
                ModelInvocationCancellationOutcomeV2.ALREADY_FINISHED
                if request.invocation_handle_ref.object_id in self._model_stopped_handles
                else ModelInvocationCancellationOutcomeV2.FAILED
            )
        else:
            runtime, run_handle, reservation_id, fence = entry
            if reservation_id != request.reservation_ref.object_id or fence != request.fencing_token:
                outcome = ModelInvocationCancellationOutcomeV2.FAILED
            else:
                try:
                    await runtime.cancel(run_handle)
                except Exception:
                    outcome = ModelInvocationCancellationOutcomeV2.FAILED
                else:
                    outcome = ModelInvocationCancellationOutcomeV2.CANCELLED
                    self._model_handles.pop(
                        request.invocation_handle_ref.object_id,
                        None,
                    )
                    self._model_stopped_handles.add(request.invocation_handle_ref.object_id)
        result = ModelInvocationCancellationResultV2.create(
            cancellation_request_ref=model_invocation_cancellation_request_ref(request),
            outcome=outcome,
            completed_at=self._clock(),
        )
        self._model_cancellations[request.cancellation_request_id] = result
        return result

    async def terminate_resources(
        self,
        request: AttachmentResourceTerminationRequestV2,
    ) -> AttachmentResourceTerminationResultV2:
        validate_attachment_resource_termination_request_identity(request)
        entry = self._resource_handles.get(request.execution_handle_ref.object_id)
        if entry is None:
            outcome = (
                AttachmentResourceTerminationOutcomeV2.ALREADY_STOPPED
                if request.execution_handle_ref.object_id in self._resource_stopped_handles
                else AttachmentResourceTerminationOutcomeV2.FAILED
            )
        else:
            runtime, run_handle, reservation_id, fence = entry
            if reservation_id != request.reservation_ref.object_id or fence != request.fencing_token:
                outcome = AttachmentResourceTerminationOutcomeV2.FAILED
            else:
                try:
                    await runtime.cancel(run_handle)
                except Exception:
                    outcome = AttachmentResourceTerminationOutcomeV2.FAILED
                else:
                    outcome = AttachmentResourceTerminationOutcomeV2.TERMINATED
                    self._resource_handles.pop(
                        request.execution_handle_ref.object_id,
                        None,
                    )
                    self._resource_stopped_handles.add(request.execution_handle_ref.object_id)
        return AttachmentResourceTerminationResultV2.create(
            termination_request_ref=attachment_resource_termination_request_ref(request),
            outcome=outcome,
            completed_at=self._clock(),
        )

    async def _execute_provider(
        self,
        request: AttachmentExecutionRequestV2,
        *,
        staging: Path,
        world_ledger: WorldLedger,
    ) -> AttachmentExecutionResultV2:
        try:
            provider = self._providers.get(request.selected_route_id)
        except KeyError:
            return self._failure_result(
                request,
                status=AttachmentExecutionStatusV2.BLOCKED_CAPABILITY,
                code=(AttachmentExecutionFailureCodeV2.SELECTED_IMPLEMENTATION_MISSING),
            )
        implementation = f"{provider.__class__.__module__}.{provider.__class__.__qualname__}"
        if request.selected_route_version is not None and request.selected_route_version != implementation:
            return self._failure_result(
                request,
                status=AttachmentExecutionStatusV2.BLOCKED_CAPABILITY,
                code=(AttachmentExecutionFailureCodeV2.SELECTED_IMPLEMENTATION_MISMATCH),
            )
        provider_request = self._resolver.resolve_provider_request(
            request,
            staging_root=staging,
            world_ledger=world_ledger,
        )
        if (
            provider_request.plan.artifact_id != request.artifact_id
            or provider_request.plan.relative_path != request.relative_path
        ):
            return self._failure_result(
                request,
                status=AttachmentExecutionStatusV2.BLOCKED_POLICY,
                code=AttachmentExecutionFailureCodeV2.MATERIAL_MISMATCH,
            )
        try:
            provider_result = await asyncio.to_thread(
                provider.generate,
                provider_request,
            )
        except Exception:
            return self._failure_result(
                request,
                status=AttachmentExecutionStatusV2.TERMINAL_FAILURE,
                code=AttachmentExecutionFailureCodeV2.PROVIDER_EXECUTION_FAILED,
            )
        if provider_result.artifact_id != request.artifact_id or provider_result.status not in {
            ArtifactStatus.BUILT,
            ArtifactStatus.VALIDATED,
        }:
            return self._failure_result(
                request,
                status=AttachmentExecutionStatusV2.TERMINAL_FAILURE,
                code=AttachmentExecutionFailureCodeV2.PROVIDER_EXECUTION_FAILED,
            )
        return self._output_result(
            request,
            staging=staging,
            worker_version=implementation,
            telemetry=self._unavailable_telemetry(),
        )

    async def _execute_runtime(
        self,
        request: AttachmentExecutionRequestV2,
        *,
        staging: Path,
        world_ledger: WorldLedger,
        resource_grant: AttachmentResourceGrantV2 | None = None,
        resource_usage: dict[str, int] | None = None,
        model_grant: ModelControlGrantV2 | None = None,
        model_observation: dict[str, object] | None = None,
    ) -> AttachmentExecutionResultV2:
        try:
            runtime_name = RuntimeName(request.selected_route_id)
            runtime = self._runtimes.get(runtime_name)
        except (KeyError, ValueError):
            return self._failure_result(
                request,
                status=AttachmentExecutionStatusV2.BLOCKED_CAPABILITY,
                code=(AttachmentExecutionFailureCodeV2.SELECTED_IMPLEMENTATION_MISSING),
            )
        runtime_request = self._resolver.resolve_runtime_request(
            request,
            staging_root=staging,
            world_ledger=world_ledger,
        )
        if runtime_request.artifact_id != request.artifact_id:
            return self._failure_result(
                request,
                status=AttachmentExecutionStatusV2.BLOCKED_POLICY,
                code=AttachmentExecutionFailureCodeV2.MATERIAL_MISMATCH,
            )
        try:
            events: list[RuntimeEvent] = []
            registered_handle: str | None = None
            resource_overrun = False
            if resource_usage is not None:
                resource_usage["process_starts"] += 1
            if model_observation is not None:
                model_observation["requests"] = 1
            async for event in runtime.run(runtime_request):
                events.append(event)
                if resource_grant is not None and event.run_handle:
                    registered_handle = event.run_handle
                    self._resource_handles[resource_grant.execution_handle_ref.object_id] = (
                        runtime,
                        event.run_handle,
                        resource_grant.reservation_ref.object_id,
                        resource_grant.fencing_token,
                    )
                if model_grant is not None and event.run_handle:
                    self._model_handles[model_grant.invocation_handle_ref.object_id] = (
                        runtime,
                        event.run_handle,
                        model_grant.reservation_ref.object_id,
                        model_grant.fencing_token,
                    )
                if (
                    model_observation is not None
                    and event.event_type is RuntimeEventType.USAGE
                    and event.usage is not None
                ):
                    model_observation.update(
                        {
                            "usage_reported": True,
                            "requests": max(event.usage.turns, 1),
                            "input_tokens": event.usage.input_tokens,
                            "output_tokens": event.usage.output_tokens,
                            "cache_creation_input_tokens": (event.usage.cache_creation_input_tokens),
                            "cache_read_input_tokens": (event.usage.cache_read_input_tokens),
                        }
                    )
                if (
                    resource_usage is not None
                    and event.event_type is RuntimeEventType.TOOL_STARTED
                    and (event.tool_name or "").casefold()
                    in {
                        "webfetch",
                        "websearch",
                        "fetch",
                        "search",
                    }
                ):
                    resource_usage["network_requests"] += 1
                    if (
                        resource_grant is not None
                        and resource_usage["network_requests"] > resource_grant.network_request_limit
                    ):
                        if event.run_handle:
                            try:
                                await runtime.cancel(event.run_handle)
                            except Exception:
                                pass
                            else:
                                self._resource_stopped_handles.add(
                                    resource_grant.execution_handle_ref.object_id
                                )
                                self._resource_handles.pop(
                                    resource_grant.execution_handle_ref.object_id,
                                    None,
                                )
                                if model_grant is not None:
                                    self._model_stopped_handles.add(
                                        model_grant.invocation_handle_ref.object_id
                                    )
                                    self._model_handles.pop(
                                        model_grant.invocation_handle_ref.object_id,
                                        None,
                                    )
                        resource_overrun = True
                        break
        except Exception:
            return self._failure_result(
                request,
                status=AttachmentExecutionStatusV2.RETRYABLE_FAILURE,
                code=AttachmentExecutionFailureCodeV2.RUNTIME_EXECUTION_FAILED,
            )
        try:
            telemetry = self._runtime_telemetry(events)
        except (TypeError, ValueError):
            return self._failure_result(
                request,
                status=AttachmentExecutionStatusV2.TERMINAL_FAILURE,
                code=AttachmentExecutionFailureCodeV2.RUNTIME_PROTOCOL_ERROR,
            )
        if resource_overrun:
            return self._failure_result(
                request,
                status=AttachmentExecutionStatusV2.BLOCKED_POLICY,
                code=AttachmentExecutionFailureCodeV2.RESOURCE_LIMIT_EXCEEDED,
                telemetry=telemetry,
            )
        terminal = events[-1] if events else None
        if resource_grant is not None:
            self._resource_handles.pop(
                resource_grant.execution_handle_ref.object_id,
                None,
            )
            if registered_handle is not None:
                self._resource_stopped_handles.add(resource_grant.execution_handle_ref.object_id)
        if model_grant is not None:
            self._model_handles.pop(model_grant.invocation_handle_ref.object_id, None)
            if registered_handle is not None:
                self._model_stopped_handles.add(model_grant.invocation_handle_ref.object_id)
        if terminal is None or terminal.event_type is not RuntimeEventType.RUNTIME_FINISHED:
            return _runtime_failure_result(
                request,
                terminal,
                telemetry=telemetry,
            )
        return self._output_result(
            request,
            staging=staging,
            worker_version=request.selected_route_version,
            telemetry=telemetry,
        )

    def _output_result(
        self,
        request: AttachmentExecutionRequestV2,
        *,
        staging: Path,
        worker_version: str | None,
        telemetry: ExecutionTelemetryV2,
    ) -> AttachmentExecutionResultV2:
        target = (staging / request.relative_path).resolve()
        if not is_path_inside(staging, target):
            return self._failure_result(
                request,
                status=AttachmentExecutionStatusV2.BLOCKED_POLICY,
                code=AttachmentExecutionFailureCodeV2.OUTPUT_PATH_VIOLATION,
                telemetry=telemetry,
            )
        if not target.exists():
            return self._failure_result(
                request,
                status=AttachmentExecutionStatusV2.TERMINAL_FAILURE,
                code=AttachmentExecutionFailureCodeV2.OUTPUT_MISSING,
                telemetry=telemetry,
            )
        digest = sha256_path(target)
        return _success_result(
            request,
            output_ref=FacadeObjectRef(
                object_type="attachment-output",
                object_id=f"attachment-output://sha256/{digest}",
                object_version="v2",
                object_sha256=digest,
            ),
            output_sha256=digest,
            worker_version=worker_version,
            telemetry=telemetry,
        )

    def _failure_result(
        self,
        request: AttachmentExecutionRequestV2,
        *,
        status: AttachmentExecutionStatusV2,
        code: AttachmentExecutionFailureCodeV2,
        telemetry: ExecutionTelemetryV2 | None = None,
    ) -> AttachmentExecutionResultV2:
        return _failure_result(
            request,
            status=status,
            code=code,
            telemetry=telemetry or self._unavailable_telemetry(),
        )

    def _unavailable_telemetry(self) -> ExecutionTelemetryV2:
        return ExecutionTelemetryV2.unavailable(observed_at=self._clock())

    def _runtime_telemetry(
        self,
        events: list[RuntimeEvent],
    ) -> ExecutionTelemetryV2:
        usage = next(
            (
                event.usage
                for event in reversed(events)
                if event.event_type is RuntimeEventType.USAGE and event.usage is not None
            ),
            None,
        )
        tool_counts = normalize_tool_family_counts(
            event.tool_name
            for event in events
            if event.event_type is RuntimeEventType.TOOL_STARTED and event.tool_name is not None
        )
        duration_us = (
            usage.duration_ms * 1_000 if usage is not None and usage.duration_ms is not None else None
        )
        return ExecutionTelemetryV2.create(
            duration_us=duration_us,
            duration_availability=(
                TelemetryAvailabilityV2.REPORTED
                if duration_us is not None
                else TelemetryAvailabilityV2.UNAVAILABLE
            ),
            tool_family_counts=tool_counts,
            tool_availability=TelemetryAvailabilityV2.DERIVED,
            reported_cost=normalize_reported_cost_usd(usage.cost_usd if usage is not None else None),
            observed_at=self._clock(),
        )

    def _staging_path(
        self,
        request: AttachmentExecutionRequestV2,
    ) -> Path:
        digest = hashlib.sha256(request.execution_request_id.encode()).hexdigest()
        return (self._staging_root / digest).resolve()

    def _cache_result(
        self,
        request: AttachmentExecutionRequestV2,
        result: AttachmentExecutionResultV2,
    ) -> AttachmentExecutionResultV2:
        self._execution_cache[request.idempotency_key] = (
            request.execution_request_sha256,
            result,
        )
        return result

    def _cache_resource_execution(
        self,
        request: AttachmentExecutionRequestV2,
        grant: AttachmentResourceGrantV2,
        result: ResourceBoundAttachmentExecutionResultV2,
    ) -> ResourceBoundAttachmentExecutionResultV2:
        self._resource_execution_cache[request.idempotency_key] = (
            request.execution_request_sha256,
            grant.resource_grant_sha256,
            result,
        )
        return result

    def _cache_model_execution(
        self,
        request: AttachmentExecutionRequestV2,
        resource_grant: AttachmentResourceGrantV2,
        model_grant: ModelControlGrantV2,
        resource_result: ResourceBoundAttachmentExecutionResultV2,
        receipt: ModelControlReceiptV2,
    ) -> tuple[
        ResourceBoundAttachmentExecutionResultV2,
        ModelControlReceiptV2,
    ]:
        self._model_execution_cache[request.idempotency_key] = (
            request.execution_request_sha256,
            resource_grant.resource_grant_sha256,
            model_grant.model_control_grant_sha256,
            resource_result,
            receipt,
        )
        return resource_result, receipt


def _success_result(
    request: AttachmentExecutionRequestV2,
    *,
    output_ref: FacadeObjectRef,
    output_sha256: str,
    worker_version: str | None,
    telemetry: ExecutionTelemetryV2,
) -> AttachmentExecutionResultV2:
    return _finalize_result(
        AttachmentExecutionResultV2(
            execution_result_id="attachment-execution-result://pending",
            execution_request_ref=attachment_execution_request_ref(request),
            artifact_id=request.artifact_id,
            attempt=request.attempt,
            status=AttachmentExecutionStatusV2.SUCCEEDED,
            world_ledger_snapshot_ref=request.world_ledger_snapshot_ref,
            selected_route_kind=request.selected_route_kind,
            selected_route_id=request.selected_route_id,
            worker_version=worker_version,
            output_ref=output_ref,
            output_sha256=output_sha256,
            retryable=False,
            failure_code=None,
            telemetry=telemetry,
            policy_version=ARTIFACT_EXECUTION_POLICY_VERSION,
            execution_result_sha256="0" * 64,
        )
    )


def _failure_result(
    request: AttachmentExecutionRequestV2,
    *,
    status: AttachmentExecutionStatusV2,
    code: AttachmentExecutionFailureCodeV2,
    telemetry: ExecutionTelemetryV2,
) -> AttachmentExecutionResultV2:
    return _finalize_result(
        AttachmentExecutionResultV2(
            execution_result_id="attachment-execution-result://pending",
            execution_request_ref=attachment_execution_request_ref(request),
            artifact_id=request.artifact_id,
            attempt=request.attempt,
            status=status,
            world_ledger_snapshot_ref=request.world_ledger_snapshot_ref,
            selected_route_kind=request.selected_route_kind,
            selected_route_id=request.selected_route_id,
            worker_version=None,
            output_ref=None,
            output_sha256=None,
            retryable=(status is AttachmentExecutionStatusV2.RETRYABLE_FAILURE),
            failure_code=code,
            telemetry=telemetry,
            policy_version=ARTIFACT_EXECUTION_POLICY_VERSION,
            execution_result_sha256="0" * 64,
        )
    )


def _runtime_failure_result(
    request: AttachmentExecutionRequestV2,
    event: RuntimeEvent | None,
    *,
    telemetry: ExecutionTelemetryV2,
) -> AttachmentExecutionResultV2:
    code = AttachmentExecutionFailureCodeV2.RUNTIME_EXECUTION_FAILED
    status = AttachmentExecutionStatusV2.RETRYABLE_FAILURE
    if event is not None:
        mapping = {
            RuntimeErrorCode.TIMEOUT: (
                AttachmentExecutionStatusV2.RETRYABLE_FAILURE,
                AttachmentExecutionFailureCodeV2.RUNTIME_TIMEOUT,
            ),
            RuntimeErrorCode.AUTH_UNAVAILABLE: (
                AttachmentExecutionStatusV2.BLOCKED_CAPABILITY,
                AttachmentExecutionFailureCodeV2.RUNTIME_AUTH_UNAVAILABLE,
            ),
            RuntimeErrorCode.TOOL_DENIED: (
                AttachmentExecutionStatusV2.BLOCKED_POLICY,
                AttachmentExecutionFailureCodeV2.RUNTIME_TOOL_DENIED,
            ),
            RuntimeErrorCode.PATH_ESCAPE: (
                AttachmentExecutionStatusV2.BLOCKED_POLICY,
                AttachmentExecutionFailureCodeV2.OUTPUT_PATH_VIOLATION,
            ),
            RuntimeErrorCode.PROTOCOL_ERROR: (
                AttachmentExecutionStatusV2.TERMINAL_FAILURE,
                AttachmentExecutionFailureCodeV2.RUNTIME_PROTOCOL_ERROR,
            ),
        }
        if event.error_code is not None:
            status, code = mapping.get(event.error_code, (status, code))
    return _failure_result(
        request,
        status=status,
        code=code,
        telemetry=telemetry,
    )


def _finalize_result(
    result: AttachmentExecutionResultV2,
) -> AttachmentExecutionResultV2:
    digest = attachment_execution_result_carried_sha256(result)
    return result.model_copy(
        update={
            "execution_result_id": f"attachment-execution-result://sha256/{digest}",
            "execution_result_sha256": digest,
        }
    )


def _world_ledger_sha256(ledger: WorldLedger) -> str:
    return _json_sha256(ledger.model_dump(mode="json", exclude_none=False))


def world_ledger_object_ref(
    ledger_id: str,
    ledger: WorldLedger,
) -> FacadeObjectRef:
    return FacadeObjectRef(
        object_type="world-ledger",
        object_id=ledger_id,
        object_version="v1",
        object_sha256=_world_ledger_sha256(ledger),
    )


def _value_sha256(value: object) -> str:
    return _json_sha256(value)


def _json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _model_usage_from_observation(
    observation: dict[str, object],
) -> ModelControlUsageV2:
    requests = _non_negative_usage_int(observation.get("requests", 0))
    input_tokens = _non_negative_usage_int(observation.get("input_tokens", 0))
    output_tokens = _non_negative_usage_int(observation.get("output_tokens", 0))
    cache_creation = _non_negative_usage_int(observation.get("cache_creation_input_tokens", 0))
    cache_read = _non_negative_usage_int(observation.get("cache_read_input_tokens", 0))
    return ModelControlUsageV2(
        requests=requests,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=cache_creation,
        cache_read_input_tokens=cache_read,
        charged_tokens=(input_tokens + output_tokens + cache_creation + cache_read),
        source=ModelControlUsageSourceV2.REPORTED,
    )


def _non_negative_usage_int(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("runtime model usage cannot be boolean")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = int(value)
        except ValueError as exc:
            raise ValueError("runtime model usage must be an integer") from exc
    elif isinstance(value, float) and value.is_integer():
        parsed = int(value)
    else:
        raise ValueError("runtime model usage must be an integer")
    if parsed < 0:
        raise ValueError("runtime model usage cannot be negative")
    return parsed


def _zero_model_usage() -> ModelControlUsageV2:
    return ModelControlUsageV2(
        requests=0,
        input_tokens=0,
        output_tokens=0,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        charged_tokens=0,
        source=ModelControlUsageSourceV2.REPORTED,
    )


def _path_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if not path.is_dir():
        return 0
    return sum(
        child.stat().st_size for child in path.rglob("*") if child.is_file() and not child.is_symlink()
    )


def _ref_key(ref: FacadeObjectRef) -> tuple[str, str, str, str]:
    return (
        ref.object_type,
        ref.object_id,
        ref.object_version,
        ref.object_sha256,
    )
