from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from env_mock_agent.facade import (
    AttachmentExecutionFailureCodeV2,
    AttachmentExecutionRequestV2,
    AttachmentExecutionResultV2,
    AttachmentExecutionRouteKindV2,
    AttachmentExecutionStatusV2,
    ExecutionTelemetryV2,
    FacadeObjectRef,
    PublicSourceFetchRequestV2,
    PublicSourceFetchResultV2,
    PublicSourceSearchRequestV2,
    PublicSourceSearchResultV2,
    ReconstructionMode,
    RegistryResourceBoundRetrievalFacade,
    attachment_execution_request_carried_sha256,
    attachment_execution_request_ref,
    attachment_execution_result_carried_sha256,
    public_source_fetch_request_carried_sha256,
    public_source_fetch_request_ref,
    public_source_search_request_carried_sha256,
    public_source_search_request_ref,
)
from env_mock_agent.facade.execution_adapter import (
    MappingAttachmentExecutionMaterialResolver,
    RegistryAttachmentExecutionFacade,
)
from env_mock_agent.facade.model_control_v2 import (
    ModelControlGrantV2,
    ModelControlReceiptOutcomeV2,
    ModelControlUsageSourceV2,
    ModelDemandModeFacadeV2,
    ModelInvocationDescriptorV2,
    model_invocation_descriptor_ref,
)
from env_mock_agent.facade.resource_v2 import (
    AttachmentResourceGrantV2,
    AttachmentResourceTerminationOutcomeV2,
    AttachmentResourceTerminationRequestV2,
    AttachmentResourceTerminationResultV2,
    AttachmentResourceUsageV2,
    attachment_resource_termination_request_ref,
)
from env_mock_agent.providers import ProviderRegistry
from env_mock_agent.runtimes import ClaudeCodeCliRuntime, RuntimeRegistry
from env_mock_agent.runtimes.base import AgentRuntime

HASH = "a" * 64
NOW = datetime(2026, 7, 31, tzinfo=UTC)


def _ref(object_type: str, *, version: str = "v2") -> FacadeObjectRef:
    return FacadeObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://r6-04/current",
        object_version=version,
        object_sha256=HASH,
    )


def _grant(
    *,
    operation_ref: FacadeObjectRef | None = None,
    process_limit: int = 1,
    network_request_limit: int = 2,
    storage_byte_limit: int = 4096,
) -> AttachmentResourceGrantV2:
    return AttachmentResourceGrantV2.create(
        reservation_ref=_ref("work-resource-reservation"),
        execution_handle_ref=_ref("resource-execution-handle", version="v1"),
        operation_ref=(operation_ref or attachment_execution_request_ref(_execution_request())),
        fencing_token=2,
        process_limit=process_limit,
        renderer_limit=0,
        network_request_limit=network_request_limit,
        storage_byte_limit=storage_byte_limit,
    )


def _execution_request(
    *,
    runtime: bool = False,
    required_runtime_tools: tuple[str, ...] = (),
) -> AttachmentExecutionRequestV2:
    value = AttachmentExecutionRequestV2(
        execution_request_id="attachment-execution-request://pending",
        execution_plan_ref=_ref("artifact-execution-plan"),
        execution_group_id="artifact-execution-group://r6-04/current",
        artifact_id="artifact://r6-04/current",
        build_spec_ref=_ref("artifact-build-spec"),
        producer_task_view_ref=_ref("producer-task-view"),
        content_contract_ref=_ref("artifact-content-contract"),
        render_contract_ref=_ref("artifact-render-contract"),
        provider_payload_ref=(None if runtime else _ref("attachment-provider-payload")),
        model_profile_ref=(_ref("model-profile") if runtime else None),
        selected_route_kind=(
            AttachmentExecutionRouteKindV2.RUNTIME if runtime else AttachmentExecutionRouteKindV2.PROVIDER
        ),
        selected_route_id="fake" if runtime else "text",
        selected_route_version=None,
        relative_path="inputs/resource.txt",
        media_type="text/plain",
        mode=ReconstructionMode.PROMPT_ONLY,
        runtime_role="builder",
        required_runtime_tools=required_runtime_tools,
        evidence_grants=(),
        world_ledger_snapshot_ref=_ref("world-ledger-snapshot"),
        locked_fact_ids=(),
        dependency_result_refs=(),
        attempt=1,
        retry_of_result_ref=None,
        idempotency_key="resource-execution-request",
        execution_request_sha256="0" * 64,
    )
    digest = attachment_execution_request_carried_sha256(value)
    return value.model_copy(
        update={
            "execution_request_id": (f"attachment-execution-request://sha256/{digest}"),
            "execution_request_sha256": digest,
        }
    )


def _search_request() -> PublicSourceSearchRequestV2:
    value = PublicSourceSearchRequestV2(
        search_request_id="public-source-search-request://pending",
        query="official source",
        provider_id="search-provider://resource-test",
        result_limit=1,
        allowed_schemes=("https",),
        allowed_host_suffixes=("example.gov",),
        query_approval_ref=_ref(
            "public-search-query-approval",
            version="v1",
        ),
        retrieval_policy_ref=_ref(
            "public-source-retrieval-policy",
        ),
        idempotency_key="resource-search-request",
        search_request_sha256="0" * 64,
    )
    digest = public_source_search_request_carried_sha256(value)
    return value.model_copy(
        update={
            "search_request_id": (f"public-source-search-request://sha256/{digest}"),
            "search_request_sha256": digest,
        }
    )


def _fetch_request() -> PublicSourceFetchRequestV2:
    value = PublicSourceFetchRequestV2(
        fetch_request_id="public-source-fetch-request://pending",
        source_uri="https://example.gov/source.txt",
        provider_id="fetch-provider://resource-test",
        max_bytes=1024,
        allowed_schemes=("https",),
        allowed_host_suffixes=("example.gov",),
        source_approval_ref=_ref(
            "public-source-approval",
            version="v1",
        ),
        retrieval_policy_ref=_ref(
            "public-source-retrieval-policy",
        ),
        idempotency_key="resource-fetch-request",
        fetch_request_sha256="0" * 64,
    )
    digest = public_source_fetch_request_carried_sha256(value)
    return value.model_copy(
        update={
            "fetch_request_id": (f"public-source-fetch-request://sha256/{digest}"),
            "fetch_request_sha256": digest,
        }
    )


class _RetrievalProbe:
    def __init__(self) -> None:
        self.search_calls = 0
        self.fetch_calls = 0

    async def search(
        self,
        request: PublicSourceSearchRequestV2,
    ) -> PublicSourceSearchResultV2:
        self.search_calls += 1
        return PublicSourceSearchResultV2(
            search_result_id="public-source-search-result://resource-test",
            search_request_ref=public_source_search_request_ref(request),
            provider_id=request.provider_id,
            hits=(),
            search_result_sha256=HASH,
        )

    async def fetch(
        self,
        request: PublicSourceFetchRequestV2,
    ) -> PublicSourceFetchResultV2:
        self.fetch_calls += 1
        return PublicSourceFetchResultV2(
            fetch_result_id="public-source-fetch-result://resource-test",
            fetch_request_ref=public_source_fetch_request_ref(request),
            requested_source_uri=request.source_uri,
            canonical_source_uri=request.source_uri,
            retrieved_at=NOW,
            status_code=200,
            media_type="text/plain",
            bytes_count=12,
            content_ref=FacadeObjectRef(
                object_type="public-source-content",
                object_id="public-source-content://resource-test",
                object_version="v1",
                object_sha256="b" * 64,
            ),
            content_sha256="b" * 64,
            provider_id=request.provider_id,
            response_metadata_sha256="c" * 64,
            fetch_result_sha256=HASH,
        )


class _OutputFacade(RegistryAttachmentExecutionFacade):
    async def _execute(self, request, **kwargs):
        observation = kwargs.get("model_observation")
        if isinstance(observation, dict):
            observation.update(
                {
                    "usage_reported": True,
                    "requests": 2,
                    "input_tokens": 100,
                    "output_tokens": 200,
                    "cache_creation_input_tokens": 10,
                    "cache_read_input_tokens": 20,
                }
            )
        target = self._staging_path(request) / request.relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"0123456789")
        output_hash = "b" * 64
        result = AttachmentExecutionResultV2(
            execution_result_id="attachment-execution-result://pending",
            execution_request_ref=attachment_execution_request_ref(request),
            artifact_id=request.artifact_id,
            attempt=1,
            status=AttachmentExecutionStatusV2.SUCCEEDED,
            world_ledger_snapshot_ref=request.world_ledger_snapshot_ref,
            selected_route_kind=request.selected_route_kind,
            selected_route_id=request.selected_route_id,
            worker_version="resource-test",
            output_ref=FacadeObjectRef(
                object_type="attachment-output",
                object_id="attachment-output://r6-04/current",
                object_version="v2",
                object_sha256=output_hash,
            ),
            output_sha256=output_hash,
            retryable=False,
            failure_code=None,
            telemetry=ExecutionTelemetryV2.unavailable(observed_at=NOW),
            execution_result_sha256="0" * 64,
        )
        digest = attachment_execution_result_carried_sha256(result)
        return result.model_copy(
            update={
                "execution_result_id": (f"attachment-execution-result://sha256/{digest}"),
                "execution_result_sha256": digest,
            }
        )


class _MalformedUsageFacade(_OutputFacade):
    async def _execute(self, request, **kwargs):
        result = await super()._execute(request, **kwargs)
        observation = kwargs.get("model_observation")
        if isinstance(observation, dict):
            observation["input_tokens"] = True
        return result


def _model_descriptor(
    request: AttachmentExecutionRequestV2,
) -> ModelInvocationDescriptorV2:
    assert request.model_profile_ref is not None
    return ModelInvocationDescriptorV2.create(
        operation_ref=attachment_execution_request_ref(request),
        model_profile_ref=request.model_profile_ref,
        mode=ModelDemandModeFacadeV2.OPAQUE_RUNTIME_ENVELOPE,
        request_allowance=3,
        input_token_allowance=200,
        output_token_allowance=300,
        cache_token_allowance=100,
        idempotency_key="runtime-model-invocation",
    )


def _model_grant(
    descriptor: ModelInvocationDescriptorV2,
) -> ModelControlGrantV2:
    return ModelControlGrantV2.create(
        reservation_ref=_ref("work-model-reservation"),
        invocation_handle_ref=_ref("model-invocation-handle", version="v1"),
        descriptor_ref=model_invocation_descriptor_ref(descriptor),
        model_profile_ref=descriptor.model_profile_ref,
        fencing_token=2,
        request_allowance=descriptor.request_allowance,
        token_allowance=descriptor.token_allowance,
        output_token_limit=descriptor.output_token_allowance,
    )


class _CancelProbe:
    def __init__(self) -> None:
        self.cancelled: list[str] = []

    async def cancel(self, run_handle: str) -> None:
        self.cancelled.append(run_handle)


def test_resource_grant_is_strict_and_usage_enforces_each_bound() -> None:
    grant = _grant()
    usage = AttachmentResourceUsageV2(
        process_starts=1,
        renderer_operations=0,
        network_requests=2,
        retained_storage_bytes=4096,
    )

    grant.validate_usage(usage)
    with pytest.raises(ValueError, match="network"):
        grant.validate_usage(usage.model_copy(update={"network_requests": 3}))
    with pytest.raises(ValueError, match="storage"):
        grant.validate_usage(usage.model_copy(update={"retained_storage_bytes": 4097}))
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        AttachmentResourceGrantV2.model_validate(
            {
                **grant.model_dump(mode="python"),
                "pid": 1234,
            }
        )


def test_termination_contract_is_opaque_and_hash_bound() -> None:
    grant = _grant()
    request = AttachmentResourceTerminationRequestV2.create(
        reservation_ref=grant.reservation_ref,
        execution_handle_ref=grant.execution_handle_ref,
        fencing_token=grant.fencing_token,
        reason="CANCELLED",
        requested_at=NOW,
    )
    request_ref = attachment_resource_termination_request_ref(request)
    result = AttachmentResourceTerminationResultV2.create(
        termination_request_ref=request_ref,
        outcome=AttachmentResourceTerminationOutcomeV2.TERMINATED,
        completed_at=NOW,
    )

    assert result.termination_request_ref == request_ref
    rendered = str(result.model_dump(mode="json"))
    for forbidden in ("pid", "path", "stderr", "exception", "credential", "runtime_output"):
        assert forbidden not in rendered


@pytest.mark.asyncio
async def test_resource_bound_execution_rejects_storage_overrun(
    tmp_path: Path,
) -> None:
    facade = _OutputFacade(
        ProviderRegistry(),
        RuntimeRegistry(),
        resolver=MappingAttachmentExecutionMaterialResolver(world_ledgers={}),
        staging_root=tmp_path,
        clock=lambda: NOW,
    )
    grant = _grant(storage_byte_limit=4)

    result = await facade.execute_with_resources(_execution_request(), grant)

    assert result.usage.retained_storage_bytes == 10
    assert result.execution_result.status is AttachmentExecutionStatusV2.BLOCKED_POLICY
    assert result.execution_result.failure_code is (AttachmentExecutionFailureCodeV2.RESOURCE_LIMIT_EXCEEDED)
    assert not facade._staging_path(_execution_request()).exists()


@pytest.mark.asyncio
async def test_runtime_execution_requires_both_resource_and_model_grants(
    tmp_path: Path,
) -> None:
    facade = _OutputFacade(
        ProviderRegistry(),
        RuntimeRegistry(),
        resolver=MappingAttachmentExecutionMaterialResolver(world_ledgers={}),
        staging_root=tmp_path,
        clock=lambda: NOW,
    )
    request = _execution_request(runtime=True)
    resource_grant = _grant(operation_ref=attachment_execution_request_ref(request))
    descriptor = _model_descriptor(request)
    model_grant = _model_grant(descriptor)

    resource_result, model_receipt = await facade.execute_with_controls(
        request,
        resource_grant,
        descriptor,
        model_grant,
    )

    assert resource_result.execution_result.status is (AttachmentExecutionStatusV2.SUCCEEDED)
    assert model_receipt.outcome is ModelControlReceiptOutcomeV2.SUCCEEDED
    assert model_receipt.usage.source is ModelControlUsageSourceV2.REPORTED
    assert model_receipt.usage.requests == 2
    assert model_receipt.usage.charged_tokens == 330


@pytest.mark.asyncio
async def test_runtime_model_usage_overrun_discards_staging_output(
    tmp_path: Path,
) -> None:
    facade = _OutputFacade(
        ProviderRegistry(),
        RuntimeRegistry(),
        resolver=MappingAttachmentExecutionMaterialResolver(world_ledgers={}),
        staging_root=tmp_path,
        clock=lambda: NOW,
    )
    request = _execution_request(runtime=True)
    resource_grant = _grant(operation_ref=attachment_execution_request_ref(request))
    baseline = _model_descriptor(request)
    descriptor = ModelInvocationDescriptorV2.create(
        operation_ref=baseline.operation_ref,
        model_profile_ref=baseline.model_profile_ref,
        mode=baseline.mode,
        request_allowance=1,
        input_token_allowance=baseline.input_token_allowance,
        output_token_allowance=baseline.output_token_allowance,
        cache_token_allowance=baseline.cache_token_allowance,
        idempotency_key="runtime-model-overrun",
    )

    resource_result, model_receipt = await facade.execute_with_controls(
        request,
        resource_grant,
        descriptor,
        _model_grant(descriptor),
    )

    assert resource_result.execution_result.status is (AttachmentExecutionStatusV2.BLOCKED_POLICY)
    assert resource_result.execution_result.failure_code is (
        AttachmentExecutionFailureCodeV2.MODEL_LIMIT_EXCEEDED
    )
    assert model_receipt.outcome is ModelControlReceiptOutcomeV2.BLOCKED_POLICY
    assert model_receipt.failure_code == (AttachmentExecutionFailureCodeV2.MODEL_LIMIT_EXCEEDED.value)
    assert model_receipt.usage.requests == 2
    assert not facade._staging_path(request).exists()


@pytest.mark.asyncio
async def test_runtime_malformed_model_usage_is_conservatively_blocked(
    tmp_path: Path,
) -> None:
    facade = _MalformedUsageFacade(
        ProviderRegistry(),
        RuntimeRegistry(),
        resolver=MappingAttachmentExecutionMaterialResolver(world_ledgers={}),
        staging_root=tmp_path,
        clock=lambda: NOW,
    )
    request = _execution_request(runtime=True)
    resource_grant = _grant(operation_ref=attachment_execution_request_ref(request))
    descriptor = _model_descriptor(request)
    grant = _model_grant(descriptor)

    resource_result, receipt = await facade.execute_with_controls(
        request,
        resource_grant,
        descriptor,
        grant,
    )

    assert resource_result.execution_result.status is (AttachmentExecutionStatusV2.BLOCKED_POLICY)
    assert resource_result.execution_result.failure_code is (
        AttachmentExecutionFailureCodeV2.MODEL_LIMIT_EXCEEDED
    )
    assert receipt.outcome is ModelControlReceiptOutcomeV2.BLOCKED_POLICY
    assert receipt.usage.source is ModelControlUsageSourceV2.CONSERVATIVE_ALLOWANCE
    assert receipt.usage.requests == grant.request_allowance
    assert receipt.usage.charged_tokens == grant.token_allowance
    assert not facade._staging_path(request).exists()


@pytest.mark.asyncio
async def test_runtime_rejects_direct_request_envelope_before_execution(
    tmp_path: Path,
) -> None:
    facade = _OutputFacade(
        ProviderRegistry(),
        RuntimeRegistry(),
        resolver=MappingAttachmentExecutionMaterialResolver(world_ledgers={}),
        staging_root=tmp_path,
        clock=lambda: NOW,
    )
    request = _execution_request(runtime=True)
    resource_grant = _grant(operation_ref=attachment_execution_request_ref(request))
    baseline = _model_descriptor(request)
    descriptor = ModelInvocationDescriptorV2.create(
        operation_ref=baseline.operation_ref,
        model_profile_ref=baseline.model_profile_ref,
        mode=ModelDemandModeFacadeV2.DIRECT_REQUEST,
        request_allowance=1,
        input_token_allowance=baseline.input_token_allowance,
        output_token_allowance=baseline.output_token_allowance,
        cache_token_allowance=baseline.cache_token_allowance,
        idempotency_key="runtime-direct-envelope",
    )

    resource_result, receipt = await facade.execute_with_controls(
        request,
        resource_grant,
        descriptor,
        _model_grant(descriptor),
    )

    assert resource_result.execution_result.status is (AttachmentExecutionStatusV2.BLOCKED_POLICY)
    assert resource_result.execution_result.failure_code is (
        AttachmentExecutionFailureCodeV2.MODEL_GRANT_INVALID
    )
    assert receipt.outcome is ModelControlReceiptOutcomeV2.BLOCKED_POLICY
    assert not facade._staging_path(request).exists()


@pytest.mark.asyncio
async def test_runtime_model_grant_mismatch_blocks_before_execution(
    tmp_path: Path,
) -> None:
    facade = _OutputFacade(
        ProviderRegistry(),
        RuntimeRegistry(),
        resolver=MappingAttachmentExecutionMaterialResolver(world_ledgers={}),
        staging_root=tmp_path,
        clock=lambda: NOW,
    )
    request = _execution_request(runtime=True)
    resource_grant = _grant(operation_ref=attachment_execution_request_ref(request))
    descriptor = _model_descriptor(request)
    wrong_descriptor = ModelInvocationDescriptorV2.create(
        operation_ref=_ref("attachment-execution-request"),
        model_profile_ref=descriptor.model_profile_ref,
        mode=descriptor.mode,
        request_allowance=descriptor.request_allowance,
        input_token_allowance=descriptor.input_token_allowance,
        output_token_allowance=descriptor.output_token_allowance,
        cache_token_allowance=descriptor.cache_token_allowance,
        idempotency_key="runtime-model-invocation-other",
    )

    resource_result, model_receipt = await facade.execute_with_controls(
        request,
        resource_grant,
        wrong_descriptor,
        _model_grant(wrong_descriptor),
    )

    assert resource_result.execution_result.status is (AttachmentExecutionStatusV2.BLOCKED_POLICY)
    assert resource_result.execution_result.failure_code is (
        AttachmentExecutionFailureCodeV2.MODEL_GRANT_INVALID
    )
    assert model_receipt.outcome is ModelControlReceiptOutcomeV2.BLOCKED_POLICY
    assert not facade._staging_path(request).exists()


@pytest.mark.asyncio
async def test_resource_bound_execution_rejects_cross_request_grant(
    tmp_path: Path,
) -> None:
    facade = _OutputFacade(
        ProviderRegistry(),
        RuntimeRegistry(),
        resolver=MappingAttachmentExecutionMaterialResolver(world_ledgers={}),
        staging_root=tmp_path,
        clock=lambda: NOW,
    )
    request = _execution_request()
    wrong_operation = _execution_request().model_copy(
        update={
            "execution_request_id": "attachment-execution-request://other",
            "execution_request_sha256": "d" * 64,
        }
    )
    grant = _grant(operation_ref=attachment_execution_request_ref(wrong_operation))

    result = await facade.execute_with_resources(request, grant)

    assert result.execution_result.status is (AttachmentExecutionStatusV2.BLOCKED_POLICY)
    assert result.execution_result.failure_code is (AttachmentExecutionFailureCodeV2.RESOURCE_GRANT_INVALID)
    assert not facade._staging_path(request).exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("process_limit", "network_request_limit", "required_runtime_tools"),
    (
        (0, 2, ()),
        (1, 0, ("WebSearch",)),
    ),
)
async def test_resource_bound_execution_rejects_missing_grant_before_runtime(
    tmp_path: Path,
    process_limit: int,
    network_request_limit: int,
    required_runtime_tools: tuple[str, ...],
) -> None:
    facade = RegistryAttachmentExecutionFacade(
        ProviderRegistry(),
        RuntimeRegistry(),
        resolver=MappingAttachmentExecutionMaterialResolver(world_ledgers={}),
        staging_root=tmp_path,
        clock=lambda: NOW,
    )
    execution_request = _execution_request(
        runtime=True,
        required_runtime_tools=required_runtime_tools,
    )
    grant = _grant(
        operation_ref=attachment_execution_request_ref(execution_request),
        process_limit=process_limit,
        network_request_limit=network_request_limit,
    )

    result = await facade.execute_with_resources(execution_request, grant)

    assert result.usage == AttachmentResourceUsageV2(
        process_starts=0,
        renderer_operations=0,
        network_requests=0,
        retained_storage_bytes=0,
    )
    assert result.execution_result.status is AttachmentExecutionStatusV2.BLOCKED_POLICY
    assert result.execution_result.failure_code is (AttachmentExecutionFailureCodeV2.RESOURCE_LIMIT_EXCEEDED)


@pytest.mark.asyncio
async def test_resource_bound_retrieval_binds_each_operation_and_usage() -> None:
    probe = _RetrievalProbe()
    facade = RegistryResourceBoundRetrievalFacade(probe)
    search = _search_request()
    fetch = _fetch_request()

    search_result = await facade.search_with_resources(
        search,
        _grant(
            operation_ref=public_source_search_request_ref(search),
            process_limit=0,
            network_request_limit=1,
            storage_byte_limit=0,
        ),
    )
    fetch_result = await facade.fetch_with_resources(
        fetch,
        _grant(
            operation_ref=public_source_fetch_request_ref(fetch),
            process_limit=0,
            network_request_limit=1,
            storage_byte_limit=0,
        ),
    )

    assert search_result.usage.network_requests == 1
    assert fetch_result.usage.network_requests == 1
    assert probe.search_calls == 1
    assert probe.fetch_calls == 1

    with pytest.raises(ValueError, match="bind the search"):
        await facade.search_with_resources(
            search,
            _grant(
                operation_ref=public_source_fetch_request_ref(fetch),
                process_limit=0,
                network_request_limit=1,
                storage_byte_limit=0,
            ),
        )
    with pytest.raises(ValueError, match="network"):
        await facade.fetch_with_resources(
            fetch,
            _grant(
                operation_ref=public_source_fetch_request_ref(fetch),
                process_limit=0,
                network_request_limit=0,
                storage_byte_limit=0,
            ),
        )
    assert probe.search_calls == 1
    assert probe.fetch_calls == 1


@pytest.mark.asyncio
async def test_resource_termination_uses_opaque_handle_and_exact_fence(
    tmp_path: Path,
) -> None:
    facade = RegistryAttachmentExecutionFacade(
        ProviderRegistry(),
        RuntimeRegistry(),
        resolver=MappingAttachmentExecutionMaterialResolver(world_ledgers={}),
        staging_root=tmp_path,
        clock=lambda: NOW,
    )
    grant = _grant()
    runtime = _CancelProbe()
    facade._resource_handles[grant.execution_handle_ref.object_id] = (
        cast(AgentRuntime, runtime),
        "runtime-handle-1",
        grant.reservation_ref.object_id,
        grant.fencing_token,
    )
    request = AttachmentResourceTerminationRequestV2.create(
        reservation_ref=grant.reservation_ref,
        execution_handle_ref=grant.execution_handle_ref,
        fencing_token=grant.fencing_token,
        reason="CANCELLED",
        requested_at=NOW,
    )

    result = await facade.terminate_resources(request)

    assert result.outcome is AttachmentResourceTerminationOutcomeV2.TERMINATED
    assert runtime.cancelled == ["runtime-handle-1"]


@pytest.mark.asyncio
async def test_resource_termination_stops_real_process_group(
    tmp_path: Path,
) -> None:
    executable = tmp_path / "resource-process"
    executable.write_text(
        "#!/bin/sh\nset -eu\nsleep 30\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    runtime = ClaudeCodeCliRuntime(str(executable))
    run_handle = "claude-cli:resource-process"
    process = await asyncio.create_subprocess_exec(
        str(executable),
        start_new_session=True,
    )
    runtime._processes[run_handle] = process
    facade = RegistryAttachmentExecutionFacade(
        ProviderRegistry(),
        RuntimeRegistry(),
        resolver=MappingAttachmentExecutionMaterialResolver(world_ledgers={}),
        staging_root=tmp_path,
        clock=lambda: NOW,
    )
    grant = _grant()
    facade._resource_handles[grant.execution_handle_ref.object_id] = (
        runtime,
        run_handle,
        grant.reservation_ref.object_id,
        grant.fencing_token,
    )
    request = AttachmentResourceTerminationRequestV2.create(
        reservation_ref=grant.reservation_ref,
        execution_handle_ref=grant.execution_handle_ref,
        fencing_token=grant.fencing_token,
        reason="CANCELLED",
        requested_at=NOW,
    )

    try:
        result = await facade.terminate_resources(request)

        assert result.outcome is (AttachmentResourceTerminationOutcomeV2.TERMINATED)
        assert process.returncode is not None
    finally:
        if process.returncode is None:
            process.terminate()
            await process.wait()
        runtime._processes.pop(run_handle, None)


@pytest.mark.asyncio
async def test_unknown_resource_handle_fails_closed(
    tmp_path: Path,
) -> None:
    facade = RegistryAttachmentExecutionFacade(
        ProviderRegistry(),
        RuntimeRegistry(),
        resolver=MappingAttachmentExecutionMaterialResolver(world_ledgers={}),
        staging_root=tmp_path,
        clock=lambda: NOW,
    )
    grant = _grant()
    request = AttachmentResourceTerminationRequestV2.create(
        reservation_ref=grant.reservation_ref,
        execution_handle_ref=grant.execution_handle_ref,
        fencing_token=grant.fencing_token,
        reason="CANCELLED",
        requested_at=NOW,
    )

    result = await facade.terminate_resources(request)

    assert result.outcome is AttachmentResourceTerminationOutcomeV2.FAILED
