from __future__ import annotations

from datetime import UTC, datetime

import pytest

from env_mock_agent.facade import FacadeObjectRef
from env_mock_agent.facade.execution_adapter import (
    MappingAttachmentExecutionMaterialResolver,
    ProviderExecutionMaterial,
    RegistryAttachmentExecutionFacade,
    RuntimeExecutionMaterial,
    world_ledger_object_ref,
)
from env_mock_agent.facade.execution_v2 import (
    AttachmentExecutionFailureCodeV2,
    AttachmentExecutionRequestV2,
    AttachmentExecutionRouteKindV2,
    AttachmentExecutionStatusV2,
    WorldLedgerSnapshotRequestV2,
    attachment_execution_request_carried_sha256,
    world_ledger_fact_id,
    world_ledger_snapshot_ref,
    world_ledger_snapshot_request_carried_sha256,
)
from env_mock_agent.providers import ProviderRegistry, TextProvider
from env_mock_agent.providers.base import ProviderRequest
from env_mock_agent.runtimes import FakeRuntime, RuntimeRegistry
from env_mock_agent.schemas import (
    ArtifactPlan,
    ArtifactResult,
    ModelProfile,
    RuntimeName,
    RuntimeRequest,
    WorldLedger,
)

HASH = "a" * 64


def _ref(
    object_type: str,
    suffix: str,
    *,
    version: str = "v2",
) -> FacadeObjectRef:
    return FacadeObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://{suffix}",
        object_version=version,
        object_sha256=HASH,
    )


def _ledger_ref(ledger: WorldLedger) -> FacadeObjectRef:
    return world_ledger_object_ref("world-ledger://current", ledger)


def _snapshot_request(
    ledger_ref: FacadeObjectRef,
    fact_ids: tuple[str, ...],
) -> WorldLedgerSnapshotRequestV2:
    request = WorldLedgerSnapshotRequestV2(
        snapshot_request_id="world-ledger-snapshot-request://pending",
        world_ledger_ref=ledger_ref,
        required_fact_ids=fact_ids,
        policy_version="artifact-execution/r5-06-v1",
        idempotency_key="world-ledger-snapshot-idempotency://current",
        snapshot_request_sha256=HASH,
    )
    digest = world_ledger_snapshot_request_carried_sha256(request)
    return request.model_copy(
        update={
            "snapshot_request_id": (f"world-ledger-snapshot-request://sha256/{digest}"),
            "snapshot_request_sha256": digest,
        }
    )


def _execution_request(
    snapshot_ref: FacadeObjectRef,
    *,
    selected_version: str,
) -> AttachmentExecutionRequestV2:
    request = AttachmentExecutionRequestV2(
        execution_request_id="attachment-execution-request://pending",
        execution_plan_ref=_ref("artifact-execution-plan", "current"),
        execution_group_id="artifact-execution-group://current",
        artifact_id="artifact://input",
        build_spec_ref=_ref("artifact-build-spec", "input"),
        producer_task_view_ref=_ref("producer-task-view", "current"),
        content_contract_ref=_ref("artifact-content-contract", "input"),
        render_contract_ref=_ref("artifact-render-contract", "input"),
        provider_payload_ref=_ref("attachment-provider-payload", "input"),
        model_profile_ref=None,
        selected_route_kind=AttachmentExecutionRouteKindV2.PROVIDER,
        selected_route_id="text",
        selected_route_version=selected_version,
        relative_path="inputs/source.txt",
        media_type="text/plain",
        mode="PROMPT_ONLY",
        runtime_role="attachment-writer",
        required_runtime_tools=("write",),
        evidence_grants=(),
        world_ledger_snapshot_ref=snapshot_ref,
        locked_fact_ids=(),
        dependency_result_refs=(),
        attempt=1,
        retry_of_result_ref=None,
        policy_version="artifact-execution/r5-06-v1",
        idempotency_key="attachment-execution-idempotency://input/1",
        execution_request_sha256=HASH,
    )
    digest = attachment_execution_request_carried_sha256(request)
    request = request.model_copy(
        update={
            "execution_request_id": (f"attachment-execution-request://sha256/{digest}"),
            "idempotency_key": (f"attachment-execution-idempotency://sha256/{digest}"),
        }
    )
    digest = attachment_execution_request_carried_sha256(request)
    return request.model_copy(update={"execution_request_sha256": digest})


def _runtime_execution_request(
    snapshot_ref: FacadeObjectRef,
) -> AttachmentExecutionRequestV2:
    provider_request = _execution_request(
        snapshot_ref,
        selected_version="unused.Provider",
    )
    request = provider_request.model_copy(
        update={
            "execution_request_id": "attachment-execution-request://pending",
            "provider_payload_ref": None,
            "model_profile_ref": _ref("model-profile", "sdk"),
            "selected_route_kind": AttachmentExecutionRouteKindV2.RUNTIME,
            "selected_route_id": RuntimeName.CLAUDE_AGENT_SDK.value,
            "selected_route_version": "fake-runtime/1",
            "idempotency_key": "attachment-execution-idempotency://pending",
            "execution_request_sha256": HASH,
        }
    )
    seed = attachment_execution_request_carried_sha256(request)
    request = request.model_copy(
        update={
            "execution_request_id": (f"attachment-execution-request://sha256/{seed}"),
            "idempotency_key": (f"attachment-execution-idempotency://sha256/{seed}"),
        }
    )
    digest = attachment_execution_request_carried_sha256(request)
    return request.model_copy(update={"execution_request_sha256": digest})


def _provider_material() -> ProviderExecutionMaterial:
    return ProviderExecutionMaterial(
        producer_task_view_ref=_ref("producer-task-view", "current"),
        content_contract_ref=_ref("artifact-content-contract", "input"),
        render_contract_ref=_ref("artifact-render-contract", "input"),
        provider_payload_ref=_ref("attachment-provider-payload", "input"),
        authorized_evidence_ref_ids=(),
        plan=ArtifactPlan(
            artifact_id="artifact://input",
            dependency_id="attachment-dependency://input",
            relative_path="inputs/source.txt",
            asset_type="txt",
            provider="text",
            content_contract={"content": "Safe input-state content."},
            render_contract={},
            validators=["text-validator"],
        ),
    )


@pytest.mark.asyncio
async def test_world_ledger_snapshot_is_deterministic_and_content_free(
    tmp_path,
) -> None:
    ledger = WorldLedger(
        locked_facts={
            "reporting_period": "2026-Q2",
            "organization": "Example Co",
        }
    )
    ledger_ref = _ledger_ref(ledger)
    resolver = MappingAttachmentExecutionMaterialResolver(
        world_ledgers={ledger_ref.object_id: ledger},
    )
    facade = RegistryAttachmentExecutionFacade(
        ProviderRegistry(),
        RuntimeRegistry(),
        resolver=resolver,
        staging_root=tmp_path,
        clock=lambda: datetime(2026, 7, 28, tzinfo=UTC),
    )
    fact_ids = tuple(
        sorted(
            (
                world_ledger_fact_id("organization"),
                world_ledger_fact_id("reporting_period"),
            )
        )
    )

    first = await facade.snapshot_world(_snapshot_request(ledger_ref, fact_ids))
    second = await facade.snapshot_world(_snapshot_request(ledger_ref, fact_ids))

    assert first == second
    assert tuple(item.fact_id for item in first.fact_locks) == fact_ids
    serialized = str(first.model_dump(mode="json"))
    assert "Example Co" not in serialized
    assert "2026-Q2" not in serialized
    assert "organization" not in serialized
    assert "reporting_period" not in serialized


@pytest.mark.asyncio
async def test_snapshot_rejects_missing_fact_and_allowed_conflicts(
    tmp_path,
) -> None:
    conflicting_ledger = WorldLedger(
        locked_facts={"organization": "Example Co"},
        allowed_conflicts=["organization"],
    )
    ledger_ref = _ledger_ref(conflicting_ledger)
    resolver = MappingAttachmentExecutionMaterialResolver(
        world_ledgers={ledger_ref.object_id: conflicting_ledger},
    )
    facade = RegistryAttachmentExecutionFacade(
        ProviderRegistry(),
        RuntimeRegistry(),
        resolver=resolver,
        staging_root=tmp_path,
        clock=lambda: datetime(2026, 7, 28, tzinfo=UTC),
    )

    with pytest.raises(ValueError, match="allowed conflicts"):
        await facade.snapshot_world(
            _snapshot_request(
                ledger_ref,
                (world_ledger_fact_id("organization"),),
            )
        )

    clean_ledger = WorldLedger(
        locked_facts={"organization": "Example Co"},
    )
    clean_ref = _ledger_ref(clean_ledger)
    clean_facade = RegistryAttachmentExecutionFacade(
        ProviderRegistry(),
        RuntimeRegistry(),
        resolver=MappingAttachmentExecutionMaterialResolver(
            world_ledgers={clean_ref.object_id: clean_ledger},
        ),
        staging_root=tmp_path / "clean",
        clock=lambda: datetime(2026, 7, 28, tzinfo=UTC),
    )
    with pytest.raises(ValueError, match="missing locked fact"):
        await clean_facade.snapshot_world(
            _snapshot_request(
                clean_ref,
                (world_ledger_fact_id("missing"),),
            )
        )
    with pytest.raises(ValueError, match="stale or mismatched"):
        await clean_facade.snapshot_world(
            _snapshot_request(
                _ref("world-ledger", "current"),
                (),
            )
        )


@pytest.mark.asyncio
async def test_provider_execution_uses_exact_selected_implementation(
    tmp_path,
) -> None:
    ledger = WorldLedger()
    ledger_ref = _ledger_ref(ledger)
    providers = ProviderRegistry()
    providers.register(TextProvider())
    resolver = MappingAttachmentExecutionMaterialResolver(
        world_ledgers={ledger_ref.object_id: ledger},
        provider_materials={_ref("artifact-build-spec", "input").object_id: (_provider_material())},
    )
    facade = RegistryAttachmentExecutionFacade(
        providers,
        RuntimeRegistry(),
        resolver=resolver,
        staging_root=tmp_path,
        clock=lambda: datetime(2026, 7, 28, tzinfo=UTC),
    )
    snapshot = await facade.snapshot_world(_snapshot_request(ledger_ref, ()))
    implementation = f"{TextProvider.__module__}.{TextProvider.__qualname__}"

    request = _execution_request(
        world_ledger_snapshot_ref(snapshot),
        selected_version=implementation,
    )
    result = await facade.execute(request)
    replay = await facade.execute(request)

    assert result.status is AttachmentExecutionStatusV2.SUCCEEDED
    assert replay is result
    assert result.selected_route_id == "text"
    assert result.output_ref is not None
    serialized = str(result.model_dump(mode="json"))
    assert str(tmp_path) not in serialized
    assert "Safe input-state content." not in serialized

    mismatch = _execution_request(
        world_ledger_snapshot_ref(snapshot),
        selected_version="other.Provider",
    )
    mismatch_result = await facade.execute(mismatch)
    assert mismatch_result.status is (AttachmentExecutionStatusV2.BLOCKED_CAPABILITY)
    assert mismatch_result.failure_code is (AttachmentExecutionFailureCodeV2.SELECTED_IMPLEMENTATION_MISMATCH)

    conflicting = request.model_copy(
        update={
            "selected_route_version": "conflicting.Provider",
            "execution_request_sha256": HASH,
        }
    )
    conflicting = conflicting.model_copy(
        update={"execution_request_sha256": (attachment_execution_request_carried_sha256(conflicting))}
    )
    with pytest.raises(ValueError, match="idempotency conflict"):
        await facade.execute(conflicting)


@pytest.mark.asyncio
async def test_provider_material_gap_is_typed_without_fallback(
    tmp_path,
) -> None:
    ledger = WorldLedger()
    ledger_ref = _ledger_ref(ledger)
    providers = ProviderRegistry()
    providers.register(TextProvider())
    facade = RegistryAttachmentExecutionFacade(
        providers,
        RuntimeRegistry(),
        resolver=MappingAttachmentExecutionMaterialResolver(
            world_ledgers={ledger_ref.object_id: ledger},
        ),
        staging_root=tmp_path,
        clock=lambda: datetime(2026, 7, 28, tzinfo=UTC),
    )
    snapshot = await facade.snapshot_world(_snapshot_request(ledger_ref, ()))
    implementation = f"{TextProvider.__module__}.{TextProvider.__qualname__}"

    result = await facade.execute(
        _execution_request(
            world_ledger_snapshot_ref(snapshot),
            selected_version=implementation,
        )
    )

    assert result.status is AttachmentExecutionStatusV2.BLOCKED_POLICY
    assert result.failure_code is (AttachmentExecutionFailureCodeV2.MATERIAL_NOT_FOUND)


@pytest.mark.asyncio
async def test_provider_material_ref_mismatch_is_blocked(
    tmp_path,
) -> None:
    ledger = WorldLedger()
    ledger_ref = _ledger_ref(ledger)
    providers = ProviderRegistry()
    providers.register(TextProvider())
    facade = RegistryAttachmentExecutionFacade(
        providers,
        RuntimeRegistry(),
        resolver=MappingAttachmentExecutionMaterialResolver(
            world_ledgers={ledger_ref.object_id: ledger},
            provider_materials={_ref("artifact-build-spec", "input").object_id: (_provider_material())},
        ),
        staging_root=tmp_path,
        clock=lambda: datetime(2026, 7, 28, tzinfo=UTC),
    )
    snapshot = await facade.snapshot_world(_snapshot_request(ledger_ref, ()))
    implementation = f"{TextProvider.__module__}.{TextProvider.__qualname__}"
    request = _execution_request(
        world_ledger_snapshot_ref(snapshot),
        selected_version=implementation,
    ).model_copy(
        update={
            "content_contract_ref": _ref(
                "artifact-content-contract",
                "wrong",
            ),
            "execution_request_sha256": HASH,
        }
    )
    request = request.model_copy(
        update={"execution_request_sha256": (attachment_execution_request_carried_sha256(request))}
    )

    result = await facade.execute(request)

    assert result.status is AttachmentExecutionStatusV2.BLOCKED_POLICY
    assert result.failure_code is (AttachmentExecutionFailureCodeV2.MATERIAL_MISMATCH)


@pytest.mark.asyncio
async def test_runtime_execution_consumes_exact_runtime_events(
    tmp_path,
) -> None:
    ledger = WorldLedger()
    ledger_ref = _ledger_ref(ledger)
    runtimes = RuntimeRegistry()
    runtimes.register(RuntimeName.CLAUDE_AGENT_SDK, FakeRuntime())
    facade = RegistryAttachmentExecutionFacade(
        ProviderRegistry(),
        runtimes,
        resolver=MappingAttachmentExecutionMaterialResolver(
            world_ledgers={ledger_ref.object_id: ledger},
            runtime_materials={
                _ref("artifact-build-spec", "input").object_id: (
                    RuntimeExecutionMaterial(
                        producer_task_view_ref=_ref(
                            "producer-task-view",
                            "current",
                        ),
                        content_contract_ref=_ref(
                            "artifact-content-contract",
                            "input",
                        ),
                        render_contract_ref=_ref(
                            "artifact-render-contract",
                            "input",
                        ),
                        model_profile_ref=_ref("model-profile", "sdk"),
                        authorized_evidence_ref_ids=(),
                        request=RuntimeRequest(
                            run_id="execution-run://input",
                            artifact_id="artifact://placeholder",
                            role="attachment-writer",
                            workspace=str(tmp_path / "placeholder"),
                            prompt="Create safe input state.",
                            allowed_tools=["write"],
                            model_profile=ModelProfile(
                                provider="test",
                                model="fake",
                            ),
                        ),
                    )
                )
            },
        ),
        staging_root=tmp_path,
        clock=lambda: datetime(2026, 7, 28, tzinfo=UTC),
    )
    snapshot = await facade.snapshot_world(_snapshot_request(ledger_ref, ()))

    result = await facade.execute(_runtime_execution_request(world_ledger_snapshot_ref(snapshot)))

    assert result.status is AttachmentExecutionStatusV2.TERMINAL_FAILURE
    assert result.failure_code is (AttachmentExecutionFailureCodeV2.OUTPUT_MISSING)


@pytest.mark.asyncio
async def test_execution_rejects_ledger_changed_after_snapshot(
    tmp_path,
) -> None:
    ledger = WorldLedger(locked_facts={"organization": "Example Co"})
    ledger_ref = _ledger_ref(ledger)
    providers = ProviderRegistry()
    providers.register(TextProvider())
    facade = RegistryAttachmentExecutionFacade(
        providers,
        RuntimeRegistry(),
        resolver=MappingAttachmentExecutionMaterialResolver(
            world_ledgers={ledger_ref.object_id: ledger},
            provider_materials={_ref("artifact-build-spec", "input").object_id: (_provider_material())},
        ),
        staging_root=tmp_path,
        clock=lambda: datetime(2026, 7, 28, tzinfo=UTC),
    )
    snapshot = await facade.snapshot_world(_snapshot_request(ledger_ref, ()))
    ledger.locked_facts["organization"] = "Changed Co"
    implementation = f"{TextProvider.__module__}.{TextProvider.__qualname__}"

    result = await facade.execute(
        _execution_request(
            world_ledger_snapshot_ref(snapshot),
            selected_version=implementation,
        )
    )

    assert result.status is AttachmentExecutionStatusV2.BLOCKED_POLICY
    assert result.failure_code is (AttachmentExecutionFailureCodeV2.WORLD_LEDGER_STALE)


class _LedgerMutatingTextProvider(TextProvider):
    def generate(self, request: ProviderRequest) -> ArtifactResult:
        result = super().generate(request)
        request.world_ledger.locked_facts["unexpected"] = "mutation"
        return result


@pytest.mark.asyncio
async def test_provider_cannot_mutate_world_ledger_snapshot(
    tmp_path,
) -> None:
    ledger = WorldLedger(locked_facts={"organization": "Example Co"})
    ledger_ref = _ledger_ref(ledger)
    providers = ProviderRegistry()
    providers.register(_LedgerMutatingTextProvider())
    facade = RegistryAttachmentExecutionFacade(
        providers,
        RuntimeRegistry(),
        resolver=MappingAttachmentExecutionMaterialResolver(
            world_ledgers={ledger_ref.object_id: ledger},
            provider_materials={_ref("artifact-build-spec", "input").object_id: (_provider_material())},
        ),
        staging_root=tmp_path,
        clock=lambda: datetime(2026, 7, 28, tzinfo=UTC),
    )
    snapshot = await facade.snapshot_world(_snapshot_request(ledger_ref, ()))
    implementation = f"{_LedgerMutatingTextProvider.__module__}.{_LedgerMutatingTextProvider.__qualname__}"

    result = await facade.execute(
        _execution_request(
            world_ledger_snapshot_ref(snapshot),
            selected_version=implementation,
        )
    )

    assert result.status is AttachmentExecutionStatusV2.BLOCKED_POLICY
    assert result.failure_code is (AttachmentExecutionFailureCodeV2.WORLD_LEDGER_MUTATION_ATTEMPT)
    assert ledger.locked_facts == {"organization": "Example Co"}
