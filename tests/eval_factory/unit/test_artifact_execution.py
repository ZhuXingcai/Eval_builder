from __future__ import annotations

import asyncio
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pytest

from env_mock_agent.facade import (
    AttachmentExecutionFailureCodeV2,
    AttachmentExecutionRequestV2,
    AttachmentExecutionResultV2,
    AttachmentExecutionStatusV2,
    AttachmentRouteCandidateKind,
    AttachmentRouteDecisionV2,
    AttachmentRouteOutcome,
    AttachmentRouteRequestV2,
    ExecutionTelemetryV2,
    FacadeObjectRef,
    attachment_execution_request_ref,
    attachment_execution_result_carried_sha256,
    attachment_route_decision_carried_sha256,
    attachment_route_request_carried_sha256,
    attachment_route_request_ref,
    world_ledger_fact_id,
)
from env_mock_agent.facade.execution_adapter import (
    MappingAttachmentExecutionMaterialResolver,
    RegistryAttachmentExecutionFacade,
    world_ledger_object_ref,
)
from env_mock_agent.providers import ProviderRegistry
from env_mock_agent.runtimes import RuntimeRegistry
from env_mock_agent.schemas import WorldLedger
from eval_factory.attachment_planning import (
    ArtifactConsistencyDefinition,
    ArtifactExecutionPlanCompiler,
    ArtifactExecutionPolicyError,
    ArtifactExecutionPreparationBuilder,
    ArtifactGroupExecutor,
    ArtifactRoutingCompilationResult,
    ArtifactRoutingRequest,
    WorldLedgerSnapshotRunner,
    artifact_routing_compilation_result_carried_sha256,
    artifact_routing_request_carried_sha256,
    artifact_routing_request_ref,
)
from eval_factory.contracts import (
    ArtifactBuildContractV2,
    ArtifactBuildSpecV2,
    ArtifactRouteEntryOutcomeV2,
    ArtifactRouteKindV2,
    ArtifactRoutePlanEntryV2,
    ArtifactRoutingAggregateOutcomeV2,
    ArtifactRoutingPlanV2,
    artifact_build_contract_carried_sha256,
    artifact_build_contract_ref,
    artifact_build_spec_v2_carried_sha256,
    artifact_routing_plan_carried_sha256,
)
from eval_factory.contracts.attachment import ArtifactBuildSpec, ReconstructionMode
from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.contracts.task import AttachmentCriticality

HASH = "a" * 64
OTHER_HASH = "b" * 64
NOW = datetime(2026, 7, 28, tzinfo=UTC)
ROOT = Path(__file__).resolve().parents[3]
GOLD_PATH = ROOT / "evals/golden/eval_factory/attachment_execution" / "r5-06-fanout-world-ledger-v1.json"


def _audit(*refs: ObjectRef) -> ContractAudit:
    return ContractAudit(
        created_at=NOW,
        created_by="artifact-execution-test",
        governing_versions=(
            VersionBinding(
                component="artifact-execution",
                version="r5-06",
            ),
        ),
        input_refs=tuple(refs),
    )


def _ref(
    object_type: str,
    suffix: str,
    *,
    version: str = "v2",
    digest: str = HASH,
) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://{suffix}",
        object_version=version,
        object_sha256=digest,
    )


def _facade_ref(ref: ObjectRef) -> FacadeObjectRef:
    return FacadeObjectRef(
        object_type=ref.object_type,
        object_id=ref.object_id,
        object_version=ref.object_version,
        object_sha256=ref.object_sha256,
    )


def _route_request(
    artifact_id: str,
    contract: ArtifactBuildContractV2,
) -> AttachmentRouteRequestV2:
    request = AttachmentRouteRequestV2(
        route_request_id="attachment-route-request://pending",
        artifact_id=artifact_id,
        build_contract_ref=_facade_ref(artifact_build_contract_ref(contract)),
        routing_policy_ref=_facade_ref(_ref("artifact-routing-policy", "current")),
        asset_type="txt",
        media_type="text/plain",
        mode="PROMPT_ONLY",
        criticality="REQUIRED",
        provider_payload_ref=_facade_ref(contract.provider_payload_ref),
        approved_provider_ids=("text",),
        required_provider_capability_ids=("attachment-provider/generate/txt/v1",),
        runtime_order=(
            "claude_agent_sdk",
            "claude_code_cli",
            "pi_rpc",
        ),
        required_runtime_tools=("write",),
        require_runtime_resume=True,
        idempotency_key=f"attachment-route-idempotency://{artifact_id[-1]}",
        route_request_sha256=HASH,
    )
    digest = attachment_route_request_carried_sha256(request)
    return request.model_copy(
        update={
            "route_request_id": f"attachment-route-request://sha256/{digest}",
            "route_request_sha256": digest,
        }
    )


def _route_decision(
    request: AttachmentRouteRequestV2,
) -> AttachmentRouteDecisionV2:
    decision = AttachmentRouteDecisionV2(
        route_decision_id="attachment-route-decision://pending",
        route_request_ref=attachment_route_request_ref(request),
        outcome=AttachmentRouteOutcome.SELECTED_PROVIDER,
        selected_kind=AttachmentRouteCandidateKind.PROVIDER,
        selected_id="text",
        selected_version="test.TextProvider",
        satisfied_capability_ids=("attachment-provider/generate/txt/v1",),
        skipped=(),
        capability_snapshot_sha256=HASH,
        probed_at=NOW,
        policy_version="artifact-routing/r5-05-v1",
        route_decision_sha256=HASH,
    )
    digest = attachment_route_decision_carried_sha256(decision)
    return decision.model_copy(
        update={
            "route_decision_id": (f"attachment-route-decision://sha256/{digest}"),
            "route_decision_sha256": digest,
        }
    )


def _contract(
    artifact_id: str,
    path: str,
    *,
    producer_task_view_ref: ObjectRef | None = None,
) -> ArtifactBuildContractV2:
    suffix = artifact_id.rsplit("/", 1)[-1]
    contract = ArtifactBuildContractV2(
        artifact_build_contract_id="artifact-build-contract://pending",
        attachment_planning_context_ref=_ref(
            "attachment-planning-context",
            "current",
        ),
        producer_task_view_ref=(producer_task_view_ref or _ref("producer-task-view", "current")),
        artifact_evidence_matrix_ref=_ref(
            "artifact-evidence-matrix",
            "current",
        ),
        artifact_evidence_row_ref=_ref(
            "artifact-evidence-row",
            suffix,
        ),
        artifact_evidence_target_ref=_ref(
            "artifact-evidence-target",
            suffix,
        ),
        attachment_dependency_id=f"attachment-dependency://{suffix}",
        artifact_id=artifact_id,
        logical_path=path,
        media_type="text/plain",
        asset_type="txt",
        mode=ReconstructionMode.PROMPT_ONLY,
        criticality=AttachmentCriticality.REQUIRED,
        content_contract_ref=_ref("artifact-content-contract", suffix),
        render_contract_ref=_ref("artifact-render-contract", suffix),
        provider_payload_ref=_ref("attachment-provider-payload", suffix),
        source_evidence_set_ref=None,
        required_provider_capability_ids=("attachment-provider/generate/txt/v1",),
        required_runtime_tools=("write",),
        runtime_role="attachment-writer",
        runtime_resume_required=True,
        validator_ids=("secret-validator", "text-validator"),
        policy_version="artifact-routing/r5-05-v1",
        artifact_build_contract_sha256=HASH,
        audit=_audit(),
    )
    digest = artifact_build_contract_carried_sha256(contract)
    return contract.model_copy(
        update={
            "artifact_build_contract_id": (f"artifact-build-contract://sha256/{digest}"),
            "artifact_build_contract_sha256": digest,
        }
    )


def _build_spec(
    contract: ArtifactBuildContractV2,
    request: AttachmentRouteRequestV2,
    decision: AttachmentRouteDecisionV2,
) -> ArtifactBuildSpecV2:
    frozen = ArtifactBuildSpec(
        artifact_build_spec_id=(f"artifact-build-spec://{contract.artifact_id.rsplit('/', 1)[-1]}"),
        artifact_evidence_row_ref=contract.artifact_evidence_row_ref,
        producer_task_view_ref=contract.producer_task_view_ref,
        relative_path=contract.logical_path,
        media_type=contract.media_type,
        mode=contract.mode,
        content_contract_ref=contract.content_contract_ref,
        render_contract_ref=contract.render_contract_ref,
        authorized_evidence_refs=(),
        provider_preference=("text",),
        runtime_preference=(),
        validator_ids=contract.validator_ids,
        retry_scope="ARTIFACT",
        build_spec_sha256=OTHER_HASH,
        audit=_audit(),
    )
    spec = ArtifactBuildSpecV2(
        artifact_build_spec_v2_id="artifact-build-spec://pending",
        attachment_planning_context_ref=(contract.attachment_planning_context_ref),
        artifact_evidence_matrix_ref=(contract.artifact_evidence_matrix_ref),
        artifact_evidence_target_ref=(contract.artifact_evidence_target_ref),
        artifact_build_contract_ref=artifact_build_contract_ref(contract),
        artifact_routing_policy_ref=_ref(
            "artifact-routing-policy",
            "current",
        ),
        facade_route_request_ref=_ref(
            "attachment-route-request",
            "current",
        ).model_copy(
            update={
                "object_id": request.route_request_id,
                "object_sha256": request.route_request_sha256,
            }
        ),
        facade_route_decision_ref=_ref(
            "attachment-route-decision",
            "current",
        ).model_copy(
            update={
                "object_id": decision.route_decision_id,
                "object_sha256": decision.route_decision_sha256,
            }
        ),
        selected_route_kind=ArtifactRouteKindV2.PROVIDER,
        selected_model_profile_ref=None,
        source_evidence_set_ref=None,
        build_spec=frozen,
        policy_version="artifact-routing/r5-05-v1",
        artifact_build_spec_v2_sha256=HASH,
        audit=_audit(),
    )
    digest = artifact_build_spec_v2_carried_sha256(spec)
    return spec.model_copy(
        update={
            "artifact_build_spec_v2_id": (f"artifact-build-spec://sha256/{digest}"),
            "artifact_build_spec_v2_sha256": digest,
        }
    )


def _routing_source(
    artifacts: tuple[tuple[str, str], ...],
    *,
    producer_task_view_ref: ObjectRef | None = None,
) -> tuple[ArtifactRoutingRequest, ArtifactRoutingCompilationResult]:
    contracts = tuple(
        sorted(
            (
                _contract(
                    artifact_id,
                    path,
                    producer_task_view_ref=producer_task_view_ref,
                )
                for artifact_id, path in artifacts
            ),
            key=lambda item: (item.logical_path, item.artifact_id),
        )
    )
    facade_requests = tuple(_route_request(contract.artifact_id, contract) for contract in contracts)
    decisions = tuple(_route_decision(item) for item in facade_requests)
    entries = tuple(
        ArtifactRoutePlanEntryV2(
            artifact_evidence_target_ref=contract.artifact_evidence_target_ref,
            attachment_dependency_id=contract.attachment_dependency_id,
            artifact_id=contract.artifact_id,
            criticality=contract.criticality,
            outcome=ArtifactRouteEntryOutcomeV2.ROUTED_PROVIDER,
            facade_route_request_ref=_ref(
                "attachment-route-request",
                "current",
            ).model_copy(
                update={
                    "object_id": request.route_request_id,
                    "object_sha256": request.route_request_sha256,
                }
            ),
            facade_route_decision=decision,
            build_spec=_build_spec(contract, request, decision),
            reasons=frozenset(),
        )
        for contract, request, decision in zip(
            contracts,
            facade_requests,
            decisions,
            strict=True,
        )
    )
    routing_plan = ArtifactRoutingPlanV2(
        artifact_routing_plan_id="artifact-routing-plan://pending",
        attachment_planning_context_ref=_ref(
            "attachment-planning-context",
            "current",
        ),
        producer_task_view_ref=(producer_task_view_ref or _ref("producer-task-view", "current")),
        artifact_evidence_matrix_ref=_ref(
            "artifact-evidence-matrix",
            "current",
        ),
        artifact_routing_policy_ref=_ref(
            "artifact-routing-policy",
            "current",
        ),
        entries=entries,
        aggregate_outcome=ArtifactRoutingAggregateOutcomeV2.ROUTED,
        routed_artifact_ids=tuple(sorted(item.artifact_id for item in entries)),
        blocked_required_artifact_ids=(),
        blocked_optional_artifact_ids=(),
        policy_version="artifact-routing/r5-05-v1",
        artifact_routing_plan_sha256=HASH,
        audit=_audit(),
    )
    plan_digest = artifact_routing_plan_carried_sha256(routing_plan)
    routing_plan = routing_plan.model_copy(
        update={
            "artifact_routing_plan_id": (f"artifact-routing-plan://sha256/{plan_digest}"),
            "artifact_routing_plan_sha256": plan_digest,
        }
    )
    request = ArtifactRoutingRequest(
        request_id="artifact-routing-request://pending",
        attachment_planning_context_ref=(routing_plan.attachment_planning_context_ref),
        producer_task_view_ref=routing_plan.producer_task_view_ref,
        artifact_evidence_matrix_ref=(routing_plan.artifact_evidence_matrix_ref),
        artifact_routing_policy_ref=(routing_plan.artifact_routing_policy_ref),
        build_contracts=contracts,
        facade_requests=facade_requests,
        missing_contract_target_refs=(),
        blocked_mode_target_refs=(),
        policy_version="artifact-routing/r5-05-v1",
        request_sha256=HASH,
        audit=_audit(),
    )
    request_digest = artifact_routing_request_carried_sha256(request)
    request = request.model_copy(
        update={
            "request_id": f"artifact-routing-request://sha256/{request_digest}",
            "request_sha256": request_digest,
        }
    )
    result = ArtifactRoutingCompilationResult(
        result_id="artifact-routing-compilation-result://pending",
        request_ref=artifact_routing_request_ref(request),
        execution_result_ref=_ref(
            "artifact-routing-execution-result",
            "current",
            version="r5-05",
        ),
        outcome=ArtifactRoutingAggregateOutcomeV2.ROUTED,
        routing_plan=routing_plan,
        policy_version="artifact-routing/r5-05-v1",
        result_sha256=HASH,
        audit=_audit(),
    )
    result_digest = artifact_routing_compilation_result_carried_sha256(result)
    result = result.model_copy(
        update={
            "result_id": (f"artifact-routing-compilation-result://sha256/{result_digest}"),
            "result_sha256": result_digest,
        }
    )
    return request, result


async def _execution_plan(
    tmp_path: Path,
    artifacts: tuple[tuple[str, str], ...],
    definitions: tuple[ArtifactConsistencyDefinition, ...],
    *,
    producer_task_view_ref: ObjectRef | None = None,
):
    request, result = _routing_source(
        artifacts,
        producer_task_view_ref=producer_task_view_ref,
    )
    fact_values = {
        "organization": "Example Co",
        "reporting_period": "2026-Q2",
    }
    ledger = WorldLedger(locked_facts=fact_values)
    ledger_ref = world_ledger_object_ref("world-ledger://current", ledger)
    facade = RegistryAttachmentExecutionFacade(
        ProviderRegistry(),
        RuntimeRegistry(),
        resolver=MappingAttachmentExecutionMaterialResolver(world_ledgers={ledger_ref.object_id: ledger}),
        staging_root=tmp_path,
        clock=lambda: NOW,
    )
    preparation = ArtifactExecutionPreparationBuilder().build(
        routing_request=request,
        routing_result=result,
        definitions=definitions,
        world_ledger_ref=ledger_ref,
        audit=_audit(),
    )
    snapshot = await WorldLedgerSnapshotRunner().run(
        preparation,
        facade=facade,
    )
    planning = ArtifactExecutionPlanCompiler().compile(
        routing_request=request,
        routing_result=result,
        preparation=preparation,
        world_ledger_snapshot=snapshot,
        audit=_audit(),
    )
    assert planning.execution_plan is not None
    return planning.execution_plan


class _ControlledExecutionFacade:
    def __init__(
        self,
        *,
        fail_once: frozenset[str] = frozenset(),
        expected_parallelism: int = 1,
    ) -> None:
        self.fail_once = fail_once
        self.expected_parallelism = expected_parallelism
        self.calls: Counter[str] = Counter()
        self.active = 0
        self.max_active = 0
        self.release = asyncio.Event()

    async def snapshot_world(self, request):
        raise AssertionError(f"snapshot is not expected: {request}")

    async def execute(
        self,
        request: AttachmentExecutionRequestV2,
    ) -> AttachmentExecutionResultV2:
        self.calls[request.artifact_id] += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        if self.active >= self.expected_parallelism:
            self.release.set()
        await asyncio.wait_for(self.release.wait(), timeout=1)
        await asyncio.sleep(0.01)
        self.active -= 1
        if request.artifact_id in self.fail_once and self.calls[request.artifact_id] == 1:
            return self._result(
                request,
                status=AttachmentExecutionStatusV2.RETRYABLE_FAILURE,
            )
        return self._result(
            request,
            status=AttachmentExecutionStatusV2.SUCCEEDED,
        )

    @staticmethod
    def _result(
        request: AttachmentExecutionRequestV2,
        *,
        status: AttachmentExecutionStatusV2,
    ) -> AttachmentExecutionResultV2:
        succeeded = status is AttachmentExecutionStatusV2.SUCCEEDED
        output_sha256 = hashlib.sha256(request.artifact_id.encode()).hexdigest()
        result = AttachmentExecutionResultV2(
            execution_result_id="attachment-execution-result://pending",
            execution_request_ref=attachment_execution_request_ref(request),
            artifact_id=request.artifact_id,
            attempt=request.attempt,
            status=status,
            world_ledger_snapshot_ref=request.world_ledger_snapshot_ref,
            selected_route_kind=request.selected_route_kind,
            selected_route_id=request.selected_route_id,
            worker_version="test.TextProvider" if succeeded else None,
            output_ref=(
                FacadeObjectRef(
                    object_type="attachment-output",
                    object_id=f"attachment-output://sha256/{output_sha256}",
                    object_version="v2",
                    object_sha256=output_sha256,
                )
                if succeeded
                else None
            ),
            output_sha256=output_sha256 if succeeded else None,
            retryable=not succeeded,
            failure_code=(None if succeeded else AttachmentExecutionFailureCodeV2.RUNTIME_EXECUTION_FAILED),
            telemetry=ExecutionTelemetryV2.unavailable(observed_at=NOW),
            policy_version="artifact-execution/r5-06-v1",
            execution_result_sha256=HASH,
        )
        digest = attachment_execution_result_carried_sha256(result)
        return result.model_copy(
            update={
                "execution_result_id": (f"attachment-execution-result://sha256/{digest}"),
                "execution_result_sha256": digest,
            }
        )


def _definition(
    artifact_id: str,
    *,
    dependencies: tuple[str, ...] = (),
    facts: tuple[str, ...] = (),
) -> ArtifactConsistencyDefinition:
    return ArtifactConsistencyDefinition(
        artifact_id=artifact_id,
        dependency_artifact_ids=dependencies,
        locked_fact_ids=facts,
    )


@pytest.mark.asyncio
async def test_independent_groups_execute_concurrently_and_shared_facts_serialize(
    tmp_path: Path,
) -> None:
    artifacts = (
        ("artifact://a", "inputs/a.txt"),
        ("artifact://b", "inputs/b.txt"),
    )
    independent = await _execution_plan(
        tmp_path / "independent",
        artifacts,
        (
            _definition("artifact://a"),
            _definition("artifact://b"),
        ),
    )
    assert len(independent.groups) == 2
    concurrent_facade = _ControlledExecutionFacade(expected_parallelism=2)
    concurrent_batch = await ArtifactGroupExecutor().run(
        independent,
        facade=concurrent_facade,
        audit=_audit(),
    )
    assert concurrent_facade.max_active == 2
    assert concurrent_batch.succeeded_artifact_ids == (
        "artifact://a",
        "artifact://b",
    )

    organization = world_ledger_fact_id("organization")
    shared = await _execution_plan(
        tmp_path / "shared",
        artifacts,
        (
            _definition("artifact://a", facts=(organization,)),
            _definition("artifact://b", facts=(organization,)),
        ),
    )
    assert len(shared.groups) == 1
    serial_facade = _ControlledExecutionFacade(expected_parallelism=1)
    await ArtifactGroupExecutor().run(
        shared,
        facade=serial_facade,
        audit=_audit(),
    )
    assert serial_facade.max_active == 1


@pytest.mark.asyncio
async def test_retry_reuses_success_and_runs_only_retryable_artifact(
    tmp_path: Path,
) -> None:
    plan = await _execution_plan(
        tmp_path,
        (
            ("artifact://a", "inputs/a.txt"),
            ("artifact://b", "inputs/b.txt"),
        ),
        (
            _definition("artifact://a"),
            _definition("artifact://b"),
        ),
    )
    facade = _ControlledExecutionFacade(
        fail_once=frozenset({"artifact://b"}),
        expected_parallelism=2,
    )
    executor = ArtifactGroupExecutor()

    first = await executor.run(plan, facade=facade, audit=_audit())
    second = await executor.run(
        plan,
        facade=facade,
        audit=_audit(),
        prior_batch=first,
    )

    assert first.retryable_artifact_ids == ("artifact://b",)
    assert second.succeeded_artifact_ids == ("artifact://a", "artifact://b")
    assert facade.calls == Counter(
        {
            "artifact://a": 1,
            "artifact://b": 2,
        }
    )
    first_a = next(item for item in first.receipts if item.artifact_id == "artifact://a")
    second_a = next(item for item in second.receipts if item.artifact_id == "artifact://a")
    assert second_a is first_a
    executor.validate_current(plan, second)

    tampered_receipt = second.receipts[0].model_copy(
        update={
            "audit": second.receipts[0].audit.model_copy(
                update={"input_refs": (_ref("private-reference", "caller-injected"),)}
            )
        }
    )
    tampered_batch = second.model_copy(
        update={
            "receipts": (
                tampered_receipt,
                *second.receipts[1:],
            )
        }
    )
    with pytest.raises(ArtifactExecutionPolicyError, match="audit lineage"):
        executor.validate_current(plan, tampered_batch)


@pytest.mark.asyncio
async def test_dependency_unblocks_after_retry_without_running_early(
    tmp_path: Path,
) -> None:
    plan = await _execution_plan(
        tmp_path,
        (
            ("artifact://a", "inputs/a.txt"),
            ("artifact://b", "inputs/b.txt"),
        ),
        (
            _definition("artifact://a"),
            _definition(
                "artifact://b",
                dependencies=("artifact://a",),
            ),
        ),
    )
    facade = _ControlledExecutionFacade(
        fail_once=frozenset({"artifact://a"}),
    )
    executor = ArtifactGroupExecutor()

    first = await executor.run(plan, facade=facade, audit=_audit())
    assert first.dependency_blocked_artifact_ids == ("artifact://b",)
    assert facade.calls["artifact://b"] == 0

    second = await executor.run(
        plan,
        facade=facade,
        audit=_audit(),
        prior_batch=first,
    )
    assert second.succeeded_artifact_ids == ("artifact://a", "artifact://b")
    assert facade.calls == Counter(
        {
            "artifact://a": 2,
            "artifact://b": 1,
        }
    )


@pytest.mark.asyncio
async def test_plan_rejects_ancestor_descendant_output_ownership(
    tmp_path: Path,
) -> None:
    request, result = _routing_source(
        (
            ("artifact://a", "inputs"),
            ("artifact://b", "inputs/b.txt"),
        )
    )
    ledger = WorldLedger()
    ledger_ref = world_ledger_object_ref("world-ledger://current", ledger)
    facade = RegistryAttachmentExecutionFacade(
        ProviderRegistry(),
        RuntimeRegistry(),
        resolver=MappingAttachmentExecutionMaterialResolver(
            world_ledgers={
                ledger_ref.object_id: ledger,
            }
        ),
        staging_root=tmp_path,
        clock=lambda: NOW,
    )
    preparation = ArtifactExecutionPreparationBuilder().build(
        routing_request=request,
        routing_result=result,
        definitions=(
            _definition("artifact://a"),
            _definition("artifact://b"),
        ),
        world_ledger_ref=ledger_ref,
        audit=_audit(),
    )
    snapshot = await WorldLedgerSnapshotRunner().run(
        preparation,
        facade=facade,
    )

    with pytest.raises(ArtifactExecutionPolicyError, match="paths overlap"):
        ArtifactExecutionPlanCompiler().compile(
            routing_request=request,
            routing_result=result,
            preparation=preparation,
            world_ledger_snapshot=snapshot,
            audit=_audit(),
        )


@pytest.mark.asyncio
async def test_execution_group_gold_is_executable(tmp_path: Path) -> None:
    payload = json.loads(GOLD_PATH.read_text(encoding="utf-8"))

    for case in payload["cases"]:
        artifacts = tuple((item["artifact_id"], item["logical_path"]) for item in case["artifacts"])
        definitions = tuple(
            _definition(
                item["artifact_id"],
                dependencies=tuple(item["dependencies"]),
                facts=tuple(sorted(world_ledger_fact_id(key) for key in item["locked_fact_keys"])),
            )
            for item in case["artifacts"]
        )
        plan = await _execution_plan(
            tmp_path / case["case_id"],
            artifacts,
            definitions,
        )

        assert [[unit.artifact_id for unit in group.units] for group in plan.groups] == case[
            "expected_groups"
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("definitions", "message"),
    [
        (
            (
                _definition(
                    "artifact://a",
                    dependencies=("artifact://missing",),
                ),
                _definition("artifact://b"),
            ),
            "unknown routed artifact",
        ),
        (
            (
                _definition(
                    "artifact://a",
                    dependencies=("artifact://b",),
                ),
                _definition(
                    "artifact://b",
                    dependencies=("artifact://a",),
                ),
            ),
            "contains a cycle",
        ),
    ],
)
async def test_plan_rejects_invalid_dependency_graph(
    tmp_path: Path,
    definitions: tuple[ArtifactConsistencyDefinition, ...],
    message: str,
) -> None:
    with pytest.raises(ArtifactExecutionPolicyError, match=message):
        await _execution_plan(
            tmp_path,
            (
                ("artifact://a", "inputs/a.txt"),
                ("artifact://b", "inputs/b.txt"),
            ),
            definitions,
        )
