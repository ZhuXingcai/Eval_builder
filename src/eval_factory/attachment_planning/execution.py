from __future__ import annotations

import asyncio
from collections.abc import Iterable
from pathlib import PurePosixPath

from env_mock_agent.facade import (
    ARTIFACT_EXECUTION_POLICY_VERSION,
    AttachmentExecutionEvidenceGrantV2,
    AttachmentExecutionFacade,
    AttachmentExecutionRequestV2,
    AttachmentExecutionResultV2,
    AttachmentExecutionRouteKindV2,
    AttachmentExecutionSourceSpanV2,
    FacadeObjectRef,
    WorldLedgerSnapshotRequestV2,
    WorldLedgerSnapshotV2,
    attachment_execution_request_carried_sha256,
    attachment_execution_request_ref,
    attachment_execution_result_ref,
    attachment_route_decision_carried_sha256,
    validate_attachment_execution_result_identity,
    validate_world_ledger_snapshot_identity,
    world_ledger_snapshot_ref,
    world_ledger_snapshot_request_carried_sha256,
    world_ledger_snapshot_request_ref,
)
from env_mock_agent.facade import (
    ReconstructionMode as FacadeReconstructionMode,
)
from eval_factory.attachment_planning.execution_models import (
    ArtifactConsistencyDefinition,
    ArtifactExecutionPlanningOutcome,
    ArtifactExecutionPlanningResult,
    ArtifactExecutionPolicyError,
    ArtifactExecutionPreparation,
)
from eval_factory.attachment_planning.routing_models import (
    ArtifactRoutingCompilationResult,
    ArtifactRoutingRequest,
    artifact_routing_compilation_result_carried_sha256,
    artifact_routing_request_carried_sha256,
    artifact_routing_request_ref,
)
from eval_factory.contracts.attachment_v2 import (
    ArtifactBuildContractV2,
    ArtifactBuildSpecV2,
    ArtifactExecutionBatchV2,
    ArtifactExecutionGroupV2,
    ArtifactExecutionPlanV2,
    ArtifactExecutionReceiptOutcomeV2,
    ArtifactExecutionReceiptV2,
    ArtifactExecutionUnitV2,
    ArtifactRouteEntryOutcomeV2,
    ArtifactRouteKindV2,
    artifact_build_contract_carried_sha256,
    artifact_build_spec_v2_carried_sha256,
    artifact_build_spec_v2_ref,
    artifact_execution_batch_carried_sha256,
    artifact_execution_batch_ref,
    artifact_execution_group_carried_sha256,
    artifact_execution_group_ref,
    artifact_execution_plan_carried_sha256,
    artifact_execution_plan_ref,
    artifact_execution_receipt_carried_sha256,
    artifact_execution_receipt_ref,
    artifact_execution_unit_carried_sha256,
    artifact_execution_unit_ref,
    artifact_routing_plan_carried_sha256,
    artifact_routing_plan_ref,
)
from eval_factory.contracts.core import (
    ContractAudit,
    EvidenceRef,
    ObjectRef,
)


class ArtifactExecutionPreparationBuilder:
    def build(
        self,
        *,
        routing_request: ArtifactRoutingRequest,
        routing_result: ArtifactRoutingCompilationResult,
        definitions: tuple[ArtifactConsistencyDefinition, ...],
        world_ledger_ref: FacadeObjectRef,
        audit: ContractAudit,
    ) -> ArtifactExecutionPreparation:
        _validate_routing_source(routing_request, routing_result)
        if routing_result.routing_plan is None:
            if definitions:
                raise ArtifactExecutionPolicyError(
                    "non-routed execution cannot contain consistency definitions"
                )
            return ArtifactExecutionPreparation(
                outcome=ArtifactExecutionPlanningOutcome.NOT_REQUIRED,
                snapshot_request=None,
                definitions=(),
                audit=_safe_audit(audit, (routing_result.request_ref,)),
            )
        routed_ids = routing_result.routing_plan.routed_artifact_ids
        if not routed_ids:
            if definitions:
                raise ArtifactExecutionPolicyError("no-routed-artifacts execution cannot contain definitions")
            return ArtifactExecutionPreparation(
                outcome=ArtifactExecutionPlanningOutcome.NO_ROUTED_ARTIFACTS,
                snapshot_request=None,
                definitions=(),
                audit=_safe_audit(
                    audit,
                    (artifact_routing_plan_ref(routing_result.routing_plan),),
                ),
            )
        normalized = tuple(sorted(definitions, key=lambda item: item.artifact_id))
        definition_ids = tuple(item.artifact_id for item in normalized)
        if len(definition_ids) != len(set(definition_ids)):
            raise ArtifactExecutionPolicyError("consistency definitions must have unique artifact IDs")
        if definition_ids != tuple(sorted(routed_ids)):
            raise ArtifactExecutionPolicyError("consistency definitions must exactly cover routed artifacts")
        fact_ids = tuple(
            sorted({fact_id for definition in normalized for fact_id in definition.locked_fact_ids})
        )
        request = WorldLedgerSnapshotRequestV2(
            snapshot_request_id="world-ledger-snapshot-request://pending",
            world_ledger_ref=world_ledger_ref,
            required_fact_ids=fact_ids,
            policy_version=ARTIFACT_EXECUTION_POLICY_VERSION,
            idempotency_key="world-ledger-snapshot-idempotency://pending",
            snapshot_request_sha256="0" * 64,
        )
        digest = world_ledger_snapshot_request_carried_sha256(request)
        request = request.model_copy(
            update={
                "snapshot_request_id": (f"world-ledger-snapshot-request://sha256/{digest}"),
                "idempotency_key": (f"world-ledger-snapshot-idempotency://sha256/{digest}"),
            }
        )
        digest = world_ledger_snapshot_request_carried_sha256(request)
        request = request.model_copy(update={"snapshot_request_sha256": digest})
        return ArtifactExecutionPreparation(
            outcome=ArtifactExecutionPlanningOutcome.PLANNED,
            snapshot_request=request,
            definitions=normalized,
            audit=_safe_audit(
                audit,
                (
                    artifact_routing_plan_ref(routing_result.routing_plan),
                    _object_ref_from_facade(world_ledger_ref),
                ),
            ),
        )


class WorldLedgerSnapshotRunner:
    async def run(
        self,
        preparation: ArtifactExecutionPreparation,
        *,
        facade: AttachmentExecutionFacade,
    ) -> WorldLedgerSnapshotV2 | None:
        if preparation.outcome is not ArtifactExecutionPlanningOutcome.PLANNED:
            return None
        if preparation.snapshot_request is None:
            raise ArtifactExecutionPolicyError("planned preparation is missing snapshot request")
        snapshot = await facade.snapshot_world(preparation.snapshot_request)
        validate_world_ledger_snapshot_identity(snapshot)
        if (
            snapshot.snapshot_request_ref != world_ledger_snapshot_request_ref(preparation.snapshot_request)
            or snapshot.world_ledger_ref != preparation.snapshot_request.world_ledger_ref
        ):
            raise ArtifactExecutionPolicyError("WorldLedger snapshot does not bind its exact request")
        expected_fact_ids = preparation.snapshot_request.required_fact_ids
        if tuple(item.fact_id for item in snapshot.fact_locks) != expected_fact_ids:
            raise ArtifactExecutionPolicyError("WorldLedger snapshot facts do not match the request")
        return snapshot


class ArtifactExecutionPlanCompiler:
    def compile(
        self,
        *,
        routing_request: ArtifactRoutingRequest,
        routing_result: ArtifactRoutingCompilationResult,
        preparation: ArtifactExecutionPreparation,
        world_ledger_snapshot: WorldLedgerSnapshotV2 | None,
        audit: ContractAudit,
    ) -> ArtifactExecutionPlanningResult:
        _validate_routing_source(routing_request, routing_result)
        if preparation.outcome is not ArtifactExecutionPlanningOutcome.PLANNED:
            if world_ledger_snapshot is not None:
                raise ArtifactExecutionPolicyError("non-planned execution cannot contain a ledger snapshot")
            return ArtifactExecutionPlanningResult(
                outcome=preparation.outcome,
                world_ledger_snapshot=None,
                execution_plan=None,
                audit=_safe_audit(audit, (routing_result.request_ref,)),
            )
        if (
            preparation.snapshot_request is None
            or world_ledger_snapshot is None
            or routing_result.routing_plan is None
        ):
            raise ArtifactExecutionPolicyError("planned execution requires routing plan and ledger snapshot")
        validate_world_ledger_snapshot_identity(world_ledger_snapshot)
        if world_ledger_snapshot.snapshot_request_ref != (
            world_ledger_snapshot_request_ref(preparation.snapshot_request)
        ):
            raise ArtifactExecutionPolicyError("ledger snapshot is stale for execution preparation")
        definitions = {item.artifact_id: item for item in preparation.definitions}
        routed_entries = tuple(
            entry
            for entry in routing_result.routing_plan.entries
            if entry.outcome
            in {
                ArtifactRouteEntryOutcomeV2.ROUTED_PROVIDER,
                ArtifactRouteEntryOutcomeV2.ROUTED_RUNTIME,
            }
        )
        contracts = {item.artifact_id: item for item in routing_request.build_contracts}
        if set(contracts) < {entry.artifact_id for entry in routed_entries}:
            raise ArtifactExecutionPolicyError("routing request is missing a routed build contract")
        paths = {
            entry.artifact_id: entry.build_spec.build_spec.relative_path
            for entry in routed_entries
            if entry.build_spec is not None
        }
        _validate_path_ownership(paths)
        _validate_dependencies(definitions)
        components = _connected_components(definitions)
        groups: list[ArtifactExecutionGroupV2] = []
        entry_by_id = {entry.artifact_id: entry for entry in routed_entries}
        for component in components:
            ordered_ids = _topological_order(
                component,
                definitions=definitions,
                paths=paths,
            )
            units: list[ArtifactExecutionUnitV2] = []
            for index, artifact_id in enumerate(ordered_ids):
                entry = entry_by_id[artifact_id]
                if entry.build_spec is None or entry.facade_route_decision is None:
                    raise ArtifactExecutionPolicyError("routed entry is missing build or route facts")
                contract = contracts[artifact_id]
                _validate_build_facts(contract, entry.build_spec)
                definition = definitions[artifact_id]
                unit = ArtifactExecutionUnitV2(
                    artifact_execution_unit_id=("artifact-execution-unit://pending"),
                    artifact_id=artifact_id,
                    logical_path=paths[artifact_id],
                    build_contract=contract,
                    build_spec=entry.build_spec,
                    facade_route_decision=entry.facade_route_decision,
                    dependency_artifact_ids=(definition.dependency_artifact_ids),
                    locked_fact_ids=definition.locked_fact_ids,
                    order_index=index,
                    policy_version=ARTIFACT_EXECUTION_POLICY_VERSION,
                    artifact_execution_unit_sha256="0" * 64,
                )
                unit_digest = artifact_execution_unit_carried_sha256(unit)
                units.append(
                    unit.model_copy(
                        update={
                            "artifact_execution_unit_id": (f"artifact-execution-unit://sha256/{unit_digest}"),
                            "artifact_execution_unit_sha256": unit_digest,
                        }
                    )
                )
            locked_facts = tuple(sorted({fact_id for unit in units for fact_id in unit.locked_fact_ids}))
            group = ArtifactExecutionGroupV2(
                artifact_execution_group_id=("artifact-execution-group://pending"),
                units=tuple(units),
                locked_fact_ids=locked_facts,
                policy_version=ARTIFACT_EXECUTION_POLICY_VERSION,
                artifact_execution_group_sha256="0" * 64,
            )
            group_digest = artifact_execution_group_carried_sha256(group)
            groups.append(
                group.model_copy(
                    update={
                        "artifact_execution_group_id": (f"artifact-execution-group://sha256/{group_digest}"),
                        "artifact_execution_group_sha256": group_digest,
                    }
                )
            )
        groups.sort(
            key=lambda group: (
                group.units[0].logical_path,
                group.units[0].artifact_id,
            )
        )
        providers = tuple(
            sorted(
                {
                    unit.facade_route_decision.selected_id
                    for group in groups
                    for unit in group.units
                    if unit.build_spec.selected_route_kind is ArtifactRouteKindV2.PROVIDER
                    and unit.facade_route_decision.selected_id is not None
                }
            )
        )
        runtimes = tuple(
            sorted(
                {
                    unit.facade_route_decision.selected_id
                    for group in groups
                    for unit in group.units
                    if unit.build_spec.selected_route_kind is ArtifactRouteKindV2.RUNTIME
                    and unit.facade_route_decision.selected_id is not None
                }
            )
        )
        plan = ArtifactExecutionPlanV2(
            artifact_execution_plan_id="artifact-execution-plan://pending",
            artifact_routing_plan_ref=artifact_routing_plan_ref(routing_result.routing_plan),
            world_ledger_snapshot=world_ledger_snapshot,
            groups=tuple(groups),
            routed_artifact_ids=tuple(sorted(entry.artifact_id for entry in routed_entries)),
            selected_provider_ids=providers,
            selected_runtime_ids=runtimes,
            policy_version=ARTIFACT_EXECUTION_POLICY_VERSION,
            artifact_execution_plan_sha256="0" * 64,
            audit=_safe_audit(
                audit,
                (
                    artifact_routing_plan_ref(routing_result.routing_plan),
                    _object_ref_from_facade(world_ledger_snapshot_ref(world_ledger_snapshot)),
                ),
            ),
        )
        plan_digest = artifact_execution_plan_carried_sha256(plan)
        plan = plan.model_copy(
            update={
                "artifact_execution_plan_id": (f"artifact-execution-plan://sha256/{plan_digest}"),
                "artifact_execution_plan_sha256": plan_digest,
            }
        )
        return ArtifactExecutionPlanningResult(
            outcome=ArtifactExecutionPlanningOutcome.PLANNED,
            world_ledger_snapshot=world_ledger_snapshot,
            execution_plan=plan,
            audit=_safe_audit(
                audit,
                (artifact_execution_plan_ref(plan),),
            ),
        )

    def validate_current(
        self,
        result: ArtifactExecutionPlanningResult,
    ) -> None:
        if result.outcome is not ArtifactExecutionPlanningOutcome.PLANNED:
            if result.execution_plan is not None or result.world_ledger_snapshot is not None:
                raise ArtifactExecutionPolicyError("non-planned execution result contains work")
            return
        if result.execution_plan is None or result.world_ledger_snapshot is None:
            raise ArtifactExecutionPolicyError("planned execution result is incomplete")
        _validate_execution_plan_identity(result.execution_plan)
        if result.execution_plan.world_ledger_snapshot != (result.world_ledger_snapshot):
            raise ArtifactExecutionPolicyError("execution plan snapshot is mismatched")


class ArtifactGroupExecutor:
    async def run(
        self,
        plan: ArtifactExecutionPlanV2,
        *,
        facade: AttachmentExecutionFacade,
        audit: ContractAudit,
        prior_batch: ArtifactExecutionBatchV2 | None = None,
    ) -> ArtifactExecutionBatchV2:
        _validate_execution_plan_identity(plan)
        prior_by_artifact: dict[str, ArtifactExecutionReceiptV2] = {}
        if prior_batch is not None:
            _validate_execution_batch_identity(prior_batch)
            if prior_batch.artifact_execution_plan_ref != artifact_execution_plan_ref(plan):
                raise ArtifactExecutionPolicyError("prior execution batch belongs to another plan")
            prior_by_artifact = {receipt.artifact_id: receipt for receipt in prior_batch.receipts}
        group_receipts = await asyncio.gather(
            *(
                self._run_group(
                    plan,
                    group,
                    facade=facade,
                    audit=audit,
                    prior_by_artifact=prior_by_artifact,
                )
                for group in plan.groups
            )
        )
        receipts = tuple(receipt for group_result in group_receipts for receipt in group_result)
        batch = ArtifactExecutionBatchV2(
            artifact_execution_batch_id="artifact-execution-batch://pending",
            artifact_execution_plan_ref=artifact_execution_plan_ref(plan),
            prior_batch_ref=(artifact_execution_batch_ref(prior_batch) if prior_batch is not None else None),
            receipts=receipts,
            succeeded_artifact_ids=_artifact_ids(
                receipts,
                {ArtifactExecutionReceiptOutcomeV2.SUCCEEDED},
            ),
            retryable_artifact_ids=_artifact_ids(
                receipts,
                {ArtifactExecutionReceiptOutcomeV2.RETRYABLE_FAILURE},
            ),
            terminal_artifact_ids=_artifact_ids(
                receipts,
                {
                    ArtifactExecutionReceiptOutcomeV2.BLOCKED_CAPABILITY,
                    ArtifactExecutionReceiptOutcomeV2.BLOCKED_POLICY,
                    ArtifactExecutionReceiptOutcomeV2.TERMINAL_FAILURE,
                },
            ),
            dependency_blocked_artifact_ids=_artifact_ids(
                receipts,
                {ArtifactExecutionReceiptOutcomeV2.BLOCKED_DEPENDENCY},
            ),
            world_ledger_snapshot_ref=_object_ref_from_facade(
                world_ledger_snapshot_ref(plan.world_ledger_snapshot)
            ),
            policy_version=ARTIFACT_EXECUTION_POLICY_VERSION,
            artifact_execution_batch_sha256="0" * 64,
            audit=_safe_audit(
                audit,
                (
                    artifact_execution_plan_ref(plan),
                    *tuple(artifact_execution_receipt_ref(receipt) for receipt in receipts),
                ),
            ),
        )
        digest = artifact_execution_batch_carried_sha256(batch)
        return batch.model_copy(
            update={
                "artifact_execution_batch_id": (f"artifact-execution-batch://sha256/{digest}"),
                "artifact_execution_batch_sha256": digest,
            }
        )

    async def _run_group(
        self,
        plan: ArtifactExecutionPlanV2,
        group: ArtifactExecutionGroupV2,
        *,
        facade: AttachmentExecutionFacade,
        audit: ContractAudit,
        prior_by_artifact: dict[str, ArtifactExecutionReceiptV2],
    ) -> tuple[ArtifactExecutionReceiptV2, ...]:
        current: dict[str, ArtifactExecutionReceiptV2] = {}
        receipts: list[ArtifactExecutionReceiptV2] = []
        for unit in group.units:
            prior = prior_by_artifact.get(unit.artifact_id)
            if prior is not None and prior.outcome not in {
                ArtifactExecutionReceiptOutcomeV2.RETRYABLE_FAILURE,
                ArtifactExecutionReceiptOutcomeV2.BLOCKED_DEPENDENCY,
            }:
                current[unit.artifact_id] = prior
                receipts.append(prior)
                continue
            dependencies = tuple(current[dependency_id] for dependency_id in unit.dependency_artifact_ids)
            failed_dependencies = tuple(
                receipt
                for receipt in dependencies
                if receipt.outcome is not ArtifactExecutionReceiptOutcomeV2.SUCCEEDED
            )
            if failed_dependencies:
                receipt = _dependency_block_receipt(
                    plan,
                    group,
                    unit,
                    failed_dependencies,
                    prior=prior,
                    audit=audit,
                )
            else:
                request = _execution_request(
                    plan,
                    group,
                    unit,
                    dependencies,
                    prior=prior,
                )
                result = await facade.execute(request)
                validate_attachment_execution_result_identity(result)
                if result.execution_request_ref != (attachment_execution_request_ref(request)):
                    raise ArtifactExecutionPolicyError("facade result does not bind its execution request")
                receipt = _attempt_receipt(
                    plan,
                    group,
                    unit,
                    request,
                    result,
                    prior=prior,
                    audit=audit,
                )
            current[unit.artifact_id] = receipt
            receipts.append(receipt)
        return tuple(receipts)

    def validate_current(
        self,
        plan: ArtifactExecutionPlanV2,
        batch: ArtifactExecutionBatchV2,
    ) -> None:
        _validate_execution_plan_identity(plan)
        _validate_execution_batch_identity(batch)
        if batch.artifact_execution_plan_ref != artifact_execution_plan_ref(plan):
            raise ArtifactExecutionPolicyError("execution batch plan is stale or mismatched")
        if {item.artifact_id for item in batch.receipts} != set(plan.routed_artifact_ids):
            raise ArtifactExecutionPolicyError("execution batch does not cover routed artifacts")


def _execution_request(
    plan: ArtifactExecutionPlanV2,
    group: ArtifactExecutionGroupV2,
    unit: ArtifactExecutionUnitV2,
    dependencies: tuple[ArtifactExecutionReceiptV2, ...],
    *,
    prior: ArtifactExecutionReceiptV2 | None,
) -> AttachmentExecutionRequestV2:
    build_spec = unit.build_spec
    frozen = build_spec.build_spec
    contract = unit.build_contract
    provider_route = build_spec.selected_route_kind is ArtifactRouteKindV2.PROVIDER
    selected_id = frozen.provider_preference[0] if provider_route else frozen.runtime_preference[0]
    attempt = (
        prior.attempt + 1
        if prior is not None and prior.outcome is ArtifactExecutionReceiptOutcomeV2.RETRYABLE_FAILURE
        else 1
    )
    retry_result_ref = (
        prior.facade_result
        if prior is not None and prior.outcome is ArtifactExecutionReceiptOutcomeV2.RETRYABLE_FAILURE
        else None
    )
    dependency_result_refs = tuple(
        sorted(
            (
                attachment_execution_result_ref(receipt.facade_result)
                for receipt in dependencies
                if receipt.facade_result is not None
            ),
            key=_facade_ref_key,
        )
    )
    request = AttachmentExecutionRequestV2(
        execution_request_id="attachment-execution-request://pending",
        execution_plan_ref=_facade_ref(artifact_execution_plan_ref(plan)),
        execution_group_id=group.artifact_execution_group_id,
        artifact_id=unit.artifact_id,
        build_spec_ref=_facade_ref(artifact_build_spec_v2_ref(build_spec)),
        producer_task_view_ref=_facade_ref(frozen.producer_task_view_ref),
        content_contract_ref=_facade_ref(frozen.content_contract_ref),
        render_contract_ref=_facade_ref(frozen.render_contract_ref),
        provider_payload_ref=(
            _facade_ref(contract.provider_payload_ref) if contract.provider_payload_ref is not None else None
        ),
        model_profile_ref=(
            _facade_ref(build_spec.selected_model_profile_ref)
            if build_spec.selected_model_profile_ref is not None
            else None
        ),
        selected_route_kind=(
            AttachmentExecutionRouteKindV2.PROVIDER
            if provider_route
            else AttachmentExecutionRouteKindV2.RUNTIME
        ),
        selected_route_id=selected_id,
        selected_route_version=unit.facade_route_decision.selected_version,
        relative_path=frozen.relative_path,
        media_type=frozen.media_type,
        mode=FacadeReconstructionMode(frozen.mode.value),
        runtime_role=contract.runtime_role,
        required_runtime_tools=contract.required_runtime_tools,
        evidence_grants=tuple(_evidence_grant(item) for item in frozen.authorized_evidence_refs),
        world_ledger_snapshot_ref=world_ledger_snapshot_ref(plan.world_ledger_snapshot),
        locked_fact_ids=unit.locked_fact_ids,
        dependency_result_refs=dependency_result_refs,
        attempt=attempt,
        retry_of_result_ref=(
            attachment_execution_result_ref(retry_result_ref) if retry_result_ref is not None else None
        ),
        policy_version=ARTIFACT_EXECUTION_POLICY_VERSION,
        idempotency_key="attachment-execution-idempotency://pending",
        execution_request_sha256="0" * 64,
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


def _evidence_grant(evidence: EvidenceRef) -> AttachmentExecutionEvidenceGrantV2:
    return AttachmentExecutionEvidenceGrantV2(
        evidence_ref_id=evidence.evidence_ref_id,
        subject_ref=_facade_ref(evidence.subject_ref),
        source_spans=tuple(
            AttachmentExecutionSourceSpanV2(
                span_id=span.span_id,
                source_trace_id=span.source_trace_id,
                raw_sha256=span.raw_sha256,
                approximate=span.approximate,
            )
            for span in sorted(
                evidence.source_spans,
                key=lambda item: (
                    item.source_trace_id,
                    item.span_id,
                    item.raw_sha256,
                    item.approximate,
                ),
            )
        ),
        polarity=evidence.polarity.value,
        capability=evidence.capability,
        capability_complete=evidence.capability_complete,
    )


def _attempt_receipt(
    plan: ArtifactExecutionPlanV2,
    group: ArtifactExecutionGroupV2,
    unit: ArtifactExecutionUnitV2,
    request: AttachmentExecutionRequestV2,
    result: AttachmentExecutionResultV2,
    *,
    prior: ArtifactExecutionReceiptV2 | None,
    audit: ContractAudit,
) -> ArtifactExecutionReceiptV2:
    receipt = ArtifactExecutionReceiptV2(
        artifact_execution_receipt_id="artifact-execution-receipt://pending",
        artifact_execution_plan_ref=artifact_execution_plan_ref(plan),
        artifact_execution_group_ref=artifact_execution_group_ref(group),
        artifact_execution_unit_ref=artifact_execution_unit_ref(unit),
        artifact_id=unit.artifact_id,
        attempt=request.attempt,
        outcome=ArtifactExecutionReceiptOutcomeV2(result.status.value),
        facade_request=request,
        facade_result=result,
        failed_dependency_receipt_refs=(),
        retry_of_receipt_ref=(artifact_execution_receipt_ref(prior) if prior is not None else None),
        policy_version=ARTIFACT_EXECUTION_POLICY_VERSION,
        artifact_execution_receipt_sha256="0" * 64,
        audit=_safe_audit(
            audit,
            (
                artifact_execution_plan_ref(plan),
                artifact_execution_group_ref(group),
                artifact_execution_unit_ref(unit),
                _object_ref_from_facade(attachment_execution_result_ref(result)),
            ),
        ),
    )
    return _finalize_receipt(receipt)


def _dependency_block_receipt(
    plan: ArtifactExecutionPlanV2,
    group: ArtifactExecutionGroupV2,
    unit: ArtifactExecutionUnitV2,
    failed_dependencies: tuple[ArtifactExecutionReceiptV2, ...],
    *,
    prior: ArtifactExecutionReceiptV2 | None,
    audit: ContractAudit,
) -> ArtifactExecutionReceiptV2:
    failed_refs = tuple(
        sorted(
            (artifact_execution_receipt_ref(receipt) for receipt in failed_dependencies),
            key=_object_ref_key,
        )
    )
    receipt = ArtifactExecutionReceiptV2(
        artifact_execution_receipt_id="artifact-execution-receipt://pending",
        artifact_execution_plan_ref=artifact_execution_plan_ref(plan),
        artifact_execution_group_ref=artifact_execution_group_ref(group),
        artifact_execution_unit_ref=artifact_execution_unit_ref(unit),
        artifact_id=unit.artifact_id,
        attempt=prior.attempt if prior is not None else 1,
        outcome=ArtifactExecutionReceiptOutcomeV2.BLOCKED_DEPENDENCY,
        facade_request=None,
        facade_result=None,
        failed_dependency_receipt_refs=failed_refs,
        retry_of_receipt_ref=(artifact_execution_receipt_ref(prior) if prior is not None else None),
        policy_version=ARTIFACT_EXECUTION_POLICY_VERSION,
        artifact_execution_receipt_sha256="0" * 64,
        audit=_safe_audit(
            audit,
            (
                artifact_execution_plan_ref(plan),
                artifact_execution_group_ref(group),
                artifact_execution_unit_ref(unit),
                *failed_refs,
            ),
        ),
    )
    return _finalize_receipt(receipt)


def _finalize_receipt(
    receipt: ArtifactExecutionReceiptV2,
) -> ArtifactExecutionReceiptV2:
    digest = artifact_execution_receipt_carried_sha256(receipt)
    return receipt.model_copy(
        update={
            "artifact_execution_receipt_id": (f"artifact-execution-receipt://sha256/{digest}"),
            "artifact_execution_receipt_sha256": digest,
        }
    )


def _validate_routing_source(
    request: ArtifactRoutingRequest,
    result: ArtifactRoutingCompilationResult,
) -> None:
    if artifact_routing_request_carried_sha256(request) != request.request_sha256:
        raise ArtifactExecutionPolicyError("routing request identity is stale")
    if artifact_routing_compilation_result_carried_sha256(result) != result.result_sha256:
        raise ArtifactExecutionPolicyError("routing compilation result identity is stale")
    if result.request_ref != artifact_routing_request_ref(request):
        raise ArtifactExecutionPolicyError("routing compilation result belongs to another request")
    if result.routing_plan is not None:
        if (
            artifact_routing_plan_carried_sha256(result.routing_plan)
            != result.routing_plan.artifact_routing_plan_sha256
        ):
            raise ArtifactExecutionPolicyError("routing plan identity is stale")
        for entry in result.routing_plan.entries:
            if (
                entry.build_spec is not None
                and artifact_build_spec_v2_carried_sha256(entry.build_spec)
                != entry.build_spec.artifact_build_spec_v2_sha256
            ):
                raise ArtifactExecutionPolicyError("routed build spec identity is stale")
            if (
                entry.facade_route_decision is not None
                and attachment_route_decision_carried_sha256(entry.facade_route_decision)
                != entry.facade_route_decision.route_decision_sha256
            ):
                raise ArtifactExecutionPolicyError("route decision identity is stale")


def _validate_build_facts(
    contract: ArtifactBuildContractV2,
    build_spec: ArtifactBuildSpecV2,
) -> None:
    if artifact_build_contract_carried_sha256(contract) != contract.artifact_build_contract_sha256:
        raise ArtifactExecutionPolicyError("build contract identity is stale")
    if artifact_build_spec_v2_carried_sha256(build_spec) != build_spec.artifact_build_spec_v2_sha256:
        raise ArtifactExecutionPolicyError("build spec identity is stale")


def _validate_path_ownership(paths: dict[str, str]) -> None:
    items = sorted(paths.items(), key=lambda item: (item[1], item[0]))
    for index, (artifact_id, path) in enumerate(items):
        current = PurePosixPath(path)
        for other_id, other_path in items[index + 1 :]:
            other = PurePosixPath(other_path)
            if current == other or current in other.parents or other in current.parents:
                raise ArtifactExecutionPolicyError(
                    f"artifact output paths overlap: {artifact_id} and {other_id}"
                )


def _validate_dependencies(
    definitions: dict[str, ArtifactConsistencyDefinition],
) -> None:
    known = set(definitions)
    for definition in definitions.values():
        unknown = set(definition.dependency_artifact_ids) - known
        if unknown:
            raise ArtifactExecutionPolicyError("artifact dependency references an unknown routed artifact")
    _topological_order(
        tuple(sorted(known)),
        definitions=definitions,
        paths={artifact_id: artifact_id for artifact_id in known},
    )


def _connected_components(
    definitions: dict[str, ArtifactConsistencyDefinition],
) -> tuple[tuple[str, ...], ...]:
    parent = {artifact_id: artifact_id for artifact_id in definitions}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        smaller, larger = sorted((left_root, right_root))
        parent[larger] = smaller

    fact_owner: dict[str, str] = {}
    for artifact_id in sorted(definitions):
        definition = definitions[artifact_id]
        for dependency_id in definition.dependency_artifact_ids:
            union(artifact_id, dependency_id)
        for fact_id in definition.locked_fact_ids:
            owner = fact_owner.setdefault(fact_id, artifact_id)
            union(artifact_id, owner)
    grouped: dict[str, list[str]] = {}
    for artifact_id in sorted(definitions):
        grouped.setdefault(find(artifact_id), []).append(artifact_id)
    return tuple(
        tuple(values)
        for _, values in sorted(
            grouped.items(),
            key=lambda item: tuple(item[1]),
        )
    )


def _topological_order(
    component: Iterable[str],
    *,
    definitions: dict[str, ArtifactConsistencyDefinition],
    paths: dict[str, str],
) -> tuple[str, ...]:
    members = set(component)
    indegree = {artifact_id: 0 for artifact_id in members}
    children: dict[str, set[str]] = {artifact_id: set() for artifact_id in members}
    for artifact_id in members:
        for dependency_id in definitions[artifact_id].dependency_artifact_ids:
            if dependency_id not in members:
                continue
            indegree[artifact_id] += 1
            children[dependency_id].add(artifact_id)
    ready = sorted(
        (artifact_id for artifact_id, degree in indegree.items() if degree == 0),
        key=lambda artifact_id: (paths[artifact_id], artifact_id),
    )
    ordered: list[str] = []
    while ready:
        artifact_id = ready.pop(0)
        ordered.append(artifact_id)
        for child in sorted(
            children[artifact_id],
            key=lambda item: (paths[item], item),
        ):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
                ready.sort(key=lambda item: (paths[item], item))
    if len(ordered) != len(members):
        raise ArtifactExecutionPolicyError("artifact dependency graph contains a cycle")
    return tuple(ordered)


def _validate_execution_plan_identity(plan: ArtifactExecutionPlanV2) -> None:
    validate_world_ledger_snapshot_identity(plan.world_ledger_snapshot)
    _validate_audit_inputs(
        plan.audit,
        (
            plan.artifact_routing_plan_ref,
            _object_ref_from_facade(world_ledger_snapshot_ref(plan.world_ledger_snapshot)),
        ),
    )
    for group in plan.groups:
        for unit in group.units:
            _validate_build_facts(unit.build_contract, unit.build_spec)
            if (
                attachment_route_decision_carried_sha256(unit.facade_route_decision)
                != unit.facade_route_decision.route_decision_sha256
            ):
                raise ArtifactExecutionPolicyError("artifact execution route decision identity is stale")
            if artifact_execution_unit_carried_sha256(unit) != unit.artifact_execution_unit_sha256:
                raise ArtifactExecutionPolicyError("artifact execution unit identity is stale")
        if artifact_execution_group_carried_sha256(group) != group.artifact_execution_group_sha256:
            raise ArtifactExecutionPolicyError("artifact execution group identity is stale")
    if artifact_execution_plan_carried_sha256(plan) != plan.artifact_execution_plan_sha256:
        raise ArtifactExecutionPolicyError("artifact execution plan identity is stale")


def _validate_execution_batch_identity(batch: ArtifactExecutionBatchV2) -> None:
    for receipt in batch.receipts:
        expected_receipt_refs = (
            receipt.artifact_execution_plan_ref,
            receipt.artifact_execution_group_ref,
            receipt.artifact_execution_unit_ref,
            *receipt.failed_dependency_receipt_refs,
        )
        if receipt.facade_result is not None:
            expected_receipt_refs += (
                _object_ref_from_facade(attachment_execution_result_ref(receipt.facade_result)),
            )
        _validate_audit_inputs(receipt.audit, expected_receipt_refs)
        if artifact_execution_receipt_carried_sha256(receipt) != receipt.artifact_execution_receipt_sha256:
            raise ArtifactExecutionPolicyError("artifact execution receipt identity is stale")
    _validate_audit_inputs(
        batch.audit,
        (
            batch.artifact_execution_plan_ref,
            *tuple(artifact_execution_receipt_ref(receipt) for receipt in batch.receipts),
        ),
    )
    if artifact_execution_batch_carried_sha256(batch) != batch.artifact_execution_batch_sha256:
        raise ArtifactExecutionPolicyError("artifact execution batch identity is stale")


def _artifact_ids(
    receipts: tuple[ArtifactExecutionReceiptV2, ...],
    outcomes: set[ArtifactExecutionReceiptOutcomeV2],
) -> tuple[str, ...]:
    return tuple(sorted(receipt.artifact_id for receipt in receipts if receipt.outcome in outcomes))


def _safe_audit(
    audit: ContractAudit,
    refs: tuple[ObjectRef, ...],
) -> ContractAudit:
    unique = {_object_ref_key(ref): ref for ref in refs}
    return audit.model_copy(update={"input_refs": tuple(unique[key] for key in sorted(unique))})


def _validate_audit_inputs(
    audit: ContractAudit,
    refs: tuple[ObjectRef, ...],
) -> None:
    unique = {_object_ref_key(ref): ref for ref in refs}
    expected = tuple(unique[key] for key in sorted(unique))
    if audit.input_refs != expected:
        raise ArtifactExecutionPolicyError("artifact execution audit lineage is stale or mismatched")
    if not any(
        binding.component == "artifact-execution" and binding.version == "r5-06"
        for binding in audit.governing_versions
    ):
        raise ArtifactExecutionPolicyError("artifact execution audit governing version is missing")


def _facade_ref(ref: ObjectRef) -> FacadeObjectRef:
    return FacadeObjectRef(
        object_type=ref.object_type,
        object_id=ref.object_id,
        object_version=ref.object_version,
        object_sha256=ref.object_sha256,
    )


def _object_ref_from_facade(ref: FacadeObjectRef) -> ObjectRef:
    return ObjectRef(
        object_type=ref.object_type,
        object_id=ref.object_id,
        object_version=ref.object_version,
        object_sha256=ref.object_sha256,
    )


def _object_ref_key(ref: ObjectRef) -> tuple[str, str, str, str]:
    return (
        ref.object_type,
        ref.object_id,
        ref.object_version,
        ref.object_sha256,
    )


def _facade_ref_key(
    ref: FacadeObjectRef,
) -> tuple[str, str, str, str]:
    return (
        ref.object_type,
        ref.object_id,
        ref.object_version,
        ref.object_sha256,
    )
