from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

from env_mock_agent.facade import (
    AttachmentRouteCandidateKind,
    AttachmentRouteDecisionV2,
    AttachmentRouteOutcome,
    AttachmentRouteRequestV2,
    AttachmentRoutingFacade,
    FacadeObjectRef,
    attachment_route_decision_carried_sha256,
    attachment_route_decision_ref,
    attachment_route_request_carried_sha256,
    attachment_route_request_ref,
)
from env_mock_agent.facade import (
    ReconstructionMode as FacadeReconstructionMode,
)
from eval_factory.attachment_planning.bridge import (
    AttachmentPlanningBridge,
)
from eval_factory.attachment_planning.models import AttachmentPlanningPolicyError
from eval_factory.attachment_planning.routing_models import (
    ARTIFACT_ROUTING_POLICY_VERSION,
    ArtifactBuildContractDefinition,
    ArtifactRoutingCompilationResult,
    ArtifactRoutingExecution,
    ArtifactRoutingExecutionOutcome,
    ArtifactRoutingExecutionResult,
    ArtifactRoutingPolicyError,
    ArtifactRoutingRequest,
    artifact_routing_compilation_result_carried_sha256,
    artifact_routing_execution_result_carried_sha256,
    artifact_routing_execution_result_ref,
    artifact_routing_request_carried_sha256,
    artifact_routing_request_ref,
)
from eval_factory.contracts.attachment import (
    ArtifactBuildSpec,
    ReconstructionMode,
)
from eval_factory.contracts.attachment_v2 import (
    ArtifactBuildContractV2,
    ArtifactBuildSpecV2,
    ArtifactEvidenceMatrixV2,
    ArtifactEvidenceRowV2,
    ArtifactEvidenceTargetV2,
    ArtifactRouteEntryOutcomeV2,
    ArtifactRouteKindV2,
    ArtifactRoutePlanEntryV2,
    ArtifactRoutingAggregateOutcomeV2,
    ArtifactRoutingPlanV2,
    ArtifactRoutingPolicyV2,
    ArtifactRoutingReasonV2,
    AttachmentPlanningContextV2,
    artifact_build_contract_carried_sha256,
    artifact_build_contract_ref,
    artifact_build_spec_v2_carried_sha256,
    artifact_evidence_matrix_carried_sha256,
    artifact_evidence_matrix_ref,
    artifact_evidence_row_carried_sha256,
    artifact_evidence_row_ref,
    artifact_evidence_target_ref,
    artifact_routing_plan_carried_sha256,
    artifact_routing_policy_carried_sha256,
    artifact_routing_policy_ref,
    attachment_planning_context_ref,
    validate_artifact_evidence_target_identity,
)
from eval_factory.contracts.core import (
    ContractAudit,
    EvidenceRef,
    ObjectRef,
)
from eval_factory.contracts.core_v2 import canonical_value_v2
from eval_factory.contracts.safety import EvidenceBundle
from eval_factory.contracts.task import AttachmentCriticality
from eval_factory.contracts.task_v2 import (
    ProducerAttachmentRequirementV2,
    ProducerStorageAuthorizationV2,
    ProducerTaskViewV2,
)
from eval_factory.provenance.bundles import (
    EvidenceCompilationPolicyError,
    projection_policy_ref,
    validate_evidence_bundle_identity,
    validate_projection_policy,
)
from eval_factory.provenance.views import (
    EvidenceViewPolicyError,
    EvidenceViewPrincipalType,
    EvidenceViewPurpose,
    EvidenceViewResult,
    validate_evidence_view_result_identity,
)


class ArtifactRoutingRequestBuilder:
    policy_version = ARTIFACT_ROUTING_POLICY_VERSION

    def build(
        self,
        *,
        attachment_planning_context: AttachmentPlanningContextV2,
        producer_task_view: ProducerTaskViewV2,
        storage_authorization: ProducerStorageAuthorizationV2,
        producer_view_result: EvidenceViewResult,
        producer_evidence_bundle: EvidenceBundle,
        artifact_evidence_matrix: ArtifactEvidenceMatrixV2 | None,
        artifact_targets: tuple[ArtifactEvidenceTargetV2, ...],
        routing_policy: ArtifactRoutingPolicyV2,
        contract_definitions: tuple[ArtifactBuildContractDefinition, ...],
        audit: ContractAudit,
    ) -> ArtifactRoutingRequest:
        self._validate_basis(
            attachment_planning_context=attachment_planning_context,
            producer_task_view=producer_task_view,
            storage_authorization=storage_authorization,
            producer_view_result=producer_view_result,
            producer_evidence_bundle=producer_evidence_bundle,
            routing_policy=routing_policy,
        )
        context_ref = attachment_planning_context_ref(attachment_planning_context)
        policy_ref = artifact_routing_policy_ref(routing_policy)
        requirements = {item.dependency_id: item for item in producer_task_view.attachment_requirements}
        if not requirements:
            if artifact_evidence_matrix is not None or artifact_targets or contract_definitions:
                raise ArtifactRoutingPolicyError("no attachment requirements cannot contain routing inputs")
            return _routing_request(
                context_ref=context_ref,
                producer_task_view_ref=(attachment_planning_context.producer_task_view_ref),
                matrix_ref=None,
                policy_ref=policy_ref,
                build_contracts=(),
                facade_requests=(),
                missing_contract_target_refs=(),
                blocked_mode_target_refs=(),
                audit=audit,
            )
        if artifact_evidence_matrix is None:
            raise ArtifactRoutingPolicyError("attachment requirements require an artifact evidence matrix")

        targets_by_ref = self._validate_matrix_and_targets(
            attachment_planning_context=attachment_planning_context,
            producer_task_view=producer_task_view,
            artifact_evidence_matrix=artifact_evidence_matrix,
            artifact_targets=artifact_targets,
        )
        definitions = self._validate_definitions(
            requirements=requirements,
            matrix=artifact_evidence_matrix,
            definitions=contract_definitions,
        )
        matrix_ref = artifact_evidence_matrix_ref(artifact_evidence_matrix)
        build_contracts: list[ArtifactBuildContractV2] = []
        missing_contract_refs: list[ObjectRef] = []
        blocked_mode_refs: list[ObjectRef] = []
        for row in artifact_evidence_matrix.rows:
            target = targets_by_ref[_ref_key(row.artifact_evidence_target_ref)]
            if row.row.selected_mode is ReconstructionMode.BLOCKED:
                blocked_mode_refs.append(row.artifact_evidence_target_ref)
                continue
            definition = definitions.get(row.attachment_dependency_id)
            if definition is None:
                missing_contract_refs.append(row.artifact_evidence_target_ref)
                continue
            build_contracts.append(
                _build_contract(
                    attachment_planning_context_ref=context_ref,
                    producer_task_view_ref=(attachment_planning_context.producer_task_view_ref),
                    artifact_evidence_matrix_ref=matrix_ref,
                    row=row,
                    target=target,
                    definition=definition,
                    audit=audit,
                )
            )
        build_contracts.sort(key=lambda item: (item.logical_path, item.artifact_id))
        facade_requests = tuple(_facade_request(contract, routing_policy) for contract in build_contracts)
        return _routing_request(
            context_ref=context_ref,
            producer_task_view_ref=(attachment_planning_context.producer_task_view_ref),
            matrix_ref=matrix_ref,
            policy_ref=policy_ref,
            build_contracts=tuple(build_contracts),
            facade_requests=facade_requests,
            missing_contract_target_refs=tuple(sorted(missing_contract_refs, key=_ref_key)),
            blocked_mode_target_refs=tuple(sorted(blocked_mode_refs, key=_ref_key)),
            audit=audit,
        )

    def _validate_basis(
        self,
        *,
        attachment_planning_context: AttachmentPlanningContextV2,
        producer_task_view: ProducerTaskViewV2,
        storage_authorization: ProducerStorageAuthorizationV2,
        producer_view_result: EvidenceViewResult,
        producer_evidence_bundle: EvidenceBundle,
        routing_policy: ArtifactRoutingPolicyV2,
    ) -> None:
        try:
            AttachmentPlanningBridge().validate_current(
                producer_task_view=producer_task_view,
                storage_authorization=storage_authorization,
                evidence_bundle=producer_evidence_bundle,
                attachment_planning_context=attachment_planning_context,
            )
            validate_evidence_view_result_identity(producer_view_result)
            validate_projection_policy(producer_view_result.projection_policy)
            validate_evidence_bundle_identity(producer_evidence_bundle)
            _validate_routing_policy_identity(routing_policy)
        except (
            AttachmentPlanningPolicyError,
            EvidenceViewPolicyError,
            EvidenceCompilationPolicyError,
            ValueError,
        ) as exc:
            raise ArtifactRoutingPolicyError("artifact routing source identity is stale or invalid") from exc
        if (
            producer_view_result.principal_id != attachment_planning_context.producer_principal_id
            or producer_view_result.principal_type is not EvidenceViewPrincipalType.ATTACHMENT_PRODUCER
            or producer_view_result.purpose is not EvidenceViewPurpose.ATTACHMENT_PRODUCTION
        ):
            raise ArtifactRoutingPolicyError("routing evidence view must use the exact attachment producer")
        expected_policy_ref = projection_policy_ref(producer_view_result.projection_policy)
        if (
            expected_policy_ref != attachment_planning_context.projection_policy_ref
            or expected_policy_ref != producer_evidence_bundle.projection_policy_ref
        ):
            raise ArtifactRoutingPolicyError("routing evidence projection policy is stale or mismatched")

    def _validate_matrix_and_targets(
        self,
        *,
        attachment_planning_context: AttachmentPlanningContextV2,
        producer_task_view: ProducerTaskViewV2,
        artifact_evidence_matrix: ArtifactEvidenceMatrixV2,
        artifact_targets: tuple[ArtifactEvidenceTargetV2, ...],
    ) -> dict[tuple[str, str, str, str], ArtifactEvidenceTargetV2]:
        _validate_matrix_identity(artifact_evidence_matrix)
        context_ref = attachment_planning_context_ref(attachment_planning_context)
        if (
            artifact_evidence_matrix.attachment_planning_context_ref != context_ref
            or artifact_evidence_matrix.producer_task_view_ref
            != attachment_planning_context.producer_task_view_ref
            or artifact_evidence_matrix.safe_evidence_bundle_ref
            != attachment_planning_context.safe_evidence_bundle_ref
        ):
            raise ArtifactRoutingPolicyError(
                "artifact evidence matrix does not bind the current R5 source graph"
            )
        requirements = {item.dependency_id: item for item in producer_task_view.attachment_requirements}
        targets_by_ref: dict[
            tuple[str, str, str, str],
            ArtifactEvidenceTargetV2,
        ] = {}
        for target in artifact_targets:
            try:
                validate_artifact_evidence_target_identity(target)
            except ValueError as exc:
                raise ArtifactRoutingPolicyError(
                    "artifact routing target identity is stale or invalid"
                ) from exc
            if target.attachment_planning_context_ref != context_ref:
                raise ArtifactRoutingPolicyError("artifact routing target context is mismatched")
            requirement = requirements.get(target.attachment_dependency_id)
            if requirement is None or requirement.criticality is not target.criticality:
                raise ArtifactRoutingPolicyError("artifact routing target does not match ProducerTaskView")
            key = _ref_key(artifact_evidence_target_ref(target))
            if key in targets_by_ref:
                raise ArtifactRoutingPolicyError("duplicate artifact routing target")
            targets_by_ref[key] = target
        row_target_keys = {
            _ref_key(row.artifact_evidence_target_ref) for row in artifact_evidence_matrix.rows
        }
        if row_target_keys != set(targets_by_ref):
            raise ArtifactRoutingPolicyError("artifact targets must exactly cover matrix rows")
        for row in artifact_evidence_matrix.rows:
            target = targets_by_ref[_ref_key(row.artifact_evidence_target_ref)]
            if (
                row.attachment_dependency_id != target.attachment_dependency_id
                or row.row.artifact_id != target.artifact_id
                or row.row.logical_path != target.logical_path
                or row.row.media_type != target.media_type
                or row.row.criticality is not target.criticality
            ):
                raise ArtifactRoutingPolicyError("artifact matrix row does not match its exact target")
        return targets_by_ref

    def _validate_definitions(
        self,
        *,
        requirements: Mapping[str, ProducerAttachmentRequirementV2],
        matrix: ArtifactEvidenceMatrixV2,
        definitions: tuple[ArtifactBuildContractDefinition, ...],
    ) -> dict[str, ArtifactBuildContractDefinition]:
        by_dependency: dict[str, ArtifactBuildContractDefinition] = {}
        row_by_dependency = {row.attachment_dependency_id: row for row in matrix.rows}
        definition_ids: set[str] = set()
        for definition in definitions:
            if definition.definition_id in definition_ids:
                raise ArtifactRoutingPolicyError("duplicate artifact build contract definition ID")
            definition_ids.add(definition.definition_id)
            if definition.attachment_dependency_id not in requirements:
                raise ArtifactRoutingPolicyError(
                    "artifact build contract definition targets an unknown dependency"
                )
            if definition.attachment_dependency_id in by_dependency:
                raise ArtifactRoutingPolicyError("duplicate artifact build contract definition")
            row = row_by_dependency[definition.attachment_dependency_id]
            if row.row.selected_mode is ReconstructionMode.BLOCKED:
                raise ArtifactRoutingPolicyError(
                    "BLOCKED evidence rows cannot receive build contract definitions"
                )
            by_dependency[definition.attachment_dependency_id] = definition
        return by_dependency


class ArtifactRoutingRunner:
    async def run(
        self,
        request: ArtifactRoutingRequest,
        *,
        facade: AttachmentRoutingFacade,
        audit: ContractAudit,
    ) -> ArtifactRoutingExecutionResult:
        _validate_request_identity(request)
        executions: list[ArtifactRoutingExecution] = []
        for facade_request in request.facade_requests:
            _validate_facade_request_identity(facade_request)
            decision = await facade.route(facade_request)
            _validate_facade_decision_identity(decision)
            executions.append(
                ArtifactRoutingExecution(
                    facade_request=facade_request,
                    facade_decision=decision,
                )
            )
        outcome = (
            ArtifactRoutingExecutionOutcome.NOT_REQUIRED
            if request.artifact_evidence_matrix_ref is None
            else ArtifactRoutingExecutionOutcome.COMPLETED
        )
        result = ArtifactRoutingExecutionResult(
            result_id="artifact-routing-execution-result://pending",
            request_ref=artifact_routing_request_ref(request),
            outcome=outcome,
            executions=tuple(executions),
            policy_version=ARTIFACT_ROUTING_POLICY_VERSION,
            result_sha256="0" * 64,
            audit=_safe_audit(
                audit,
                (
                    artifact_routing_request_ref(request),
                    *tuple(
                        _factory_ref(attachment_route_decision_ref(execution.facade_decision))
                        for execution in executions
                    ),
                ),
            ),
        )
        digest = artifact_routing_execution_result_carried_sha256(result)
        return result.model_copy(
            update={
                "result_id": (f"artifact-routing-execution-result://sha256/{digest}"),
                "result_sha256": digest,
            }
        )


class ArtifactBuildSpecCompiler:
    policy_version = ARTIFACT_ROUTING_POLICY_VERSION

    def compile(
        self,
        *,
        request: ArtifactRoutingRequest,
        execution_result: ArtifactRoutingExecutionResult,
        attachment_planning_context: AttachmentPlanningContextV2,
        producer_task_view: ProducerTaskViewV2,
        storage_authorization: ProducerStorageAuthorizationV2,
        producer_view_result: EvidenceViewResult,
        producer_evidence_bundle: EvidenceBundle,
        artifact_evidence_matrix: ArtifactEvidenceMatrixV2 | None,
        artifact_targets: tuple[ArtifactEvidenceTargetV2, ...],
        routing_policy: ArtifactRoutingPolicyV2,
        audit: ContractAudit,
    ) -> ArtifactRoutingCompilationResult:
        _validate_request_identity(request)
        _validate_execution_result_identity(execution_result)
        ArtifactRoutingRequestBuilder()._validate_basis(
            attachment_planning_context=attachment_planning_context,
            producer_task_view=producer_task_view,
            storage_authorization=storage_authorization,
            producer_view_result=producer_view_result,
            producer_evidence_bundle=producer_evidence_bundle,
            routing_policy=routing_policy,
        )
        if (
            request.attachment_planning_context_ref
            != attachment_planning_context_ref(attachment_planning_context)
            or request.producer_task_view_ref != attachment_planning_context.producer_task_view_ref
            or request.artifact_routing_policy_ref != artifact_routing_policy_ref(routing_policy)
            or execution_result.request_ref != artifact_routing_request_ref(request)
        ):
            raise ArtifactRoutingPolicyError("routing request or execution source graph is mismatched")
        execution_by_request = self._validate_executions(
            request,
            execution_result,
        )
        if request.artifact_evidence_matrix_ref is None:
            if (
                artifact_evidence_matrix is not None
                or artifact_targets
                or producer_task_view.attachment_requirements
                or execution_result.outcome is not ArtifactRoutingExecutionOutcome.NOT_REQUIRED
            ):
                raise ArtifactRoutingPolicyError("NOT_REQUIRED routing result has unexpected source work")
            return _compilation_result(
                request=request,
                execution_result=execution_result,
                outcome=ArtifactRoutingAggregateOutcomeV2.NOT_REQUIRED,
                routing_plan=None,
                audit=audit,
            )
        if artifact_evidence_matrix is None:
            raise ArtifactRoutingPolicyError("routing request matrix is missing from authoritative inputs")
        targets_by_ref = ArtifactRoutingRequestBuilder()._validate_matrix_and_targets(
            attachment_planning_context=attachment_planning_context,
            producer_task_view=producer_task_view,
            artifact_evidence_matrix=artifact_evidence_matrix,
            artifact_targets=artifact_targets,
        )
        if request.artifact_evidence_matrix_ref != artifact_evidence_matrix_ref(artifact_evidence_matrix):
            raise ArtifactRoutingPolicyError("routing request matrix ref is stale or mismatched")
        contract_by_target = {
            _ref_key(contract.artifact_evidence_target_ref): contract for contract in request.build_contracts
        }
        missing_keys = {_ref_key(ref) for ref in request.missing_contract_target_refs}
        blocked_keys = {_ref_key(ref) for ref in request.blocked_mode_target_refs}
        entries: list[ArtifactRoutePlanEntryV2] = []
        for row in artifact_evidence_matrix.rows:
            target_key = _ref_key(row.artifact_evidence_target_ref)
            target = targets_by_ref[target_key]
            if target_key in blocked_keys:
                entries.append(
                    _blocked_entry(
                        target,
                        ArtifactRouteEntryOutcomeV2.BLOCKED_POLICY,
                        ArtifactRoutingReasonV2.EVIDENCE_MODE_BLOCKED,
                    )
                )
                continue
            if target_key in missing_keys:
                entries.append(
                    _blocked_entry(
                        target,
                        ArtifactRouteEntryOutcomeV2.BLOCKED_CAPABILITY,
                        ArtifactRoutingReasonV2.BUILD_CONTRACT_MISSING,
                    )
                )
                continue
            contract = contract_by_target.get(target_key)
            if contract is None:
                raise ArtifactRoutingPolicyError("routing request target partition is incomplete")
            facade_request = _facade_request_for_contract(
                request,
                contract,
            )
            execution = execution_by_request[facade_request.route_request_id]
            decision = execution.facade_decision
            if decision.outcome is AttachmentRouteOutcome.BLOCKED_CAPABILITY:
                entries.append(
                    ArtifactRoutePlanEntryV2(
                        artifact_evidence_target_ref=(row.artifact_evidence_target_ref),
                        attachment_dependency_id=(row.attachment_dependency_id),
                        artifact_id=row.row.artifact_id,
                        criticality=row.row.criticality,
                        outcome=(ArtifactRouteEntryOutcomeV2.BLOCKED_CAPABILITY),
                        facade_route_request_ref=_factory_ref(attachment_route_request_ref(facade_request)),
                        facade_route_decision=decision,
                        build_spec=None,
                        reasons=frozenset({ArtifactRoutingReasonV2.ROUTE_UNAVAILABLE}),
                    )
                )
                continue
            build_spec = _compile_build_spec(
                context_ref=request.attachment_planning_context_ref,
                matrix_ref=request.artifact_evidence_matrix_ref,
                row=row,
                target=target,
                contract=contract,
                routing_policy=routing_policy,
                facade_request=facade_request,
                facade_decision=decision,
                producer_task_view=producer_task_view,
                storage_authorization=storage_authorization,
                producer_evidence_bundle=producer_evidence_bundle,
                audit=audit,
            )
            entry_outcome = (
                ArtifactRouteEntryOutcomeV2.ROUTED_PROVIDER
                if build_spec.selected_route_kind is ArtifactRouteKindV2.PROVIDER
                else ArtifactRouteEntryOutcomeV2.ROUTED_RUNTIME
            )
            entries.append(
                ArtifactRoutePlanEntryV2(
                    artifact_evidence_target_ref=(row.artifact_evidence_target_ref),
                    attachment_dependency_id=row.attachment_dependency_id,
                    artifact_id=row.row.artifact_id,
                    criticality=row.row.criticality,
                    outcome=entry_outcome,
                    facade_route_request_ref=_factory_ref(attachment_route_request_ref(facade_request)),
                    facade_route_decision=decision,
                    build_spec=build_spec,
                    reasons=frozenset(),
                )
            )
        outcome = _aggregate_outcome(tuple(entries))
        routed = tuple(
            sorted(
                item.artifact_id
                for item in entries
                if item.outcome
                in {
                    ArtifactRouteEntryOutcomeV2.ROUTED_PROVIDER,
                    ArtifactRouteEntryOutcomeV2.ROUTED_RUNTIME,
                }
            )
        )
        blocked_required = tuple(
            sorted(
                item.artifact_id
                for item in entries
                if item.outcome
                in {
                    ArtifactRouteEntryOutcomeV2.BLOCKED_CAPABILITY,
                    ArtifactRouteEntryOutcomeV2.BLOCKED_POLICY,
                }
                and item.criticality is not AttachmentCriticality.OPTIONAL
            )
        )
        blocked_optional = tuple(
            sorted(
                item.artifact_id
                for item in entries
                if item.outcome
                in {
                    ArtifactRouteEntryOutcomeV2.BLOCKED_CAPABILITY,
                    ArtifactRouteEntryOutcomeV2.BLOCKED_POLICY,
                }
                and item.criticality is AttachmentCriticality.OPTIONAL
            )
        )
        plan = ArtifactRoutingPlanV2(
            artifact_routing_plan_id="artifact-routing-plan://pending",
            attachment_planning_context_ref=(request.attachment_planning_context_ref),
            producer_task_view_ref=request.producer_task_view_ref,
            artifact_evidence_matrix_ref=(request.artifact_evidence_matrix_ref),
            artifact_routing_policy_ref=(request.artifact_routing_policy_ref),
            entries=tuple(entries),
            aggregate_outcome=outcome,
            routed_artifact_ids=routed,
            blocked_required_artifact_ids=blocked_required,
            blocked_optional_artifact_ids=blocked_optional,
            policy_version=ARTIFACT_ROUTING_POLICY_VERSION,
            artifact_routing_plan_sha256="0" * 64,
            audit=_safe_audit(
                audit,
                (
                    request.attachment_planning_context_ref,
                    request.producer_task_view_ref,
                    request.artifact_evidence_matrix_ref,
                    request.artifact_routing_policy_ref,
                    *tuple(artifact_build_contract_ref(contract) for contract in request.build_contracts),
                ),
            ),
        )
        plan_digest = artifact_routing_plan_carried_sha256(plan)
        plan = plan.model_copy(
            update={
                "artifact_routing_plan_id": (f"artifact-routing-plan://sha256/{plan_digest}"),
                "artifact_routing_plan_sha256": plan_digest,
            }
        )
        return _compilation_result(
            request=request,
            execution_result=execution_result,
            outcome=outcome,
            routing_plan=plan,
            audit=audit,
        )

    def validate_current(
        self,
        *,
        request: ArtifactRoutingRequest,
        execution_result: ArtifactRoutingExecutionResult,
        compilation_result: ArtifactRoutingCompilationResult,
        attachment_planning_context: AttachmentPlanningContextV2,
        producer_task_view: ProducerTaskViewV2,
        storage_authorization: ProducerStorageAuthorizationV2,
        producer_view_result: EvidenceViewResult,
        producer_evidence_bundle: EvidenceBundle,
        artifact_evidence_matrix: ArtifactEvidenceMatrixV2 | None,
        artifact_targets: tuple[ArtifactEvidenceTargetV2, ...],
        routing_policy: ArtifactRoutingPolicyV2,
    ) -> None:
        try:
            _validate_compilation_result_identity(compilation_result)
            rebuilt = self.compile(
                request=request,
                execution_result=execution_result,
                attachment_planning_context=attachment_planning_context,
                producer_task_view=producer_task_view,
                storage_authorization=storage_authorization,
                producer_view_result=producer_view_result,
                producer_evidence_bundle=producer_evidence_bundle,
                artifact_evidence_matrix=artifact_evidence_matrix,
                artifact_targets=artifact_targets,
                routing_policy=routing_policy,
                audit=compilation_result.audit,
            )
            if _without_audit_actor_time(rebuilt) != _without_audit_actor_time(compilation_result):
                raise ArtifactRoutingPolicyError(
                    "artifact routing result does not match authoritative inputs"
                )
        except ArtifactRoutingPolicyError as exc:
            raise ArtifactRoutingPolicyError(f"current artifact routing validation failed: {exc}") from exc

    def _validate_executions(
        self,
        request: ArtifactRoutingRequest,
        execution_result: ArtifactRoutingExecutionResult,
    ) -> dict[str, ArtifactRoutingExecution]:
        expected_ids = tuple(item.route_request_id for item in request.facade_requests)
        observed_ids = tuple(item.facade_request.route_request_id for item in execution_result.executions)
        if observed_ids != expected_ids:
            raise ArtifactRoutingPolicyError("routing executions must exactly cover facade requests")
        for expected, execution in zip(
            request.facade_requests,
            execution_result.executions,
            strict=True,
        ):
            if execution.facade_request != expected:
                raise ArtifactRoutingPolicyError("routing execution request is stale or mismatched")
            _validate_facade_request_identity(execution.facade_request)
            _validate_facade_decision_identity(execution.facade_decision)
            _validate_decision_against_request(
                execution.facade_request,
                execution.facade_decision,
            )
        return {item.facade_request.route_request_id: item for item in execution_result.executions}


def _build_contract(
    *,
    attachment_planning_context_ref: ObjectRef,
    producer_task_view_ref: ObjectRef,
    artifact_evidence_matrix_ref: ObjectRef,
    row: ArtifactEvidenceRowV2,
    target: ArtifactEvidenceTargetV2,
    definition: ArtifactBuildContractDefinition,
    audit: ContractAudit,
) -> ArtifactBuildContractV2:
    contract = ArtifactBuildContractV2(
        artifact_build_contract_id="artifact-build-contract://pending",
        attachment_planning_context_ref=attachment_planning_context_ref,
        producer_task_view_ref=producer_task_view_ref,
        artifact_evidence_matrix_ref=artifact_evidence_matrix_ref,
        artifact_evidence_row_ref=artifact_evidence_row_ref(row),
        artifact_evidence_target_ref=artifact_evidence_target_ref(target),
        attachment_dependency_id=target.attachment_dependency_id,
        artifact_id=target.artifact_id,
        logical_path=target.logical_path,
        media_type=target.media_type,
        asset_type=definition.asset_type,
        mode=row.row.selected_mode,
        criticality=target.criticality,
        content_contract_ref=definition.content_contract_ref,
        render_contract_ref=definition.render_contract_ref,
        provider_payload_ref=definition.provider_payload_ref,
        source_evidence_set_ref=definition.source_evidence_set_ref,
        required_provider_capability_ids=(definition.required_provider_capability_ids),
        required_runtime_tools=definition.required_runtime_tools,
        runtime_role=definition.runtime_role,
        runtime_resume_required=definition.runtime_resume_required,
        validator_ids=definition.validator_ids,
        policy_version=ARTIFACT_ROUTING_POLICY_VERSION,
        artifact_build_contract_sha256="0" * 64,
        audit=_safe_audit(
            audit,
            (
                attachment_planning_context_ref,
                producer_task_view_ref,
                artifact_evidence_matrix_ref,
                artifact_evidence_row_ref(row),
                artifact_evidence_target_ref(target),
                definition.content_contract_ref,
                definition.render_contract_ref,
                *(() if definition.provider_payload_ref is None else (definition.provider_payload_ref,)),
                *(
                    ()
                    if definition.source_evidence_set_ref is None
                    else (definition.source_evidence_set_ref,)
                ),
            ),
        ),
    )
    digest = artifact_build_contract_carried_sha256(contract)
    return contract.model_copy(
        update={
            "artifact_build_contract_id": (f"artifact-build-contract://sha256/{digest}"),
            "artifact_build_contract_sha256": digest,
        }
    )


def _facade_request(
    contract: ArtifactBuildContractV2,
    policy: ArtifactRoutingPolicyV2,
) -> AttachmentRouteRequestV2:
    request = AttachmentRouteRequestV2(
        route_request_id="attachment-route-request://pending",
        artifact_id=contract.artifact_id,
        build_contract_ref=_facade_ref(artifact_build_contract_ref(contract)),
        routing_policy_ref=_facade_ref(artifact_routing_policy_ref(policy)),
        asset_type=contract.asset_type,
        media_type=contract.media_type,
        mode=FacadeReconstructionMode(contract.mode.value),
        criticality=contract.criticality.value,
        provider_payload_ref=(
            None if contract.provider_payload_ref is None else _facade_ref(contract.provider_payload_ref)
        ),
        approved_provider_ids=policy.approved_provider_ids,
        required_provider_capability_ids=(contract.required_provider_capability_ids),
        runtime_order=policy.runtime_order,
        required_runtime_tools=contract.required_runtime_tools,
        require_runtime_resume=contract.runtime_resume_required,
        idempotency_key=(
            "attachment-route-idempotency://sha256/"
            + hashlib.sha256(
                (f"{contract.artifact_id}:{contract.artifact_build_contract_sha256}").encode()
            ).hexdigest()
        ),
        route_request_sha256="0" * 64,
    )
    digest = attachment_route_request_carried_sha256(request)
    return request.model_copy(
        update={
            "route_request_id": (f"attachment-route-request://sha256/{digest}"),
            "route_request_sha256": digest,
        }
    )


def _routing_request(
    *,
    context_ref: ObjectRef,
    producer_task_view_ref: ObjectRef,
    matrix_ref: ObjectRef | None,
    policy_ref: ObjectRef,
    build_contracts: tuple[ArtifactBuildContractV2, ...],
    facade_requests: tuple[AttachmentRouteRequestV2, ...],
    missing_contract_target_refs: tuple[ObjectRef, ...],
    blocked_mode_target_refs: tuple[ObjectRef, ...],
    audit: ContractAudit,
) -> ArtifactRoutingRequest:
    refs = (
        context_ref,
        producer_task_view_ref,
        policy_ref,
        *(() if matrix_ref is None else (matrix_ref,)),
        *tuple(artifact_build_contract_ref(contract) for contract in build_contracts),
        *missing_contract_target_refs,
        *blocked_mode_target_refs,
    )
    request = ArtifactRoutingRequest(
        request_id="artifact-routing-request://pending",
        attachment_planning_context_ref=context_ref,
        producer_task_view_ref=producer_task_view_ref,
        artifact_evidence_matrix_ref=matrix_ref,
        artifact_routing_policy_ref=policy_ref,
        build_contracts=build_contracts,
        facade_requests=facade_requests,
        missing_contract_target_refs=missing_contract_target_refs,
        blocked_mode_target_refs=blocked_mode_target_refs,
        policy_version=ARTIFACT_ROUTING_POLICY_VERSION,
        request_sha256="0" * 64,
        audit=_safe_audit(audit, refs),
    )
    digest = artifact_routing_request_carried_sha256(request)
    return request.model_copy(
        update={
            "request_id": f"artifact-routing-request://sha256/{digest}",
            "request_sha256": digest,
        }
    )


def _compile_build_spec(
    *,
    context_ref: ObjectRef,
    matrix_ref: ObjectRef,
    row: ArtifactEvidenceRowV2,
    target: ArtifactEvidenceTargetV2,
    contract: ArtifactBuildContractV2,
    routing_policy: ArtifactRoutingPolicyV2,
    facade_request: AttachmentRouteRequestV2,
    facade_decision: AttachmentRouteDecisionV2,
    producer_task_view: ProducerTaskViewV2,
    storage_authorization: ProducerStorageAuthorizationV2,
    producer_evidence_bundle: EvidenceBundle,
    audit: ContractAudit,
) -> ArtifactBuildSpecV2:
    if facade_decision.selected_id is None:
        raise ArtifactRoutingPolicyError("selected route decision is missing its selected ID")
    if facade_decision.outcome is AttachmentRouteOutcome.SELECTED_PROVIDER:
        if (
            facade_decision.selected_kind is not AttachmentRouteCandidateKind.PROVIDER
            or facade_decision.selected_id not in routing_policy.approved_provider_ids
            or contract.provider_payload_ref is None
            or not set(contract.required_provider_capability_ids).issubset(
                facade_decision.satisfied_capability_ids
            )
        ):
            raise ArtifactRoutingPolicyError("provider route does not satisfy the exact build contract")
        route_kind = ArtifactRouteKindV2.PROVIDER
        providers: tuple[str, ...] = (facade_decision.selected_id,)
        runtimes: tuple[str, ...] = ()
        model_profile_ref = None
    elif facade_decision.outcome is AttachmentRouteOutcome.SELECTED_RUNTIME:
        if (
            facade_decision.selected_kind is not AttachmentRouteCandidateKind.RUNTIME
            or facade_decision.selected_id not in routing_policy.runtime_order
        ):
            raise ArtifactRoutingPolicyError("runtime route is not approved by routing policy")
        if (
            not facade_decision.skipped
            or facade_decision.skipped[0].candidate_kind is not AttachmentRouteCandidateKind.PROVIDER
        ):
            raise ArtifactRoutingPolicyError("runtime route requires provider-first skip lineage")
        route_kind = ArtifactRouteKindV2.RUNTIME
        providers = ()
        runtimes = (facade_decision.selected_id,)
        index = routing_policy.runtime_order.index(facade_decision.selected_id)
        model_profile_ref = routing_policy.runtime_model_profile_refs[index]
    else:
        raise ArtifactRoutingPolicyError("blocked route decision cannot compile a build spec")
    authorized_evidence = _authorized_evidence(
        row=row,
        storage_authorization=storage_authorization,
        producer_evidence_bundle=producer_evidence_bundle,
    )
    row_ref = artifact_evidence_row_ref(row)
    frozen = ArtifactBuildSpec(
        artifact_build_spec_id="artifact-build-spec://pending",
        artifact_evidence_row_ref=row_ref,
        producer_task_view_ref=ObjectRef(
            object_type="producer-task-view",
            object_id=producer_task_view.producer_task_view_id,
            object_version="v2",
            object_sha256=producer_task_view.producer_task_view_sha256,
        ),
        relative_path=contract.logical_path,
        media_type=contract.media_type,
        mode=contract.mode,
        content_contract_ref=contract.content_contract_ref,
        render_contract_ref=contract.render_contract_ref,
        authorized_evidence_refs=authorized_evidence,
        provider_preference=providers,
        runtime_preference=runtimes,
        validator_ids=contract.validator_ids,
        retry_scope="ARTIFACT",
        build_spec_sha256="0" * 64,
        audit=_safe_audit(
            audit,
            (
                row_ref,
                contract.producer_task_view_ref,
                contract.content_contract_ref,
                contract.render_contract_ref,
                *tuple(evidence.subject_ref for evidence in authorized_evidence),
            ),
        ),
    )
    frozen_digest = _frozen_build_spec_carried_sha256(frozen)
    frozen = frozen.model_copy(
        update={
            "artifact_build_spec_id": (f"artifact-build-spec://sha256/{frozen_digest}"),
            "build_spec_sha256": frozen_digest,
        }
    )
    request_ref = _factory_ref(attachment_route_request_ref(facade_request))
    decision_ref = _factory_ref(attachment_route_decision_ref(facade_decision))
    spec = ArtifactBuildSpecV2(
        artifact_build_spec_v2_id="artifact-build-spec://pending",
        attachment_planning_context_ref=context_ref,
        artifact_evidence_matrix_ref=matrix_ref,
        artifact_evidence_target_ref=artifact_evidence_target_ref(target),
        artifact_build_contract_ref=artifact_build_contract_ref(contract),
        artifact_routing_policy_ref=artifact_routing_policy_ref(routing_policy),
        facade_route_request_ref=request_ref,
        facade_route_decision_ref=decision_ref,
        selected_route_kind=route_kind,
        selected_model_profile_ref=model_profile_ref,
        source_evidence_set_ref=contract.source_evidence_set_ref,
        build_spec=frozen,
        policy_version=ARTIFACT_ROUTING_POLICY_VERSION,
        artifact_build_spec_v2_sha256="0" * 64,
        audit=_safe_audit(
            audit,
            (
                context_ref,
                matrix_ref,
                artifact_evidence_target_ref(target),
                artifact_build_contract_ref(contract),
                artifact_routing_policy_ref(routing_policy),
                request_ref,
                decision_ref,
                *(() if model_profile_ref is None else (model_profile_ref,)),
                *(() if contract.source_evidence_set_ref is None else (contract.source_evidence_set_ref,)),
            ),
        ),
    )
    digest = artifact_build_spec_v2_carried_sha256(spec)
    return spec.model_copy(
        update={
            "artifact_build_spec_v2_id": (f"artifact-build-spec://sha256/{digest}"),
            "artifact_build_spec_v2_sha256": digest,
        }
    )


def _authorized_evidence(
    *,
    row: ArtifactEvidenceRowV2,
    storage_authorization: ProducerStorageAuthorizationV2,
    producer_evidence_bundle: EvidenceBundle,
) -> tuple[EvidenceRef, ...]:
    bundle_by_id = {item.evidence_ref_id: item for item in producer_evidence_bundle.evidence}
    authorized_subjects = set(storage_authorization.authorized_subject_refs)
    selected = tuple(
        sorted(
            (
                bundle_by_id[item.evidence_ref_id]
                for item in row.row.evidence
                if item.evidence_ref_id in bundle_by_id
                and bundle_by_id[item.evidence_ref_id].subject_ref in authorized_subjects
                and bundle_by_id[item.evidence_ref_id] == item
            ),
            key=lambda item: item.evidence_ref_id,
        )
    )
    return selected


def _frozen_build_spec_carried_sha256(
    spec: ArtifactBuildSpec,
) -> str:
    return _payload_sha256(
        spec.model_dump(
            mode="json",
            exclude={
                "artifact_build_spec_id",
                "build_spec_sha256",
                "audit",
            },
            exclude_none=False,
        )
    )


def _blocked_entry(
    target: ArtifactEvidenceTargetV2,
    outcome: ArtifactRouteEntryOutcomeV2,
    reason: ArtifactRoutingReasonV2,
) -> ArtifactRoutePlanEntryV2:
    return ArtifactRoutePlanEntryV2(
        artifact_evidence_target_ref=artifact_evidence_target_ref(target),
        attachment_dependency_id=target.attachment_dependency_id,
        artifact_id=target.artifact_id,
        criticality=target.criticality,
        outcome=outcome,
        facade_route_request_ref=None,
        facade_route_decision=None,
        build_spec=None,
        reasons=frozenset({reason}),
    )


def _facade_request_for_contract(
    request: ArtifactRoutingRequest,
    contract: ArtifactBuildContractV2,
) -> AttachmentRouteRequestV2:
    for facade_request in request.facade_requests:
        ref = facade_request.build_contract_ref
        if (
            ref.object_id == contract.artifact_build_contract_id
            and ref.object_sha256 == contract.artifact_build_contract_sha256
        ):
            return facade_request
    raise ArtifactRoutingPolicyError("artifact build contract has no exact facade request")


def _aggregate_outcome(
    entries: tuple[ArtifactRoutePlanEntryV2, ...],
) -> ArtifactRoutingAggregateOutcomeV2:
    if any(
        item.outcome is ArtifactRouteEntryOutcomeV2.BLOCKED_POLICY
        and item.criticality is not AttachmentCriticality.OPTIONAL
        for item in entries
    ):
        return ArtifactRoutingAggregateOutcomeV2.BLOCKED_POLICY
    if any(
        item.outcome is ArtifactRouteEntryOutcomeV2.BLOCKED_CAPABILITY
        and item.criticality is not AttachmentCriticality.OPTIONAL
        for item in entries
    ):
        return ArtifactRoutingAggregateOutcomeV2.BLOCKED_CAPABILITY
    if any(
        item.outcome
        in {
            ArtifactRouteEntryOutcomeV2.BLOCKED_CAPABILITY,
            ArtifactRouteEntryOutcomeV2.BLOCKED_POLICY,
        }
        for item in entries
    ):
        return ArtifactRoutingAggregateOutcomeV2.PARTIALLY_ROUTED
    return ArtifactRoutingAggregateOutcomeV2.ROUTED


def _compilation_result(
    *,
    request: ArtifactRoutingRequest,
    execution_result: ArtifactRoutingExecutionResult,
    outcome: ArtifactRoutingAggregateOutcomeV2,
    routing_plan: ArtifactRoutingPlanV2 | None,
    audit: ContractAudit,
) -> ArtifactRoutingCompilationResult:
    result = ArtifactRoutingCompilationResult(
        result_id="artifact-routing-compilation-result://pending",
        request_ref=artifact_routing_request_ref(request),
        execution_result_ref=artifact_routing_execution_result_ref(execution_result),
        outcome=outcome,
        routing_plan=routing_plan,
        policy_version=ARTIFACT_ROUTING_POLICY_VERSION,
        result_sha256="0" * 64,
        audit=_safe_audit(
            audit,
            (
                artifact_routing_request_ref(request),
                artifact_routing_execution_result_ref(execution_result),
                *(
                    ()
                    if routing_plan is None
                    else (
                        ObjectRef(
                            object_type="artifact-routing-plan",
                            object_id=(routing_plan.artifact_routing_plan_id),
                            object_version="v2",
                            object_sha256=(routing_plan.artifact_routing_plan_sha256),
                        ),
                    )
                ),
            ),
        ),
    )
    digest = artifact_routing_compilation_result_carried_sha256(result)
    return result.model_copy(
        update={
            "result_id": (f"artifact-routing-compilation-result://sha256/{digest}"),
            "result_sha256": digest,
        }
    )


def _validate_routing_policy_identity(
    policy: ArtifactRoutingPolicyV2,
) -> None:
    digest = artifact_routing_policy_carried_sha256(policy)
    if (
        policy.policy_version != ARTIFACT_ROUTING_POLICY_VERSION
        or policy.artifact_routing_policy_sha256 != digest
        or policy.artifact_routing_policy_id != f"artifact-routing-policy://sha256/{digest}"
    ):
        raise ArtifactRoutingPolicyError("artifact routing policy identity is stale or invalid")
    expected_refs = tuple(
        sorted(
            (
                *policy.runtime_model_profile_refs,
                policy.provider_capability_policy_ref,
                policy.runtime_capability_policy_ref,
                policy.validator_policy_ref,
            ),
            key=_ref_key,
        )
    )
    if policy.audit.input_refs != expected_refs:
        raise ArtifactRoutingPolicyError("artifact routing policy audit lineage is stale or invalid")


def _validate_matrix_identity(matrix: ArtifactEvidenceMatrixV2) -> None:
    for row in matrix.rows:
        digest = artifact_evidence_row_carried_sha256(row)
        if (
            row.artifact_evidence_row_sha256 != digest
            or row.artifact_evidence_row_id != f"artifact-evidence-row://sha256/{digest}"
        ):
            raise ArtifactRoutingPolicyError("artifact evidence row identity is stale or invalid")
    digest = artifact_evidence_matrix_carried_sha256(matrix)
    if (
        matrix.artifact_evidence_matrix_sha256 != digest
        or matrix.artifact_evidence_matrix_id != f"artifact-evidence-matrix://sha256/{digest}"
    ):
        raise ArtifactRoutingPolicyError("artifact evidence matrix identity is stale or invalid")


def _validate_request_identity(request: ArtifactRoutingRequest) -> None:
    digest = artifact_routing_request_carried_sha256(request)
    if (
        request.policy_version != ARTIFACT_ROUTING_POLICY_VERSION
        or request.request_sha256 != digest
        or request.request_id != f"artifact-routing-request://sha256/{digest}"
    ):
        raise ArtifactRoutingPolicyError("artifact routing request identity is stale or invalid")
    for contract in request.build_contracts:
        contract_digest = artifact_build_contract_carried_sha256(contract)
        if (
            contract.policy_version != ARTIFACT_ROUTING_POLICY_VERSION
            or contract.artifact_build_contract_sha256 != contract_digest
            or contract.artifact_build_contract_id != f"artifact-build-contract://sha256/{contract_digest}"
        ):
            raise ArtifactRoutingPolicyError("artifact build contract identity is stale or invalid")
        expected_contract_refs = tuple(
            sorted(
                (
                    contract.attachment_planning_context_ref,
                    contract.producer_task_view_ref,
                    contract.artifact_evidence_matrix_ref,
                    contract.artifact_evidence_row_ref,
                    contract.artifact_evidence_target_ref,
                    contract.content_contract_ref,
                    contract.render_contract_ref,
                    *(() if contract.provider_payload_ref is None else (contract.provider_payload_ref,)),
                    *(
                        ()
                        if contract.source_evidence_set_ref is None
                        else (contract.source_evidence_set_ref,)
                    ),
                ),
                key=_ref_key,
            )
        )
        if (
            contract.audit.governing_versions != request.audit.governing_versions
            or contract.audit.input_refs != expected_contract_refs
        ):
            raise ArtifactRoutingPolicyError("artifact build contract audit lineage is stale or invalid")
    expected_request_refs = tuple(
        sorted(
            (
                request.attachment_planning_context_ref,
                request.producer_task_view_ref,
                request.artifact_routing_policy_ref,
                *(
                    ()
                    if request.artifact_evidence_matrix_ref is None
                    else (request.artifact_evidence_matrix_ref,)
                ),
                *tuple(artifact_build_contract_ref(contract) for contract in request.build_contracts),
                *request.missing_contract_target_refs,
                *request.blocked_mode_target_refs,
            ),
            key=_ref_key,
        )
    )
    if request.audit.input_refs != expected_request_refs:
        raise ArtifactRoutingPolicyError("artifact routing request audit lineage is stale or invalid")
    for facade_request in request.facade_requests:
        _validate_facade_request_identity(facade_request)


def _validate_execution_result_identity(
    result: ArtifactRoutingExecutionResult,
) -> None:
    digest = artifact_routing_execution_result_carried_sha256(result)
    if (
        result.policy_version != ARTIFACT_ROUTING_POLICY_VERSION
        or result.result_sha256 != digest
        or result.result_id != f"artifact-routing-execution-result://sha256/{digest}"
    ):
        raise ArtifactRoutingPolicyError("artifact routing execution result identity is stale or invalid")


def _validate_compilation_result_identity(
    result: ArtifactRoutingCompilationResult,
) -> None:
    digest = artifact_routing_compilation_result_carried_sha256(result)
    if (
        result.policy_version != ARTIFACT_ROUTING_POLICY_VERSION
        or result.result_sha256 != digest
        or result.result_id != f"artifact-routing-compilation-result://sha256/{digest}"
    ):
        raise ArtifactRoutingPolicyError("artifact routing compilation result identity is stale or invalid")
    if result.routing_plan is not None:
        plan_digest = artifact_routing_plan_carried_sha256(result.routing_plan)
        if (
            result.routing_plan.artifact_routing_plan_sha256 != plan_digest
            or result.routing_plan.artifact_routing_plan_id != f"artifact-routing-plan://sha256/{plan_digest}"
        ):
            raise ArtifactRoutingPolicyError("artifact routing plan identity is stale or invalid")


def _validate_facade_request_identity(
    request: AttachmentRouteRequestV2,
) -> None:
    digest = attachment_route_request_carried_sha256(request)
    if (
        request.route_request_sha256 != digest
        or request.route_request_id != f"attachment-route-request://sha256/{digest}"
    ):
        raise ArtifactRoutingPolicyError("facade route request identity is stale or invalid")


def _validate_facade_decision_identity(
    decision: AttachmentRouteDecisionV2,
) -> None:
    digest = attachment_route_decision_carried_sha256(decision)
    if (
        decision.route_decision_sha256 != digest
        or decision.route_decision_id != f"attachment-route-decision://sha256/{digest}"
    ):
        raise ArtifactRoutingPolicyError("facade route decision identity is stale or invalid")


def _validate_decision_against_request(
    request: AttachmentRouteRequestV2,
    decision: AttachmentRouteDecisionV2,
) -> None:
    runtime_skips = tuple(
        item.candidate_id
        for item in decision.skipped
        if item.candidate_kind is AttachmentRouteCandidateKind.RUNTIME
    )
    if decision.outcome is AttachmentRouteOutcome.SELECTED_PROVIDER:
        if (
            decision.selected_id not in request.approved_provider_ids
            or runtime_skips
            or decision.satisfied_capability_ids != request.required_provider_capability_ids
        ):
            raise ArtifactRoutingPolicyError("provider decision does not match its route request")
        return
    if decision.outcome is AttachmentRouteOutcome.SELECTED_RUNTIME:
        if decision.selected_id not in request.runtime_order:
            raise ArtifactRoutingPolicyError("runtime decision is not present in route request order")
        selected_index = request.runtime_order.index(decision.selected_id)
        if (
            runtime_skips != request.runtime_order[:selected_index]
            or decision.satisfied_capability_ids != request.required_runtime_tools
        ):
            raise ArtifactRoutingPolicyError("runtime decision skips or capabilities do not match request")
        return
    if runtime_skips != request.runtime_order or decision.satisfied_capability_ids:
        raise ArtifactRoutingPolicyError("blocked decision must exhaust requested runtime order")


def _facade_ref(ref: ObjectRef) -> FacadeObjectRef:
    return FacadeObjectRef(
        object_type=ref.object_type,
        object_id=ref.object_id,
        object_version=ref.object_version,
        object_sha256=ref.object_sha256,
    )


def _factory_ref(ref: FacadeObjectRef) -> ObjectRef:
    return ObjectRef(
        object_type=ref.object_type,
        object_id=ref.object_id,
        object_version=ref.object_version,
        object_sha256=ref.object_sha256,
    )


def _safe_audit(
    audit: ContractAudit,
    refs: tuple[ObjectRef, ...],
) -> ContractAudit:
    return ContractAudit(
        created_at=audit.created_at,
        created_by=audit.created_by,
        governing_versions=audit.governing_versions,
        input_refs=tuple(sorted(set(refs), key=_ref_key)),
    )


def _ref_key(ref: ObjectRef) -> tuple[str, str, str, str]:
    return (
        ref.object_type,
        ref.object_id,
        ref.object_version,
        ref.object_sha256,
    )


def _payload_sha256(payload: object) -> str:
    encoded = json.dumps(
        canonical_value_v2(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _without_audit_actor_time(value: object) -> object:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json", exclude_none=False)
    if isinstance(value, dict):
        normalized: dict[str, object] = {}
        for key, item in value.items():
            if key == "audit" and isinstance(item, dict):
                normalized[key] = {
                    "schema_version": item.get("schema_version"),
                    "governing_versions": _without_audit_actor_time(item.get("governing_versions", [])),
                    "input_refs": _without_audit_actor_time(item.get("input_refs", [])),
                }
            else:
                normalized[key] = _without_audit_actor_time(item)
        return normalized
    if isinstance(value, list):
        return [_without_audit_actor_time(item) for item in value]
    return value
