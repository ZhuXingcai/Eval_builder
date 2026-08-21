from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol, cast

from env_mock_agent.facade.contracts import FacadeObjectRef
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
from env_mock_agent.facade.semantic_review_v2 import (
    ATTACHMENT_REPAIR_POLICY_VERSION,
    ATTACHMENT_SEMANTIC_REVIEW_POLICY_VERSION,
    AttachmentRepairFailureCodeV2,
    AttachmentRepairOutcomeV2,
    AttachmentRepairRequestV2,
    AttachmentRepairResultV2,
    AttachmentSemanticFindingResolutionV2,
    AttachmentSemanticResolutionDispositionV2,
    AttachmentSemanticReviewerRoleV2,
    AttachmentSemanticReviewFailureCodeV2,
    AttachmentSemanticReviewFindingCodeV2,
    AttachmentSemanticReviewFindingV2,
    AttachmentSemanticReviewOutcomeV2,
    AttachmentSemanticReviewRequestV2,
    AttachmentSemanticReviewResultV2,
    SemanticCleanContextAttestationV2,
    attachment_repair_request_ref,
    attachment_repair_result_carried_sha256,
    attachment_repair_result_ref,
    attachment_semantic_finding_policy,
    attachment_semantic_finding_resolution_carried_sha256,
    attachment_semantic_finding_resolution_ref,
    attachment_semantic_review_finding_carried_sha256,
    attachment_semantic_review_finding_ref,
    attachment_semantic_review_request_ref,
    attachment_semantic_review_result_carried_sha256,
    attachment_semantic_review_result_ref,
    validate_attachment_repair_request_identity,
    validate_attachment_semantic_review_request_identity,
)


@dataclass(frozen=True)
class ProviderSemanticReviewMaterial:
    context_view_ref: FacadeObjectRef
    included_ref_inventory: tuple[FacadeObjectRef, ...]
    denied_data_families: tuple[str, ...]
    allowed_evidence_ref_ids: frozenset[str]
    safe_payload: Mapping[str, object]


@dataclass(frozen=True)
class ProviderAttachmentRepairMaterial:
    source_output_ref: FacadeObjectRef
    source_output_bytes: bytes
    logical_path: str
    media_type: str


class AttachmentSemanticReviewMaterialResolver(Protocol):
    def resolve_review_material(
        self,
        request: AttachmentSemanticReviewRequestV2,
    ) -> ProviderSemanticReviewMaterial: ...

    def resolve_repair_material(
        self,
        request: AttachmentRepairRequestV2,
    ) -> ProviderAttachmentRepairMaterial: ...


class RepairedAttachmentOutputRegistrar(Protocol):
    def register_repaired_output(
        self,
        *,
        request: AttachmentRepairRequestV2,
        material: ProviderAttachmentRepairMaterial,
        output_ref: FacadeObjectRef,
        output_bytes: bytes,
    ) -> None: ...


class MappingAttachmentSemanticReviewMaterialResolver:
    def __init__(
        self,
        *,
        review_materials: Mapping[str, ProviderSemanticReviewMaterial],
        repair_materials: Mapping[str, ProviderAttachmentRepairMaterial],
    ) -> None:
        self._review_materials = dict(review_materials)
        self._repair_materials = dict(repair_materials)

    def resolve_review_material(
        self,
        request: AttachmentSemanticReviewRequestV2,
    ) -> ProviderSemanticReviewMaterial:
        try:
            return self._review_materials[request.context_view_ref.object_id]
        except KeyError as exc:
            raise LookupError("semantic review context material is unavailable") from exc

    def resolve_repair_material(
        self,
        request: AttachmentRepairRequestV2,
    ) -> ProviderAttachmentRepairMaterial:
        try:
            return self._repair_materials[request.repair_plan_ref.object_id]
        except KeyError as exc:
            raise LookupError("attachment repair material is unavailable") from exc


@dataclass(frozen=True)
class ProviderSemanticFindingDecision:
    code: AttachmentSemanticReviewFindingCodeV2
    subject_refs: tuple[FacadeObjectRef, ...]
    artifact_ids: tuple[str, ...]
    evidence_ref_ids: tuple[str, ...]
    predecessor_finding_ref: FacadeObjectRef | None = None


@dataclass(frozen=True)
class ProviderSemanticResolutionDecision:
    prior_finding_ref: FacadeObjectRef
    old_subject_refs: tuple[FacadeObjectRef, ...]
    current_subject_refs: tuple[FacadeObjectRef, ...]
    repair_plan_ref: FacadeObjectRef
    repair_result_refs: tuple[FacadeObjectRef, ...]
    disposition: AttachmentSemanticResolutionDispositionV2
    successor_finding_ref: FacadeObjectRef | None
    evidence_ref_ids: tuple[str, ...]


@dataclass(frozen=True)
class ProviderSemanticReviewDecision:
    outcome: AttachmentSemanticReviewOutcomeV2
    findings: tuple[ProviderSemanticFindingDecision, ...] = ()
    resolutions: tuple[ProviderSemanticResolutionDecision, ...] = ()
    failure_code: AttachmentSemanticReviewFailureCodeV2 | None = None

    @classmethod
    def accepted(cls) -> ProviderSemanticReviewDecision:
        return cls(outcome=AttachmentSemanticReviewOutcomeV2.ACCEPTED)

    @classmethod
    def requires_repair(
        cls,
        *,
        findings: tuple[ProviderSemanticFindingDecision, ...],
    ) -> ProviderSemanticReviewDecision:
        return cls(
            outcome=AttachmentSemanticReviewOutcomeV2.REQUIRES_REPAIR,
            findings=findings,
        )

    @classmethod
    def rejected(
        cls,
        *,
        findings: tuple[ProviderSemanticFindingDecision, ...],
    ) -> ProviderSemanticReviewDecision:
        return cls(
            outcome=AttachmentSemanticReviewOutcomeV2.REJECTED,
            findings=findings,
        )


@dataclass(frozen=True)
class ProviderSemanticReviewContext:
    context_id: str
    request: AttachmentSemanticReviewRequestV2
    material: ProviderSemanticReviewMaterial


class SemanticReviewerBackend(Protocol):
    async def review(
        self,
        context: ProviderSemanticReviewContext,
    ) -> ProviderSemanticReviewDecision: ...


@dataclass(frozen=True)
class ProviderAttachmentRepairDecision:
    outcome: AttachmentRepairOutcomeV2
    output_bytes: bytes | None = None
    worker_version: str | None = None
    failure_code: AttachmentRepairFailureCodeV2 | None = None

    @classmethod
    def succeeded(
        cls,
        *,
        output_bytes: bytes,
        worker_version: str,
    ) -> ProviderAttachmentRepairDecision:
        return cls(
            outcome=AttachmentRepairOutcomeV2.SUCCEEDED,
            output_bytes=output_bytes,
            worker_version=worker_version,
        )


@dataclass(frozen=True)
class ProviderAttachmentRepairContext:
    context_id: str
    request: AttachmentRepairRequestV2
    material: ProviderAttachmentRepairMaterial


class AttachmentRepairBackend(Protocol):
    async def repair(
        self,
        context: ProviderAttachmentRepairContext,
    ) -> ProviderAttachmentRepairDecision: ...


class RegistryAttachmentSemanticReviewFacade:
    def __init__(
        self,
        *,
        resolver: AttachmentSemanticReviewMaterialResolver,
        review_backends: Mapping[
            AttachmentSemanticReviewerRoleV2,
            SemanticReviewerBackend,
        ],
        repair_backend: AttachmentRepairBackend | None,
        repaired_output_registrar: RepairedAttachmentOutputRegistrar | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._resolver = resolver
        self._review_backends = dict(review_backends)
        self._repair_backend = repair_backend
        self._repaired_output_registrar = repaired_output_registrar
        self._clock = clock or (lambda: datetime.now(UTC))
        self._review_cache: dict[
            str,
            tuple[str, AttachmentSemanticReviewResultV2],
        ] = {}
        self._repair_cache: dict[
            str,
            tuple[str, AttachmentRepairResultV2],
        ] = {}
        self._model_review_cache: dict[
            str,
            tuple[
                str,
                AttachmentSemanticReviewResultV2 | None,
                ModelControlReceiptV2,
            ],
        ] = {}
        self._model_repair_cache: dict[
            str,
            tuple[
                str,
                AttachmentRepairResultV2 | None,
                ModelControlReceiptV2,
            ],
        ] = {}
        self._model_active: dict[
            str,
            tuple[asyncio.Task[object], str, int],
        ] = {}
        self._model_stopped: set[str] = set()
        self._model_cancellations: dict[
            str,
            ModelInvocationCancellationResultV2,
        ] = {}

    async def review_with_model_control(
        self,
        *,
        request: AttachmentSemanticReviewRequestV2,
        descriptor: ModelInvocationDescriptorV2,
        grant: ModelControlGrantV2,
    ) -> tuple[
        AttachmentSemanticReviewResultV2 | None,
        ModelControlReceiptV2,
    ]:
        validate_attachment_semantic_review_request_identity(request)
        validate_model_invocation_descriptor_identity(descriptor)
        validate_model_control_grant_identity(grant)
        grant.validate_descriptor(descriptor)
        if descriptor.mode is not ModelDemandModeFacadeV2.DIRECT_REQUEST:
            raise ValueError("semantic review model control requires DIRECT_REQUEST mode")
        if descriptor.operation_ref != attachment_semantic_review_request_ref(request):
            raise ValueError("model descriptor does not bind the semantic review request")
        if descriptor.model_profile_ref != request.model_profile_ref:
            raise ValueError("model descriptor does not bind the semantic review profile")
        request_key = (
            f"{request.semantic_review_request_sha256}:"
            f"{descriptor.invocation_descriptor_sha256}:"
            f"{grant.model_control_grant_sha256}"
        )
        cached = self._model_review_cache.get(descriptor.idempotency_key)
        if cached is not None:
            if cached[0] != request_key:
                raise ValueError("model-controlled semantic review idempotency conflict")
            return cached[1], cached[2]
        handle_id = grant.invocation_handle_ref.object_id
        task = cast(
            asyncio.Task[object],
            asyncio.create_task(self.review(request)),
        )
        self._model_active[handle_id] = (
            task,
            grant.reservation_ref.object_id,
            grant.fencing_token,
        )
        try:
            result = cast(AttachmentSemanticReviewResultV2, await task)
            receipt = self._semantic_review_model_receipt(
                descriptor=descriptor,
                grant=grant,
                result=result,
            )
        finally:
            self._model_active.pop(handle_id, None)
            self._model_stopped.add(handle_id)
        self._model_review_cache[descriptor.idempotency_key] = (
            request_key,
            result,
            receipt,
        )
        return result, receipt

    async def repair_with_model_control(
        self,
        *,
        request: AttachmentRepairRequestV2,
        descriptor: ModelInvocationDescriptorV2,
        grant: ModelControlGrantV2,
    ) -> tuple[AttachmentRepairResultV2 | None, ModelControlReceiptV2]:
        validate_attachment_repair_request_identity(request)
        validate_model_invocation_descriptor_identity(descriptor)
        validate_model_control_grant_identity(grant)
        grant.validate_descriptor(descriptor)
        if descriptor.mode is not ModelDemandModeFacadeV2.DIRECT_REQUEST:
            raise ValueError("attachment repair model control requires DIRECT_REQUEST mode")
        if descriptor.operation_ref != attachment_repair_request_ref(request):
            raise ValueError("model descriptor does not bind the attachment repair request")
        request_key = (
            f"{request.repair_request_sha256}:"
            f"{descriptor.invocation_descriptor_sha256}:"
            f"{grant.model_control_grant_sha256}"
        )
        cached = self._model_repair_cache.get(descriptor.idempotency_key)
        if cached is not None:
            if cached[0] != request_key:
                raise ValueError("model-controlled attachment repair idempotency conflict")
            return cached[1], cached[2]
        handle_id = grant.invocation_handle_ref.object_id
        task = cast(
            asyncio.Task[object],
            asyncio.create_task(self.repair(request)),
        )
        self._model_active[handle_id] = (
            task,
            grant.reservation_ref.object_id,
            grant.fencing_token,
        )
        try:
            result = cast(AttachmentRepairResultV2, await task)
            receipt = self._attachment_repair_model_receipt(
                descriptor=descriptor,
                grant=grant,
                result=result,
            )
        finally:
            self._model_active.pop(handle_id, None)
            self._model_stopped.add(handle_id)
        self._model_repair_cache[descriptor.idempotency_key] = (
            request_key,
            result,
            receipt,
        )
        return result, receipt

    async def cancel_model_invocation(
        self,
        request: ModelInvocationCancellationRequestV2,
    ) -> ModelInvocationCancellationResultV2:
        validate_model_invocation_cancellation_request_identity(request)
        cached = self._model_cancellations.get(request.cancellation_request_id)
        if cached is not None:
            return cached
        handle_id = request.invocation_handle_ref.object_id
        active = self._model_active.get(handle_id)
        if active is None:
            outcome = (
                ModelInvocationCancellationOutcomeV2.ALREADY_FINISHED
                if handle_id in self._model_stopped
                else ModelInvocationCancellationOutcomeV2.FAILED
            )
        else:
            task, reservation_id, fence = active
            if reservation_id != request.reservation_ref.object_id or fence != request.fencing_token:
                outcome = ModelInvocationCancellationOutcomeV2.FAILED
            else:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
                outcome = ModelInvocationCancellationOutcomeV2.CANCELLED
        result = ModelInvocationCancellationResultV2.create(
            cancellation_request_ref=(model_invocation_cancellation_request_ref(request)),
            outcome=outcome,
            completed_at=self._clock(),
        )
        self._model_cancellations[request.cancellation_request_id] = result
        return result

    def _semantic_review_model_receipt(
        self,
        *,
        descriptor: ModelInvocationDescriptorV2,
        grant: ModelControlGrantV2,
        result: AttachmentSemanticReviewResultV2,
    ) -> ModelControlReceiptV2:
        if result.outcome is not AttachmentSemanticReviewOutcomeV2.BLOCKED:
            return self._model_receipt(
                descriptor=descriptor,
                grant=grant,
                outcome=ModelControlReceiptOutcomeV2.SUCCEEDED,
                result_ref=attachment_semantic_review_result_ref(result),
                usage=ModelControlUsageV2.conservative(
                    requests=grant.request_allowance,
                    tokens=grant.token_allowance,
                ),
                failure_code=None,
            )
        no_model_failure = {
            AttachmentSemanticReviewFailureCodeV2.CONTEXT_MISSING,
            AttachmentSemanticReviewFailureCodeV2.CONTEXT_MISMATCH,
            AttachmentSemanticReviewFailureCodeV2.MODEL_UNAVAILABLE,
        }
        if result.failure_code in no_model_failure:
            outcome = (
                ModelControlReceiptOutcomeV2.BLOCKED_CAPABILITY
                if result.failure_code is AttachmentSemanticReviewFailureCodeV2.MODEL_UNAVAILABLE
                else ModelControlReceiptOutcomeV2.BLOCKED_POLICY
            )
            usage = _zero_model_usage()
        else:
            outcome = ModelControlReceiptOutcomeV2.FAILED
            usage = ModelControlUsageV2.conservative(
                requests=grant.request_allowance,
                tokens=grant.token_allowance,
            )
        return self._model_receipt(
            descriptor=descriptor,
            grant=grant,
            outcome=outcome,
            result_ref=None,
            usage=usage,
            failure_code=(
                result.failure_code.value if result.failure_code is not None else "SEMANTIC_REVIEW_FAILED"
            ),
        )

    def _attachment_repair_model_receipt(
        self,
        *,
        descriptor: ModelInvocationDescriptorV2,
        grant: ModelControlGrantV2,
        result: AttachmentRepairResultV2,
    ) -> ModelControlReceiptV2:
        if result.outcome is AttachmentRepairOutcomeV2.SUCCEEDED:
            return self._model_receipt(
                descriptor=descriptor,
                grant=grant,
                outcome=ModelControlReceiptOutcomeV2.SUCCEEDED,
                result_ref=attachment_repair_result_ref(result),
                usage=ModelControlUsageV2.conservative(
                    requests=grant.request_allowance,
                    tokens=grant.token_allowance,
                ),
                failure_code=None,
            )
        if result.outcome in {
            AttachmentRepairOutcomeV2.BLOCKED_CAPABILITY,
            AttachmentRepairOutcomeV2.BLOCKED_POLICY,
        }:
            outcome = (
                ModelControlReceiptOutcomeV2.BLOCKED_CAPABILITY
                if result.outcome is AttachmentRepairOutcomeV2.BLOCKED_CAPABILITY
                else ModelControlReceiptOutcomeV2.BLOCKED_POLICY
            )
            usage = _zero_model_usage()
        else:
            outcome = ModelControlReceiptOutcomeV2.FAILED
            usage = ModelControlUsageV2.conservative(
                requests=grant.request_allowance,
                tokens=grant.token_allowance,
            )
        return self._model_receipt(
            descriptor=descriptor,
            grant=grant,
            outcome=outcome,
            result_ref=None,
            usage=usage,
            failure_code=(
                result.failure_code.value if result.failure_code is not None else "ATTACHMENT_REPAIR_FAILED"
            ),
        )

    def _model_receipt(
        self,
        *,
        descriptor: ModelInvocationDescriptorV2,
        grant: ModelControlGrantV2,
        outcome: ModelControlReceiptOutcomeV2,
        result_ref: FacadeObjectRef | None,
        usage: ModelControlUsageV2,
        failure_code: str | None,
    ) -> ModelControlReceiptV2:
        return ModelControlReceiptV2.create(
            descriptor_ref=model_invocation_descriptor_ref(descriptor),
            grant_ref=model_control_grant_ref(grant),
            outcome=outcome,
            result_ref=result_ref,
            usage=usage,
            backpressure_ref=None,
            failure_code=failure_code,
            completed_at=self._clock(),
        )

    async def review(
        self,
        request: AttachmentSemanticReviewRequestV2,
    ) -> AttachmentSemanticReviewResultV2:
        validate_attachment_semantic_review_request_identity(request)
        cached = self._review_cache.get(request.idempotency_key)
        if cached is not None:
            request_sha256, result = cached
            if request_sha256 != request.semantic_review_request_sha256:
                raise ValueError("semantic review idempotency conflict")
            return result
        try:
            material = self._resolver.resolve_review_material(request)
        except LookupError:
            return self._cache_review(
                request,
                self._blocked_review(
                    request,
                    AttachmentSemanticReviewFailureCodeV2.CONTEXT_MISSING,
                ),
            )
        if not _review_material_matches(request, material):
            return self._cache_review(
                request,
                self._blocked_review(
                    request,
                    AttachmentSemanticReviewFailureCodeV2.CONTEXT_MISMATCH,
                ),
            )
        backend = self._review_backends.get(request.reviewer_role)
        if backend is None:
            return self._cache_review(
                request,
                self._blocked_review(
                    request,
                    AttachmentSemanticReviewFailureCodeV2.MODEL_UNAVAILABLE,
                ),
            )
        context_id = (
            f"semantic-review-context://sha256/{_sha256_text(request.semantic_review_request_sha256)}"
        )
        try:
            decision = await backend.review(
                ProviderSemanticReviewContext(
                    context_id=context_id,
                    request=request,
                    material=material,
                )
            )
            _validate_decision_evidence(decision, material)
            findings = tuple(self._finding(request, item) for item in decision.findings)
            resolutions = tuple(self._resolution(request, item) for item in decision.resolutions)
            _validate_resolution_decisions(
                request,
                findings=findings,
                resolutions=resolutions,
            )
            attestation = SemanticCleanContextAttestationV2.create(
                semantic_review_request_ref=attachment_semantic_review_request_ref(request),
                stage_run_ref=request.stage_run_ref,
                round=request.round,
                reviewer_role=request.reviewer_role,
                context_view_ref=request.context_view_ref,
                included_ref_inventory=material.included_ref_inventory,
                denied_data_families=tuple(material.denied_data_families),
            )
            result = self._review_result(
                request,
                outcome=decision.outcome,
                attestation=attestation,
                findings=findings,
                resolutions=resolutions,
                failure_code=decision.failure_code,
            )
        except (TypeError, ValueError):
            result = self._blocked_review(
                request,
                AttachmentSemanticReviewFailureCodeV2.BACKEND_OUTPUT_INVALID,
            )
        except Exception:
            result = self._blocked_review(
                request,
                AttachmentSemanticReviewFailureCodeV2.BACKEND_FAILED,
            )
        return self._cache_review(request, result)

    async def repair(
        self,
        request: AttachmentRepairRequestV2,
    ) -> AttachmentRepairResultV2:
        validate_attachment_repair_request_identity(request)
        cached = self._repair_cache.get(request.idempotency_key)
        if cached is not None:
            request_sha256, result = cached
            if request_sha256 != request.repair_request_sha256:
                raise ValueError("attachment repair idempotency conflict")
            return result
        try:
            material = self._resolver.resolve_repair_material(request)
        except LookupError:
            return self._cache_repair(
                request,
                self._repair_failure(
                    request,
                    AttachmentRepairOutcomeV2.BLOCKED_POLICY,
                    AttachmentRepairFailureCodeV2.REPAIR_MATERIAL_MISSING,
                ),
            )
        if (
            material.source_output_ref != request.output_ref
            or hashlib.sha256(material.source_output_bytes).hexdigest() != request.output_sha256
        ):
            return self._cache_repair(
                request,
                self._repair_failure(
                    request,
                    AttachmentRepairOutcomeV2.BLOCKED_POLICY,
                    AttachmentRepairFailureCodeV2.REPAIR_MATERIAL_MISMATCH,
                ),
            )
        if self._repair_backend is None:
            return self._cache_repair(
                request,
                self._repair_failure(
                    request,
                    AttachmentRepairOutcomeV2.BLOCKED_CAPABILITY,
                    AttachmentRepairFailureCodeV2.REPAIR_BACKEND_UNAVAILABLE,
                ),
            )
        try:
            decision = await self._repair_backend.repair(
                ProviderAttachmentRepairContext(
                    context_id=(
                        f"attachment-repair-context://sha256/{_sha256_text(request.repair_request_sha256)}"
                    ),
                    request=request,
                    material=material,
                )
            )
        except Exception:
            decision = ProviderAttachmentRepairDecision(
                outcome=AttachmentRepairOutcomeV2.RETRYABLE_FAILURE,
                failure_code=(AttachmentRepairFailureCodeV2.REPAIR_BACKEND_FAILED),
            )
        if decision.outcome is AttachmentRepairOutcomeV2.SUCCEEDED:
            if decision.output_bytes is None or decision.worker_version is None:
                result = self._repair_failure(
                    request,
                    AttachmentRepairOutcomeV2.BLOCKED_POLICY,
                    AttachmentRepairFailureCodeV2.OUTPUT_INVALID,
                )
            else:
                digest = hashlib.sha256(decision.output_bytes).hexdigest()
                if digest == request.output_sha256:
                    result = self._repair_failure(
                        request,
                        AttachmentRepairOutcomeV2.BLOCKED_POLICY,
                        AttachmentRepairFailureCodeV2.OUTPUT_UNCHANGED,
                    )
                elif self._repaired_output_registrar is None:
                    result = self._repair_failure(
                        request,
                        AttachmentRepairOutcomeV2.BLOCKED_CAPABILITY,
                        AttachmentRepairFailureCodeV2.VALIDATION_REGISTRATION_UNAVAILABLE,
                    )
                else:
                    output_ref = _repaired_output_ref(digest)
                    try:
                        self._repaired_output_registrar.register_repaired_output(
                            request=request,
                            material=material,
                            output_ref=output_ref,
                            output_bytes=decision.output_bytes,
                        )
                    except Exception:
                        result = self._repair_failure(
                            request,
                            AttachmentRepairOutcomeV2.BLOCKED_POLICY,
                            AttachmentRepairFailureCodeV2.VALIDATION_REGISTRATION_FAILED,
                        )
                    else:
                        result = self._repair_success(
                            request,
                            output_ref=output_ref,
                            worker_version=decision.worker_version,
                        )
        else:
            result = self._repair_failure(
                request,
                decision.outcome,
                decision.failure_code or AttachmentRepairFailureCodeV2.REPAIR_BACKEND_FAILED,
            )
        return self._cache_repair(request, result)

    def _finding(
        self,
        request: AttachmentSemanticReviewRequestV2,
        decision: ProviderSemanticFindingDecision,
    ) -> AttachmentSemanticReviewFindingV2:
        policy = attachment_semantic_finding_policy(decision.code)
        value = AttachmentSemanticReviewFindingV2(
            finding_id="attachment-semantic-review-finding://pending",
            semantic_review_request_ref=attachment_semantic_review_request_ref(request),
            round=request.round,
            scope=policy.scope,
            code=decision.code,
            candidate_revision_ref=request.candidate_revision_ref,
            subject_refs=_sorted_refs(decision.subject_refs),
            artifact_ids=tuple(sorted(decision.artifact_ids)),
            evidence_ref_ids=tuple(sorted(decision.evidence_ref_ids)),
            predecessor_finding_ref=decision.predecessor_finding_ref,
            severity=policy.severity,
            non_waivable=policy.non_waivable,
            artifact_repair_allowed=policy.artifact_repair_allowed,
            finding_sha256="0" * 64,
        )
        digest = attachment_semantic_review_finding_carried_sha256(value)
        return value.model_copy(
            update={
                "finding_id": (f"attachment-semantic-review-finding://sha256/{digest}"),
                "finding_sha256": digest,
            }
        )

    def _review_result(
        self,
        request: AttachmentSemanticReviewRequestV2,
        *,
        outcome: AttachmentSemanticReviewOutcomeV2,
        attestation: SemanticCleanContextAttestationV2,
        findings: tuple[AttachmentSemanticReviewFindingV2, ...],
        resolutions: tuple[AttachmentSemanticFindingResolutionV2, ...],
        failure_code: AttachmentSemanticReviewFailureCodeV2 | None,
    ) -> AttachmentSemanticReviewResultV2:
        value = AttachmentSemanticReviewResultV2(
            semantic_review_result_id="attachment-semantic-review-result://pending",
            semantic_review_request_ref=attachment_semantic_review_request_ref(request),
            candidate_revision_ref=request.candidate_revision_ref,
            deterministic_validation_result_ref=(request.deterministic_validation_result_ref),
            round=request.round,
            reviewer_role=request.reviewer_role,
            outcome=outcome,
            clean_context_attestation=attestation,
            findings=findings,
            resolutions=resolutions,
            failure_code=failure_code,
            policy_version=ATTACHMENT_SEMANTIC_REVIEW_POLICY_VERSION,
            semantic_review_result_sha256="0" * 64,
        )
        digest = attachment_semantic_review_result_carried_sha256(value)
        return value.model_copy(
            update={
                "semantic_review_result_id": (f"attachment-semantic-review-result://sha256/{digest}"),
                "semantic_review_result_sha256": digest,
            }
        )

    def _resolution(
        self,
        request: AttachmentSemanticReviewRequestV2,
        decision: ProviderSemanticResolutionDecision,
    ) -> AttachmentSemanticFindingResolutionV2:
        value = AttachmentSemanticFindingResolutionV2(
            resolution_id="attachment-semantic-finding-resolution://pending",
            semantic_review_request_ref=attachment_semantic_review_request_ref(request),
            prior_finding_ref=decision.prior_finding_ref,
            old_subject_refs=_sorted_refs(decision.old_subject_refs),
            current_subject_refs=_sorted_refs(decision.current_subject_refs),
            repair_plan_ref=decision.repair_plan_ref,
            repair_result_refs=_sorted_refs(decision.repair_result_refs),
            deterministic_revalidation_ref=(request.deterministic_validation_result_ref),
            disposition=decision.disposition,
            successor_finding_ref=decision.successor_finding_ref,
            evidence_ref_ids=tuple(sorted(decision.evidence_ref_ids)),
            resolution_sha256="0" * 64,
        )
        digest = attachment_semantic_finding_resolution_carried_sha256(value)
        return value.model_copy(
            update={
                "resolution_id": (f"attachment-semantic-finding-resolution://sha256/{digest}"),
                "resolution_sha256": digest,
            }
        )

    def _blocked_review(
        self,
        request: AttachmentSemanticReviewRequestV2,
        code: AttachmentSemanticReviewFailureCodeV2,
    ) -> AttachmentSemanticReviewResultV2:
        attestation = SemanticCleanContextAttestationV2.create(
            semantic_review_request_ref=attachment_semantic_review_request_ref(request),
            stage_run_ref=request.stage_run_ref,
            round=request.round,
            reviewer_role=request.reviewer_role,
            context_view_ref=request.context_view_ref,
            included_ref_inventory=(),
        )
        return self._review_result(
            request,
            outcome=AttachmentSemanticReviewOutcomeV2.BLOCKED,
            attestation=attestation,
            findings=(),
            resolutions=(),
            failure_code=code,
        )

    def _repair_success(
        self,
        request: AttachmentRepairRequestV2,
        *,
        output_ref: FacadeObjectRef,
        worker_version: str,
    ) -> AttachmentRepairResultV2:
        value = AttachmentRepairResultV2(
            repair_result_id="attachment-repair-result://pending",
            repair_request_ref=attachment_repair_request_ref(request),
            candidate_revision_ref=request.candidate_revision_ref,
            artifact_id=request.artifact_id,
            source_artifact_version_ref=request.artifact_version_ref,
            source_output_ref=request.output_ref,
            targeted_finding_refs=request.targeted_finding_refs,
            attempt=request.attempt,
            outcome=AttachmentRepairOutcomeV2.SUCCEEDED,
            worker_version=worker_version,
            output_ref=output_ref,
            output_sha256=output_ref.object_sha256,
            retryable=False,
            failure_code=None,
            policy_version=ATTACHMENT_REPAIR_POLICY_VERSION,
            repair_result_sha256="0" * 64,
        )
        return _finalize_repair(value)

    def _repair_failure(
        self,
        request: AttachmentRepairRequestV2,
        outcome: AttachmentRepairOutcomeV2,
        code: AttachmentRepairFailureCodeV2,
    ) -> AttachmentRepairResultV2:
        value = AttachmentRepairResultV2(
            repair_result_id="attachment-repair-result://pending",
            repair_request_ref=attachment_repair_request_ref(request),
            candidate_revision_ref=request.candidate_revision_ref,
            artifact_id=request.artifact_id,
            source_artifact_version_ref=request.artifact_version_ref,
            source_output_ref=request.output_ref,
            targeted_finding_refs=request.targeted_finding_refs,
            attempt=request.attempt,
            outcome=outcome,
            worker_version=None,
            output_ref=None,
            output_sha256=None,
            retryable=outcome is AttachmentRepairOutcomeV2.RETRYABLE_FAILURE,
            failure_code=code,
            policy_version=ATTACHMENT_REPAIR_POLICY_VERSION,
            repair_result_sha256="0" * 64,
        )
        return _finalize_repair(value)

    def _cache_review(
        self,
        request: AttachmentSemanticReviewRequestV2,
        result: AttachmentSemanticReviewResultV2,
    ) -> AttachmentSemanticReviewResultV2:
        self._review_cache[request.idempotency_key] = (
            request.semantic_review_request_sha256,
            result,
        )
        return result

    def _cache_repair(
        self,
        request: AttachmentRepairRequestV2,
        result: AttachmentRepairResultV2,
    ) -> AttachmentRepairResultV2:
        self._repair_cache[request.idempotency_key] = (
            request.repair_request_sha256,
            result,
        )
        return result


def _finalize_repair(value: AttachmentRepairResultV2) -> AttachmentRepairResultV2:
    digest = attachment_repair_result_carried_sha256(value)
    return value.model_copy(
        update={
            "repair_result_id": f"attachment-repair-result://sha256/{digest}",
            "repair_result_sha256": digest,
        }
    )


def _validate_resolution_decisions(
    request: AttachmentSemanticReviewRequestV2,
    *,
    findings: tuple[AttachmentSemanticReviewFindingV2, ...],
    resolutions: tuple[AttachmentSemanticFindingResolutionV2, ...],
) -> None:
    expected_prior_refs = set(request.required_resolution_finding_refs)
    observed_prior_refs = {item.prior_finding_ref for item in resolutions}
    if request.prior_repair_plan_refs:
        if observed_prior_refs != expected_prior_refs:
            raise ValueError("semantic review must resolve every repaired prior finding")
    elif resolutions:
        raise ValueError("semantic review cannot resolve findings without repair lineage")
    current_subject_universe = {
        request.candidate_revision_ref,
        *request.current_artifact_version_refs,
        *request.current_output_refs,
    }
    current_finding_refs = {attachment_semantic_review_finding_ref(item) for item in findings}
    for resolution in resolutions:
        if (
            resolution.semantic_review_request_ref != attachment_semantic_review_request_ref(request)
            or resolution.repair_plan_ref not in request.prior_repair_plan_refs
            or not set(resolution.repair_result_refs) <= set(request.prior_repair_result_refs)
            or resolution.deterministic_revalidation_ref != request.deterministic_validation_result_ref
            or not set(resolution.current_subject_refs) <= current_subject_universe
            or request.candidate_revision_ref not in resolution.current_subject_refs
        ):
            raise ValueError("semantic resolution does not match current repair lineage")
        if (
            resolution.successor_finding_ref is not None
            and resolution.successor_finding_ref not in current_finding_refs
        ):
            raise ValueError("semantic resolution successor is not a current finding")
        attachment_semantic_finding_resolution_ref(resolution)


def _validate_decision_evidence(
    decision: ProviderSemanticReviewDecision,
    material: ProviderSemanticReviewMaterial,
) -> None:
    observed = {
        *(evidence_id for finding in decision.findings for evidence_id in finding.evidence_ref_ids),
        *(evidence_id for resolution in decision.resolutions for evidence_id in resolution.evidence_ref_ids),
    }
    if not observed <= material.allowed_evidence_ref_ids:
        raise ValueError("semantic review returned unauthorized evidence IDs")


def _review_material_matches(
    request: AttachmentSemanticReviewRequestV2,
    material: ProviderSemanticReviewMaterial,
) -> bool:
    required_refs = {
        request.candidate_revision_ref,
        request.deterministic_validation_result_ref,
        request.context_view_ref,
        *request.current_artifact_version_refs,
        *request.current_output_refs,
        *request.prior_finding_refs,
        *request.prior_resolution_refs,
        *request.prior_repair_plan_refs,
        *request.prior_repair_result_refs,
    }
    expected_denied = (
        "BUILD_TRANSCRIPT",
        "HIDDEN_REASONING",
        "RAW_PRIVATE_REFERENCE",
        "RAW_TRACE",
    )
    return (
        material.context_view_ref == request.context_view_ref
        and len(material.included_ref_inventory) == len(set(material.included_ref_inventory))
        and required_refs <= set(material.included_ref_inventory)
        and tuple(sorted(material.denied_data_families)) == expected_denied
    )


def _repaired_output_ref(digest: str) -> FacadeObjectRef:
    return FacadeObjectRef(
        object_type="attachment-output",
        object_id=f"attachment-output://sha256/{digest}",
        object_version="v2",
        object_sha256=digest,
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _ref_key(ref: FacadeObjectRef) -> tuple[str, str, str, str]:
    return (
        ref.object_type,
        ref.object_id,
        ref.object_version,
        ref.object_sha256,
    )


def _sorted_refs(
    refs: tuple[FacadeObjectRef, ...],
) -> tuple[FacadeObjectRef, ...]:
    unique = {_ref_key(ref): ref for ref in refs}
    return tuple(unique[key] for key in sorted(unique))


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
