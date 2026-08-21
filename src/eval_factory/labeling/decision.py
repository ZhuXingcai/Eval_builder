from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Literal

from pydantic import Field, field_validator, model_validator

from eval_factory.contracts.core import (
    ContractAudit,
    ContractModel,
    EvidenceRef,
    Identifier,
    ObjectRef,
    Sha256,
)
from eval_factory.contracts.labeling_v2 import (
    LabelDecisionV2,
    LabelDecisionValueV2,
    LabelExecutionStatus,
    LabelSpecV2,
    LabelUnresolvedReason,
)
from eval_factory.labeling.semantic import SemanticResidualOutcome, SemanticResidualResult
from eval_factory.labeling.structured import StructuredLabelResult

LABEL_DECISION_MERGE_POLICY_VERSION: Literal["label-decision-merge/r3-04-v1"] = (
    "label-decision-merge/r3-04-v1"
)


class LabelDecisionMergePolicyError(RuntimeError):
    pass


class LabelDecisionRoute(StrEnum):
    FINAL = "FINAL"
    TYPED_UNRESOLVED_QUEUE = "TYPED_UNRESOLVED_QUEUE"
    USER_INSPECTION_QUEUE = "USER_INSPECTION_QUEUE"


_DENIED_DECISION_OBJECT_TYPES = frozenset(
    {
        "answer-bearing",
        "final-answer",
        "final-output",
        "grader-rule",
        "hidden-pass-condition",
        "hidden-selection-signal",
        "private-reference",
        "quarantine",
        "quarantine-subject",
        "raw-trace",
        "raw-traj",
        "trace-raw",
    }
)


class LabelDecisionRoutingPolicy(ContractModel):
    schema_version: Literal["eval-factory/label-decision-routing-policy/r3-04"] = (
        "eval-factory/label-decision-routing-policy/r3-04"
    )
    policy_id: Identifier = "label-decision-routing-policy://default"
    route_low_confidence_to_user_inspection: bool = False
    route_conflicts_to_user_inspection: bool = False
    policy_version: Literal["label-decision-merge/r3-04-v1"] = LABEL_DECISION_MERGE_POLICY_VERSION


class LabelDecisionMergeRequest(ContractModel):
    schema_version: Literal["eval-factory/label-decision-merge-request/r3-04"] = (
        "eval-factory/label-decision-merge-request/r3-04"
    )
    label_spec: LabelSpecV2
    structured_result: StructuredLabelResult
    semantic_result: SemanticResidualResult | None = None
    routing_policy: LabelDecisionRoutingPolicy = Field(default_factory=LabelDecisionRoutingPolicy)
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_merge_request(self) -> LabelDecisionMergeRequest:
        if self.structured_result.trace_envelope_ref.object_type != "trace-envelope":
            raise ValueError("structured_result trace_envelope_ref must reference trace-envelope")
        return self


class LabelDecisionMergeResult(ContractModel):
    schema_version: Literal["eval-factory/label-decision-merge-result/r3-04"] = (
        "eval-factory/label-decision-merge-result/r3-04"
    )
    label_decision_merge_result_id: Identifier
    label_decision: LabelDecisionV2
    route: LabelDecisionRoute
    routing_reasons: frozenset[LabelUnresolvedReason] = frozenset()
    policy_version: Literal["label-decision-merge/r3-04-v1"] = LABEL_DECISION_MERGE_POLICY_VERSION
    result_sha256: Sha256
    audit: ContractAudit

    @field_validator("route", mode="before")
    @classmethod
    def parse_route(cls, value: object) -> LabelDecisionRoute:
        if isinstance(value, LabelDecisionRoute):
            return value
        if isinstance(value, str):
            return LabelDecisionRoute(value)
        raise TypeError("route must be a LabelDecisionRoute")

    @field_validator("routing_reasons", mode="before")
    @classmethod
    def parse_reasons(cls, value: object) -> frozenset[LabelUnresolvedReason]:
        if isinstance(value, frozenset):
            return frozenset(
                item if isinstance(item, LabelUnresolvedReason) else LabelUnresolvedReason(item)
                for item in value
            )
        if isinstance(value, (list, tuple, set)):
            return frozenset(LabelUnresolvedReason(item) for item in value)
        raise TypeError("routing_reasons must be a collection")

    @model_validator(mode="after")
    def validate_route(self) -> LabelDecisionMergeResult:
        if self.route is LabelDecisionRoute.FINAL:
            if self.label_decision.execution_status is not LabelExecutionStatus.FINAL:
                raise ValueError("FINAL route requires FINAL label execution status")
            if self.routing_reasons:
                raise ValueError("FINAL route cannot carry routing reasons")
        if (
            self.route is LabelDecisionRoute.USER_INSPECTION_QUEUE
            and LabelUnresolvedReason.USER_INSPECTION_REQUIRED not in self.routing_reasons
        ):
            raise ValueError("user-inspection route requires USER_INSPECTION_REQUIRED reason")
        return self


class LabelDecisionMerger:
    policy_version = LABEL_DECISION_MERGE_POLICY_VERSION

    def merge(self, request: LabelDecisionMergeRequest) -> LabelDecisionMergeResult:
        _validate_merge_request(request)
        if request.structured_result.unresolved_reasons:
            return self._abstain(
                request,
                reasons=request.structured_result.unresolved_reasons,
                semantic_evidence=_semantic_evidence(request.semantic_result),
            )
        if request.structured_result.semantic_evaluation_required:
            return self._merge_semantic(request)
        return self._merge_structured_only(request)

    def _merge_structured_only(self, request: LabelDecisionMergeRequest) -> LabelDecisionMergeResult:
        structured = request.structured_result
        has_positive = bool(structured.positive_evidence)
        has_negative = bool(structured.negative_evidence)
        if has_positive and has_negative:
            return self._abstain(
                request,
                reasons=frozenset({LabelUnresolvedReason.CONFLICTING_EVIDENCE}),
            )
        if has_positive and structured.structured_capability_complete:
            return self._final_decision(
                request,
                decision=LabelDecisionValueV2.MATCH,
                confidence=1.0,
                positive_evidence=structured.positive_evidence,
                negative_evidence=(),
                semantic_evidence=(),
            )
        if has_negative and _negative_evidence_complete(structured.negative_evidence):
            return self._final_decision(
                request,
                decision=LabelDecisionValueV2.NO_MATCH,
                confidence=1.0,
                positive_evidence=(),
                negative_evidence=structured.negative_evidence,
                semantic_evidence=(),
            )
        if not structured.structured_capability_complete:
            return self._abstain(
                request,
                reasons=frozenset({LabelUnresolvedReason.INCOMPLETE_STRUCTURED_CAPABILITY}),
            )
        return self._abstain(
            request,
            reasons=frozenset({LabelUnresolvedReason.MISSING_EVIDENCE}),
        )

    def _merge_semantic(self, request: LabelDecisionMergeRequest) -> LabelDecisionMergeResult:
        semantic = request.semantic_result
        if semantic is None:
            return self._abstain(
                request,
                reasons=frozenset({LabelUnresolvedReason.MISSING_EVIDENCE}),
            )
        if semantic.outcome in {SemanticResidualOutcome.ABSTAIN, SemanticResidualOutcome.UNRESOLVED}:
            return self._abstain(
                request,
                reasons=semantic.unresolved_reasons,
                semantic_evidence=semantic.semantic_evidence,
            )
        if semantic.confidence < request.label_spec.review_threshold:
            return self._abstain(
                request,
                reasons=frozenset({LabelUnresolvedReason.LOW_CONFIDENCE}),
                semantic_evidence=semantic.semantic_evidence,
            )
        if semantic.confidence < request.label_spec.decision_threshold:
            return self._abstain(
                request,
                reasons=frozenset({LabelUnresolvedReason.LOW_CONFIDENCE}),
                semantic_evidence=semantic.semantic_evidence,
            )
        conflict = _semantic_conflicts_with_structured(semantic, request.structured_result)
        if conflict:
            return self._abstain(
                request,
                reasons=frozenset({LabelUnresolvedReason.CONFLICTING_EVIDENCE}),
                semantic_evidence=semantic.semantic_evidence,
            )
        if semantic.outcome is SemanticResidualOutcome.MATCH:
            return self._final_decision(
                request,
                decision=LabelDecisionValueV2.MATCH,
                confidence=semantic.confidence,
                positive_evidence=request.structured_result.positive_evidence,
                negative_evidence=(),
                semantic_evidence=semantic.semantic_evidence,
            )
        if semantic.outcome is SemanticResidualOutcome.NO_MATCH:
            if request.structured_result.structured_capability_complete and _negative_evidence_complete(
                request.structured_result.negative_evidence
            ):
                return self._final_decision(
                    request,
                    decision=LabelDecisionValueV2.NO_MATCH,
                    confidence=semantic.confidence,
                    positive_evidence=(),
                    negative_evidence=request.structured_result.negative_evidence,
                    semantic_evidence=semantic.semantic_evidence,
                )
            return self._abstain(
                request,
                reasons=frozenset({LabelUnresolvedReason.MISSING_EVIDENCE}),
                semantic_evidence=semantic.semantic_evidence,
            )
        raise LabelDecisionMergePolicyError("unsupported semantic residual outcome")

    def _final_decision(
        self,
        request: LabelDecisionMergeRequest,
        *,
        decision: LabelDecisionValueV2,
        confidence: float,
        positive_evidence: tuple[EvidenceRef, ...],
        negative_evidence: tuple[EvidenceRef, ...],
        semantic_evidence: tuple[EvidenceRef, ...],
    ) -> LabelDecisionMergeResult:
        label_decision = _label_decision(
            request=request,
            decision=decision,
            execution_status=LabelExecutionStatus.FINAL,
            confidence=confidence,
            positive_evidence=positive_evidence,
            negative_evidence=negative_evidence,
            semantic_evidence=semantic_evidence,
            unresolved_reasons=frozenset(),
        )
        return _merge_result(
            label_decision=label_decision,
            route=LabelDecisionRoute.FINAL,
            routing_reasons=frozenset(),
            audit=request.audit,
        )

    def _abstain(
        self,
        request: LabelDecisionMergeRequest,
        *,
        reasons: frozenset[LabelUnresolvedReason],
        semantic_evidence: tuple[EvidenceRef, ...] = (),
    ) -> LabelDecisionMergeResult:
        if not reasons:
            reasons = frozenset({LabelUnresolvedReason.MISSING_EVIDENCE})
        route, execution_status, routed_reasons = _route_for_reasons(reasons, request.routing_policy)
        label_decision = _label_decision(
            request=request,
            decision=LabelDecisionValueV2.ABSTAIN,
            execution_status=execution_status,
            confidence=_abstain_confidence(request.semantic_result),
            positive_evidence=(),
            negative_evidence=(),
            semantic_evidence=semantic_evidence,
            unresolved_reasons=routed_reasons,
        )
        return _merge_result(
            label_decision=label_decision,
            route=route,
            routing_reasons=routed_reasons,
            audit=request.audit,
        )


def _validate_merge_request(request: LabelDecisionMergeRequest) -> None:
    expected_label_ref = _label_spec_ref(request.label_spec)
    try:
        _validate_label_spec_identity(request.structured_result.label_spec_ref, expected_label_ref)
        if not request.structured_result.semantic_evaluation_required and request.semantic_result is not None:
            raise LabelDecisionMergePolicyError(
                "semantic result supplied when semantic evaluation is not required"
            )
        if request.semantic_result is not None:
            _validate_label_spec_identity(request.semantic_result.label_spec_ref, expected_label_ref)
            if request.semantic_result.trace_envelope_ref != request.structured_result.trace_envelope_ref:
                raise LabelDecisionMergePolicyError("semantic and structured trace envelope refs must match")
        for evidence in _request_evidence(request):
            _validate_safe_ref(evidence.subject_ref)
    except ValueError as exc:
        raise LabelDecisionMergePolicyError(str(exc)) from exc


def _label_decision(
    *,
    request: LabelDecisionMergeRequest,
    decision: LabelDecisionValueV2,
    execution_status: LabelExecutionStatus,
    confidence: float,
    positive_evidence: tuple[EvidenceRef, ...],
    negative_evidence: tuple[EvidenceRef, ...],
    semantic_evidence: tuple[EvidenceRef, ...],
    unresolved_reasons: frozenset[LabelUnresolvedReason],
) -> LabelDecisionV2:
    semantic = request.semantic_result
    model_profile = semantic.model_profile if semantic_evidence and semantic is not None else None
    prompt_version = semantic.prompt_version if semantic_evidence and semantic is not None else None
    seed = _decision_seed(
        request=request,
        decision=decision,
        execution_status=execution_status,
        confidence=confidence,
        positive_evidence=positive_evidence,
        negative_evidence=negative_evidence,
        semantic_evidence=semantic_evidence,
        unresolved_reasons=unresolved_reasons,
        model_profile=model_profile,
        prompt_version=prompt_version,
    )
    return LabelDecisionV2(
        label_decision_id=_stable_id("label-decision", seed),
        label_spec_ref=_label_spec_ref(request.label_spec),
        trace_envelope_ref=request.structured_result.trace_envelope_ref,
        decision=decision,
        execution_status=execution_status,
        positive_evidence=_sort_evidence(positive_evidence),
        negative_evidence=_sort_evidence(negative_evidence),
        semantic_evidence=_sort_evidence(semantic_evidence),
        structured_capability_complete=request.structured_result.structured_capability_complete,
        confidence=confidence,
        rule_version=request.structured_result.policy_version,
        model_profile=model_profile,
        prompt_version=prompt_version,
        unresolved_reasons=unresolved_reasons,
        policy_version=LABEL_DECISION_MERGE_POLICY_VERSION,
        decision_sha256=_stable_hash(seed),
        audit=request.audit,
    )


def _merge_result(
    *,
    label_decision: LabelDecisionV2,
    route: LabelDecisionRoute,
    routing_reasons: frozenset[LabelUnresolvedReason],
    audit: ContractAudit,
) -> LabelDecisionMergeResult:
    seed = {
        "label_decision": _decision_payload(label_decision),
        "route": route.value,
        "routing_reasons": sorted(reason.value for reason in routing_reasons),
        "policy_version": LABEL_DECISION_MERGE_POLICY_VERSION,
    }
    return LabelDecisionMergeResult(
        label_decision_merge_result_id=_stable_id("label-decision-merge-result", seed),
        label_decision=label_decision,
        route=route,
        routing_reasons=routing_reasons,
        result_sha256=_stable_hash(seed),
        audit=audit,
    )


def _route_for_reasons(
    reasons: frozenset[LabelUnresolvedReason],
    policy: LabelDecisionRoutingPolicy,
) -> tuple[LabelDecisionRoute, LabelExecutionStatus, frozenset[LabelUnresolvedReason]]:
    if (
        LabelUnresolvedReason.LOW_CONFIDENCE in reasons and policy.route_low_confidence_to_user_inspection
    ) or (
        LabelUnresolvedReason.CONFLICTING_EVIDENCE in reasons and policy.route_conflicts_to_user_inspection
    ):
        routed = frozenset((*reasons, LabelUnresolvedReason.USER_INSPECTION_REQUIRED))
        return LabelDecisionRoute.USER_INSPECTION_QUEUE, LabelExecutionStatus.REVIEW_REQUIRED, routed
    return LabelDecisionRoute.TYPED_UNRESOLVED_QUEUE, LabelExecutionStatus.UNRESOLVED, reasons


def _abstain_confidence(semantic: SemanticResidualResult | None) -> float:
    return semantic.confidence if semantic is not None else 0.0


def _semantic_evidence(semantic: SemanticResidualResult | None) -> tuple[EvidenceRef, ...]:
    return () if semantic is None else semantic.semantic_evidence


def _semantic_conflicts_with_structured(
    semantic: SemanticResidualResult,
    structured: StructuredLabelResult,
) -> bool:
    if semantic.outcome is SemanticResidualOutcome.MATCH and structured.negative_evidence:
        return True
    return bool(semantic.outcome is SemanticResidualOutcome.NO_MATCH and structured.positive_evidence)


def _negative_evidence_complete(evidence: tuple[EvidenceRef, ...]) -> bool:
    return bool(evidence) and all(item.capability_complete for item in evidence)


def _request_evidence(request: LabelDecisionMergeRequest) -> tuple[EvidenceRef, ...]:
    semantic = () if request.semantic_result is None else request.semantic_result.semantic_evidence
    return (
        *request.structured_result.positive_evidence,
        *request.structured_result.negative_evidence,
        *semantic,
    )


def _validate_label_spec_identity(observed: ObjectRef, expected: ObjectRef) -> None:
    if observed.object_type != "label-spec":
        raise ValueError("label spec reference must reference label-spec")
    if observed.object_id != expected.object_id or observed.object_version != expected.object_version:
        raise ValueError("label spec reference mismatch")


def _validate_safe_ref(ref: ObjectRef) -> None:
    if ref.object_type in _DENIED_DECISION_OBJECT_TYPES:
        raise ValueError("unsafe label decision evidence reference")


def _label_spec_ref(label_spec: LabelSpecV2) -> ObjectRef:
    return ObjectRef(
        object_type="label-spec",
        object_id=label_spec.label_spec_id,
        object_version=label_spec.label_version,
        object_sha256=label_spec.label_spec_sha256,
    )


def _structured_result_ref(result: StructuredLabelResult) -> ObjectRef:
    return ObjectRef(
        object_type="structured-label-result",
        object_id=result.structured_label_result_id,
        object_version=result.policy_version,
        object_sha256=result.result_sha256,
    )


def _semantic_result_ref(result: SemanticResidualResult | None) -> ObjectRef | None:
    if result is None:
        return None
    return ObjectRef(
        object_type="semantic-residual-result",
        object_id=result.semantic_residual_result_id,
        object_version=result.policy_version,
        object_sha256=result.result_sha256,
    )


def _decision_seed(
    *,
    request: LabelDecisionMergeRequest,
    decision: LabelDecisionValueV2,
    execution_status: LabelExecutionStatus,
    confidence: float,
    positive_evidence: tuple[EvidenceRef, ...],
    negative_evidence: tuple[EvidenceRef, ...],
    semantic_evidence: tuple[EvidenceRef, ...],
    unresolved_reasons: frozenset[LabelUnresolvedReason],
    model_profile: str | None,
    prompt_version: str | None,
) -> dict[str, object]:
    semantic_ref = _semantic_result_ref(request.semantic_result)
    return {
        "label_spec_ref": _label_spec_ref(request.label_spec).model_dump(mode="json", exclude_none=False),
        "structured_label_result_ref": _structured_result_ref(request.structured_result).model_dump(
            mode="json", exclude_none=False
        ),
        "semantic_residual_result_ref": (
            None if semantic_ref is None else semantic_ref.model_dump(mode="json", exclude_none=False)
        ),
        "trace_envelope_ref": request.structured_result.trace_envelope_ref.model_dump(
            mode="json", exclude_none=False
        ),
        "decision": decision.value,
        "execution_status": execution_status.value,
        "positive_evidence": [
            item.model_dump(mode="json", exclude_none=False) for item in _sort_evidence(positive_evidence)
        ],
        "negative_evidence": [
            item.model_dump(mode="json", exclude_none=False) for item in _sort_evidence(negative_evidence)
        ],
        "semantic_evidence": [
            item.model_dump(mode="json", exclude_none=False) for item in _sort_evidence(semantic_evidence)
        ],
        "structured_capability_complete": request.structured_result.structured_capability_complete,
        "confidence": confidence,
        "rule_version": request.structured_result.policy_version,
        "model_profile": model_profile,
        "prompt_version": prompt_version,
        "unresolved_reasons": sorted(reason.value for reason in unresolved_reasons),
        "policy_version": LABEL_DECISION_MERGE_POLICY_VERSION,
    }


def _decision_payload(decision: LabelDecisionV2) -> dict[str, object]:
    return {
        "label_decision_id": decision.label_decision_id,
        "label_spec_ref": decision.label_spec_ref.model_dump(mode="json", exclude_none=False),
        "trace_envelope_ref": decision.trace_envelope_ref.model_dump(mode="json", exclude_none=False),
        "decision": decision.decision.value,
        "execution_status": decision.execution_status.value,
        "positive_evidence": [
            item.model_dump(mode="json", exclude_none=False)
            for item in _sort_evidence(decision.positive_evidence)
        ],
        "negative_evidence": [
            item.model_dump(mode="json", exclude_none=False)
            for item in _sort_evidence(decision.negative_evidence)
        ],
        "semantic_evidence": [
            item.model_dump(mode="json", exclude_none=False)
            for item in _sort_evidence(decision.semantic_evidence)
        ],
        "structured_capability_complete": decision.structured_capability_complete,
        "confidence": decision.confidence,
        "rule_version": decision.rule_version,
        "model_profile": decision.model_profile,
        "prompt_version": decision.prompt_version,
        "unresolved_reasons": sorted(reason.value for reason in decision.unresolved_reasons),
        "policy_version": decision.policy_version,
        "decision_sha256": decision.decision_sha256,
    }


def _sort_evidence(evidence: tuple[EvidenceRef, ...]) -> tuple[EvidenceRef, ...]:
    return tuple(sorted(evidence, key=lambda item: item.evidence_ref_id))


def _stable_id(kind: str, payload: object) -> str:
    return f"{kind}://sha256/{_stable_hash(payload)}"


def _stable_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
