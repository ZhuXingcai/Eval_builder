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
    LabelSpecV2,
    LabelUnresolvedReason,
)
from eval_factory.contracts.safety import EvidenceBundle
from eval_factory.labeling.structured import StructuredLabelResult

SEMANTIC_RESIDUAL_POLICY_VERSION: Literal["semantic-residual/r3-03-v1"] = "semantic-residual/r3-03-v1"


class SemanticResidualPolicyError(RuntimeError):
    pass


class SemanticResidualOutcome(StrEnum):
    MATCH = "MATCH"
    NO_MATCH = "NO_MATCH"
    ABSTAIN = "ABSTAIN"
    UNRESOLVED = "UNRESOLVED"


_DENIED_SEMANTIC_OBJECT_TYPES = frozenset(
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


class SemanticResidualRequest(ContractModel):
    schema_version: Literal["eval-factory/semantic-residual-request/r3-03"] = (
        "eval-factory/semantic-residual-request/r3-03"
    )
    semantic_residual_request_id: Identifier
    label_spec_ref: ObjectRef
    structured_label_result_ref: ObjectRef
    trace_envelope_ref: ObjectRef
    residual_id: Identifier
    question: str = Field(min_length=1, max_length=4000)
    evidence_bundle_ref: ObjectRef
    projection_policy_ref: ObjectRef
    evidence_refs: tuple[EvidenceRef, ...] = Field(min_length=1)
    allowed_evidence_types: tuple[Identifier, ...] = Field(min_length=1)
    abstain_conditions: tuple[str, ...] = Field(min_length=1)
    model_profile: Identifier
    prompt_version: str = Field(min_length=1, max_length=128)
    returned_characters: int = Field(ge=0)
    max_characters: int = Field(ge=0)
    policy_version: Literal["semantic-residual/r3-03-v1"] = SEMANTIC_RESIDUAL_POLICY_VERSION
    request_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_request_boundary(self) -> SemanticResidualRequest:
        if self.label_spec_ref.object_type != "label-spec":
            raise ValueError("label_spec_ref must reference label-spec")
        if self.structured_label_result_ref.object_type != "structured-label-result":
            raise ValueError("structured_label_result_ref must reference structured-label-result")
        if self.trace_envelope_ref.object_type != "trace-envelope":
            raise ValueError("trace_envelope_ref must reference trace-envelope")
        if self.evidence_bundle_ref.object_type != "evidence-bundle":
            raise ValueError("evidence_bundle_ref must reference evidence-bundle")
        if self.projection_policy_ref.object_type != "projection-policy":
            raise ValueError("projection_policy_ref must reference projection-policy")
        if self.returned_characters > self.max_characters:
            raise ValueError("returned characters exceed authorized budget")
        for ref in (
            self.label_spec_ref,
            self.structured_label_result_ref,
            self.trace_envelope_ref,
            self.evidence_bundle_ref,
            self.projection_policy_ref,
        ):
            _validate_safe_ref(ref)
        for evidence in self.evidence_refs:
            _validate_safe_ref(evidence.subject_ref)
        return self


class SemanticResidualResult(ContractModel):
    schema_version: Literal["eval-factory/semantic-residual-result/r3-03"] = (
        "eval-factory/semantic-residual-result/r3-03"
    )
    semantic_residual_result_id: Identifier
    semantic_residual_request_ref: ObjectRef
    label_spec_ref: ObjectRef
    trace_envelope_ref: ObjectRef
    outcome: SemanticResidualOutcome
    confidence: float = Field(ge=0, le=1)
    semantic_evidence: tuple[EvidenceRef, ...] = ()
    model_profile: Identifier
    prompt_version: str = Field(min_length=1, max_length=128)
    unresolved_reasons: frozenset[LabelUnresolvedReason] = frozenset()
    explanation_code: Identifier | None = None
    policy_version: Literal["semantic-residual/r3-03-v1"] = SEMANTIC_RESIDUAL_POLICY_VERSION
    result_sha256: Sha256
    audit: ContractAudit

    @field_validator("outcome", mode="before")
    @classmethod
    def parse_outcome(cls, value: object) -> SemanticResidualOutcome:
        if isinstance(value, SemanticResidualOutcome):
            return value
        if isinstance(value, str):
            return SemanticResidualOutcome(value)
        raise TypeError("outcome must be a SemanticResidualOutcome")

    @field_validator("unresolved_reasons", mode="before")
    @classmethod
    def parse_reasons(cls, value: object) -> frozenset[LabelUnresolvedReason]:
        if isinstance(value, frozenset):
            return frozenset(
                item if isinstance(item, LabelUnresolvedReason) else LabelUnresolvedReason(item)
                for item in value
            )
        if isinstance(value, (list, tuple, set)):
            return frozenset(LabelUnresolvedReason(item) for item in value)
        raise TypeError("unresolved_reasons must be a collection")

    @model_validator(mode="after")
    def validate_result_boundary(self) -> SemanticResidualResult:
        if self.semantic_residual_request_ref.object_type != "semantic-residual-request":
            raise ValueError("semantic_residual_request_ref must reference semantic-residual-request")
        if self.label_spec_ref.object_type != "label-spec":
            raise ValueError("label_spec_ref must reference label-spec")
        if self.trace_envelope_ref.object_type != "trace-envelope":
            raise ValueError("trace_envelope_ref must reference trace-envelope")
        if self.outcome in {SemanticResidualOutcome.MATCH, SemanticResidualOutcome.NO_MATCH}:
            if not self.semantic_evidence:
                raise ValueError("semantic match outcomes require semantic evidence")
            if self.unresolved_reasons:
                raise ValueError("semantic match outcomes cannot carry unresolved reasons")
        if (
            self.outcome
            in {
                SemanticResidualOutcome.ABSTAIN,
                SemanticResidualOutcome.UNRESOLVED,
            }
            and not self.unresolved_reasons
        ):
            raise ValueError("semantic abstain/unresolved outcomes require unresolved reasons")
        for evidence in self.semantic_evidence:
            _validate_safe_ref(evidence.subject_ref)
        return self


class FakeSemanticResidualFixture(ContractModel):
    schema_version: Literal["eval-factory/fake-semantic-residual-fixture/r3-03"] = (
        "eval-factory/fake-semantic-residual-fixture/r3-03"
    )
    fixture_id: Identifier
    outcome: SemanticResidualOutcome
    confidence: float = Field(ge=0, le=1)
    selected_evidence_ref_ids: tuple[Identifier, ...] = ()
    unresolved_reasons: frozenset[LabelUnresolvedReason] = frozenset()
    explanation_code: Identifier | None = None
    model_available: bool = True

    @field_validator("outcome", mode="before")
    @classmethod
    def parse_outcome(cls, value: object) -> SemanticResidualOutcome:
        if isinstance(value, SemanticResidualOutcome):
            return value
        if isinstance(value, str):
            return SemanticResidualOutcome(value)
        raise TypeError("outcome must be a SemanticResidualOutcome")

    @field_validator("unresolved_reasons", mode="before")
    @classmethod
    def parse_reasons(cls, value: object) -> frozenset[LabelUnresolvedReason]:
        if isinstance(value, frozenset):
            return frozenset(
                item if isinstance(item, LabelUnresolvedReason) else LabelUnresolvedReason(item)
                for item in value
            )
        if isinstance(value, (list, tuple, set)):
            return frozenset(LabelUnresolvedReason(item) for item in value)
        raise TypeError("unresolved_reasons must be a collection")


class SemanticResidualRequestBuilder:
    policy_version = SEMANTIC_RESIDUAL_POLICY_VERSION

    def build(
        self,
        *,
        label_spec: LabelSpecV2,
        structured_result: StructuredLabelResult,
        evidence_bundle: EvidenceBundle,
        audit: ContractAudit,
    ) -> SemanticResidualRequest:
        residual = label_spec.semantic_residual
        if residual is None:
            raise SemanticResidualPolicyError("label spec has no semantic residual")
        if not structured_result.prerequisites_satisfied:
            raise SemanticResidualPolicyError("structured prerequisites are not satisfied")
        if not structured_result.semantic_evaluation_required:
            raise SemanticResidualPolicyError("semantic residual evaluation is not required")
        if structured_result.trace_envelope_ref.object_id != evidence_bundle.trace_ir_version_id:
            raise SemanticResidualPolicyError("structured result and evidence bundle trace mismatch")
        if evidence_bundle.purpose != residual.evidence_bundle_purpose:
            raise SemanticResidualPolicyError("evidence bundle purpose does not match semantic residual")
        if evidence_bundle.projection_policy_ref.object_type != "projection-policy":
            raise SemanticResidualPolicyError("semantic residual requires a projection-policy ref")
        if evidence_bundle.returned_characters > evidence_bundle.max_characters:
            raise SemanticResidualPolicyError("evidence bundle exceeds semantic residual budget")
        if evidence_bundle.tainted_content_included is not False:
            raise SemanticResidualPolicyError("tainted evidence bundle cannot enter semantic residual")
        if not evidence_bundle.evidence:
            raise SemanticResidualPolicyError("semantic residual requires safe evidence")
        for evidence in evidence_bundle.evidence:
            try:
                _validate_safe_ref(evidence.subject_ref)
            except ValueError as exc:
                raise SemanticResidualPolicyError("unsafe semantic evidence reference") from exc

        evidence_refs = tuple(sorted(evidence_bundle.evidence, key=lambda item: item.evidence_ref_id))
        seed = _request_seed(
            label_spec=label_spec,
            structured_result=structured_result,
            evidence_bundle=evidence_bundle,
            evidence_refs=evidence_refs,
        )
        return SemanticResidualRequest(
            semantic_residual_request_id=_stable_id("semantic-residual-request", seed),
            label_spec_ref=_label_spec_ref(label_spec),
            structured_label_result_ref=_structured_result_ref(structured_result),
            trace_envelope_ref=structured_result.trace_envelope_ref,
            residual_id=residual.residual_id,
            question=residual.question,
            evidence_bundle_ref=_evidence_bundle_ref(evidence_bundle),
            projection_policy_ref=evidence_bundle.projection_policy_ref,
            evidence_refs=evidence_refs,
            allowed_evidence_types=residual.allowed_evidence_types,
            abstain_conditions=residual.abstain_conditions,
            model_profile=residual.model_profile,
            prompt_version=residual.prompt_version,
            returned_characters=evidence_bundle.returned_characters,
            max_characters=evidence_bundle.max_characters,
            request_sha256=_stable_hash(seed),
            audit=audit,
        )


class FakeSemanticResidualRunner:
    policy_version = SEMANTIC_RESIDUAL_POLICY_VERSION

    def run(
        self,
        request: SemanticResidualRequest,
        *,
        fixture: FakeSemanticResidualFixture,
        audit: ContractAudit,
    ) -> SemanticResidualResult:
        evidence_by_id = {item.evidence_ref_id: item for item in request.evidence_refs}
        selected = _selected_evidence(fixture, evidence_by_id)
        if not fixture.model_available:
            if fixture.outcome is not SemanticResidualOutcome.UNRESOLVED:
                raise SemanticResidualPolicyError("unavailable model must yield unresolved outcome")
            if LabelUnresolvedReason.MODEL_UNAVAILABLE not in fixture.unresolved_reasons:
                raise SemanticResidualPolicyError("unavailable model requires MODEL_UNAVAILABLE reason")
        if fixture.outcome in {SemanticResidualOutcome.MATCH, SemanticResidualOutcome.NO_MATCH}:
            if not fixture.model_available:
                raise SemanticResidualPolicyError("semantic match outcome requires available model")
            if not selected:
                raise SemanticResidualPolicyError("semantic match outcome requires selected evidence")
            if fixture.unresolved_reasons:
                raise SemanticResidualPolicyError("semantic match outcome cannot carry unresolved reason")
        if fixture.outcome is SemanticResidualOutcome.ABSTAIN and not fixture.unresolved_reasons:
            raise SemanticResidualPolicyError("semantic abstain requires unresolved reason")
        if fixture.outcome is SemanticResidualOutcome.UNRESOLVED:
            if not fixture.unresolved_reasons:
                raise SemanticResidualPolicyError("semantic unresolved requires unresolved reason")
            if selected:
                raise SemanticResidualPolicyError("semantic unresolved cannot carry selected evidence")
        semantic_evidence = () if fixture.outcome is SemanticResidualOutcome.UNRESOLVED else selected
        seed = _result_seed(
            request=request,
            fixture=fixture,
            semantic_evidence=semantic_evidence,
        )
        return SemanticResidualResult(
            semantic_residual_result_id=_stable_id("semantic-residual-result", seed),
            semantic_residual_request_ref=_request_ref(request),
            label_spec_ref=request.label_spec_ref,
            trace_envelope_ref=request.trace_envelope_ref,
            outcome=fixture.outcome,
            confidence=fixture.confidence,
            semantic_evidence=semantic_evidence,
            model_profile=request.model_profile,
            prompt_version=request.prompt_version,
            unresolved_reasons=fixture.unresolved_reasons,
            explanation_code=fixture.explanation_code,
            result_sha256=_stable_hash(seed),
            audit=audit,
        )


def _validate_safe_ref(ref: ObjectRef) -> None:
    if ref.object_type in _DENIED_SEMANTIC_OBJECT_TYPES:
        raise ValueError("unsafe semantic evidence reference")


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


def _evidence_bundle_ref(bundle: EvidenceBundle) -> ObjectRef:
    return ObjectRef(
        object_type="evidence-bundle",
        object_id=bundle.evidence_bundle_id,
        object_version="v1",
        object_sha256=bundle.bundle_sha256,
    )


def _request_ref(request: SemanticResidualRequest) -> ObjectRef:
    return ObjectRef(
        object_type="semantic-residual-request",
        object_id=request.semantic_residual_request_id,
        object_version=request.policy_version,
        object_sha256=request.request_sha256,
    )


def _selected_evidence(
    fixture: FakeSemanticResidualFixture,
    evidence_by_id: dict[str, EvidenceRef],
) -> tuple[EvidenceRef, ...]:
    selected: list[EvidenceRef] = []
    for evidence_ref_id in fixture.selected_evidence_ref_ids:
        evidence = evidence_by_id.get(evidence_ref_id)
        if evidence is None:
            raise SemanticResidualPolicyError("selected evidence is not present in request")
        selected.append(evidence)
    return tuple(sorted(selected, key=lambda item: item.evidence_ref_id))


def _request_seed(
    *,
    label_spec: LabelSpecV2,
    structured_result: StructuredLabelResult,
    evidence_bundle: EvidenceBundle,
    evidence_refs: tuple[EvidenceRef, ...],
) -> dict[str, object]:
    residual = label_spec.semantic_residual
    assert residual is not None
    return {
        "label_spec_ref": _label_spec_ref(label_spec).model_dump(mode="json", exclude_none=False),
        "structured_label_result_ref": _structured_result_ref(structured_result).model_dump(
            mode="json", exclude_none=False
        ),
        "trace_envelope_ref": structured_result.trace_envelope_ref.model_dump(
            mode="json", exclude_none=False
        ),
        "residual_id": residual.residual_id,
        "question": residual.question,
        "evidence_bundle_ref": _evidence_bundle_ref(evidence_bundle).model_dump(
            mode="json", exclude_none=False
        ),
        "projection_policy_ref": evidence_bundle.projection_policy_ref.model_dump(
            mode="json", exclude_none=False
        ),
        "evidence_refs": [item.model_dump(mode="json", exclude_none=False) for item in evidence_refs],
        "allowed_evidence_types": list(residual.allowed_evidence_types),
        "abstain_conditions": list(residual.abstain_conditions),
        "model_profile": residual.model_profile,
        "prompt_version": residual.prompt_version,
        "returned_characters": evidence_bundle.returned_characters,
        "max_characters": evidence_bundle.max_characters,
        "policy_version": SEMANTIC_RESIDUAL_POLICY_VERSION,
    }


def _result_seed(
    *,
    request: SemanticResidualRequest,
    fixture: FakeSemanticResidualFixture,
    semantic_evidence: tuple[EvidenceRef, ...],
) -> dict[str, object]:
    return {
        "semantic_residual_request_ref": _request_ref(request).model_dump(mode="json", exclude_none=False),
        "label_spec_ref": request.label_spec_ref.model_dump(mode="json", exclude_none=False),
        "trace_envelope_ref": request.trace_envelope_ref.model_dump(mode="json", exclude_none=False),
        "outcome": fixture.outcome.value,
        "confidence": fixture.confidence,
        "semantic_evidence": [item.model_dump(mode="json", exclude_none=False) for item in semantic_evidence],
        "model_profile": request.model_profile,
        "prompt_version": request.prompt_version,
        "unresolved_reasons": sorted(reason.value for reason in fixture.unresolved_reasons),
        "explanation_code": fixture.explanation_code,
        "model_available": fixture.model_available,
        "policy_version": SEMANTIC_RESIDUAL_POLICY_VERSION,
    }


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
