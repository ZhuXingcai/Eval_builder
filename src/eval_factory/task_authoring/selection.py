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
    SelectionContextV2,
    label_decision_ref,
    selection_context_ref,
)
from eval_factory.contracts.safety import EvidenceBundle
from eval_factory.labeling.decision import LABEL_DECISION_MERGE_POLICY_VERSION
from eval_factory.provenance.bundles import EVIDENCE_COMPILATION_POLICY_VERSION
from eval_factory.task_authoring.models import (
    TASK_EPISODE_CONSUMER_STAGE,
    TASK_EPISODE_EVIDENCE_PURPOSE,
    is_safe_task_authoring_ref,
)

SELECTION_CONTEXT_FIREWALL_POLICY_VERSION: Literal["selection-context-firewall/r4-02-v1"] = (
    "selection-context-firewall/r4-02-v1"
)


class SelectionContextFirewallPolicyError(RuntimeError):
    pass


class SelectionContextFirewallOutcome(StrEnum):
    ALLOWED = "ALLOWED"
    BLOCKED_DECISION = "BLOCKED_DECISION"
    BLOCKED_EVIDENCE = "BLOCKED_EVIDENCE"


class SelectionContextFirewallReason(StrEnum):
    NO_DECISIONS = "NO_DECISIONS"
    DECISION_NOT_MATCH = "DECISION_NOT_MATCH"
    DECISION_NOT_FINAL = "DECISION_NOT_FINAL"
    DECISION_UNRESOLVED = "DECISION_UNRESOLVED"
    MISSING_SAFE_EVIDENCE = "MISSING_SAFE_EVIDENCE"


class SelectionContextFirewallRequest(ContractModel):
    schema_version: Literal["eval-factory/selection-context-firewall-request/r4-02"] = (
        "eval-factory/selection-context-firewall-request/r4-02"
    )
    decisions: tuple[LabelDecisionV2, ...] = ()
    evidence_bundle: EvidenceBundle
    task_authoring_note_refs: tuple[ObjectRef, ...] = ()
    restricted_signal_refs: tuple[ObjectRef, ...] = ()
    audit: ContractAudit


class SelectionContextFirewallResult(ContractModel):
    schema_version: Literal["eval-factory/selection-context-firewall-result/r4-02"] = (
        "eval-factory/selection-context-firewall-result/r4-02"
    )
    selection_context_firewall_result_id: Identifier
    outcome: SelectionContextFirewallOutcome
    selection_context: SelectionContextV2 | None = None
    blocked_reasons: frozenset[SelectionContextFirewallReason] = frozenset()
    excluded_signal_count: int = Field(ge=0)
    input_sha256: Sha256
    policy_version: Literal["selection-context-firewall/r4-02-v1"] = SELECTION_CONTEXT_FIREWALL_POLICY_VERSION
    result_sha256: Sha256
    audit: ContractAudit

    @field_validator("outcome", mode="before")
    @classmethod
    def parse_outcome(cls, value: object) -> SelectionContextFirewallOutcome:
        if isinstance(value, SelectionContextFirewallOutcome):
            return value
        if isinstance(value, str):
            return SelectionContextFirewallOutcome(value)
        raise TypeError("outcome must be a SelectionContextFirewallOutcome")

    @field_validator("blocked_reasons", mode="before")
    @classmethod
    def parse_blocked_reasons(
        cls,
        value: object,
    ) -> frozenset[SelectionContextFirewallReason]:
        if isinstance(value, (frozenset, set, tuple, list)):
            return frozenset(
                item
                if isinstance(item, SelectionContextFirewallReason)
                else SelectionContextFirewallReason(item)
                for item in value
            )
        raise TypeError("blocked_reasons must be a collection")

    @model_validator(mode="after")
    def validate_result_shape(self) -> SelectionContextFirewallResult:
        if self.outcome is SelectionContextFirewallOutcome.ALLOWED:
            if self.selection_context is None:
                raise ValueError("ALLOWED outcome requires SelectionContext")
            if self.blocked_reasons:
                raise ValueError("ALLOWED outcome cannot carry blocked reasons")
            return self
        if self.selection_context is not None:
            raise ValueError("blocked firewall outcome cannot carry SelectionContext")
        if not self.blocked_reasons:
            raise ValueError("blocked firewall outcome requires reasons")
        decision_reasons = {
            SelectionContextFirewallReason.NO_DECISIONS,
            SelectionContextFirewallReason.DECISION_NOT_MATCH,
            SelectionContextFirewallReason.DECISION_NOT_FINAL,
            SelectionContextFirewallReason.DECISION_UNRESOLVED,
        }
        if (
            self.outcome is SelectionContextFirewallOutcome.BLOCKED_DECISION
            and not self.blocked_reasons.issubset(decision_reasons)
        ):
            raise ValueError("BLOCKED_DECISION requires decision reasons")
        if (
            self.outcome is SelectionContextFirewallOutcome.BLOCKED_EVIDENCE
            and self.blocked_reasons != frozenset({SelectionContextFirewallReason.MISSING_SAFE_EVIDENCE})
        ):
            raise ValueError("BLOCKED_EVIDENCE requires missing-safe-evidence reason")
        return self


class SelectionContextFirewall:
    policy_version = SELECTION_CONTEXT_FIREWALL_POLICY_VERSION

    def apply(
        self,
        request: SelectionContextFirewallRequest,
    ) -> SelectionContextFirewallResult:
        _validate_bundle(request.evidence_bundle)
        _validate_notes(
            request.task_authoring_note_refs,
            request.evidence_bundle,
        )
        _validate_decisions(request.decisions, request.evidence_bundle)

        input_sha256 = _input_hash(request)
        excluded_hashes = _excluded_signal_hashes(
            request.decisions,
            request.restricted_signal_refs,
        )
        audit = _sanitized_audit(request)
        blocked_reasons = _decision_blocked_reasons(request.decisions)
        if blocked_reasons:
            return _result(
                outcome=SelectionContextFirewallOutcome.BLOCKED_DECISION,
                selection_context=None,
                blocked_reasons=blocked_reasons,
                excluded_signal_count=len(excluded_hashes),
                input_sha256=input_sha256,
                audit=audit,
            )
        if not request.evidence_bundle.evidence:
            return _result(
                outcome=SelectionContextFirewallOutcome.BLOCKED_EVIDENCE,
                selection_context=None,
                blocked_reasons=frozenset({SelectionContextFirewallReason.MISSING_SAFE_EVIDENCE}),
                excluded_signal_count=len(excluded_hashes),
                input_sha256=input_sha256,
                audit=audit,
            )

        context = _selection_context(
            request=request,
            excluded_hashes=excluded_hashes,
            audit=audit,
        )
        return _result(
            outcome=SelectionContextFirewallOutcome.ALLOWED,
            selection_context=context,
            blocked_reasons=frozenset(),
            excluded_signal_count=len(excluded_hashes),
            input_sha256=input_sha256,
            audit=audit,
        )


def _validate_decisions(
    decisions: tuple[LabelDecisionV2, ...],
    bundle: EvidenceBundle,
) -> None:
    decision_ids: set[str] = set()
    label_identities: set[tuple[str, str]] = set()
    trace_ref: ObjectRef | None = None
    for decision in decisions:
        if decision.policy_version != LABEL_DECISION_MERGE_POLICY_VERSION:
            raise SelectionContextFirewallPolicyError("decision policy is not current")
        if decision.label_decision_id != (f"label-decision://sha256/{decision.decision_sha256}"):
            raise SelectionContextFirewallPolicyError(
                "decision identity is not an opaque carried-hash identity"
            )
        if decision.label_decision_id in decision_ids:
            raise SelectionContextFirewallPolicyError("duplicate decision identity")
        decision_ids.add(decision.label_decision_id)

        label_identity = (
            decision.label_spec_ref.object_id,
            decision.label_spec_ref.object_version,
        )
        if label_identity in label_identities:
            raise SelectionContextFirewallPolicyError("duplicate label-spec identity")
        label_identities.add(label_identity)

        if trace_ref is None:
            trace_ref = decision.trace_envelope_ref
        elif decision.trace_envelope_ref != trace_ref:
            raise SelectionContextFirewallPolicyError("decision trace envelopes do not match")
        if decision.trace_envelope_ref.object_id != bundle.trace_ir_version_id:
            raise SelectionContextFirewallPolicyError("decision and evidence bundle trace mismatch")
        _validate_decision_refs(decision)


def _validate_decision_refs(decision: LabelDecisionV2) -> None:
    refs = (
        label_decision_ref(decision),
        decision.label_spec_ref,
        decision.trace_envelope_ref,
        *decision.audit.input_refs,
        *(evidence.subject_ref for evidence in _decision_evidence(decision)),
    )
    if any(not is_safe_task_authoring_ref(ref) for ref in refs):
        raise SelectionContextFirewallPolicyError("unsafe decision reference")


def _validate_bundle(bundle: EvidenceBundle) -> None:
    if bundle.consumer_stage != TASK_EPISODE_CONSUMER_STAGE:
        raise SelectionContextFirewallPolicyError("unexpected evidence bundle consumer stage")
    if bundle.purpose != TASK_EPISODE_EVIDENCE_PURPOSE:
        raise SelectionContextFirewallPolicyError("unexpected evidence bundle purpose")
    if bundle.projection_policy_ref.object_type != "projection-policy" or not is_safe_task_authoring_ref(
        bundle.projection_policy_ref
    ):
        raise SelectionContextFirewallPolicyError("invalid evidence bundle projection policy")
    if bundle.model_dump(mode="python")["tainted_content_included"] is not False:
        raise SelectionContextFirewallPolicyError("tainted evidence bundle cannot enter task authoring")
    if bundle.returned_characters > bundle.max_characters:
        raise SelectionContextFirewallPolicyError("evidence bundle exceeds authorized budget")
    evidence_ids: set[str] = set()
    for evidence in bundle.evidence:
        if evidence.evidence_ref_id in evidence_ids:
            raise SelectionContextFirewallPolicyError("duplicate bundle evidence identity")
        evidence_ids.add(evidence.evidence_ref_id)
        if not is_safe_task_authoring_ref(evidence.subject_ref):
            raise SelectionContextFirewallPolicyError("unsafe evidence reference")
        if any(span.source_trace_id != bundle.source_trace_id for span in evidence.source_spans):
            raise SelectionContextFirewallPolicyError(
                "evidence source trace does not match bundle source trace"
            )
    expected_hash = _stable_hash(_bundle_seed(bundle))
    if (
        bundle.bundle_sha256 != expected_hash
        or bundle.evidence_bundle_id != f"evidence-bundle://sha256/{expected_hash}"
    ):
        raise SelectionContextFirewallPolicyError("evidence bundle identity is stale or invalid")


def _validate_notes(
    notes: tuple[ObjectRef, ...],
    bundle: EvidenceBundle,
) -> None:
    note_keys = [_ref_key(ref) for ref in notes]
    if len(note_keys) != len(set(note_keys)):
        raise SelectionContextFirewallPolicyError("duplicate note reference")
    authorized = {_ref_key(item.subject_ref) for item in bundle.evidence}
    for note in notes:
        if not is_safe_task_authoring_ref(note):
            raise SelectionContextFirewallPolicyError("unsafe note reference")
        if _ref_key(note) not in authorized:
            raise SelectionContextFirewallPolicyError(
                "note reference is not authorized by the evidence bundle"
            )


def _decision_blocked_reasons(
    decisions: tuple[LabelDecisionV2, ...],
) -> frozenset[SelectionContextFirewallReason]:
    if not decisions:
        return frozenset({SelectionContextFirewallReason.NO_DECISIONS})
    reasons: set[SelectionContextFirewallReason] = set()
    if any(item.decision is not LabelDecisionValueV2.MATCH for item in decisions):
        reasons.add(SelectionContextFirewallReason.DECISION_NOT_MATCH)
    if any(item.execution_status is not LabelExecutionStatus.FINAL for item in decisions):
        reasons.add(SelectionContextFirewallReason.DECISION_NOT_FINAL)
    if any(item.unresolved_reasons for item in decisions):
        reasons.add(SelectionContextFirewallReason.DECISION_UNRESOLVED)
    return frozenset(reasons)


def _excluded_signal_hashes(
    decisions: tuple[LabelDecisionV2, ...],
    restricted_signal_refs: tuple[ObjectRef, ...],
) -> tuple[Sha256, ...]:
    values = {ref.object_sha256 for ref in restricted_signal_refs}
    for decision in decisions:
        values.add(decision.decision_sha256)
        values.add(decision.label_spec_ref.object_sha256)
        values.update(evidence.subject_ref.object_sha256 for evidence in _decision_evidence(decision))
    return tuple(sorted(values))


def _selection_context(
    *,
    request: SelectionContextFirewallRequest,
    excluded_hashes: tuple[Sha256, ...],
    audit: ContractAudit,
) -> SelectionContextV2:
    decision_refs = _sort_refs(tuple(label_decision_ref(item) for item in request.decisions))
    bundle_ref = _evidence_bundle_ref(request.evidence_bundle)
    notes = _sort_refs(request.task_authoring_note_refs)
    trace_ref = request.decisions[0].trace_envelope_ref
    candidate_seed = {
        "trace_envelope_ref": _ref_payload(trace_ref),
        "approved_label_decision_refs": [_ref_payload(ref) for ref in decision_refs],
        "policy_version": SELECTION_CONTEXT_FIREWALL_POLICY_VERSION,
    }
    candidate_id = _stable_id("candidate", candidate_seed)
    context_seed = {
        "candidate_id": candidate_id,
        "approved_label_decision_refs": [_ref_payload(ref) for ref in decision_refs],
        "safe_evidence_bundle_ref": _ref_payload(bundle_ref),
        "task_authoring_note_refs": [_ref_payload(ref) for ref in notes],
        "excluded_signal_hashes": list(excluded_hashes),
        "projection_policy_ref": _ref_payload(request.evidence_bundle.projection_policy_ref),
        "policy_version": SELECTION_CONTEXT_FIREWALL_POLICY_VERSION,
    }
    context_hash = _stable_hash(context_seed)
    return SelectionContextV2(
        selection_context_id=f"selection-context://sha256/{context_hash}",
        candidate_id=candidate_id,
        approved_label_decision_refs=decision_refs,
        safe_evidence_bundle_ref=bundle_ref,
        task_authoring_note_refs=notes,
        excluded_signal_hashes=excluded_hashes,
        projection_policy_ref=request.evidence_bundle.projection_policy_ref,
        policy_version=SELECTION_CONTEXT_FIREWALL_POLICY_VERSION,
        selection_context_sha256=context_hash,
        audit=audit,
    )


def _result(
    *,
    outcome: SelectionContextFirewallOutcome,
    selection_context: SelectionContextV2 | None,
    blocked_reasons: frozenset[SelectionContextFirewallReason],
    excluded_signal_count: int,
    input_sha256: Sha256,
    audit: ContractAudit,
) -> SelectionContextFirewallResult:
    context_ref = (
        None if selection_context is None else _ref_payload(selection_context_ref(selection_context))
    )
    seed = {
        "input_sha256": input_sha256,
        "outcome": outcome.value,
        "selection_context_ref": context_ref,
        "blocked_reasons": sorted(item.value for item in blocked_reasons),
        "excluded_signal_count": excluded_signal_count,
        "policy_version": SELECTION_CONTEXT_FIREWALL_POLICY_VERSION,
    }
    result_hash = _stable_hash(seed)
    return SelectionContextFirewallResult(
        selection_context_firewall_result_id=(f"selection-context-firewall-result://sha256/{result_hash}"),
        outcome=outcome,
        selection_context=selection_context,
        blocked_reasons=blocked_reasons,
        excluded_signal_count=excluded_signal_count,
        input_sha256=input_sha256,
        result_sha256=result_hash,
        audit=audit,
    )


def _input_hash(request: SelectionContextFirewallRequest) -> Sha256:
    decisions = sorted(
        request.decisions,
        key=lambda item: _ref_key(label_decision_ref(item)),
    )
    return _stable_hash(
        {
            "decisions": [
                item.model_dump(
                    mode="json",
                    exclude={"audit"},
                    exclude_none=False,
                )
                for item in decisions
            ],
            "evidence_bundle": request.evidence_bundle.model_dump(
                mode="json",
                exclude={"audit"},
                exclude_none=False,
            ),
            "task_authoring_note_refs": [
                _ref_payload(ref) for ref in _sort_refs(request.task_authoring_note_refs)
            ],
            "restricted_signal_refs": [
                _ref_payload(ref)
                for ref in sorted(
                    request.restricted_signal_refs,
                    key=_ref_key,
                )
            ],
            "policy_version": SELECTION_CONTEXT_FIREWALL_POLICY_VERSION,
        }
    )


def _sanitized_audit(
    request: SelectionContextFirewallRequest,
) -> ContractAudit:
    safe_refs = (
        *(label_decision_ref(item) for item in request.decisions),
        _evidence_bundle_ref(request.evidence_bundle),
        request.evidence_bundle.projection_policy_ref,
        *request.task_authoring_note_refs,
    )
    unique_refs = {_ref_key(ref): ref for ref in safe_refs}
    return ContractAudit(
        created_at=request.audit.created_at,
        created_by=request.audit.created_by,
        governing_versions=request.audit.governing_versions,
        input_refs=tuple(unique_refs[key] for key in sorted(unique_refs)),
    )


def _bundle_seed(bundle: EvidenceBundle) -> dict[str, object]:
    return {
        "source_trace_id": bundle.source_trace_id,
        "trace_ir_version_id": bundle.trace_ir_version_id,
        "consumer_stage": bundle.consumer_stage,
        "purpose": bundle.purpose,
        "projection_policy_ref": _ref_payload(bundle.projection_policy_ref),
        "evidence": [item.model_dump(mode="json", exclude_none=False) for item in bundle.evidence],
        "excluded_subject_refs": [_ref_payload(ref) for ref in bundle.excluded_subject_refs],
        "returned_characters": bundle.returned_characters,
        "max_characters": bundle.max_characters,
        "policy_version": EVIDENCE_COMPILATION_POLICY_VERSION,
    }


def _decision_evidence(
    decision: LabelDecisionV2,
) -> tuple[EvidenceRef, ...]:
    return (
        *decision.positive_evidence,
        *decision.negative_evidence,
        *decision.semantic_evidence,
    )


def _evidence_bundle_ref(bundle: EvidenceBundle) -> ObjectRef:
    return ObjectRef(
        object_type="evidence-bundle",
        object_id=bundle.evidence_bundle_id,
        object_version="v1",
        object_sha256=bundle.bundle_sha256,
    )


def _sort_refs(refs: tuple[ObjectRef, ...]) -> tuple[ObjectRef, ...]:
    return tuple(sorted(refs, key=_ref_key))


def _ref_payload(ref: ObjectRef) -> dict[str, object]:
    return ref.model_dump(mode="json", exclude_none=False)


def _ref_key(ref: ObjectRef) -> tuple[str, str, str, str]:
    return (
        ref.object_type,
        ref.object_id,
        ref.object_version,
        ref.object_sha256,
    )


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
