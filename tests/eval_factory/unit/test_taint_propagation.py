from __future__ import annotations

import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.contracts.safety import (
    ContentRiskLabel,
    Disposition,
    OriginClass,
    ProvenanceDecision,
    TaintLabel,
    Visibility,
)
from eval_factory.provenance import (
    DerivationVerification,
    DerivedContentOperation,
    PropagationUncertainty,
    TaintPropagationEngine,
    TaintPropagationRequest,
)

HASH = "a" * 64
OTHER_HASH = "b" * 64


def _audit(created_at: datetime = datetime(2026, 7, 23, tzinfo=UTC)) -> ContractAudit:
    return ContractAudit(
        created_at=created_at,
        created_by="taint-propagation-test",
        governing_versions=(VersionBinding(component="taint-propagation", version="r2-03"),),
    )


def _ref(object_type: str, object_id: str, digest: str = HASH) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=object_id,
        object_version="v1",
        object_sha256=digest,
    )


def _parent_decision(
    *,
    subject_id: str,
    taints: frozenset[TaintLabel] = frozenset(),
    risks: frozenset[ContentRiskLabel] = frozenset(),
    disposition: Disposition = Disposition.ALLOW_INPUT_EVIDENCE,
    origin: OriginClass = OriginClass.PREEXISTING_WORKSPACE_INPUT,
    digest: str = HASH,
) -> ProvenanceDecision:
    subject = _ref("subject", subject_id, digest)
    return ProvenanceDecision(
        provenance_decision_id=f"provenance-decision://{subject_id.rsplit('/', 1)[-1]}",
        subject_ref=subject,
        origin_class=origin,
        taint_labels=taints,
        content_risk_labels=risks,
        visibility=Visibility.STAGE_PROJECTION,
        disposition=disposition,
        rule_ids=("parent-test/v1",),
        source_event_refs=(_ref("trace-event", f"trace-event://{subject_id.rsplit('/', 1)[-1]}"),),
        confidence=1.0,
        review_required=disposition in {Disposition.NEEDS_REVIEW, Disposition.QUARANTINE, Disposition.REJECT},
        policy_version="parent-provenance/test-v1",
        subject_sha256=subject.object_sha256,
        audit=_audit(),
    )


def _request(
    *,
    operation: DerivedContentOperation,
    parents: tuple[ProvenanceDecision, ...],
    verification: DerivationVerification = DerivationVerification.VERIFIED,
    child_id: str = "subject://child",
    child_digest: str = OTHER_HASH,
    origin: OriginClass = OriginClass.AGENT_GENERATED_INTERMEDIATE,
    risks: frozenset[ContentRiskLabel] = frozenset(),
    taints: frozenset[TaintLabel] = frozenset(),
    audit: ContractAudit | None = None,
) -> TaintPropagationRequest:
    return TaintPropagationRequest(
        child_ref=_ref("subject", child_id, child_digest),
        child_origin_class=origin,
        visibility=Visibility.STAGE_PROJECTION,
        operation=operation,
        parent_decisions=parents,
        transform_verification=verification,
        target_content_risk_labels=risks,
        extra_taint_labels=taints,
        rule_id=f"rule://{operation.value.lower()}",
        source_event_refs=(_ref("trace-event", f"trace-event://{operation.value.lower()}"),),
        confidence=1.0,
        audit=audit or _audit(),
    )


def test_copy_preserves_parent_taint_risk_lineage_and_disposition_floor() -> None:
    parent = _parent_decision(
        subject_id="subject://parent",
        taints=frozenset({TaintLabel.SENSITIVE_SOURCE_DERIVED}),
        risks=frozenset({ContentRiskLabel.PROMPT_INJECTION}),
        disposition=Disposition.QUARANTINE,
    )

    result = TaintPropagationEngine().propagate(
        _request(operation=DerivedContentOperation.COPY, parents=(parent,))
    )

    assert result.taint_edge.parent_refs == (parent.subject_ref,)
    assert result.taint_edge.child_ref == result.child_decision.subject_ref
    assert result.taint_edge.transform_verified is True
    assert TaintLabel.SENSITIVE_SOURCE_DERIVED in result.child_decision.taint_labels
    assert ContentRiskLabel.PROMPT_INJECTION in result.child_decision.content_risk_labels
    assert result.child_decision.disposition is Disposition.QUARANTINE
    assert result.child_decision.derived_from == (parent.subject_ref,)


def test_unverified_summary_adds_unknown_derivation_and_quarantines() -> None:
    parent = _parent_decision(subject_id="subject://source")

    result = TaintPropagationEngine().propagate(
        _request(
            operation=DerivedContentOperation.SUMMARY,
            parents=(parent,),
            verification=DerivationVerification.UNVERIFIED,
        )
    )

    assert result.taint_edge.transform_verified is False
    assert PropagationUncertainty.UNVERIFIED_TRANSFORM in result.uncertainties
    assert TaintLabel.UNKNOWN_DERIVATION in result.child_decision.taint_labels
    assert result.child_decision.disposition is Disposition.QUARANTINE


def test_translation_from_final_output_derived_parent_remains_rejected() -> None:
    parent = _parent_decision(
        subject_id="subject://final",
        taints=frozenset({TaintLabel.FINAL_OUTPUT_DERIVED}),
        disposition=Disposition.REJECT,
        origin=OriginClass.AGENT_GENERATED_FINAL,
    )

    result = TaintPropagationEngine().propagate(
        _request(operation=DerivedContentOperation.TRANSLATION, parents=(parent,))
    )

    assert TaintLabel.FINAL_OUTPUT_DERIVED in result.child_decision.taint_labels
    assert result.child_decision.disposition is Disposition.REJECT
    assert result.child_decision.review_required is True


def test_embedding_projection_never_clears_parent_taint() -> None:
    parent = _parent_decision(
        subject_id="subject://private-reference",
        taints=frozenset({TaintLabel.PRIVATE_REFERENCE_DERIVED}),
        disposition=Disposition.REJECT,
    )

    result = TaintPropagationEngine().propagate(
        _request(operation=DerivedContentOperation.EMBEDDING, parents=(parent,))
    )

    assert result.child_decision.derived_from == (parent.subject_ref,)
    assert TaintLabel.PRIVATE_REFERENCE_DERIVED in result.child_decision.taint_labels
    assert result.child_decision.disposition is Disposition.REJECT


def test_multi_parent_propagation_unions_labels_and_uses_most_restrictive_disposition() -> None:
    first = _parent_decision(
        subject_id="subject://first",
        taints=frozenset({TaintLabel.SENSITIVE_SOURCE_DERIVED}),
        risks=frozenset({ContentRiskLabel.PROMPT_INJECTION}),
        disposition=Disposition.QUARANTINE,
    )
    second = _parent_decision(
        subject_id="subject://second",
        taints=frozenset({TaintLabel.GRADER_RULE_DERIVED}),
        risks=frozenset({ContentRiskLabel.HIDDEN_PASS_CONDITION}),
        disposition=Disposition.REJECT,
        digest=OTHER_HASH,
    )

    result = TaintPropagationEngine().propagate(
        _request(operation=DerivedContentOperation.STRUCTURAL_PROJECTION, parents=(second, first))
    )

    assert result.taint_edge.parent_refs == (first.subject_ref, second.subject_ref)
    assert result.child_decision.derived_from == (first.subject_ref, second.subject_ref)
    assert result.child_decision.taint_labels == frozenset(
        {TaintLabel.SENSITIVE_SOURCE_DERIVED, TaintLabel.GRADER_RULE_DERIVED}
    )
    assert result.child_decision.content_risk_labels == frozenset(
        {ContentRiskLabel.PROMPT_INJECTION, ContentRiskLabel.HIDDEN_PASS_CONDITION}
    )
    assert result.child_decision.disposition is Disposition.REJECT


def test_missing_parent_unknown_operation_fails_closed() -> None:
    result = TaintPropagationEngine().propagate(
        _request(
            operation=DerivedContentOperation.UNKNOWN,
            parents=(),
            verification=DerivationVerification.AMBIGUOUS,
        )
    )

    assert PropagationUncertainty.MISSING_PARENT in result.uncertainties
    assert PropagationUncertainty.UNKNOWN_OPERATION in result.uncertainties
    assert TaintLabel.UNKNOWN_DERIVATION in result.child_decision.taint_labels
    assert ContentRiskLabel.UNSCANNABLE_CONTENT in result.child_decision.content_risk_labels
    assert result.child_decision.disposition is Disposition.QUARANTINE


def test_ids_ignore_audit_timestamp() -> None:
    parent = _parent_decision(subject_id="subject://stable")
    first = TaintPropagationEngine().propagate(
        _request(
            operation=DerivedContentOperation.COPY,
            parents=(parent,),
            audit=_audit(datetime(2026, 7, 23, tzinfo=UTC)),
        )
    )
    second = TaintPropagationEngine().propagate(
        _request(
            operation=DerivedContentOperation.COPY,
            parents=(parent,),
            audit=_audit(datetime(2026, 7, 24, tzinfo=UTC)),
        )
    )

    assert first.taint_edge.taint_edge_id == second.taint_edge.taint_edge_id
    assert first.child_decision.provenance_decision_id == second.child_decision.provenance_decision_id
    assert first.canonical_sha256() != second.canonical_sha256()


def test_ids_are_stable_across_python_hash_seed(tmp_path: Path) -> None:
    script = tmp_path / "check_taint_seed.py"
    script.write_text(
        """
from datetime import UTC, datetime
from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.contracts.safety import Disposition, OriginClass, ProvenanceDecision, Visibility
from eval_factory.provenance import DerivedContentOperation, TaintPropagationEngine, TaintPropagationRequest

def ref(kind, object_id, digest='a' * 64):
    return ObjectRef(object_type=kind, object_id=object_id, object_version='v1', object_sha256=digest)

audit = ContractAudit(
    created_at=datetime(2026, 7, 23, tzinfo=UTC),
    created_by='seed-test',
    governing_versions=(VersionBinding(component='taint-propagation', version='r2-03'),),
)
parent = ProvenanceDecision(
    provenance_decision_id='provenance-decision://seed-parent',
    subject_ref=ref('subject', 'subject://seed-parent'),
    origin_class=OriginClass.PREEXISTING_WORKSPACE_INPUT,
    visibility=Visibility.STAGE_PROJECTION,
    disposition=Disposition.ALLOW_INPUT_EVIDENCE,
    rule_ids=('parent-test/v1',),
    source_event_refs=(ref('trace-event', 'trace-event://seed-parent'),),
    confidence=1.0,
    review_required=False,
    policy_version='parent-provenance/test-v1',
    subject_sha256='a' * 64,
    audit=audit,
)
request = TaintPropagationRequest(
    child_ref=ref('subject', 'subject://seed-child', 'b' * 64),
    child_origin_class=OriginClass.AGENT_GENERATED_INTERMEDIATE,
    visibility=Visibility.STAGE_PROJECTION,
    operation=DerivedContentOperation.COPY,
    parent_decisions=(parent,),
    rule_id='rule://copy',
    source_event_refs=(ref('trace-event', 'trace-event://copy'),),
    confidence=1.0,
    audit=audit,
)
result = TaintPropagationEngine().propagate(request)
print(result.taint_edge.taint_edge_id)
print(result.child_decision.provenance_decision_id)
""",
        encoding="utf-8",
    )
    outputs = []
    for seed in ("1", "99"):
        result = subprocess.run(
            [sys.executable, str(script)],
            check=False,
            capture_output=True,
            text=True,
            env={**dict(PYTHONHASHSEED=seed), "PYTHONPATH": "src"},
        )
        assert result.returncode == 0, result.stderr
        outputs.append(result.stdout.strip())

    assert len(set(outputs)) == 1


def test_request_model_rejects_raw_text_fields() -> None:
    parent = _parent_decision(subject_id="subject://strict")
    request = _request(operation=DerivedContentOperation.COPY, parents=(parent,))

    with pytest.raises(ValidationError):
        TaintPropagationRequest.model_validate(
            {
                **request.model_dump(mode="json"),
                "raw_trace_text": "forbidden",
            }
        )
