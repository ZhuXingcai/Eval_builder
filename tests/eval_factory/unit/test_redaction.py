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
    ConfiguredPiiRule,
    RedactionEngine,
    RedactionFindingKind,
    RedactionPolicyError,
    RedactionRequest,
)

HASH = "a" * 64
SOURCE_TEXT = "api_key = fake_token_value_123456789\nowner_id=CUST-12345\n"


def _audit(created_at: datetime = datetime(2026, 7, 23, tzinfo=UTC)) -> ContractAudit:
    return ContractAudit(
        created_at=created_at,
        created_by="redaction-test",
        governing_versions=(VersionBinding(component="redaction", version="r2-04"),),
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
    risks: frozenset[ContentRiskLabel] = frozenset(),
    taints: frozenset[TaintLabel] = frozenset(),
    disposition: Disposition = Disposition.REJECT,
) -> ProvenanceDecision:
    subject = _ref("subject", "subject://raw")
    return ProvenanceDecision(
        provenance_decision_id="provenance-decision://raw",
        subject_ref=subject,
        origin_class=OriginClass.PREEXISTING_WORKSPACE_INPUT,
        taint_labels=taints,
        content_risk_labels=risks,
        visibility=Visibility.PRIVILEGED_AUDIT,
        disposition=disposition,
        rule_ids=("parent-test/v1",),
        source_event_refs=(_ref("trace-event", "trace-event://raw"),),
        confidence=1.0,
        review_required=disposition in {Disposition.NEEDS_REVIEW, Disposition.QUARANTINE, Disposition.REJECT},
        policy_version="parent-provenance/test-v1",
        subject_sha256=subject.object_sha256,
        audit=_audit(),
    )


def _request(
    *,
    source_text: str = SOURCE_TEXT,
    parent: ProvenanceDecision | None = None,
    configured_pii_rules: tuple[ConfiguredPiiRule, ...] = (),
    audit: ContractAudit | None = None,
) -> RedactionRequest:
    parent_decision = parent or _parent_decision(
        risks=frozenset({ContentRiskLabel.SECRET, ContentRiskLabel.RESTRICTED_PII})
    )
    return RedactionRequest(
        subject_ref=parent_decision.subject_ref,
        source_text=source_text,
        parent_decision=parent_decision,
        configured_pii_rules=configured_pii_rules,
        source_event_refs=(_ref("trace-event", "trace-event://redaction"),),
        audit=audit or _audit(),
    )


def test_builtin_secret_is_redacted_without_storing_matched_value() -> None:
    result = RedactionEngine().redact(_request())

    assert "fake_token_value_123456789" not in result.redacted_text
    secret_findings = [item for item in result.findings if item.kind is RedactionFindingKind.SECRET]
    assert secret_findings
    assert all(
        "fake_token_value_123456789" not in str(item.model_dump(mode="json")) for item in result.findings
    )
    assert ContentRiskLabel.SECRET not in result.child_decision.content_risk_labels


def test_configured_pii_rule_is_detected_and_redacted() -> None:
    rule = ConfiguredPiiRule(rule_id="configured-pii/customer-id/v1", pattern=r"CUST-\d{5}")

    result = RedactionEngine().redact(_request(configured_pii_rules=(rule,)))

    assert "CUST-12345" not in result.redacted_text
    pii_findings = [item for item in result.findings if item.kind is RedactionFindingKind.RESTRICTED_PII]
    assert len(pii_findings) == 1
    assert pii_findings[0].rule_id == rule.rule_id
    assert ContentRiskLabel.RESTRICTED_PII not in result.child_decision.content_risk_labels


def test_invalid_configured_pii_rule_fails_closed() -> None:
    rule = ConfiguredPiiRule(rule_id="configured-pii/bad/v1", pattern="[")

    with pytest.raises(RedactionPolicyError, match="invalid configured PII rule"):
        RedactionEngine().redact(_request(configured_pii_rules=(rule,)))


def test_redacted_subject_ref_is_distinct_and_validated_clean() -> None:
    rule = ConfiguredPiiRule(rule_id="configured-pii/customer-id/v1", pattern=r"CUST-\d{5}")

    result = RedactionEngine().redact(_request(configured_pii_rules=(rule,)))

    assert result.redacted_ref != result.original_ref
    assert result.redacted_ref.object_sha256 == result.redacted_sha256
    assert result.post_validation_passed is True
    assert result.taint_edge.parent_refs == (result.original_ref,)
    assert result.taint_edge.child_ref == result.redacted_ref


def test_non_waivable_restrictions_survive_redaction() -> None:
    parent = _parent_decision(
        risks=frozenset({ContentRiskLabel.SECRET, ContentRiskLabel.ANSWER_BEARING}),
        taints=frozenset({TaintLabel.FINAL_OUTPUT_DERIVED}),
        disposition=Disposition.REJECT,
    )

    result = RedactionEngine().redact(_request(parent=parent, source_text=SOURCE_TEXT))

    assert ContentRiskLabel.SECRET not in result.child_decision.content_risk_labels
    assert ContentRiskLabel.ANSWER_BEARING in result.child_decision.content_risk_labels
    assert TaintLabel.FINAL_OUTPUT_DERIVED in result.child_decision.taint_labels
    assert result.child_decision.disposition is Disposition.REJECT


def test_sensitive_source_taint_survives_redaction() -> None:
    parent = _parent_decision(
        risks=frozenset({ContentRiskLabel.SECRET}),
        taints=frozenset({TaintLabel.SENSITIVE_SOURCE_DERIVED}),
        disposition=Disposition.REJECT,
    )

    result = RedactionEngine().redact(_request(parent=parent, source_text=SOURCE_TEXT))

    assert TaintLabel.SENSITIVE_SOURCE_DERIVED in result.child_decision.taint_labels
    assert result.child_decision.disposition is not Disposition.REJECT


def test_ids_and_hashes_ignore_audit_timestamp() -> None:
    request = _request(audit=_audit(datetime(2026, 7, 23, tzinfo=UTC)))
    first = RedactionEngine().redact(request)
    second = RedactionEngine().redact(_request(audit=_audit(datetime(2026, 7, 24, tzinfo=UTC))))

    assert first.redacted_ref == second.redacted_ref
    assert first.taint_edge.taint_edge_id == second.taint_edge.taint_edge_id
    assert first.child_decision.provenance_decision_id == second.child_decision.provenance_decision_id
    assert first.canonical_sha256() != second.canonical_sha256()


def test_ids_are_stable_across_python_hash_seed(tmp_path: Path) -> None:
    script = tmp_path / "check_redaction_seed.py"
    script.write_text(
        """
from datetime import UTC, datetime
from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.contracts.safety import ContentRiskLabel, Disposition, OriginClass, ProvenanceDecision, Visibility
from eval_factory.provenance import RedactionEngine, RedactionRequest

def ref(kind, object_id):
    return ObjectRef(object_type=kind, object_id=object_id, object_version='v1', object_sha256='a' * 64)

audit = ContractAudit(
    created_at=datetime(2026, 7, 23, tzinfo=UTC),
    created_by='seed-test',
    governing_versions=(VersionBinding(component='redaction', version='r2-04'),),
)
subject = ref('subject', 'subject://raw')
parent = ProvenanceDecision(
    provenance_decision_id='provenance-decision://raw',
    subject_ref=subject,
    origin_class=OriginClass.PREEXISTING_WORKSPACE_INPUT,
    visibility=Visibility.PRIVILEGED_AUDIT,
    disposition=Disposition.REJECT,
    content_risk_labels=frozenset({ContentRiskLabel.SECRET}),
    rule_ids=('parent-test/v1',),
    source_event_refs=(ref('trace-event', 'trace-event://raw'),),
    confidence=1.0,
    review_required=True,
    policy_version='parent-provenance/test-v1',
    subject_sha256='a' * 64,
    audit=audit,
)
request = RedactionRequest(
    subject_ref=subject,
    source_text='api_key = fake_token_value_123456789\\n',
    parent_decision=parent,
    source_event_refs=(ref('trace-event', 'trace-event://redaction'),),
    audit=audit,
)
result = RedactionEngine().redact(request)
print(result.redacted_ref.object_id)
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


def test_request_model_rejects_unexpected_raw_trace_field() -> None:
    request = _request()

    with pytest.raises(ValidationError):
        RedactionRequest.model_validate(
            {
                **request.model_dump(mode="json"),
                "raw_trace_text": "forbidden",
            }
        )
