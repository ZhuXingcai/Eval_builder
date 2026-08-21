from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime

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
    ProvenanceDecisionInput,
    ProvenanceDecisionTable,
    ProvenancePolicyError,
)

HASH = "a" * 64


def _audit(created_at: datetime = datetime(2026, 7, 23, tzinfo=UTC)) -> ContractAudit:
    return ContractAudit(
        created_at=created_at,
        created_by="provenance-test",
        governing_versions=(VersionBinding(component="provenance-decision-table", version="r2-01"),),
    )


def _ref(object_type: str = "subject", digest: str = HASH) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://example/v1",
        object_version="v1",
        object_sha256=digest,
    )


def _request(
    *,
    origin: OriginClass,
    visibility: Visibility = Visibility.STAGE_PROJECTION,
    taints: frozenset[TaintLabel] = frozenset(),
    risks: frozenset[ContentRiskLabel] = frozenset(),
    rule_ids: tuple[str, ...] = ("test-rule/v1",),
    audit: ContractAudit | None = None,
) -> ProvenanceDecisionInput:
    return ProvenanceDecisionInput(
        subject_ref=_ref(),
        origin_class=origin,
        visibility=visibility,
        taint_labels=taints,
        content_risk_labels=risks,
        derived_from=(),
        source_event_refs=(_ref("trace-event"),),
        rule_ids=rule_ids,
        confidence=1.0,
        audit=audit or _audit(),
    )


@pytest.mark.parametrize(
    ("origin", "expected"),
    [
        (OriginClass.USER_SUPPLIED_INPUT, Disposition.ALLOW_INPUT_EVIDENCE),
        (OriginClass.PREEXISTING_WORKSPACE_INPUT, Disposition.ALLOW_INPUT_EVIDENCE),
        (OriginClass.SYSTEM_OR_HARNESS_CONTEXT, Disposition.ALLOW_INPUT_EVIDENCE),
        (OriginClass.AGENT_RETRIEVED_EXTERNAL, Disposition.ALLOW_EXTERNAL_LEAD_ONLY),
        (OriginClass.AGENT_GENERATED_INTERMEDIATE, Disposition.QUARANTINE),
        (OriginClass.AGENT_GENERATED_FINAL, Disposition.REJECT),
        (OriginClass.UNKNOWN, Disposition.QUARANTINE),
    ],
)
def test_decision_table_classifies_origin_safety_gold(
    origin: OriginClass,
    expected: Disposition,
) -> None:
    decision = ProvenanceDecisionTable().decide(_request(origin=origin))

    assert decision.origin_class is origin
    assert decision.disposition is expected
    assert decision.policy_version == ProvenanceDecisionTable.policy_version
    assert decision.subject_sha256 == HASH


@pytest.mark.parametrize(
    ("taints", "risks", "expected"),
    [
        (frozenset({TaintLabel.FINAL_OUTPUT_DERIVED}), frozenset(), Disposition.REJECT),
        (frozenset({TaintLabel.PRIVATE_REFERENCE_DERIVED}), frozenset(), Disposition.REJECT),
        (frozenset({TaintLabel.GRADER_RULE_DERIVED}), frozenset(), Disposition.REJECT),
        (frozenset(), frozenset({ContentRiskLabel.ANSWER_BEARING}), Disposition.REJECT),
        (frozenset(), frozenset({ContentRiskLabel.HIDDEN_PASS_CONDITION}), Disposition.REJECT),
        (frozenset(), frozenset({ContentRiskLabel.SECRET}), Disposition.REJECT),
        (frozenset({TaintLabel.UNKNOWN_DERIVATION}), frozenset(), Disposition.QUARANTINE),
        (frozenset(), frozenset({ContentRiskLabel.UNSCANNABLE_CONTENT}), Disposition.QUARANTINE),
        (frozenset(), frozenset({ContentRiskLabel.PROMPT_INJECTION}), Disposition.QUARANTINE),
    ],
)
def test_decision_table_applies_non_waivable_and_fail_closed_rules(
    taints: frozenset[TaintLabel],
    risks: frozenset[ContentRiskLabel],
    expected: Disposition,
) -> None:
    decision = ProvenanceDecisionTable().decide(
        _request(
            origin=OriginClass.USER_SUPPLIED_INPUT,
            taints=taints,
            risks=risks,
        )
    )

    assert decision.disposition is expected
    assert decision.review_required is True


def test_decision_table_allows_structure_only_without_content_evidence() -> None:
    decision = ProvenanceDecisionTable().decide(
        _request(
            origin=OriginClass.PREEXISTING_WORKSPACE_INPUT,
            visibility=Visibility.STAGE_PROJECTION,
            risks=frozenset(),
            rule_ids=("structure-only/v1",),
        ),
        requested_disposition=Disposition.ALLOW_STRUCTURE_ONLY,
    )

    assert decision.disposition is Disposition.ALLOW_STRUCTURE_ONLY
    assert decision.review_required is False


def test_decision_table_rejects_requested_allow_for_non_waivable_subject() -> None:
    with pytest.raises(ProvenancePolicyError, match="cannot satisfy requested disposition"):
        ProvenanceDecisionTable().decide(
            _request(
                origin=OriginClass.AGENT_GENERATED_FINAL,
                taints=frozenset({TaintLabel.FINAL_OUTPUT_DERIVED}),
                risks=frozenset({ContentRiskLabel.ANSWER_BEARING}),
            ),
            requested_disposition=Disposition.ALLOW_INPUT_EVIDENCE,
        )


def test_contract_validator_remains_last_line_of_defense() -> None:
    with pytest.raises(ValidationError, match="cannot be allowed"):
        ProvenanceDecision(
            provenance_decision_id="provenance-decision://manual",
            subject_ref=_ref(),
            origin_class=OriginClass.AGENT_GENERATED_FINAL,
            taint_labels=frozenset({TaintLabel.FINAL_OUTPUT_DERIVED}),
            content_risk_labels=frozenset(),
            visibility=Visibility.CONTESTANT_VISIBLE,
            disposition=Disposition.ALLOW_INPUT_EVIDENCE,
            rule_ids=("manual-invalid/v1",),
            confidence=1.0,
            review_required=True,
            policy_version=ProvenanceDecisionTable.policy_version,
            subject_sha256=HASH,
            audit=_audit(),
        )


def test_decision_id_is_stable_across_audit_timestamp() -> None:
    table = ProvenanceDecisionTable()
    first = table.decide(
        _request(
            origin=OriginClass.USER_SUPPLIED_INPUT,
            audit=_audit(datetime(2026, 7, 23, tzinfo=UTC)),
        )
    )
    second = table.decide(
        _request(
            origin=OriginClass.USER_SUPPLIED_INPUT,
            audit=_audit(datetime(2026, 7, 24, tzinfo=UTC)),
        )
    )

    assert first.provenance_decision_id == second.provenance_decision_id
    assert first.canonical_sha256() != second.canonical_sha256()


def test_decision_id_is_stable_across_python_hash_seed(tmp_path) -> None:
    script = tmp_path / "check_decision.py"
    script.write_text(
        """
from datetime import UTC, datetime
from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.contracts.safety import OriginClass, Visibility
from eval_factory.provenance import ProvenanceDecisionInput, ProvenanceDecisionTable

subject = ObjectRef(object_type='subject', object_id='subject://example/v1', object_version='v1', object_sha256='a' * 64)
event = ObjectRef(object_type='trace-event', object_id='trace-event://example/v1', object_version='v1', object_sha256='a' * 64)
audit = ContractAudit(
    created_at=datetime(2026, 7, 23, tzinfo=UTC),
    created_by='hash-seed-test',
    governing_versions=(VersionBinding(component='provenance-decision-table', version='r2-01'),),
)
request = ProvenanceDecisionInput(
    subject_ref=subject,
    origin_class=OriginClass.USER_SUPPLIED_INPUT,
    visibility=Visibility.STAGE_PROJECTION,
    source_event_refs=(event,),
    rule_ids=('test-rule/v1',),
    confidence=1.0,
    audit=audit,
)
print(ProvenanceDecisionTable().decide(request).provenance_decision_id)
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
    assert outputs[0].startswith("provenance-decision://sha256/")


def test_decision_input_forbids_unknown_raw_payload_fields() -> None:
    with pytest.raises(ValidationError):
        ProvenanceDecisionInput.model_validate(
            {
                **_request(origin=OriginClass.USER_SUPPLIED_INPUT).model_dump(mode="json"),
                "raw_trace_text": "do not allow this",
            }
        )


def test_decision_payload_contains_no_raw_trace_text() -> None:
    decision = ProvenanceDecisionTable().decide(_request(origin=OriginClass.USER_SUPPLIED_INPUT))
    rendered = json.dumps(decision.model_dump(mode="json"), sort_keys=True)

    assert "raw_trace_text" not in rendered
    assert "Needle" not in rendered
