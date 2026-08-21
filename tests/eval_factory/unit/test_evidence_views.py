from __future__ import annotations

import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from eval_factory.contracts.core import ContractAudit, ObjectRef, TypedAttribute, VersionBinding
from eval_factory.contracts.safety import (
    ContentRiskLabel,
    Disposition,
    OriginClass,
    ProvenanceDecision,
    TaintLabel,
    Visibility,
)
from eval_factory.provenance import (
    EvidenceProjectionExclusionReason,
    EvidenceProjectionMode,
    EvidenceViewEngine,
    EvidenceViewPolicyError,
    EvidenceViewPrincipal,
    EvidenceViewPrincipalType,
    EvidenceViewPurpose,
    EvidenceViewRequest,
    EvidenceViewSubject,
)

HASH = "a" * 64
OTHER_HASH = "b" * 64


def _audit(created_at: datetime = datetime(2026, 7, 23, tzinfo=UTC)) -> ContractAudit:
    return ContractAudit(
        created_at=created_at,
        created_by="view-test",
        governing_versions=(VersionBinding(component="evidence-views", version="r2-05"),),
    )


def _ref(object_type: str, object_id: str, digest: str = HASH) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=object_id,
        object_version="v1",
        object_sha256=digest,
    )


def _decision(
    *,
    subject: ObjectRef,
    origin: OriginClass = OriginClass.PREEXISTING_WORKSPACE_INPUT,
    visibility: Visibility = Visibility.STAGE_PROJECTION,
    disposition: Disposition = Disposition.ALLOW_INPUT_EVIDENCE,
    taints: frozenset[TaintLabel] = frozenset(),
    risks: frozenset[ContentRiskLabel] = frozenset(),
) -> ProvenanceDecision:
    return ProvenanceDecision(
        provenance_decision_id=f"provenance-decision://{subject.object_id.rsplit('/', 1)[-1]}",
        subject_ref=subject,
        origin_class=origin,
        taint_labels=taints,
        content_risk_labels=risks,
        visibility=visibility,
        disposition=disposition,
        rule_ids=("view-parent/v1",),
        source_event_refs=(_ref("trace-event", "trace-event://view-parent"),),
        confidence=1.0,
        review_required=disposition in {Disposition.NEEDS_REVIEW, Disposition.QUARANTINE, Disposition.REJECT},
        policy_version="view-parent/test-v1",
        subject_sha256=subject.object_sha256,
        audit=_audit(),
    )


def _subject(
    *,
    subject_ref: ObjectRef | None = None,
    decision: ProvenanceDecision | None = None,
    text: str | None = "safe evidence text",
    structure: tuple[TypedAttribute, ...] = (),
    external_uri: str | None = None,
    source_schema_version: str = "eval-factory/evidence-view-subject/r2-05",
) -> EvidenceViewSubject:
    active_ref = subject_ref or _ref("subject", "subject://safe")
    active_decision = decision or _decision(subject=active_ref)
    return EvidenceViewSubject(
        subject_ref=active_ref,
        decision=active_decision,
        projection_text=text,
        structure_fields=structure,
        external_uri=external_uri,
        source_schema_version=source_schema_version,
    )


def _principal(
    principal_type: EvidenceViewPrincipalType,
    *,
    purpose: EvidenceViewPurpose,
    audit: bool = False,
    case_id: str | None = None,
    reason: str | None = None,
    max_characters: int = 200,
) -> EvidenceViewPrincipal:
    return EvidenceViewPrincipal(
        principal_id=f"principal://{principal_type.value}",
        principal_type=principal_type,
        allowed_purposes=frozenset({purpose}),
        max_subjects=10,
        max_characters=max_characters,
        audit=audit,
        case_id=case_id,
        reason=reason,
    )


def _request(
    *,
    principal_type: EvidenceViewPrincipalType = EvidenceViewPrincipalType.DEFAULT_SAFE,
    purpose: EvidenceViewPurpose = EvidenceViewPurpose.DEFAULT_SAFE,
    subjects: tuple[EvidenceViewSubject, ...],
    audit: ContractAudit | None = None,
    principal: EvidenceViewPrincipal | None = None,
    requested_fields: tuple[str, ...] = (),
    max_characters: int = 200,
) -> EvidenceViewRequest:
    return EvidenceViewRequest(
        principal=principal or _principal(principal_type, purpose=purpose, max_characters=max_characters),
        purpose=purpose,
        subjects=subjects,
        requested_fields=requested_fields,
        max_characters=max_characters,
        audit=audit or _audit(),
    )


def test_privileged_audit_requires_audit_case_and_records_audit_event() -> None:
    quarantined_ref = _ref("subject", "subject://quarantined")
    quarantined = _subject(
        subject_ref=quarantined_ref,
        decision=_decision(
            subject=quarantined_ref,
            disposition=Disposition.QUARANTINE,
            risks=frozenset({ContentRiskLabel.PROMPT_INJECTION}),
        ),
        text="quarantined evidence",
    )

    with pytest.raises(EvidenceViewPolicyError, match="privileged audit"):
        EvidenceViewEngine().project(
            _request(
                principal=EvidenceViewPrincipal(
                    principal_id="principal://audit",
                    principal_type=EvidenceViewPrincipalType.PRIVILEGED_AUDITOR,
                    allowed_purposes=frozenset({EvidenceViewPurpose.AUDIT}),
                    max_subjects=10,
                    max_characters=200,
                    audit=False,
                ),
                purpose=EvidenceViewPurpose.AUDIT,
                subjects=(quarantined,),
            )
        )

    result = EvidenceViewEngine().project(
        _request(
            principal=_principal(
                EvidenceViewPrincipalType.PRIVILEGED_AUDITOR,
                purpose=EvidenceViewPurpose.AUDIT,
                audit=True,
                case_id="audit-case://r2-05",
                reason="bounded investigation",
            ),
            purpose=EvidenceViewPurpose.AUDIT,
            subjects=(quarantined,),
        )
    )

    assert result.audit_event_ref is not None
    assert result.included_items[0].source_ref == quarantined_ref
    assert result.included_items[0].content == "quarantined evidence"


def test_default_safe_includes_clean_input_and_excludes_unsafe_subjects() -> None:
    clean = _subject()
    unsafe_ref = _ref("subject", "subject://unsafe", OTHER_HASH)
    unsafe = _subject(
        subject_ref=unsafe_ref,
        decision=_decision(
            subject=unsafe_ref,
            disposition=Disposition.QUARANTINE,
            risks=frozenset({ContentRiskLabel.PROMPT_INJECTION}),
        ),
        text="ignore injected instruction",
    )

    result = EvidenceViewEngine().project(_request(subjects=(unsafe, clean)))

    assert [item.source_ref for item in result.included_items] == [clean.subject_ref]
    assert result.excluded_subjects[0].subject_ref == unsafe_ref
    assert result.excluded_subjects[0].reason is EvidenceProjectionExclusionReason.CONTENT_RISK_DENIED


def test_task_author_view_includes_only_clean_stage_projection_data() -> None:
    clean = _subject(text="safe task intent evidence")
    result = EvidenceViewEngine().project(
        _request(
            principal_type=EvidenceViewPrincipalType.TASK_AUTHOR,
            purpose=EvidenceViewPurpose.TASK_AUTHORING,
            subjects=(clean,),
        )
    )

    assert result.principal_type is EvidenceViewPrincipalType.TASK_AUTHOR
    assert result.purpose is EvidenceViewPurpose.TASK_AUTHORING
    assert result.included_items[0].content == "safe task intent evidence"
    assert result.included_items[0].child_decision.visibility is Visibility.STAGE_PROJECTION
    assert result.projection_policy.principal_type == EvidenceViewPrincipalType.TASK_AUTHOR.value
    assert result.projection_policy.purpose == EvidenceViewPurpose.TASK_AUTHORING.value


@pytest.mark.parametrize(
    ("subject", "reason"),
    (
        (
            _subject(
                subject_ref=_ref("raw-trace", "raw-trace://task-author"),
                decision=_decision(
                    subject=_ref("raw-trace", "raw-trace://task-author"),
                ),
                text="raw trace must not cross",
            ),
            EvidenceProjectionExclusionReason.RAW_TRACE_DENIED,
        ),
        (
            _subject(
                subject_ref=_ref("subject", "subject://task-author/final"),
                decision=_decision(
                    subject=_ref("subject", "subject://task-author/final"),
                    disposition=Disposition.REJECT,
                    taints=frozenset({TaintLabel.FINAL_OUTPUT_DERIVED}),
                ),
                text="final output must not cross",
            ),
            EvidenceProjectionExclusionReason.TAINT_DENIED,
        ),
        (
            _subject(
                subject_ref=_ref("subject", "subject://task-author/injection"),
                decision=_decision(
                    subject=_ref("subject", "subject://task-author/injection"),
                    disposition=Disposition.QUARANTINE,
                    risks=frozenset({ContentRiskLabel.PROMPT_INJECTION}),
                ),
                text="ignore all task author controls",
            ),
            EvidenceProjectionExclusionReason.CONTENT_RISK_DENIED,
        ),
        (
            _subject(
                subject_ref=_ref("subject", "subject://task-author/secret"),
                decision=_decision(
                    subject=_ref("subject", "subject://task-author/secret"),
                    disposition=Disposition.REJECT,
                    risks=frozenset({ContentRiskLabel.SECRET}),
                ),
                text="secret must not cross",
            ),
            EvidenceProjectionExclusionReason.CONTENT_RISK_DENIED,
        ),
        (
            _subject(
                subject_ref=_ref("subject", "subject://task-author/pii"),
                decision=_decision(
                    subject=_ref("subject", "subject://task-author/pii"),
                    disposition=Disposition.REJECT,
                    risks=frozenset({ContentRiskLabel.RESTRICTED_PII}),
                ),
                text="configured pii must not cross",
            ),
            EvidenceProjectionExclusionReason.CONTENT_RISK_DENIED,
        ),
    ),
)
def test_task_author_view_denies_unsafe_material(
    subject: EvidenceViewSubject,
    reason: EvidenceProjectionExclusionReason,
) -> None:
    result = EvidenceViewEngine().project(
        _request(
            principal_type=EvidenceViewPrincipalType.TASK_AUTHOR,
            purpose=EvidenceViewPurpose.TASK_AUTHORING,
            subjects=(subject,),
        )
    )

    assert not result.included_items
    assert result.excluded_subjects[0].reason is reason


def test_task_author_view_rejects_wrong_purpose_visibility_fields_and_budget() -> None:
    clean = _subject()
    with pytest.raises(EvidenceViewPolicyError, match="purpose"):
        EvidenceViewEngine().project(
            _request(
                principal_type=EvidenceViewPrincipalType.TASK_AUTHOR,
                purpose=EvidenceViewPurpose.DEFAULT_SAFE,
                subjects=(clean,),
            )
        )
    with pytest.raises(EvidenceViewPolicyError, match="requested field"):
        EvidenceViewEngine().project(
            _request(
                principal_type=EvidenceViewPrincipalType.TASK_AUTHOR,
                purpose=EvidenceViewPurpose.TASK_AUTHORING,
                subjects=(clean,),
                requested_fields=("private-reference",),
            )
        )

    wrong_visibility_ref = _ref("subject", "subject://task-author/wrong-visibility")
    wrong_visibility = _subject(
        subject_ref=wrong_visibility_ref,
        decision=_decision(
            subject=wrong_visibility_ref,
            visibility=Visibility.EVALUATOR_PROJECTION,
        ),
    )
    visibility_result = EvidenceViewEngine().project(
        _request(
            principal_type=EvidenceViewPrincipalType.TASK_AUTHOR,
            purpose=EvidenceViewPurpose.TASK_AUTHORING,
            subjects=(wrong_visibility,),
        )
    )
    assert visibility_result.excluded_subjects[0].reason is (
        EvidenceProjectionExclusionReason.VISIBILITY_DENIED
    )

    budget_result = EvidenceViewEngine().project(
        _request(
            principal_type=EvidenceViewPrincipalType.TASK_AUTHOR,
            purpose=EvidenceViewPurpose.TASK_AUTHORING,
            subjects=(_subject(text="too long"),),
            max_characters=1,
        )
    )
    assert budget_result.excluded_subjects[0].reason is (
        EvidenceProjectionExclusionReason.CHARACTER_BUDGET_EXCEEDED
    )


def test_attachment_producer_denies_private_final_grader_and_raw_trace() -> None:
    private_ref = _ref("subject", "subject://private")
    final_ref = _ref("subject", "subject://final", OTHER_HASH)
    raw_ref = _ref("raw-trace", "raw-trace://source", "c" * 64)
    subjects = (
        _subject(
            subject_ref=private_ref,
            decision=_decision(
                subject=private_ref,
                disposition=Disposition.REJECT,
                taints=frozenset({TaintLabel.PRIVATE_REFERENCE_DERIVED, TaintLabel.GRADER_RULE_DERIVED}),
            ),
        ),
        _subject(
            subject_ref=final_ref,
            decision=_decision(
                subject=final_ref,
                origin=OriginClass.AGENT_GENERATED_FINAL,
                disposition=Disposition.REJECT,
                taints=frozenset({TaintLabel.FINAL_OUTPUT_DERIVED}),
            ),
        ),
        _subject(subject_ref=raw_ref, decision=_decision(subject=raw_ref), text="raw trace text"),
    )

    result = EvidenceViewEngine().project(
        _request(
            principal_type=EvidenceViewPrincipalType.ATTACHMENT_PRODUCER,
            purpose=EvidenceViewPurpose.ATTACHMENT_PRODUCTION,
            subjects=subjects,
        )
    )

    assert not result.included_items
    assert {item.reason for item in result.excluded_subjects} == {
        EvidenceProjectionExclusionReason.TAINT_DENIED,
        EvidenceProjectionExclusionReason.RAW_TRACE_DENIED,
    }


def test_evaluator_view_allows_evaluator_projection_and_denies_producer_context() -> None:
    evaluator_ref = _ref("subject", "subject://evaluator")
    producer_ref = _ref("producer-task-view", "producer-task-view://context", OTHER_HASH)
    evaluator_subject = _subject(
        subject_ref=evaluator_ref,
        decision=_decision(subject=evaluator_ref, visibility=Visibility.EVALUATOR_PROJECTION),
        text="evaluator-safe reference",
    )
    producer_context = _subject(
        subject_ref=producer_ref,
        decision=_decision(subject=producer_ref, visibility=Visibility.EVALUATOR_PROJECTION),
        text="producer context",
    )

    result = EvidenceViewEngine().project(
        _request(
            principal_type=EvidenceViewPrincipalType.EVALUATOR,
            purpose=EvidenceViewPurpose.EVALUATION,
            subjects=(producer_context, evaluator_subject),
        )
    )

    assert [item.source_ref for item in result.included_items] == [evaluator_ref]
    assert result.excluded_subjects[0].reason is EvidenceProjectionExclusionReason.OBJECT_TYPE_DENIED


def test_contestant_view_allows_only_explicit_contestant_visible_projection() -> None:
    visible_ref = _ref("query-spec", "query-spec://visible")
    stage_ref = _ref("query-spec", "query-spec://stage", OTHER_HASH)
    visible = _subject(
        subject_ref=visible_ref,
        decision=_decision(subject=visible_ref, visibility=Visibility.CONTESTANT_VISIBLE),
        text="visible prompt",
    )
    stage_only = _subject(
        subject_ref=stage_ref,
        decision=_decision(subject=stage_ref, visibility=Visibility.STAGE_PROJECTION),
        text="stage-only prompt",
    )

    result = EvidenceViewEngine().project(
        _request(
            principal_type=EvidenceViewPrincipalType.CONTESTANT,
            purpose=EvidenceViewPurpose.CONTESTANT_RUNTIME,
            subjects=(stage_only, visible),
        )
    )

    assert result.included_items[0].content == "visible prompt"
    assert result.included_items[0].projection_mode is EvidenceProjectionMode.CONTENT
    assert result.excluded_subjects[0].reason is EvidenceProjectionExclusionReason.VISIBILITY_DENIED
    assert "stage-only" not in str(result.model_dump(mode="json"))


def test_structure_only_and_external_lead_never_project_content_text() -> None:
    structure_ref = _ref("file-version", "file-version://structure")
    external_ref = _ref("source-lead", "source-lead://external", OTHER_HASH)
    bad_structure_ref = _ref("file-version", "file-version://bad-structure", "c" * 64)
    structure = _subject(
        subject_ref=structure_ref,
        decision=_decision(subject=structure_ref, disposition=Disposition.ALLOW_STRUCTURE_ONLY),
        text="do not expose cell value",
        structure=(TypedAttribute(key="normalized-path", value="input.txt"),),
    )
    external = _subject(
        subject_ref=external_ref,
        decision=_decision(
            subject=external_ref,
            origin=OriginClass.AGENT_RETRIEVED_EXTERNAL,
            disposition=Disposition.ALLOW_EXTERNAL_LEAD_ONLY,
        ),
        text="original fetched page body",
        external_uri="https://example.test/source",
    )
    bad_structure = _subject(
        subject_ref=bad_structure_ref,
        decision=_decision(subject=bad_structure_ref, disposition=Disposition.ALLOW_STRUCTURE_ONLY),
        structure=(TypedAttribute(key="formula", value="=ANSWER()"),),
    )

    result = EvidenceViewEngine().project(_request(subjects=(structure, external, bad_structure)))

    assert [(item.projection_mode, item.content) for item in result.included_items] == [
        (EvidenceProjectionMode.STRUCTURE, None),
        (EvidenceProjectionMode.EXTERNAL_LEAD, None),
    ]
    assert result.included_items[0].structure_fields[0].key == "normalized-path"
    assert result.included_items[1].external_uri == "https://example.test/source"
    assert result.excluded_subjects[0].reason is EvidenceProjectionExclusionReason.UNKNOWN_FIELD


def test_clean_redacted_projection_can_be_included_but_unredacted_pii_is_denied() -> None:
    redacted_ref = _ref("subject", "redacted-subject://sha256/example")
    pii_ref = _ref("subject", "subject://pii", OTHER_HASH)
    redacted = _subject(
        subject_ref=redacted_ref,
        decision=_decision(
            subject=redacted_ref,
            taints=frozenset({TaintLabel.SENSITIVE_SOURCE_DERIVED}),
        ),
        text="owner [REDACTED:RESTRICTED-PII]",
    )
    pii = _subject(
        subject_ref=pii_ref,
        decision=_decision(
            subject=pii_ref,
            disposition=Disposition.QUARANTINE,
            risks=frozenset({ContentRiskLabel.RESTRICTED_PII}),
        ),
        text="owner id CUST-12345",
    )

    result = EvidenceViewEngine().project(_request(subjects=(pii, redacted)))

    assert [item.source_ref for item in result.included_items] == [redacted_ref]
    assert result.excluded_subjects[0].reason is EvidenceProjectionExclusionReason.CONTENT_RISK_DENIED


def test_unknown_requested_or_source_fields_fail_closed() -> None:
    with pytest.raises(EvidenceViewPolicyError, match="requested field"):
        EvidenceViewEngine().project(_request(subjects=(_subject(),), requested_fields=("raw-trace-text",)))
    with pytest.raises(EvidenceViewPolicyError, match="source schema"):
        EvidenceViewEngine().project(
            _request(subjects=(_subject(source_schema_version="unknown/schema/v1"),))
        )


def test_ids_and_hashes_ignore_audit_timestamp() -> None:
    subject = _subject()
    first = EvidenceViewEngine().project(
        _request(subjects=(subject,), audit=_audit(datetime(2026, 7, 23, tzinfo=UTC)))
    )
    second = EvidenceViewEngine().project(
        _request(subjects=(subject,), audit=_audit(datetime(2026, 7, 24, tzinfo=UTC)))
    )

    assert first.view_result_id == second.view_result_id
    assert first.projection_policy.projection_policy_id == second.projection_policy.projection_policy_id
    assert first.included_items[0].projected_ref == second.included_items[0].projected_ref
    assert first.canonical_sha256() != second.canonical_sha256()


def test_ids_are_stable_across_python_hash_seed(tmp_path: Path) -> None:
    script = tmp_path / "check_view_seed.py"
    script.write_text(
        """
from datetime import UTC, datetime
from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.contracts.safety import Disposition, OriginClass, ProvenanceDecision, Visibility
from eval_factory.provenance import (
    EvidenceViewEngine,
    EvidenceViewPrincipal,
    EvidenceViewPrincipalType,
    EvidenceViewPurpose,
    EvidenceViewRequest,
    EvidenceViewSubject,
)

def ref(kind, object_id):
    return ObjectRef(object_type=kind, object_id=object_id, object_version='v1', object_sha256='a' * 64)

audit = ContractAudit(
    created_at=datetime(2026, 7, 23, tzinfo=UTC),
    created_by='seed-test',
    governing_versions=(VersionBinding(component='evidence-views', version='r2-05'),),
)
subject = ref('subject', 'subject://safe')
decision = ProvenanceDecision(
    provenance_decision_id='provenance-decision://safe',
    subject_ref=subject,
    origin_class=OriginClass.PREEXISTING_WORKSPACE_INPUT,
    visibility=Visibility.STAGE_PROJECTION,
    disposition=Disposition.ALLOW_INPUT_EVIDENCE,
    rule_ids=('view-parent/v1',),
    source_event_refs=(ref('trace-event', 'trace-event://safe'),),
    confidence=1.0,
    review_required=False,
    policy_version='view-parent/test-v1',
    subject_sha256='a' * 64,
    audit=audit,
)
principal = EvidenceViewPrincipal(
    principal_id='principal://default-safe',
    principal_type=EvidenceViewPrincipalType.DEFAULT_SAFE,
    allowed_purposes=frozenset({EvidenceViewPurpose.DEFAULT_SAFE}),
    max_subjects=10,
    max_characters=200,
)
request = EvidenceViewRequest(
    principal=principal,
    purpose=EvidenceViewPurpose.DEFAULT_SAFE,
    subjects=(EvidenceViewSubject(subject_ref=subject, decision=decision, projection_text='safe evidence text'),),
    max_characters=200,
    audit=audit,
)
result = EvidenceViewEngine().project(request)
print(result.view_result_id)
print(result.projection_policy.projection_policy_id)
print(result.included_items[0].projected_ref.object_id)
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
    request = _request(subjects=(_subject(),))

    with pytest.raises(ValidationError):
        EvidenceViewRequest.model_validate(
            {
                **request.model_dump(mode="json"),
                "raw_trace_text": "forbidden",
            }
        )
