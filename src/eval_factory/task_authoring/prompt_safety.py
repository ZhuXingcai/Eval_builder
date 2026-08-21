from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

from eval_factory.contracts.core import ContractAudit, Identifier, ObjectRef
from eval_factory.contracts.core_v2 import canonical_value_v2
from eval_factory.contracts.safety import ProvenanceDecision
from eval_factory.contracts.task_v2 import (
    TASK_PROMPT_SAFETY_REQUIRED_CATEGORIES_V2,
    PromptLeakageCategoryV2,
    PromptLeakageDetectorV2,
    PromptLeakageFingerprintV2,
    PromptLeakageMatchKindV2,
    PromptLeakageReferenceSetV2,
    TaskDraftPromptSafetyStatusV2,
    TaskDraftV2,
    TaskPromptSafetyCheckOutcomeV2,
    TaskPromptSafetyCheckV2,
    TaskPromptSafetyFindingV2,
    TaskPromptSafetyGateStatusV2,
    TaskPromptSafetyGateV2,
    prompt_leakage_reference_set_ref,
    task_draft_carried_sha256,
    task_draft_payload_sha256,
    task_draft_ref,
    task_prompt_safety_gate_ref,
)
from eval_factory.provenance.injection import (
    CandidateTaskPromptBoundaryRequest,
    PromptBoundaryEnforcementRequest,
    PromptBoundaryEnforcementResult,
    PromptBoundarySegment,
    PromptBoundarySourceRole,
    PromptBoundarySurface,
    PromptInjectionBoundaryEnforcer,
)
from eval_factory.task_authoring.leakage_fingerprints import (
    digest_prompt_leakage_tokens,
    match_prompt_leakage_fingerprints,
    normalize_prompt_leakage_tokens,
)
from eval_factory.task_authoring.prompt_safety_models import (
    PROMPT_LEAKAGE_FINGERPRINT_POLICY_VERSION,
    PROMPT_LEAKAGE_NORMALIZATION_VERSION,
    TASK_PROMPT_SAFETY_POLICY_VERSION,
    FakeTaskPromptSafetyFixture,
    PromptDeterministicScan,
    RestrictedPromptLeakageSource,
    TaskPromptSafetyFindingFixture,
    TaskPromptSafetyOutcome,
    TaskPromptSafetyPolicyError,
    TaskPromptSafetyProposal,
    TaskPromptSafetyReason,
    TaskPromptSafetyRequest,
    TaskPromptSafetyRequirementView,
    TaskPromptSafetyResult,
    deterministic_scan_ref,
)

_MIN_TOKEN_WINDOW = 4
_MAX_TOKEN_WINDOW = 8
_MAX_FINGERPRINTS = 20_000
_DETERMINISTIC_DETECTOR_ID = "task-prompt-safety-detector://deterministic-fingerprint"
_SEMANTIC_DETECTOR_ID = "task-prompt-safety-detector://semantic-residual"


class PromptLeakageReferenceSetCompiler:
    policy_version = PROMPT_LEAKAGE_FINGERPRINT_POLICY_VERSION

    def compile(
        self,
        *,
        trace_envelope_ref: ObjectRef,
        sources: tuple[RestrictedPromptLeakageSource, ...],
        complete_categories: frozenset[PromptLeakageCategoryV2],
        audit: ContractAudit,
    ) -> PromptLeakageReferenceSetV2:
        if trace_envelope_ref.object_type != "trace-envelope":
            raise TaskPromptSafetyPolicyError("trace_envelope_ref must reference trace-envelope")
        source_ids = tuple(item.source_id for item in sources)
        if len(source_ids) != len(set(source_ids)):
            raise TaskPromptSafetyPolicyError("duplicate restricted prompt leakage source")
        if not complete_categories.issubset(TASK_PROMPT_SAFETY_REQUIRED_CATEGORIES_V2):
            raise TaskPromptSafetyPolicyError("reference set contains unsupported complete category")
        for source in sources:
            if source.category in complete_categories and not source.capability_complete:
                raise TaskPromptSafetyPolicyError("complete category contains incomplete source capability")

        fingerprints_by_key: dict[
            tuple[
                PromptLeakageCategoryV2,
                tuple[str, str, str, str],
                PromptLeakageMatchKindV2,
                str,
                int,
            ],
            PromptLeakageFingerprintV2,
        ] = {}
        for source in sorted(sources, key=lambda item: item.source_id):
            try:
                source = RestrictedPromptLeakageSource.model_validate(source.model_dump(mode="python"))
            except ValueError as exc:
                raise TaskPromptSafetyPolicyError(str(exc)) from exc
            for fingerprint in _fingerprints_for_source(source):
                key = (
                    fingerprint.category,
                    _ref_key(fingerprint.source_subject_ref),
                    fingerprint.match_kind,
                    fingerprint.digest_sha256,
                    fingerprint.token_count,
                )
                fingerprints_by_key.setdefault(key, fingerprint)
                if len(fingerprints_by_key) > _MAX_FINGERPRINTS:
                    raise TaskPromptSafetyPolicyError("prompt leakage fingerprint budget exceeded")
        fingerprints = tuple(
            fingerprints_by_key[key]
            for key in sorted(
                fingerprints_by_key,
                key=lambda item: (
                    item[0].value,
                    item[1],
                    item[2].value,
                    item[3],
                    item[4],
                ),
            )
        )
        seed = _reference_set_seed(
            trace_envelope_ref=trace_envelope_ref,
            fingerprints=fingerprints,
            complete_categories=complete_categories,
        )
        digest = _stable_hash(seed)
        return PromptLeakageReferenceSetV2(
            reference_set_id=f"prompt-leakage-reference-set://sha256/{digest}",
            trace_envelope_ref=trace_envelope_ref,
            fingerprints=fingerprints,
            complete_categories=complete_categories,
            fingerprint_policy_version=self.policy_version,
            reference_set_sha256=digest,
            audit=_safe_audit(
                audit,
                (
                    trace_envelope_ref,
                    *(
                        ref
                        for source in sources
                        for ref in (
                            source.source_subject_ref,
                            _provenance_decision_ref(source.source_provenance_decision),
                            source.classification_evidence_ref,
                        )
                    ),
                ),
            ),
        )


class TaskPromptSafetyRequestBuilder:
    policy_version = TASK_PROMPT_SAFETY_POLICY_VERSION

    def build(
        self,
        *,
        task_draft: TaskDraftV2,
        leakage_reference_set: PromptLeakageReferenceSetV2,
        model_profile: Identifier,
        prompt_version: str,
        audit: ContractAudit,
    ) -> TaskPromptSafetyRequest:
        _validate_pending_task_draft(task_draft)
        _validate_reference_set_identity(leakage_reference_set)
        scan = _scan_prompt(
            task_draft=task_draft,
            leakage_reference_set=leakage_reference_set,
            audit=audit,
        )
        boundary = _enforce_candidate_prompt_as_data(
            task_draft=task_draft,
            leakage_reference_set=leakage_reference_set,
            audit=audit,
        )
        draft_ref = task_draft_ref(task_draft)
        reference_set_ref = prompt_leakage_reference_set_ref(leakage_reference_set)
        scan_ref = deterministic_scan_ref(scan)
        boundary_ref = _prompt_boundary_ref(boundary)
        requirement_views = tuple(
            TaskPromptSafetyRequirementView(
                requirement_id=item.requirement_id,
                statement=item.statement,
                criticality=item.criticality,
            )
            for item in sorted(
                task_draft.requirement_lineage,
                key=lambda item: item.requirement_id,
            )
        )
        seed = _request_seed(
            task_draft_ref_value=draft_ref,
            leakage_reference_set_ref=reference_set_ref,
            deterministic_scan_ref_value=scan_ref,
            prompt_boundary_enforcement_ref=boundary_ref,
            visible_prompt=task_draft.visible_prompt,
            task_intent=task_draft.task_intent,
            evaluation_claim=task_draft.evaluation_claim,
            required_capabilities=task_draft.required_capabilities,
            allowed_tools=task_draft.allowed_tools,
            forbidden_outputs=task_draft.forbidden_outputs,
            requirement_views=requirement_views,
            complete_categories=leakage_reference_set.complete_categories,
            deterministic_scan=scan,
            model_profile=model_profile,
            prompt_version=prompt_version,
        )
        digest = _stable_hash(seed)
        return TaskPromptSafetyRequest(
            request_id=f"task-prompt-safety-request://sha256/{digest}",
            task_draft_ref=draft_ref,
            leakage_reference_set_ref=reference_set_ref,
            deterministic_scan_ref=scan_ref,
            prompt_boundary_enforcement_ref=boundary_ref,
            visible_prompt=task_draft.visible_prompt,
            task_intent=task_draft.task_intent,
            evaluation_claim=task_draft.evaluation_claim,
            required_capabilities=task_draft.required_capabilities,
            allowed_tools=task_draft.allowed_tools,
            forbidden_outputs=task_draft.forbidden_outputs,
            requirement_views=requirement_views,
            complete_categories=leakage_reference_set.complete_categories,
            deterministic_scan=scan,
            model_profile=model_profile,
            prompt_version=prompt_version,
            untrusted_data_marker=True,
            policy_version=self.policy_version,
            request_sha256=digest,
            audit=_safe_audit(
                audit,
                (
                    scan_ref,
                    reference_set_ref,
                    boundary_ref,
                    draft_ref,
                ),
            ),
        )


class FakeTaskPromptSafetyRunner:
    policy_version = TASK_PROMPT_SAFETY_POLICY_VERSION

    def run(
        self,
        request: TaskPromptSafetyRequest,
        *,
        fixture: FakeTaskPromptSafetyFixture,
        audit: ContractAudit,
    ) -> TaskPromptSafetyProposal:
        _validate_request_integrity(request)
        try:
            fixture = FakeTaskPromptSafetyFixture.model_validate(fixture.model_dump(mode="python"))
        except ValueError as exc:
            raise TaskPromptSafetyPolicyError(str(exc)) from exc
        findings = tuple(
            sorted(
                fixture.findings,
                key=lambda item: (
                    item.category.value,
                    item.prompt_span_start,
                    item.prompt_span_end,
                    item.rule_id,
                ),
            )
        )
        seed = _proposal_seed(
            request_ref=_request_ref(request),
            outcome=fixture.outcome,
            findings=findings,
            unresolved_reasons=fixture.unresolved_reasons,
            model_profile=request.model_profile,
            prompt_version=request.prompt_version,
        )
        digest = _stable_hash(seed)
        return TaskPromptSafetyProposal(
            proposal_id=f"task-prompt-safety-assessment://sha256/{digest}",
            request_ref=_request_ref(request),
            outcome=fixture.outcome,
            findings=findings,
            unresolved_reasons=fixture.unresolved_reasons,
            model_profile=request.model_profile,
            prompt_version=request.prompt_version,
            policy_version=self.policy_version,
            proposal_sha256=digest,
            audit=_safe_audit(audit, (_request_ref(request),)),
        )


class TaskPromptSafetyCompiler:
    policy_version = TASK_PROMPT_SAFETY_POLICY_VERSION

    def compile(
        self,
        *,
        request: TaskPromptSafetyRequest,
        proposal: TaskPromptSafetyProposal | None,
        task_draft: TaskDraftV2,
        leakage_reference_set: PromptLeakageReferenceSetV2,
        audit: ContractAudit,
    ) -> TaskPromptSafetyResult:
        _validate_pending_task_draft(task_draft)
        _validate_reference_set_identity(leakage_reference_set)
        _validate_request_integrity(request)
        expected = TaskPromptSafetyRequestBuilder().build(
            task_draft=task_draft,
            leakage_reference_set=leakage_reference_set,
            model_profile=request.model_profile,
            prompt_version=request.prompt_version,
            audit=audit,
        )
        if expected.request_id != request.request_id or expected.request_sha256 != request.request_sha256:
            raise TaskPromptSafetyPolicyError("request no longer matches authoritative prompt safety inputs")
        if proposal is not None:
            _validate_proposal_integrity(proposal)
            if proposal.request_ref != _request_ref(request):
                raise TaskPromptSafetyPolicyError("proposal request ref is stale or mismatched")
            if proposal.model_profile != request.model_profile:
                raise TaskPromptSafetyPolicyError("proposal model profile is mismatched")
            if proposal.prompt_version != request.prompt_version:
                raise TaskPromptSafetyPolicyError("proposal prompt version is mismatched")

        deterministic_findings = request.deterministic_scan.findings
        complete = request.complete_categories.issuperset(TASK_PROMPT_SAFETY_REQUIRED_CATEGORIES_V2)
        if deterministic_findings:
            if proposal is not None and proposal.outcome is TaskPromptSafetyOutcome.PASSED:
                raise TaskPromptSafetyPolicyError(
                    "semantic proposal cannot clear deterministic prompt findings"
                )
            semantic_findings = (
                _semantic_findings(request, proposal)
                if proposal is not None and proposal.outcome is TaskPromptSafetyOutcome.BLOCKED
                else ()
            )
            return _terminal_result(
                request=request,
                proposal=proposal,
                source_draft=task_draft,
                findings=tuple(
                    sorted(
                        (*deterministic_findings, *semantic_findings),
                        key=_finding_key,
                    )
                ),
                semantic_evaluated=(
                    proposal is not None
                    and proposal.outcome
                    in {
                        TaskPromptSafetyOutcome.PASSED,
                        TaskPromptSafetyOutcome.BLOCKED,
                    }
                ),
                outcome=TaskPromptSafetyOutcome.BLOCKED,
                audit=audit,
            )

        if not complete:
            reasons = {TaskPromptSafetyReason.INCOMPLETE_REFERENCE_COVERAGE}
            if proposal is not None:
                if proposal.outcome is not TaskPromptSafetyOutcome.BLOCKED_CAPABILITY:
                    raise TaskPromptSafetyPolicyError(
                        "incomplete reference coverage cannot produce terminal prompt safety"
                    )
                reasons.update(proposal.unresolved_reasons)
            return _unresolved_result(
                request=request,
                proposal=proposal,
                outcome=TaskPromptSafetyOutcome.BLOCKED_CAPABILITY,
                reasons=frozenset(reasons),
                audit=audit,
            )

        if proposal is None:
            return _unresolved_result(
                request=request,
                proposal=None,
                outcome=TaskPromptSafetyOutcome.BLOCKED_CAPABILITY,
                reasons=frozenset({TaskPromptSafetyReason.MISSING_SEMANTIC_ASSESSMENT}),
                audit=audit,
            )
        if proposal.outcome in {
            TaskPromptSafetyOutcome.ABSTAIN,
            TaskPromptSafetyOutcome.BLOCKED_CAPABILITY,
        }:
            return _unresolved_result(
                request=request,
                proposal=proposal,
                outcome=proposal.outcome,
                reasons=proposal.unresolved_reasons,
                audit=audit,
            )
        if proposal.outcome is TaskPromptSafetyOutcome.BLOCKED:
            return _terminal_result(
                request=request,
                proposal=proposal,
                source_draft=task_draft,
                findings=_semantic_findings(request, proposal),
                semantic_evaluated=True,
                outcome=TaskPromptSafetyOutcome.BLOCKED,
                audit=audit,
            )
        return _terminal_result(
            request=request,
            proposal=proposal,
            source_draft=task_draft,
            findings=(),
            semantic_evaluated=True,
            outcome=TaskPromptSafetyOutcome.PASSED,
            audit=audit,
        )


def validate_prompt_leakage_reference_set_identity(
    reference_set: PromptLeakageReferenceSetV2,
) -> None:
    _validate_reference_set_identity(reference_set)


def validate_task_prompt_safety_gate_identity(
    gate: TaskPromptSafetyGateV2,
) -> None:
    gate_seed = {
        "source_task_draft_ref": _ref_payload(gate.source_task_draft_ref),
        "leakage_reference_set_ref": _ref_payload(gate.leakage_reference_set_ref),
        "deterministic_scan_ref": _ref_payload(gate.deterministic_scan_ref),
        "semantic_assessment_ref": _maybe_ref_payload(gate.semantic_assessment_ref),
        "visible_prompt_sha256": gate.visible_prompt_sha256,
        "status": gate.status.value,
        "checks": [item.model_dump(mode="json", exclude_none=False) for item in gate.checks],
        "findings": [item.model_dump(mode="json", exclude_none=False) for item in gate.findings],
        "policy_version": gate.policy_version,
        "model_profile": gate.model_profile,
        "prompt_version": gate.prompt_version,
    }
    observed = _stable_hash(gate_seed)
    if gate.gate_sha256 != observed:
        raise TaskPromptSafetyPolicyError("prompt safety gate carried hash is stale or invalid")
    if gate.task_prompt_safety_gate_id != (f"task-prompt-safety-gate://sha256/{observed}"):
        raise TaskPromptSafetyPolicyError("prompt safety gate carried identity is stale or invalid")


def _fingerprints_for_source(
    source: RestrictedPromptLeakageSource,
) -> tuple[PromptLeakageFingerprintV2, ...]:
    values: list[PromptLeakageFingerprintV2] = []
    if source.source_text is not None:
        tokens = normalize_prompt_leakage_tokens(source.source_text)
        if not tokens:
            raise TaskPromptSafetyPolicyError("restricted leakage source text is unscannable")
        values.append(
            _fingerprint(
                source,
                PromptLeakageMatchKindV2.NORMALIZED_FULL_TEXT,
                digest_prompt_leakage_tokens(tokens),
                len(tokens),
            )
        )
        for size in range(
            _MIN_TOKEN_WINDOW,
            min(_MAX_TOKEN_WINDOW, len(tokens)) + 1,
        ):
            for start in range(0, len(tokens) - size + 1):
                values.append(
                    _fingerprint(
                        source,
                        PromptLeakageMatchKindV2.TOKEN_WINDOW,
                        digest_prompt_leakage_tokens(tokens[start : start + size]),
                        size,
                    )
                )
    else:
        for path in source.logical_paths:
            tokens = normalize_prompt_leakage_tokens(path)
            if not tokens:
                raise TaskPromptSafetyPolicyError("restricted leakage source path is unscannable")
            values.append(
                _fingerprint(
                    source,
                    PromptLeakageMatchKindV2.PATH_COMPONENT,
                    digest_prompt_leakage_tokens(tokens),
                    len(tokens),
                )
            )
    unique: dict[
        tuple[PromptLeakageMatchKindV2, str, int],
        PromptLeakageFingerprintV2,
    ] = {}
    for item in values:
        unique.setdefault(
            (item.match_kind, item.digest_sha256, item.token_count),
            item,
        )
    return tuple(
        unique[key]
        for key in sorted(
            unique,
            key=lambda item: (item[0].value, item[1], item[2]),
        )
    )


def _fingerprint(
    source: RestrictedPromptLeakageSource,
    match_kind: PromptLeakageMatchKindV2,
    digest: str,
    token_count: int,
) -> PromptLeakageFingerprintV2:
    seed = {
        "category": source.category.value,
        "source_subject_ref": _ref_payload(source.source_subject_ref),
        "source_provenance_decision_ref": _ref_payload(
            _provenance_decision_ref(source.source_provenance_decision)
        ),
        "classification_evidence_ref": _ref_payload(source.classification_evidence_ref),
        "match_kind": match_kind.value,
        "digest_sha256": digest,
        "token_count": token_count,
        "normalization_version": PROMPT_LEAKAGE_NORMALIZATION_VERSION,
    }
    return PromptLeakageFingerprintV2(
        fingerprint_id=_stable_id("prompt-leakage-fingerprint", seed),
        category=source.category,
        source_subject_ref=source.source_subject_ref,
        source_provenance_decision_ref=_provenance_decision_ref(source.source_provenance_decision),
        classification_evidence_ref=source.classification_evidence_ref,
        match_kind=match_kind,
        digest_sha256=digest,
        token_count=token_count,
        normalization_version=PROMPT_LEAKAGE_NORMALIZATION_VERSION,
    )


def _scan_prompt(
    *,
    task_draft: TaskDraftV2,
    leakage_reference_set: PromptLeakageReferenceSetV2,
    audit: ContractAudit,
) -> PromptDeterministicScan:
    findings = tuple(
        _deterministic_finding(
            category=match.category,
            start=match.span_start,
            end=match.span_end,
            fingerprint_ids=match.fingerprint_ids,
        )
        for match in match_prompt_leakage_fingerprints(
            task_draft.visible_prompt,
            leakage_reference_set.fingerprints,
        )
    )
    draft_ref = task_draft_ref(task_draft)
    reference_set_ref = prompt_leakage_reference_set_ref(leakage_reference_set)
    prompt_hash = hashlib.sha256(task_draft.visible_prompt.encode()).hexdigest()
    seed = _scan_seed(
        task_draft_ref_value=draft_ref,
        leakage_reference_set_ref=reference_set_ref,
        visible_prompt_sha256=prompt_hash,
        complete_categories=leakage_reference_set.complete_categories,
        findings=findings,
    )
    digest = _stable_hash(seed)
    return PromptDeterministicScan(
        scan_id=f"task-prompt-deterministic-scan://sha256/{digest}",
        task_draft_ref=draft_ref,
        leakage_reference_set_ref=reference_set_ref,
        visible_prompt_sha256=prompt_hash,
        complete_categories=leakage_reference_set.complete_categories,
        findings=findings,
        policy_version=TASK_PROMPT_SAFETY_POLICY_VERSION,
        scan_sha256=digest,
        audit=_safe_audit(
            audit,
            (draft_ref, reference_set_ref),
        ),
    )


def _deterministic_finding(
    *,
    category: PromptLeakageCategoryV2,
    start: int,
    end: int,
    fingerprint_ids: tuple[str, ...],
) -> TaskPromptSafetyFindingV2:
    seed = {
        "category": category.value,
        "detector": PromptLeakageDetectorV2.DETERMINISTIC_FINGERPRINT.value,
        "prompt_span_start": start,
        "prompt_span_end": end,
        "matched_fingerprint_ids": list(fingerprint_ids),
        "rule_id": "task-prompt-safety-rule://deterministic-fingerprint",
        "non_waivable": True,
    }
    return TaskPromptSafetyFindingV2(
        finding_id=_stable_id("task-prompt-safety-finding", seed),
        category=category,
        detector=PromptLeakageDetectorV2.DETERMINISTIC_FINGERPRINT,
        prompt_span_start=start,
        prompt_span_end=end,
        matched_fingerprint_ids=fingerprint_ids,
        rule_id="task-prompt-safety-rule://deterministic-fingerprint",
        non_waivable=True,
    )


def _semantic_findings(
    request: TaskPromptSafetyRequest,
    proposal: TaskPromptSafetyProposal,
) -> tuple[TaskPromptSafetyFindingV2, ...]:
    values: list[TaskPromptSafetyFindingV2] = []
    seen: set[tuple[PromptLeakageCategoryV2, int, int, str]] = set()
    for fixture in proposal.findings:
        if fixture.prompt_span_end > len(request.visible_prompt):
            raise TaskPromptSafetyPolicyError("semantic prompt finding span exceeds visible prompt")
        key = (
            fixture.category,
            fixture.prompt_span_start,
            fixture.prompt_span_end,
            fixture.rule_id,
        )
        if key in seen:
            raise TaskPromptSafetyPolicyError("duplicate semantic prompt safety finding")
        seen.add(key)
        seed: dict[str, object] = {
            "category": fixture.category.value,
            "detector": PromptLeakageDetectorV2.SEMANTIC_RESIDUAL.value,
            "prompt_span_start": fixture.prompt_span_start,
            "prompt_span_end": fixture.prompt_span_end,
            "matched_fingerprint_ids": [],
            "rule_id": fixture.rule_id,
            "non_waivable": True,
        }
        values.append(
            TaskPromptSafetyFindingV2(
                finding_id=_stable_id("task-prompt-safety-finding", seed),
                category=fixture.category,
                detector=PromptLeakageDetectorV2.SEMANTIC_RESIDUAL,
                prompt_span_start=fixture.prompt_span_start,
                prompt_span_end=fixture.prompt_span_end,
                matched_fingerprint_ids=(),
                rule_id=fixture.rule_id,
                non_waivable=True,
            )
        )
    return tuple(sorted(values, key=_finding_key))


def _terminal_result(
    *,
    request: TaskPromptSafetyRequest,
    proposal: TaskPromptSafetyProposal | None,
    source_draft: TaskDraftV2,
    findings: tuple[TaskPromptSafetyFindingV2, ...],
    semantic_evaluated: bool,
    outcome: TaskPromptSafetyOutcome,
    audit: ContractAudit,
) -> TaskPromptSafetyResult:
    status = (
        TaskPromptSafetyGateStatusV2.PASSED
        if outcome is TaskPromptSafetyOutcome.PASSED
        else TaskPromptSafetyGateStatusV2.BLOCKED
    )
    checks = _compile_checks(
        findings=findings,
        semantic_evaluated=semantic_evaluated,
        passed=status is TaskPromptSafetyGateStatusV2.PASSED,
    )
    proposal_ref = None if proposal is None else _proposal_ref(proposal)
    gate_seed = {
        "source_task_draft_ref": _ref_payload(request.task_draft_ref),
        "leakage_reference_set_ref": _ref_payload(request.leakage_reference_set_ref),
        "deterministic_scan_ref": _ref_payload(request.deterministic_scan_ref),
        "semantic_assessment_ref": _maybe_ref_payload(proposal_ref),
        "visible_prompt_sha256": request.deterministic_scan.visible_prompt_sha256,
        "status": status.value,
        "checks": [item.model_dump(mode="json", exclude_none=False) for item in checks],
        "findings": [item.model_dump(mode="json", exclude_none=False) for item in findings],
        "policy_version": TASK_PROMPT_SAFETY_POLICY_VERSION,
        "model_profile": (proposal.model_profile if proposal is not None and semantic_evaluated else None),
        "prompt_version": (proposal.prompt_version if proposal is not None and semantic_evaluated else None),
    }
    gate_hash = _stable_hash(gate_seed)
    gate = TaskPromptSafetyGateV2(
        task_prompt_safety_gate_id=(f"task-prompt-safety-gate://sha256/{gate_hash}"),
        source_task_draft_ref=request.task_draft_ref,
        leakage_reference_set_ref=request.leakage_reference_set_ref,
        deterministic_scan_ref=request.deterministic_scan_ref,
        semantic_assessment_ref=proposal_ref if semantic_evaluated else None,
        visible_prompt_sha256=request.deterministic_scan.visible_prompt_sha256,
        status=status,
        checks=checks,
        findings=findings,
        policy_version=TASK_PROMPT_SAFETY_POLICY_VERSION,
        model_profile=(proposal.model_profile if proposal is not None and semantic_evaluated else None),
        prompt_version=(proposal.prompt_version if proposal is not None and semantic_evaluated else None),
        gate_sha256=gate_hash,
        audit=_safe_audit(
            audit,
            tuple(
                ref
                for ref in (
                    request.task_draft_ref,
                    request.leakage_reference_set_ref,
                    request.deterministic_scan_ref,
                    proposal_ref,
                )
                if ref is not None
            ),
        ),
    )
    promoted = _promote_task_draft(
        source=source_draft,
        gate=gate,
        status=(
            TaskDraftPromptSafetyStatusV2.PASSED
            if outcome is TaskPromptSafetyOutcome.PASSED
            else TaskDraftPromptSafetyStatusV2.BLOCKED
        ),
        audit=audit,
    )
    result_seed = _result_seed(
        request_ref=_request_ref(request),
        proposal_ref=proposal_ref,
        outcome=outcome,
        gate_ref=task_prompt_safety_gate_ref(gate),
        task_draft_ref_value=task_draft_ref(promoted),
        reasons=frozenset(),
    )
    digest = _stable_hash(result_seed)
    return TaskPromptSafetyResult(
        result_id=f"task-prompt-safety-result://sha256/{digest}",
        request_ref=_request_ref(request),
        proposal_ref=proposal_ref,
        outcome=outcome,
        task_prompt_safety_gate=gate,
        task_draft=promoted,
        unresolved_reasons=frozenset(),
        policy_version=TASK_PROMPT_SAFETY_POLICY_VERSION,
        result_sha256=digest,
        audit=_safe_audit(
            audit,
            (
                _request_ref(request),
                task_prompt_safety_gate_ref(gate),
                task_draft_ref(promoted),
                *((proposal_ref,) if proposal_ref is not None else ()),
            ),
        ),
    )


def _unresolved_result(
    *,
    request: TaskPromptSafetyRequest,
    proposal: TaskPromptSafetyProposal | None,
    outcome: TaskPromptSafetyOutcome,
    reasons: frozenset[TaskPromptSafetyReason],
    audit: ContractAudit,
) -> TaskPromptSafetyResult:
    proposal_ref = None if proposal is None else _proposal_ref(proposal)
    result_seed = _result_seed(
        request_ref=_request_ref(request),
        proposal_ref=proposal_ref,
        outcome=outcome,
        gate_ref=None,
        task_draft_ref_value=None,
        reasons=reasons,
    )
    digest = _stable_hash(result_seed)
    return TaskPromptSafetyResult(
        result_id=f"task-prompt-safety-result://sha256/{digest}",
        request_ref=_request_ref(request),
        proposal_ref=proposal_ref,
        outcome=outcome,
        task_prompt_safety_gate=None,
        task_draft=None,
        unresolved_reasons=reasons,
        policy_version=TASK_PROMPT_SAFETY_POLICY_VERSION,
        result_sha256=digest,
        audit=_safe_audit(
            audit,
            (
                _request_ref(request),
                *((proposal_ref,) if proposal_ref is not None else ()),
            ),
        ),
    )


def _compile_checks(
    *,
    findings: tuple[TaskPromptSafetyFindingV2, ...],
    semantic_evaluated: bool,
    passed: bool,
) -> tuple[TaskPromptSafetyCheckV2, ...]:
    findings_by_category: dict[
        PromptLeakageCategoryV2,
        list[TaskPromptSafetyFindingV2],
    ] = {}
    for finding in findings:
        findings_by_category.setdefault(finding.category, []).append(finding)
    checks: list[TaskPromptSafetyCheckV2] = []
    for category in sorted(
        TASK_PROMPT_SAFETY_REQUIRED_CATEGORIES_V2,
        key=lambda item: item.value,
    ):
        category_findings = findings_by_category.get(category, [])
        if category_findings:
            detectors = {
                (
                    _DETERMINISTIC_DETECTOR_ID
                    if item.detector is PromptLeakageDetectorV2.DETERMINISTIC_FINGERPRINT
                    else _SEMANTIC_DETECTOR_ID
                )
                for item in category_findings
            }
            checks.append(
                TaskPromptSafetyCheckV2(
                    category=category,
                    outcome=TaskPromptSafetyCheckOutcomeV2.FAILED,
                    finding_ids=tuple(sorted(item.finding_id for item in category_findings)),
                    detector_ids=tuple(sorted(detectors)),
                )
            )
        elif passed or semantic_evaluated:
            checks.append(
                TaskPromptSafetyCheckV2(
                    category=category,
                    outcome=TaskPromptSafetyCheckOutcomeV2.PASSED,
                    finding_ids=(),
                    detector_ids=(_SEMANTIC_DETECTOR_ID,),
                )
            )
        else:
            checks.append(
                TaskPromptSafetyCheckV2(
                    category=category,
                    outcome=TaskPromptSafetyCheckOutcomeV2.NOT_EVALUATED,
                    finding_ids=(),
                    detector_ids=(),
                )
            )
    return tuple(checks)


def _promote_task_draft(
    *,
    source: TaskDraftV2,
    gate: TaskPromptSafetyGateV2,
    status: TaskDraftPromptSafetyStatusV2,
    audit: ContractAudit,
) -> TaskDraftV2:
    gate_ref = task_prompt_safety_gate_ref(gate)
    source_ref = task_draft_ref(source)
    payload = {
        "task_version": source.task_version + 1,
        "supersedes_task_draft_ref": _ref_payload(source_ref),
        "selection_context_ref": _ref_payload(source.selection_context_ref),
        "task_episode_refs": [_ref_payload(ref) for ref in source.task_episode_refs],
        "visible_prompt": source.visible_prompt,
        "task_intent": source.task_intent,
        "evaluation_claim": source.evaluation_claim,
        "required_capabilities": list(source.required_capabilities),
        "allowed_tools": list(source.allowed_tools),
        "forbidden_outputs": list(source.forbidden_outputs),
        "attachment_dependencies": [
            item.model_dump(mode="json", exclude_none=False) for item in source.attachment_dependencies
        ],
        "requirement_lineage": [
            item.model_dump(mode="json", exclude_none=False) for item in source.requirement_lineage
        ],
        "prompt_requirement_ids": list(source.prompt_requirement_ids),
        "uncertainties": list(source.uncertainties),
        "prompt_safety_status": status.value,
        "prompt_safety_gate_ref": _ref_payload(gate_ref),
        "model_profile": source.model_profile,
        "prompt_version": source.prompt_version,
        "policy_version": source.policy_version,
    }
    digest = task_draft_payload_sha256(payload)
    return TaskDraftV2(
        task_draft_id=f"task-draft://sha256/{digest}",
        task_version=source.task_version + 1,
        supersedes_task_draft_ref=source_ref,
        selection_context_ref=source.selection_context_ref,
        task_episode_refs=source.task_episode_refs,
        visible_prompt=source.visible_prompt,
        task_intent=source.task_intent,
        evaluation_claim=source.evaluation_claim,
        required_capabilities=source.required_capabilities,
        allowed_tools=source.allowed_tools,
        forbidden_outputs=source.forbidden_outputs,
        attachment_dependencies=source.attachment_dependencies,
        requirement_lineage=source.requirement_lineage,
        prompt_requirement_ids=source.prompt_requirement_ids,
        uncertainties=source.uncertainties,
        prompt_safety_status=status,
        prompt_safety_gate_ref=gate_ref,
        model_profile=source.model_profile,
        prompt_version=source.prompt_version,
        policy_version=source.policy_version,
        task_draft_sha256=digest,
        audit=_safe_audit(audit, (source_ref, gate_ref)),
    )


def _enforce_candidate_prompt_as_data(
    *,
    task_draft: TaskDraftV2,
    leakage_reference_set: PromptLeakageReferenceSetV2,
    audit: ContractAudit,
) -> PromptBoundaryEnforcementResult:
    prompt_hash = hashlib.sha256(task_draft.visible_prompt.encode()).hexdigest()
    prompt_ref = ObjectRef(
        object_type="task-draft-visible-prompt",
        object_id=f"task-draft-visible-prompt://sha256/{prompt_hash}",
        object_version="v2",
        object_sha256=prompt_hash,
    )
    segment = PromptBoundarySegment(
        segment_id=f"prompt-boundary-segment://task-prompt/{prompt_hash}",
        surface=PromptBoundarySurface.SAFETY_REVIEW_DATA,
        source_role=PromptBoundarySourceRole.CANDIDATE_TASK_TEXT,
        source_ref=prompt_ref,
        content_sha256=prompt_hash,
        untrusted_data_marker=True,
    )
    boundary_request = PromptBoundaryEnforcementRequest(
        boundary_id=_stable_id(
            "prompt-boundary",
            {
                "task_draft_ref": _ref_payload(task_draft_ref(task_draft)),
                "leakage_reference_set_ref": _ref_payload(
                    prompt_leakage_reference_set_ref(leakage_reference_set)
                ),
                "prompt_ref": _ref_payload(prompt_ref),
            },
        ),
        segments=(segment,),
        audit=_safe_audit(
            audit,
            (
                task_draft_ref(task_draft),
                prompt_leakage_reference_set_ref(leakage_reference_set),
                prompt_ref,
            ),
        ),
    )
    return PromptInjectionBoundaryEnforcer().validate_candidate_task_prompt_boundary(
        CandidateTaskPromptBoundaryRequest(
            task_draft_ref=task_draft_ref(task_draft),
            visible_prompt_ref=prompt_ref,
            boundary_request=boundary_request,
            prompt_segment_id=segment.segment_id,
        )
    )


def _validate_pending_task_draft(draft: TaskDraftV2) -> None:
    if draft.prompt_safety_status is not TaskDraftPromptSafetyStatusV2.PENDING:
        raise TaskPromptSafetyPolicyError("prompt safety requires a pending TaskDraft")
    observed = task_draft_carried_sha256(draft)
    if draft.task_draft_sha256 != observed:
        raise TaskPromptSafetyPolicyError("TaskDraft carried hash is stale or invalid")
    if draft.task_draft_id != f"task-draft://sha256/{observed}":
        raise TaskPromptSafetyPolicyError("TaskDraft carried identity is stale or invalid")


def _validate_reference_set_identity(
    reference_set: PromptLeakageReferenceSetV2,
) -> None:
    seed = _reference_set_seed(
        trace_envelope_ref=reference_set.trace_envelope_ref,
        fingerprints=reference_set.fingerprints,
        complete_categories=reference_set.complete_categories,
    )
    observed = _stable_hash(seed)
    if reference_set.reference_set_sha256 != observed:
        raise TaskPromptSafetyPolicyError("reference set carried hash is stale or invalid")
    if reference_set.reference_set_id != f"prompt-leakage-reference-set://sha256/{observed}":
        raise TaskPromptSafetyPolicyError("reference set carried identity is stale or invalid")


def _validate_request_integrity(request: TaskPromptSafetyRequest) -> None:
    seed = _request_seed(
        task_draft_ref_value=request.task_draft_ref,
        leakage_reference_set_ref=request.leakage_reference_set_ref,
        deterministic_scan_ref_value=request.deterministic_scan_ref,
        prompt_boundary_enforcement_ref=request.prompt_boundary_enforcement_ref,
        visible_prompt=request.visible_prompt,
        task_intent=request.task_intent,
        evaluation_claim=request.evaluation_claim,
        required_capabilities=request.required_capabilities,
        allowed_tools=request.allowed_tools,
        forbidden_outputs=request.forbidden_outputs,
        requirement_views=request.requirement_views,
        complete_categories=request.complete_categories,
        deterministic_scan=request.deterministic_scan,
        model_profile=request.model_profile,
        prompt_version=request.prompt_version,
    )
    observed = _stable_hash(seed)
    if request.request_sha256 != observed:
        raise TaskPromptSafetyPolicyError("request carried hash is stale or invalid")
    if request.request_id != f"task-prompt-safety-request://sha256/{observed}":
        raise TaskPromptSafetyPolicyError("request identity is stale or invalid")
    scan_seed = _scan_seed(
        task_draft_ref_value=request.deterministic_scan.task_draft_ref,
        leakage_reference_set_ref=(request.deterministic_scan.leakage_reference_set_ref),
        visible_prompt_sha256=request.deterministic_scan.visible_prompt_sha256,
        complete_categories=request.deterministic_scan.complete_categories,
        findings=request.deterministic_scan.findings,
    )
    scan_hash = _stable_hash(scan_seed)
    if request.deterministic_scan.scan_sha256 != scan_hash:
        raise TaskPromptSafetyPolicyError("deterministic scan carried hash is stale or invalid")
    if request.deterministic_scan.scan_id != f"task-prompt-deterministic-scan://sha256/{scan_hash}":
        raise TaskPromptSafetyPolicyError("deterministic scan identity is stale or invalid")


def _validate_proposal_integrity(
    proposal: TaskPromptSafetyProposal,
) -> None:
    seed = _proposal_seed(
        request_ref=proposal.request_ref,
        outcome=proposal.outcome,
        findings=proposal.findings,
        unresolved_reasons=proposal.unresolved_reasons,
        model_profile=proposal.model_profile,
        prompt_version=proposal.prompt_version,
    )
    observed = _stable_hash(seed)
    if proposal.proposal_sha256 != observed:
        raise TaskPromptSafetyPolicyError("proposal carried hash is stale or invalid")
    if proposal.proposal_id != f"task-prompt-safety-assessment://sha256/{observed}":
        raise TaskPromptSafetyPolicyError("proposal identity is stale or invalid")


def _reference_set_seed(
    *,
    trace_envelope_ref: ObjectRef,
    fingerprints: tuple[PromptLeakageFingerprintV2, ...],
    complete_categories: frozenset[PromptLeakageCategoryV2],
) -> dict[str, object]:
    return {
        "trace_envelope_ref": _ref_payload(trace_envelope_ref),
        "fingerprints": [item.model_dump(mode="json", exclude_none=False) for item in fingerprints],
        "complete_categories": sorted(item.value for item in complete_categories),
        "fingerprint_policy_version": (PROMPT_LEAKAGE_FINGERPRINT_POLICY_VERSION),
    }


def _scan_seed(
    *,
    task_draft_ref_value: ObjectRef,
    leakage_reference_set_ref: ObjectRef,
    visible_prompt_sha256: str,
    complete_categories: frozenset[PromptLeakageCategoryV2],
    findings: tuple[TaskPromptSafetyFindingV2, ...],
) -> dict[str, object]:
    return {
        "task_draft_ref": _ref_payload(task_draft_ref_value),
        "leakage_reference_set_ref": _ref_payload(leakage_reference_set_ref),
        "visible_prompt_sha256": visible_prompt_sha256,
        "complete_categories": sorted(item.value for item in complete_categories),
        "findings": [item.model_dump(mode="json", exclude_none=False) for item in findings],
        "policy_version": TASK_PROMPT_SAFETY_POLICY_VERSION,
    }


def _request_seed(
    *,
    task_draft_ref_value: ObjectRef,
    leakage_reference_set_ref: ObjectRef,
    deterministic_scan_ref_value: ObjectRef,
    prompt_boundary_enforcement_ref: ObjectRef,
    visible_prompt: str,
    task_intent: str,
    evaluation_claim: str,
    required_capabilities: tuple[str, ...],
    allowed_tools: tuple[str, ...],
    forbidden_outputs: tuple[str, ...],
    requirement_views: tuple[TaskPromptSafetyRequirementView, ...],
    complete_categories: frozenset[PromptLeakageCategoryV2],
    deterministic_scan: PromptDeterministicScan,
    model_profile: str,
    prompt_version: str,
) -> dict[str, object]:
    deterministic_scan_payload = deterministic_scan.model_dump(
        mode="json",
        exclude={"audit"},
        exclude_none=False,
    )
    deterministic_scan_payload["complete_categories"] = sorted(
        item.value for item in deterministic_scan.complete_categories
    )
    return {
        "task_draft_ref": _ref_payload(task_draft_ref_value),
        "leakage_reference_set_ref": _ref_payload(leakage_reference_set_ref),
        "deterministic_scan_ref": _ref_payload(deterministic_scan_ref_value),
        "prompt_boundary_enforcement_ref": _ref_payload(prompt_boundary_enforcement_ref),
        "visible_prompt": visible_prompt,
        "task_intent": task_intent,
        "evaluation_claim": evaluation_claim,
        "required_capabilities": list(required_capabilities),
        "allowed_tools": list(allowed_tools),
        "forbidden_outputs": list(forbidden_outputs),
        "requirement_views": [item.model_dump(mode="json", exclude_none=False) for item in requirement_views],
        "complete_categories": sorted(item.value for item in complete_categories),
        "deterministic_scan": deterministic_scan_payload,
        "model_profile": model_profile,
        "prompt_version": prompt_version,
        "untrusted_data_marker": True,
        "policy_version": TASK_PROMPT_SAFETY_POLICY_VERSION,
    }


def _proposal_seed(
    *,
    request_ref: ObjectRef,
    outcome: TaskPromptSafetyOutcome,
    findings: tuple[TaskPromptSafetyFindingFixture, ...],
    unresolved_reasons: frozenset[TaskPromptSafetyReason],
    model_profile: str,
    prompt_version: str,
) -> dict[str, object]:
    return {
        "request_ref": _ref_payload(request_ref),
        "outcome": outcome.value,
        "findings": [item.model_dump(mode="json", exclude_none=False) for item in findings],
        "unresolved_reasons": sorted(item.value for item in unresolved_reasons),
        "model_profile": model_profile,
        "prompt_version": prompt_version,
        "policy_version": TASK_PROMPT_SAFETY_POLICY_VERSION,
    }


def _result_seed(
    *,
    request_ref: ObjectRef,
    proposal_ref: ObjectRef | None,
    outcome: TaskPromptSafetyOutcome,
    gate_ref: ObjectRef | None,
    task_draft_ref_value: ObjectRef | None,
    reasons: frozenset[TaskPromptSafetyReason],
) -> dict[str, object]:
    return {
        "request_ref": _ref_payload(request_ref),
        "proposal_ref": _maybe_ref_payload(proposal_ref),
        "outcome": outcome.value,
        "gate_ref": _maybe_ref_payload(gate_ref),
        "task_draft_ref": _maybe_ref_payload(task_draft_ref_value),
        "unresolved_reasons": sorted(item.value for item in reasons),
        "policy_version": TASK_PROMPT_SAFETY_POLICY_VERSION,
    }


def _finding_key(
    finding: TaskPromptSafetyFindingV2,
) -> tuple[str, int, int, str]:
    return (
        finding.category.value,
        finding.prompt_span_start,
        finding.prompt_span_end,
        finding.finding_id,
    )


def _provenance_decision_ref(decision: ProvenanceDecision) -> ObjectRef:
    payload = decision.model_dump(
        mode="json",
        exclude={"audit"},
        exclude_none=False,
    )
    return ObjectRef(
        object_type="provenance-decision",
        object_id=decision.provenance_decision_id,
        object_version=decision.policy_version,
        object_sha256=_stable_hash(payload),
    )


def _prompt_boundary_ref(
    result: PromptBoundaryEnforcementResult,
) -> ObjectRef:
    return ObjectRef(
        object_type="prompt-boundary-enforcement",
        object_id=result.enforcement_id,
        object_version=result.policy_version,
        object_sha256=result.enforcement_sha256,
    )


def _request_ref(request: TaskPromptSafetyRequest) -> ObjectRef:
    return ObjectRef(
        object_type="task-prompt-safety-request",
        object_id=request.request_id,
        object_version=request.policy_version,
        object_sha256=request.request_sha256,
    )


def _proposal_ref(proposal: TaskPromptSafetyProposal) -> ObjectRef:
    return ObjectRef(
        object_type="task-prompt-safety-assessment",
        object_id=proposal.proposal_id,
        object_version=proposal.policy_version,
        object_sha256=proposal.proposal_sha256,
    )


def _safe_audit(
    audit: ContractAudit,
    refs: tuple[ObjectRef, ...],
) -> ContractAudit:
    unique = {_ref_key(ref): ref for ref in refs}
    return ContractAudit(
        created_at=audit.created_at,
        created_by=audit.created_by,
        governing_versions=audit.governing_versions,
        input_refs=tuple(unique[key] for key in sorted(unique)),
    )


def _ref_payload(ref: ObjectRef) -> dict[str, object]:
    return ref.model_dump(mode="json", exclude_none=False)


def _maybe_ref_payload(ref: ObjectRef | None) -> dict[str, object] | None:
    if ref is None:
        return None
    return _ref_payload(ref)


def _ref_key(ref: ObjectRef) -> tuple[str, str, str, str]:
    return (
        ref.object_type,
        ref.object_id,
        ref.object_version,
        ref.object_sha256,
    )


def _stable_hash(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        canonical_value_v2(dict(payload)),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _stable_id(kind: str, payload: Mapping[str, object]) -> str:
    return f"{kind}://sha256/{_stable_hash(payload)}"
