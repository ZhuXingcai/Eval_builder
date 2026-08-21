from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
from test_cross_item_safety import (
    _passed_fixture,
    _ScanFacade,
)
from test_cross_item_safety import (
    _policy as _cross_item_policy,
)
from test_duplicate_detection import (
    _FingerprintFacade,
    _graph,
    _quality_result,
    _task_contract_set,
)
from test_duplicate_detection import (
    _policy as _duplicate_policy,
)
from test_task_prompt_safety import (
    _compile as _compile_prompt_safety,
)
from test_task_prompt_safety import (
    _pending_draft,
)
from test_task_prompt_safety import (
    _source as _restricted_source,
)

from eval_factory.batch_quality.cross_item_safety import (
    CrossItemSafetyCompiler,
    CrossItemSafetyItemSource,
)
from eval_factory.batch_quality.duplicates import (
    DuplicateDetectionCompiler,
    DuplicateDetectionItemSource,
)
from eval_factory.batch_quality.reports import (
    BatchQualityCompiler,
    BatchQualityItemSource,
    BatchQualityPolicyError,
)
from eval_factory.contracts.batch_quality_v2 import (
    BatchFindingCategoryV2,
    BatchFindingScopeV2,
    BatchQualityPolicyV2,
    ItemLineageAuditOutcomeV2,
    LineageAuditCheckCodeV2,
    LineageAuditCheckOutcomeV2,
    batch_quality_report_v2_ref,
    lineage_audit_check_v2_ref,
)
from eval_factory.contracts.core import (
    ContractAudit,
    EvidencePolarity,
    EvidenceRef,
    ObjectRef,
    SourceSpanRef,
    VersionBinding,
)
from eval_factory.contracts.labeling_v2 import (
    LabelDecisionV2,
    LabelDecisionValueV2,
    LabelExecutionStatus,
    SelectionContextV2,
    label_decision_ref,
    selection_context_ref,
)
from eval_factory.contracts.quality import Severity
from eval_factory.contracts.quality_v2 import environment_artifact_ref
from eval_factory.contracts.task_v2 import (
    TASK_PROMPT_SAFETY_REQUIRED_CATEGORIES_V2,
    PromptLeakageCategoryV2,
    task_draft_carried_sha256,
)
from eval_factory.contracts.trace import (
    CapabilityStatus,
    ParseQuality,
    TraceCapability,
    TraceEnvelope,
)
from eval_factory.labeling import LABEL_DECISION_MERGE_POLICY_VERSION
from eval_factory.task_authoring import (
    PROMPT_LEAKAGE_FINGERPRINT_POLICY_VERSION,
    SELECTION_CONTEXT_FIREWALL_POLICY_VERSION,
    PromptLeakageReferenceSetCompiler,
)

NOW = datetime(2026, 8, 1, tzinfo=UTC)
GOLD_PATH = (
    Path(__file__).resolve().parents[3]
    / "evals/golden/eval_factory/batch_quality"
    / "r7-03-finding-ownership-v1.json"
)


def _digest(seed: str) -> str:
    return hashlib.sha256(seed.encode()).hexdigest()


def _stable_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _ref(
    object_type: str,
    suffix: str,
    *,
    version: str = "v2",
    digest: str | None = None,
) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://r7-03/{suffix}",
        object_version=version,
        object_sha256=digest or _digest(f"{object_type}:{suffix}:{version}"),
    )


def _audit(
    *,
    created_at: datetime = NOW,
    created_by: str = "batch-quality-test",
) -> ContractAudit:
    return ContractAudit(
        created_at=created_at,
        created_by=created_by,
        governing_versions=(
            VersionBinding(
                component="batch-quality",
                version="r7-03-v1",
            ),
        ),
        input_refs=(),
    )


def _policy(
    *,
    max_findings: int = 100_000,
    max_direct_evidence_refs: int = 100_000,
) -> BatchQualityPolicyV2:
    return BatchQualityPolicyV2.create(
        max_items=100,
        max_label_decisions=10_000,
        max_package_members=100_000,
        max_lineage_checks=10_000,
        max_direct_evidence_refs=max_direct_evidence_refs,
        max_findings=max_findings,
        max_finding_subject_refs=1_000_000,
        max_report_revisions=100,
        max_blocked_items=100,
        audit=_audit(),
    )


def _trace_envelope(source_ref: ObjectRef, *, suffix: str) -> TraceEnvelope:
    return TraceEnvelope(
        source_trace_id=source_ref.object_id,
        trace_ir_version_id=f"trace-ir://r7-03/{suffix}",
        source_uri=f"file:///restricted/{suffix}.json",
        raw_sha256=source_ref.object_sha256,
        adapter_name="raw-traj-v1",
        adapter_version=source_ref.object_version,
        trace_ir_schema_version="v1",
        repair_policy_version="trace-repair/r1-02-v1",
        segmentation_policy_version="trace-segmentation/r1-04-v1",
        parse_quality=ParseQuality.STRICT,
        capabilities=(
            TraceCapability(
                capability="conversation_events",
                status=CapabilityStatus.COMPLETE,
            ),
        ),
        event_refs=(),
        tool_call_refs=(),
        file_observation_refs=(),
        segment_refs=(),
        repair_map_refs=(),
        unrecoverable_span_refs=(),
        audit=_audit(),
    )


def _trace_envelope_ref(envelope: TraceEnvelope) -> ObjectRef:
    return ObjectRef(
        object_type="trace-envelope",
        object_id=envelope.trace_ir_version_id,
        object_version="stored-manifest/v1",
        object_sha256=envelope.canonical_sha256(),
    )


def _label_decision(
    envelope: TraceEnvelope,
    *,
    suffix: str,
) -> LabelDecisionV2:
    envelope_ref = _trace_envelope_ref(envelope)
    evidence = EvidenceRef(
        evidence_ref_id=f"evidence-ref://r7-03/{suffix}",
        subject_ref=_ref("interaction-segment", suffix),
        source_spans=(
            SourceSpanRef(
                span_id=f"source-span://r7-03/{suffix}",
                source_trace_id=envelope.source_trace_id,
                raw_sha256=envelope.raw_sha256,
            ),
        ),
        polarity=EvidencePolarity.POSITIVE,
        capability="batch-lineage",
        capability_complete=True,
    )
    decision_sha256 = _digest(f"label-decision:{suffix}")
    return LabelDecisionV2(
        label_decision_id=f"label-decision://sha256/{decision_sha256}",
        label_spec_ref=_ref("label-spec", suffix),
        trace_envelope_ref=envelope_ref,
        decision=LabelDecisionValueV2.MATCH,
        execution_status=LabelExecutionStatus.FINAL,
        positive_evidence=(evidence,),
        negative_evidence=(),
        semantic_evidence=(),
        structured_capability_complete=True,
        confidence=1.0,
        rule_version="structured-labeling/r3-02-v1",
        model_profile=None,
        prompt_version=None,
        unresolved_reasons=frozenset(),
        policy_version=LABEL_DECISION_MERGE_POLICY_VERSION,
        decision_sha256=decision_sha256,
        audit=_audit(),
    )


def _selection_context(
    envelope: TraceEnvelope,
    decision: LabelDecisionV2,
    *,
    suffix: str,
) -> SelectionContextV2:
    decision_refs = (label_decision_ref(decision),)
    envelope_ref = _trace_envelope_ref(envelope)
    candidate_seed = {
        "trace_envelope_ref": envelope_ref.model_dump(
            mode="json",
            exclude_none=False,
        ),
        "approved_label_decision_refs": [
            ref.model_dump(mode="json", exclude_none=False) for ref in decision_refs
        ],
        "policy_version": SELECTION_CONTEXT_FIREWALL_POLICY_VERSION,
    }
    candidate_id = f"candidate://sha256/{_stable_hash(candidate_seed)}"
    bundle_ref = _ref("evidence-bundle", suffix)
    projection_ref = _ref("projection-policy", suffix)
    excluded = tuple(
        sorted(
            {
                decision.decision_sha256,
                decision.label_spec_ref.object_sha256,
                decision.positive_evidence[0].subject_ref.object_sha256,
            }
        )
    )
    context_seed = {
        "candidate_id": candidate_id,
        "approved_label_decision_refs": [
            ref.model_dump(mode="json", exclude_none=False) for ref in decision_refs
        ],
        "safe_evidence_bundle_ref": bundle_ref.model_dump(
            mode="json",
            exclude_none=False,
        ),
        "task_authoring_note_refs": [],
        "excluded_signal_hashes": list(excluded),
        "projection_policy_ref": projection_ref.model_dump(
            mode="json",
            exclude_none=False,
        ),
        "policy_version": SELECTION_CONTEXT_FIREWALL_POLICY_VERSION,
    }
    context_sha256 = _stable_hash(context_seed)
    return SelectionContextV2(
        selection_context_id=f"selection-context://sha256/{context_sha256}",
        candidate_id=candidate_id,
        approved_label_decision_refs=decision_refs,
        safe_evidence_bundle_ref=bundle_ref,
        task_authoring_note_refs=(),
        excluded_signal_hashes=excluded,
        projection_policy_ref=projection_ref,
        policy_version=SELECTION_CONTEXT_FIREWALL_POLICY_VERSION,
        selection_context_sha256=context_sha256,
        audit=_audit(),
    )


def _source(
    *,
    item_id: str,
    source_trace_ref: ObjectRef,
    suffix: str,
    prompt: str,
    category: PromptLeakageCategoryV2,
    restricted_text: str,
    attachments: tuple[tuple[str, str, str, str], ...] = (),
) -> BatchQualityItemSource:
    envelope = _trace_envelope(source_trace_ref, suffix=suffix)
    envelope_ref = _trace_envelope_ref(envelope)
    label = _label_decision(envelope, suffix=suffix)
    selection = _selection_context(
        envelope,
        label,
        suffix=suffix,
    )
    pending = _pending_draft(visible_prompt=prompt).model_copy(
        update={"selection_context_ref": selection_context_ref(selection)}
    )
    pending_digest = task_draft_carried_sha256(pending)
    pending = pending.model_copy(
        update={
            "task_draft_id": f"task-draft://sha256/{pending_digest}",
            "task_draft_sha256": pending_digest,
        }
    )
    restricted_source = _restricted_source(
        category,
        text=restricted_text,
        suffix=f"r7-03-{suffix}",
    )
    reference_set = PromptLeakageReferenceSetCompiler().compile(
        trace_envelope_ref=envelope_ref,
        sources=(restricted_source,),
        complete_categories=TASK_PROMPT_SAFETY_REQUIRED_CATEGORIES_V2,
        audit=_audit(),
    )
    assert reference_set.fingerprint_policy_version == PROMPT_LEAKAGE_FINGERPRINT_POLICY_VERSION
    _, _, prompt_result = _compile_prompt_safety(
        pending,
        reference_set,
        _passed_fixture(),
    )
    assert prompt_result.task_draft is not None
    assert prompt_result.task_prompt_safety_gate is not None
    draft = prompt_result.task_draft
    contract_set = _task_contract_set(
        draft,
        suffix=f"r7-03-{suffix}",
    )
    quality = _quality_result(
        suffix=f"r7-03-{suffix}",
        contract_set=contract_set,
        attachments=attachments,
    )
    return BatchQualityItemSource(
        item_id=item_id,
        source_trace_ref=source_trace_ref,
        trace_envelope=envelope,
        label_decisions=(label,),
        selection_context=selection,
        task_draft=draft,
        task_prompt_safety_gate=prompt_result.task_prompt_safety_gate,
        leakage_reference_set=reference_set,
        task_contract_set=contract_set,
        item_quality=quality,
    )


async def _inputs(
    *,
    prompts: tuple[str, str],
    restricted_texts: tuple[str, str],
    categories: tuple[
        PromptLeakageCategoryV2,
        PromptLeakageCategoryV2,
    ] = (
        PromptLeakageCategoryV2.FINAL_ANSWER,
        PromptLeakageCategoryV2.PRIVATE_REFERENCE,
    ),
    duplicate_task_threshold: int = 10_000,
    duplicate_attachment_threshold: int = 10_000,
    attachments: tuple[
        tuple[tuple[str, str, str, str], ...],
        tuple[tuple[str, str, str, str], ...],
    ] = ((), ()),
    attachment_fingerprints: tuple[
        tuple[str | None, ...],
        tuple[str | None, ...],
    ]
    | None = None,
):
    graph, item_ids = _graph(2)
    sources = tuple(
        _source(
            item_id=item_id,
            source_trace_ref=source_ref,
            suffix=f"item-{index}",
            prompt=prompts[index],
            category=categories[index],
            restricted_text=restricted_texts[index],
            attachments=attachments[index],
        )
        for index, (item_id, source_ref) in enumerate(
            zip(
                item_ids,
                graph.source_trace_refs,
                strict=True,
            )
        )
    )
    duplicate_policy = _duplicate_policy(
        task_threshold=duplicate_task_threshold,
        attachment_threshold=duplicate_attachment_threshold,
    )
    configured_fingerprints: dict[str, str | None] = {}
    if attachment_fingerprints is not None:
        for source, expected_fingerprints in zip(
            sources,
            attachment_fingerprints,
            strict=True,
        ):
            environment = source.item_quality.environment_spec
            assert environment is not None
            if len(environment.artifacts) != len(expected_fingerprints):
                raise AssertionError("attachment fingerprint fixture length differs")
            configured_fingerprints.update(
                {
                    environment_artifact_ref(artifact).object_id: fingerprint
                    for artifact, fingerprint in zip(
                        environment.artifacts,
                        expected_fingerprints,
                        strict=True,
                    )
                }
            )
    duplicate_result = await DuplicateDetectionCompiler().compile(
        resolved_job_work_graph=graph,
        policy=duplicate_policy,
        sources=tuple(
            DuplicateDetectionItemSource(
                item_id=source.item_id,
                task_draft=source.task_draft,
                task_contract_set=source.task_contract_set,
                item_quality=source.item_quality,
            )
            for source in sources
        ),
        facade=_FingerprintFacade(
            fingerprints=configured_fingerprints,
        ),
        audit=_audit(),
    )
    cross_policy = _cross_item_policy()
    cross_result = await CrossItemSafetyCompiler().compile(
        resolved_job_work_graph=graph,
        policy=cross_policy,
        sources=tuple(
            CrossItemSafetyItemSource(
                item_id=source.item_id,
                task_draft=source.task_draft,
                task_prompt_safety_gate=source.task_prompt_safety_gate,
                leakage_reference_set=source.leakage_reference_set,
                task_contract_set=source.task_contract_set,
                item_quality=source.item_quality,
            )
            for source in sources
        ),
        facade=_ScanFacade(),
        audit=_audit(),
    )
    return (
        graph,
        item_ids,
        sources,
        duplicate_policy,
        duplicate_result,
        cross_policy,
        cross_result,
    )


def _compile(
    inputs,
    *,
    sources: tuple[BatchQualityItemSource, ...] | None = None,
    policy: BatchQualityPolicyV2 | None = None,
    audit: ContractAudit | None = None,
):
    (
        graph,
        _,
        input_sources,
        duplicate_policy,
        duplicate_result,
        cross_policy,
        cross_result,
    ) = inputs
    return BatchQualityCompiler().compile(
        resolved_job_work_graph=graph,
        policy=policy or _policy(),
        sources=sources or input_sources,
        duplicate_policy=duplicate_policy,
        duplicate_result=duplicate_result,
        cross_item_policy=cross_policy,
        cross_item_result=cross_result,
        audit=audit or _audit(),
    )


@pytest.mark.asyncio
async def test_clean_batch_has_complete_lineage_and_is_approvable() -> None:
    inputs = await _inputs(
        prompts=(
            "Inspect alpha workspace inputs and summarize the design.",
            "Analyze beta workspace requirements and report the structure.",
        ),
        restricted_texts=(
            "alpha answer one two three four five six seven eight",
            "beta private reference nine ten eleven twelve thirteen fourteen",
        ),
    )

    report = _compile(inputs)

    assert report.approvable is True
    assert report.findings == ()
    assert report.item_quality_report_revisions == ()
    assert report.blocked_item_ids == ()
    assert all(value.outcome is ItemLineageAuditOutcomeV2.PASSED for value in report.lineage_audits)
    assert all(
        next(
            check for check in value.checks if check.code is LineageAuditCheckCodeV2.PACKAGE_MEMBER_DERIVATION
        ).outcome.value
        == "SKIPPED"
        for value in report.lineage_audits
    )


@pytest.mark.asyncio
async def test_exact_task_duplicate_is_one_p1_batch_finding() -> None:
    prompt = "Inspect the supplied workspace and summarize its design."
    inputs = await _inputs(
        prompts=(prompt, prompt),
        restricted_texts=(
            "alpha answer one two three four five six seven eight",
            "beta private reference nine ten eleven twelve thirteen fourteen",
        ),
    )

    report = _compile(inputs)

    assert report.approvable is False
    assert report.open_p1_count == 1
    assert report.open_p0_count == 0
    assert report.item_quality_report_revisions == ()
    assert len(report.findings) == 1
    finding = report.findings[0]
    assert finding.category is BatchFindingCategoryV2.EXACT_TASK_DUPLICATE
    assert finding.scope is BatchFindingScopeV2.BATCH
    assert finding.severity is Severity.P1
    assert finding.non_waivable is False
    assert finding.blocked_item_ids == tuple(sorted(inputs[1]))


@pytest.mark.asyncio
async def test_near_task_duplicate_is_p2_and_batch_remains_approvable() -> None:
    inputs = await _inputs(
        prompts=(
            "Inspect the alpha workspace and summarize its design.",
            "Calculate the beta ledger and explain monthly totals.",
        ),
        restricted_texts=(
            "alpha answer one two three four five six seven eight",
            "beta private reference nine ten eleven twelve thirteen fourteen",
        ),
        duplicate_task_threshold=0,
    )

    report = _compile(inputs)

    assert report.approvable is True
    assert report.open_p0_count == 0
    assert report.open_p1_count == 0
    assert report.open_p2_count == 1
    assert report.blocked_item_ids == ()
    assert report.item_quality_report_revisions == ()
    assert len(report.findings) == 1
    finding = report.findings[0]
    assert finding.category is BatchFindingCategoryV2.NEAR_TASK_DUPLICATE
    assert finding.scope is BatchFindingScopeV2.BATCH
    assert finding.severity is Severity.P2
    assert finding.release_blocking is False
    assert finding.blocked_item_ids == ()


@pytest.mark.asyncio
async def test_exact_and_near_attachment_duplicates_are_p2_nonblockers() -> None:
    shared_content = _digest("shared attachment content")
    inputs = await _inputs(
        prompts=(
            "Inspect the alpha attachment and report its structure.",
            "Analyze the beta attachment and report its layout.",
        ),
        restricted_texts=(
            "alpha answer one two three four five six seven eight",
            "beta private reference nine ten eleven twelve thirteen fourteen",
        ),
        duplicate_attachment_threshold=9_000,
        attachments=(
            (
                ("exact", "inputs/exact-a.txt", "text/plain", shared_content),
                ("near", "inputs/near-a.txt", "text/plain", _digest("near-a")),
            ),
            (
                ("exact", "inputs/exact-b.txt", "text/plain", shared_content),
                ("near", "inputs/near-b.txt", "text/plain", _digest("near-b")),
            ),
        ),
        attachment_fingerprints=(
            ("f" * 64, "0" * 64),
            ("f" * 64, "f" + "0" * 63),
        ),
    )

    report = _compile(inputs)

    categories = tuple(finding.category for finding in report.findings)
    assert categories.count(BatchFindingCategoryV2.EXACT_ATTACHMENT_DUPLICATE) == 1
    assert categories.count(BatchFindingCategoryV2.NEAR_ATTACHMENT_DUPLICATE) == 1
    assert report.approvable is True
    assert report.open_p2_count == 2
    assert report.blocked_item_ids == ()
    assert report.item_quality_report_revisions == ()
    assert all(finding.scope is BatchFindingScopeV2.BATCH for finding in report.findings)
    assert all(finding.release_blocking is False for finding in report.findings)


@pytest.mark.asyncio
async def test_duplicate_cluster_navigation_does_not_fabricate_finding() -> None:
    inputs = await _inputs(
        prompts=(
            "Inspect the alpha attachment chain and report its structure.",
            "Analyze the beta attachment chain and report its layout.",
        ),
        restricted_texts=(
            "alpha answer one two three four five six seven eight",
            "beta private reference nine ten eleven twelve thirteen fourteen",
        ),
        duplicate_attachment_threshold=9_000,
        attachments=(
            (
                ("left", "inputs/left.txt", "text/plain", _digest("left")),
                ("right", "inputs/right.txt", "text/plain", _digest("right")),
            ),
            (("middle", "inputs/middle.txt", "text/plain", _digest("middle")),),
        ),
        attachment_fingerprints=(
            ("0" * 64, "f" * 10 + "0" * 54),
            ("f" * 5 + "0" * 59,),
        ),
    )

    report = _compile(inputs)
    attachment_clusters = tuple(
        cluster for cluster in inputs[4].duplicate_clusters if cluster.subject_kind.value == "ATTACHMENT"
    )

    assert len(attachment_clusters) == 1
    assert len(attachment_clusters[0].member_subject_refs) == 3
    assert len(attachment_clusters[0].direct_pair_refs) == 2
    assert len(report.duplicate_cluster_refs) == 1
    assert report.finding_count == 2
    assert report.open_p2_count == 2
    assert {finding.direct_evidence_ref for finding in report.findings} == set(
        attachment_clusters[0].direct_pair_refs
    )


@pytest.mark.asyncio
async def test_directed_answer_contamination_revises_only_target_item() -> None:
    leaked = "foreign final answer alpha one two three four five six seven"
    inputs = await _inputs(
        prompts=(
            "Analyze clean source material.",
            f"Analyze target material. {leaked}.",
        ),
        restricted_texts=(
            leaked,
            "unrelated target private reference nine ten eleven twelve",
        ),
    )

    report = _compile(inputs)

    assert report.approvable is False
    assert report.open_p0_count == 1
    assert report.blocked_item_ids == (inputs[1][1],)
    assert len(report.item_quality_report_revisions) == 1
    revision = report.item_quality_report_revisions[0]
    assert revision.item_id == inputs[1][1]
    current_report_by_item = dict(
        zip(
            report.item_ids,
            report.current_item_quality_report_refs,
            strict=True,
        )
    )
    assert current_report_by_item[inputs[2][0].item_id] == (inputs[2][0].item_quality.quality_report_ref)
    finding = report.findings[0]
    assert finding.category is BatchFindingCategoryV2.CROSS_ITEM_CONTAMINATION
    assert finding.scope is BatchFindingScopeV2.ITEM
    assert finding.owner_item_id == inputs[1][1]
    assert finding.severity is Severity.P0
    assert finding.non_waivable is True


@pytest.mark.asyncio
async def test_directed_control_leakage_is_p1_on_target_item() -> None:
    leaked = "foreign grader threshold alpha one two three four five six seven"
    inputs = await _inputs(
        prompts=(
            "Analyze clean source material.",
            f"Analyze target material. {leaked}.",
        ),
        restricted_texts=(
            leaked,
            "unrelated target private reference nine ten eleven twelve",
        ),
        categories=(
            PromptLeakageCategoryV2.GRADER_RULE,
            PromptLeakageCategoryV2.PRIVATE_REFERENCE,
        ),
    )

    report = _compile(inputs)

    assert report.approvable is False
    assert report.open_p0_count == 0
    assert report.open_p1_count == 1
    assert report.blocked_item_ids == (inputs[1][1],)
    assert len(report.item_quality_report_revisions) == 1
    assert report.item_quality_report_revisions[0].item_id == inputs[1][1]
    finding = report.findings[0]
    assert finding.category is BatchFindingCategoryV2.CROSS_ITEM_LEAKAGE
    assert finding.scope is BatchFindingScopeV2.ITEM
    assert finding.owner_item_id == inputs[1][1]
    assert finding.severity is Severity.P1
    assert finding.non_waivable is True


@pytest.mark.asyncio
async def test_full_answer_reuse_is_p0_batch_finding_without_item_revision() -> None:
    shared = "shared private answer one two three four five six seven eight"
    inputs = await _inputs(
        prompts=(
            "Analyze clean left material.",
            "Analyze clean right material.",
        ),
        restricted_texts=(shared, shared),
    )

    report = _compile(inputs)

    finding = next(
        value for value in report.findings if value.category is BatchFindingCategoryV2.ANSWER_REUSE
    )
    assert finding.scope is BatchFindingScopeV2.BATCH
    assert finding.severity is Severity.P0
    assert finding.blocked_item_ids == tuple(sorted(inputs[1]))
    assert report.item_quality_report_revisions == ()


@pytest.mark.asyncio
async def test_missing_label_selection_lineage_creates_item_revision() -> None:
    inputs = await _inputs(
        prompts=(
            "Analyze clean left material.",
            "Analyze clean right material.",
        ),
        restricted_texts=(
            "left answer one two three four five six seven eight",
            "right reference nine ten eleven twelve thirteen fourteen",
        ),
    )
    sources = inputs[2]
    original_label = sources[0].label_decisions[0]
    alternate_sha256 = _digest("alternate-label-decision")
    alternate_label = original_label.model_copy(
        update={
            "label_decision_id": (f"label-decision://sha256/{alternate_sha256}"),
            "decision_sha256": alternate_sha256,
            "trace_envelope_ref": _ref(
                "trace-envelope",
                "alternate",
                version="stored-manifest/v1",
            ),
        }
    )
    broken = replace(
        sources[0],
        label_decisions=(alternate_label,),
    )

    report = _compile(
        inputs,
        sources=(broken, sources[1]),
    )

    assert report.approvable is False
    assert report.blocked_item_ids == (inputs[1][0],)
    assert len(report.item_quality_report_revisions) == 1
    assert report.item_quality_report_revisions[0].item_id == inputs[1][0]
    lineage = next(value for value in report.lineage_audits if value.item_id == inputs[1][0])
    assert lineage.outcome is ItemLineageAuditOutcomeV2.FAILED
    failed_codes = {check.code for check in lineage.checks if check.outcome.value == "FAILED"}
    assert LineageAuditCheckCodeV2.SOURCE_TRACE_LABELS in failed_codes
    assert LineageAuditCheckCodeV2.LABEL_SELECTION in failed_codes


@pytest.mark.asyncio
async def test_cross_item_lineage_alias_is_one_batch_check_and_finding() -> None:
    inputs = await _inputs(
        prompts=(
            "Analyze clean left material.",
            "Analyze clean right material.",
        ),
        restricted_texts=(
            "left answer one two three four five six seven eight",
            "right reference nine ten eleven twelve thirteen fourteen",
        ),
    )
    sources = inputs[2]
    aliased = replace(
        sources[1],
        label_decisions=(
            *sources[1].label_decisions,
            sources[0].label_decisions[0],
        ),
    )

    report = _compile(
        inputs,
        sources=(sources[0], aliased),
    )

    alias_checks = tuple(
        check
        for check in report.batch_lineage_checks
        if check.code is LineageAuditCheckCodeV2.CROSS_ITEM_LINEAGE_ALIAS
    )
    alias_findings = tuple(
        finding
        for finding in report.findings
        if finding.category is BatchFindingCategoryV2.CROSS_ITEM_LINEAGE_ALIAS
    )
    assert len(alias_checks) == 1
    assert alias_checks[0].outcome is LineageAuditCheckOutcomeV2.FAILED
    assert alias_checks[0].failed_count == 1
    assert alias_checks[0].item_ids == tuple(sorted(inputs[1]))
    assert len(alias_findings) == 1
    assert alias_findings[0].scope is BatchFindingScopeV2.BATCH
    assert alias_findings[0].affected_item_ids == tuple(sorted(inputs[1]))
    assert alias_findings[0].blocked_item_ids == tuple(sorted(inputs[1]))
    assert alias_findings[0].direct_evidence_ref == lineage_audit_check_v2_ref(alias_checks[0])
    assert all(
        check.code is not LineageAuditCheckCodeV2.CROSS_ITEM_LINEAGE_ALIAS
        for lineage in report.lineage_audits
        for check in lineage.checks
    )


@pytest.mark.asyncio
async def test_result_is_stable_and_validate_current_rebuilds_without_io() -> None:
    inputs = await _inputs(
        prompts=(
            "Analyze clean left material.",
            "Analyze clean right material.",
        ),
        restricted_texts=(
            "left answer one two three four five six seven eight",
            "right reference nine ten eleven twelve thirteen fourteen",
        ),
    )
    first = _compile(inputs)
    second = _compile(
        inputs,
        sources=tuple(reversed(inputs[2])),
        audit=_audit(
            created_at=datetime(2026, 8, 2, tzinfo=UTC),
            created_by="different-actor",
        ),
    )

    assert first.batch_quality_report_id == second.batch_quality_report_id
    assert first.report_sha256 == second.report_sha256
    assert batch_quality_report_v2_ref(first) == batch_quality_report_v2_ref(second)
    BatchQualityCompiler().validate_current(
        first,
        resolved_job_work_graph=inputs[0],
        policy=_policy(),
        sources=inputs[2],
        duplicate_policy=inputs[3],
        duplicate_result=inputs[4],
        cross_item_policy=inputs[5],
        cross_item_result=inputs[6],
    )


@pytest.mark.asyncio
async def test_stale_source_and_budget_fail_without_report() -> None:
    inputs = await _inputs(
        prompts=(
            "Analyze clean left material.",
            "Analyze clean right material.",
        ),
        restricted_texts=(
            "left answer one two three four five six seven eight",
            "right reference nine ten eleven twelve thirteen fourteen",
        ),
    )
    stale = replace(
        inputs[2][0],
        selection_context=inputs[2][0].selection_context.model_copy(
            update={"selection_context_sha256": "f" * 64}
        ),
    )
    with pytest.raises(BatchQualityPolicyError, match="selection context"):
        _compile(
            inputs,
            sources=(stale, inputs[2][1]),
        )
    with pytest.raises(BatchQualityPolicyError, match="finding budget"):
        _compile(
            await _inputs(
                prompts=(
                    "Inspect the supplied workspace and summarize its design.",
                    "Inspect the supplied workspace and summarize its design.",
                ),
                restricted_texts=(
                    "left answer one two three four five six seven eight",
                    "right reference nine ten eleven twelve thirteen fourteen",
                ),
            ),
            policy=_policy(max_findings=0),
        )
    duplicate_inputs = await _inputs(
        prompts=(
            "Inspect the supplied workspace and summarize its design.",
            "Inspect the supplied workspace and summarize its design.",
        ),
        restricted_texts=(
            "left answer one two three four five six seven eight",
            "right reference nine ten eleven twelve thirteen fourteen",
        ),
    )
    duplicate_report = _compile(duplicate_inputs)
    lineage_evidence_count = sum(
        len(check.evidence_refs) for lineage in duplicate_report.lineage_audits for check in lineage.checks
    ) + sum(len(check.evidence_refs) for check in duplicate_report.batch_lineage_checks)
    with pytest.raises(BatchQualityPolicyError, match="direct evidence limit"):
        _compile(
            duplicate_inputs,
            policy=_policy(
                max_direct_evidence_refs=lineage_evidence_count,
            ),
        )


def test_finding_ownership_gold_is_complete_and_content_free() -> None:
    payload = json.loads(GOLD_PATH.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "eval-factory/batch-quality-gold/r7-03-v1"
    assert payload["policy_version"] == "batch-quality/r7-03-v1"
    assert payload["ownership_mode"] == "TARGET_ITEM_RELATION_BATCH"
    assert payload["direct_evidence_authority"] is True
    assert payload["stable_hash_seeds"] == [1, 321]
    scenarios = {scenario["scenario_id"]: scenario for scenario in payload["scenarios"]}
    assert set(scenarios) == {
        "clean",
        "exact-task",
        "near-task",
        "attachment-p2",
        "directed-p0",
        "directed-p1",
        "answer-reuse",
        "single-item-lineage",
        "cross-item-lineage-alias",
        "connected-transitive",
        "stale-source",
        "preflight-budget",
    }
    assert scenarios["near-task"]["approvable"] is True
    assert scenarios["attachment-p2"]["open_p2_count"] == 2
    assert scenarios["cross-item-lineage-alias"]["batch_finding_count"] == 1
    transitive = scenarios["connected-transitive"]
    assert transitive["cluster_member_count"] == 3
    assert transitive["direct_finding_count"] == 2
    assert transitive["fabricated_finding_count"] == 0
    serialized = json.dumps(payload, sort_keys=True).casefold()
    for forbidden in (
        "raw_trace",
        "visible_prompt",
        "source_text",
        "restricted_text",
        "private_reference",
        "grader_rule",
        "hidden_condition",
        "attachment_bytes",
        "physical_path",
        "token_inventory",
        "provider_payload",
        "credential",
        "diagnostic",
    ):
        assert forbidden not in serialized
