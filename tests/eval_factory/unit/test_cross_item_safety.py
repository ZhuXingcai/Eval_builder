from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from test_duplicate_detection import (
    _graph,
    _quality_result,
    _task_contract_set,
)
from test_task_prompt_safety import (
    _compile as _compile_prompt_safety,
)
from test_task_prompt_safety import (
    _pending_draft,
    _reference_set,
)
from test_task_prompt_safety import (
    _source as _restricted_source,
)

from env_mock_agent.facade import (
    AttachmentCrossItemSafetyScanFailureCodeV2,
    AttachmentCrossItemSafetyScanRequestV2,
    AttachmentCrossItemSafetyScanResultV2,
    AttachmentCrossItemSafetyScanStatusV2,
    attachment_cross_item_safety_scan_request_ref,
    attachment_cross_item_safety_scan_result_carried_sha256,
)
from eval_factory.batch_quality.cross_item_safety import (
    CrossItemSafetyCompiler,
    CrossItemSafetyItemSource,
    CrossItemSafetyPolicyError,
)
from eval_factory.contracts.core import ContractAudit, VersionBinding
from eval_factory.contracts.cross_item_safety_v2 import (
    AnswerReuseMatchKindV2,
    CrossItemSafetyClusterKindV2,
    CrossItemSafetyEvidenceKindV2,
    CrossItemSafetyPolicyV2,
    CrossItemSafetySurfaceKindV2,
)
from eval_factory.contracts.task_v2 import (
    PromptLeakageCategoryV2,
    PromptLeakageReferenceSetV2,
)
from eval_factory.task_authoring import (
    FakeTaskPromptSafetyFixture,
    TaskPromptSafetyOutcome,
)

NOW = datetime(2026, 8, 1, tzinfo=UTC)
GOLD_PATH = (
    Path(__file__).resolve().parents[3]
    / "evals/golden/eval_factory/batch_quality"
    / "r7-02-batch-safety-v1.json"
)


def _audit() -> ContractAudit:
    return ContractAudit(
        created_at=NOW,
        created_by="cross-item-safety-test",
        governing_versions=(
            VersionBinding(
                component="cross-item-safety",
                version="r7-02-v1",
            ),
        ),
        input_refs=(),
    )


def _policy(
    *,
    min_shared_windows: int = 2,
    min_coverage_bps: int = 8_000,
    max_prompt_characters: int = 100_000,
    max_prompt_scan_comparisons: int = 10_000_000,
    max_inventory_members: int = 10_000,
    max_match_count: int = 10_000,
) -> CrossItemSafetyPolicyV2:
    from env_mock_agent.facade import (
        AttachmentCrossItemSafetyScanLimitsV2,
    )

    limits = AttachmentCrossItemSafetyScanLimitsV2(
        max_file_bytes=10_000_000,
        max_extracted_characters=1_000_000,
        max_inventory_members=max_inventory_members,
        max_nested_depth=4,
        max_expanded_bytes=50_000_000,
        max_compression_ratio_milli=1_000_000,
        max_fingerprint_count=100_000,
        max_match_count=max_match_count,
    )
    return CrossItemSafetyPolicyV2.create(
        min_shared_answer_windows=min_shared_windows,
        min_answer_reuse_coverage_bps=min_coverage_bps,
        max_items=100,
        max_attachments=1_000,
        max_total_fingerprints=100_000,
        max_foreign_fingerprints_per_target=100_000,
        max_prompt_characters=max_prompt_characters,
        max_prompt_scan_comparisons=max_prompt_scan_comparisons,
        max_answer_source_pairs=1_000_000,
        max_answer_window_comparisons=10_000_000,
        max_visible_matches=100_000,
        max_reuse_pairs=100_000,
        max_clusters=100_000,
        attachment_scan_limits=limits,
        audit=_audit(),
    )


def _passed_fixture() -> FakeTaskPromptSafetyFixture:
    return FakeTaskPromptSafetyFixture(
        fixture_id="fake-task-prompt-safety-fixture://r7-02/passed",
        outcome=TaskPromptSafetyOutcome.PASSED,
        findings=(),
        unresolved_reasons=frozenset(),
        model_available=True,
    )


def _source(
    *,
    item_id: str,
    suffix: str,
    prompt: str,
    category: PromptLeakageCategoryV2,
    restricted_text: str,
    attachments: tuple[tuple[str, str, str, str], ...] = (),
) -> CrossItemSafetyItemSource:
    reference_set = _reference_set(
        _restricted_source(
            category,
            text=restricted_text,
            suffix=f"r7-02-{suffix}",
        )
    )
    return _source_from_reference_set(
        item_id=item_id,
        suffix=suffix,
        prompt=prompt,
        reference_set=reference_set,
        attachments=attachments,
    )


def _source_from_reference_set(
    *,
    item_id: str,
    suffix: str,
    prompt: str,
    reference_set: PromptLeakageReferenceSetV2,
    attachments: tuple[tuple[str, str, str, str], ...] = (),
) -> CrossItemSafetyItemSource:
    pending = _pending_draft(visible_prompt=prompt)
    _, _, result = _compile_prompt_safety(
        pending,
        reference_set,
        _passed_fixture(),
    )
    assert result.task_draft is not None
    assert result.task_prompt_safety_gate is not None
    draft = result.task_draft
    contract_set = _task_contract_set(draft, suffix=f"r7-02-{suffix}")
    quality = _quality_result(
        suffix=f"r7-02-{suffix}",
        contract_set=contract_set,
        attachments=attachments,
    )
    return CrossItemSafetyItemSource(
        item_id=item_id,
        task_draft=draft,
        task_prompt_safety_gate=result.task_prompt_safety_gate,
        leakage_reference_set=reference_set,
        task_contract_set=contract_set,
        item_quality=quality,
    )


class _ScanFacade:
    def __init__(
        self,
        *,
        matched_by_item: dict[str, tuple[str, ...]] | None = None,
        blocked_items: set[str] | None = None,
        scanned_member_count_by_item: dict[str, int] | None = None,
    ) -> None:
        self.matched_by_item = matched_by_item or {}
        self.blocked_items = blocked_items or set()
        self.scanned_member_count_by_item = scanned_member_count_by_item or {}
        self.calls: list[str] = []

    async def scan(
        self,
        request: AttachmentCrossItemSafetyScanRequestV2,
    ) -> AttachmentCrossItemSafetyScanResultV2:
        self.calls.append(request.target_item_id)
        blocked = request.target_item_id in self.blocked_items
        matched = () if blocked else self.matched_by_item.get(request.target_item_id, ())
        pending = AttachmentCrossItemSafetyScanResultV2(
            scan_result_id="attachment-cross-item-safety-scan-result://pending",
            scan_request_ref=attachment_cross_item_safety_scan_request_ref(request),
            environment_artifact_ref=request.environment_artifact_ref,
            output_ref=request.output_ref,
            output_sha256=request.content_sha256,
            status=(
                AttachmentCrossItemSafetyScanStatusV2.BLOCKED
                if blocked
                else AttachmentCrossItemSafetyScanStatusV2.PASSED
            ),
            matched_fingerprint_ids=tuple(sorted(matched)),
            scanned_member_count=(
                0
                if blocked
                else self.scanned_member_count_by_item.get(
                    request.target_item_id,
                    1,
                )
            ),
            scan_complete=not blocked,
            failure_code=(
                AttachmentCrossItemSafetyScanFailureCodeV2.CONTENT_UNSCANNABLE if blocked else None
            ),
            extractor_version=(None if blocked else "attachment-cross-item-safety-extractor/r7-02-v1"),
            normalization_version=request.normalization_version,
            policy_version=request.policy_version,
            result_sha256="0" * 64,
        )
        digest = attachment_cross_item_safety_scan_result_carried_sha256(pending)
        return pending.model_copy(
            update={
                "scan_result_id": (f"attachment-cross-item-safety-scan-result://sha256/{digest}"),
                "result_sha256": digest,
            }
        )


def _first_fingerprint_id(source: CrossItemSafetyItemSource) -> str:
    return source.leakage_reference_set.fingerprints[0].fingerprint_id


@pytest.mark.asyncio
async def test_foreign_final_answer_in_prompt_is_directed_contamination() -> None:
    graph, item_ids = _graph(2)
    leaked = "approved foreign answer is forty two for account alpha"
    left = _source(
        item_id=item_ids[0],
        suffix="left",
        prompt="Analyze the clean left input without relying on hidden output.",
        category=PromptLeakageCategoryV2.FINAL_ANSWER,
        restricted_text=leaked,
    )
    right = _source(
        item_id=item_ids[1],
        suffix="right",
        prompt=f"Inspect the workspace. {leaked}. Explain the conclusion.",
        category=PromptLeakageCategoryV2.PRIVATE_REFERENCE,
        restricted_text="unrelated private reference for the right item",
    )

    result = await CrossItemSafetyCompiler().compile(
        resolved_job_work_graph=graph,
        policy=_policy(),
        sources=(right, left),
        facade=_ScanFacade(),
        audit=_audit(),
    )

    assert len(result.visible_matches) == 1
    match = result.visible_matches[0]
    assert match.evidence_kind is CrossItemSafetyEvidenceKindV2.CONTAMINATION
    assert match.target_surface is CrossItemSafetySurfaceKindV2.PROMPT
    assert match.source_item_id == item_ids[0]
    assert match.target_item_id == item_ids[1]
    assert all(value.source_item_id != value.target_item_id for value in result.visible_matches)


@pytest.mark.parametrize(
    ("category", "expected_kind"),
    (
        (
            PromptLeakageCategoryV2.FINAL_ANSWER,
            CrossItemSafetyEvidenceKindV2.CONTAMINATION,
        ),
        (
            PromptLeakageCategoryV2.COMPLETED_DELIVERABLE,
            CrossItemSafetyEvidenceKindV2.CONTAMINATION,
        ),
        (
            PromptLeakageCategoryV2.PRIVATE_REFERENCE,
            CrossItemSafetyEvidenceKindV2.LEAKAGE,
        ),
        (
            PromptLeakageCategoryV2.GRADER_RULE,
            CrossItemSafetyEvidenceKindV2.LEAKAGE,
        ),
        (
            PromptLeakageCategoryV2.HIDDEN_PASS_CONDITION,
            CrossItemSafetyEvidenceKindV2.LEAKAGE,
        ),
        (
            PromptLeakageCategoryV2.HIDDEN_SELECTION_SIGNAL,
            CrossItemSafetyEvidenceKindV2.LEAKAGE,
        ),
        (
            PromptLeakageCategoryV2.TRAJECTORY_SPECIFIC_STEP,
            CrossItemSafetyEvidenceKindV2.LEAKAGE,
        ),
    ),
)
@pytest.mark.asyncio
async def test_all_restricted_categories_have_exact_prompt_classification(
    category: PromptLeakageCategoryV2,
    expected_kind: CrossItemSafetyEvidenceKindV2,
) -> None:
    graph, item_ids = _graph(2)
    marker = category.value.casefold().replace("_", "-")
    restricted = f"foreign {marker} marker alpha beta gamma delta epsilon zeta"
    left = _source(
        item_id=item_ids[0],
        suffix=f"category-{marker}",
        prompt="Analyze a clean source Item.",
        category=category,
        restricted_text=restricted,
    )
    right = _source(
        item_id=item_ids[1],
        suffix=f"target-{marker}",
        prompt=f"Analyze the target Item with {restricted}.",
        category=PromptLeakageCategoryV2.FINAL_ANSWER,
        restricted_text=f"unrelated target answer for {marker}",
    )

    result = await CrossItemSafetyCompiler().compile(
        resolved_job_work_graph=graph,
        policy=_policy(),
        sources=(left, right),
        facade=_ScanFacade(),
        audit=_audit(),
    )

    match = next(value for value in result.visible_matches if value.source_item_id == item_ids[0])
    assert match.category is category
    assert match.evidence_kind is expected_kind


@pytest.mark.asyncio
async def test_full_and_partial_answer_reuse_follow_strict_policy() -> None:
    graph, item_ids = _graph(3)
    shared_full = "approved answer contains customer alpha quarterly total forty two"
    partial_left = "prefix one two three four five six seven eight nine ten suffix left"
    partial_right = "different one two three four five six seven eight nine ten ending"
    sources = (
        _source(
            item_id=item_ids[0],
            suffix="full-left",
            prompt="Analyze the first clean input.",
            category=PromptLeakageCategoryV2.FINAL_ANSWER,
            restricted_text=shared_full,
        ),
        _source(
            item_id=item_ids[1],
            suffix="full-right",
            prompt="Analyze the second clean input.",
            category=PromptLeakageCategoryV2.PRIVATE_REFERENCE,
            restricted_text=shared_full,
        ),
        _source(
            item_id=item_ids[2],
            suffix="partial",
            prompt="Analyze the third clean input.",
            category=PromptLeakageCategoryV2.COMPLETED_DELIVERABLE,
            restricted_text=partial_right,
        ),
    )

    full_result = await CrossItemSafetyCompiler().compile(
        resolved_job_work_graph=graph,
        policy=_policy(min_coverage_bps=6_000),
        sources=sources,
        facade=_ScanFacade(),
        audit=_audit(),
    )

    assert any(
        pair.match_kind is AnswerReuseMatchKindV2.FULL_SOURCE for pair in full_result.answer_reuse_pairs
    )
    assert all(
        cluster.cluster_kind is CrossItemSafetyClusterKindV2.ANSWER_REUSE for cluster in full_result.clusters
    )

    partial_graph, partial_ids = _graph(2)
    partial_sources = (
        _source(
            item_id=partial_ids[0],
            suffix="partial-left",
            prompt="Analyze clean left material.",
            category=PromptLeakageCategoryV2.FINAL_ANSWER,
            restricted_text=partial_left,
        ),
        _source(
            item_id=partial_ids[1],
            suffix="partial-right",
            prompt="Analyze clean right material.",
            category=PromptLeakageCategoryV2.PRIVATE_REFERENCE,
            restricted_text=partial_right,
        ),
    )
    partial_result = await CrossItemSafetyCompiler().compile(
        resolved_job_work_graph=partial_graph,
        policy=_policy(min_coverage_bps=6_000),
        sources=partial_sources,
        facade=_ScanFacade(),
        audit=_audit(),
    )

    assert len(partial_result.answer_reuse_pairs) == 1
    pair = partial_result.answer_reuse_pairs[0]
    assert pair.match_kind is AnswerReuseMatchKindV2.PARTIAL_WINDOWS
    assert pair.shared_window_count >= 2
    assert pair.coverage_bps >= 6_000


@pytest.mark.parametrize(
    ("left_text", "right_text", "minimum_coverage"),
    (
        (
            "left alpha beta gamma delta epsilon zeta eta theta tail",
            "right alpha beta gamma delta epsilon zeta eta theta ending",
            1,
        ),
        (
            ("left alpha beta gamma delta epsilon zeta eta theta iota left-one left-two left-three"),
            ("right alpha beta gamma delta epsilon zeta eta theta iota right-one right-two right-three"),
            8_000,
        ),
    ),
    ids=("single-shared-window", "below-shorter-source-coverage"),
)
@pytest.mark.asyncio
async def test_partial_answer_reuse_rejects_count_and_coverage_boundaries(
    left_text: str,
    right_text: str,
    minimum_coverage: int,
) -> None:
    graph, item_ids = _graph(2)
    sources = (
        _source(
            item_id=item_ids[0],
            suffix="negative-left",
            prompt="Analyze clean left material.",
            category=PromptLeakageCategoryV2.FINAL_ANSWER,
            restricted_text=left_text,
        ),
        _source(
            item_id=item_ids[1],
            suffix="negative-right",
            prompt="Analyze clean right material.",
            category=PromptLeakageCategoryV2.PRIVATE_REFERENCE,
            restricted_text=right_text,
        ),
    )

    result = await CrossItemSafetyCompiler().compile(
        resolved_job_work_graph=graph,
        policy=_policy(min_coverage_bps=minimum_coverage),
        sources=sources,
        facade=_ScanFacade(),
        audit=_audit(),
    )

    assert result.answer_reuse_pairs == ()
    assert result.clusters == ()


@pytest.mark.asyncio
async def test_transitive_answer_reuse_cluster_keeps_only_direct_pairs() -> None:
    graph, item_ids = _graph(3)
    left_segment = "a1 a2 a3 a4 a5 a6 a7 a8 a9"
    right_segment = "c1 c2 c3 c4 c5 c6 c7 c8 c9"
    bridge = f"prefix {left_segment} separator {right_segment} suffix"
    sources = (
        _source(
            item_id=item_ids[0],
            suffix="chain-left",
            prompt="Analyze clean chain left material.",
            category=PromptLeakageCategoryV2.FINAL_ANSWER,
            restricted_text=left_segment,
        ),
        _source(
            item_id=item_ids[1],
            suffix="chain-bridge",
            prompt="Analyze clean chain bridge material.",
            category=PromptLeakageCategoryV2.PRIVATE_REFERENCE,
            restricted_text=bridge,
        ),
        _source(
            item_id=item_ids[2],
            suffix="chain-right",
            prompt="Analyze clean chain right material.",
            category=PromptLeakageCategoryV2.COMPLETED_DELIVERABLE,
            restricted_text=right_segment,
        ),
    )

    result = await CrossItemSafetyCompiler().compile(
        resolved_job_work_graph=graph,
        policy=_policy(),
        sources=sources,
        facade=_ScanFacade(),
        audit=_audit(),
    )

    assert len(result.answer_reuse_pairs) == 2
    assert {frozenset((pair.left_item_id, pair.right_item_id)) for pair in result.answer_reuse_pairs} == {
        frozenset((item_ids[0], item_ids[1])),
        frozenset((item_ids[1], item_ids[2])),
    }
    cluster = result.clusters[0]
    assert cluster.cluster_kind is CrossItemSafetyClusterKindV2.ANSWER_REUSE
    assert cluster.member_item_ids == tuple(sorted(item_ids))
    assert len(cluster.direct_evidence_refs) == 2


@pytest.mark.asyncio
async def test_result_identity_is_stable_across_order_and_audit_metadata() -> None:
    graph, item_ids = _graph(2)
    shared = "approved stable answer alpha beta gamma delta epsilon zeta"
    sources = (
        _source(
            item_id=item_ids[0],
            suffix="stable-left",
            prompt="Analyze stable clean left material.",
            category=PromptLeakageCategoryV2.FINAL_ANSWER,
            restricted_text=shared,
        ),
        _source(
            item_id=item_ids[1],
            suffix="stable-right",
            prompt="Analyze stable clean right material.",
            category=PromptLeakageCategoryV2.PRIVATE_REFERENCE,
            restricted_text=shared,
        ),
    )
    compiler = CrossItemSafetyCompiler()
    policy = _policy()
    first_audit = _audit()
    second_audit = first_audit.model_copy(
        update={
            "created_at": datetime(2026, 8, 2, tzinfo=UTC),
            "created_by": "different-cross-item-actor",
        }
    )

    first = await compiler.compile(
        resolved_job_work_graph=graph,
        policy=policy,
        sources=sources,
        facade=_ScanFacade(),
        audit=first_audit,
    )
    second = await compiler.compile(
        resolved_job_work_graph=graph,
        policy=policy,
        sources=tuple(reversed(sources)),
        facade=_ScanFacade(),
        audit=second_audit,
    )

    assert first.cross_item_safety_result_id == second.cross_item_safety_result_id
    assert first.result_sha256 == second.result_sha256
    assert tuple(value.pair_sha256 for value in first.answer_reuse_pairs) == tuple(
        value.pair_sha256 for value in second.answer_reuse_pairs
    )
    assert tuple(value.cluster_sha256 for value in first.clusters) == tuple(
        value.cluster_sha256 for value in second.clusters
    )


@pytest.mark.asyncio
async def test_foreign_attachment_match_and_currentness_reuse_provider_result() -> None:
    graph, item_ids = _graph(2)
    leaked = "approved foreign answer is forty two for account alpha"
    left = _source(
        item_id=item_ids[0],
        suffix="left-attachment",
        prompt="Analyze clean left material.",
        category=PromptLeakageCategoryV2.FINAL_ANSWER,
        restricted_text=leaked,
    )
    right = _source(
        item_id=item_ids[1],
        suffix="right-attachment",
        prompt=f"Analyze right material while noting: {leaked}.",
        category=PromptLeakageCategoryV2.PRIVATE_REFERENCE,
        restricted_text="unrelated right reference",
        attachments=(
            (
                "input",
                "inputs/source.txt",
                "text/plain",
                hashlib.sha256(b"attachment").hexdigest(),
            ),
        ),
    )
    facade = _ScanFacade(
        matched_by_item={
            item_ids[1]: (_first_fingerprint_id(left),),
        }
    )
    compiler = CrossItemSafetyCompiler()
    policy = _policy()

    result = await compiler.compile(
        resolved_job_work_graph=graph,
        policy=policy,
        sources=(left, right),
        facade=facade,
        audit=_audit(),
    )

    assert facade.calls == [item_ids[1]]
    assert len(result.attachment_scan_evidence) == 1
    assert len(result.visible_matches) == 2
    attachment_match = next(
        value
        for value in result.visible_matches
        if value.target_surface is CrossItemSafetySurfaceKindV2.ATTACHMENT
    )
    assert attachment_match.evidence_kind is CrossItemSafetyEvidenceKindV2.CONTAMINATION
    compiler.validate_current(
        result,
        resolved_job_work_graph=graph,
        policy=policy,
        sources=(right, left),
    )
    assert facade.calls == [item_ids[1]]


@pytest.mark.asyncio
async def test_prompt_length_and_r4_policy_versions_fail_before_provider() -> None:
    graph, item_ids = _graph(2)
    left = _source(
        item_id=item_ids[0],
        suffix="bounded-left",
        prompt="Analyze clean left material.",
        category=PromptLeakageCategoryV2.FINAL_ANSWER,
        restricted_text="approved foreign answer is forty two",
    )
    right = _source(
        item_id=item_ids[1],
        suffix="bounded-right",
        prompt="Analyze clean right material.",
        category=PromptLeakageCategoryV2.PRIVATE_REFERENCE,
        restricted_text="unrelated right reference",
    )
    facade = _ScanFacade()

    with pytest.raises(CrossItemSafetyPolicyError, match="prompt character limit"):
        await CrossItemSafetyCompiler().compile(
            resolved_job_work_graph=graph,
            policy=_policy(max_prompt_characters=1),
            sources=(left, right),
            facade=facade,
            audit=_audit(),
        )

    stale_reference_set = left.leakage_reference_set.model_copy(
        update={"fingerprint_policy_version": "task-prompt-leakage-fingerprint/stale"}
    )
    with pytest.raises(CrossItemSafetyPolicyError, match="policy version is stale"):
        await CrossItemSafetyCompiler().compile(
            resolved_job_work_graph=graph,
            policy=_policy(),
            sources=(
                CrossItemSafetyItemSource(
                    item_id=left.item_id,
                    task_draft=left.task_draft,
                    task_prompt_safety_gate=left.task_prompt_safety_gate,
                    leakage_reference_set=stale_reference_set,
                    task_contract_set=left.task_contract_set,
                    item_quality=left.item_quality,
                ),
                right,
            ),
            facade=facade,
            audit=_audit(),
        )
    assert facade.calls == []


@pytest.mark.asyncio
async def test_provider_result_counts_cannot_exceed_request_limits() -> None:
    graph, item_ids = _graph(2)
    left = _source(
        item_id=item_ids[0],
        suffix="provider-limit-left",
        prompt="Analyze clean left material.",
        category=PromptLeakageCategoryV2.FINAL_ANSWER,
        restricted_text="approved foreign answer is forty two for account alpha",
    )
    right = _source(
        item_id=item_ids[1],
        suffix="provider-limit-right",
        prompt="Analyze clean right material.",
        category=PromptLeakageCategoryV2.PRIVATE_REFERENCE,
        restricted_text="unrelated right reference",
        attachments=(
            (
                "input",
                "inputs/source.txt",
                "text/plain",
                hashlib.sha256(b"attachment").hexdigest(),
            ),
        ),
    )
    matched_ids = tuple(
        fingerprint.fingerprint_id for fingerprint in left.leakage_reference_set.fingerprints[:2]
    )

    with pytest.raises(CrossItemSafetyPolicyError, match="exceeded request limits"):
        await CrossItemSafetyCompiler().compile(
            resolved_job_work_graph=graph,
            policy=_policy(max_match_count=1),
            sources=(left, right),
            facade=_ScanFacade(
                matched_by_item={item_ids[1]: matched_ids},
            ),
            audit=_audit(),
        )
    with pytest.raises(CrossItemSafetyPolicyError, match="exceeded request limits"):
        await CrossItemSafetyCompiler().compile(
            resolved_job_work_graph=graph,
            policy=_policy(max_inventory_members=1),
            sources=(left, right),
            facade=_ScanFacade(
                scanned_member_count_by_item={item_ids[1]: 2},
            ),
            audit=_audit(),
        )


@pytest.mark.asyncio
async def test_answer_source_with_multiple_categories_is_counted_once() -> None:
    graph, item_ids = _graph(2)
    shared = "approved shared answer is forty two for account alpha"
    primary = _restricted_source(
        PromptLeakageCategoryV2.FINAL_ANSWER,
        text=shared,
        suffix="r7-02-multi-category",
    )
    secondary = primary.model_copy(
        update={
            "source_id": "restricted-prompt-leakage-source://r7-02-multi-category-secondary",
            "category": PromptLeakageCategoryV2.COMPLETED_DELIVERABLE,
        }
    )
    left = _source_from_reference_set(
        item_id=item_ids[0],
        suffix="multi-category-left",
        prompt="Analyze clean left material.",
        reference_set=_reference_set(primary, secondary),
    )
    right = _source(
        item_id=item_ids[1],
        suffix="multi-category-right",
        prompt="Analyze clean right material.",
        category=PromptLeakageCategoryV2.PRIVATE_REFERENCE,
        restricted_text=shared,
    )

    result = await CrossItemSafetyCompiler().compile(
        resolved_job_work_graph=graph,
        policy=_policy(),
        sources=(left, right),
        facade=_ScanFacade(),
        audit=_audit(),
    )

    assert len(result.answer_source_subject_refs) == 2
    assert len(result.answer_reuse_pairs) == 1
    assert result.answer_reuse_pairs[0].match_kind is AnswerReuseMatchKindV2.FULL_SOURCE


@pytest.mark.asyncio
async def test_blocked_scan_and_preflight_budget_produce_no_partial_result() -> None:
    graph, item_ids = _graph(2)
    left = _source(
        item_id=item_ids[0],
        suffix="blocked-left",
        prompt="Analyze clean left material.",
        category=PromptLeakageCategoryV2.FINAL_ANSWER,
        restricted_text="approved foreign answer is forty two",
    )
    right = _source(
        item_id=item_ids[1],
        suffix="blocked-right",
        prompt="Analyze clean right material.",
        category=PromptLeakageCategoryV2.PRIVATE_REFERENCE,
        restricted_text="unrelated right reference",
        attachments=(
            (
                "input",
                "inputs/source.txt",
                "text/plain",
                hashlib.sha256(b"attachment").hexdigest(),
            ),
        ),
    )
    blocked = _ScanFacade(blocked_items={item_ids[1]})

    with pytest.raises(CrossItemSafetyPolicyError, match="blocked"):
        await CrossItemSafetyCompiler().compile(
            resolved_job_work_graph=graph,
            policy=_policy(),
            sources=(left, right),
            facade=blocked,
            audit=_audit(),
        )

    no_call = _ScanFacade()
    with pytest.raises(CrossItemSafetyPolicyError, match="prompt scan budget"):
        await CrossItemSafetyCompiler().compile(
            resolved_job_work_graph=graph,
            policy=_policy(max_prompt_scan_comparisons=0),
            sources=(left, right),
            facade=no_call,
            audit=_audit(),
        )
    assert no_call.calls == []


def test_batch_safety_gold_is_complete_and_content_free() -> None:
    payload = json.loads(GOLD_PATH.read_text(encoding="utf-8"))

    assert payload["policy_version"] == "cross-item-safety/r7-02-v1"
    assert payload["classification_mode"] == "CATEGORY_EXACT"
    assert payload["clustering_mode"] == "CONNECTED_COMPONENTS"
    assert payload["direct_evidence_authority"] is True
    assert payload["stable_hash_seeds"] == [1, 321]
    assert {scenario["scenario_id"] for scenario in payload["scenarios"]} == {
        "prompt-contamination",
        "prompt-leakage",
        "attachment-contamination",
        "attachment-leakage",
        "clean-unrelated",
        "same-item-excluded",
        "full-answer-reuse",
        "partial-answer-reuse",
        "partial-below-window-count",
        "partial-below-coverage",
        "connected-transitive",
        "no-attachment",
        "blocked-attachment",
        "stale-source",
        "preflight-budget",
    }
    transitive = next(
        scenario for scenario in payload["scenarios"] if scenario["scenario_id"] == "connected-transitive"
    )
    assert transitive["cluster_member_count"] == 3
    assert transitive["direct_evidence_count"] == 2
    assert transitive["transitive_unobserved_pair_count"] == 1
    serialized = json.dumps(payload, sort_keys=True).casefold()
    for forbidden in (
        "raw_trace",
        "visible_prompt",
        "source_text",
        "extracted_text",
        "matched_text",
        "token_inventory",
        "window_inventory",
        "physical_path",
        "member_path",
        "private_reference",
        "grader_rule",
        "hidden_condition",
        "provider_payload",
        "release_decision",
        "batch_quality_report",
    ):
        assert forbidden not in serialized
