from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from label_quality_fixtures import (
    fixture_audit,
    frozen_authority,
    frozen_request,
    observation_set,
    quality_policy,
)

from eval_factory.contracts.label_quality_v2 import (
    LabelQualityClassMetricV2,
    LabelQualityEvaluationReportV2,
    LabelQualityEvidenceClassV2,
    LabelQualityLabelOutcomeV2,
    LabelQualityMetricNameV2,
    LabelQualityReasonCodeV2,
    StructuredLabelQualitySummaryV2,
)
from eval_factory.contracts.labeling_v2 import LabelDecisionValueV2
from eval_factory.statistics.label_quality import (
    LabelQualityEvaluator,
    wilson_score_interval,
)
from eval_factory.statistics.label_quality_builder import (
    LabelQualityEvaluationBuilder,
)
from eval_factory.statistics.label_quality_models import (
    LabelQualityAcceptedRunV1,
    LabelQualityResultSetV1,
)
from eval_factory.statistics.label_quality_store import (
    LabelQualityAcceptedRunStore,
    LabelQualityMaterialStore,
    LabelQualityReportStore,
    LabelQualityStoreConflictError,
    LabelQualityStoreFaultPoint,
    LabelQualityStoreInjectedCrash,
    LabelQualityStoreIntegrityError,
    LabelQualityStoreTypeError,
    StaticLabelQualityStoreFaultInjector,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
R8_01_GOLD = (
    REPO_ROOT / "evals/golden/eval_factory/statistics/r8-01-independent-label-test-set-freeze-v1.json"
)


def _frozen(tmp_path: Path):
    persistence, specs, frozen = frozen_authority(tmp_path / "source")
    manifest = frozen.dataset_manifest
    assert manifest is not None
    material = persistence.material_store.get_test_set(manifest.private_material_ref)
    observations = observation_set(frozen, specs, material.members)
    policy = quality_policy(frozen, specs)
    compilation = LabelQualityEvaluationBuilder().compile_frozen(
        persistence=persistence,
        expected_result=frozen,
        policy=policy,
        request=frozen_request(frozen, specs, observations),
        audit=fixture_audit(),
    )
    report = LabelQualityEvaluator().evaluate(
        compilation=compilation,
        policy=policy,
        audit=fixture_audit(),
    )
    return compilation, report


def _stores(
    tmp_path: Path,
    *,
    fault_points: frozenset[LabelQualityStoreFaultPoint] = frozenset(),
) -> LabelQualityAcceptedRunStore:
    injector = StaticLabelQualityStoreFaultInjector(crash_points=fault_points)
    return LabelQualityAcceptedRunStore(
        material_store=LabelQualityMaterialStore(
            tmp_path / "private",
            max_private_bytes=100_000_000,
            max_members=10_000,
            fault_injector=injector,
        ),
        report_store=LabelQualityReportStore(
            tmp_path / "report",
            max_report_bytes=10_000_000,
            fault_injector=injector,
        ),
        fault_injector=injector,
    )


def _snapshot(path: Path) -> tuple[int, int]:
    files = tuple(value for value in path.rglob("*") if value.is_file())
    return len(files), sum(value.stat().st_size for value in files)


def test_material_and_report_codecs_round_trip_complete_closure(
    tmp_path: Path,
) -> None:
    compilation, report = _frozen(tmp_path)
    assert compilation.observation_set is not None
    assert compilation.result_set is not None
    stores = _stores(tmp_path / "stores")

    stores.material_store.put_observation_set(compilation.observation_set)
    for pair in compilation.result_set.pair_results:
        stores.material_store.put_pair_result(pair)
    stores.material_store.put_result_set(compilation.result_set)
    stores.report_store.put(report)

    assert (
        stores.material_store.get_observation_set(compilation.observation_set.to_ref())
        == compilation.observation_set
    )
    assert stores.material_store.get_result_set(compilation.result_set.to_ref()) == compilation.result_set
    assert stores.report_store.get(report.to_ref()) == report


def test_first_authority_exact_replay_is_physical_noop_and_changed_request_conflicts(
    tmp_path: Path,
) -> None:
    compilation, report = _frozen(tmp_path)
    stores = _stores(tmp_path / "stores")
    request_sha = hashlib.sha256(b"exact-frozen-request").hexdigest()

    first = stores.persist(
        acceptance_key="label-quality-acceptance://r8-02/frozen/v1",
        request_sha256=request_sha,
        compilation=compilation,
        report=report,
        audit=fixture_audit(),
    )
    before = _snapshot(tmp_path / "stores")
    replay = stores.persist(
        acceptance_key="label-quality-acceptance://r8-02/frozen/v1",
        request_sha256=request_sha,
        compilation=compilation,
        report=report,
        audit=fixture_audit().model_copy(update={"created_by": "replay-actor"}),
    )
    after = _snapshot(tmp_path / "stores")

    assert first.accepted_run_id == replay.accepted_run_id
    assert before == after
    assert stores.get_accepted_report("label-quality-acceptance://r8-02/frozen/v1") == report
    with pytest.raises(LabelQualityStoreConflictError, match="request"):
        stores.persist(
            acceptance_key="label-quality-acceptance://r8-02/frozen/v1",
            request_sha256=hashlib.sha256(b"changed").hexdigest(),
            compilation=compilation,
            report=report,
            audit=fixture_audit(),
        )


def test_mixed_observation_and_pair_closure_is_rejected_before_acceptance(
    tmp_path: Path,
) -> None:
    compilation, report = _frozen(tmp_path)
    assert compilation.observation_set is not None
    assert compilation.result_set is not None
    stores = _stores(tmp_path / "stores")
    pairs = compilation.result_set.pair_results
    decision = pairs[0].decision
    assert decision is not None
    mixed_pair = pairs[0].create(
        reference=pairs[0].reference,
        decision=decision.model_copy(update={"confidence": 0.5}),
        status=pairs[0].status,
        audit=fixture_audit(),
    )
    mixed = LabelQualityResultSetV1.create(
        dataset_manifest_ref=compilation.result_set.dataset_manifest_ref,
        observation_set_ref=compilation.observation_set.to_ref(),
        pair_results=(
            mixed_pair,
            *pairs[1:],
        ),
        audit=fixture_audit(),
    )
    changed = compilation.__class__(
        policy=compilation.policy,
        prerequisite=compilation.prerequisite,
        observation_set=compilation.observation_set,
        result_set=mixed,
        evidence_sha256=compilation.evidence_sha256,
        evidence_class=compilation.evidence_class,
    )
    mixed_report = LabelQualityEvaluationReportV2.create(
        policy_ref=report.policy_ref,
        evidence_class=report.evidence_class,
        prerequisite=report.prerequisite,
        observation_closure_sha256=report.observation_closure_sha256,
        result_closure_sha256=mixed.result_set_sha256,
        structured_summaries=report.structured_summaries,
        semantic_summaries=report.semantic_summaries,
        audit=fixture_audit(),
    )

    with pytest.raises(LabelQualityStoreIntegrityError, match="observation"):
        stores.persist(
            acceptance_key="label-quality-acceptance://r8-02/mixed",
            request_sha256=hashlib.sha256(b"mixed").hexdigest(),
            compilation=changed,
            report=mixed_report,
            audit=fixture_audit(),
        )
    with pytest.raises(LabelQualityStoreIntegrityError, match="not found"):
        stores.get_accepted_report("label-quality-acceptance://r8-02/mixed")


def test_public_summary_must_be_derived_from_private_pair_results(
    tmp_path: Path,
) -> None:
    compilation, report = _frozen(tmp_path)
    original = report.structured_summaries[0]
    forged_summary = StructuredLabelQualitySummaryV2(
        label_spec_ref=original.label_spec_ref,
        reference_match_count=50,
        reference_no_match_count=50,
        true_positive_count=49,
        false_positive_count=0,
        true_negative_count=50,
        false_negative_count=1,
        observed_abstain_count=0,
        incomplete_observation_count=0,
        missing_observation_count=0,
        precision=LabelQualityClassMetricV2.measured(
            metric=LabelQualityMetricNameV2.PRECISION,
            decision=LabelDecisionValueV2.MATCH,
            numerator=49,
            denominator=49,
        ),
        precision_interval=wilson_score_interval(49, 49),
        recall=LabelQualityClassMetricV2.measured(
            metric=LabelQualityMetricNameV2.RECALL,
            decision=LabelDecisionValueV2.MATCH,
            numerator=49,
            denominator=50,
        ),
        recall_interval=wilson_score_interval(49, 50),
        outcome=LabelQualityLabelOutcomeV2.FAILED,
        reason_codes=(LabelQualityReasonCodeV2.STRUCTURED_THRESHOLD_NOT_MET,),
    )
    forged_report = LabelQualityEvaluationReportV2.create(
        policy_ref=report.policy_ref,
        evidence_class=report.evidence_class,
        prerequisite=report.prerequisite,
        observation_closure_sha256=report.observation_closure_sha256,
        result_closure_sha256=report.result_closure_sha256,
        structured_summaries=(
            forged_summary,
            report.structured_summaries[1],
        ),
        semantic_summaries=report.semantic_summaries,
        audit=report.audit,
    )

    with pytest.raises(LabelQualityStoreIntegrityError, match="private evaluation"):
        _stores(tmp_path / "stores").persist(
            acceptance_key="label-quality-acceptance://r8-02/forged-report",
            request_sha256=hashlib.sha256(b"forged-report").hexdigest(),
            compilation=compilation,
            report=forged_report,
            audit=fixture_audit(),
        )


def test_pending_first_authority_contains_no_private_closure(
    tmp_path: Path,
) -> None:
    _persistence, specs, frozen = frozen_authority(tmp_path / "source")
    policy = quality_policy(
        frozen,
        specs,
        evidence_class=LabelQualityEvidenceClassV2.REPOSITORY_PENDING_ONLY,
    )
    compilation = LabelQualityEvaluationBuilder().compile_repository_pending(
        payload=R8_01_GOLD.read_bytes(),
        policy=policy,
    )
    report = LabelQualityEvaluator().evaluate(
        compilation=compilation,
        policy=policy,
        audit=fixture_audit(),
    )
    stores = _stores(tmp_path / "stores")

    accepted = stores.persist(
        acceptance_key="label-quality-acceptance://r8-02/repository-pending",
        request_sha256=hashlib.sha256(R8_01_GOLD.read_bytes()).hexdigest(),
        compilation=compilation,
        report=report,
        audit=fixture_audit(),
    )

    assert isinstance(accepted, LabelQualityAcceptedRunV1)
    assert accepted.observation_set_ref is None
    assert accepted.result_set_ref is None
    assert stores.get_accepted_report("label-quality-acceptance://r8-02/repository-pending") == report


def test_crash_before_acceptance_leaves_no_authority_and_replay_recovers(
    tmp_path: Path,
) -> None:
    compilation, report = _frozen(tmp_path)
    crashing = _stores(
        tmp_path / "stores",
        fault_points=frozenset({LabelQualityStoreFaultPoint.BEFORE_ACCEPTANCE_INDEX}),
    )
    with pytest.raises(LabelQualityStoreInjectedCrash):
        crashing.persist(
            acceptance_key="label-quality-acceptance://r8-02/crash",
            request_sha256=hashlib.sha256(b"crash").hexdigest(),
            compilation=compilation,
            report=report,
            audit=fixture_audit(),
        )

    clean = _stores(tmp_path / "stores")
    with pytest.raises(LabelQualityStoreIntegrityError, match="not found"):
        clean.get_accepted_report("label-quality-acceptance://r8-02/crash")
    accepted = clean.persist(
        acceptance_key="label-quality-acceptance://r8-02/crash",
        request_sha256=hashlib.sha256(b"crash").hexdigest(),
        compilation=compilation,
        report=report,
        audit=fixture_audit(),
    )
    assert accepted.report_ref == report.to_ref()


def test_corruption_and_symlink_roots_fail_closed(
    tmp_path: Path,
) -> None:
    compilation, report = _frozen(tmp_path)
    stores = _stores(tmp_path / "stores")
    stores.persist(
        acceptance_key="label-quality-acceptance://r8-02/corrupt",
        request_sha256=hashlib.sha256(b"corrupt").hexdigest(),
        compilation=compilation,
        report=report,
        audit=fixture_audit(),
    )
    report_envelope = next((tmp_path / "stores/report/envelopes").rglob("*.json"))
    report_envelope.write_text("{}")
    with pytest.raises(LabelQualityStoreIntegrityError):
        stores.get_accepted_report("label-quality-acceptance://r8-02/corrupt")

    real = tmp_path / "real-root"
    real.mkdir()
    link = tmp_path / "linked-root"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(LabelQualityStoreTypeError):
        LabelQualityMaterialStore(
            link,
            max_private_bytes=1_000,
            max_members=300,
        )
