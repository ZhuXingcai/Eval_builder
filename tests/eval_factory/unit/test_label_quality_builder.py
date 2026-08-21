from __future__ import annotations

import json
from pathlib import Path

import pytest
from label_quality_fixtures import (
    candidates,
    decision,
    fixture_audit,
    freeze_source,
    frozen_authority,
    frozen_request,
    label_specs,
    observation_set,
    quality_policy,
    service,
)
from pydantic import ValidationError
from test_independent_label_test_set_freeze import _candidate_pool

from eval_factory.contracts.label_quality_v2 import (
    LABEL_QUALITY_REPOSITORY_PENDING_SHA256,
    LabelQualityEvidenceClassV2,
    LabelQualityPrerequisiteOutcomeV2,
)
from eval_factory.contracts.labeling_v2 import (
    LabelDecisionValueV2,
    LabelExecutionStatus,
    LabelUnresolvedReason,
)
from eval_factory.statistics.label_quality_builder import (
    LabelQualityEvaluationAuthorizationError,
    LabelQualityEvaluationBuilder,
    LabelQualityEvaluationPolicyError,
)
from eval_factory.statistics.label_quality_models import (
    LabelQualityPairStatusV1,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
R8_01_GOLD = (
    REPO_ROOT / "evals/golden/eval_factory/statistics/r8-01-independent-label-test-set-freeze-v1.json"
)


def test_repository_pending_admission_binds_exact_gold_without_private_access(
    tmp_path: Path,
) -> None:
    persistence, specs, frozen = frozen_authority(tmp_path)
    del persistence
    policy = quality_policy(
        frozen,
        specs,
        evidence_class=LabelQualityEvidenceClassV2.REPOSITORY_PENDING_ONLY,
    )

    compilation = LabelQualityEvaluationBuilder().compile_repository_pending(
        payload=R8_01_GOLD.read_bytes(),
        policy=policy,
    )

    assert compilation.prerequisite.outcome is LabelQualityPrerequisiteOutcomeV2.STATISTICAL_GATE_PENDING
    assert compilation.prerequisite.available_semantic_count == 91
    assert compilation.prerequisite.required_semantic_count == 100
    assert compilation.prerequisite.dataset_manifest_ref is None
    assert compilation.prerequisite.access_receipt_ref is None
    assert compilation.observation_set is None
    assert compilation.result_set is None
    assert compilation.evidence_sha256 == LABEL_QUALITY_REPOSITORY_PENDING_SHA256

    payload = json.loads(R8_01_GOLD.read_text())
    payload["real_corpus"]["eligible_source_count"] = 92
    with pytest.raises(LabelQualityEvaluationPolicyError, match="approved"):
        LabelQualityEvaluationBuilder().compile_repository_pending(
            payload=json.dumps(payload).encode(),
            policy=policy,
        )


def test_typed_r8_01_pending_result_is_reportable_without_private_access(
    tmp_path: Path,
) -> None:
    specs = label_specs()
    persistence = service(tmp_path)
    source = freeze_source(specs)
    all_candidates = candidates(specs)
    semantic_ref = all_candidates[-1].label_spec_ref
    selected = tuple(candidate for candidate in all_candidates if candidate.label_spec_ref != semantic_ref)
    selected += tuple(candidate for candidate in all_candidates if candidate.label_spec_ref == semantic_ref)[
        :91
    ]
    source["candidate_pool"] = _candidate_pool(selected)
    pending = persistence.freeze(
        idempotency_key="r8-02-pending",
        **source,  # type: ignore[arg-type]
    )
    policy = quality_policy(pending, specs)

    compilation = LabelQualityEvaluationBuilder().compile_pending_result(
        expected_result=pending,
        policy=policy,
    )

    assert compilation.prerequisite.outcome is LabelQualityPrerequisiteOutcomeV2.STATISTICAL_GATE_PENDING
    assert compilation.prerequisite.freeze_result_ref == pending.to_ref()
    assert compilation.prerequisite.repository_evidence_ref is None
    assert compilation.prerequisite.shortage_count == 1
    assert compilation.prerequisite.available_semantic_count == 91
    assert compilation.observation_set is None
    assert compilation.result_set is None


def test_observation_set_is_order_and_audit_stable_and_rejects_duplicate_pairs(
    tmp_path: Path,
) -> None:
    _persistence, specs, frozen = frozen_authority(tmp_path)
    manifest = frozen.dataset_manifest
    assert manifest is not None
    material = _persistence.material_store.get_test_set(manifest.private_material_ref)
    first = observation_set(frozen, specs, material.members)
    second = first.create(
        dataset_manifest_ref=manifest.to_ref(),
        label_specs=tuple(reversed(specs)),
        decisions=tuple(reversed(first.decisions)),
        audit=fixture_audit().model_copy(update={"created_by": "r8-02-other-actor"}),
    )

    assert first.to_ref() == second.to_ref()
    assert first.canonical_sha256() != second.canonical_sha256()
    with pytest.raises(ValidationError, match="duplicate"):
        first.create(
            dataset_manifest_ref=manifest.to_ref(),
            label_specs=specs,
            decisions=(first.decisions[0], first.decisions[0]),
            audit=fixture_audit(),
        )


def test_frozen_builder_uses_current_r8_01_access_and_pairs_all_members(
    tmp_path: Path,
) -> None:
    persistence, specs, frozen = frozen_authority(tmp_path)
    manifest = frozen.dataset_manifest
    assert manifest is not None
    material = persistence.material_store.get_test_set(manifest.private_material_ref)
    observations = observation_set(frozen, specs, material.members)
    request = frozen_request(frozen, specs, observations)

    compilation = LabelQualityEvaluationBuilder().compile_frozen(
        persistence=persistence,
        expected_result=frozen,
        policy=quality_policy(frozen, specs),
        request=request,
        audit=fixture_audit(),
    )

    assert compilation.prerequisite.outcome is LabelQualityPrerequisiteOutcomeV2.FROZEN
    assert compilation.prerequisite.material_verified is True
    assert compilation.prerequisite.selected_member_count == 300
    assert compilation.observation_set == observations
    assert compilation.result_set is not None
    assert len(compilation.result_set.pair_results) == 300
    assert {pair.status for pair in compilation.result_set.pair_results} == {
        LabelQualityPairStatusV1.FINAL_MATCH,
        LabelQualityPairStatusV1.FINAL_NO_MATCH,
        LabelQualityPairStatusV1.VALID_ABSTAIN,
    }


def test_missing_observation_is_pending_evidence_but_extra_is_policy_error(
    tmp_path: Path,
) -> None:
    persistence, specs, frozen = frozen_authority(tmp_path)
    manifest = frozen.dataset_manifest
    assert manifest is not None
    material = persistence.material_store.get_test_set(manifest.private_material_ref)
    missing = observation_set(frozen, specs, material.members, omit_last=True)

    compilation = LabelQualityEvaluationBuilder().compile_frozen(
        persistence=persistence,
        expected_result=frozen,
        policy=quality_policy(frozen, specs),
        request=frozen_request(frozen, specs, missing),
        audit=fixture_audit(),
    )

    assert compilation.result_set is not None
    assert (
        sum(value.status is LabelQualityPairStatusV1.MISSING for value in compilation.result_set.pair_results)
        == 1
    )

    outsider = material.members[0].model_copy(
        update={
            "trace_envelope_ref": material.members[0].trace_envelope_ref.model_copy(
                update={"object_id": "trace-envelope://r8-02/outsider"}
            ),
            "source_trace_id": "source-trace://r8-02/outsider",
        }
    )
    extra = decision(
        outsider,
        specs[0],
        observed=LabelDecisionValueV2.MATCH,
    )
    with pytest.raises(LabelQualityEvaluationPolicyError, match="outside frozen"):
        observations = observation_set(
            frozen,
            specs,
            material.members,
            extra_decisions=(extra,),
        )
        LabelQualityEvaluationBuilder().compile_frozen(
            persistence=persistence,
            expected_result=frozen,
            policy=quality_policy(frozen, specs),
            request=frozen_request(frozen, specs, observations),
            audit=fixture_audit(),
        )


def test_semantic_model_unavailable_abstain_is_incomplete_not_valid_class(
    tmp_path: Path,
) -> None:
    persistence, specs, frozen = frozen_authority(tmp_path)
    manifest = frozen.dataset_manifest
    assert manifest is not None
    material = persistence.material_store.get_test_set(manifest.private_material_ref)
    recovery = specs[2]
    target = next(
        member
        for member in material.members
        if member.label_spec_ref.object_id == recovery.label_spec_id
        and member.expected_decision is LabelDecisionValueV2.ABSTAIN
    )
    base = observation_set(frozen, specs, material.members)
    unavailable = decision(
        target,
        recovery,
        observed=LabelDecisionValueV2.ABSTAIN,
        model_unavailable=True,
    )
    decisions = tuple(
        unavailable
        if value.trace_envelope_ref == target.trace_envelope_ref
        and value.label_spec_ref == target.label_spec_ref
        else value
        for value in base.decisions
    )
    changed = base.create(
        dataset_manifest_ref=manifest.to_ref(),
        label_specs=specs,
        decisions=decisions,
        audit=fixture_audit(),
    )

    compilation = LabelQualityEvaluationBuilder().compile_frozen(
        persistence=persistence,
        expected_result=frozen,
        policy=quality_policy(frozen, specs),
        request=frozen_request(frozen, specs, changed),
        audit=fixture_audit(),
    )

    assert compilation.result_set is not None
    pair = next(
        value
        for value in compilation.result_set.pair_results
        if value.reference.trace_envelope_ref == target.trace_envelope_ref
        and value.reference.label_spec_ref == target.label_spec_ref
    )
    assert pair.status is LabelQualityPairStatusV1.MODEL_UNAVAILABLE


def test_builder_rejects_stale_policy_and_frozen_authority(
    tmp_path: Path,
) -> None:
    persistence, specs, frozen = frozen_authority(tmp_path)
    policy = quality_policy(frozen, specs)
    with pytest.raises(LabelQualityEvaluationPolicyError, match="policy"):
        LabelQualityEvaluationBuilder().compile_repository_pending(
            payload=R8_01_GOLD.read_bytes(),
            policy=policy.model_copy(update={"policy_sha256": "f" * 64}),
        )

    manifest = frozen.dataset_manifest
    assert manifest is not None
    material = persistence.material_store.get_test_set(manifest.private_material_ref)
    observations = observation_set(frozen, specs, material.members)
    request = frozen_request(frozen, specs, observations)
    with pytest.raises(LabelQualityEvaluationPolicyError, match="authority"):
        LabelQualityEvaluationBuilder().compile_frozen(
            persistence=persistence,
            expected_result=frozen,
            policy=policy,
            request=request.model_copy(update={"dataset_series_id": "independent-label-test-set://other"}),
            audit=fixture_audit(),
        )


def test_builder_rejects_stale_label_spec_and_denied_access(
    tmp_path: Path,
) -> None:
    persistence, specs, frozen = frozen_authority(tmp_path)
    manifest = frozen.dataset_manifest
    assert manifest is not None
    material = persistence.material_store.get_test_set(manifest.private_material_ref)
    observations = observation_set(frozen, specs, material.members)
    request = frozen_request(frozen, specs, observations)
    stale = specs[0].model_copy(update={"requirement": "changed without rehash"})
    with pytest.raises(LabelQualityEvaluationPolicyError, match="behavior hash"):
        LabelQualityEvaluationBuilder().compile_frozen(
            persistence=persistence,
            expected_result=frozen,
            policy=quality_policy(frozen, specs),
            request=request.model_copy(update={"label_specs": (stale, *specs[1:])}),
            audit=fixture_audit(),
        )

    denied_principal = request.principal.model_copy(
        update={
            "principal_ref": request.principal.principal_ref.model_copy(
                update={"object_id": "statistical-principal://r8-02/other"}
            )
        }
    )
    with pytest.raises(LabelQualityEvaluationAuthorizationError, match="access"):
        LabelQualityEvaluationBuilder().compile_frozen(
            persistence=persistence,
            expected_result=frozen,
            policy=quality_policy(frozen, specs),
            request=request.model_copy(update={"principal": denied_principal}),
            audit=fixture_audit(),
        )


def test_pair_classifier_fails_closed_on_versions_and_distinguishes_abstains(
    tmp_path: Path,
) -> None:
    persistence, specs, frozen = frozen_authority(tmp_path)
    manifest = frozen.dataset_manifest
    assert manifest is not None
    material = persistence.material_store.get_test_set(manifest.private_material_ref)
    policy = quality_policy(frozen, specs)
    structured = specs[0]
    structured_candidate = next(
        value for value in material.members if value.label_spec_ref.object_id == structured.label_spec_id
    )
    structured_abstain = decision(
        structured_candidate,
        structured,
        observed=LabelDecisionValueV2.ABSTAIN,
    )
    assert (
        LabelQualityEvaluationBuilder._classify(
            reference=structured_candidate,
            decision=structured_abstain,
            spec=structured,
            policy=policy,
        )
        is LabelQualityPairStatusV1.ABSTAIN
    )

    semantic = specs[2]
    semantic_candidate = next(
        value
        for value in material.members
        if value.label_spec_ref.object_id == semantic.label_spec_id
        and value.expected_decision is LabelDecisionValueV2.ABSTAIN
    )
    valid = decision(
        semantic_candidate,
        semantic,
        observed=LabelDecisionValueV2.ABSTAIN,
    )
    invalid_abstain = valid.model_copy(update={"semantic_evidence": ()})
    with pytest.raises(LabelQualityEvaluationPolicyError, match="configuration"):
        LabelQualityEvaluationBuilder._classify(
            reference=semantic_candidate,
            decision=invalid_abstain,
            spec=semantic,
            policy=policy,
        )
    unproven_abstain = invalid_abstain.model_copy(
        update={
            "model_profile": None,
            "prompt_version": None,
        }
    )
    assert (
        LabelQualityEvaluationBuilder._classify(
            reference=semantic_candidate,
            decision=unproven_abstain,
            spec=semantic,
            policy=policy,
        )
        is LabelQualityPairStatusV1.INCOMPLETE
    )
    inspection_only = valid.model_copy(
        update={"unresolved_reasons": frozenset({LabelUnresolvedReason.USER_INSPECTION_REQUIRED})}
    )
    assert (
        LabelQualityEvaluationBuilder._classify(
            reference=semantic_candidate,
            decision=inspection_only,
            spec=semantic,
            policy=policy,
        )
        is LabelQualityPairStatusV1.INCOMPLETE
    )
    nonfinal_match = decision(
        semantic_candidate,
        semantic,
        observed=LabelDecisionValueV2.MATCH,
    ).model_copy(
        update={
            "execution_status": LabelExecutionStatus.UNRESOLVED,
            "unresolved_reasons": frozenset({LabelUnresolvedReason.LOW_CONFIDENCE}),
        }
    )
    assert (
        LabelQualityEvaluationBuilder._classify(
            reference=semantic_candidate,
            decision=nonfinal_match,
            spec=semantic,
            policy=policy,
        )
        is LabelQualityPairStatusV1.INCOMPLETE
    )
    with pytest.raises(LabelQualityEvaluationPolicyError, match="rule version"):
        LabelQualityEvaluationBuilder._classify(
            reference=semantic_candidate,
            decision=valid.model_copy(update={"rule_version": "wrong-rule/v1"}),
            spec=semantic,
            policy=policy,
        )
    with pytest.raises(LabelQualityEvaluationPolicyError, match="configuration"):
        LabelQualityEvaluationBuilder._classify(
            reference=semantic_candidate,
            decision=valid.model_copy(update={"prompt_version": "wrong-prompt/v1"}),
            spec=semantic,
            policy=policy,
        )
    with pytest.raises(LabelQualityEvaluationPolicyError, match="configuration"):
        LabelQualityEvaluationBuilder._classify(
            reference=structured_candidate,
            decision=decision(
                structured_candidate,
                structured,
                observed=LabelDecisionValueV2.MATCH,
            ).model_copy(
                update={
                    "model_profile": "unbound-model",
                    "prompt_version": "unbound-prompt/v1",
                }
            ),
            spec=structured,
            policy=policy,
        )
    with pytest.raises(LabelQualityEvaluationPolicyError, match="model-unavailable"):
        LabelQualityEvaluationBuilder._classify(
            reference=semantic_candidate,
            decision=valid.model_copy(
                update={
                    "decision": LabelDecisionValueV2.MATCH,
                    "semantic_evidence": (),
                    "model_profile": None,
                    "prompt_version": None,
                    "unresolved_reasons": frozenset({LabelUnresolvedReason.MODEL_UNAVAILABLE}),
                }
            ),
            spec=semantic,
            policy=policy,
        )
    with pytest.raises(LabelQualityEvaluationPolicyError, match="trace-label"):
        LabelQualityEvaluationBuilder._classify(
            reference=semantic_candidate,
            decision=valid.model_copy(
                update={
                    "trace_envelope_ref": valid.trace_envelope_ref.model_copy(
                        update={"object_id": "trace-envelope://r8-02/wrong"}
                    )
                }
            ),
            spec=semantic,
            policy=policy,
        )
