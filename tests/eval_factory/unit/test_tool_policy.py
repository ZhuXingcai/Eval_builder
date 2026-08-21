from __future__ import annotations

import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from eval_factory.contracts import (
    EvaluatorExecutionModeV2,
    RubricCriterionV2,
    RubricJudgedObjectKindV2,
    RubricJudgedObjectV2,
    RubricReachabilityV2,
    RubricSetV2,
    RubricSourceModeV2,
    TaskDraftPromptSafetyStatusV2,
    TaskDraftV2,
    TaskRequirementLineageV2,
    ToolDenyReasonV2,
    ToolRuleActionV2,
    evaluator_spec_ref,
    rubric_criterion_carried_sha256,
    rubric_reachability_carried_sha256,
    rubric_set_carried_sha256,
    task_draft_carried_sha256,
    task_draft_ref,
    tool_policy_carried_sha256,
    tool_rule_carried_sha256,
)
from eval_factory.contracts.core import (
    ContractAudit,
    EvidencePolarity,
    EvidenceRef,
    ObjectRef,
    SourceSpanRef,
    VersionBinding,
)
from eval_factory.contracts.task import (
    AttachmentCriticality,
    EvidencePriority,
    ReferenceMode,
    RequirementConflict,
    RubricVisibility,
)
from eval_factory.contracts.trace import ToolFamily
from eval_factory.provenance import (
    PromptBoundarySourceRole,
    PromptBoundarySurface,
    PromptInjectionBoundaryEnforcer,
)
from eval_factory.task_authoring import (
    EvaluationContractCompiler,
    EvaluationContractOutcome,
    EvaluatorBindingDefinition,
    ToolCapabilityCatalog,
    ToolCapabilityDefinition,
    ToolPolicyCompilationOutcome,
    ToolPolicyCompiler,
    ToolPolicyPolicyError,
    tool_capability_catalog_carried_sha256,
    tool_capability_definition_carried_sha256,
)

HASH = "a" * 64
OTHER_HASH = "b" * 64
THIRD_HASH = "c" * 64
EVALUATOR_BINDING = "evaluator-binding://tool-policy/default"
ROOT = Path(__file__).resolve().parents[3]


def _audit(
    created_at: datetime = datetime(2026, 7, 27, tzinfo=UTC),
    *,
    input_refs: tuple[ObjectRef, ...] = (),
) -> ContractAudit:
    return ContractAudit(
        created_at=created_at,
        created_by="tool-policy-test",
        governing_versions=(VersionBinding(component="tool-policy", version="r4-07"),),
        input_refs=input_refs,
    )


def _ref(
    object_type: str,
    suffix: str,
    *,
    digest: str = HASH,
    version: str = "v1",
) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://{suffix}",
        object_version=version,
        object_sha256=digest,
    )


def _evidence(suffix: str) -> EvidenceRef:
    return EvidenceRef(
        evidence_ref_id=f"evidence-ref://tool-policy/{suffix}",
        subject_ref=_ref("file-version-projection", suffix),
        source_spans=(
            SourceSpanRef(
                span_id=f"source-span://tool-policy/{suffix}",
                source_trace_id="source-trace://tool-policy",
                raw_sha256=HASH,
            ),
        ),
        polarity=EvidencePolarity.POSITIVE,
        capability="tool-policy",
        capability_complete=True,
    )


def _lineage() -> TaskRequirementLineageV2:
    return TaskRequirementLineageV2(
        requirement_id="requirement://tool-policy/visible",
        statement="Inspect the visible input-state workspace.",
        criticality=AttachmentCriticality.CRITICAL,
        evidence_priority=EvidencePriority.EXPLICIT_REQUIREMENT,
        evidence=(_evidence("visible"),),
        task_episode_refs=(_ref("task-episode", "primary", version="v2"),),
        conflict_status=RequirementConflict.NONE,
    )


def _draft(
    *,
    allowed_tools: tuple[str, ...] = ("file-read",),
    prompt_safety_status: TaskDraftPromptSafetyStatusV2 = (TaskDraftPromptSafetyStatusV2.PENDING),
    audit: ContractAudit | None = None,
) -> TaskDraftV2:
    gate_ref = (
        None
        if prompt_safety_status is TaskDraftPromptSafetyStatusV2.PENDING
        else _ref("task-prompt-safety-gate", "tool-policy", version="v2")
    )
    draft = TaskDraftV2(
        task_draft_id="task-draft://pending",
        task_version=1,
        supersedes_task_draft_ref=None,
        selection_context_ref=_ref("selection-context", "tool-policy", version="v2"),
        task_episode_refs=(_ref("task-episode", "primary", version="v2"),),
        visible_prompt="Inspect the workspace using only the explicitly allowed tools.",
        task_intent="Evaluate an input-state workspace inspection.",
        evaluation_claim="The task is solvable with the exact declared tool set.",
        required_capabilities=("workspace-inspection",),
        allowed_tools=allowed_tools,
        forbidden_outputs=("original final answer", "private grader controls"),
        attachment_dependencies=(),
        requirement_lineage=(_lineage(),),
        prompt_requirement_ids=("requirement://tool-policy/visible",),
        uncertainties=(),
        prompt_safety_status=prompt_safety_status,
        prompt_safety_gate_ref=gate_ref,
        model_profile="internal-task-author-v1",
        prompt_version="task-draft-authoring-prompt/v1",
        policy_version="task-draft-authoring/r4-03-v1",
        task_draft_sha256=HASH,
        audit=audit or _audit(),
    )
    digest = task_draft_carried_sha256(draft)
    return draft.model_copy(
        update={
            "task_draft_id": f"task-draft://sha256/{digest}",
            "task_draft_sha256": digest,
        }
    )


def _criterion(
    *,
    allowed_tool_ids: tuple[str, ...] = ("file-read",),
) -> RubricCriterionV2:
    reachability = RubricReachabilityV2(
        reachability_id="rubric-reachability://pending",
        prompt_requirement_ids=("requirement://tool-policy/visible",),
        attachment_dependency_ids=(),
        allowed_tool_ids=allowed_tool_ids,
        evidence=(_evidence("visible"),),
        capability_complete=True,
        reachability_sha256=HASH,
    )
    reachability_digest = rubric_reachability_carried_sha256(reachability)
    reachability = reachability.model_copy(
        update={
            "reachability_id": f"rubric-reachability://sha256/{reachability_digest}",
            "reachability_sha256": reachability_digest,
        }
    )
    criterion = RubricCriterionV2(
        criterion_id="rubric-criterion://pending",
        judged_object=RubricJudgedObjectV2(
            judged_object_id="judged-object://tool-policy/behavior",
            kind=RubricJudgedObjectKindV2.TOOL_BEHAVIOR,
            description="The contestant tool behavior visible to the evaluator.",
        ),
        description="Uses only the exact allowed tool capability.",
        weight=1.0,
        reachability=reachability,
        visibility=RubricVisibility.EVALUATOR_ONLY,
        evaluator_binding=EVALUATOR_BINDING,
        approval_status="CANDIDATE",
        criterion_sha256=HASH,
    )
    criterion_digest = rubric_criterion_carried_sha256(criterion)
    return criterion.model_copy(
        update={
            "criterion_id": f"rubric-criterion://sha256/{criterion_digest}",
            "criterion_sha256": criterion_digest,
        }
    )


def _rubric_set(
    task_draft: TaskDraftV2,
    *,
    allowed_tool_ids: tuple[str, ...] | None = None,
    audit: ContractAudit | None = None,
) -> RubricSetV2:
    tools = task_draft.allowed_tools if allowed_tool_ids is None else allowed_tool_ids
    criterion = _criterion(allowed_tool_ids=tools)
    rubric = RubricSetV2(
        rubric_set_id="rubric-set://pending",
        rubric_version=1,
        supersedes_rubric_set_ref=None,
        task_draft_ref=task_draft_ref(task_draft),
        source_mode=RubricSourceModeV2.GENERATED,
        criteria=(criterion,),
        total_weight=criterion.weight,
        imported_from_ref=None,
        model_profile="internal-rubric-author-v1",
        prompt_version="rubric-authoring-prompt/v1",
        policy_version="rubric-authoring/r4-05-v1",
        rubric_set_sha256=HASH,
        audit=audit or _audit(),
    )
    digest = rubric_set_carried_sha256(rubric)
    return rubric.model_copy(
        update={
            "rubric_set_id": f"rubric-set://sha256/{digest}",
            "rubric_set_sha256": digest,
        }
    )


def _evaluator_spec(rubric_set: RubricSetV2):
    compiled = EvaluationContractCompiler().compile(
        rubric_set=rubric_set,
        binding_definitions=(
            EvaluatorBindingDefinition(
                evaluator_binding_id=EVALUATOR_BINDING,
                evaluator_type="contract-evaluator",
                execution_mode=EvaluatorExecutionModeV2.DETERMINISTIC,
                evaluator_version="r4-06-v1",
                input_contract_ref=_ref(
                    "evaluator-input-contract",
                    "tool-policy",
                ),
                output_contract_ref=_ref(
                    "evaluator-output-contract",
                    "tool-policy",
                ),
                evaluator_principal_id="principal://evaluator/tool-policy",
                model_profile_ref=None,
                reference_refs=(),
                timeout_seconds=300,
            ),
        ),
        reference_mode=ReferenceMode.NONE,
        audit=_audit(),
    )
    assert compiled.outcome is EvaluationContractOutcome.COMPILED
    assert compiled.evaluator_spec is not None
    return compiled.evaluator_spec


def _definition(
    tool_id: str,
    family: ToolFamily,
    *,
    contestant_eligible: bool = True,
    digest: str = HASH,
    **overrides: object,
) -> ToolCapabilityDefinition:
    values: dict[str, object] = {
        "definition_id": "tool-capability-definition://pending",
        "tool_id": tool_id,
        "tool_family": family,
        "capability_ref": _ref(
            "tool-capability",
            tool_id,
            digest=digest,
        ),
        "enforcement_profile_ref": _ref(
            "tool-enforcement-profile",
            tool_id,
            digest=digest,
        ),
        "constraint_profile_ref": _ref(
            "tool-constraint-profile",
            tool_id,
            digest=digest,
        ),
        "contestant_descriptor_ref": _ref(
            "contestant-tool-descriptor",
            tool_id,
            digest=digest,
        ),
        "contestant_constraint_profile_ref": _ref(
            "contestant-tool-constraint-profile",
            tool_id,
            digest=digest,
        ),
        "contestant_eligible": contestant_eligible,
        "definition_sha256": HASH,
    }
    values.update(overrides)
    definition = ToolCapabilityDefinition(**values)
    definition_digest = tool_capability_definition_carried_sha256(definition)
    return definition.model_copy(
        update={
            "definition_id": (f"tool-capability-definition://sha256/{definition_digest}"),
            "definition_sha256": definition_digest,
        }
    )


def _catalog(
    definitions: tuple[ToolCapabilityDefinition, ...] | None = None,
    *,
    version: str = "tool-capability-catalog/r4-07-v1",
    audit: ContractAudit | None = None,
) -> ToolCapabilityCatalog:
    if definitions is None:
        resolved = (
            _definition("file-read", ToolFamily.FILE_READ),
            _definition("shell", ToolFamily.SHELL, digest=OTHER_HASH),
            _definition(
                "internal-admin",
                ToolFamily.UNKNOWN,
                contestant_eligible=False,
                digest=THIRD_HASH,
            ),
        )
    else:
        resolved = definitions
    catalog = ToolCapabilityCatalog(
        catalog_id="tool-capability-catalog://pending",
        catalog_version=version,
        definitions=resolved,
        catalog_sha256=HASH,
        audit=audit or _audit(),
    )
    digest = tool_capability_catalog_carried_sha256(catalog)
    return catalog.model_copy(
        update={
            "catalog_id": f"tool-capability-catalog://sha256/{digest}",
            "catalog_sha256": digest,
        }
    )


def _inputs(
    *,
    allowed_tools: tuple[str, ...] = ("file-read",),
    rubric_tools: tuple[str, ...] | None = None,
    catalog: ToolCapabilityCatalog | None = None,
    audit: ContractAudit | None = None,
):
    draft = _draft(allowed_tools=allowed_tools, audit=audit)
    rubric = _rubric_set(
        draft,
        allowed_tool_ids=allowed_tools if rubric_tools is None else rubric_tools,
        audit=audit,
    )
    return draft, rubric, _evaluator_spec(rubric), catalog or _catalog()


def _compile(
    *,
    allowed_tools: tuple[str, ...] = ("file-read",),
    rubric_tools: tuple[str, ...] | None = None,
    catalog: ToolCapabilityCatalog | None = None,
    audit: ContractAudit | None = None,
):
    draft, rubric, evaluator, resolved_catalog = _inputs(
        allowed_tools=allowed_tools,
        rubric_tools=rubric_tools,
        catalog=catalog,
        audit=audit,
    )
    result = ToolPolicyCompiler().compile(
        task_draft=draft,
        rubric_set=rubric,
        evaluator_spec=evaluator,
        catalog=resolved_catalog,
        audit=audit or _audit(),
    )
    return draft, rubric, evaluator, resolved_catalog, result


def test_static_catalog_requires_exact_content_free_definitions() -> None:
    definition = _definition("file-read", ToolFamily.FILE_READ)

    assert definition.definition_sha256 == (tool_capability_definition_carried_sha256(definition))
    with pytest.raises(ValidationError, match="UNKNOWN"):
        _definition(
            "unknown",
            ToolFamily.UNKNOWN,
            contestant_eligible=True,
        )
    with pytest.raises(ValidationError, match="capability_ref"):
        _definition(
            "file-read",
            ToolFamily.FILE_READ,
            capability_ref=_ref("runtime-credential", "forbidden"),
        )
    with pytest.raises(ValidationError):
        ToolCapabilityDefinition.model_validate(
            {
                **definition.model_dump(mode="json"),
                "vendor_tool_name": "Read",
            }
        )
    with pytest.raises(ValidationError, match="content-free"):
        _definition(
            "file-read",
            ToolFamily.FILE_READ,
            contestant_descriptor_ref=_ref(
                "contestant-tool-descriptor",
                "private-reference",
            ),
        )


def test_catalog_rejects_empty_duplicate_and_stale_definitions() -> None:
    with pytest.raises(ValidationError):
        _catalog(definitions=())

    definition = _definition("file-read", ToolFamily.FILE_READ)
    with pytest.raises(ValidationError, match="tool IDs"):
        _catalog(
            definitions=(
                definition,
                _definition(
                    "file-read",
                    ToolFamily.FILE_READ,
                    digest=OTHER_HASH,
                ),
            )
        )

    stale = definition.model_copy(update={"definition_sha256": OTHER_HASH})
    catalog = _catalog(definitions=(stale,))
    draft, rubric, evaluator, _ = _inputs(catalog=catalog)
    with pytest.raises(ToolPolicyPolicyError, match="definition"):
        ToolPolicyCompiler().compile(
            task_draft=draft,
            rubric_set=rubric,
            evaluator_spec=evaluator,
            catalog=catalog,
            audit=_audit(),
        )


def test_catalog_hash_is_stable_across_order_and_audit() -> None:
    first_definition = _definition("file-read", ToolFamily.FILE_READ)
    second_definition = _definition(
        "shell",
        ToolFamily.SHELL,
        digest=OTHER_HASH,
    )
    first = _catalog(
        (first_definition, second_definition),
        audit=_audit(datetime(2026, 7, 27, tzinfo=UTC)),
    )
    second = _catalog(
        (second_definition, first_definition),
        audit=_audit(datetime(2026, 7, 28, tzinfo=UTC)),
    )

    assert first.catalog_id == second.catalog_id
    assert first.catalog_sha256 == second.catalog_sha256
    assert first.canonical_sha256() != second.canonical_sha256()


def test_compiler_emits_complete_allow_deny_inventory_and_public_projection() -> None:
    draft, rubric, evaluator, catalog, result = _compile()

    assert result.outcome is ToolPolicyCompilationOutcome.COMPILED
    assert result.tool_policy is not None
    assert result.contestant_projection is not None
    assert result.unresolved_reasons == frozenset()
    policy = result.tool_policy
    projection = result.contestant_projection
    assert tuple(rule.tool_id for rule in policy.rules) == (
        "file-read",
        "internal-admin",
        "shell",
    )
    rule_by_tool = {rule.tool_id: rule for rule in policy.rules}
    assert rule_by_tool["file-read"].action is ToolRuleActionV2.ALLOW
    assert rule_by_tool["file-read"].deny_reason is None
    assert rule_by_tool["shell"].action is ToolRuleActionV2.DENY
    assert rule_by_tool["shell"].deny_reason is ToolDenyReasonV2.NOT_REQUIRED
    assert rule_by_tool["internal-admin"].action is ToolRuleActionV2.DENY
    assert rule_by_tool["internal-admin"].deny_reason is ToolDenyReasonV2.CATALOG_POLICY_DENIED
    assert all(rule.rule_sha256 == tool_rule_carried_sha256(rule) for rule in policy.rules)
    assert tuple(rule.tool_id for rule in projection.rules) == ("file-read",)
    assert projection.source_task_draft_sha256 == draft.task_draft_sha256
    assert policy.task_draft_ref == task_draft_ref(draft)
    assert policy.rubric_set_ref.object_sha256 == rubric.rubric_set_sha256
    assert policy.evaluator_spec_ref == evaluator_spec_ref(evaluator)
    assert policy.tool_catalog_ref.object_sha256 == catalog.catalog_sha256
    assert policy.control_boundary_enforcement_ref.object_type == ("prompt-boundary-enforcement")
    assert policy.tool_policy_sha256 == tool_policy_carried_sha256(policy)


def test_compiler_sends_only_exact_content_free_catalog_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = []
    original_enforce = PromptInjectionBoundaryEnforcer.enforce

    def capture_enforce(self, request):
        captured.append(request)
        return original_enforce(self, request)

    monkeypatch.setattr(
        PromptInjectionBoundaryEnforcer,
        "enforce",
        capture_enforce,
    )
    _, _, _, catalog, result = _compile()

    assert result.tool_policy is not None
    assert len(captured) == 1
    request = captured[0]
    assert len(request.segments) == 1
    segment = request.segments[0]
    assert segment.surface is PromptBoundarySurface.TOOL_POLICY_CONFIG
    assert segment.source_role is PromptBoundarySourceRole.STATIC_POLICY_CONFIG
    assert segment.source_ref.object_type == "tool-capability-catalog"
    assert segment.source_ref.object_sha256 == catalog.catalog_sha256
    assert segment.content_sha256 == catalog.catalog_sha256
    assert segment.untrusted_data_marker is False
    assert segment.text_preview is None
    assert result.tool_policy.control_boundary_enforcement_ref.object_type == ("prompt-boundary-enforcement")


def test_projection_discloses_only_allowed_public_descriptors() -> None:
    _, _, _, _, result = _compile()
    assert result.contestant_projection is not None

    serialized = str(result.contestant_projection.model_dump(mode="json"))
    assert "file-read" in serialized
    for forbidden in (
        "internal-admin",
        "shell",
        "tool-capability://",
        "tool-enforcement-profile",
        "CATALOG_POLICY_DENIED",
        "NOT_REQUIRED",
        "task-draft://",
        "rubric-set://",
        "evaluator-spec://",
        "prompt-boundary-enforcement://",
        "runtime-credential",
        "vendor",
        "command",
        "path",
    ):
        assert forbidden not in serialized
    assert all(
        rule.contestant_descriptor_ref.object_type == "contestant-tool-descriptor"
        and rule.contestant_constraint_profile_ref.object_type == "contestant-tool-constraint-profile"
        for rule in result.contestant_projection.rules
    )
    assert {ref.object_type for ref in result.contestant_projection.audit.input_refs} <= {
        "contestant-tool-descriptor",
        "contestant-tool-constraint-profile",
    }


def test_empty_allowed_tools_compile_to_explicit_deny_all() -> None:
    _, _, _, catalog, result = _compile(
        allowed_tools=(),
        rubric_tools=(),
    )

    assert result.outcome is ToolPolicyCompilationOutcome.COMPILED
    assert result.tool_policy is not None
    assert result.contestant_projection is not None
    assert len(result.tool_policy.rules) == len(catalog.definitions)
    assert all(rule.action is ToolRuleActionV2.DENY for rule in result.tool_policy.rules)
    assert result.tool_policy.default_action == "DENY"
    assert result.contestant_projection.rules == ()
    assert result.contestant_projection.default_action == "DENY"


def test_missing_catalog_definition_blocks_without_partial_output() -> None:
    catalog = _catalog(definitions=(_definition("shell", ToolFamily.SHELL),))
    _, _, _, _, result = _compile(catalog=catalog)

    assert result.outcome is ToolPolicyCompilationOutcome.BLOCKED_CAPABILITY
    assert result.tool_policy is None
    assert result.contestant_projection is None
    assert result.unresolved_reasons


@pytest.mark.parametrize(
    "definition",
    [
        pytest.param(
            _definition(
                "file-read",
                ToolFamily.FILE_READ,
                contestant_eligible=False,
            ),
            id="ineligible",
        ),
        pytest.param(
            _definition(
                "file-read",
                ToolFamily.UNKNOWN,
                contestant_eligible=False,
            ),
            id="unknown-family",
        ),
    ],
)
def test_requested_denied_catalog_tool_blocks_without_partial_output(
    definition: ToolCapabilityDefinition,
) -> None:
    catalog = _catalog(definitions=(definition,))
    _, _, _, _, result = _compile(catalog=catalog)

    assert result.outcome is ToolPolicyCompilationOutcome.BLOCKED_POLICY
    assert result.tool_policy is None
    assert result.contestant_projection is None
    assert result.unresolved_reasons


def test_compiler_rejects_cross_contract_and_stale_nested_inputs() -> None:
    draft, rubric, evaluator, catalog = _inputs()
    compiler = ToolPolicyCompiler()

    other_draft = _draft(allowed_tools=("shell",))
    with pytest.raises(ToolPolicyPolicyError, match="TaskDraft"):
        compiler.compile(
            task_draft=other_draft,
            rubric_set=rubric,
            evaluator_spec=evaluator,
            catalog=catalog,
            audit=_audit(),
        )

    wrong_rubric = _rubric_set(
        draft,
        allowed_tool_ids=("shell",),
    )
    with pytest.raises(ToolPolicyPolicyError, match="rubric"):
        compiler.compile(
            task_draft=draft,
            rubric_set=wrong_rubric,
            evaluator_spec=_evaluator_spec(wrong_rubric),
            catalog=catalog,
            audit=_audit(),
        )

    stale_reachability = rubric.criteria[0].reachability.model_copy(
        update={"reachability_sha256": OTHER_HASH}
    )
    stale_criterion = rubric.criteria[0].model_copy(update={"reachability": stale_reachability})
    stale_rubric = rubric.model_copy(update={"criteria": (stale_criterion,)})
    with pytest.raises(ToolPolicyPolicyError, match="reachability"):
        compiler.compile(
            task_draft=draft,
            rubric_set=stale_rubric,
            evaluator_spec=evaluator,
            catalog=catalog,
            audit=_audit(),
        )


def test_compiler_rejects_blocked_task_and_stale_catalog() -> None:
    blocked = _draft(prompt_safety_status=TaskDraftPromptSafetyStatusV2.BLOCKED)
    rubric = _rubric_set(blocked)
    with pytest.raises(ToolPolicyPolicyError, match="BLOCKED"):
        ToolPolicyCompiler().compile(
            task_draft=blocked,
            rubric_set=rubric,
            evaluator_spec=_evaluator_spec(rubric),
            catalog=_catalog(),
            audit=_audit(),
        )

    draft, rubric, evaluator, catalog = _inputs()
    with pytest.raises(ToolPolicyPolicyError, match="catalog"):
        ToolPolicyCompiler().compile(
            task_draft=draft,
            rubric_set=rubric,
            evaluator_spec=evaluator,
            catalog=catalog.model_copy(update={"catalog_sha256": OTHER_HASH}),
            audit=_audit(),
        )


def test_compiler_rebuilds_audits_from_validated_dependencies() -> None:
    caller_only = _ref("private-reference", "caller-only")
    _, _, _, _, result = _compile(audit=_audit(input_refs=(caller_only,)))
    assert result.tool_policy is not None
    assert result.contestant_projection is not None

    assert caller_only not in result.audit.input_refs
    assert caller_only not in result.tool_policy.audit.input_refs
    assert caller_only not in result.contestant_projection.audit.input_refs
    assert {ref.object_type for ref in result.tool_policy.audit.input_refs} == {
        "task-draft",
        "rubric-set",
        "evaluator-spec",
        "tool-capability-catalog",
        "prompt-boundary-enforcement",
        "contestant-tool-policy",
    }


def test_validate_current_accepts_exact_and_rejects_policy_or_projection_mutation() -> None:
    draft, rubric, evaluator, catalog, result = _compile()
    assert result.tool_policy is not None
    assert result.contestant_projection is not None
    compiler = ToolPolicyCompiler()

    compiler.validate_current(
        task_draft=draft,
        rubric_set=rubric,
        evaluator_spec=evaluator,
        catalog=catalog,
        tool_policy=result.tool_policy,
        contestant_projection=result.contestant_projection,
    )

    first_rule = result.tool_policy.rules[0]
    tampered_rule = first_rule.model_copy(
        update={"constraint_profile_ref": _ref("tool-constraint-profile", "tampered")}
    )
    tampered_policy = result.tool_policy.model_copy(
        update={"rules": (tampered_rule, *result.tool_policy.rules[1:])}
    )
    with pytest.raises(ToolPolicyPolicyError, match="current"):
        compiler.validate_current(
            task_draft=draft,
            rubric_set=rubric,
            evaluator_spec=evaluator,
            catalog=catalog,
            tool_policy=tampered_policy,
            contestant_projection=result.contestant_projection,
        )

    public_rule = result.contestant_projection.rules[0].model_copy(
        update={
            "contestant_descriptor_ref": _ref(
                "contestant-tool-descriptor",
                "tampered",
            )
        }
    )
    tampered_projection = result.contestant_projection.model_copy(update={"rules": (public_rule,)})
    with pytest.raises(ToolPolicyPolicyError, match="current"):
        compiler.validate_current(
            task_draft=draft,
            rubric_set=rubric,
            evaluator_spec=evaluator,
            catalog=catalog,
            tool_policy=result.tool_policy,
            contestant_projection=tampered_projection,
        )


def test_compile_is_stable_across_catalog_order_and_audit_time() -> None:
    definitions = _catalog().definitions
    first = _compile(
        catalog=_catalog(
            definitions,
            audit=_audit(datetime(2026, 7, 27, tzinfo=UTC)),
        ),
        audit=_audit(datetime(2026, 7, 27, tzinfo=UTC)),
    )[-1]
    second = _compile(
        catalog=_catalog(
            tuple(reversed(definitions)),
            audit=_audit(datetime(2026, 7, 28, tzinfo=UTC)),
        ),
        audit=_audit(datetime(2026, 7, 28, tzinfo=UTC)),
    )[-1]
    assert first.tool_policy is not None
    assert second.tool_policy is not None
    assert first.contestant_projection is not None
    assert second.contestant_projection is not None

    assert first.tool_policy.tool_policy_id == second.tool_policy.tool_policy_id
    assert (
        first.contestant_projection.contestant_tool_policy_id
        == second.contestant_projection.contestant_tool_policy_id
    )
    assert first.result_id == second.result_id


def test_r4_07_identities_are_stable_across_python_hash_seed() -> None:
    code = """
import runpy

ns = runpy.run_path("tests/eval_factory/unit/test_tool_policy.py")
result = ns["_compile"]()[-1]
policy = result.tool_policy
projection = result.contestant_projection
assert policy is not None and projection is not None
print(policy.tool_policy_id)
print(projection.contestant_tool_policy_id)
print(result.result_id)
"""
    outputs = []
    for seed in ("1", "99"):
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "PYTHONHASHSEED": seed,
                "PYTHONPATH": "src",
            },
        )
        assert completed.returncode == 0, completed.stderr
        outputs.append(completed.stdout.strip())

    assert len(set(outputs)) == 1
