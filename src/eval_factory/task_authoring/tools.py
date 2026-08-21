from __future__ import annotations

from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.task_v2 import (
    ContestantToolPolicyV2,
    ContestantToolRuleV2,
    EvaluatorSpecV2,
    RubricSetV2,
    TaskDraftPromptSafetyStatusV2,
    TaskDraftV2,
    ToolDenyReasonV2,
    ToolPolicyV2,
    ToolRuleActionV2,
    ToolRuleV2,
    contestant_tool_policy_carried_sha256,
    contestant_tool_policy_ref,
    contestant_tool_rule_carried_sha256,
    evaluator_binding_carried_sha256,
    evaluator_failure_rule_carried_sha256,
    evaluator_spec_carried_sha256,
    evaluator_spec_ref,
    rubric_criterion_carried_sha256,
    rubric_reachability_carried_sha256,
    rubric_set_carried_sha256,
    rubric_set_ref,
    task_draft_carried_sha256,
    task_draft_ref,
    tool_policy_carried_sha256,
    tool_policy_ref,
    tool_rule_carried_sha256,
)
from eval_factory.contracts.trace import ToolFamily
from eval_factory.provenance.injection import (
    PromptBoundaryEnforcementRequest,
    PromptBoundaryEnforcementResult,
    PromptBoundarySegment,
    PromptBoundarySourceRole,
    PromptBoundarySurface,
    PromptInjectionBoundaryEnforcer,
    PromptInjectionBoundaryPolicyError,
)
from eval_factory.task_authoring.tool_models import (
    TOOL_POLICY_POLICY_VERSION,
    ToolCapabilityCatalog,
    ToolCapabilityDefinition,
    ToolPolicyCompilationOutcome,
    ToolPolicyCompilationReason,
    ToolPolicyCompilationResult,
    ToolPolicyPolicyError,
    tool_capability_catalog_carried_sha256,
    tool_capability_catalog_ref,
    tool_capability_definition_carried_sha256,
    tool_policy_payload_sha256,
)


class ToolPolicyCompiler:
    policy_version = TOOL_POLICY_POLICY_VERSION

    def compile(
        self,
        *,
        task_draft: TaskDraftV2,
        rubric_set: RubricSetV2,
        evaluator_spec: EvaluatorSpecV2,
        catalog: ToolCapabilityCatalog,
        audit: ContractAudit,
        previous_tool_policy: ToolPolicyV2 | None = None,
    ) -> ToolPolicyCompilationResult:
        _validate_task_draft(task_draft)
        _validate_rubric_set(rubric_set)
        _validate_evaluator_spec(evaluator_spec)
        _validate_catalog(catalog)
        if previous_tool_policy is not None:
            _validate_tool_policy(previous_tool_policy)

        draft_ref = task_draft_ref(task_draft)
        rubric_ref = rubric_set_ref(rubric_set)
        evaluator_ref = evaluator_spec_ref(evaluator_spec)
        catalog_ref = tool_capability_catalog_ref(catalog)
        if rubric_set.task_draft_ref != draft_ref:
            raise ToolPolicyPolicyError("RubricSet TaskDraft ref is stale or mismatched")
        if evaluator_spec.rubric_set_ref != rubric_ref:
            raise ToolPolicyPolicyError("EvaluatorSpec RubricSet ref is stale or mismatched")

        allowed_tools = set(task_draft.allowed_tools)
        rubric_tools = {
            tool_id
            for criterion in rubric_set.criteria
            for tool_id in criterion.reachability.allowed_tool_ids
        }
        if not rubric_tools.issubset(allowed_tools):
            raise ToolPolicyPolicyError("rubric reachability tools must remain in TaskDraft allowed tools")

        definitions = _definition_map(catalog.definitions)
        missing_tools = allowed_tools - set(definitions)
        if missing_tools:
            return _compilation_result(
                task_draft_ref=draft_ref,
                rubric_set_ref=rubric_ref,
                evaluator_spec_ref=evaluator_ref,
                tool_catalog_ref=catalog_ref,
                outcome=ToolPolicyCompilationOutcome.BLOCKED_CAPABILITY,
                tool_policy=None,
                contestant_projection=None,
                unresolved_reasons=frozenset({ToolPolicyCompilationReason.MISSING_CATALOG_DEFINITION}),
                audit=audit,
            )

        blocked_reasons: set[ToolPolicyCompilationReason] = set()
        for tool_id in allowed_tools:
            definition = definitions[tool_id]
            if not definition.contestant_eligible:
                blocked_reasons.add(ToolPolicyCompilationReason.TOOL_NOT_CONTESTANT_ELIGIBLE)
            if definition.tool_family is ToolFamily.UNKNOWN:
                blocked_reasons.add(ToolPolicyCompilationReason.UNKNOWN_TOOL_FAMILY)
        if blocked_reasons:
            return _compilation_result(
                task_draft_ref=draft_ref,
                rubric_set_ref=rubric_ref,
                evaluator_spec_ref=evaluator_ref,
                tool_catalog_ref=catalog_ref,
                outcome=ToolPolicyCompilationOutcome.BLOCKED_POLICY,
                tool_policy=None,
                contestant_projection=None,
                unresolved_reasons=frozenset(blocked_reasons),
                audit=audit,
            )

        control_enforcement = _enforce_catalog_control(catalog, audit)
        control_ref = _prompt_boundary_ref(control_enforcement)
        rules = tuple(
            _compile_rule(
                definition,
                allowed=definition.tool_id in allowed_tools,
            )
            for definition in sorted(
                catalog.definitions,
                key=lambda item: item.tool_id,
            )
        )
        contestant_rules = tuple(
            _compile_contestant_rule(definitions[tool_id]) for tool_id in sorted(allowed_tools)
        )
        projection = _compile_contestant_projection(
            task_draft=task_draft,
            rules=contestant_rules,
            audit=audit,
        )
        projection_ref = contestant_tool_policy_ref(projection)
        policy = ToolPolicyV2(
            tool_policy_id="tool-policy://pending",
            tool_policy_version=(
                1 if previous_tool_policy is None else previous_tool_policy.tool_policy_version + 1
            ),
            supersedes_tool_policy_ref=(
                None if previous_tool_policy is None else tool_policy_ref(previous_tool_policy)
            ),
            task_draft_ref=draft_ref,
            rubric_set_ref=rubric_ref,
            evaluator_spec_ref=evaluator_ref,
            tool_catalog_ref=catalog_ref,
            control_boundary_enforcement_ref=control_ref,
            rules=rules,
            default_action="DENY",
            contestant_projection_ref=projection_ref,
            policy_version=self.policy_version,
            tool_policy_sha256="0" * 64,
            audit=_safe_audit(
                audit,
                (
                    draft_ref,
                    rubric_ref,
                    evaluator_ref,
                    catalog_ref,
                    control_ref,
                    projection_ref,
                ),
            ),
        )
        policy_digest = tool_policy_carried_sha256(policy)
        policy = policy.model_copy(
            update={
                "tool_policy_id": f"tool-policy://sha256/{policy_digest}",
                "tool_policy_sha256": policy_digest,
            }
        )
        return _compilation_result(
            task_draft_ref=draft_ref,
            rubric_set_ref=rubric_ref,
            evaluator_spec_ref=evaluator_ref,
            tool_catalog_ref=catalog_ref,
            outcome=ToolPolicyCompilationOutcome.COMPILED,
            tool_policy=policy,
            contestant_projection=projection,
            unresolved_reasons=frozenset(),
            audit=audit,
        )

    def validate_current(
        self,
        *,
        task_draft: TaskDraftV2,
        rubric_set: RubricSetV2,
        evaluator_spec: EvaluatorSpecV2,
        catalog: ToolCapabilityCatalog,
        tool_policy: ToolPolicyV2,
        contestant_projection: ContestantToolPolicyV2,
    ) -> None:
        try:
            _validate_tool_policy(tool_policy)
            _validate_contestant_projection(contestant_projection)
            if tool_policy.contestant_projection_ref != contestant_tool_policy_ref(contestant_projection):
                raise ToolPolicyPolicyError("ToolPolicy contestant projection ref is stale or mismatched")
            rebuilt = self.compile(
                task_draft=task_draft,
                rubric_set=rubric_set,
                evaluator_spec=evaluator_spec,
                catalog=catalog,
                audit=tool_policy.audit,
            )
            if (
                rebuilt.outcome is not ToolPolicyCompilationOutcome.COMPILED
                or rebuilt.tool_policy is None
                or rebuilt.contestant_projection is None
            ):
                raise ToolPolicyPolicyError("current authoritative inputs no longer compile a ToolPolicy")
            if not _same_contract_except_audit_actor_time(
                rebuilt.tool_policy,
                tool_policy,
            ):
                raise ToolPolicyPolicyError("current ToolPolicy does not match authoritative inputs")
            if not _same_contract_except_audit_actor_time(
                rebuilt.contestant_projection,
                contestant_projection,
            ):
                raise ToolPolicyPolicyError(
                    "current contestant projection does not match authoritative inputs"
                )
        except ToolPolicyPolicyError as exc:
            raise ToolPolicyPolicyError(f"current ToolPolicy validation failed: {exc}") from exc


def _validate_task_draft(task_draft: TaskDraftV2) -> None:
    digest = task_draft_carried_sha256(task_draft)
    if task_draft.task_draft_sha256 != digest or task_draft.task_draft_id != f"task-draft://sha256/{digest}":
        raise ToolPolicyPolicyError("TaskDraft identity is stale or mismatched")
    if task_draft.prompt_safety_status is TaskDraftPromptSafetyStatusV2.BLOCKED:
        raise ToolPolicyPolicyError("BLOCKED TaskDraft cannot compile a ToolPolicy")


def _validate_rubric_set(rubric_set: RubricSetV2) -> None:
    for criterion in rubric_set.criteria:
        reachability_digest = rubric_reachability_carried_sha256(criterion.reachability)
        if (
            criterion.reachability.reachability_sha256 != reachability_digest
            or criterion.reachability.reachability_id != f"rubric-reachability://sha256/{reachability_digest}"
        ):
            raise ToolPolicyPolicyError("rubric reachability identity is stale or mismatched")
        criterion_digest = rubric_criterion_carried_sha256(criterion)
        if (
            criterion.criterion_sha256 != criterion_digest
            or criterion.criterion_id != f"rubric-criterion://sha256/{criterion_digest}"
        ):
            raise ToolPolicyPolicyError("rubric criterion identity is stale or mismatched")
    digest = rubric_set_carried_sha256(rubric_set)
    if rubric_set.rubric_set_sha256 != digest or rubric_set.rubric_set_id != f"rubric-set://sha256/{digest}":
        raise ToolPolicyPolicyError("RubricSet identity is stale or mismatched")


def _validate_evaluator_spec(evaluator_spec: EvaluatorSpecV2) -> None:
    for binding in evaluator_spec.bindings:
        if binding.binding_sha256 != evaluator_binding_carried_sha256(binding):
            raise ToolPolicyPolicyError("EvaluatorSpec binding identity is stale or mismatched")
    for rule in evaluator_spec.failure_rules:
        if rule.rule_sha256 != evaluator_failure_rule_carried_sha256(rule):
            raise ToolPolicyPolicyError("EvaluatorSpec failure rule identity is stale or mismatched")
    digest = evaluator_spec_carried_sha256(evaluator_spec)
    if (
        evaluator_spec.evaluator_spec_sha256 != digest
        or evaluator_spec.evaluator_spec_id != f"evaluator-spec://sha256/{digest}"
    ):
        raise ToolPolicyPolicyError("EvaluatorSpec identity is stale or mismatched")


def _validate_catalog(catalog: ToolCapabilityCatalog) -> None:
    for definition in catalog.definitions:
        digest = tool_capability_definition_carried_sha256(definition)
        if (
            definition.definition_sha256 != digest
            or definition.definition_id != f"tool-capability-definition://sha256/{digest}"
        ):
            raise ToolPolicyPolicyError("tool catalog definition identity is stale or mismatched")
    digest = tool_capability_catalog_carried_sha256(catalog)
    if catalog.catalog_sha256 != digest or catalog.catalog_id != f"tool-capability-catalog://sha256/{digest}":
        raise ToolPolicyPolicyError("tool catalog identity is stale or mismatched")


def _definition_map(
    definitions: tuple[ToolCapabilityDefinition, ...],
) -> dict[str, ToolCapabilityDefinition]:
    result: dict[str, ToolCapabilityDefinition] = {}
    for definition in definitions:
        if definition.tool_id in result:
            raise ToolPolicyPolicyError("duplicate tool catalog definition")
        result[definition.tool_id] = definition
    return result


def _compile_rule(
    definition: ToolCapabilityDefinition,
    *,
    allowed: bool,
) -> ToolRuleV2:
    if allowed:
        action = ToolRuleActionV2.ALLOW
        deny_reason = None
    else:
        action = ToolRuleActionV2.DENY
        deny_reason = (
            ToolDenyReasonV2.CATALOG_POLICY_DENIED
            if (not definition.contestant_eligible or definition.tool_family is ToolFamily.UNKNOWN)
            else ToolDenyReasonV2.NOT_REQUIRED
        )
    rule = ToolRuleV2(
        rule_id=definition.tool_id,
        tool_id=definition.tool_id,
        tool_family=definition.tool_family,
        action=action,
        capability_ref=definition.capability_ref,
        enforcement_profile_ref=definition.enforcement_profile_ref,
        constraint_profile_ref=definition.constraint_profile_ref,
        deny_reason=deny_reason,
        rule_sha256="0" * 64,
    )
    return rule.model_copy(update={"rule_sha256": tool_rule_carried_sha256(rule)})


def _compile_contestant_rule(
    definition: ToolCapabilityDefinition,
) -> ContestantToolRuleV2:
    rule = ContestantToolRuleV2(
        rule_id=definition.tool_id,
        tool_id=definition.tool_id,
        tool_family=definition.tool_family,
        action="ALLOW",
        contestant_descriptor_ref=definition.contestant_descriptor_ref,
        contestant_constraint_profile_ref=(definition.contestant_constraint_profile_ref),
        rule_sha256="0" * 64,
    )
    return rule.model_copy(update={"rule_sha256": contestant_tool_rule_carried_sha256(rule)})


def _compile_contestant_projection(
    *,
    task_draft: TaskDraftV2,
    rules: tuple[ContestantToolRuleV2, ...],
    audit: ContractAudit,
) -> ContestantToolPolicyV2:
    public_refs = tuple(
        ref
        for rule in rules
        for ref in (
            rule.contestant_descriptor_ref,
            rule.contestant_constraint_profile_ref,
        )
    )
    projection = ContestantToolPolicyV2(
        contestant_tool_policy_id="contestant-tool-policy://pending",
        source_task_draft_sha256=task_draft.task_draft_sha256,
        rules=rules,
        default_action="DENY",
        policy_version=TOOL_POLICY_POLICY_VERSION,
        contestant_tool_policy_sha256="0" * 64,
        audit=_safe_audit(audit, public_refs),
    )
    digest = contestant_tool_policy_carried_sha256(projection)
    return projection.model_copy(
        update={
            "contestant_tool_policy_id": (f"contestant-tool-policy://sha256/{digest}"),
            "contestant_tool_policy_sha256": digest,
        }
    )


def _enforce_catalog_control(
    catalog: ToolCapabilityCatalog,
    audit: ContractAudit,
) -> PromptBoundaryEnforcementResult:
    catalog_ref = tool_capability_catalog_ref(catalog)
    segment = PromptBoundarySegment(
        segment_id=f"prompt-boundary-segment://tool-policy/{catalog.catalog_sha256}",
        surface=PromptBoundarySurface.TOOL_POLICY_CONFIG,
        source_role=PromptBoundarySourceRole.STATIC_POLICY_CONFIG,
        source_ref=catalog_ref,
        content_sha256=catalog.catalog_sha256,
        untrusted_data_marker=False,
        text_preview=None,
    )
    request = PromptBoundaryEnforcementRequest(
        boundary_id=f"prompt-boundary://tool-policy/{catalog.catalog_sha256}",
        segments=(segment,),
        audit=_safe_audit(audit, (catalog_ref,)),
    )
    try:
        result = PromptInjectionBoundaryEnforcer().enforce(request)
    except PromptInjectionBoundaryPolicyError as exc:
        raise ToolPolicyPolicyError("tool catalog control boundary is invalid") from exc
    if (
        result.trace_injection_execution_count != 0
        or result.accepted_segments != (segment,)
        or result.rejected_segments
    ):
        raise ToolPolicyPolicyError("tool catalog control boundary did not remain static")
    return result


def _prompt_boundary_ref(
    result: PromptBoundaryEnforcementResult,
) -> ObjectRef:
    return ObjectRef(
        object_type="prompt-boundary-enforcement",
        object_id=result.enforcement_id,
        object_version=result.policy_version,
        object_sha256=result.enforcement_sha256,
    )


def _compilation_result(
    *,
    task_draft_ref: ObjectRef,
    rubric_set_ref: ObjectRef,
    evaluator_spec_ref: ObjectRef,
    tool_catalog_ref: ObjectRef,
    outcome: ToolPolicyCompilationOutcome,
    tool_policy: ToolPolicyV2 | None,
    contestant_projection: ContestantToolPolicyV2 | None,
    unresolved_reasons: frozenset[ToolPolicyCompilationReason],
    audit: ContractAudit,
) -> ToolPolicyCompilationResult:
    policy_ref = tool_policy_ref(tool_policy) if tool_policy is not None else None
    projection_ref = (
        contestant_tool_policy_ref(contestant_projection) if contestant_projection is not None else None
    )
    payload = {
        "task_draft_ref": _ref_payload(task_draft_ref),
        "rubric_set_ref": _ref_payload(rubric_set_ref),
        "evaluator_spec_ref": _ref_payload(evaluator_spec_ref),
        "tool_catalog_ref": _ref_payload(tool_catalog_ref),
        "outcome": outcome.value,
        "tool_policy_ref": _maybe_ref_payload(policy_ref),
        "contestant_projection_ref": _maybe_ref_payload(projection_ref),
        "unresolved_reasons": sorted(item.value for item in unresolved_reasons),
        "policy_version": TOOL_POLICY_POLICY_VERSION,
    }
    digest = tool_policy_payload_sha256(payload)
    refs = tuple(
        ref
        for ref in (
            task_draft_ref,
            rubric_set_ref,
            evaluator_spec_ref,
            tool_catalog_ref,
            policy_ref,
            projection_ref,
        )
        if ref is not None
    )
    return ToolPolicyCompilationResult(
        result_id=f"tool-policy-compilation-result://sha256/{digest}",
        task_draft_ref=task_draft_ref,
        rubric_set_ref=rubric_set_ref,
        evaluator_spec_ref=evaluator_spec_ref,
        tool_catalog_ref=tool_catalog_ref,
        outcome=outcome,
        tool_policy=tool_policy,
        contestant_projection=contestant_projection,
        unresolved_reasons=unresolved_reasons,
        policy_version=TOOL_POLICY_POLICY_VERSION,
        result_sha256=digest,
        audit=_safe_audit(audit, refs),
    )


def _validate_tool_policy(policy: ToolPolicyV2) -> None:
    for rule in policy.rules:
        if rule.rule_id != rule.tool_id:
            raise ToolPolicyPolicyError("ToolPolicy rule ID must equal its tool ID")
        if rule.rule_sha256 != tool_rule_carried_sha256(rule):
            raise ToolPolicyPolicyError("ToolPolicy rule identity is stale or mismatched")
    digest = tool_policy_carried_sha256(policy)
    if policy.tool_policy_sha256 != digest or policy.tool_policy_id != f"tool-policy://sha256/{digest}":
        raise ToolPolicyPolicyError("ToolPolicy identity is stale or mismatched")


def _validate_contestant_projection(
    projection: ContestantToolPolicyV2,
) -> None:
    for rule in projection.rules:
        if rule.rule_id != rule.tool_id:
            raise ToolPolicyPolicyError("contestant rule ID must equal its tool ID")
        if rule.rule_sha256 != contestant_tool_rule_carried_sha256(rule):
            raise ToolPolicyPolicyError("contestant rule identity is stale or mismatched")
    digest = contestant_tool_policy_carried_sha256(projection)
    if (
        projection.contestant_tool_policy_sha256 != digest
        or projection.contestant_tool_policy_id != f"contestant-tool-policy://sha256/{digest}"
    ):
        raise ToolPolicyPolicyError("contestant projection identity is stale or mismatched")


def _same_contract_except_audit_actor_time(
    expected: ToolPolicyV2 | ContestantToolPolicyV2,
    observed: ToolPolicyV2 | ContestantToolPolicyV2,
) -> bool:
    if expected.model_dump(mode="json", exclude={"audit"}) != observed.model_dump(
        mode="json",
        exclude={"audit"},
    ):
        return False
    return (
        expected.audit.governing_versions == observed.audit.governing_versions
        and expected.audit.input_refs == observed.audit.input_refs
    )


def _safe_audit(
    audit: ContractAudit,
    refs: tuple[ObjectRef, ...],
) -> ContractAudit:
    return ContractAudit(
        created_at=audit.created_at,
        created_by=audit.created_by,
        governing_versions=audit.governing_versions,
        input_refs=tuple(sorted(set(refs), key=_ref_key)),
    )


def _ref_key(ref: ObjectRef) -> tuple[str, str, str, str]:
    return (
        ref.object_type,
        ref.object_id,
        ref.object_version,
        ref.object_sha256,
    )


def _ref_payload(ref: ObjectRef) -> dict[str, object]:
    return ref.model_dump(mode="json", exclude_none=False)


def _maybe_ref_payload(ref: ObjectRef | None) -> dict[str, object] | None:
    if ref is None:
        return None
    return _ref_payload(ref)
