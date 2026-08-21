from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol

from eval_factory.agent_system.private_store import FactoryPrivateObjectStore
from eval_factory.ai_gateway.protocols import AIGateway
from eval_factory.contracts.ai_gateway_v2 import (
    GatewayInvocationRequestV2,
    GatewayInvocationResultV2,
    GatewayInvocationStatusV2,
    GatewayReceiptV2,
    ModelRouteRequestV2,
    PromptTemplateV2,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.harness.runtime_models import (
    GatewayJournalStateV1,
    HarnessGatewayJournalV1,
    HarnessMessageRoleV1,
    HarnessMessageV1,
    HarnessRequirementPolicyV1,
    HarnessTurnOutcomeV1,
    HarnessTurnResultV1,
    ProviderEvidenceClassV1,
    RequirementIntakeV1,
    RequirementInterpretationOutcomeV1,
    RequirementInterpretationProposalV1,
    RequirementInterpretationV1,
)
from eval_factory.harness.session_store import (
    HarnessSessionStore,
    HarnessTurnStart,
)


class GatewayInvocationLookup(Protocol):
    def get_invocation(
        self,
        request_ref: ObjectRef,
    ) -> (
        tuple[
            GatewayInvocationRequestV2,
            GatewayReceiptV2,
            GatewayInvocationResultV2,
        ]
        | None
    ): ...


class GatewayRequirementAgentError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class GatewayRequirementAgentConfig:
    prompt: PromptTemplateV2
    agent_definition_ref: ObjectRef
    allowed_model_profile_refs: tuple[ObjectRef, ...]
    budget_reservation_ref: ObjectRef
    data_classification: str
    residency: str
    input_token_budget: int
    output_token_budget: int
    max_cost_micro_usd: int
    minimum_quality_basis_points: int
    evidence_class: ProviderEvidenceClassV1
    assistant_chunk_characters: int = 512

    def __post_init__(self) -> None:
        if self.assistant_chunk_characters < 64 or self.assistant_chunk_characters > 4_096:
            raise ValueError("assistant chunk size must be between 64 and 4096")
        if not self.allowed_model_profile_refs:
            raise ValueError("requirement Agent requires an allowed model profile")


class GatewayRequirementAgentLoop:
    def __init__(
        self,
        *,
        gateway: AIGateway,
        lookup: GatewayInvocationLookup,
        private_store: FactoryPrivateObjectStore,
        session_store: HarnessSessionStore,
        config: GatewayRequirementAgentConfig,
    ) -> None:
        self.gateway = gateway
        self.lookup = lookup
        self.private_store = private_store
        self.session_store = session_store
        self.config = config

    async def run(
        self,
        start: HarnessTurnStart,
        *,
        audit: ContractAudit,
    ) -> HarnessTurnResultV1:
        if start.replay is not None:
            return start.replay
        current = self.session_store.get_journal_for_command(start.command.to_ref())
        if current is not None:
            return self.reconcile(start, audit=audit)
        invocation, prepared = self.prepare(start, audit=audit)

        result = await self.gateway.invoke(
            invocation,
            route=self.gateway.get_route(prepared.route_ref),
        )
        closure = self.lookup.get_invocation(invocation.to_ref())
        if closure is None or closure[2] != result:
            return self._complete_unknown(
                start,
                prepared=prepared,
                reason_code="GATEWAY_RESULT_NOT_PROVABLE",
                audit=audit,
            )
        return self._complete_from_gateway(
            start,
            prepared=prepared,
            closure=closure,
            audit=audit,
        )

    def prepare(
        self,
        start: HarnessTurnStart,
        *,
        audit: ContractAudit,
    ) -> tuple[GatewayInvocationRequestV2, HarnessGatewayJournalV1]:
        invocation, prepared = self._prepare(start, audit=audit)
        self.session_store.record_prepared_journal(
            session_id=start.state.session_id,
            command=start.command,
            journal=prepared,
        )
        return invocation, prepared

    def reconcile(
        self,
        start: HarnessTurnStart,
        *,
        audit: ContractAudit,
    ) -> HarnessTurnResultV1:
        replay = self.session_store.get_turn_for_command(start.command.to_ref())
        if replay is not None:
            return replay
        journal = self.session_store.get_journal_for_command(start.command.to_ref())
        if journal is None:
            raise GatewayRequirementAgentError("session command has no Gateway journal to reconcile")
        if journal.state is GatewayJournalStateV1.UNKNOWN_OUTCOME:
            raise GatewayRequirementAgentError("Gateway invocation outcome requires external verification")
        if journal.state is GatewayJournalStateV1.COMMITTED:
            closure = self.lookup.get_invocation(journal.invocation_request_ref)
            if closure is None:
                raise GatewayRequirementAgentError("committed session journal lacks Gateway authority")
            if journal.predecessor_journal_ref is None:
                raise GatewayRequirementAgentError("committed session journal lacks prepared predecessor")
            return self._complete_from_gateway(
                start,
                prepared=self.session_store.get_journal(journal.predecessor_journal_ref),
                closure=closure,
                audit=audit,
            )
        closure = self.lookup.get_invocation(journal.invocation_request_ref)
        if closure is None:
            return self._complete_unknown(
                start,
                prepared=journal,
                reason_code="GATEWAY_OUTCOME_UNKNOWN",
                audit=audit,
            )
        return self._complete_from_gateway(
            start,
            prepared=journal,
            closure=closure,
            audit=audit,
        )

    def _prepare(
        self,
        start: HarnessTurnStart,
        *,
        audit: ContractAudit,
    ) -> tuple[GatewayInvocationRequestV2, HarnessGatewayJournalV1]:
        intake = RequirementIntakeV1(
            session_ref=start.state.session_ref,
            command_ref=start.command.to_ref(),
            user_message_ref=start.user_message.to_ref(),
            user_text=start.user_message.content,
            artifact_envelope_refs=start.user_message.artifact_envelope_refs,
        )
        rendering_ref = self.private_store.put_model(
            object_type="prompt-rendering",
            value=intake,
        )
        turn_ref = ObjectRef(
            object_type="harness-turn",
            object_id=_turn_id(start.command.object_id),
            object_version="v1",
            object_sha256=start.command.object_sha256,
        )
        route = self.gateway.route(
            ModelRouteRequestV2.create(
                route_request_id=(
                    f"model-route-request://harness-requirement/{_suffix(start.command.object_id)}"
                ),
                agent_task_ref=turn_ref,
                agent_definition_ref=self.config.agent_definition_ref,
                task_kind="harness-requirement-interpretation",
                prompt_template_ref=self.config.prompt.to_ref(),
                required_capabilities=(self.config.prompt.required_model_capabilities),
                data_classification=self.config.data_classification,
                residency=self.config.residency,
                input_token_budget=self.config.input_token_budget,
                output_token_budget=self.config.output_token_budget,
                max_cost_micro_usd=self.config.max_cost_micro_usd,
                minimum_quality_basis_points=(self.config.minimum_quality_basis_points),
                allowed_model_profile_refs=tuple(
                    sorted(
                        self.config.allowed_model_profile_refs,
                        key=_ref_key,
                    )
                ),
                budget_reservation_ref=self.config.budget_reservation_ref,
                route_version=1,
                predecessor_route_ref=None,
                failed_receipt_ref=None,
                generator_model_profile_ref=None,
                audit=audit,
            )
        )
        invocation = GatewayInvocationRequestV2.create(
            invocation_request_id=(
                f"gateway-invocation-request://harness-requirement/{_suffix(start.command.object_id)}"
            ),
            agent_task_ref=turn_ref,
            route_decision_ref=route.to_ref(),
            prompt_template_ref=self.config.prompt.to_ref(),
            prompt_rendering_ref=rendering_ref,
            output_schema_ref=self.config.prompt.output_schema_ref,
            rag_result_refs=(),
            idempotency_key=(f"gateway-invoke-harness-requirement-{_suffix(start.command.object_id)}"),
            audit=audit,
        )
        prepared = HarnessGatewayJournalV1.create(
            journal_id=(f"harness-gateway-journal://{_suffix(start.command.object_id)}/prepared"),
            session_ref=start.state.session_ref,
            command_ref=start.command.to_ref(),
            state=GatewayJournalStateV1.PREPARED,
            evidence_class=self.config.evidence_class,
            route_ref=route.to_ref(),
            invocation_request_ref=invocation.to_ref(),
            prompt_template_ref=self.config.prompt.to_ref(),
            model_profile_ref=route.selected_model_profile_ref,
            predecessor_journal_ref=None,
            receipt_ref=None,
            invocation_result_ref=None,
            output_ref=None,
            gateway_status=None,
            usage=None,
            failure_code=None,
            audit=audit,
        )
        return invocation, prepared

    def _complete_from_gateway(
        self,
        start: HarnessTurnStart,
        *,
        prepared: HarnessGatewayJournalV1,
        closure: tuple[
            GatewayInvocationRequestV2,
            GatewayReceiptV2,
            GatewayInvocationResultV2,
        ],
        audit: ContractAudit,
    ) -> HarnessTurnResultV1:
        invocation, receipt, result = closure
        if (
            invocation.to_ref() != prepared.invocation_request_ref
            or result.receipt_ref != receipt.to_ref()
            or result.route_decision_ref != prepared.route_ref
        ):
            raise GatewayRequirementAgentError("Gateway closure differs from prepared session authority")
        committed = HarnessGatewayJournalV1.create(
            journal_id=(f"harness-gateway-journal://{_suffix(start.command.object_id)}/committed"),
            session_ref=start.state.session_ref,
            command_ref=start.command.to_ref(),
            state=GatewayJournalStateV1.COMMITTED,
            evidence_class=self.config.evidence_class,
            route_ref=prepared.route_ref,
            invocation_request_ref=prepared.invocation_request_ref,
            prompt_template_ref=prepared.prompt_template_ref,
            model_profile_ref=prepared.model_profile_ref,
            predecessor_journal_ref=prepared.to_ref(),
            receipt_ref=receipt.to_ref(),
            invocation_result_ref=result.to_ref(),
            output_ref=result.output_ref,
            gateway_status=result.status,
            usage=receipt.usage,
            failure_code=result.failure_code,
            audit=audit,
        )
        if result.status is not GatewayInvocationStatusV2.SUCCEEDED or result.output_ref is None:
            return self._complete_blocked(
                start,
                prepared=prepared,
                journal=committed,
                assistant_text="模型调用未完成, 当前需求未被解释。",
                interpretation_outcome=(RequirementInterpretationOutcomeV1.BLOCKED_CAPABILITY),
                turn_outcome=HarnessTurnOutcomeV1.BLOCKED_CAPABILITY,
                reason_code=result.failure_code or "MODEL_PROVIDER_FAILED",
                audit=audit,
            )
        proposal = self.private_store.get_model(
            result.output_ref,
            RequirementInterpretationProposalV1,
        )
        return self._complete_proposal(
            start,
            prepared=prepared,
            journal=committed,
            proposal_ref=result.output_ref,
            proposal=proposal,
            gateway_result_ref=result.to_ref(),
            audit=audit,
        )

    def _complete_proposal(
        self,
        start: HarnessTurnStart,
        *,
        prepared: HarnessGatewayJournalV1,
        journal: HarnessGatewayJournalV1,
        proposal_ref: ObjectRef,
        proposal: RequirementInterpretationProposalV1,
        gateway_result_ref: ObjectRef,
        audit: ContractAudit,
    ) -> HarnessTurnResultV1:
        messages = self._assistant_messages(
            start,
            proposal.assistant_message,
            audit=audit,
        )
        final_message = next(value for value in messages if value.final)
        if proposal.outcome == "READY":
            interpretation_outcome = RequirementInterpretationOutcomeV1.READY
            turn_outcome = HarnessTurnOutcomeV1.READY
            reasons: tuple[str, ...] = ()
        elif proposal.outcome == "CLARIFICATION_REQUIRED":
            interpretation_outcome = RequirementInterpretationOutcomeV1.CLARIFICATION_REQUIRED
            turn_outcome = HarnessTurnOutcomeV1.CLARIFICATION_REQUIRED
            reasons = ()
        else:
            interpretation_outcome = RequirementInterpretationOutcomeV1.ABSTAINED
            turn_outcome = HarnessTurnOutcomeV1.BLOCKED_POLICY
            reasons = ("SEMANTIC_ABSTAINED",)
        interpretation = RequirementInterpretationV1.create(
            interpretation_id=(f"requirement-interpretation://{_suffix(start.command.object_id)}"),
            session_ref=start.state.session_ref,
            command_ref=start.command.to_ref(),
            user_message_ref=start.user_message.to_ref(),
            assistant_message_ref=final_message.to_ref(),
            proposal_ref=proposal_ref,
            gateway_result_ref=gateway_result_ref,
            outcome=interpretation_outcome,
            evidence_class=self.config.evidence_class,
            goals=proposal.goals,
            constraints=proposal.constraints,
            assumptions=proposal.assumptions,
            source_expectations=proposal.source_expectations,
            target_capabilities=proposal.target_capabilities,
            quality_intent=proposal.quality_intent,
            delivery_intent=proposal.delivery_intent,
            missing_field_codes=proposal.missing_field_codes,
            clarification_questions=proposal.clarification_questions,
            reason_codes=reasons,
            audit=audit,
        )
        policy = None
        if proposal.outcome == "READY":
            assert proposal.max_model_requests is not None
            assert proposal.max_model_tokens is not None
            assert proposal.max_cost_micro_usd is not None
            policy = HarnessRequirementPolicyV1.create(
                policy_id=(f"harness-requirement-policy://{_suffix(interpretation.object_id)}"),
                interpretation_ref=interpretation.to_ref(),
                data_classification=self.config.data_classification,
                residency=self.config.residency,
                max_model_requests=proposal.max_model_requests,
                max_model_tokens=proposal.max_model_tokens,
                max_cost_micro_usd=proposal.max_cost_micro_usd,
                audit=audit,
            )
        return self.session_store.complete_turn(
            session_id=start.state.session_id,
            command=start.command,
            prepared_journal=prepared,
            committed_journal=journal,
            assistant_messages=messages,
            interpretation=interpretation,
            requirement_policy=policy,
            outcome=turn_outcome,
            audit=audit,
        )

    def _complete_unknown(
        self,
        start: HarnessTurnStart,
        *,
        prepared: HarnessGatewayJournalV1,
        reason_code: str,
        audit: ContractAudit,
    ) -> HarnessTurnResultV1:
        unknown = HarnessGatewayJournalV1.create(
            journal_id=(f"harness-gateway-journal://{_suffix(start.command.object_id)}/unknown"),
            session_ref=start.state.session_ref,
            command_ref=start.command.to_ref(),
            state=GatewayJournalStateV1.UNKNOWN_OUTCOME,
            evidence_class=self.config.evidence_class,
            route_ref=prepared.route_ref,
            invocation_request_ref=prepared.invocation_request_ref,
            prompt_template_ref=prepared.prompt_template_ref,
            model_profile_ref=prepared.model_profile_ref,
            predecessor_journal_ref=prepared.to_ref(),
            receipt_ref=None,
            invocation_result_ref=None,
            output_ref=None,
            gateway_status=None,
            usage=None,
            failure_code=reason_code,
            audit=audit,
        )
        return self._complete_blocked(
            start,
            prepared=prepared,
            journal=unknown,
            assistant_text=("上一次模型调用结果无法被证明, 已停止自动重试并等待核验。"),
            interpretation_outcome=(RequirementInterpretationOutcomeV1.BLOCKED_CAPABILITY),
            turn_outcome=HarnessTurnOutcomeV1.UNKNOWN_OUTCOME,
            reason_code=reason_code,
            audit=audit,
        )

    def _complete_blocked(
        self,
        start: HarnessTurnStart,
        *,
        prepared: HarnessGatewayJournalV1,
        journal: HarnessGatewayJournalV1,
        assistant_text: str,
        interpretation_outcome: RequirementInterpretationOutcomeV1,
        turn_outcome: HarnessTurnOutcomeV1,
        reason_code: str,
        audit: ContractAudit,
    ) -> HarnessTurnResultV1:
        messages = self._assistant_messages(start, assistant_text, audit=audit)
        final_message = next(value for value in messages if value.final)
        interpretation = RequirementInterpretationV1.create(
            interpretation_id=(f"requirement-interpretation://{_suffix(start.command.object_id)}"),
            session_ref=start.state.session_ref,
            command_ref=start.command.to_ref(),
            user_message_ref=start.user_message.to_ref(),
            assistant_message_ref=final_message.to_ref(),
            proposal_ref=None,
            gateway_result_ref=None,
            outcome=interpretation_outcome,
            evidence_class=self.config.evidence_class,
            goals=(),
            constraints=(),
            assumptions=(),
            source_expectations=(),
            target_capabilities=(),
            quality_intent=None,
            delivery_intent=None,
            missing_field_codes=(),
            clarification_questions=(),
            reason_codes=(reason_code,),
            audit=audit,
        )
        return self.session_store.complete_turn(
            session_id=start.state.session_id,
            command=start.command,
            prepared_journal=prepared,
            committed_journal=journal,
            assistant_messages=messages,
            interpretation=interpretation,
            requirement_policy=None,
            outcome=turn_outcome,
            audit=audit,
        )

    def _assistant_messages(
        self,
        start: HarnessTurnStart,
        text: str,
        *,
        audit: ContractAudit,
    ) -> tuple[HarnessMessageV1, ...]:
        chunks = tuple(
            text[index : index + self.config.assistant_chunk_characters]
            for index in range(0, len(text), self.config.assistant_chunk_characters)
        )
        messages = [
            HarnessMessageV1.create(
                message_id=(
                    f"harness-message://{start.state.session_id}/"
                    f"{_suffix(start.command.object_id)}/chunk/{index}"
                ),
                session_ref=start.state.session_ref,
                role=HarnessMessageRoleV1.ASSISTANT,
                content=chunk,
                artifact_envelope_refs=(),
                chunk_index=index,
                final=False,
                created_at=audit.created_at,
                audit=audit,
            )
            for index, chunk in enumerate(chunks)
        ]
        messages.append(
            HarnessMessageV1.create(
                message_id=(
                    f"harness-message://{start.state.session_id}/{_suffix(start.command.object_id)}/final"
                ),
                session_ref=start.state.session_ref,
                role=HarnessMessageRoleV1.ASSISTANT,
                content=text,
                artifact_envelope_refs=(),
                chunk_index=None,
                final=True,
                created_at=audit.created_at,
                audit=audit,
            )
        )
        return tuple(messages)


def _turn_id(command_object_id: str) -> str:
    return f"harness-turn://{_suffix(command_object_id)}"


def _suffix(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:32]


def _ref_key(value: ObjectRef) -> tuple[str, str, str, str]:
    return (
        value.object_type,
        value.object_id,
        value.object_version,
        value.object_sha256,
    )


__all__ = [
    "GatewayInvocationLookup",
    "GatewayRequirementAgentConfig",
    "GatewayRequirementAgentError",
    "GatewayRequirementAgentLoop",
]
