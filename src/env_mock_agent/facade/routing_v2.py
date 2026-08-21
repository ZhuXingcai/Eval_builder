from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, Literal, Protocol

from pydantic import (
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from env_mock_agent.facade.contracts import (
    FacadeModel,
    FacadeObjectRef,
    Identifier,
    ReconstructionMode,
    Sha256,
)
from env_mock_agent.schemas import RuntimeName

if TYPE_CHECKING:
    from env_mock_agent.providers.registry import ProviderRegistry
    from env_mock_agent.runtimes.registry import RuntimeRegistry
    from env_mock_agent.runtimes.router import RuntimeRouter

CapabilityToken = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/+-]{0,127}$",
    ),
]
ARTIFACT_ROUTING_POLICY_VERSION: Literal["artifact-routing/r5-05-v1"] = "artifact-routing/r5-05-v1"


class AttachmentRouteCandidateKind(StrEnum):
    PROVIDER = "PROVIDER"
    RUNTIME = "RUNTIME"


class AttachmentRouteOutcome(StrEnum):
    SELECTED_PROVIDER = "SELECTED_PROVIDER"
    SELECTED_RUNTIME = "SELECTED_RUNTIME"
    BLOCKED_CAPABILITY = "BLOCKED_CAPABILITY"


class AttachmentRouteSkipCode(StrEnum):
    PROVIDER_PAYLOAD_MISSING = "PROVIDER_PAYLOAD_MISSING"
    PROVIDER_UNSUPPORTED_ASSET_TYPE = "PROVIDER_UNSUPPORTED_ASSET_TYPE"
    PROVIDER_NOT_APPROVED = "PROVIDER_NOT_APPROVED"
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"
    PROVIDER_CAPABILITY_MISSING = "PROVIDER_CAPABILITY_MISSING"
    PROVIDER_PROBE_FAILED = "PROVIDER_PROBE_FAILED"
    RUNTIME_NOT_REGISTERED = "RUNTIME_NOT_REGISTERED"
    RUNTIME_UNAVAILABLE = "RUNTIME_UNAVAILABLE"
    RUNTIME_TOOL_MISSING = "RUNTIME_TOOL_MISSING"
    RUNTIME_RESUME_UNSUPPORTED = "RUNTIME_RESUME_UNSUPPORTED"
    RUNTIME_PROBE_FAILED = "RUNTIME_PROBE_FAILED"


class AttachmentRouteRequestV2(FacadeModel):
    schema_version: Literal["env-mock-agent/attachment-route-request/v2"] = (
        "env-mock-agent/attachment-route-request/v2"
    )
    route_request_id: Identifier
    artifact_id: Identifier
    build_contract_ref: FacadeObjectRef
    routing_policy_ref: FacadeObjectRef
    asset_type: CapabilityToken
    media_type: str = Field(
        min_length=3,
        max_length=255,
        pattern=r"^[a-z0-9.+-]+/[a-z0-9.+-]+$",
    )
    mode: ReconstructionMode
    criticality: Literal["CRITICAL", "REQUIRED", "OPTIONAL"]
    provider_payload_ref: FacadeObjectRef | None = None
    approved_provider_ids: tuple[Identifier, ...] = Field(min_length=1)
    required_provider_capability_ids: tuple[Identifier, ...]
    runtime_order: tuple[Identifier, ...] = Field(min_length=1)
    required_runtime_tools: tuple[CapabilityToken, ...]
    require_runtime_resume: bool
    idempotency_key: Identifier
    route_request_sha256: Sha256

    @field_validator("mode", mode="before")
    @classmethod
    def parse_mode(cls, value: object) -> ReconstructionMode:
        if isinstance(value, ReconstructionMode):
            return value
        if isinstance(value, str):
            return ReconstructionMode(value)
        raise TypeError("mode must be a ReconstructionMode")

    @model_validator(mode="after")
    def validate_request(self) -> AttachmentRouteRequestV2:
        _require_ref(
            self.build_contract_ref,
            "artifact-build-contract",
            "v2",
            "build_contract_ref",
        )
        _require_ref(
            self.routing_policy_ref,
            "artifact-routing-policy",
            "v2",
            "routing_policy_ref",
        )
        if self.provider_payload_ref is not None:
            _require_ref(
                self.provider_payload_ref,
                "attachment-provider-payload",
                "v2",
                "provider_payload_ref",
            )
        if self.mode is ReconstructionMode.BLOCKED:
            raise ValueError("route request cannot use BLOCKED mode")
        for label, values in (
            ("approved provider IDs", self.approved_provider_ids),
            (
                "required provider capability IDs",
                self.required_provider_capability_ids,
            ),
            ("required runtime tools", self.required_runtime_tools),
        ):
            _require_sorted_unique(label, values)
        if len(self.runtime_order) != len(set(self.runtime_order)):
            raise ValueError("runtime order must be unique")
        allowed_runtimes = {
            RuntimeName.CLAUDE_AGENT_SDK.value,
            RuntimeName.CLAUDE_CODE_CLI.value,
            RuntimeName.PI_RPC.value,
        }
        if any(item not in allowed_runtimes for item in self.runtime_order):
            raise ValueError("runtime order contains an unapproved runtime")
        return self


class AttachmentRouteSkipV2(FacadeModel):
    schema_version: Literal["env-mock-agent/attachment-route-skip/v2"] = (
        "env-mock-agent/attachment-route-skip/v2"
    )
    candidate_kind: AttachmentRouteCandidateKind
    candidate_id: Identifier
    code: AttachmentRouteSkipCode
    missing_capability_ids: tuple[CapabilityToken, ...] = ()

    @field_validator("candidate_kind", mode="before")
    @classmethod
    def parse_candidate_kind(
        cls,
        value: object,
    ) -> AttachmentRouteCandidateKind:
        if isinstance(value, AttachmentRouteCandidateKind):
            return value
        if isinstance(value, str):
            return AttachmentRouteCandidateKind(value)
        raise TypeError("candidate_kind must be an AttachmentRouteCandidateKind")

    @field_validator("code", mode="before")
    @classmethod
    def parse_code(cls, value: object) -> AttachmentRouteSkipCode:
        if isinstance(value, AttachmentRouteSkipCode):
            return value
        if isinstance(value, str):
            return AttachmentRouteSkipCode(value)
        raise TypeError("code must be an AttachmentRouteSkipCode")

    @model_validator(mode="after")
    def validate_skip(self) -> AttachmentRouteSkipV2:
        _require_sorted_unique(
            "missing capability IDs",
            self.missing_capability_ids,
        )
        provider_code = self.code.value.startswith("PROVIDER_")
        if provider_code != (self.candidate_kind is AttachmentRouteCandidateKind.PROVIDER):
            raise ValueError("route skip code must match candidate kind")
        return self


class AttachmentRouteDecisionV2(FacadeModel):
    schema_version: Literal["env-mock-agent/attachment-route-decision/v2"] = (
        "env-mock-agent/attachment-route-decision/v2"
    )
    route_decision_id: Identifier
    route_request_ref: FacadeObjectRef
    outcome: AttachmentRouteOutcome
    selected_kind: AttachmentRouteCandidateKind | None = None
    selected_id: Identifier | None = None
    selected_version: str | None = Field(
        default=None,
        min_length=1,
        max_length=256,
    )
    satisfied_capability_ids: tuple[CapabilityToken, ...]
    skipped: tuple[AttachmentRouteSkipV2, ...]
    capability_snapshot_sha256: Sha256
    probed_at: datetime
    policy_version: Literal["artifact-routing/r5-05-v1"] = ARTIFACT_ROUTING_POLICY_VERSION
    route_decision_sha256: Sha256

    @field_validator("outcome", mode="before")
    @classmethod
    def parse_outcome(cls, value: object) -> AttachmentRouteOutcome:
        if isinstance(value, AttachmentRouteOutcome):
            return value
        if isinstance(value, str):
            return AttachmentRouteOutcome(value)
        raise TypeError("outcome must be an AttachmentRouteOutcome")

    @field_validator("selected_kind", mode="before")
    @classmethod
    def parse_selected_kind(
        cls,
        value: object,
    ) -> AttachmentRouteCandidateKind | None:
        if value is None or isinstance(value, AttachmentRouteCandidateKind):
            return value
        if isinstance(value, str):
            return AttachmentRouteCandidateKind(value)
        raise TypeError("selected_kind must be an AttachmentRouteCandidateKind")

    @model_validator(mode="after")
    def validate_decision(self) -> AttachmentRouteDecisionV2:
        _require_ref(
            self.route_request_ref,
            "attachment-route-request",
            "v2",
            "route_request_ref",
        )
        _require_sorted_unique(
            "satisfied capability IDs",
            self.satisfied_capability_ids,
        )
        skipped_candidates = tuple((item.candidate_kind, item.candidate_id) for item in self.skipped)
        if len(skipped_candidates) != len(set(skipped_candidates)):
            raise ValueError("skipped route candidates must be unique")
        runtime_skip_seen = False
        for item in self.skipped:
            if item.candidate_kind is AttachmentRouteCandidateKind.RUNTIME:
                runtime_skip_seen = True
            elif runtime_skip_seen:
                raise ValueError("provider skips must precede runtime skips")
        selected = self.outcome in {
            AttachmentRouteOutcome.SELECTED_PROVIDER,
            AttachmentRouteOutcome.SELECTED_RUNTIME,
        }
        if selected:
            if self.selected_kind is None or self.selected_id is None:
                raise ValueError("selected route requires candidate kind and ID")
            expected_kind = (
                AttachmentRouteCandidateKind.PROVIDER
                if self.outcome is AttachmentRouteOutcome.SELECTED_PROVIDER
                else AttachmentRouteCandidateKind.RUNTIME
            )
            if self.selected_kind is not expected_kind:
                raise ValueError("selected route kind does not match route outcome")
            if (self.selected_kind, self.selected_id) in skipped_candidates:
                raise ValueError("selected route candidate cannot also be skipped")
        elif (
            self.selected_kind is not None
            or self.selected_id is not None
            or self.selected_version is not None
            or self.satisfied_capability_ids
            or not self.skipped
        ):
            raise ValueError("blocked capability decision requires skips and no selection")
        if self.probed_at.tzinfo is None:
            raise ValueError("route probe time must be timezone-aware")
        return self


class AttachmentRoutingFacade(Protocol):
    async def route(
        self,
        request: AttachmentRouteRequestV2,
    ) -> AttachmentRouteDecisionV2: ...


class RegistryAttachmentRoutingFacade:
    def __init__(
        self,
        providers: ProviderRegistry,
        runtimes: RuntimeRegistry,
        *,
        clock: Callable[[], datetime],
    ) -> None:
        from env_mock_agent.runtimes.router import RuntimeRouter

        self._providers = providers
        self._runtime_router: RuntimeRouter = RuntimeRouter(runtimes)
        self._clock = clock

    async def route(
        self,
        request: AttachmentRouteRequestV2,
    ) -> AttachmentRouteDecisionV2:
        _validate_request_identity(request)
        skipped: list[AttachmentRouteSkipV2] = []
        snapshots: list[dict[str, object]] = []
        provider_candidate = request.approved_provider_ids[0]
        if request.provider_payload_ref is None:
            skipped.append(
                _skip(
                    AttachmentRouteCandidateKind.PROVIDER,
                    provider_candidate,
                    AttachmentRouteSkipCode.PROVIDER_PAYLOAD_MISSING,
                )
            )
            snapshots.append(
                {
                    "candidate_kind": "PROVIDER",
                    "candidate_id": provider_candidate,
                    "state": "PAYLOAD_MISSING",
                }
            )
        else:
            provider = self._providers.for_asset_type(request.asset_type)
            if provider is None:
                skipped.append(
                    _skip(
                        AttachmentRouteCandidateKind.PROVIDER,
                        provider_candidate,
                        (AttachmentRouteSkipCode.PROVIDER_UNSUPPORTED_ASSET_TYPE),
                    )
                )
                snapshots.append(
                    {
                        "candidate_kind": "PROVIDER",
                        "candidate_id": provider_candidate,
                        "state": "UNSUPPORTED_ASSET_TYPE",
                    }
                )
            elif provider.name not in request.approved_provider_ids:
                skipped.append(
                    _skip(
                        AttachmentRouteCandidateKind.PROVIDER,
                        provider.name,
                        AttachmentRouteSkipCode.PROVIDER_NOT_APPROVED,
                    )
                )
                snapshots.append(
                    {
                        "candidate_kind": "PROVIDER",
                        "candidate_id": provider.name,
                        "state": "NOT_APPROVED",
                    }
                )
            else:
                try:
                    capability = provider.probe()
                except Exception:
                    skipped.append(
                        _skip(
                            AttachmentRouteCandidateKind.PROVIDER,
                            provider.name,
                            AttachmentRouteSkipCode.PROVIDER_PROBE_FAILED,
                        )
                    )
                    snapshots.append(
                        {
                            "candidate_kind": "PROVIDER",
                            "candidate_id": provider.name,
                            "state": "PROBE_FAILED",
                        }
                    )
                else:
                    capability_ids = tuple(
                        sorted({_provider_capability_id(asset_type) for asset_type in capability.asset_types})
                    )
                    snapshots.append(
                        {
                            "candidate_kind": "PROVIDER",
                            "candidate_id": provider.name,
                            "available": capability.available,
                            "asset_types": sorted(item.casefold() for item in capability.asset_types),
                            "capability_ids": list(capability_ids),
                            "implementation": (
                                f"{provider.__class__.__module__}.{provider.__class__.__qualname__}"
                            ),
                        }
                    )
                    missing = tuple(
                        sorted(set(request.required_provider_capability_ids) - set(capability_ids))
                    )
                    if not capability.available:
                        skipped.append(
                            _skip(
                                AttachmentRouteCandidateKind.PROVIDER,
                                provider.name,
                                (AttachmentRouteSkipCode.PROVIDER_UNAVAILABLE),
                            )
                        )
                    elif missing:
                        skipped.append(
                            _skip(
                                AttachmentRouteCandidateKind.PROVIDER,
                                provider.name,
                                (AttachmentRouteSkipCode.PROVIDER_CAPABILITY_MISSING),
                                missing,
                            )
                        )
                    else:
                        return _decision(
                            request=request,
                            outcome=(AttachmentRouteOutcome.SELECTED_PROVIDER),
                            selected_kind=(AttachmentRouteCandidateKind.PROVIDER),
                            selected_id=provider.name,
                            selected_version=(
                                f"{provider.__class__.__module__}.{provider.__class__.__qualname__}"
                            ),
                            satisfied_capability_ids=(request.required_provider_capability_ids),
                            skipped=tuple(skipped),
                            snapshots=snapshots,
                            probed_at=self._clock(),
                        )

        from env_mock_agent.runtimes.router import RuntimeRoutingRequest

        runtime_names = [RuntimeName(item) for item in request.runtime_order]
        runtime, routing = await self._runtime_router.select(
            preferred=runtime_names,
            routing_request=RuntimeRoutingRequest(
                artifact_type=request.asset_type,
                role="attachment-writer",
                required_tools=list(request.required_runtime_tools),
                preferred=runtime_names,
                require_resume=request.require_runtime_resume,
            ),
        )
        for name in runtime_names:
            reason = routing.skipped.get(name.value)
            if reason is None:
                break
            code, missing = _runtime_skip(reason)
            skipped.append(
                _skip(
                    AttachmentRouteCandidateKind.RUNTIME,
                    name.value,
                    code,
                    missing,
                )
            )
            snapshots.append(
                {
                    "candidate_kind": "RUNTIME",
                    "candidate_id": name.value,
                    "code": code.value,
                    "missing_capability_ids": list(missing),
                }
            )
        if runtime is not None and routing.selected is not None:
            snapshots.append(
                {
                    "candidate_kind": "RUNTIME",
                    "candidate_id": routing.selected.value,
                    "selected": True,
                    "version": routing.runtime_version,
                    "tools": sorted(item.casefold() for item in routing.tools),
                    "resume_required": request.require_runtime_resume,
                }
            )
            return _decision(
                request=request,
                outcome=AttachmentRouteOutcome.SELECTED_RUNTIME,
                selected_kind=AttachmentRouteCandidateKind.RUNTIME,
                selected_id=routing.selected.value,
                selected_version=routing.runtime_version,
                satisfied_capability_ids=tuple(sorted(request.required_runtime_tools)),
                skipped=tuple(skipped),
                snapshots=snapshots,
                probed_at=self._clock(),
            )
        return _decision(
            request=request,
            outcome=AttachmentRouteOutcome.BLOCKED_CAPABILITY,
            selected_kind=None,
            selected_id=None,
            selected_version=None,
            satisfied_capability_ids=(),
            skipped=tuple(skipped),
            snapshots=snapshots,
            probed_at=self._clock(),
        )


def attachment_route_request_carried_sha256(
    request: AttachmentRouteRequestV2,
) -> str:
    return _payload_sha256(
        request.model_dump(
            mode="json",
            exclude={"route_request_id", "route_request_sha256"},
            exclude_none=False,
        )
    )


def attachment_route_request_ref(
    request: AttachmentRouteRequestV2,
) -> FacadeObjectRef:
    return FacadeObjectRef(
        object_type="attachment-route-request",
        object_id=request.route_request_id,
        object_version="v2",
        object_sha256=request.route_request_sha256,
    )


def attachment_route_decision_carried_sha256(
    decision: AttachmentRouteDecisionV2,
) -> str:
    return _payload_sha256(
        decision.model_dump(
            mode="json",
            exclude={
                "route_decision_id",
                "route_decision_sha256",
            },
            exclude_none=False,
        )
    )


def attachment_route_decision_ref(
    decision: AttachmentRouteDecisionV2,
) -> FacadeObjectRef:
    return FacadeObjectRef(
        object_type="attachment-route-decision",
        object_id=decision.route_decision_id,
        object_version="v2",
        object_sha256=decision.route_decision_sha256,
    )


def _decision(
    *,
    request: AttachmentRouteRequestV2,
    outcome: AttachmentRouteOutcome,
    selected_kind: AttachmentRouteCandidateKind | None,
    selected_id: str | None,
    selected_version: str | None,
    satisfied_capability_ids: tuple[str, ...],
    skipped: tuple[AttachmentRouteSkipV2, ...],
    snapshots: list[dict[str, object]],
    probed_at: datetime,
) -> AttachmentRouteDecisionV2:
    decision = AttachmentRouteDecisionV2(
        route_decision_id="attachment-route-decision://pending",
        route_request_ref=attachment_route_request_ref(request),
        outcome=outcome,
        selected_kind=selected_kind,
        selected_id=selected_id,
        selected_version=selected_version,
        satisfied_capability_ids=satisfied_capability_ids,
        skipped=skipped,
        capability_snapshot_sha256=_payload_sha256(snapshots),
        probed_at=probed_at,
        policy_version=ARTIFACT_ROUTING_POLICY_VERSION,
        route_decision_sha256="0" * 64,
    )
    digest = attachment_route_decision_carried_sha256(decision)
    return decision.model_copy(
        update={
            "route_decision_id": (f"attachment-route-decision://sha256/{digest}"),
            "route_decision_sha256": digest,
        }
    )


def _skip(
    candidate_kind: AttachmentRouteCandidateKind,
    candidate_id: str,
    code: AttachmentRouteSkipCode,
    missing_capability_ids: tuple[str, ...] = (),
) -> AttachmentRouteSkipV2:
    return AttachmentRouteSkipV2(
        candidate_kind=candidate_kind,
        candidate_id=candidate_id,
        code=code,
        missing_capability_ids=missing_capability_ids,
    )


def _runtime_skip(
    reason: str,
) -> tuple[AttachmentRouteSkipCode, tuple[str, ...]]:
    if reason == "runtime not registered":
        return AttachmentRouteSkipCode.RUNTIME_NOT_REGISTERED, ()
    if reason == "runtime probe failed":
        return AttachmentRouteSkipCode.RUNTIME_PROBE_FAILED, ()
    if reason.startswith("missing required tools: "):
        missing = tuple(
            sorted(
                item.strip()
                for item in reason.removeprefix("missing required tools: ").split(",")
                if item.strip()
            )
        )
        return AttachmentRouteSkipCode.RUNTIME_TOOL_MISSING, missing
    if reason == "runtime does not support cross-process session resume":
        return (
            AttachmentRouteSkipCode.RUNTIME_RESUME_UNSUPPORTED,
            (),
        )
    return AttachmentRouteSkipCode.RUNTIME_UNAVAILABLE, ()


def _provider_capability_id(asset_type: str) -> str:
    normalized = asset_type.casefold().removeprefix("file_").lstrip(".")
    return f"attachment-provider/generate/{normalized}/v1"


def _validate_request_identity(
    request: AttachmentRouteRequestV2,
) -> None:
    digest = attachment_route_request_carried_sha256(request)
    if (
        request.route_request_sha256 != digest
        or request.route_request_id != f"attachment-route-request://sha256/{digest}"
    ):
        raise ValueError("attachment route request identity is stale")


def _require_ref(
    ref: FacadeObjectRef,
    expected_type: str,
    expected_version: str,
    field_name: str,
) -> None:
    if ref.object_type != expected_type or ref.object_version != expected_version:
        raise ValueError(f"{field_name} must reference {expected_type} {expected_version}")


def _require_sorted_unique(
    field_name: str,
    values: tuple[str, ...],
) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must be unique")
    if values != tuple(sorted(values)):
        raise ValueError(f"{field_name} must be sorted")


def _payload_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            default=_json_default,
        ).encode()
    ).hexdigest()


def _json_default(value: object) -> str:
    if isinstance(value, (datetime, StrEnum)):
        return value.isoformat() if isinstance(value, datetime) else value.value
    raise TypeError(f"unsupported canonical value: {type(value).__name__}")
