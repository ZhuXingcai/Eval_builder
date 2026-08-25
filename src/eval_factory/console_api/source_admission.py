from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

from eval_factory.console_api.agent_contracts import (
    AgentShellSourceAdmissionCommandV1,
    AgentShellSourceAdmissionContractV1,
    AgentShellSourceAdmissionV1,
    AgentShellSourceFileKindV1,
    AgentShellSourceFileV1,
)
from eval_factory.contracts.core import (
    ContractAudit,
    ObjectRef,
    VersionBinding,
)
from eval_factory.harness import (
    HarnessSessionConcurrencyError,
    HarnessSessionService,
    HarnessSessionStatusV1,
)
from eval_factory.harness.source_admission import (
    SOURCE_MANIFEST_COLUMNS,
    HarnessSourceAdmissionConflictError,
    HarnessSourceAdmissionLimits,
    HarnessSourceAdmissionStore,
    HarnessSourceAdmissionWrite,
    SourceUploadClaim,
    SourceUploadPart,
)


class HarnessSourceAdmissionService:
    def __init__(
        self,
        *,
        sessions: HarnessSessionService,
        store: HarnessSourceAdmissionStore,
        trace_schema_ref: ObjectRef,
        producer_capability_ref: ObjectRef,
        governing_versions: tuple[VersionBinding, ...],
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if (
            trace_schema_ref.object_type != "json-schema"
            or trace_schema_ref.object_version != "v1"
            or producer_capability_ref.object_type != "harness-capability-definition"
            or producer_capability_ref.object_version != "v1"
        ):
            raise ValueError(
                "source admission authority refs are invalid",
            )
        self._sessions = sessions
        self.store = store
        self._trace_schema_ref = trace_schema_ref
        self._producer_capability_ref = producer_capability_ref
        self._governing_versions = governing_versions
        self._clock = clock or (lambda: datetime.now(UTC))

    @property
    def limits(self) -> HarnessSourceAdmissionLimits:
        return self.store.limits

    def contract(self) -> AgentShellSourceAdmissionContractV1:
        return AgentShellSourceAdmissionContractV1(
            max_source_files=self.limits.max_source_files,
            max_manifest_bytes=self.limits.max_manifest_bytes,
            max_source_bytes=self.limits.max_source_bytes,
            max_total_source_bytes=self.limits.max_total_source_bytes,
            max_request_bytes=self.limits.max_request_bytes,
        )

    def admit(
        self,
        session_id: str,
        command: AgentShellSourceAdmissionCommandV1,
        *,
        manifest: SourceUploadPart,
        traces: tuple[SourceUploadPart, ...],
        principal: str,
    ) -> AgentShellSourceAdmissionV1:
        projection = self._sessions.get_session(session_id)
        session = projection.session
        if session.status is HarnessSessionStatusV1.CLOSED:
            raise HarnessSourceAdmissionConflictError(
                "closed session cannot admit sources",
            )
        if session.session_version != command.expected_session_version:
            raise HarnessSessionConcurrencyError(
                "source admission uses stale session authority",
            )
        audit = ContractAudit(
            created_at=self._clock(),
            created_by=principal,
            governing_versions=self._governing_versions,
            input_refs=tuple(
                sorted(
                    {
                        session.session_ref,
                        self._trace_schema_ref,
                        self._producer_capability_ref,
                    },
                    key=lambda item: (
                        item.object_type,
                        item.object_id,
                        item.object_version,
                        item.object_sha256,
                    ),
                )
            ),
        )
        write = self.store.admit(
            session_ref=session.session_ref,
            session_id=session_id,
            session_version=session.session_version,
            manifest_claim=SourceUploadClaim(
                relative_name="manifest.csv",
                expected_sha256=command.manifest_sha256,
                expected_size_bytes=command.manifest_size_bytes,
            ),
            trace_claims=tuple(
                SourceUploadClaim(
                    relative_name=item.relative_name,
                    expected_sha256=item.expected_sha256,
                    expected_size_bytes=item.expected_size_bytes,
                )
                for item in command.trace_files
            ),
            manifest_upload=manifest,
            trace_uploads=traces,
            trace_schema_ref=self._trace_schema_ref,
            producer_capability_ref=self._producer_capability_ref,
            audit=audit,
            idempotency_key=command.idempotency_key,
        )
        return self._public(write)

    def get(self, session_id: str) -> AgentShellSourceAdmissionV1:
        self._sessions.get_session(session_id)
        return self._public(self.store.get_for_session(session_id))

    @staticmethod
    def _public(
        write: HarnessSourceAdmissionWrite,
    ) -> AgentShellSourceAdmissionV1:
        admission = write.admission
        manifest = AgentShellSourceFileV1(
            kind=AgentShellSourceFileKindV1.MANIFEST,
            relative_name="manifest.csv",
            media_type="text/csv",
            size_bytes=admission.manifest_size_bytes,
            sha256=admission.manifest_sha256,
        )
        traces = tuple(
            AgentShellSourceFileV1(
                kind=AgentShellSourceFileKindV1.TRACE,
                relative_name=member.relative_name,
                media_type="application/x-ndjson",
                size_bytes=member.size_bytes,
                sha256=member.raw_sha256,
                source_ref=member.source_ref,
                artifact_envelope_ref=member.artifact_envelope_ref,
            )
            for member in admission.members
        )
        return AgentShellSourceAdmissionV1(
            admission_ref=admission.to_ref(),
            session_id=admission.session_id,
            session_version=admission.session_version,
            manifest_sha256=admission.manifest_sha256,
            source_count=len(admission.members),
            total_source_bytes=admission.total_source_bytes,
            files=(manifest, *traces),
            artifact_envelope_refs=admission.artifact_envelope_refs,
            created_at=admission.audit.created_at,
        )


__all__ = [
    "SOURCE_MANIFEST_COLUMNS",
    "HarnessSourceAdmissionService",
]
