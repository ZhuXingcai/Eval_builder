from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass

from eval_factory.agent_system.dataset_runtime import (
    FactoryDatasetCoreInput,
)
from eval_factory.contracts.core import ObjectRef
from eval_factory.contracts.orchestration import TraceSourceRef
from eval_factory.harness.contracts import sorted_refs
from eval_factory.trace.source_registry import TraceSourceRegistry


class GenericAgentTraceSourceAdmissionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class GenericAgentTraceSourceAdmission:
    source_refs: tuple[ObjectRef, ...]
    sources: dict[ObjectRef, TraceSourceRef]


class GenericAgentTraceSourceAdmissionService:
    _COLUMNS = (
        "instance_id",
        "sid",
        "p_date",
        "business",
        "category",
        "pool_id",
        "pool_category",
    )

    def admit(
        self,
        *,
        core_input: FactoryDatasetCoreInput,
        registry: TraceSourceRegistry,
    ) -> GenericAgentTraceSourceAdmission:
        manifest_bytes = core_input.manifest_path.read_bytes()
        if hashlib.sha256(manifest_bytes).hexdigest() != core_input.expected_manifest_sha256:
            raise GenericAgentTraceSourceAdmissionError(
                "trace manifest differs from approved authority",
            )
        entries = self._entries(core_input)
        sources: dict[ObjectRef, TraceSourceRef] = {}
        for instance_id, sid in entries:
            source_trace_id = f"source-trace://{instance_id}/{sid}"
            source_path = core_input.raw_root / f"{instance_id}_{sid}.jsonl"
            if (
                source_path.is_symlink()
                or not source_path.is_file()
                or source_path.parent != core_input.raw_root
            ):
                raise GenericAgentTraceSourceAdmissionError(
                    "trace source file is missing or unsafe",
                )
            registered = registry.register(
                source_path,
                source_trace_id=source_trace_id,
                source_uri=source_path.as_uri(),
            )
            source = registered.source
            reference = ObjectRef(
                object_type="trace-source",
                object_id=source.source_trace_id,
                object_version="v2",
                object_sha256=source.raw_sha256,
            )
            if reference in sources and sources[reference] != source:
                raise GenericAgentTraceSourceAdmissionError(
                    "trace source authority conflicts",
                )
            sources[reference] = source
        source_refs = sorted_refs(sources)
        if len(source_refs) != len(entries):
            raise GenericAgentTraceSourceAdmissionError(
                "trace source manifest is not a unique partition",
            )
        return GenericAgentTraceSourceAdmission(
            source_refs=source_refs,
            sources=sources,
        )

    def _entries(
        self,
        core_input: FactoryDatasetCoreInput,
    ) -> tuple[tuple[str, str], ...]:
        with core_input.manifest_path.open(
            encoding="utf-8",
            newline="",
        ) as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != self._COLUMNS:
                raise GenericAgentTraceSourceAdmissionError(
                    "trace manifest columns are not canonical",
                )
            entries = tuple(
                (
                    str(row["instance_id"]),
                    str(row["sid"]),
                )
                for row in reader
            )
        if not entries or len(set(entries)) != len(entries):
            raise GenericAgentTraceSourceAdmissionError(
                "trace manifest must contain unique members",
            )
        return entries


__all__ = [
    "GenericAgentTraceSourceAdmission",
    "GenericAgentTraceSourceAdmissionError",
    "GenericAgentTraceSourceAdmissionService",
]
