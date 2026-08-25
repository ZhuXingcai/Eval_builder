from __future__ import annotations

import csv
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass

from eval_factory.agent_system.dataset_runtime import (
    FactoryDatasetCoreInput,
)
from eval_factory.contracts.core import ObjectRef
from eval_factory.contracts.orchestration import TraceSourceRef
from eval_factory.harness.contracts import sorted_refs
from eval_factory.harness.source_admission import (
    SOURCE_MANIFEST_COLUMNS,
)
from eval_factory.trace.source_registry import TraceSourceRegistry


class GenericAgentTraceSourceAdmissionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class GenericAgentTraceSourceAdmission:
    source_refs: tuple[ObjectRef, ...]
    sources: dict[ObjectRef, TraceSourceRef]


class GenericAgentTraceSourceAdmissionService:
    def admit(
        self,
        *,
        core_input: FactoryDatasetCoreInput,
        registry: TraceSourceRegistry,
        source_ref_overrides: Mapping[str, ObjectRef] | None = None,
    ) -> GenericAgentTraceSourceAdmission:
        manifest_bytes = core_input.manifest_path.read_bytes()
        if hashlib.sha256(manifest_bytes).hexdigest() != core_input.expected_manifest_sha256:
            raise GenericAgentTraceSourceAdmissionError(
                "trace manifest differs from approved authority",
            )
        entries = self._entries(core_input)
        expected_names = {f"{instance_id}_{sid}.jsonl" for instance_id, sid in entries}
        overrides = dict(source_ref_overrides or {})
        if overrides and set(overrides) != expected_names:
            raise GenericAgentTraceSourceAdmissionError(
                "Harness source refs differ from the trace manifest",
            )
        sources: dict[ObjectRef, TraceSourceRef] = {}
        for instance_id, sid in entries:
            relative_name = f"{instance_id}_{sid}.jsonl"
            source_path = core_input.raw_root / relative_name
            if (
                source_path.is_symlink()
                or not source_path.is_file()
                or source_path.parent != core_input.raw_root
            ):
                raise GenericAgentTraceSourceAdmissionError(
                    "trace source file is missing or unsafe",
                )
            override = overrides.get(relative_name)
            if override is not None and (
                override.object_type != "trace-source" or override.object_version != "v2"
            ):
                raise GenericAgentTraceSourceAdmissionError(
                    "Harness source ref has the wrong type or version",
                )
            source_trace_id = (
                override.object_id if override is not None else f"source-trace://{instance_id}/{sid}"
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
            if override is not None and reference != override:
                raise GenericAgentTraceSourceAdmissionError(
                    "Harness source ref differs from registered source bytes",
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
            if tuple(reader.fieldnames or ()) != SOURCE_MANIFEST_COLUMNS:
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
