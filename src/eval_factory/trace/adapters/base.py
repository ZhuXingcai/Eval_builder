from __future__ import annotations

from pathlib import Path
from typing import Protocol

from eval_factory.trace.models import TraceProbeResult, TraceProbeResultV2


class TraceAdapter(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    def probe(
        self,
        source: Path,
    ) -> TraceProbeResult | TraceProbeResultV2: ...
