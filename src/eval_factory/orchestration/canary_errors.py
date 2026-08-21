from __future__ import annotations

from eval_factory.contracts.resource_v2 import (
    ResourceAdmissionOutcomeV2,
)


class CanaryDriverError(RuntimeError):
    pass


class CanaryResourceAdmissionError(CanaryDriverError):
    def __init__(
        self,
        outcome: ResourceAdmissionOutcomeV2,
    ) -> None:
        self.outcome = outcome
        super().__init__(f"canary resource admission ended as {outcome.value}")


class CanaryDriverInjectedCrash(CanaryDriverError):
    pass


class CanaryProfileError(RuntimeError):
    pass
