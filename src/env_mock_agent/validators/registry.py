from __future__ import annotations

from env_mock_agent.schemas import ValidationFinding
from env_mock_agent.validators.base import ArtifactValidator, ValidationRequest


class ValidatorRegistry:
    def __init__(self) -> None:
        self._validators: dict[str, ArtifactValidator] = {}

    def register(self, validator: ArtifactValidator) -> None:
        self._validators[validator.name] = validator

    def get(self, name: str) -> ArtifactValidator:
        try:
            return self._validators[name]
        except KeyError as exc:
            raise KeyError(f"validator not registered: {name}") from exc

    def validate(
        self,
        request: ValidationRequest,
        names: list[str] | None = None,
    ) -> list[ValidationFinding]:
        selected = names or list(self._validators)
        findings: list[ValidationFinding] = []
        for name in selected:
            findings.extend(self.get(name).validate(request))
        return findings

    @classmethod
    def default(cls) -> ValidatorRegistry:
        from env_mock_agent.validators.code import CodeProjectValidator
        from env_mock_agent.validators.common import CommonValidator
        from env_mock_agent.validators.docx import DocxValidator
        from env_mock_agent.validators.leakage import LeakageValidator
        from env_mock_agent.validators.metadata import MetadataValidator
        from env_mock_agent.validators.pdf import PdfValidator
        from env_mock_agent.validators.secrets import SecretValidator
        from env_mock_agent.validators.structured import StructuredDataValidator
        from env_mock_agent.validators.text import TextValidator
        from env_mock_agent.validators.traceability import TraceabilityValidator
        from env_mock_agent.validators.xlsx import XlsxValidator

        registry = cls()
        for validator in (
            CommonValidator(),
            TextValidator(),
            StructuredDataValidator(),
            DocxValidator(),
            XlsxValidator(),
            PdfValidator(),
            CodeProjectValidator(),
            MetadataValidator(),
            SecretValidator(),
            LeakageValidator(),
            TraceabilityValidator(),
        ):
            registry.register(validator)
        return registry
