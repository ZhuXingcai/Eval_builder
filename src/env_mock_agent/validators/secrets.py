from __future__ import annotations

import re

from env_mock_agent.schemas import FindingCategory, Severity, ValidationFinding
from env_mock_agent.validators.base import (
    ArtifactValidator,
    ValidationRequest,
    finding,
)
from env_mock_agent.validators.extract import extract_text

SECRET_PATTERNS = {
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "bearer_token": re.compile(r"(?i)\bbearer\s+[a-z0-9._~+/=-]{20,}"),
    "assigned_secret": re.compile(
        r"(?i)\b(?:api[_-]?key|secret|access[_-]?token)\b\s*[:=]\s*[\"']?[a-z0-9._~+/=-]{16,}"
    ),
}


class SecretValidator(ArtifactValidator):
    name = "secrets"

    def validate(self, request: ValidationRequest) -> list[ValidationFinding]:
        text = extract_text(request.artifact_path())
        findings: list[ValidationFinding] = []
        for name, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                findings.append(
                    finding(
                        self.name,
                        request,
                        Severity.P0,
                        FindingCategory.SECURITY,
                        f"possible real credential detected: {name}",
                        repair_action="remove the credential and replace it with an explicit placeholder",
                    )
                )
        return findings
