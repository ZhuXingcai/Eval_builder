from __future__ import annotations

import pymupdf

from env_mock_agent.schemas import FindingCategory, Severity, ValidationFinding
from env_mock_agent.validators.base import (
    ArtifactValidator,
    ValidationRequest,
    finding,
)


class PdfValidator(ArtifactValidator):
    name = "pdf"

    def validate(self, request: ValidationRequest) -> list[ValidationFinding]:
        path = request.artifact_path()
        try:
            document = pymupdf.open(path)  # type: ignore[no-untyped-call]
        except Exception as exc:
            return [
                finding(
                    self.name,
                    request,
                    Severity.P0,
                    FindingCategory.STRUCTURE,
                    f"PDF cannot be parsed: {exc}",
                    repair_action="regenerate with the PDF provider",
                )
            ]
        findings: list[ValidationFinding] = []
        try:
            minimum_pages = request.plan.content_contract.get("min_pages", 1)
            if isinstance(minimum_pages, int) and document.page_count < minimum_pages:
                findings.append(
                    finding(
                        self.name,
                        request,
                        Severity.P1,
                        FindingCategory.COVERAGE,
                        f"PDF has {document.page_count} pages; expected {minimum_pages}",
                    )
                )
            text = "\n".join(
                document[index].get_text()  # type: ignore[no-untyped-call]
                for index in range(document.page_count)
            )
            minimum_characters = request.plan.content_contract.get("min_characters", 1)
            if isinstance(minimum_characters, int) and len(text.strip()) < minimum_characters:
                findings.append(
                    finding(
                        self.name,
                        request,
                        Severity.P1,
                        FindingCategory.COVERAGE,
                        f"PDF has {len(text.strip())} extracted characters; expected {minimum_characters}",
                    )
                )
        finally:
            document.close()  # type: ignore[no-untyped-call]
        return findings
