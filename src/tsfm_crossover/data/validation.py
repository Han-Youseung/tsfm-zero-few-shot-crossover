"""Structured validation diagnostics; never silently repairs input."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    severity: Literal["warning", "error"]
    message: str
    column: str | None = None
    row: int | None = None


@dataclass
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def add(
        self, code: str, severity: Literal["warning", "error"], message: str, **where: object
    ) -> None:
        self.issues.append(ValidationIssue(code, severity, message, **where))

    def as_dict(self) -> dict[str, object]:
        return {"valid": self.valid, "issues": [asdict(issue) for issue in self.issues]}
