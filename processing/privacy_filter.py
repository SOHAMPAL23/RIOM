
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class PrivacyFilterConfig:
    redact_card_numbers: bool = True
    redact_ssn: bool = True
    redact_api_keys: bool = True
    redact_emails: bool = False  # Emails often carry useful meeting context


# ---------------------------------------------------------------------------
# Compiled regex patterns
# ---------------------------------------------------------------------------

_PATTERNS: dict[str, re.Pattern] = {
    "CARD_NUMBER": re.compile(
        r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|"
        r"6(?:011|5[0-9]{2})[0-9]{12})\b"
    ),
    "SSN": re.compile(r"\b(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b"),
    "API_KEY": re.compile(
        r"\b(?:sk-|pk_|Bearer\s+)[A-Za-z0-9_\-]{20,}\b",
        re.IGNORECASE,
    ),
    "EMAIL": re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
}


class PrivacyFilter:
    def __init__(
        self,
        config: PrivacyFilterConfig | None = None,
        redact_card_numbers: bool = True,
        redact_ssn: bool = True,
        redact_api_keys: bool = True,
        redact_emails: bool = False,
        **kwargs,
    ) -> None:
        if config is not None:
            self._config = config
        else:
            self._config = PrivacyFilterConfig(
                redact_card_numbers=redact_card_numbers,
                redact_ssn=redact_ssn,
                redact_api_keys=redact_api_keys,
                redact_emails=redact_emails,
            )

    def filter(self, text: str) -> str:
        """
        Returns a copy of `text` with PII replaced by [REDACTED:<TYPE>].
        """
        if self._config.redact_card_numbers:
            text = _PATTERNS["CARD_NUMBER"].sub("[REDACTED:CARD_NUMBER]", text)
        if self._config.redact_ssn:
            text = _PATTERNS["SSN"].sub("[REDACTED:SSN]", text)
        if self._config.redact_api_keys:
            text = _PATTERNS["API_KEY"].sub("[REDACTED:API_KEY]", text)
        if self._config.redact_emails:
            text = _PATTERNS["EMAIL"].sub("[REDACTED:EMAIL]", text)
        return text
