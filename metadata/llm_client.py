from __future__ import annotations

import time
import logging

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Calls an OpenAI-compatible chat completion endpoint.

    Args:
        api_key:      API key for the provider.
        model:        Model identifier (e.g. "gpt-4o-mini").
        base_url:     Override for non-OpenAI providers.
        max_retries:  Number of retry attempts on transient errors.
        timeout:      Request timeout in seconds.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
        max_retries: int = 3,
        timeout: float = 30.0,
    ) -> None:
        self._model = model
        self._max_retries = max_retries
        self._timeout = timeout

        try:
            from openai import OpenAI  # type: ignore
            kwargs: dict = {"api_key": api_key, "timeout": timeout}
            if base_url:
                kwargs["base_url"] = base_url
            self._client = OpenAI(**kwargs)
        except ImportError as e:
            raise RuntimeError(
                "openai package is required. Install with: pip install openai"
            ) from e

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """
        Send a chat completion request and return the response text.

        Args:
            system_prompt: Instructions for the LLM role.
            user_prompt:   The OCR text or user-facing content.

        Returns:
            Raw response string (expected to be JSON).

        Raises:
            RuntimeError: If all retries are exhausted.
        """
        for attempt in range(1, self._max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1,  # Low temperature for deterministic extraction
                )
                content = response.choices[0].message.content
                return content or ""
            except Exception as exc:
                logger.warning("LLM call failed (attempt %d/%d): %s", attempt, self._max_retries, exc)
                if attempt < self._max_retries:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise RuntimeError(f"LLM call failed after {self._max_retries} attempts") from exc

        return ""  # Unreachable but satisfies type-checker
