"""
LLM Client interface supporting local Ollama and OpenAI-compatible endpoints.

Key features:
- Temperature 0 for deterministic extraction.
- Configurable timeouts and retries with exponential backoff.
- Mock client for deterministic unit testing without network or GPU.
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """Abstract interface for LLM extraction models."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate response text for the given prompt."""
        pass


class OllamaClient(LLMClient):
    """Client for local Ollama instances."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2",
        temperature: float = 0.0,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if system_prompt:
            payload["system"] = system_prompt

        backoff = 1.0
        last_err = None

        for attempt in range(self.max_retries):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(url, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    return str(data.get("response", "")).strip()
            except Exception as e:
                last_err = e
                logger.warning(f"Ollama request failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                time.sleep(backoff)
                backoff *= 2.0

        raise RuntimeError(f"Ollama generation failed after {self.max_retries} attempts: {last_err}")


class OpenAICompatibleClient(LLMClient):
    """Client for any OpenAI-compatible API endpoint."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        url = f"{self.base_url}/chat/completions"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
        }

        backoff = 1.0
        last_err = None

        for attempt in range(self.max_retries):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(url, headers=headers, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    return data["choices"][0]["message"]["content"].strip()
            except Exception as e:
                last_err = e
                logger.warning(f"OpenAI request failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                time.sleep(backoff)
                backoff *= 2.0

        raise RuntimeError(f"OpenAI generation failed after {self.max_retries} attempts: {last_err}")


class MockDeterministicLLMClient(LLMClient):
    """
    Mock LLM client for deterministic testing of schema validation,
    retry loops, and caching.
    """

    def __init__(self, canned_responses: Optional[List[str]] = None) -> None:
        self.canned_responses = canned_responses or []
        self.call_count = 0
        self.prompts_received: List[str] = []

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        self.prompts_received.append(prompt)
        current_idx = self.call_count
        self.call_count += 1

        if current_idx < len(self.canned_responses):
            return self.canned_responses[current_idx]

        # Default valid response
        return json.dumps({
            "ticker": "TEST.NS",
            "sentiment": 0.5,
            "event_type": "earnings",
            "surprise": "positive",
            "confidence": 0.85,
            "evidence_ids": ["art_001"],
            "summary": "Solid quarterly profit expansion exceeding estimates.",
        })

