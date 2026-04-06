import os
import time
from typing import Dict, Any, Optional, Generator
from openai import OpenAI
from src.core.llm_provider import LLMProvider

GITHUB_MODELS_BASE_URL = "https://models.inference.ai.azure.com/"


class OpenAIProvider(LLMProvider):
    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        resolved_api_key = api_key or os.getenv("GITHUB_TOKEN") or os.getenv("OPENAI_API_KEY")
        resolved_model_name = model_name or os.getenv("DEFAULT_MODEL", "gpt-4o-mini")

        if not resolved_api_key:
            raise ValueError("Missing API key. Set GITHUB_TOKEN or OPENAI_API_KEY.")

        super().__init__(resolved_model_name, resolved_api_key)

        resolved_base_url = self._resolve_base_url(base_url, resolved_api_key)
        client_kwargs = {"api_key": self.api_key}
        if resolved_base_url:
            client_kwargs["base_url"] = resolved_base_url

        self.base_url = resolved_base_url
        self.client = OpenAI(**client_kwargs)

    def _resolve_base_url(self, base_url: Optional[str], api_key: str) -> Optional[str]:
        if base_url:
            return base_url

        env_base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("GITHUB_MODELS_BASE_URL")
        if env_base_url:
            return env_base_url

        if os.getenv("GITHUB_TOKEN") or api_key.startswith("gh"):
            return GITHUB_MODELS_BASE_URL

        return None

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        start_time = time.time()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
        )

        end_time = time.time()
        latency_ms = int((end_time - start_time) * 1000)

        content = response.choices[0].message.content
        response_usage = getattr(response, "usage", None)
        usage = {
            "prompt_tokens": getattr(response_usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(response_usage, "completion_tokens", 0) or 0,
            "total_tokens": getattr(response_usage, "total_tokens", 0) or 0,
        }

        return {
            "content": content,
            "usage": usage,
            "latency_ms": latency_ms,
            "provider": "openai",
        }

    def stream(self, prompt: str, system_prompt: Optional[str] = None) -> Generator[str, None, None]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        stream = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=True
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
