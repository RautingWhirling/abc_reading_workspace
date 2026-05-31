from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


class LLMClientError(RuntimeError):
    pass


class OpenAICompatibleLLMClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        provider: str = "openai_compatible",
        timeout: int = 60,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.provider = provider
        self.timeout = timeout

    @classmethod
    def from_env_files(cls, workspace_root: str | Path) -> "OpenAICompatibleLLMClient | None":
        env_values: dict[str, str] = {}
        root = Path(workspace_root)
        for path in (
            root / ".env",
            root / "src" / "influence_strategy" / ".env",
        ):
            env_values.update(_read_env_file(path))

        provider = (
            env_values.get("LLM_PROVIDER")
            or os.environ.get("LLM_PROVIDER")
            or ""
        ).strip()

        api_key = _first_non_empty(
            env_values.get("LLM_API_KEY"),
            os.environ.get("LLM_API_KEY"),
            env_values.get("DEEPSEEK_API_KEY"),
            os.environ.get("DEEPSEEK_API_KEY"),
            env_values.get("OPENAI_API_KEY"),
            os.environ.get("OPENAI_API_KEY"),
            env_values.get("DASHSCOPE_API_KEY"),
            os.environ.get("DASHSCOPE_API_KEY"),
        )
        base_url = _first_non_empty(
            env_values.get("LLM_BASE_URL"),
            os.environ.get("LLM_BASE_URL"),
            env_values.get("DEEPSEEK_BASE_URL"),
            os.environ.get("DEEPSEEK_BASE_URL"),
            env_values.get("OPENAI_BASE_URL"),
            os.environ.get("OPENAI_BASE_URL"),
            env_values.get("OPENAI_API_BASE"),
            os.environ.get("OPENAI_API_BASE"),
            env_values.get("DASHSCOPE_BASE_URL"),
            os.environ.get("DASHSCOPE_BASE_URL"),
            "https://api.openai.com/v1",
        )
        model = _first_non_empty(
            env_values.get("LLM_MODEL"),
            os.environ.get("LLM_MODEL"),
            env_values.get("DEEPSEEK_MODEL"),
            os.environ.get("DEEPSEEK_MODEL"),
            env_values.get("OPENAI_MODEL"),
            os.environ.get("OPENAI_MODEL"),
            env_values.get("MODEL_NAME"),
            os.environ.get("MODEL_NAME"),
            "gpt-4o-mini",
        )

        if not api_key:
            return None

        resolved_provider = provider or _infer_provider(base_url=base_url, model=model)
        return cls(
            api_key=api_key,
            base_url=base_url,
            model=model,
            provider=resolved_provider,
        )

    def describe(self) -> dict[str, str]:
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
        }

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.4,
            "response_format": {"type": "json_object"},
        }
        try:
            return self._post_chat_completions(payload)
        except LLMClientError:
            payload.pop("response_format", None)
            return self._post_chat_completions(payload)

    def _post_chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            raise LLMClientError(f"LLM request failed with HTTP {exc.code}: {body[:300]}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise LLMClientError(f"LLM request failed: {exc}") from exc

        try:
            content = response_payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMClientError("LLM response missing choices[0].message.content") from exc

        return _parse_json_content(str(content))


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    result: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            result[key] = value
    return result


def _parse_json_content(content: str) -> dict[str, Any]:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start < 0 or end <= start:
            raise LLMClientError("LLM response is not valid JSON.")
        parsed = json.loads(content[start : end + 1])
    if not isinstance(parsed, dict):
        raise LLMClientError("LLM JSON response must be an object.")
    return parsed


def _first_non_empty(*values: str | None) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _infer_provider(*, base_url: str, model: str) -> str:
    combined = f"{base_url} {model}".lower()
    if "deepseek" in combined:
        return "deepseek"
    if "dashscope" in combined or "qwen" in combined:
        return "dashscope"
    if "openai" in combined or "gpt" in combined:
        return "openai"
    return "openai_compatible"
