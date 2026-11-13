# openai_client.py — Lightweight OpenAI Chat Completions client (zero new deps, reuses httpx)
import os
import time
from types import SimpleNamespace
import httpx


class _CompletionsAPI:
    def __init__(self, owner: "OpenAIChatClient"):
        self._owner = owner

    def create(self, **kwargs):
        return self._owner._create_chat_completion(**kwargs)


class _ChatNamespace:
    def __init__(self, owner: "OpenAIChatClient"):
        self.completions = _CompletionsAPI(owner)


class OpenAIChatClient:
    """
    Minimal OpenAI Chat Completions wrapper using installed httpx.
    Handles gpt-6-luna reasoning_effort, max_completion_tokens, base64 image_url
    inputs, transient retries, fast failure on quota/auth errors, and token/latency stats.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 45.0,
        max_retries: int = 3,
    ):
        self.api_key = (api_key or os.getenv("OPENAI_API_KEY") or "").strip()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._http = httpx.Client(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=self.timeout,
        )
        self.chat = _ChatNamespace(self)
        self.reset_stats()

    def reset_stats(self):
        self.call_count = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.latencies_ms: list[float] = []

    def get_stats(self) -> dict:
        return {
            "call_count": self.call_count,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "latencies_ms": list(self.latencies_ms),
        }

    def _build_payload(self, **kwargs) -> dict:
        payload = {k: v for k, v in kwargs.items() if v is not None}
        model = str(payload.get("model", ""))
        if "max_tokens" in payload and "max_completion_tokens" not in payload:
            payload["max_completion_tokens"] = payload.pop("max_tokens")
        else:
            payload.pop("max_tokens", None)
        if "gpt-6" in model or "gpt-5" in model or model.startswith("o"):
            payload.setdefault("reasoning_effort", "low")
            if payload.get("reasoning_effort") != "none":
                payload.pop("temperature", None)
        return payload

    def _create_chat_completion(self, **kwargs):
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is empty")
        payload = self._build_payload(**kwargs)
        last_exc: Exception | None = None
        for attempt in range(max(1, self.max_retries)):
            t0 = time.perf_counter()
            try:
                resp = self._http.post("/chat/completions", json=payload)
                dt_ms = (time.perf_counter() - t0) * 1000.0
                if resp.status_code == 200:
                    data = resp.json()
                    usage = data.get("usage") or {}
                    self.call_count += 1
                    self.prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
                    self.completion_tokens += int(usage.get("completion_tokens", 0) or 0)
                    self.latencies_ms.append(round(dt_ms, 2))
                    choices = [
                        SimpleNamespace(
                            message=SimpleNamespace(content=(c.get("message") or {}).get("content") or "")
                        )
                        for c in (data.get("choices") or [])
                    ]
                    return SimpleNamespace(
                        choices=choices,
                        usage=SimpleNamespace(
                            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
                            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
                        ),
                    )
                err_txt = resp.text[:400]
                low = err_txt.lower()
                if resp.status_code in (400, 401, 403, 404) or "insufficient_quota" in low or "invalid_api_key" in low:
                    raise RuntimeError(f"OpenAI HTTP {resp.status_code}: {err_txt}")
                retry_after = resp.headers.get("retry-after")
                wait_s = float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() else 3.0 * (attempt + 1)
                last_exc = RuntimeError(f"OpenAI HTTP {resp.status_code}: {err_txt}")
                if attempt + 1 < self.max_retries:
                    time.sleep(min(20.0, wait_s))
            except Exception as exc:
                last_exc = exc
                low = str(exc).lower()
                if "openai http 40" in low or "insufficient_quota" in low or "invalid_api_key" in low:
                    raise
                if attempt + 1 < self.max_retries:
                    time.sleep(3.0 * (attempt + 1))
        raise last_exc or RuntimeError("OpenAI request failed")

