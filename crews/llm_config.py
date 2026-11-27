"""LLM helpers for local Ollama (including an ngrok tunnel from Render)."""

import os
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from crewai import LLM

GROQ_MODEL = "openai/gpt-oss-120b"
DEFAULT_OLLAMA_MODEL = "llama3.1"
NGROK_SKIP_HEADER = {"ngrok-skip-browser-warning": "true"}


def ollama_base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")


def ollama_model_name() -> str:
    return os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL).strip() or DEFAULT_OLLAMA_MODEL


def ollama_reachable(base_url: str | None = None, timeout: float = 8.0) -> bool:
    """Probe Ollama. Timeout is long enough for a cold ngrok hop."""
    url = (base_url or ollama_base_url()).rstrip("/")
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            return False
        req = Request(
            f"{url}/api/tags",
            headers={**NGROK_SKIP_HEADER, "User-Agent": "al-ai-engine"},
        )
        with urlopen(req, timeout=timeout) as resp:
            return 200 <= getattr(resp, "status", 200) < 300
    except Exception:
        return False


def ollama_llm() -> LLM:
    """CrewAI client for Ollama on localhost or an ngrok HTTPS URL."""
    base = ollama_base_url()
    os.environ["OLLAMA_API_BASE"] = base
    return LLM(
        model=f"ollama/{ollama_model_name()}",
        base_url=base,
        temperature=0.1,
        timeout=300,
        extra_headers=NGROK_SKIP_HEADER,
    )


def groq_llm() -> LLM:
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "Ollama is not reachable and GROQ_API_KEY is not set. "
            "Start Ollama (and ngrok if this is Render), set OLLAMA_BASE_URL, "
            "or add GROQ_API_KEY."
        )
    return LLM(
        model=f"openai/{GROQ_MODEL}",
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.1,
    )


def affordability_llm() -> LLM:
    """Prefer Ollama (local or ngrok). Fall back to Groq only in auto mode."""
    preference = os.getenv("AFFORDABILITY_LLM", "auto").strip().lower()
    if preference in ("groq", "cloud"):
        return groq_llm()
    if preference in ("ollama", "local"):
        return ollama_llm()
    if ollama_reachable():
        return ollama_llm()
    return groq_llm()
