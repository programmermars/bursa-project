"""Answer generation with page citations. Providers: ollama (local), groq (free tier), none."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import requests

from .retrieve import Hit

SYSTEM = (
    "You are an assistant for internal auditors and risk analysts reading Malaysian listed companies' "
    "annual reports. Answer ONLY from the numbered sources provided. After every factual sentence add "
    "its citation in the form [S1] or [S2][S3]. If the sources do not contain the answer, reply exactly: "
    "\"Not found in the provided reports.\" Do not use outside knowledge. Be concise; use bullet points "
    "for lists of risks or findings."
)


@dataclass
class Answer:
    text: str
    hits: list[Hit]
    provider: str
    citations: list[int] = field(default_factory=list)

    @property
    def cited_pages(self) -> list[tuple[str, int]]:
        return [(self.hits[i - 1].chunk.company, self.hits[i - 1].chunk.page)
                for i in self.citations if 0 < i <= len(self.hits)]

    @property
    def invalid_citations(self) -> list[int]:
        return [i for i in self.citations if not 0 < i <= len(self.hits)]


def format_sources(hits: list[Hit]) -> str:
    return "\n\n".join(
        f"[S{n}] ({h.chunk.company} {h.chunk.year}, page {h.chunk.page}"
        + (f", {h.chunk.section}" if h.chunk.section else "") + f")\n{h.chunk.text}"
        for n, h in enumerate(hits, start=1)
    )


def parse_citations(text: str) -> list[int]:
    seen: list[int] = []
    for m in re.finditer(r"\[S(\d+)\]", text):
        n = int(m.group(1))
        if n not in seen:
            seen.append(n)
    return seen


def _ollama(settings, messages) -> str:
    try:
        r = requests.post(
            f"{settings.ollama_url.rstrip('/')}/api/chat",
            json={"model": settings.ollama_llm_model, "messages": messages, "stream": False,
                  "options": {"temperature": settings.temperature, "num_ctx": 8192}},
            timeout=600,
        )
        r.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(
            f"Could not reach Ollama at {settings.ollama_url}. Run `ollama serve` and "
            f"`ollama pull {settings.ollama_llm_model}`, or set LLM_PROVIDER=groq / none. ({e})"
        ) from e
    return r.json()["message"]["content"]


def _groq(settings, messages) -> str:
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is empty. Get a free key at console.groq.com or set LLM_PROVIDER=ollama / none.")
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {settings.groq_api_key}"},
        json={"model": settings.groq_model, "messages": messages, "temperature": settings.temperature},
        timeout=120,
    )
    if r.status_code == 429:
        raise RuntimeError("Groq free-tier rate limit reached. Wait a minute or switch to LLM_PROVIDER=ollama.")
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def answer(settings, question: str, hits: list[Hit], provider: str | None = None) -> Answer:
    provider = (provider or settings.llm_provider).lower()
    if not hits:
        return Answer("Not found in the provided reports.", hits, provider)
    if provider == "none":
        return Answer("Retrieval-only mode (LLM_PROVIDER=none): read the source passages below.", hits, provider)
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Sources:\n\n{format_sources(hits)}\n\nQuestion: {question}"},
    ]
    if provider == "ollama":
        text = _ollama(settings, messages)
    elif provider == "groq":
        text = _groq(settings, messages)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER '{provider}' (use ollama, groq or none)")
    return Answer(text.strip(), hits, provider, parse_citations(text))
