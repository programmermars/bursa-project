# Hardware guide

| Machine | Embeddings | Answers | Settings in `.env` |
|---|---|---|---|
| Desktop with NVIDIA RTX 2060 (6 GB VRAM) | Ollama `nomic-embed-text` on GPU | Ollama `qwen2.5:7b` (Q4, about 4.7 GB VRAM) | defaults |
| Laptop without a dedicated GPU (e.g. Dell) | Ollama `nomic-embed-text` on CPU (fine, small model) | Groq free tier, or Ollama `llama3.2:3b` on CPU (slow) | `LLM_PROVIDER=groq` + `GROQ_API_KEY`, or `OLLAMA_LLM_MODEL=llama3.2:3b` |
| Laptop using the desktop over Tailscale | none locally | the desktop's Ollama | `OLLAMA_URL=http://<desktop-tailscale-ip>:11434` |
| Anything, no installs | TF-IDF/LSA | none | `EMBED_BACKEND=tfidf`, `LLM_PROVIDER=none` |

Notes:

- On the desktop, keep other GPU apps closed while running `qwen2.5:7b`; if it does not fit, use `qwen2.5:3b`.
- To serve Ollama to the laptop over Tailscale, start it on the desktop with `OLLAMA_HOST=0.0.0.0 ollama serve`.
- The index must be rebuilt (`rag ingest`) whenever you change `EMBED_BACKEND` or the embedding model.
