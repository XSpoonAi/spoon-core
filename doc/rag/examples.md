Examples
========

Two runnable examples demonstrate RAG usage through agents:

ReAct Agent
-----------

Run: `python examples/rag_react_agent_demo.py`

Flow:
- `rag_ingest` indexes a directory or URLs (default: `./doc`).
- `rag_search` returns top snippets with sources.
- `rag_qa` produces an answer with [n] citations.

Graph Agent
-----------

Run: `python examples/rag_graph_agent_demo.py`

Flow: `RAGIngestNode -> RAGRetrieveNode -> RAGAnswerNode` implemented inline using `RagIndex`, `RagRetriever`, `RagQA`.

Configuration
-------------

Use env vars to switch backends and embeddings without code changes. For offline smoke:

```
export RAG_BACKEND=faiss
python examples/rag_react_agent_demo.py
```

To use AnyRoute or OpenAI embeddings, set `ANYROUTE_*` or `OPENAI_API_KEY` respectively.

Tip: running from source without installing the package? Either install with `pip install -e .` or prepend the repo root to `PYTHONPATH` when invoking examples:

```
PYTHONPATH=. RAG_BACKEND=faiss RAG_FAKE_QA=1 python examples/smoke/rag_faiss_smoke.py
```
