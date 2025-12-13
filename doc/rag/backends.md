Backends & Embeddings
=====================

Vector Stores
-------------

The RAG registry supports the following backends via `RAG_BACKEND`:

- `faiss` (default): 本地与离线友好。提供真实 FAISS 适配器（如未安装 `faiss` 将自动回退到内存余弦检索）。
- `pinecone`: 云端向量库（需要 `PINECONE_API_KEY`，默认索引名 `spoon-rag`，可用 `RAG_PINECONE_INDEX` 指定）。已提供真实适配器。
- `qdrant`: 本地/云端（需要 `pip install qdrant-client` 且服务默认 `http://localhost:6333`）。已提供真实适配器。
  - 支持嵌入式模式：`QDRANT_URL=:memory:` 或设置 `QDRANT_PATH=:memory:` 可在本地无服务运行。
- `chroma`: 本地（需要 `pip install chromadb`，默认持久目录 `${RAG_DIR:-.rag_store}/chroma`）。已提供真实适配器。

说明：为了保证离线测试稳定，`faiss` 仍默认以内存向量库运行。其余后端需要按需安装/配置，适配器层保持统一 API。

Embeddings
----------

- AnyRoute (OpenAI-compatible): set `ANYROUTE_API_KEY` and `ANYROUTE_BASE_URL` (optional `ANYROUTE_MODEL`).
- OpenAI: set `OPENAI_API_KEY` (uses `text-embedding-3-small` by default).
- Hash (fallback): deterministic offline embedding for tests and demos (no env needed).

Backend Smoke 测试
------------------

- FAISS：
  ```bash
  # 如未安装 faiss，会自动回退到内存实现
  RAG_BACKEND=faiss RAG_FAKE_QA=1 python examples/smoke/rag_faiss_smoke.py
  ```

- Pinecone：
  ```bash
  export PINECONE_API_KEY=...
  RAG_BACKEND=pinecone RAG_FAKE_QA=1 python examples/smoke/rag_pinecone_smoke.py
  ```

- Qdrant（需本地/远程服务与 `qdrant-client`）
  ```bash
  pip install qdrant-client
  export QDRANT_URL=http://localhost:6333  # 如需
  RAG_BACKEND=qdrant RAG_FAKE_QA=1 python examples/smoke/rag_qdrant_smoke.py
  ```

- Chroma（需 `chromadb`）
  ```bash
  pip install chromadb
  RAG_BACKEND=chroma RAG_FAKE_QA=1 python examples/smoke/rag_chroma_smoke.py
  ```
