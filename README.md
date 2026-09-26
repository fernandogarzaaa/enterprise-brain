# Enterprise Brain

Unified AI intelligence layer: ask questions in natural language across your company's documents, contracts, and data. (Beta)

## Quickstart

```bash
cp .env.example .env
docker compose up --build
# open http://localhost:8002
# login: admin@clusterx.local / ChangeMe123!
```

Local dev (SQLite):

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python -m app.seed
uvicorn app.main:app --port 8002
```

## How it works

**Knowledge sources.** Two kinds, both first-class:

- *Documents* — upload PDF, TXT, Markdown, or CSV on the Knowledge Sources page. Each file is ingested immediately: text extraction (pypdf for PDFs) → overlapping chunks (600 chars, 120 overlap, snapped to sentence boundaries) → embeddings → stored.
- *Connectors* — CRM, ERP, Google Drive, SharePoint, Email, Contracts, Vendors, or Custom. A connector is a named record with a status and a staging directory at `data/staging/<name>/`. Drop export files there and press **Sync**; the sync imports every supported file through the same ingest pipeline and updates the connector's `last_sync_at`. Nothing is faked: if the staging directory is empty, sync honestly reports zero files.

**Embeddings.** `EMBEDDINGS=hashed` (default) uses deterministic hashed numpy embeddings — zero extra dependencies. `EMBEDDINGS=st` lazily imports `sentence-transformers` if installed and falls back to hashed with a logged warning if not (it is intentionally *not* in `requirements.txt`).

**Vector store.** Retrieval is abstracted in `app/vectorstore.py`:

- `PythonCosineStore` — cosine similarity in Python over the embeddings on each `Chunk` row. Works on any database, always.
- `PgVectorStore` — used only when `DATABASE_URL` points at PostgreSQL. Best-effort: creates the `vector` extension and a `chunk_vectors` side table (256-dim, matching the default hashed embeddings) and queries with the `<=>` operator. Any setup or query failure falls back to the Python store, and `upsert` never blocks ingest.

The test suite runs on SQLite, so the Python path is the tested one; pgvector is exercised in production via `docker compose` (the compose DB is the `pgvector/pgvector` image).

**Ask.** `POST /api/ask {question, source_ids?}` retrieves top-k chunks and builds a cited answer. Local mode uses an extractive summarizer: the sentences covering the most query terms, cited as `[1]`, `[2]` against their source names. When `LLM_BASE_URL` + `LLM_API_KEY` are set, the retrieved context is passed through the LLM for a richer answer with the same citation mapping. Every question is stored and shown in Recent Queries; answers can be starred into Saved Answers.

**Insights.** Query counts, per-source chunk coverage, and top query topics (stopword-filtered term frequencies over asked questions).

**Team.** Admins can invite members; each invite creates a member user and returns a one-time temporary password to share.

## Pages

Ask (`/`) · Knowledge Sources (`/sources`) · Insights (`/insights`) · Saved Answers (`/saved`) · Team (`/team`) · Settings (`/settings`)

## Seed data

`python -m app.seed` creates the admin user plus 13 sources through the real ingest pipeline: 10 sample documents (Acme Corp MSA, Q3 revenue by client, approved vendors, invoice/expense policy, data retention, Globex SOW, remote-work security, CloudServe contract, leave policy, sales pipeline) and 3 connectors (Salesforce CRM, NetSuite ERP, Google Drive) whose staged export files are synced for real.

## Tests

```bash
python -m pytest -q
ruff check app
```

## License

MIT
