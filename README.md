# Enterprise Brain

[![ci](https://github.com/fernandogarzaaa/enterprise-brain/actions/workflows/ci.yml/badge.svg)](https://github.com/fernandogarzaaa/enterprise-brain/actions/workflows/ci.yml)
[![release](https://img.shields.io/github/v/release/fernandogarzaaa/enterprise-brain)](https://github.com/fernandogarzaaa/enterprise-brain/releases)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Unified AI intelligence layer: ask questions in natural language across your company's
documents, contracts, and data — and get answers with citations back to the source.

- **Knowledge sources** — upload PDF, Word, TXT, Markdown, CSV, HTML or JSON, or create
  connectors (CRM, ERP, Google Drive, SharePoint, email, contracts, vendors, custom) that
  incrementally index the exports you drop into them.
- **Cited answers** — every answer cites `[1]`, `[2]` back to the sources it used. Works fully
  offline with an extractive answerer; plug in any OpenAI-compatible LLM (Groq, OpenRouter,
  OpenAI, a local Ollama/vLLM) for generated answers with automatic fallback.
- **Team ready** — admin/member roles, invites, password resets, deactivation, personal API
  keys, login rate limiting, per-user query history, shared starred answers, usage insights.
- **Production essentials** — env-based config validated at startup, structured JSON logs with
  request ids, Alembic migrations (SQLite or PostgreSQL + pgvector), health/readiness probes,
  security headers, Docker image and Compose stack, CLI for ops tasks.

## Quickstart

### Docker Compose (PostgreSQL + pgvector)

```bash
cp .env.example .env
# edit .env: set SECRET_KEY, ADMIN_EMAIL, ADMIN_PASSWORD, POSTGRES_PASSWORD
#            (and DEMO_DATA=true to load sample documents)
docker compose up -d --build
open http://localhost:8002
```

### Local (SQLite)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e .                       # or: pip install enterprise_brain-1.0.0-py3-none-any.whl
export ADMIN_EMAIL=admin@example.com ADMIN_PASSWORD='ChangeMe123!' DEMO_DATA=true
enterprise-brain serve                 # http://localhost:8002
```

Sign in with the `ADMIN_EMAIL` / `ADMIN_PASSWORD` you set. The first admin is created only when
the database has no users; afterwards manage people on the **Team** page or with
`enterprise-brain create-admin`.

### Use an LLM (optional, free options)

Extractive answers need nothing extra. For generated answers point the app at any
OpenAI-compatible endpoint:

| Provider | `LLM_BASE_URL` | `LLM_CHAT_MODEL` | `LLM_API_KEY` |
|---|---|---|---|
| Groq (free tier) | `https://api.groq.com/openai/v1` | `llama-3.1-8b-instant` | your Groq key |
| OpenRouter (free models) | `https://openrouter.ai/api/v1` | a `:free` model | your OpenRouter key |
| Ollama (local) | `http://localhost:11434/v1` | `llama3.1` | *(empty)* |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` | your OpenAI key |

If the LLM fails or times out, the extractive answer is returned with a warning instead of an
error. `LLM_PROVIDER=mock` runs a deterministic offline stand-in for demos and tests.

## Configuration

All settings come from environment variables (or a `.env` file); see
[`.env.example`](.env.example) for the full list. The important ones:

| Variable | Default | Notes |
|---|---|---|
| `ENVIRONMENT` | `development` | `production` refuses to start with a weak `SECRET_KEY` or default admin password, enables secure cookies and HSTS, JSON logs, and hides `/docs`. |
| `SECRET_KEY` | random per process in dev | **Required in production** (32+ chars). Signs session tokens. |
| `DATABASE_URL` | `sqlite:///./data/enterprise_brain.db` | `postgresql+psycopg2://…` for Postgres; `postgres://` URLs are accepted. |
| `DATA_DIR` / `STAGING_DIR` | `./data` / `DATA_DIR/staging` | Uploaded files and connector staging folders. |
| `ADMIN_EMAIL` / `ADMIN_PASSWORD` | empty | Bootstrap admin when there are no users. |
| `DEMO_DATA` | `false` | Load 10 sample documents and 3 connectors (idempotent). |
| `LLM_PROVIDER` | `auto` | `auto` · `none` · `mock` · `openai`. |
| `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_CHAT_MODEL` | — | Any OpenAI-compatible chat API. |
| `EMBEDDINGS` | `hashed` | `hashed` (local), `openai` (any `/embeddings` API via `EMBED_BASE_URL`/`EMBED_MODEL`), `st` (sentence-transformers, `pip install enterprise-brain[st]`). Run `enterprise-brain reindex` after changing. |
| `MAX_UPLOAD_MB` | `25` | Per-file limit for uploads and staged files. |
| `LOGIN_RATE_LIMIT` / `LOGIN_RATE_WINDOW_SECONDS` | `10` / `300` | Failed logins per IP+email per window. |
| `CORS_ORIGINS` | empty | Comma-separated browser origins allowed to call the API. |
| `SYNC_INTERVAL_MINUTES` | `0` | Background sync of all connectors (enable on one instance only). |
| `LOG_LEVEL` / `LOG_FORMAT` | `INFO` / `auto` | `auto` = JSON in production, text otherwise. |

Validate a configuration without starting the server: `enterprise-brain check-config`.

## Using it

**Web UI** — Ask (`/`), Knowledge Sources, Insights, Saved Answers, Team, Settings (password,
API keys, and the server configuration for admins).

**API** — interactive docs at `/docs` (disabled in production). Authenticate with a session
cookie, a bearer token from `POST /api/auth/login`, or a personal API key
(`X-API-Key: eb_…` or `Authorization: Bearer eb_…`, created under Settings).

```bash
KEY=eb_xxxxxxxxxxxx_...
curl -s -X POST localhost:8002/api/ask -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
  -d '{"question": "Which invoices are overdue?"}'
curl -s -X POST localhost:8002/api/sources/upload -H "X-API-Key: $KEY" -F file=@contract.pdf
```

| Endpoint | Purpose |
|---|---|
| `GET /api/health`, `GET /api/health/ready` | Liveness / readiness (checks the database). |
| `POST /api/auth/login`, `POST /api/auth/logout`, `GET /api/auth/me` | Sessions. |
| `POST /api/auth/password`, `GET/POST/DELETE /api/auth/api-keys` | Account security. |
| `POST /api/ask` | `{question, source_ids?, top_k?}` → `{answer, citations, mode, latency_ms, warnings}`. |
| `GET /api/queries`, `GET /api/queries/{id}` | Your question history (admins: `?everyone=true`). |
| `POST /api/queries/{id}/save`, `GET /api/saved`, `DELETE /api/saved/{id}` | Starred answers. |
| `GET /api/sources`, `GET /api/sources/{id}`, `POST /api/sources/upload`, `DELETE /api/sources/{id}` | Documents. |
| `POST /api/connectors`, `POST /api/connectors/{id}/files`, `POST /api/connectors/{id}/sync` | Connectors. |
| `GET /api/team`, `POST /api/team/invite`, `PATCH/DELETE /api/team/{id}`, `POST /api/team/{id}/reset-password` | Team (changes are admin-only). |
| `GET /api/insights` | Usage analytics. |

**CLI** — `enterprise-brain serve | migrate | create-admin | seed-demo | ingest FILE… |
sync [ID…|--all] | ask "QUESTION" [--json] | reindex | healthcheck | check-config`.

## Architecture

```
            ┌──────────── FastAPI app (enterprise_brain.main) ─────────────┐
 browser ─▶ │ pages (Jinja)  ─┐                                             │
 API/CLI ─▶ │ /api routers ───┼─▶ ask ──▶ vector store ──▶ DB (chunks)      │
            │                 │     └──▶ chat provider (none/mock/openai)   │
            │                 ├─▶ sources ─▶ ingest: extract → chunk → embed│
            │ auth: cookie │ bearer JWT │ API key; rate limit; CSRF origin  │
            └──────────────────────────────────────────────────────────────┘
   SQLAlchemy 2 + Alembic   ·   SQLite or PostgreSQL (+ pgvector mirror table)
```

- **Ingest** (`ingest.py`): text extraction (pypdf, python-docx, HTML/CSV/JSON parsers) →
  overlapping ~600-char chunks snapped to sentence boundaries → embeddings tagged with the
  embedding model id → stored per document. Each file is a `Document` tracked by SHA-256, so
  connector syncs are incremental (new / changed / unchanged / removed).
- **Retrieval** (`vectorstore.py`): exact cosine similarity with numpy over the current
  model's embeddings; on PostgreSQL with pgvector, embeddings are mirrored into
  `chunk_vectors` and ranked with `<=>`, falling back to numpy on any error.
- **Answering** (`ask.py`): extractive answer from the best-matching sentences with citations;
  with an LLM, the full retrieved chunks are sent numbered by source and invalid citation
  markers are stripped. Every question is stored with its mode and latency.
- **Security** (`security.py`, `deps.py`): bcrypt passwords, HS256 JWT sessions revoked by a
  per-user token version (password change, reset, deactivation), hashed API keys, sliding
  window login limiter, same-origin check for cookie-authenticated writes, CSP and other
  security headers.

## Deploy

- **Docker**: `docker build -t enterprise-brain .` produces a non-root image with a
  `HEALTHCHECK`; mount a volume at `/data`. `docker compose up -d` runs it with PostgreSQL +
  pgvector.
- **Any VM / PaaS**: `pip install enterprise_brain-1.0.0-py3-none-any.whl` (from the
  [release](https://github.com/fernandogarzaaa/enterprise-brain/releases)) and run
  `enterprise-brain serve --workers 2` behind a TLS-terminating reverse proxy. Set
  `FORWARDED_ALLOW_IPS` to the proxy address so client IPs (used by the login limiter) are
  correct.
- **Migrations** run automatically at startup (`AUTO_MIGRATE=true`, once in the parent process
  before workers start) or explicitly with `enterprise-brain migrate`. Upgrading from the 0.x
  beta is automatic: its tables are altered in place and existing users, sources, chunks and
  saved answers are kept.
- **Backups**: back up the database and `DATA_DIR` (uploaded originals and staging folders).
- **Scaling notes**: the login limiter is per process; for strict limits across replicas add a
  proxy rate limit. Enable `SYNC_INTERVAL_MINUTES` on one instance only (or run
  `enterprise-brain sync --all` from cron).

## Development

```bash
pip install -r requirements-dev.txt && pip install -e .
pytest                                   # SQLite; set TEST_DATABASE_URL=postgresql+psycopg2://… for Postgres
ruff check enterprise_brain tests && ruff format --check enterprise_brain tests
mypy enterprise_brain tests
testteeth run --pytest-args tests/unit   # mutation testing of the core modules
```

The stylesheet is pre-built with Tailwind (no CDN at runtime); after editing templates run
`tailwindcss -c frontend/tailwind.config.js -i frontend/input.css -o enterprise_brain/static/app.css --minify`.

## License

[MIT](LICENSE). Security issues: see [SECURITY.md](SECURITY.md).
