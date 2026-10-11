# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-10-11

First production release. The 0.x beta is upgraded in place on first start.

### Added
- Installable package `enterprise-brain` with a CLI: `serve`, `migrate`, `create-admin`,
  `seed-demo`, `ingest`, `sync`, `ask`, `reindex`, `healthcheck`, `check-config`.
- Alembic migrations for SQLite and PostgreSQL, including an automatic in-place upgrade of beta
  databases; pgvector mirror table with cosine ranking and automatic numpy fallback.
- OpenAI-compatible LLM answers (Groq, OpenRouter, OpenAI, Ollama, vLLM, ...) using the full
  retrieved chunks, with timeouts, bounded retries with backoff, citation validation, and
  fallback to extractive answers; deterministic `mock` provider for offline tests and demos.
- Pluggable embeddings: local hashed (default), any OpenAI-compatible `/embeddings` API, or
  sentence-transformers; embeddings are tagged by model and `reindex` re-embeds everything.
- Word (.docx), HTML and JSON ingestion alongside PDF, TXT, Markdown and CSV; CSV rows are
  indexed as `column: value` sentences.
- Incremental connector sync by SHA-256 (new / updated / unchanged / removed files), per-file
  error reporting, uploads into a connector over the API/UI, and optional scheduled sync.
- Personal API keys (`X-API-Key` / `Bearer eb_...`), password change, admin password reset,
  role changes, deactivation and deletion with last-admin protection.
- Login rate limiting, session revocation through token versions, same-origin check for
  cookie-authenticated writes, CSP / frame / content-type / referrer / HSTS headers.
- Structured logging (JSON in production) with `X-Request-ID` propagation and request logs.
- `/api/health/ready` readiness probe (database check) and `/api/version`.
- Insights: unanswered questions, answer modes, average latency, member count.
- Non-root Docker image with healthcheck; Compose stack with PostgreSQL + pgvector.
- CI (ruff, mypy --strict, pytest on Python 3.11-3.13, PostgreSQL + pgvector job, wheel and
  Docker smoke tests) and a tag-triggered release workflow.
- Test suite of unit and integration tests, hardened with mutation testing.

### Changed
- Code moved from `app/` to the `enterprise_brain` package; templates, static assets and demo
  data ship inside the wheel. The UI no longer loads Tailwind, HTMX, Alpine or Chart.js from
  CDNs (pre-built CSS and vendored Chart.js).
- Configuration is validated at startup; production refuses weak secrets and default admin
  passwords. `SEED_ON_STARTUP` is replaced by `DEMO_DATA`, and the bootstrap admin is only
  created when `ADMIN_EMAIL`/`ADMIN_PASSWORD` are set.
- Recent queries are private to each member (admins can list everyone's); saved answers stay
  team-wide.
- `passlib` and `python-jose` replaced by `bcrypt` and `PyJWT`; unused `reportlab` removed.

### Fixed
- Re-syncing a connector no longer duplicates every chunk.
- Deleting a source now removes its chunks, documents, pgvector rows and files on disk.
- The LLM path used 160-character snippets as context; it now uses the full retrieved chunks.
- pgvector queries used a `:param::vector` cast that SQLAlchemy cannot bind, and fetched every
  chunk per query.
- Connectors with the same name no longer share a staging folder.
- Upload size is limited and client filenames are sanitised.

## [0.1.0-beta] - 2026-09

- Initial beta: ask UI, document uploads, staged connectors, insights, team invites.
