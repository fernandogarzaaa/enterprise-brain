# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| 1.0.x | Yes |
| < 1.0 (beta) | No — upgrade to 1.0 (the database is migrated in place) |

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security problems.

Report vulnerabilities privately through GitHub's
[private vulnerability reporting](https://github.com/fernandogarzaaa/enterprise-brain/security/advisories/new)
or by email to **fernandogarzaaa@gmail.com**. Include a description, steps to reproduce, and the
affected version or commit.

You can expect an acknowledgement within a few days. Please allow a reasonable window to
investigate and release a fix before any public disclosure.

## Hardening checklist for deployments

- Run with `ENVIRONMENT=production` and a random `SECRET_KEY` of at least 32 characters
  (the app refuses to start otherwise).
- Serve over HTTPS behind a reverse proxy; production mode sets `Secure` cookies and HSTS.
  Set `FORWARDED_ALLOW_IPS` to your proxy so client addresses are trusted only from it.
- Use a strong `ADMIN_PASSWORD` for the bootstrap admin and change it after first sign-in;
  remove it from the environment once the admin exists.
- Keep `.env` and API keys out of version control. API keys are shown once and stored only as
  SHA-256 hashes; revoke unused keys under Settings.
- Restrict `CORS_ORIGINS` to the exact origins that need browser access to the API.
- Add a proxy-level rate limit if you run several replicas (the login limiter is per process).
- Back up the database and `DATA_DIR`; uploaded documents may contain sensitive data.
- When an external LLM or embedding API is configured, retrieved document text is sent to that
  provider. Use a local model (e.g. Ollama) or `LLM_PROVIDER=none` for data that must not leave
  your network.

## Security features

bcrypt password hashing, HS256 session tokens revoked on password change/reset/deactivation,
hashed API keys, sliding-window login rate limiting, same-origin enforcement for
cookie-authenticated state changes (plus `SameSite=Lax`), CSP, `X-Frame-Options: DENY`,
`nosniff`, strict referrer policy, sanitised upload filenames and size limits, input
validation on every endpoint, and a non-root container image.
