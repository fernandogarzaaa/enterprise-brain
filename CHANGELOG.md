# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added / Changed
- .gitignore for Python caches, local .env and runtime data (data/*.db, data/uploads/)
- requirements.lock pinning transitive dependencies (uv pip compile, Python 3.12)
- Dependabot config for pip, Docker base image and GitHub Actions (weekly)
- SECURITY.md with private reporting contact
