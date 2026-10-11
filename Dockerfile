# syntax=docker/dockerfile:1
FROM python:3.12-slim AS build
WORKDIR /src
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
COPY requirements.txt ./
RUN pip wheel --wheel-dir /wheels -r requirements.txt
COPY pyproject.toml README.md LICENSE ./
COPY enterprise_brain ./enterprise_brain
RUN pip wheel --no-deps --wheel-dir /wheels .

FROM python:3.12-slim
LABEL org.opencontainers.image.title="Enterprise Brain" \
      org.opencontainers.image.source="https://github.com/fernandogarzaaa/enterprise-brain" \
      org.opencontainers.image.licenses="MIT"
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DATA_DIR=/data \
    DATABASE_URL=sqlite:////data/enterprise_brain.db \
    PORT=8002
RUN useradd --create-home --uid 10001 app && mkdir -p /data && chown app:app /data
COPY --from=build /wheels /wheels
RUN pip install --no-index --find-links /wheels enterprise-brain[postgres] && rm -rf /wheels
USER app
WORKDIR /home/app
VOLUME ["/data"]
EXPOSE 8002
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 CMD ["enterprise-brain", "healthcheck"]
CMD ["enterprise-brain", "serve"]
