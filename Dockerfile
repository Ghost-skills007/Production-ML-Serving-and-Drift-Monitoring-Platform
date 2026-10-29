# syntax=docker/dockerfile:1

# ---- Build stage: install pinned dependencies into a virtualenv ----
FROM python:3.11-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# ---- Runtime stage: slim, non-root, healthchecked ----
FROM python:3.11-slim AS runtime
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PATH="/opt/venv/bin:$PATH" \
    MODEL_STORE=local MODEL_DIR=/app/models

# Create a non-root user.
RUN groupadd --system appuser && useradd --system --gid appuser --home /app appuser

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY app ./app
COPY pipeline ./pipeline
COPY monitoring ./monitoring
COPY scripts ./scripts
COPY requirements.txt ./

# Models directory (populated at runtime by the entrypoint or a mounted volume).
RUN mkdir -p /app/models && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status==200 else 1)"

CMD ["bash", "scripts/entrypoint.sh"]
