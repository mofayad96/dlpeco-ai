FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    AI_HOST=0.0.0.0 \
    AI_PORT=8001 \
    DISTILBERT_MODEL_PATH=/app/ai/models/finetuned_distilbert \
    ARABERT_MODEL_PATH=/app/ai/models/finetuned_arabert \
    SENTENCE_TRANSFORMERS_HOME=/app/ai/models/sentence_transformers \
    TRANSFORMERS_OFFLINE=0 \
    HF_HOME=/app/ai/models/huggingface \
    TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor

WORKDIR /app

# System packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user (UID/GID 1000)
RUN groupadd -g 1000 app && \
    useradd -m -u 1000 -g app -s /bin/bash app

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Copy application
COPY . /app/ai/

# Create writable directories
RUN mkdir -p \
    /tmp/torchinductor \
    /app/ai/models/huggingface && \
    chown -R app:app /app /tmp/torchinductor

# Switch to non-root user
USER app

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8001/health || exit 1

CMD ["uvicorn", "ai.inference_api:app", "--host", "0.0.0.0", "--port", "8001"]