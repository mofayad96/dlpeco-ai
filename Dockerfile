FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1     PYTHONDONTWRITEBYTECODE=1     PYTHONPATH=/app     AI_HOST=0.0.0.0     AI_PORT=8001     AI_ENABLE_LLM=0     DISTILBERT_MODEL_PATH=/app/ai/models/finetuned_distilbert     ARABERT_MODEL_PATH=/app/ai/models/finetuned_arabert     SENTENCE_TRANSFORMERS_HOME=/app/ai/models/sentence_transformers     TRANSFORMERS_OFFLINE=0

WORKDIR /app

RUN apt-get update     && apt-get install -y --no-install-recommends build-essential curl     && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip     && pip install --no-cache-dir -r /tmp/requirements.txt

COPY . /app/ai/

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3     CMD curl -fsS http://127.0.0.1:8001/health || exit 1

CMD ["uvicorn", "ai.inference_api:app", "--host", "0.0.0.0", "--port", "8001"]
