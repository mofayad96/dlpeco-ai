FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source files and model weights
COPY . /app/ai/

# Set environment variables
ENV PYTHONPATH=/app
ENV AI_SERVICE_TOKEN=default-secret-token

# Expose the port
EXPOSE 8001

# Command to run the API using the module path
CMD ["uvicorn", "ai.inference_api:app", "--host", "0.0.0.0", "--port", "8001"]
