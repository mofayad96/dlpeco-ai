import os
import time
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel
from contextlib import asynccontextmanager
import uvicorn
from pathlib import Path
import sys

# Ensure parent directory is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai.ai_orchestrator import AIOrchestrator

# Security Token (should be set via environment variable)
AI_SERVICE_TOKEN = os.getenv("AI_SERVICE_TOKEN")
if not AI_SERVICE_TOKEN:
    print("WARNING: AI_SERVICE_TOKEN not set. Using default insecure token.")
    AI_SERVICE_TOKEN = "default-secret-token"

# Orchestrator instance
orchestrator = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize and Warm-up
    global orchestrator
    print("[API] Starting up: Initializing AI Orchestrator...")
    orchestrator = AIOrchestrator()
    print("[API] Warming up models...")
    # Warm-up request to force loading of weights into memory
    orchestrator.classify({"text": "warmup", "metadata": {"channel": "web"}})
    print("[API] AI Orchestrator Ready.")
    yield
    # Shutdown: Clean up if necessary
    print("[API] Shutting down.")

app = FastAPI(title="DLPEco AI Inference API", version="1.0.0", lifespan=lifespan)

async def verify_token(x_ai_token: str = Header(None)):
    if x_ai_token != AI_SERVICE_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid or missing X-AI-Token")

class ClassifyPayload(BaseModel):
    text: Optional[str] = ""
    metadata: Optional[Dict[str, Any]] = {}

class EmailPayload(BaseModel):
    raw_email: Optional[str] = ""
    subject: Optional[str] = ""
    body: Optional[str] = ""
    from_addr: Optional[str] = ""
    to_addrs: Optional[List[str]] = []
    attachment_text: Optional[str] = ""
    metadata: Optional[Dict[str, Any]] = {}

class WebPayload(BaseModel):
    raw_http: Optional[str] = ""
    url: Optional[str] = ""
    content: Optional[str] = ""
    file_text: Optional[str] = ""
    metadata: Optional[Dict[str, Any]] = {}

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": time.time()}

@app.post("/classify")
async def classify(payload: ClassifyPayload, x_ai_token: str = Header(None)):
    await verify_token(x_ai_token)
    # Using model_dump() for Pydantic v2
    return orchestrator.classify(payload.model_dump())

@app.post("/classify_email")
async def classify_email(payload: EmailPayload, x_ai_token: str = Header(None)):
    await verify_token(x_ai_token)
    return orchestrator.classify_email(**payload.model_dump())

@app.post("/classify_web")
async def classify_web(payload: WebPayload, x_ai_token: str = Header(None)):
    await verify_token(x_ai_token)
    return orchestrator.classify_web(**payload.model_dump())

if __name__ == "__main__":
    uvicorn.run("inference_api:app", host="0.0.0.0", port=8001, reload=False)
