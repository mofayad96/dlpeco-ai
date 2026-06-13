import os
import time
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel
import uvicorn

# Ensure the parent directory is in sys.path so 'ai.' imports work
import sys
from pathlib import Path
# We assume this is run from the directory containing the 'ai' folder
# or that the 'ai' folder is in the python path.
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from ai.ai_orchestrator import AIOrchestrator
except ImportError:
    # Fallback if run from within the ai directory
    from ai_orchestrator import AIOrchestrator

app = FastAPI(title="DLPEco AI Inference API", version="1.0.0")

# Security Token (should be set via environment variable)
AI_SERVICE_TOKEN = os.getenv("AI_SERVICE_TOKEN")
if not AI_SERVICE_TOKEN:
    print("WARNING: AI_SERVICE_TOKEN not set. Using default insecure token.")
    AI_SERVICE_TOKEN = "default-secret-token"

# Initialize Orchestrator lazily or at startup
orchestrator = None

@app.on_event("startup")
async def startup_event():
    global orchestrator
    print("[API] Initializing AI Orchestrator...")
    orchestrator = AIOrchestrator()
    print("[API] AI Orchestrator Ready.")

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
    result = orchestrator.classify(payload.dict())
    return result

@app.post("/classify_email")
async def classify_email(payload: EmailPayload, x_ai_token: str = Header(None)):
    await verify_token(x_ai_token)
    result = orchestrator.classify_email(**payload.dict())
    return result

@app.post("/classify_web")
async def classify_web(payload: WebPayload, x_ai_token: str = Header(None)):
    await verify_token(x_ai_token)
    result = orchestrator.classify_web(**payload.dict())
    return result

if __name__ == "__main__":
    uvicorn.run("inference_api:app", host="0.0.0.0", port=8001, reload=False)
