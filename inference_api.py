import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from ai.ai_orchestrator import AIOrchestrator, LABELS

AI_SERVICE_TOKEN = os.getenv("AI_SERVICE_TOKEN")
if not AI_SERVICE_TOKEN:
    raise RuntimeError("AI_SERVICE_TOKEN must be set; use the same value in every client X-API-Key header")

DISTILBERT_MODEL_PATH = Path(os.getenv("DISTILBERT_MODEL_PATH", "/app/ai/models/finetuned_distilbert"))
ARABERT_MODEL_PATH = Path(os.getenv("ARABERT_MODEL_PATH", "/app/ai/models/finetuned_arabert"))

orchestrator: AIOrchestrator | None = None
startup_status: dict[str, Any] = {"ready": False, "models": {}}


def _validate_model_dir(path: Path, name: str) -> dict[str, Any]:
    required_any = {
        "weights": ["model.safetensors", "pytorch_model.bin", "tf_model.h5"],
        "tokenizer": ["tokenizer.json", "vocab.txt", "spiece.model"],
    }
    if not path.exists() or not path.is_dir():
        raise RuntimeError(f"{name} model directory does not exist: {path}")

    files = {child.name for child in path.iterdir() if child.is_file()}
    missing = []
    if "config.json" not in files:
        missing.append("config.json")
    for group, options in required_any.items():
        if not any(option in files for option in options):
            missing.append(f"one of {options}")

    if missing:
        raise RuntimeError(f"{name} model directory is incomplete at {path}; missing {', '.join(missing)}")

    return {"path": str(path), "files": sorted(files)}


def _scores(result: dict[str, Any]) -> dict[str, float]:
    raw_scores = result.get("scores") or result.get("all_scores", {}).get("fused") or {}
    scores = {label: round(float(raw_scores.get(label, 0.0)), 4) for label in LABELS}
    total = sum(scores.values())
    if total <= 0:
        label = result.get("label", "Public")
        confidence = float(result.get("confidence") or 0.0)
        scores = {item: 0.0 for item in LABELS}
        scores[label if label in scores else "Public"] = round(confidence, 4)
        total = sum(scores.values())
    if total > 0 and abs(total - 1.0) > 0.05:
        scores = {label: round(value / total, 4) for label, value in scores.items()}
    return scores


def _normalize_response(result: dict[str, Any], channel: str) -> dict[str, Any]:
    label = result.get("label") if result.get("label") in LABELS else "Public"
    
    # Extract specific violations from the LLM/Preprocessor metadata
    llm_info = result.get("llm") or {}
    violations = list(llm_info.get("sensitivity_indicators") or [])
    
    return {
        "label": label,
        "confidence": round(float(result.get("confidence") or 0.0), 4),
        "scores": _scores(result),
        "context_domain": result.get("context_domain") or result.get("primary_domain") or "General",
        "violations": violations,
        "compliance_tags": list(result.get("compliance_tags") or []),
        "channel": channel,
        "language": result.get("language") or "en",
    }


def _log_result(desc: str, r: dict[str, Any]):
    import sys
    ch = r.get("channel", {})
    lm = r.get("llm", {})
    print(f"\n[{desc}]")
    print(f"  Label       : {r.get('label')} ({r.get('confidence', 0):.2%})")
    print(f"  Domain      : {r.get('primary_domain')} → {r.get('domains')}")
    print(f"  Language    : {r.get('language')}")
    print(f"  Channel     : ext={ch.get('is_external')} "
          f"cloud={ch.get('is_cloud_host')} "
          f"dir={ch.get('direction')} "
          f"fired={ch.get('channel_fired')}")
    print(f"  Compliance  : {r.get('compliance_tags')}")
    print(f"  LLM model   : {lm.get('model_used')} (fallback={lm.get('fallback')})")
    print(f"  LLM context : {lm.get('business_context')}")
    print(f"  LLM signals : {lm.get('sensitivity_indicators')}")
    print(f"  Latency     : {r.get('latency_ms', 0)}ms")
    sys.stdout.flush()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global orchestrator, startup_status
    print("[API] Validating model directories...")
    startup_status["models"] = {
        "distilbert": _validate_model_dir(DISTILBERT_MODEL_PATH, "DistilBERT"),
        "arabert": _validate_model_dir(ARABERT_MODEL_PATH, "AraBERT"),
    }
    print("[API] Starting AI orchestrator...")
    orchestrator = AIOrchestrator()
    print("[API] Warming up models...")
    orchestrator.classify({"text": "warmup", "metadata": {"channel": "management"}})
    startup_status["ready"] = True
    print("[API] AI inference service ready.")
    yield
    startup_status["ready"] = False
    print("[API] Shutting down.")


app = FastAPI(title="DLPEco AI Inference API", version="1.0.0", lifespan=lifespan)


def verify_token(x_api_key: str = Header(None, alias="X-API-Key")) -> None:
    if x_api_key != AI_SERVICE_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid or missing X-API-Key")


class ClassifyPayload(BaseModel):
    text: Optional[str] = ""
    scan_scope: Optional[List[str]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EmailPayload(BaseModel):
    raw_email: Optional[str] = ""
    subject: Optional[str] = ""
    body: Optional[str] = ""
    body_text: Optional[str] = ""
    from_addr: Optional[str] = ""
    to_addrs: List[str] = Field(default_factory=list)
    attachment_filenames: List[str] = Field(default_factory=list)
    attachment_text: Optional[str] = ""
    scan_scope: Optional[List[str]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WebPayload(BaseModel):
    raw_http: Optional[str] = ""
    url: Optional[str] = ""
    method: Optional[str] = ""
    request_body_snippet: Optional[str] = ""
    content: Optional[str] = ""
    content_type: Optional[str] = ""
    destination_domain: Optional[str] = ""
    user_agent: Optional[str] = ""
    file_text: Optional[str] = ""
    scan_scope: Optional[List[str]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


@app.get("/health")
async def health():
    return {"status": "healthy" if startup_status["ready"] else "starting", "timestamp": time.time()}


@app.post("/classify")
async def classify(payload: ClassifyPayload, x_api_key: str = Header(None, alias="X-API-Key")):
    verify_token(x_api_key)
    result = orchestrator.classify(payload.model_dump())
    channel = payload.metadata.get("channel") or "ManagementChannel"
    
    # Apply scan_scope filtering
    if payload.scan_scope:
        result["llm"]["sensitivity_indicators"] = [
            v for v in (result["llm"].get("sensitivity_indicators") or [])
            if v in payload.scan_scope
        ]
        # If the specific violations requested are NOT found, downgrade label
        if not result["llm"]["sensitivity_indicators"] and result["label"] == "Restricted":
            result["label"] = "Internal" # Downgrade to baseline sensitive if specific one missed

    _log_result("Classify: Generic request", result)
    return _normalize_response(result, str(channel))


@app.post("/classify_email")
async def classify_email(payload: EmailPayload, x_api_key: str = Header(None, alias="X-API-Key")):
    verify_token(x_api_key)
    body = payload.body or payload.body_text or ""
    metadata = dict(payload.metadata)
    metadata.setdefault("attachment_filenames", payload.attachment_filenames)
    metadata.setdefault("attachment_count", len(payload.attachment_filenames))
    result = orchestrator.classify_email(
        raw_email=payload.raw_email or "",
        subject=payload.subject or "",
        body=body,
        from_addr=payload.from_addr or "",
        to_addrs=payload.to_addrs,
        attachment_text=payload.attachment_text or "",
        metadata=metadata,
    )
    
    # Apply scan_scope filtering
    if payload.scan_scope:
        result["llm"]["sensitivity_indicators"] = [
            v for v in (result["llm"].get("sensitivity_indicators") or [])
            if v in payload.scan_scope
        ]
        if not result["llm"]["sensitivity_indicators"] and result["label"] == "Restricted":
            result["label"] = "Internal"

    desc = f"Email: {payload.subject}" if payload.subject else "Email: No subject"
    _log_result(desc, result)
    return _normalize_response(result, "EmailChannel")


@app.post("/classify_web")
async def classify_web(payload: WebPayload, x_api_key: str = Header(None, alias="X-API-Key")):
    verify_token(x_api_key)
    metadata = dict(payload.metadata)
    if payload.method:
        metadata.setdefault("method", payload.method.upper())
        metadata.setdefault("protocol", "HTTP")
    if payload.content_type:
        metadata.setdefault("content_type", payload.content_type)
    if payload.destination_domain:
        metadata.setdefault("host", payload.destination_domain)
    if payload.user_agent:
        metadata.setdefault("user_agent", payload.user_agent)
    content = payload.content or payload.request_body_snippet or ""
    result = orchestrator.classify_web(
        raw_http=payload.raw_http or "",
        url=payload.url or "",
        content=content,
        file_text=payload.file_text or "",
        metadata=metadata,
    )

    # Apply scan_scope filtering
    if payload.scan_scope:
        result["llm"]["sensitivity_indicators"] = [
            v for v in (result["llm"].get("sensitivity_indicators") or [])
            if v in payload.scan_scope
        ]
        if not result["llm"]["sensitivity_indicators"] and result["label"] == "Restricted":
            result["label"] = "Internal"

    desc = f"Web: {payload.url}" if payload.url else "Web: Request"
    _log_result(desc, result)
    return _normalize_response(result, "WebChannel")


if __name__ == "__main__":
    uvicorn.run("ai.inference_api:app", host="0.0.0.0", port=8001, reload=False)
