"""
FastAPI inference service for DLPEco edge agents (VM2 email, VM3 web).

Endpoints mirror dlp_core.classifiers.ml.handler expectations:
  POST /classify_email
  POST /classify_web
  POST /classify
  GET  /health
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from ai.ai_orchestrator import AIOrchestrator

API_TOKEN = os.getenv("AI_SERVICE_TOKEN", "default-secret-token")
BLOCK_LABELS = {"Confidential", "Restricted"}
DEFAULT_MIN_CONFIDENCE = float(os.getenv("AI_SECONDARY_LAYER_MIN_CONFIDENCE", "0.75"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [AI] %(message)s",
    stream=sys.stdout,
    force=True,
)
log = logging.getLogger("ai.inference")

app = FastAPI(title="DLPEco AI Inference", version="1.0.0")
_orchestrator: AIOrchestrator | None = None


def _orch() -> AIOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AIOrchestrator()
    return _orchestrator


def _preview(text: str, limit: int = 160) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _block_decision(label: str, confidence: float) -> tuple[bool, str]:
    if label not in BLOCK_LABELS:
        return False, f"label {label!r} is not a DLP block label (need one of {sorted(BLOCK_LABELS)})"
    if confidence < DEFAULT_MIN_CONFIDENCE:
        return False, (
            f"confidence {confidence:.1%} below threshold {DEFAULT_MIN_CONFIDENCE:.1%}"
        )
    return True, "meets secondary-layer quarantine criteria"


def _log_request(route: str, *, text_len: int, preview: str, metadata: dict[str, Any]) -> None:
    meta_bits = ", ".join(f"{k}={v!r}" for k, v in sorted(metadata.items()) if v not in (None, "", [], {}))
    log.info(
        "REQUEST %s text_len=%s preview=%r metadata=[%s]",
        route,
        text_len,
        preview,
        meta_bits,
    )


def _log_result(route: str, result: dict[str, Any], *, elapsed_ms: float) -> None:
    label = str(result.get("label") or "")
    confidence = float(result.get("confidence") or 0.0)
    would_block, reason = _block_decision(label, confidence)
    violations = result.get("violations") or result.get("llm", {}).get("sensitivity_indicators") or []
    log.info(
        "RESULT %s label=%s confidence=%.1f%% domain=%s language=%s "
        "would_block=%s reason=%s violations=%s compliance_tags=%s latency=%.1fms",
        route,
        label,
        confidence * 100,
        result.get("primary_domain") or result.get("context_domain"),
        result.get("language"),
        would_block,
        reason,
        violations[:8],
        (result.get("compliance_tags") or [])[:8],
        elapsed_ms,
    )
    if not would_block:
        log.info(
            "RESULT %s NOT quarantined — %s (triggered_by=%s channel_fired=%s)",
            route,
            reason,
            result.get("triggered_by"),
            (result.get("channel") or {}).get("channel_fired"),
        )
    else:
        log.info("RESULT %s WOULD quarantine for admin review", route)


def _auth(x_api_key: str | None) -> None:
    if not x_api_key or x_api_key != API_TOKEN:
        log.warning("AUTH rejected — missing or invalid X-API-Key")
        raise HTTPException(status_code=401, detail="Invalid API key")


class ClassifyEmailRequest(BaseModel):
    raw_email: str = ""
    subject: str = ""
    body: str = ""
    from_addr: str = ""
    to_addrs: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    scan_scope: list[str] = Field(default_factory=list)


class ClassifyWebRequest(BaseModel):
    raw_http: str = ""
    url: str = ""
    content: str = ""
    file_text: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    scan_scope: list[str] = Field(default_factory=list)


class ClassifyRequest(BaseModel):
    text: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    scan_scope: list[str] = Field(default_factory=list)


@app.middleware("http")
async def log_non_health_requests(request: Request, call_next):
    if request.url.path == "/health":
        return await call_next(request)
    started = time.perf_counter()
    client = request.client.host if request.client else "unknown"
    log.info("HTTP %s %s from %s", request.method, request.url.path, client)
    try:
        response = await call_next(request)
    except Exception as exc:
        log.exception("HTTP %s %s failed after %.1fms: %s", request.method, request.url.path, (time.perf_counter() - started) * 1000, exc)
        raise
    elapsed_ms = (time.perf_counter() - started) * 1000
    log.info(
        "HTTP %s %s -> %s in %.1fms",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


@app.on_event("startup")
def _startup_banner() -> None:
    log.info(
        "Inference API ready — block_labels=%s min_confidence=%.0f%% token_set=%s",
        sorted(BLOCK_LABELS),
        DEFAULT_MIN_CONFIDENCE * 100,
        bool(API_TOKEN),
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/classify_email")
def classify_email(
    req: ClassifyEmailRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> dict[str, Any]:
    _auth(x_api_key)
    text = req.raw_email or req.body or ""
    _log_request(
        "/classify_email",
        text_len=len(text),
        preview=_preview(text),
        metadata=req.metadata,
    )
    started = time.perf_counter()
    result = _orch().classify_email(
        raw_email=req.raw_email,
        subject=req.subject,
        body=req.body,
        from_addr=req.from_addr,
        to_addrs=req.to_addrs,
        metadata=req.metadata,
    )
    _log_result("/classify_email", result, elapsed_ms=(time.perf_counter() - started) * 1000)
    return result


@app.post("/classify_web")
def classify_web(
    req: ClassifyWebRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> dict[str, Any]:
    _auth(x_api_key)
    text = req.raw_http or req.content or ""
    _log_request(
        "/classify_web",
        text_len=len(text),
        preview=_preview(text),
        metadata=req.metadata,
    )
    started = time.perf_counter()
    result = _orch().classify_web(
        raw_http=req.raw_http,
        url=req.url,
        content=req.content,
        file_text=req.file_text,
        metadata=req.metadata,
    )
    _log_result("/classify_web", result, elapsed_ms=(time.perf_counter() - started) * 1000)
    return result


@app.post("/classify")
def classify(
    req: ClassifyRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> dict[str, Any]:
    _auth(x_api_key)
    _log_request(
        "/classify",
        text_len=len(req.text or ""),
        preview=_preview(req.text),
        metadata=req.metadata,
    )
    started = time.perf_counter()
    result = _orch().classify({
        "text": req.text,
        "metadata": req.metadata,
    })
    _log_result("/classify", result, elapsed_ms=(time.perf_counter() - started) * 1000)
    return result
