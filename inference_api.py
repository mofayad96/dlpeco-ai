"""
FastAPI inference service for DLPEco edge agents (VM2 email, VM3 web).

Endpoints mirror dlp_core.classifiers.ml.handler expectations:
  POST /classify_email
  POST /classify_web
  POST /classify
  GET  /health
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from ai.ai_orchestrator import AIOrchestrator

API_TOKEN = os.getenv("AI_SERVICE_TOKEN", "default-secret-token")

app = FastAPI(title="DLPEco AI Inference", version="1.0.0")
_orchestrator: AIOrchestrator | None = None


def _orch() -> AIOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AIOrchestrator()
    return _orchestrator


def _auth(x_api_key: str | None) -> None:
    if not x_api_key or x_api_key != API_TOKEN:
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


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/classify_email")
def classify_email(
    req: ClassifyEmailRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> dict[str, Any]:
    _auth(x_api_key)
    return _orch().classify_email(
        raw_email=req.raw_email,
        subject=req.subject,
        body=req.body,
        from_addr=req.from_addr,
        to_addrs=req.to_addrs,
        metadata=req.metadata,
    )


@app.post("/classify_web")
def classify_web(
    req: ClassifyWebRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> dict[str, Any]:
    _auth(x_api_key)
    return _orch().classify_web(
        raw_http=req.raw_http,
        url=req.url,
        content=req.content,
        file_text=req.file_text,
        metadata=req.metadata,
    )


@app.post("/classify")
def classify(
    req: ClassifyRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> dict[str, Any]:
    _auth(x_api_key)
    return _orch().classify({
        "text": req.text,
        "metadata": req.metadata,
    })
