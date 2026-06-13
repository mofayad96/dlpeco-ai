"""
============================================================
FILE 2: ai_orchestrator.py  —  v6  (LLM-integrated)
============================================================

FIXES vs v5:

  FIX A — channel_fired now only triggers when label is Restricted
           or Confidential. Previously fired on ANY web/email traffic
           including benign public contact forms.

  FIX B — LLM domain merge capped: only merges llm_domains when
           LLM returns 3 or fewer domains (uncertain LLM outputs
           with long domain lists are ignored). Total domains
           capped at 4 to prevent domain list explosion on
           complex payloads like GraphQL PII exports.

  FIX C — LLM safety cap: if LLM explicitly found zero sensitive
           signals and contains_sensitive=False, final label is
           capped at Internal. Prevents DistilBERT/AraBERT from
           over-classifying benign internal traffic (e.g. IT
           maintenance notices) when LLM found nothing sensitive.

  UNCHANGED from v5:
    LLM 1-6 — full LLM preprocessing integration
    FIX 1   — Channel-aware text building
    FIX 2   — Metadata token prefix builder
    FIX 3   — compliance_tags from classify() result dict
    FIX 4   — Web payload parser
    FIX 5   — Email payload parser
    FIX 6   — Channel escalation boost

WHAT THIS FILE DOES:
  0. LLM preprocessing (Gemma via Ollama, or local fallback)
  1. Parse raw web/email traffic into structured payload
  2. Detect language
  3. Build semantic text (with metadata tokens) for BERT models
  4. Build channel text (with full HTTP/SMTP structure) for context
  5. Run DistilBERT (English), AraBERT (Arabic)
  6. Run context classifier with channel text + LLM domain hints
  7. Fuse semantic scores
  7b. LLM safety cap (FIX C)
  8. Apply context risk + channel risk weighting
  9. Return structured result to inference_api.py
============================================================
"""

import re
import sys
import time
import torch
import numpy as np
from pathlib import Path

# ── MODEL PATHS ───────────────────────────────────────────────────────────
DISTILBERT_PATH = "ai/models/finetuned_distilbert"
ARABERT_PATH    = "ai/models/finetuned_arabert"

# ── LABELS ────────────────────────────────────────────────────────────────
LABELS   = ["Public", "Internal", "Confidential", "Restricted"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}

# ── FUSION WEIGHTS ────────────────────────────────────────────────────────
DISTILBERT_WEIGHT = 0.55
ARABIC_WEIGHT     = 0.55

# ── CONTEXT RISK WEIGHTING ────────────────────────────────────────────────
SEMANTIC_W = 0.70
CONTEXT_W  = 0.30

# ── CHANNEL ESCALATION ────────────────────────────────────────────────────
CHANNEL_ESCALATION_BOOST = 0.05

# ── CLOUD STORAGE HOSTS ───────────────────────────────────────────────────
CLOUD_EXFIL_HOSTS = {
    "dropbox.com", "wetransfer.com", "drive.google.com",
    "onedrive.live.com", "mega.nz", "box.com",
    "mediafire.com", "sendspace.com", "anonfiles.com",
    "personal.onedrive.com", "4shared.com",
}

# ── PERSONAL EMAIL DOMAINS ────────────────────────────────────────────────
PERSONAL_EMAIL_DOMAINS = {
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com",
    "protonmail.com", "icloud.com", "aol.com", "live.com",
    "yandex.com", "mail.com",
}

# ── SENSITIVE FILE EXTENSIONS ─────────────────────────────────────────────
SENSITIVE_EXTENSIONS = {
    ".csv", ".sql", ".db", ".sqlite", ".env", ".key", ".pem",
    ".xlsx", ".xls", ".mdb", ".bak", ".dump", ".tar", ".gz",
}

MAX_TOKEN_LENGTH = 128

# ── FIX B: domain merge limits ────────────────────────────────────────────
LLM_DOMAIN_MERGE_MAX  = 3   # only merge when LLM returns this many or fewer domains
DOMAIN_DISPLAY_CAP    = 4   # max domains shown in result


# ══════════════════════════════════════════════════════════════════════════
# LANGUAGE DETECTION
# ══════════════════════════════════════════════════════════════════════════

def detect_language(text: str) -> str:
    arabic = len(re.findall(r"[\u0600-\u06FF]", text))
    latin  = len(re.findall(r"[a-zA-Z]", text))
    total  = arabic + latin or 1
    ratio  = arabic / total
    if ratio > 0.70: return "ar"
    if ratio > 0.15: return "mixed"
    return "en"


# ══════════════════════════════════════════════════════════════════════════
# WEB PAYLOAD PARSER
# ══════════════════════════════════════════════════════════════════════════

def parse_web_payload(raw: str) -> dict:
    result = {
        "method":            "",
        "host":              "",
        "path":              "",
        "filename":          "",
        "content_body":      "",
        "is_cloud_host":     False,
        "is_sensitive_file": False,
        "metadata":          {"channel": "web"},
    }
    if not raw:
        return result

    lines = raw.strip().splitlines()
    if lines:
        m = re.match(r"^(GET|POST|PUT|DELETE|PATCH|OPTIONS)\s+(\S+)", lines[0], re.I)
        if m:
            result["method"] = m.group(1).upper()
            result["path"]   = m.group(2)

    for line in lines[1:]:
        if ":" not in line:
            break
        key, _, val = line.partition(":")
        key = key.strip().lower()
        val = val.strip()
        if key == "host":
            result["host"] = val
            result["metadata"]["url_category"] = _categorise_host(val)
            result["is_cloud_host"] = val.lower() in CLOUD_EXFIL_HOSTS
        elif key == "content-type":
            result["metadata"]["content_type"] = val

    fn_match = re.search(r'filename[="\s]*([^\s";\n]+)', raw, re.I)
    if fn_match:
        result["filename"] = fn_match.group(1).strip('"')
        ext = Path(result["filename"]).suffix.lower()
        result["is_sensitive_file"] = ext in SENSITIVE_EXTENSIONS
        result["metadata"]["file_type"] = ext.lstrip(".")

    parts = re.split(r"\n\s*\n", raw, maxsplit=1)
    if len(parts) > 1:
        result["content_body"] = parts[1].strip()[:800]

    result["metadata"]["method"]        = result["method"]
    result["metadata"]["is_external"]   = True
    result["metadata"]["host"]          = result["host"]
    result["metadata"]["is_cloud_host"] = result["is_cloud_host"]
    return result


def _categorise_host(host: str) -> str:
    host = host.lower()
    if host in CLOUD_EXFIL_HOSTS:
        return "shadow_it"
    if any(h in host for h in ("drive.google", "sharepoint", "onedrive")):
        return "cloud_storage"
    if any(h in host for h in ("slack.com", "teams.microsoft", "zoom.us")):
        return "collaboration"
    if any(h in host for h in ("gmail", "yahoo", "hotmail", "protonmail")):
        return "webmail"
    return "unknown"


# ══════════════════════════════════════════════════════════════════════════
# EMAIL PAYLOAD PARSER
# ══════════════════════════════════════════════════════════════════════════

def parse_email_payload(raw: str) -> dict:
    result = {
        "from_addr":   "",
        "to_addrs":    [],
        "subject":     "",
        "body":        "",
        "attachments": [],
        "is_external": False,
        "direction":   "unknown",
        "metadata":    {"channel": "email"},
    }
    if not raw:
        return result

    lines       = raw.strip().splitlines()
    header_done = False
    body_lines  = []

    for line in lines:
        if not header_done:
            if not line.strip():
                header_done = True
                continue
            lower = line.lower()
            if lower.startswith("from:"):
                result["from_addr"] = line.split(":", 1)[1].strip()
            elif lower.startswith("to:"):
                to_raw = line.split(":", 1)[1].strip()
                result["to_addrs"] = [a.strip() for a in re.split(r"[,;]", to_raw)]
                for addr in result["to_addrs"]:
                    domain = addr.split("@")[-1].lower() if "@" in addr else ""
                    if domain in PERSONAL_EMAIL_DOMAINS:
                        result["is_external"] = True
                        break
            elif lower.startswith("subject:"):
                result["subject"] = line.split(":", 1)[1].strip()
        else:
            body_lines.append(line)

    result["body"] = "\n".join(body_lines).strip()[:800]

    for fn_match in re.finditer(r'filename[="\s]*([^\s";\n]+)', raw, re.I):
        fn = fn_match.group(1).strip('"')
        if fn not in result["attachments"]:
            result["attachments"].append(fn)

    if result["is_external"] or any(
        d in PERSONAL_EMAIL_DOMAINS
        for d in [result["from_addr"].split("@")[-1].lower()]
    ):
        result["direction"] = "outbound"
    else:
        result["direction"] = "internal"

    result["metadata"].update({
        "is_external":      result["is_external"],
        "direction":        result["direction"],
        "attachment_count": len(result["attachments"]),
        "recipient_count":  len(result["to_addrs"]),
        "protocol":         "smtp",
        "file_type":        Path(result["attachments"][0]).suffix.lstrip(".")
                            if result["attachments"] else "",
    })
    return result


# ══════════════════════════════════════════════════════════════════════════
# METADATA TOKEN BUILDER
# ══════════════════════════════════════════════════════════════════════════

_URL_CATEGORY_TOKENS = {
    "cloud_storage": "[URL:cloud_storage]",
    "file_sharing":  "[URL:file_sharing]",
    "webmail":       "[URL:webmail]",
    "social_media":  "[URL:social_media]",
    "shadow_it":     "[URL:shadow_it]",
    "business":      "[URL:business]",
    "unknown":       "[URL:unknown]",
}

_FILE_TYPE_TOKENS = {
    "csv":  "[FILE:csv]", "xlsx": "[FILE:xlsx]", "docx": "[FILE:docx]",
    "pdf":  "[FILE:pdf]", "zip":  "[FILE:zip]",  "sql":  "[FILE:sql]",
    "txt":  "[FILE:txt]", "pptx": "[FILE:pptx]", "py":   "[FILE:py]",
    "sh":   "[FILE:sh]",  "env":  "[FILE:env]",  "key":  "[FILE:key]",
}


def build_metadata_prefix(metadata: dict) -> str:
    if not metadata:
        return ""
    tokens = []
    if metadata.get("is_external") is True:
        tokens.append("[EXTERNAL]")
    elif metadata.get("is_external") is False:
        tokens.append("[INTERNAL]")

    direction = (metadata.get("direction") or "").upper()
    if direction in ("OUTBOUND", "INBOUND", "INTERNAL"):
        tokens.append(f"[{direction}]")

    protocol = (metadata.get("protocol") or "").upper()
    if protocol in ("SMTP", "HTTP", "HTTPS", "DNS"):
        tokens.append(f"[{protocol}]")

    url_cat = (metadata.get("url_category") or "").lower()
    if url_cat:
        tokens.append(_URL_CATEGORY_TOKENS.get(url_cat, f"[URL:{url_cat}]"))

    file_type = (metadata.get("file_type") or "").lower().lstrip(".")
    if file_type:
        tokens.append(_FILE_TYPE_TOKENS.get(file_type, f"[FILE:{file_type}]"))

    att = metadata.get("attachment_count")
    if att and att > 0:
        tokens.append(f"[ATT:{att}]")

    rcpt = metadata.get("recipient_count")
    if rcpt and rcpt > 0:
        tokens.append(f"[RCPT:{rcpt}]")

    if metadata.get("is_cloud_host"):
        tokens.append("[SHADOW_IT]")

    return " ".join(tokens) + " " if tokens else ""


# ══════════════════════════════════════════════════════════════════════════
# DIRECT MODEL LOADER
# ══════════════════════════════════════════════════════════════════════════

class DirectModelLoader:
    def __init__(self, model_path: str, name: str):
        self.name       = name
        self.model_path = model_path
        self.tokenizer  = None
        self.model      = None
        self.loaded     = False
        self._load()

    def _load(self):
        if not Path(self.model_path).exists():
            print(f"[{self.name}] Model not found at {self.model_path}")
            return
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            print(f"[{self.name}] Loading from {self.model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model     = AutoModelForSequenceClassification.from_pretrained(
                self.model_path
            )
            self.model.eval()
            self.loaded = True
            print(f"[{self.name}] Ready.")
        except Exception as e:
            print(f"[{self.name}] Failed to load: {e}")

    def predict(self, text: str) -> dict:
        if not self.loaded:
            return None
        try:
            inputs = self.tokenizer(
                text, return_tensors="pt",
                truncation=True, padding=True,
                max_length=MAX_TOKEN_LENGTH,
            )
            with torch.no_grad():
                logits = self.model(**inputs).logits
            probs    = torch.softmax(logits, dim=-1)[0]
            pred_id  = int(torch.argmax(probs))
            id2label = self.model.config.id2label
            label    = id2label.get(pred_id, LABELS[pred_id] if pred_id < len(LABELS) else "Public")
            conf     = float(probs[pred_id])
            all_scores = {
                id2label.get(i, f"label_{i}"): round(float(p), 4)
                for i, p in enumerate(probs)
            }
            return {"label": label, "confidence": round(conf, 4), "all_scores": all_scores}
        except Exception as e:
            print(f"[{self.name}] Inference error: {e}")
            return None


class LazyLoader:
    def __init__(self, factory):
        self._factory  = factory
        self._instance = None

    def get(self):
        if self._instance is None:
            self._instance = self._factory()
        return self._instance


# ══════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR  v6
# ══════════════════════════════════════════════════════════════════════════

class AIOrchestrator:

    def __init__(self, llm_model: str = None):
        print("[Orchestrator] Initializing v6 (LLM-integrated, web + email aware)...")
        self._llm = LazyLoader(self._load_llm_preprocessor(llm_model))
        self._distilbert = LazyLoader(
            lambda: DirectModelLoader(DISTILBERT_PATH, "DistilBERT")
        )
        self._arabert = LazyLoader(
            lambda: DirectModelLoader(ARABERT_PATH, "AraBERT")
        )
        self._context = LazyLoader(self._load_context_classifier)
        print("[Orchestrator] Ready. Models load on first request.")

    def _load_llm_preprocessor(self, model: str = None):
        def _factory():
            sys.path.insert(0, str(Path(__file__).parent.parent))
            try:
                from ai.llm.llm_preprocessor import LLMPreprocessor
                kwargs = {"model": model} if model else {}
                prep   = LLMPreprocessor(**kwargs)
                status = prep.status()
                print(f"[Orchestrator] LLM preprocessor: "
                      f"mode={status['mode']}  model={status['model']}")
                return prep
            except ImportError:
                print("[Orchestrator] llm_preprocessor.py not found — "
                      "preprocessing skipped. Place at ai/llm/llm_preprocessor.py")
                return None
            except Exception as e:
                print(f"[Orchestrator] LLM preprocessor failed to load: {e}")
                return None
        return _factory

    def _load_context_classifier(self):
        sys.path.insert(0, str(Path(__file__).parent.parent))
        try:
            from ai.models.context_classifier import ContextClassifier
            return ContextClassifier()
        except Exception as e:
            print(f"[Orchestrator] ContextClassifier failed: {e}")
            return None

    # ── Helpers ───────────────────────────────────────────────────────────

    def _uniform(self) -> dict:
        return {l: 0.25 for l in LABELS}

    def _to_label_probs(self, result: dict) -> dict:
        if not result:
            return self._uniform()
        scores = result.get("all_scores", {})
        out = {l: 0.01 for l in LABELS}
        for key, val in scores.items():
            if key in out:
                out[key] = float(val)
        if max(out.values()) <= 0.01:
            label = result.get("label")
            conf  = result.get("confidence", 0.5)
            if label and label in out:
                out = {l: 0.01 for l in LABELS}
                out[label] = conf
        total = sum(out.values()) or 1
        return {l: round(v / total, 4) for l, v in out.items()}

    def _fuse(self, probs_list: list, weights: list) -> dict:
        fused   = {l: 0.0 for l in LABELS}
        total_w = sum(weights)
        for probs, w in zip(probs_list, weights):
            for label in LABELS:
                fused[label] += probs.get(label, 0.0) * (w / total_w)
        total = sum(fused.values()) or 1
        return {l: round(v / total, 4) for l, v in fused.items()}

    def _apply_context_risk(self, semantic_conf: float,
                             context_risk: float) -> tuple:
        final      = round(min(semantic_conf * SEMANTIC_W + context_risk * CONTEXT_W, 1.0), 4)
        risk_delta = round(final - semantic_conf, 4)
        return final, risk_delta

    # ── Text builders ─────────────────────────────────────────────────────

    def _build_semantic_text(self, payload: dict, metadata: dict) -> str:
        meta_prefix = build_metadata_prefix(metadata)
        parts = []
        subject = payload.get("subject", "")
        if subject:
            parts.append(f"Subject: {subject}")
        body = payload.get("text") or payload.get("body", "")
        if body:
            parts.append(body[:600])
        att = payload.get("attachment_text", "")
        if att:
            parts.append(f"Attachment: {att[:400]}")
        filename = payload.get("filename", "")
        if filename:
            parts.append(f"filename={filename}")
        content = "\n".join(parts)
        return (meta_prefix + content).strip()[:1500]

    def _build_channel_text(self, payload: dict) -> str:
        parts = []
        raw = payload.get("raw", "")
        if raw:
            parts.append(raw[:1000])
        else:
            if payload.get("method"):
                host = payload.get("host", "")
                path = payload.get("path", "")
                parts.append(f"{payload['method']} {path} HTTP/1.1")
                if host:
                    parts.append(f"Host: {host}")
            from_addr = payload.get("from_addr", "")
            to_addrs  = payload.get("to_addrs", [])
            subject   = payload.get("subject", "")
            if from_addr:
                parts.append(f"From: {from_addr}")
            if to_addrs:
                parts.append(f"To: {', '.join(to_addrs)}")
            if subject:
                parts.append(f"Subject: {subject}")
            filename = payload.get("filename", "")
            if filename:
                parts.append(f"filename={filename}")
            body = payload.get("text") or payload.get("body", "")
            if body:
                parts.append(body[:400])
            att = payload.get("attachment_text", "")
            if att:
                parts.append(f"Attachment: {att[:300]}")
        return "\n".join(parts).strip()[:1500]

    # ── Main classify ─────────────────────────────────────────────────────

    def classify(self, payload: dict) -> dict:
        t_start  = time.time()
        metadata = payload.get("metadata") or {}

        # ── Step 0: LLM preprocessing ──────────────────────────────────────
        llm_result     = None
        llm_domains    = []
        llm_indicators = []
        llm_used       = "none"

        llm_prep = self._llm.get()
        if llm_prep is not None:
            try:
                raw_for_llm = (
                    payload.get("raw", "")
                    or payload.get("text", "")
                    or payload.get("body", "")
                )
                subject = payload.get("subject", "")
                if subject:
                    raw_for_llm = f"Subject: {subject}\n\n{raw_for_llm}"
                filename = payload.get("filename", "")
                if filename and filename not in raw_for_llm:
                    raw_for_llm = f"filename={filename}\n{raw_for_llm}"
                att = payload.get("attachment_text", "")
                if att:
                    raw_for_llm += f"\n\nAttachment:\n{att[:600]}"

                if raw_for_llm.strip():
                    llm_result     = llm_prep.process(raw_for_llm)
                    llm_domains    = llm_result.domains_detected or []
                    llm_indicators = llm_result.sensitivity_indicators or []
                    llm_used       = llm_result.model_used
            except Exception as e:
                print(f"[Orchestrator] LLM preprocessing error: {e}")
                llm_result = None

        # ── Step 1: Build texts ────────────────────────────────────────────
        if llm_result and llm_result.processed and llm_result.classifier_input():
            meta_prefix   = build_metadata_prefix(metadata)
            semantic_text = (meta_prefix + llm_result.classifier_input()).strip()[:1500]
        else:
            semantic_text = self._build_semantic_text(payload, metadata)

        channel_text = self._build_channel_text(payload)

        # ── Step 2: Language detection ────────────────────────────────────
        if llm_result and llm_result.language in ("en", "ar", "mixed"):
            language = llm_result.language
        else:
            language = detect_language(semantic_text)

        raw_preds = {}
        probs     = {}

        # ── Step 3: DistilBERT (English) ──────────────────────────────────
        if language in ("en", "mixed"):
            db_model = self._distilbert.get()
            if db_model and db_model.loaded:
                r = db_model.predict(semantic_text)
                if r:
                    raw_preds["distilbert"] = r
                    probs["distilbert"]     = self._to_label_probs(r)
                else:
                    probs["distilbert"] = self._uniform()
            else:
                probs["distilbert"] = self._uniform()

        # ── Step 4: AraBERT (Arabic) ──────────────────────────────────────
        if language in ("ar", "mixed"):
            ar_model = self._arabert.get()
            if ar_model and ar_model.loaded:
                r = ar_model.predict(semantic_text)
                if r:
                    raw_preds["arabic"] = r
                    probs["arabic"]     = self._to_label_probs(r)
                else:
                    probs["arabic"] = self._uniform()
            else:
                probs["arabic"] = self._uniform()

        # ── Step 5: Context classification ────────────────────────────────
        ctx_clf    = self._context.get()
        ctx_result = {
            "primary_domain":  "General",
            "domains":         ["General"],
            "confidence":      0.0,
            "context_risk":    0.2,
            "compliance_tags": [],
            "all_scores":      {},
            "mode":            "unavailable",
        }
        if ctx_clf:
            try:
                preliminary = "Internal"
                if probs:
                    all_p = list(probs.values())[0]
                    preliminary = max(all_p, key=all_p.get)

                ctx_result = ctx_clf.classify(channel_text, sensitivity=preliminary)

                # FIX B — only merge LLM domains when confident (≤3 domains returned)
                # Avoids domain list explosion when LLM is uncertain
                if llm_domains and len(llm_domains) <= LLM_DOMAIN_MERGE_MAX:
                    merged = list(ctx_result.get("domains", []))
                    for d in llm_domains:
                        if d not in merged:
                            merged.append(d)
                    # Cap total domains to avoid noise in audit log
                    ctx_result["domains"]         = merged[:DOMAIN_DISPLAY_CAP]
                    ctx_result["llm_domain_hint"] = llm_domains
                elif llm_domains:
                    # LLM returned too many domains — uncertain, don't merge
                    ctx_result["llm_domain_hint"] = llm_domains

            except Exception as e:
                print(f"[Orchestrator] Context error: {e}")

        # ── Step 6: Fuse semantic scores ──────────────────────────────────
        prob_list, weight_list = [], []
        triggered_by = "uniform"

        if "distilbert" in probs and "arabic" in probs:
            prob_list    = [probs["distilbert"], probs["arabic"]]
            weight_list  = [DISTILBERT_WEIGHT, ARABIC_WEIGHT]
            triggered_by = "distilbert+arabert"
        elif "arabic" in probs:
            prob_list, weight_list = [probs["arabic"]], [1.0]
            triggered_by = "arabert"
        elif "distilbert" in probs:
            prob_list, weight_list = [probs["distilbert"]], [1.0]
            triggered_by = "distilbert"
        else:
            prob_list, weight_list = [self._uniform()], [1.0]

        fused       = self._fuse(prob_list, weight_list)
        final_label = max(fused, key=fused.get)
        sem_conf    = fused[final_label]

        # ── Step 6b: LLM safety cap (FIX C) ──────────────────────────────
        # If LLM explicitly found ZERO sensitive signals and
        # contains_sensitive=False, cap at Internal.
        # Prevents over-classification of benign internal traffic
        # (e.g. IT maintenance notices, internal announcements).
        # Only applies when LLM ran successfully (not fallback).
        if (llm_result
                and llm_result.processed
                and not llm_result.fallback_used
                and not llm_result.contains_sensitive_content
                and not llm_result.sensitivity_indicators):
            if final_label not in ("Public", "Internal"):
                final_label = "Internal"
                sem_conf    = fused.get("Internal", 0.5)

        # ── Step 7: Context risk weighting ────────────────────────────────
        context_risk           = ctx_result.get("context_risk", 0.2)
        final_conf, risk_delta = self._apply_context_risk(sem_conf, context_risk)

        # ── Step 8: Channel escalation (FIX A) ───────────────────────────
        # FIX A: channel_fired only True when label is actually sensitive.
        # Previously fired on ANY web/email traffic including public forms.
        active_domains = ctx_result.get("domains", ["General"])
        channel_fired  = (
            ("WebChannel" in active_domains or "EmailChannel" in active_domains)
            and final_label in ("Restricted", "Confidential")   # FIX A
        )
        if channel_fired:
            final_conf = round(min(final_conf + CHANNEL_ESCALATION_BOOST, 1.0), 4)
            risk_delta = round(final_conf - sem_conf, 4)

        # ── Step 9: Compliance tags ────────────────────────────────────────
        compliance_tags = ctx_result.get("compliance_tags", [])
        if ctx_clf:
            try:
                ctx_final       = ctx_clf.classify(channel_text, sensitivity=final_label)
                compliance_tags = ctx_final.get("compliance_tags", [])
            except Exception:
                pass

        elapsed = round((time.time() - t_start) * 1000, 1)

        channel_info = {
            "is_external":   metadata.get("is_external", False),
            "is_cloud_host": metadata.get("is_cloud_host", False),
            "direction":     metadata.get("direction", ""),
            "protocol":      metadata.get("protocol", ""),
            "file_type":     metadata.get("file_type", ""),
            "url_category":  metadata.get("url_category", ""),
            "channel_fired": channel_fired,
        }

        return {
            "label":           final_label,
            "confidence":      final_conf,
            "primary_domain":  ctx_result.get("primary_domain", "General"),
            "domains":         active_domains,
            "language":        language,
            "risk_delta":      risk_delta,
            "context_risk":    context_risk,
            "compliance_tags": compliance_tags,
            "channel":         channel_info,
            "latency_ms":      elapsed,
            "triggered_by":    triggered_by,
            "llm": {
                "model_used":             llm_used,
                "fallback":               llm_result.fallback_used if llm_result else True,
                "business_context":       llm_result.business_context if llm_result else "",
                "sensitivity_indicators": llm_indicators,
                "domain_hints":           llm_domains,
                "contains_sensitive":     llm_result.contains_sensitive_content if llm_result else False,
                "encoding_detected":      llm_result.encoding_detected if llm_result else "none",
            },
            "all_scores": {
                "distilbert": probs.get("distilbert", {}),
                "arabert":    probs.get("arabic", {}),
                "context":    ctx_result.get("all_scores", {}),
                "fused":      fused,
            },
        }

    # ── Public entry points ───────────────────────────────────────────────

    def classify_email(self,
                       raw_email:       str  = "",
                       subject:         str  = "",
                       body:            str  = "",
                       from_addr:       str  = "",
                       to_addrs:        list = None,
                       attachment_text: str  = "",
                       metadata:        dict = None) -> dict:
        if raw_email:
            parsed = parse_email_payload(raw_email)
            parsed["raw"]             = raw_email
            parsed["attachment_text"] = attachment_text
            parsed["metadata"].update(metadata or {})
            return self.classify(parsed)

        to_addrs = to_addrs or []
        is_ext   = any(
            a.split("@")[-1].lower() in PERSONAL_EMAIL_DOMAINS
            for a in to_addrs if "@" in a
        )
        meta = {
            "channel":          "email",
            "is_external":      is_ext,
            "direction":        "outbound" if is_ext else "internal",
            "protocol":         "smtp",
            "attachment_count": 1 if attachment_text else 0,
            "recipient_count":  len(to_addrs),
        }
        meta.update(metadata or {})
        return self.classify({
            "subject":         subject,
            "text":            body,
            "attachment_text": attachment_text,
            "from_addr":       from_addr,
            "to_addrs":        to_addrs,
            "metadata":        meta,
        })

    def classify_web(self,
                     raw_http:  str  = "",
                     url:       str  = "",
                     content:   str  = "",
                     file_text: str  = "",
                     metadata:  dict = None) -> dict:
        if raw_http:
            parsed = parse_web_payload(raw_http)
            parsed["raw"]             = raw_http
            parsed["attachment_text"] = file_text
            parsed["metadata"].update(metadata or {})
            return self.classify(parsed)

        host     = re.sub(r"https?://", "", url).split("/")[0] if url else ""
        is_cloud = host in CLOUD_EXFIL_HOSTS
        url_cat  = _categorise_host(host) if host else "unknown"
        meta = {
            "channel":       "web",
            "is_external":   True,
            "url":           url,
            "host":          host,
            "url_category":  url_cat,
            "is_cloud_host": is_cloud,
        }
        meta.update(metadata or {})
        return self.classify({
            "text":            content,
            "attachment_text": file_text,
            "metadata":        meta,
        })


# ══════════════════════════════════════════════════════════════════════════
# TEST
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    orc = AIOrchestrator()
    print("\n" + "=" * 70)
    print("ORCHESTRATOR END-TO-END TEST  v6")
    print("=" * 70)

    tests = [
        ("Web: cloud upload of payroll CSV",
         "classify_web",
         {"raw_http": (
             "POST /upload HTTP/1.1\n"
             "Host: dropbox.com\n"
             "Content-Type: multipart/form-data\n"
             "filename=payroll_Q4.csv\n\n"
             "employee_id,name,national_id,bank_account,routing,net_salary\n"
             "12345,Ahmed Kamal,29001011234567,1234567890,021000021,25000"
         )}),

        ("Web: GraphQL query exporting PII",
         "classify_web",
         {"raw_http": (
             'POST /graphql HTTP/1.1\n'
             'Host: internal-app.company.com\n\n'
             '{"query": "{ users { id email ssn credit_card national_id salary } }"}'
         )}),

        ("Web: public contact form",
         "classify_web",
         {"raw_http": (
             "POST /contact HTTP/1.1\n"
             "Host: company.com\n\n"
             "name=Jane+Doe&email=jane@gmail.com&message=Hello+I+would+like+more+info"
         )}),

        ("Email: outbound to Gmail with medical record",
         "classify_email",
         {"raw_email": (
             "From: doctor@hospital.com\n"
             "To: patient@gmail.com\n"
             "Subject: Your test results\n\n"
             "Dear Ahmed, your HIV test came back positive. "
             "Your NID: 29001011234567. Please contact us."
         )}),

        ("Email: M&A merger doc to external",
         "classify_email",
         {"raw_email": (
             "From: cfo@company.com\n"
             "To: advisor@goldmansachs.com\n"
             "Subject: Q4 merger strategy FINAL\n\n"
             "Please find attached the acquisition term sheet. "
             "Board has not been informed yet. Strictly confidential."
         )}),

        ("Email: internal IT maintenance notice",
         "classify_email",
         {"raw_email": (
             "From: it@company.com\n"
             "To: staff@company.com\n"
             "Subject: VPN maintenance Saturday\n\n"
             "Dear team, VPN will be offline Saturday 8am for maintenance."
         )}),

        ("Arabic email: Egyptian NID + medical",
         "classify_email",
         {"raw_email": (
             "From: dr.sami@hospital-eg.com\n"
             "To: patient@gmail.com\n"
             "Subject: نتيجة التحليل\n\n"
             "مريض محمد عبدالله، الرقم القومي: 29001011234567، "
             "التشخيص: السكر النوع التاني، الدواء: ميتفورمين"
         )}),

        ("Plain text: English confidential",
         "classify",
         {"text": "Net loss EGP 2.3M. Merger planned. Board unaware.",
          "metadata": {"channel": "email", "is_external": True}}),
    ]

    for desc, method, payload in tests:
        if method == "classify_web":
            r = orc.classify_web(**payload)
        elif method == "classify_email":
            r = orc.classify_email(**payload)
        else:
            r = orc.classify(payload)

        ch = r.get("channel", {})
        lm = r.get("llm", {})
        print(f"\n[{desc}]")
        print(f"  Label       : {r['label']} ({r['confidence']:.2%})")
        print(f"  Domain      : {r['primary_domain']} → {r['domains']}")
        print(f"  Language    : {r['language']}")
        print(f"  Channel     : ext={ch.get('is_external')} "
              f"cloud={ch.get('is_cloud_host')} "
              f"dir={ch.get('direction')} "
              f"fired={ch.get('channel_fired')}")
        print(f"  Compliance  : {r['compliance_tags']}")
        print(f"  LLM model   : {lm.get('model_used')} (fallback={lm.get('fallback')})")
        print(f"  LLM context : {lm.get('business_context')}")
        print(f"  LLM signals : {lm.get('sensitivity_indicators')}")
        print(f"  Latency     : {r['latency_ms']}ms")