"""
============================================================
llm_preprocessor.py  --  v3  (Gemma intelligent understanding)
============================================================
ROLE:
  Intelligent language understanding layer BEFORE classifiers.
  Gemma reads raw traffic, understands its business meaning,
  extracts sensitive content verbatim, and returns structured
  data that the orchestrator feeds to DistilBERT/AraBERT/Context.

PIPELINE:
  Raw traffic (email/web/any format)
    down
  LLMPreprocessor (this file)
    down  structured JSON
  AIOrchestrator
    down  routes to
  DistilBERT (English sensitivity)
  AraBERT    (Arabic sensitivity)
  ContextClassifier (domain)

CHANGES vs v2:
  FIX 1 -- format="json" added to Ollama call to force valid JSON output,
           fixes malformed/incomplete JSON on mixed Arabic/English content.
  FIX 2 -- Enriched classifier_input(): emits Context + Domains on
           separate lines before segments, giving classifiers stronger signal.
  FIX 3 -- language_confidence field added. Enables smart routing in
           orchestrator (AraBERT vs DistilBERT vs both).
  FIX 4 -- File saved as UTF-8 with no BOM. No null bytes.

ORCHESTRATOR ROUTING HINT:
  lang       = result.language
  confidence = result.language_confidence

  if lang == "ar" and confidence >= 0.7:   AraBERT only
  elif lang == "en" and confidence >= 0.7: DistilBERT only
  else:                                    run both, take higher score

INSTALL:
  pip install ollama
  ollama pull gemma:2b          # fast, 1.4GB
  ollama pull gemma:7b          # better quality, 4.7GB (recommended)
  ollama pull qwen2.5:7b        # best for Arabic+English mixed
  ollama serve                  # start Ollama server
============================================================
"""

import re
import json
import time
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ── CONFIG ────────────────────────────────────────────────────────────────
OLLAMA_HOST       = "http://localhost:11434"
GEMMA_MODEL       = "qwen2.5:7b"       # change to gemma:7b or qwen2.5:7b for better quality
MAX_INPUT_CHARS   = 3000
MAX_SEGMENT_CHARS = 400

# ── SYSTEM PROMPT ─────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an intelligent content analyzer for a DLP (Data Loss Prevention) security system.

Your job is to UNDERSTAND raw network traffic and extract its business meaning for security classification.

You MUST respond with ONLY valid JSON. No explanation, no markdown, no code blocks.

CRITICAL RULES:
1. NEVER summarize away sensitive information.
   Always preserve verbatim: names, IDs, medical data, financial figures, credentials, legal content.
2. Arabic text must remain Arabic. English must remain English. Mixed content stays mixed. DO NOT TRANSLATE.
3. Extract the ACTUAL sensitive content -- classifiers need the real text, not a description of it.
4. For long content: extract the most sensitive sections verbatim as important_segments.
5. Decode URL-encoding (%20 to space), strip HTTP headers, extract JSON/HTML body content.
6. language_confidence: float 0.0 to 1.0 indicating certainty about detected language.

Return this exact JSON structure:
{
  "clean_text": "main readable content with all sensitive data preserved verbatim",
  "business_context": "business purpose in one phrase e.g. Employee payroll data or Medical patient record or M&A strategy",
  "language": "en or ar or mixed",
  "language_confidence": 0.95,
  "content_type": "email or web_form or api_payload or file_upload or code or chat or document or general",
  "contains_sensitive_content": true,
  "sensitivity_indicators": ["national_id", "credit_card", "medical_record"],
  "domains_detected": ["Finance", "Medical", "Legal", "HR", "IT", "Security", "Operations", "CustomerData", "InternalComms"],
  "important_segments": ["verbatim sensitive excerpt 1", "verbatim sensitive excerpt 2"],
  "encoding_detected": "none or base64 or url_encoded or html_entities or multipart or json or xml"
}

sensitivity_indicators examples: national_id, credit_card, iban, medical_diagnosis, prescription,
credentials, api_key, private_key, salary_data, employee_records, merger_details, legal_contract,
customer_database, personal_data, egyptian_nid, egyptian_iban"""


# ══════════════════════════════════════════════════════════════════════════
# RESULT DATACLASS
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class PreprocessResult:
    clean_text:                 str   = ""
    business_context:           str   = ""
    language:                   str   = "en"
    language_confidence:        float = 0.0
    content_type:               str   = "general"
    contains_sensitive_content: bool  = False
    sensitivity_indicators:     list  = field(default_factory=list)
    domains_detected:           list  = field(default_factory=list)
    important_segments:         list  = field(default_factory=list)
    encoding_detected:          str   = "none"
    original_length:            int   = 0
    processed:                  bool  = False
    latency_ms:                 float = 0.0
    model_used:                 str   = ""
    fallback_used:              bool  = False
    error:                      str   = ""

    def classifier_input(self) -> str:
        """
        Build enriched text for DistilBERT/AraBERT classifiers.
        Puts business_context and domains on separate lines as signal boost,
        then appends verbatim sensitive segments.
        Falls back to clean_text when no segments available.
        """
        parts = []
        if self.business_context:
            parts.append(f"Context:{self.business_context}")
        if self.domains_detected:
            parts.append(f"Domains:{','.join(self.domains_detected)}")
        if self.important_segments:
            parts.append(" | ".join(self.important_segments[:5]))
        elif self.clean_text:
            parts.append(self.clean_text)
        return "\n".join(parts) if parts else self.clean_text

    def to_orchestrator_payload(self) -> dict:
        """Build payload dict for ai_orchestrator.classify()."""
        return {
            "text":                self.classifier_input(),
            "business_context":    self.business_context,
            "llm_domains":         self.domains_detected,
            "llm_sensitivity":     self.sensitivity_indicators,
            "contains_sensitive":  self.contains_sensitive_content,
            "language_hint":       self.language,
            "language_confidence": self.language_confidence,
        }


# ══════════════════════════════════════════════════════════════════════════
# LOCAL FALLBACK -- runs when Ollama is not available
# ══════════════════════════════════════════════════════════════════════════

class LocalFallbackPreprocessor:
    """
    Rule-based preprocessor -- no LLM needed.
    Handles common web content formats.
    Used automatically when Ollama is not running.
    """

    def process(self, raw_text: str) -> PreprocessResult:
        result                 = PreprocessResult()
        result.fallback_used   = True
        result.original_length = len(raw_text)
        result.model_used      = "local_fallback"
        text = raw_text.strip()

        # Strip HTTP headers -- keep body only
        if re.match(r"^(GET|POST|PUT|DELETE|PATCH|HEAD)\s", text):
            parts = re.split(r"\n\n|\r\n\r\n", text, maxsplit=1)
            text  = parts[1].strip() if len(parts) > 1 else text
            result.content_type = "api_payload"

        # URL decode
        if "%" in text and re.search(r"%[0-9A-Fa-f]{2}", text):
            try:
                from urllib.parse import unquote_plus
                text = unquote_plus(text)
                result.encoding_detected = "url_encoded"
            except Exception:
                pass

        # Extract JSON body
        stripped = text.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
                text   = self._flatten_json(parsed)
                result.content_type      = "api_payload"
                result.encoding_detected = "json"
            except Exception:
                pass

        # Strip HTML
        if "<" in text and ">" in text:
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"&[a-zA-Z]+;", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            result.content_type      = "web_form"
            result.encoding_detected = "html_entities"

        # Base64 decode
        b64 = re.search(r"(?:[A-Za-z0-9+/]{40,}={0,2})", text)
        if b64:
            try:
                import base64
                decoded = base64.b64decode(b64.group()).decode("utf-8", errors="ignore")
                if len(decoded) > 20:
                    text = text.replace(b64.group(), f"[decoded: {decoded[:300]}]")
                    result.encoding_detected = "base64"
            except Exception:
                pass

        # Language detection with confidence
        arabic = len(re.findall(r"[\u0600-\u06FF]", text))
        latin  = len(re.findall(r"[a-zA-Z]", text))
        total  = arabic + latin or 1
        ratio  = arabic / total

        if ratio > 0.7:
            result.language            = "ar"
            result.language_confidence = round(ratio, 2)
        elif ratio > 0.15:
            result.language            = "mixed"
            result.language_confidence = round(1.0 - abs(ratio - 0.5) * 2, 2)
        else:
            result.language            = "en"
            result.language_confidence = round(1.0 - ratio, 2)

        # Sensitivity signal detection
        signals, segments = [], []

        # English regex patterns
        en_checks = [
            (r"\b[23]\d{13}\b",                                        "egyptian_nid"),
            (r"\b\d{3}-\d{2}-\d{4}\b",                                "ssn"),
            (r"\b[A-Z]{1,2}\d{6,9}\b",                                "passport"),
            (r"\b4\d{3}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b",      "credit_card"),
            (r"\b5[1-5]\d{2}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b", "mastercard"),
            (r"(?i)cvv[\s:=]?\d{3,4}",                                 "cvv"),
            (r"\bEG\d{2}[A-Z0-9]{27}\b",                              "egyptian_iban"),
            (r"\b[A-Z]{2}\d{2}[A-Z0-9]{4,30}\b",                      "iban"),
            (r"\b01[0125]\d{8}\b",                                     "egyptian_phone"),
            (r"AKIA[0-9A-Z]{16}",                                      "aws_key"),
            (r"BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY",                      "private_key"),
            (r"(?i)(?:password|passwd|pwd)[=:\s]\S+",                  "credentials"),
            (r"(?i)(?:api[_\-]?key|secret|token)[=:\s]\S{10,}",       "api_key"),
        ]
        for pattern_str, signal in en_checks:
            m = re.search(pattern_str, text)
            if m:
                signals.append(signal)
                start = max(0, m.start() - 40)
                end   = min(len(text), m.end() + 40)
                segments.append(text[start:end].strip())

        # Arabic keyword detection
        ar_keywords = {
            "\u0627\u0644\u0631\u0642\u0645 \u0627\u0644\u0642\u0648\u0645\u064a": "egyptian_nid",
            "\u0631\u0642\u0645 \u0642\u0648\u0645\u064a": "egyptian_nid",
            "\u0628\u0637\u0627\u0642\u0629 \u0627\u0626\u062a\u0645\u0627\u0646": "credit_card",
            "\u0627\u0644\u0633\u062c\u0644 \u0627\u0644\u0637\u0628\u064a": "medical_record",
            "\u0627\u0644\u062a\u0634\u062e\u064a\u0635": "medical_diagnosis",
            "\u0648\u0635\u0641\u0629 \u0637\u0628\u064a\u0629": "prescription",
            "\u0643\u0644\u0645\u0629 \u0627\u0644\u0645\u0631\u0648\u0631": "credentials",
            "\u0627\u0644\u0628\u0627\u0633\u0648\u0648\u0631\u062f": "credentials",
            "\u0643\u0634\u0641 \u0627\u0644\u0645\u0631\u062a\u0628\u0627\u062a": "salary_data",
            "\u0631\u0627\u062a\u0628": "salary_data",
            "\u062a\u0633\u0631\u0628 \u0628\u064a\u0627\u0646\u0627\u062a": "data_leak",
            "\u0633\u0631\u064a \u0644\u0644\u063a\u0627\u064a\u0629": "highly_confidential",
            "\u0627\u0646\u062f\u0645\u0627\u062c": "merger_details",
            "\u0627\u0633\u062a\u062d\u0648\u0627\u0630": "acquisition",
        }
        for keyword, signal in ar_keywords.items():
            if keyword in text:
                if signal not in signals:
                    signals.append(signal)
                idx   = text.index(keyword)
                start = max(0, idx - 40)
                end   = min(len(text), idx + 80)
                seg   = text[start:end].strip()
                if seg not in segments:
                    segments.append(seg)

        result.sensitivity_indicators     = signals
        result.contains_sensitive_content = len(signals) > 0
        result.important_segments         = segments[:5]

        # Domain hints
        domain_map = {
            "medical_record":    "Medical",    "medical_diagnosis": "Medical",
            "prescription":      "Medical",    "salary_data":       "HR",
            "employee_records":  "HR",         "merger_details":    "Finance",
            "acquisition":       "Finance",    "credit_card":       "CustomerData",
            "mastercard":        "CustomerData","iban":             "CustomerData",
            "egyptian_iban":     "CustomerData","credentials":      "Security",
            "api_key":           "Security",   "private_key":       "Security",
            "aws_key":           "Security",   "data_leak":         "Security",
            "egyptian_nid":      "HR",         "ssn":               "HR",
        }
        result.domains_detected = list(dict.fromkeys(
            domain_map[s] for s in signals if s in domain_map
        ))
        result.clean_text       = text[:MAX_INPUT_CHARS]
        result.processed        = True
        result.business_context = (
            f"Content with: {', '.join(signals[:3])}"
            if signals else "General content"
        )
        return result

    def _flatten_json(self, obj, depth=0) -> str:
        if depth > 5:                      return ""
        if isinstance(obj, str):           return obj
        if isinstance(obj, (int, float)):  return str(obj)
        if isinstance(obj, dict):
            return " ".join(
                f"{k}: {self._flatten_json(v, depth+1)}"
                for k, v in obj.items()
            )
        if isinstance(obj, list):
            return " ".join(self._flatten_json(i, depth+1) for i in obj[:10])
        return ""


# ══════════════════════════════════════════════════════════════════════════
# GEMMA PREPROCESSOR
# ══════════════════════════════════════════════════════════════════════════

class GemmaPreprocessor:

    def __init__(self, model: str = GEMMA_MODEL):
        self.model     = model
        self._ready    = False
        self._fallback = LocalFallbackPreprocessor()
        self._check_ollama()

    def _check_ollama(self):
        try:
            import ollama
            client     = ollama.Client(host=OLLAMA_HOST)
            models     = client.list()
            model_list = []

            # Handle both old and new ollama API response formats
            if hasattr(models, "models"):
                for m in models.models:
                    model_list.append(m.model if hasattr(m, "model") else str(m))
            elif isinstance(models, dict):
                model_list = [m.get("name", "") for m in models.get("models", [])]

            if any(self.model in m for m in model_list):
                self._ready = True
                print(f"[LLMPreprocessor] {self.model} ready via Ollama.")
            else:
                print(f"[LLMPreprocessor] {self.model} not found.")
                print(f"  Available models: {model_list}")
                print(f"  Run: ollama pull {self.model}")
                print(f"  Falling back to local preprocessor.")

        except ImportError:
            print("[LLMPreprocessor] ollama package not installed.")
            print("  Run: pip install ollama")
            print("  Falling back to local preprocessor.")
        except Exception as e:
            print(f"[LLMPreprocessor] Ollama not running: {e}")
            print(f"  Run: ollama serve")
            print(f"  Falling back to local preprocessor.")

    def _build_prompt(self, raw_text: str) -> str:
        truncated = raw_text[:MAX_INPUT_CHARS]
        if len(raw_text) > MAX_INPUT_CHARS:
            truncated += f"\n[content truncated -- {len(raw_text) - MAX_INPUT_CHARS} more chars]"
        return f"Analyze this content for DLP classification:\n\n{truncated}"

    def _parse_response(self, text: str) -> dict:
        text = text.strip()
        # Remove markdown fences if present
        text = re.sub(r"```(?:json)?", "", text).strip()
        # Try direct JSON parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Find first JSON object
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
        logger.warning(f"[LLMPreprocessor] Cannot parse JSON response: {text[:150]}")
        return {}

    def process(self, raw_text: str) -> PreprocessResult:
        t_start = time.time()
        result  = PreprocessResult(original_length=len(raw_text))

        if not self._ready:
            r               = self._fallback.process(raw_text)
            r.fallback_used = True
            return r

        try:
            import ollama
            client = ollama.Client(host=OLLAMA_HOST)

            response = client.chat(
                model    = self.model,
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": self._build_prompt(raw_text)},
                ],
                format  = "json",   # FIX 1: forces valid JSON, fixes Arabic/English failures
                options = {
                    "temperature": 0.05,   # very low = consistent structured output
                    "top_p":       0.9,
                    "num_predict": 700,
                },
            )

            raw_resp = response["message"]["content"]
            parsed   = self._parse_response(raw_resp)

            if parsed:
                result.clean_text                 = parsed.get("clean_text", "")[:MAX_INPUT_CHARS]
                result.business_context           = parsed.get("business_context", "")
                result.language                   = parsed.get("language", "en")
                result.language_confidence        = float(parsed.get("language_confidence", 0.0))
                result.content_type               = parsed.get("content_type", "general")
                result.contains_sensitive_content = bool(parsed.get("contains_sensitive_content", False))
                result.sensitivity_indicators     = parsed.get("sensitivity_indicators", [])
                result.domains_detected           = parsed.get("domains_detected", [])
                result.important_segments         = [
                    s[:MAX_SEGMENT_CHARS]
                    for s in parsed.get("important_segments", [])[:5]
                ]
                result.encoding_detected          = parsed.get("encoding_detected", "none")
                result.processed                  = True
                result.model_used                 = self.model
            else:
                # LLM responded but JSON unparseable -- use fallback
                r               = self._fallback.process(raw_text)
                r.fallback_used = True
                r.error         = "LLM response unparseable -- used fallback"
                result          = r

        except Exception as e:
            logger.error(f"[LLMPreprocessor] Error calling Ollama: {e}")
            r               = self._fallback.process(raw_text)
            r.fallback_used = True
            r.error         = str(e)
            result          = r

        result.latency_ms = round((time.time() - t_start) * 1000, 1)
        return result


# ══════════════════════════════════════════════════════════════════════════
# PUBLIC INTERFACE -- used by ai_orchestrator.py
# ══════════════════════════════════════════════════════════════════════════

class LLMPreprocessor:
    """
    Main entry point. Called by ai_orchestrator.py before classifiers.

    Usage:
        prep    = LLMPreprocessor()
        result  = prep.process(raw_traffic_text)
        text    = result.classifier_input()      # send to DistilBERT/AraBERT
        payload = result.to_orchestrator_payload() # send to orchestrator

    Routing hint for orchestrator:
        if result.language == "ar"    and result.language_confidence >= 0.7: AraBERT only
        if result.language == "en"    and result.language_confidence >= 0.7: DistilBERT only
        if result.language == "mixed" or result.language_confidence < 0.7:   run both
    """

    def __init__(self, model: str = GEMMA_MODEL):
        self._processor = GemmaPreprocessor(model=model)

    def process(self, raw_text: str) -> PreprocessResult:
        return self._processor.process(raw_text)

    def process_email(self, subject: str = "", body: str = "",
                      attachment_text: str = "") -> PreprocessResult:
        parts = []
        if subject:         parts.append(f"Subject: {subject}")
        if body:            parts.append(body)
        if attachment_text: parts.append(f"Attachment:\n{attachment_text[:800]}")
        return self.process("\n\n".join(parts))

    def process_web(self, url: str = "", content: str = "",
                    file_text: str = "") -> PreprocessResult:
        parts = []
        if url:       parts.append(f"URL: {url}")
        if content:   parts.append(f"Content:\n{content}")
        if file_text: parts.append(f"File:\n{file_text[:800]}")
        return self.process("\n\n".join(parts))

    def is_using_llm(self) -> bool:
        return self._processor._ready

    def status(self) -> dict:
        return {
            "llm_available": self._processor._ready,
            "model":         self._processor.model,
            "ollama_host":   OLLAMA_HOST,
            "mode":          "gemma" if self._processor._ready else "local_fallback",
        }


# ══════════════════════════════════════════════════════════════════════════
# TEST
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    prep = LLMPreprocessor()
    print(f"\nStatus: {prep.status()}\n")
    print("=" * 65)

    tests = [
        ("Arabic internal slang",
         "\u062a\u0630\u0643\u064a\u0631: \u062f\u064a\u062f\u0644\u0627\u064a\u0646 "
         "\u0627\u0644\u062a\u0642\u0627\u0631\u064a\u0631 \u0627\u0644\u0634\u0647\u0631\u064a\u0629 "
         "\u0627\u0644\u062e\u0645\u064a\u0633 \u0627\u0644\u062c\u0627\u064a. "
         "\u0644\u0644\u062a\u064a\u0645 \u0627\u0644\u062f\u0627\u062e\u0644\u064a \u0628\u0633. "
         "\u0645\u0634 \u0644\u0644\u0634\u064a\u0631 \u0628\u0631\u0627"),

        ("Egyptian NID + medical",
         "\u0645\u0631\u064a\u0636 \u0645\u062d\u0645\u062f\u060c "
         "\u0627\u0644\u0631\u0642\u0645 \u0627\u0644\u0642\u0648\u0645\u064a: 29001011234567\u060c "
         "\u0627\u0644\u062a\u0634\u062e\u064a\u0635: \u0627\u0644\u0633\u0643\u0631\u060c "
         "\u0627\u0644\u062f\u0648\u0627\u0621: \u0645\u064a\u062a\u0641\u0648\u0631\u0645\u064a\u0646 500mg"),

        ("Financial confidential -- short English",
         "Net loss EGP 2.3M. Merger planned. Board unaware."),

        ("URL-encoded PII",
         "cardNumber=4111111111111111&cvv=123&expiryMonth=12&expiryYear=26&nationalId=29001011234567"),

        ("JSON credential leak",
         '{"db_host": "prod-db.internal", "db_user": "root", "db_password": "Admin@Pr0d"}'),

        ("Mixed Arabic/English internal",
         "Dear team, \u0627\u0644\u0640 VPN "
         "\u0647\u064a\u0643\u0648\u0646 \u0623\u0648\u0641\u0644\u0627\u064a\u0646 \u0627\u0644\u0633\u0628\u062a. "
         "Internal maintenance \u0641\u0642\u0637. "
         "\u0644\u0644\u062a\u064a\u0645 \u0627\u0644\u062f\u0627\u062e\u0644\u064a \u0628\u0633"),

        ("HTTP POST with sensitive file",
         "POST /upload HTTP/1.1\nHost: dropbox.com\n\n"
         "filename=customer_db_export.csv\n"
         "customer_id,national_id,credit_card,cvv -- 80000 records"),
    ]

    for desc, raw in tests:
        print(f"\n[{desc}]")
        r = prep.process(raw)
        print(f"  Model        : {r.model_used} (fallback={r.fallback_used})")
        print(f"  Language     : {r.language} (confidence={r.language_confidence})")
        print(f"  Content type : {r.content_type}")
        print(f"  Business ctx : {r.business_context}")
        print(f"  Sensitive    : {r.contains_sensitive_content}")
        print(f"  Indicators   : {r.sensitivity_indicators}")
        print(f"  Domains      : {r.domains_detected}")
        print(f"  Segments     : {r.important_segments[:2]}")
        print(f"  Classifier input:")
        for line in r.classifier_input().splitlines():
            print(f"    {line[:120]}")
        print(f"  Latency      : {r.latency_ms}ms")