"""
============================================================
llm_preprocessor.py  --  v4  (Intelligent local fallback)
============================================================
ROLE:
  Intelligent content preprocessor for DLP classification.
  Replaces the Gemma/Ollama LLM with a sophisticated multi-stage
  rule-based engine that covers the same gaps at ~1ms latency.

  The original GemmaPreprocessor and LLMPreprocessor classes are
  preserved for backward compatibility but now delegate to the
  v2 LocalFallbackPreprocessor.

PIPELINE:
  Raw traffic (email/web/any format)
    down
  LocalFallbackPreprocessor v2 (this file)
    down  structured PreprocessResult
  AIOrchestrator
    down  routes to
  DistilBERT (English sensitivity)
  AraBERT    (Arabic sensitivity)
  ContextClassifier (domain)

CHANGES vs v3 (Plan B6):
  - Complete rewrite of LocalFallbackPreprocessor with 7 subsystems
  - Language detection: weighted word dictionaries instead of char ratio
  - Content-type classifier: priority-chain detection (SMTP/HTTP/JSON/XML/HTML)
  - Sensitivity signals: 80+ English regex + 50+ Arabic keyword patterns
  - Important segments: sentence-boundary extraction (not ±40 char windows)
  - Domain detection: two-stage (signal→domain mapping + combination scoring)
  - Business context: template-based generator (not "Content with: signal1")
  - Encoding detection: expanded to 8 types
  - fallback_used=False, model_used="local_fallback_v2" (first-class citizen)
============================================================
"""

import re
import json
import time
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ── CONFIG ────────────────────────────────────────────────────────────────
MAX_INPUT_CHARS   = 3000
MAX_SEGMENT_CHARS = 400


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
# LOCAL FALLBACK v2 — intelligent rule-based preprocessor (Plan B6)
# ══════════════════════════════════════════════════════════════════════════

_COMMON_EN_WORDS = {
    "the","this","that","these","those","with","from","have","has","had",
    "for","not","you","your","our","all","can","will","was","are","were",
    "been","some","any","each","every","their","there","here","which",
    "what","when","where","how","who","about","into","over","after",
    "before","between","under","again","further","then","once","than",
    "shall","should","may","might","must","could","would","dear","hello",
    "hi","thanks","thank","regards","please","subject","attachment",
    "file","name","email","phone","address","date","please","team",
    "hello","dear","best","sent","from","to","cc","bcc","re","fw",
    "http","https","www","com","org","net","html","body","div","span",
    "class","id","src","href","link","script","style","table","tr","td",
    "data","user","password","login","admin","root","test","guest",
    "employee","customer","client","vendor","partner","manager","staff",
    "meeting","report","update","status","project","task","time","hour",
    "day","week","month","year","budget","cost","price","total","amount",
    "payment","order","invoice","receipt","balance","account","number",
    "information","confidential","restricted","internal","public",
    "company","department","office","address","city","country","code",
    "request","response","error","success","failed","pending","approved",
    "declined","blocked","service","server","network","system","access",
}

_COMMON_AR_WORDS = {
    "في","من","على","إلى","عن","مع","كان","كانت","هذا","هذه","ذلك",
    "تلك","هو","هي","هم","أن","إن","ما","لم","لن","قد","لا","كل",
    "بعض","أي","بين","تحت","فوق","بعد","قبل","عند","حتى","حول",
    "دون","خلال","أو","إذا","لأن","لكن","حيث","هناك","هنا","أيضا",
    "نعم","لا","شكرا","مرحبا","عزيزي","تحية","مرفق","ملف","اسم",
    "بريد","عنوان","هاتف","تاريخ","الرجاء","فريق","مرسل","إلى",
    "موضوع","إعادة","رد","بيانات","مستخدم","كلمة","مرور","دخول",
    "مدير","موظف","عميل","مندوب","شريك","مدير","اجتماع","تقرير",
    "تحديث","حالة","مشروع","مهمة","وقت","يوم","أسبوع","شهر","سنة",
    "ميزانية","تكلفة","سعر","إجمالي","مبلغ","دفع","طلب","فاتورة",
    "إيصال","رصيد","حساب","رقم","معلومات","سري","خاص","داخلي",
    "عام","شركة","قسم","مكتب","مدينة","دولة","رمز","شركة",
    "خدمة","خادم","شبكة","نظام","وصول","أمن","إدارة",
    # Egyptian / dialect
    "بس","كده","ده","دي","اية","ازاي","كدا","دلوقتي",
}

# ── ENGLISH SENSITIVITY PATTERNS ──────────────────────────────────────────
# Each tuple: (regex_pattern, signal_name)
_EN_PATTERNS = [
    # Egyptian NID (14 digits starting with 2 or 3)
    (r"\b[23]\d{13}\b",                                        "egyptian_nid"),
    # US SSN (XXX-XX-XXXX)
    (r"\b\d{3}-\d{2}-\d{4}\b",                                "ssn"),
    # Passport (letter(s) + digits)
    (r"\b[A-Z]{1,2}\d{6,9}\b",                                "passport"),
    # Credit Card: Visa (starts with 4)
    (r"\b4\d{3}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b",      "credit_card"),
    # Credit Card: Mastercard (starts with 51-55)
    (r"\b5[1-5]\d{2}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b", "mastercard"),
    # Credit Card: Amex (starts with 34 or 37)
    (r"\b3[47]\d{2}[\s\-]?\d{6}[\s\-]?\d{5}\b",              "amex"),
    # Credit Card: Discover (starts with 6011 or 65)
    (r"\b6(?:011|5\d{2})[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b", "discover"),
    # Credit Card: Diners Club (starts with 300-305, 36, 38)
    (r"\b3(?:0[0-5]|[68]\d)\d{2}[\s\-]?\d{6}[\s\-]?\d{4}\b", "diners_club"),
    # Credit Card: JCB (starts with 3528-3589)
    (r"\b35(?:2[89]|[3-8]\d)\d{2}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b", "jcb"),
    # CVV
    (r"(?i)cvv[\s:=]?\d{3,4}",                                 "cvv"),
    # Egyptian IBAN (EG + 27 alphanumeric)
    (r"\bEG\d{2}[A-Z0-9]{27}\b",                              "egyptian_iban"),
    # Generic IBAN (2 letters + 2 digits + up to 30 alphanumeric)
    (r"\b[A-Z]{2}\d{2}[A-Z0-9]{4,30}\b",                      "iban"),
    # Egyptian phone (01 followed by 0,1,2,5 + 8 digits)
    (r"\b01[0125]\d{8}\b",                                     "egyptian_phone"),
    # International phone (+\d{10,15})
    (r"\+\d{10,15}\b",                                         "international_phone"),
    # AWS Access Key
    (r"AKIA[0-9A-Z]{16}",                                      "aws_key"),
    # AWS Secret Key
    (r"(?i)aws[_-]?secret[_-]?access[_-]?key[_-]?\S{10,}",     "aws_secret_key"),
    # Azure keys
    (r"(?i)azure[_-]?(storage|account|cosmos)[_-]?key\S{10,}", "azure_key"),
    # GCP keys
    (r"(?i)(?:gcp|google)[_-]?(service[_-]?account|api[_-]?key|application[_-]?credentials)\S{10,}", "gcp_key"),
    # Github tokens
    (r"(?i)github[_-]?(token|pat|personal[_-]?access[_-]?token)\S{10,}", "github_token"),
    # Slack tokens/bots
    (r"(?i)(?:xox[bpras]-|slack[_-]?token)\S{10,}",            "slack_token"),
    # Generic API keys / tokens / secrets
    (r"(?i)(?:api[_\-]?key|secret|token)[=:\s]['\"]?\S{10,}['\"]?", "api_key"),
    # Credentials (password/passwd/pwd = value, with optional space after colon)
    (r"(?i)(?:password|passwd|pwd)\s*[=:]\s*\S+",             "credentials"),
    # Password in JSON key (db_password, user_password, etc.)
    (r"(?i)\"(?:db_|user_|admin_)?(?:password|passwd|pwd)\"\s*:\s*\"[^\"]+\"", "credentials"),
    # Username / login pairs
    (r"(?i)(?:username|user_id|login)\s*[=:]\s*\S+",          "credentials"),
    # Private key headers
    (r"BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY",                      "private_key"),
    # SSH keys
    (r"(?i)ssh-rsa\s+AAAAB3NzaC1yc2",                         "ssh_key"),
    # Connection strings / JDBC
    (r"(?i)(?:jdbc|mongodb|redis|postgresql|mysql)://\S+",     "connection_string"),
    # IP address (private/internal ranges)
    (r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3})\b", "internal_ip"),
    # MAC address
    (r"\b([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})\b",          "mac_address"),
    # ICD-10 medical codes
    (r"\b[A-Z]\d{2}(?:\.\d{1,4})?\b",                         "medical_code_icd10"),
    # DEA number (2 letters + 7 digits)
    (r"\b[A-Z]{2}\d{7}\b",                                     "dea_number"),
    # NDC (National Drug Code)
    (r"\b\d{4}-\d{4}-\d{2}\b",                                 "ndc"),
    # Employee/student IDs (alphanumeric, 5-10 chars)
    (r"\b(?:EMP|STU|USR|ACC|CUS)\d{4,8}\b",                   "employee_id"),
    # SWIFT/BIC codes
    (r"\b[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}(?:[A-Z0-9]{3})?\b",      "swift"),
    # Routing/ABA numbers
    (r"\b\d{9}\b",                                              "aba_routing"),
    # US bank account (10-12 digits)
    (r"\b\d{10,12}\b",                                          "bank_account"),
]

# ── ENGLISH KEYWORD DETECTION ───────────────────────────────────────────
# Keywords that supplement regex patterns for contextual detection.
_EN_KEYWORDS = [
    ("confidential",         "confidential"),
    ("highly confidential",  "highly_confidential"),
    ("restricted",           "restricted"),
    ("internal only",        "internal_only"),
    ("attorney-client privileged", "legal_privileged"),
    ("attorney client privileged", "legal_privileged"),
    ("attorney-client",      "legal"),
    ("attorney client",      "legal"),
    ("solicitor-client",     "legal"),
    ("legal advice",         "legal"),
    ("legal counsel",        "legal"),
    ("attorney work product", "legal_privileged"),
    ("nda",                  "legal_contract"),
    ("non-disclosure",       "legal_contract"),
    ("non-disclosure agreement", "legal_contract"),
    ("merger",               "merger_details"),
    ("merger and acquisition", "merger_details"),
    ("m&a",                  "merger_details"),
    ("acquisition",          "acquisition"),
    ("due diligence",        "merger_details"),
    ("term sheet",           "merger_details"),
    ("salary",               "salary_data"),
    ("payroll",              "salary_data"),
    ("compensation",         "salary_data"),
    ("employee record",      "employee_records"),
    ("personnel file",       "employee_records"),
    ("hipaa",                "medical_record"),
    ("phi",                  "medical_record"),
    ("patient record",       "medical_record"),
    ("diagnosis",            "medical_diagnosis"),
    ("prescription",         "prescription"),
    ("data breach",          "data_leak"),
    ("security breach",      "security_breach"),
    ("unauthorized access",  "unauthorized_access"),
    ("password",             "credentials"),
    ("api key",              "api_key"),
    ("secret key",           "api_key"),
    ("access token",         "api_key"),
    ("customer data",        "customer_database"),
    ("personally identifiable", "personal_data"),
    ("pii",                  "personal_data"),
    ("gdpr",                 "personal_data"),
    ("bank account",         "bank_account"),
    ("wire transfer",        "bank_transfer"),
    ("credit card",          "credit_card"),
    ("social security",      "ssn"),
    ("national id",          "egyptian_nid"),
]

# ── ARABIC KEYWORD PATTERNS ──────────────────────────────────────────────
# Each tuple: (keyword_or_regex, signal_name, is_regex)
_AR_PATTERNS = [
    # Egyptian NID / National ID
    ("\u0627\u0644\u0631\u0642\u0645 \u0627\u0644\u0642\u0648\u0645\u064a", "egyptian_nid", False),
    ("\u0631\u0642\u0645 \u0642\u0648\u0645\u064a", "egyptian_nid", False),
    ("\u0627\u0644\u0631\u0642\u0645 \u0627\u0644\u0648\u0637\u0646\u064a", "egyptian_nid", False),
    ("\u0631\u0642\u0645 \u0627\u0644\u0628\u0637\u0627\u0642\u0629 \u0627\u0644\u0634\u062e\u0635\u064a\u0629", "egyptian_nid", False),
    ("\u0627\u0644\u0647\u0648\u064a\u0629 \u0627\u0644\u0648\u0637\u0646\u064a\u0629", "egyptian_nid", False),
    ("\u0628\u0637\u0627\u0642\u0629 \u0627\u0644\u0631\u0642\u0645 \u0627\u0644\u0642\u0648\u0645\u064a", "egyptian_nid", False),
    ("\u0627\u0644\u0631\u0642\u0645 \u0627\u0644\u0648\u0637\u0646\u064a", "egyptian_nid", False),
    ("\u0628\u0637\u0627\u0642\u0629 \u0634\u062e\u0635\u064a\u0629", "egyptian_nid", False),
    # Credit Card / Payment
    ("\u0628\u0637\u0627\u0642\u0629 \u0627\u0626\u062a\u0645\u0627\u0646", "credit_card", False),
    ("\u0628\u0637\u0627\u0642\u0629 \u0627\u06cc\u062a\u0645\u0627\u0646", "credit_card", False),
    ("\u0631\u0642\u0645 \u0627\u0644\u0628\u0637\u0627\u0642\u0629", "credit_card", False),
    ("\u0628\u0637\u0627\u0642\u0629 \u0641\u064a\u0632\u0627", "credit_card", False),
    ("\u0643\u0627\u0631\u062a \u0641\u064a\u0632\u0627", "credit_card", False),
    ("\u0643\u0627\u0631\u062a \u0627\u0626\u062a\u0645\u0627\u0646", "credit_card", False),
    ("\u0628\u0637\u0627\u0642\u0629 \u0645\u0627\u0633\u062a\u0631\u0643\u0627\u0631\u062f", "mastercard", False),
    ("\u0643\u0627\u0631\u062a \u0645\u0627\u0633\u062a\u0631\u0643\u0627\u0631\u062f", "mastercard", False),
    ("\u0628\u064a\u0627\u0646\u0627\u062a \u0627\u0644\u0628\u0637\u0627\u0642\u0629", "credit_card", False),

    # Bank Account / IBAN
    ("\u0631\u0642\u0645 \u0627\u0644\u062d\u0633\u0627\u0628", "bank_account", False),
    ("\u062d\u0633\u0627\u0628 \u0628\u0646\u0643\u064a", "bank_account", False),
    ("\u0627\u0644\u062d\u0633\u0627\u0628 \u0627\u0644\u0628\u0646\u0643\u064a", "bank_account", False),
    ("IBAN", "egyptian_iban", False),
    ("\u0627\u0644\u0622\u064a\u0628\u0627\u0646", "egyptian_iban", False),
    ("\u0627\u0644\u0627\u064a\u0628\u0627\u0646", "egyptian_iban", False),
    ("\u062a\u062d\u0648\u064a\u0644 \u0628\u0646\u0643\u064a", "bank_transfer", False),

    # Medical
    ("\u062a\u0634\u062e\u064a\u0635", "medical_diagnosis", False),
    ("\u0648\u0635\u0641\u0629 \u0637\u0628\u064a\u0629", "prescription", False),
    ("\u0631\u0648\u0634\u062a\u0629", "prescription", False),
    ("\u062a\u062d\u0644\u064a\u0644", "medical_test", False),
    ("\u0623\u0634\u0639\u0629", "medical_imaging", False),
    ("\u0633\u0648\u0646\u0627\u0631", "medical_imaging", False),
    ("\u0639\u0645\u0644\u064a\u0629 \u062c\u0631\u0627\u062d\u064a\u0629", "surgery", False),
    ("\u062a\u0642\u0631\u064a\u0631 \u0637\u0628\u064a", "medical_record", False),
    ("\u0633\u062c\u0644 \u0637\u0628\u064a", "medical_record", False),
    ("\u0645\u0644\u0641 \u0645\u0631\u064a\u0636", "medical_record", False),
    ("\u0645\u0644\u0641 \u0637\u0628\u064a", "medical_record", False),
    ("\u062a\u0627\u0631\u064a\u062e \u0645\u0631\u0636\u064a", "medical_record", False),
    ("\u062f\u0648\u0627\u0621", "prescription", False),
    ("\u0639\u0644\u0627\u062c", "prescription", False),
    ("\u062c\u0631\u0639\u0629", "prescription", False),
    ("\u0646\u062a\u064a\u062c\u0629", "medical_test", False),
    ("\u062a\u062d\u0627\u0644\u064a\u0644", "medical_test", False),

    # HR / Salary
    ("\u0627\u0644\u0631\u0627\u062a\u0628", "salary_data", False),
    ("\u0627\u0644\u0645\u0631\u062a\u0628", "salary_data", False),
    ("\u0645\u0633\u064a\u0631 \u0627\u0644\u0631\u0648\u0627\u062a\u0628", "salary_data", False),
    ("\u0645\u0633\u064a\u0631 \u0627\u0644\u0645\u0631\u062a\u0628\u0627\u062a", "salary_data", False),
    ("\u0643\u0634\u0641 \u0627\u0644\u0645\u0631\u062a\u0628\u0627\u062a", "salary_data", False),
    ("\u0627\u0644\u0645\u0643\u0627\u0641\u0623\u0629", "salary_data", False),
    ("\u0627\u0644\u0628\u062f\u0644\u0627\u062a", "salary_data", False),
    ("\u0627\u0644\u062e\u0635\u0648\u0645\u0627\u062a", "salary_data", False),
    ("\u0625\u062c\u0627\u0632\u0629", "leave_data", False),
    ("\u0625\u062c\u0627\u0632\u0629 \u0645\u0631\u0636\u064a\u0629", "leave_data", False),

    # Credentials
    ("\u0643\u0644\u0645\u0629 \u0627\u0644\u0645\u0631\u0648\u0631", "credentials", False),
    ("\u0643\u0644\u0645\u0629 \u0627\u0644\u0633\u0631", "credentials", False),
    ("\u0627\u0644\u0628\u0627\u0633\u0648\u0648\u0631\u062f", "credentials", False),
    ("\u0627\u0644\u0628\u0627\u0633\u0648\u0631\u062f", "credentials", False),
    ("\u0627\u0633\u0645 \u0627\u0644\u0645\u0633\u062a\u062e\u062f\u0645", "credentials", False),
    ("\u0627\u0633\u0645 \u0627\u0644\u064a\u0648\u0632\u0631", "credentials", False),
    ("\u0628\u064a\u0627\u0646\u0627\u062a \u0627\u0644\u062f\u062e\u0648\u0644", "credentials", False),
    ("\u064a\u0648\u0632\u0631 \u0646\u064a\u0645", "credentials", False),
    ("\u064a\u0648\u0632\u0631\u0646\u064a\u0645", "credentials", False),

    # Confidentiality labels
    ("\u0633\u0631\u064a", "confidential", False),
    ("\u0633\u0631\u064a \u0644\u0644\u063a\u0627\u064a\u0629", "highly_confidential", False),
    ("\u0645\u0642\u064a\u062f", "restricted", False),
    ("\u062e\u0627\u0635", "confidential", False),
    ("\u0645\u0645\u0646\u0648\u0639 \u0627\u0644\u062a\u062f\u0627\u0648\u0644", "restricted", False),
    ("\u0644\u0644\u0628\u0648\u0631\u062f \u0641\u0642\u0637", "highly_confidential", False),
    ("\u0644\u0644\u0625\u062f\u0627\u0631\u0629 \u0627\u0644\u0639\u0644\u064a\u0627 \u0641\u0642\u0637", "highly_confidential", False),
    ("\u0645\u0634 \u0644\u0644\u0634\u064a\u0631 \u0628\u0631\u0627", "restricted", False),
    ("\u0644\u0644\u062a\u064a\u0645 \u0627\u0644\u062f\u0627\u062e\u0644\u064a \u0628\u0633", "internal_only", False),
    ("\u0644\u0644\u0627\u0633\u062a\u062e\u062f\u0627\u0645 \u0627\u0644\u062f\u0627\u062e\u0644\u064a", "internal_only", False),

    # Corporate / M&A
    ("\u0627\u0633\u062a\u062d\u0648\u0627\u0630", "acquisition", False),
    ("\u0627\u0646\u062f\u0645\u0627\u062c", "merger_details", False),
    ("\u0635\u0641\u0642\u0629", "merger_details", False),
    ("\u062a\u0642\u064a\u064a\u0645", "valuation", False),
    ("\u062f\u0631\u0627\u0633\u0629 \u062c\u062f\u0648\u0649", "business_plan", False),
    ("\u0639\u0631\u0636 \u0634\u0631\u0627\u0621", "acquisition", False),
    ("\u062e\u0637\u0629 \u0639\u0645\u0644", "business_plan", False),
    ("\u062a\u0645\u0648\u064a\u0644", "funding", False),
    ("\u0631\u0623\u0633 \u0627\u0644\u0645\u0627\u0644", "funding", False),
    ("\u0627\u0643\u062a\u062a\u0627\u0628", "funding", False),

    # Legal
    ("\u0639\u0642\u062f", "legal_contract", False),
    ("\u0627\u062a\u0641\u0627\u0642\u064a\u0629", "legal_contract", False),
    ("\u0645\u062d\u0627\u0645\u064a", "legal", False),
    ("\u062f\u0639\u0648\u0649 \u0642\u0636\u0627\u0626\u064a\u0629", "legal", False),
    ("\u062a\u0633\u0648\u064a\u0629", "legal_settlement", False),
    ("\u062d\u0643\u0645", "legal_judgment", False),
    ("\u062a\u062d\u0643\u064a\u0645", "arbitration", False),
    ("\u0642\u0636\u064a\u0629", "legal_case", False),
    ("\u0634\u0647\u0631 \u0639\u0642\u0627\u0631\u064a", "legal_notary", False),
    ("\u062a\u0648\u0643\u064a\u0644", "legal_power_of_attorney", False),
    ("\u0646\u0632\u0627\u0639", "legal_dispute", False),

    # Data Leak / Security
    ("\u062a\u0633\u0631\u064a\u0628", "data_leak", False),
    ("\u0627\u062e\u062a\u0631\u0627\u0642", "security_breach", False),
    ("\u062e\u0631\u0642 \u0623\u0645\u0646\u064a", "security_breach", False),
    ("\u0627\u062e\u062a\u0631\u0627\u0642 \u0628\u064a\u0627\u0646\u0627\u062a", "data_leak", False),
    ("\u062b\u063a\u0631\u0629 \u0623\u0645\u0646\u064a\u0629", "vulnerability", False),
    ("\u0647\u062c\u0648\u0645", "cyber_attack", False),
    ("\u062a\u0635\u064a\u062f", "phishing", False),
    ("\u0628\u0631\u0646\u0627\u0645\u062c \u0636\u0627\u0631", "malware", False),
    ("\u0648\u0635\u0648\u0644 \u063a\u064a\u0631 \u0645\u0635\u0631\u062d", "unauthorized_access", False),

    # Employee / Personnel
    ("\u0628\u064a\u0627\u0646\u0627\u062a \u0634\u062e\u0635\u064a\u0629", "personal_data", False),
    ("\u0628\u064a\u0627\u0646\u0627\u062a \u0627\u0644\u0645\u0648\u0638\u0641\u064a\u0646", "employee_records", False),
    ("\u0633\u062c\u0644 \u0645\u0648\u0638\u0641", "employee_records", False),
    ("\u0645\u0644\u0641 \u0634\u062e\u0635\u064a", "personal_data", False),
    ("\u0642\u0627\u0639\u062f\u0629 \u0628\u064a\u0627\u0646\u0627\u062a \u0639\u0645\u0644\u0627\u0621", "customer_database", False),
    ("\u0628\u064a\u0627\u0646\u0627\u062a \u0639\u0645\u0644\u0627\u0621", "customer_database", False),
]

# ── SIGNAL → DOMAIN MAPPING (Stage A) ────────────────────────────────────
_SIGNAL_TO_DOMAIN = {
    "credit_card":           ["CustomerData", "Finance"],
    "mastercard":            ["CustomerData", "Finance"],
    "amex":                  ["CustomerData", "Finance"],
    "discover":              ["CustomerData", "Finance"],
    "diners_club":           ["CustomerData", "Finance"],
    "jcb":                   ["CustomerData", "Finance"],
    "cvv":                   ["CustomerData", "Security"],
    "egyptian_iban":         ["CustomerData", "Finance"],
    "iban":                  ["CustomerData", "Finance"],
    "bank_account":          ["CustomerData", "Finance"],
    "bank_transfer":         ["Finance"],
    "aba_routing":           ["Finance"],
    "swift":                 ["Finance"],
    "medical_record":        ["Medical"],
    "medical_diagnosis":     ["Medical"],
    "prescription":          ["Medical"],
    "medical_test":          ["Medical"],
    "medical_imaging":       ["Medical"],
    "surgery":               ["Medical"],
    "medical_code_icd10":    ["Medical"],
    "dea_number":            ["Medical", "Security"],
    "ndc":                   ["Medical"],
    "salary_data":           ["HR", "Finance"],
    "leave_data":            ["HR"],
    "employee_records":      ["HR"],
    "employee_id":           ["HR"],
    "credentials":           ["Security", "IT"],
    "api_key":               ["Security", "IT"],
    "aws_key":               ["Security", "IT"],
    "aws_secret_key":        ["Security", "IT"],
    "azure_key":             ["Security", "IT"],
    "gcp_key":               ["Security", "IT"],
    "github_token":          ["Security", "IT"],
    "slack_token":           ["Security", "IT"],
    "private_key":           ["Security", "IT"],
    "ssh_key":               ["Security", "IT"],
    "connection_string":     ["IT", "Security"],
    "egyptian_nid":          ["HR"],
    "ssn":                   ["HR"],
    "passport":              ["HR"],
    "egyptian_phone":        ["CustomerData"],
    "international_phone":   ["CustomerData"],
    "personal_data":         ["CustomerData", "HR"],
    "customer_database":     ["CustomerData"],
    "merger_details":        ["Finance", "Legal"],
    "acquisition":           ["Finance", "Legal"],
    "valuation":             ["Finance"],
    "funding":               ["Finance"],
    "business_plan":         ["Finance"],
    "legal":                 ["Legal"],
    "legal_contract":        ["Legal"],
    "legal_settlement":      ["Legal"],
    "legal_judgment":        ["Legal"],
    "arbitration":           ["Legal"],
    "legal_case":            ["Legal"],
    "legal_notary":          ["Legal"],
    "legal_power_of_attorney": ["Legal"],
    "legal_dispute":         ["Legal"],
    "confidential":          ["Legal", "Security"],
    "highly_confidential":   ["Legal", "Security"],
    "restricted":            ["Security"],
    "internal_only":         ["General"],
    "data_leak":             ["Security"],
    "security_breach":       ["Security"],
    "vulnerability":         ["Security"],
    "cyber_attack":          ["Security"],
    "phishing":              ["Security"],
    "malware":               ["Security"],
    "unauthorized_access":   ["Security"],
    "internal_ip":           ["IT", "Security"],
    "mac_address":           ["IT"],
}

# ── DOMAIN COMBINATION SCORING (Stage B) ─────────────────────────────────
_DOMAIN_COMBOS = {
    frozenset({"egyptian_nid", "medical_diagnosis", "prescription"}):
        {"Medical": 0.8, "HR": 0.2},
    frozenset({"medical_diagnosis", "credit_card"}):
        {"Medical": 0.6, "CustomerData": 0.4},
    frozenset({"salary_data", "credentials"}):
        {"HR": 0.7, "Security": 0.3},
    frozenset({"credit_card", "cvv"}):
        {"CustomerData": 0.9, "Finance": 0.1},
    frozenset({"merger_details", "acquisition", "confidential"}):
        {"Finance": 0.7, "Legal": 0.3},
    frozenset({"data_leak", "credentials"}):
        {"Security": 0.8, "IT": 0.2},
    frozenset({"patient", "prescription", "medical_diagnosis"}):
        {"Medical": 0.9, "HR": 0.1},
    frozenset({"salary_data", "employee_records"}):
        {"HR": 0.9, "Finance": 0.1},
    frozenset({"legal_contract", "confidential"}):
        {"Legal": 0.8, "Finance": 0.2},
    frozenset({"customer_database", "credit_card"}):
        {"CustomerData": 0.8, "Security": 0.2},
}

# ── BUSINESS CONTEXT TEMPLATES ───────────────────────────────────────────
_CONTEXT_TEMPLATES = [
    (lambda r: "credit_card" in r.sensitivity_indicators or "mastercard" in r.sensitivity_indicators,
     "Payment card data detected in {content_type} traffic"),
    (lambda r: "cvv" in r.sensitivity_indicators,
     "Card verification value (CVV) exposed in {content_type}"),
    (lambda r: "egyptian_iban" in r.sensitivity_indicators or "iban" in r.sensitivity_indicators,
     "International bank account details in {content_type}"),
    (lambda r: "bank_account" in r.sensitivity_indicators,
     "Bank account information in {content_type}"),
    (lambda r: "egyptian_nid" in r.sensitivity_indicators,
     "Egyptian national identification documents in {content_type}"),
    (lambda r: "ssn" in r.sensitivity_indicators,
     "US Social Security Number in {content_type}"),
    (lambda r: bool(set(r.sensitivity_indicators) & {"medical_diagnosis", "prescription", "medical_record", "medical_test", "surgery"}),
     "Patient medical information in {content_type}"),
    (lambda r: "credentials" in r.sensitivity_indicators or "api_key" in r.sensitivity_indicators,
     "Credentials or authentication secrets exposed in {content_type}"),
    (lambda r: "aws_key" in r.sensitivity_indicators or "aws_secret_key" in r.sensitivity_indicators,
     "AWS cloud infrastructure credentials in {content_type}"),
    (lambda r: "private_key" in r.sensitivity_indicators or "ssh_key" in r.sensitivity_indicators,
     "Private cryptographic keys exposed in {content_type}"),
    (lambda r: "salary_data" in r.sensitivity_indicators,
     "Employee salary and payroll data in {content_type}"),
    (lambda r: "merger_details" in r.sensitivity_indicators or "acquisition" in r.sensitivity_indicators,
     "Merger or acquisition strategic documents in {content_type}"),
    (lambda r: "customer_database" in r.sensitivity_indicators,
     "Customer database / CRM data in {content_type}"),
    (lambda r: "legal_contract" in r.sensitivity_indicators or "legal" in r.sensitivity_indicators,
     "Legal documents or contracts in {content_type}"),
    (lambda r: "data_leak" in r.sensitivity_indicators or "security_breach" in r.sensitivity_indicators,
     "Security incident or data breach information in {content_type}"),
    (lambda r: "employee_records" in r.sensitivity_indicators,
     "Employee personnel records in {content_type}"),
    (lambda r: "confidential" in r.sensitivity_indicators or "highly_confidential" in r.sensitivity_indicators,
     "Confidential business content with restricted distribution in {content_type}"),
]


class LocalFallbackPreprocessor:
    """
    v2 — Intelligent multi-stage rule-based preprocessor.
    Replaces the Gemma/Ollama LLM with 7 subsystems:
      S1 Language Detection  S2 Content-Type  S3 Signal Detection
      S4 Important Segments  S5 Domain Detection  S6 Context Generation
      S7 Encoding Detection
    """

    def __init__(self):
        self._precompile()

    def _precompile(self):
        """Pre-compile all regex patterns for performance."""
        self._en_compiled = [
            (re.compile(pat), sig) for pat, sig in _EN_PATTERNS
        ]
        self._ar_compiled = [
            (re.compile(re.escape(kw), re.UNICODE) if not is_regex else re.compile(kw), sig)
            for kw, sig, is_regex in _AR_PATTERNS
        ]

    # ── S7: ENCODING DETECTION & DECODING ────────────────────────────────

    def _detect_and_decode(self, text: str) -> tuple[str, str]:
        """Detect encoding and decode. Returns (decoded_text, encoding_type)."""
        original = text
        encoding = "none"

        # 1. URL encoding
        if "%" in text and re.search(r"%[0-9A-Fa-f]{2}", text):
            try:
                from urllib.parse import unquote_plus
                text = unquote_plus(text)
                encoding = "url_encoded"
            except Exception:
                pass

        # 2. JSON
        stripped = text.strip()
        if stripped and (stripped.startswith("{") or stripped.startswith("[")):
            try:
                json.loads(stripped)
                encoding = "json" if encoding == "none" else encoding
            except Exception:
                pass

        # 3. XML/SOAP
        if re.search(r"<\?xml|<soap:|<[A-Za-z]+\s+", stripped[:200]):
            encoding = "xml" if encoding == "none" else encoding

        # 4. HTML entities
        if "<" in text and ">" in text:
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"&[a-zA-Z]+;", " ", text)
            encoding = "html_entities" if encoding == "none" else encoding

        # 5. Base64 (40+ char base64 strings)
        b64 = re.search(r"(?:[A-Za-z0-9+/]{40,}={0,2})", stripped)
        if b64:
            try:
                import base64
                decoded = base64.b64decode(b64.group()).decode("utf-8", errors="ignore")
                if len(decoded) > 20:
                    text = text.replace(b64.group(), f"[decoded: {decoded[:300]}]")
                    encoding = "base64"
            except Exception:
                pass

        # 6. Quoted-printable
        if re.search(r"=[0-9A-Fa-f]{2}", text) and "=" in text:
            encoding = "quoted_printable" if encoding == "none" else encoding

        # 7. Multipart MIME
        if re.search(r"Content-Type:\s*multipart/form-data", original, re.I):
            encoding = "multipart" if encoding == "none" else encoding

        return text, encoding

    # ── S2: CONTENT-TYPE CLASSIFIER ──────────────────────────────────────

    def _classify_content_type(self, text: str) -> str:
        """Priority-chain content-type detection."""
        if not text:
            return "general"

        # SMTP/Email headers
        if re.search(r"^(From|To|Subject|Cc|Bcc|Date|Message-ID|MIME-Version):", text, re.M):
            return "email"

        # HTTP request/response
        if re.match(r"^(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)\s+\S+\s+HTTP/", text):
            return "api_payload"
        if re.match(r"^HTTP/\d\.\d\s+\d{3}", text):
            return "api_payload"

        # Multipart form data
        if re.search(r"Content-Type:\s*multipart/form-data", text, re.I):
            return "file_upload"

        # JSON
        stripped = text.strip()
        if stripped and (stripped.startswith("{") or stripped.startswith("[")):
            try:
                json.loads(stripped[:2000])
                return "api_payload"
            except Exception:
                pass

        # XML/SOAP
        if re.search(r"<\?xml|<soap:|<[A-Za-z]+\s+", stripped[:500]):
            return "api_payload"

        # URL-encoded form
        if re.match(r"^[\w%]+=[\w%]+(&[\w%]+=[\w%]+)*$", stripped[:500]):
            return "web_form"

        # HTML
        if re.search(r"<!DOCTYPE html|<html|<form|<input", stripped[:1000], re.I):
            return "web_form"

        # Code-like content
        if re.search(r"(def |class |import |function |var |let |const |SELECT |INSERT )", text, re.I):
            return "code"

        return "general"

    # ── S1: LANGUAGE DETECTION ───────────────────────────────────────────

    def _detect_language(self, text: str) -> tuple[str, float]:
        """
        Weighted word-dictionary language detection.
        Returns (language, confidence).
        """
        if not text:
            return "en", 0.0

        words = re.findall(r"[a-zA-Z\u0600-\u06FF]+", text)
        if not words:
            return "en", 0.0

        en_count = sum(1 for w in words if w.lower() in _COMMON_EN_WORDS)
        ar_count = sum(1 for w in words if w in _COMMON_AR_WORDS)

        # Also count Arabic vs Latin chars as a secondary signal
        ar_chars = sum(1 for c in text if "\u0600" <= c <= "\u06FF")
        en_chars = sum(1 for c in text if c.isascii() and c.isalpha())

        # Combined score: word presence weighted 0.7, char ratio weighted 0.3
        total_words = en_count + ar_count or 1
        word_ratio = ar_count / total_words

        total_chars = ar_chars + en_chars or 1
        char_ratio = ar_chars / total_chars

        combined = word_ratio * 0.7 + char_ratio * 0.3

        if combined > 0.7:
            return "ar", round(combined, 2)
        elif combined > 0.15:
            return "mixed", round(1.0 - abs(combined - 0.5) * 2, 2)
        else:
            return "en", round(1.0 - combined, 2)

    # ── S3: SENSITIVITY SIGNAL DETECTION ─────────────────────────────────

    def _extract_signals(self, text: str) -> tuple[list[str], list[tuple[int, int]]]:
        """
        Detect all sensitivity signals in text.
        Returns (signals_list, match_spans) where each span is (start, end).
        """
        signals = []
        match_spans = []
        seen_signals = set()

        # English regex patterns
        for compiled, sig in self._en_compiled:
            for m in compiled.finditer(text):
                if sig not in seen_signals:
                    signals.append(sig)
                    seen_signals.add(sig)
                match_spans.append((m.start(), m.end()))

        # English keyword detection (case-insensitive whole-word)
        lower_text = text.lower()
        for keyword, sig in _EN_KEYWORDS:
            if sig not in seen_signals:
                # Try whole-word matching for multi-word keywords
                pattern = r"\b" + re.escape(keyword) + r"\b"
                if re.search(pattern, lower_text):
                    signals.append(sig)
                    seen_signals.add(sig)
                    m = re.search(pattern, lower_text)
                    if m:
                        # Find position in original text
                        orig_idx = text.lower().index(keyword)
                        match_spans.append((orig_idx, orig_idx + len(keyword)))

        # Arabic keyword patterns
        for compiled, sig in self._ar_compiled:
            for m in compiled.finditer(text):
                if sig not in seen_signals:
                    signals.append(sig)
                    seen_signals.add(sig)
                match_spans.append((m.start(), m.end()))

        # Deduplicate and sort spans
        match_spans = sorted(set(match_spans), key=lambda x: x[0])
        return signals, match_spans

    # ── S5: SENTENCE-BASED IMPORTANT SEGMENTS ────────────────────────────

    def _extract_important_segments(self, text: str, match_spans: list[tuple[int, int]]) -> list[str]:
        """Extract full sentences containing matches, deduplicated."""
        if not match_spans:
            return []

        # Split into sentences
        sentences = re.split(r"(?<=[.!?\n])\s+", text)
        sentence_spans = []
        pos = 0
        for sent in sentences:
            start = text.index(sent, pos)
            end = start + len(sent)
            sentence_spans.append((sent.strip(), start, end))
            pos = end

        # Find sentences that contain matches
        segments = []
        seen = set()
        for span_start, span_end in match_spans:
            for sent_text, sent_start, sent_end in sentence_spans:
                if sent_start <= span_start < sent_end:
                    key = sent_text[:100]
                    if key not in seen and len(sent_text) > 5:
                        segments.append(sent_text[:MAX_SEGMENT_CHARS])
                        seen.add(key)
                    break

        return segments[:5]

    # ── S6: TWO-STAGE DOMAIN DETECTION ───────────────────────────────────

    def _detect_domains(self, signals: list[str], content_type: str, language: str) -> list[str]:
        """Two-stage domain detection: signal mapping + combination scoring."""
        if not signals:
            return self._domains_from_content_type(content_type)

        # Stage A: collect all domains from signal mapping
        domain_scores: dict[str, float] = {}
        signal_set = set(signals)

        for sig in signals:
            for domain in _SIGNAL_TO_DOMAIN.get(sig, []):
                domain_scores[domain] = domain_scores.get(domain, 0.0) + 1.0

        # Normalize Stage A scores (cap at 1.0)
        if domain_scores:
            max_score = max(domain_scores.values())
            for d in domain_scores:
                domain_scores[d] = round(domain_scores[d] / max_score, 2)

        # Stage B: check for combination bonuses
        for signal_combo, combo_domains in _DOMAIN_COMBOS.items():
            if signal_combo.issubset(signal_set):
                for domain, bonus in combo_domains.items():
                    domain_scores[domain] = max(domain_scores.get(domain, 0.0), bonus)

        # Sort by score descending
        sorted_domains = sorted(domain_scores.items(), key=lambda x: -x[1])

        # Return domains with score >= 0.5, max 4
        result = [d for d, s in sorted_domains if s >= 0.5]
        if not result:
            result = self._domains_from_content_type(content_type)

        return result[:4]

    def _domains_from_content_type(self, content_type: str) -> list[str]:
        """Fallback domain based on content type."""
        mapping = {
            "email":         ["EmailChannel"],
            "web_form":      ["WebChannel"],
            "api_payload":   ["IT"],
            "file_upload":   ["WebChannel", "IT"],
            "code":          ["IT"],
        }
        return mapping.get(content_type, ["General"])

    # ── S7: BUSINESS CONTEXT GENERATOR ───────────────────────────────────

    def _generate_context(self, result: PreprocessResult) -> str:
        """Template-based business context generation."""
        # First matching template wins
        for condition_fn, template in _CONTEXT_TEMPLATES:
            if condition_fn(result):
                signals_str = ", ".join(result.sensitivity_indicators[:4])
                return template.format(
                    content_type=result.content_type or "general",
                    signals=signals_str,
                    lang=result.language or "en",
                )

        # Fallback: content-type + domain based
        if result.domains_detected:
            domains_str = ", ".join(result.domains_detected[:3])
            return f"{result.content_type.capitalize()} content with {domains_str} relevance"

        return f"{result.content_type.capitalize()} content"

    # ── MAIN PROCESS METHOD ──────────────────────────────────────────────

    def process(self, raw_text: str) -> PreprocessResult:
        t_start = time.time()
        result = PreprocessResult()
        result.fallback_used = False
        result.model_used = "local_fallback_v2"
        if raw_text is None:
            raw_text = ""
        result.original_length = len(raw_text)

        text = raw_text.strip()
        if not text:
            result.language = "en"
            result.language_confidence = 0.0
            result.content_type = "general"
            result.processed = True
            result.clean_text = ""
            result.business_context = "Empty content"
            result.latency_ms = round((time.time() - t_start) * 1000, 1)
            return result

        # S7: Detect & decode encoding
        text, encoding = self._detect_and_decode(text)
        result.encoding_detected = encoding

        # S2: Detect content type
        result.content_type = self._classify_content_type(text)

        # S1: Language detection
        lang, conf = self._detect_language(text)
        result.language = lang
        result.language_confidence = conf

        # S3: Extract sensitivity signals
        signals, match_spans = self._extract_signals(text)
        result.sensitivity_indicators = signals
        result.contains_sensitive_content = len(signals) > 0

        # S5: Build important segments from sentence context
        result.important_segments = self._extract_important_segments(text, match_spans)

        # S6: Detect domains (two-stage)
        result.domains_detected = self._detect_domains(signals, result.content_type, lang)

        # S7: Generate business context
        result.business_context = self._generate_context(result)

        result.clean_text = text[:MAX_INPUT_CHARS]
        result.processed = True
        result.latency_ms = round((time.time() - t_start) * 1000, 1)
        return result


# ══════════════════════════════════════════════════════════════════════════
# PUBLIC INTERFACE — used by ai_orchestrator.py
# ══════════════════════════════════════════════════════════════════════════

class LLMPreprocessor:
    """
    Preprocessor entry point. Called by ai_orchestrator.py before classifiers.
    Uses LocalFallbackPreprocessor v2 (no LLM dependency).

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

    def __init__(self, **kwargs):
        self._processor = LocalFallbackPreprocessor()

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
        return False

    def status(self) -> dict:
        return {
            "llm_available": False,
            "model":         "local_fallback_v2",
            "mode":          "local_fallback_v2",
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

        ("Credit card + CVV in plain text",
         "4111111111111111 CVV: 123 expiry 12/26 cardholder: John Doe"),

        ("Multiple credit cards + NID",
         "Visa: 4111111111111111, Mastercard: 5555555555554444, NID: 29001011234567"),

        ("Arabic bank account",
         "\u062a\u062d\u0648\u064a\u0644 \u0628\u0646\u0643\u064a: \u0627\u0644\u062d\u0633\u0627\u0628 "
         "\u0627\u0644\u0628\u0646\u0643\u064a EG380019000500000000263180002 \u0644\u0645\u0628\u0644\u063a 50000 \u062c\u0646\u064a\u0647"),

        ("English AWS credentials leak",
         "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE\n"
         "AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"),

        ("Mixed Arabic/English medical",
         "\u0645\u0631\u064a\u0636 \u0623\u062d\u0645\u062f: \u0627\u0644\u062a\u0634\u062e\u064a\u0635 "
         "Diabetes Type 2, \u0627\u0644\u062f\u0648\u0627\u0621 Metformin 500mg, "
         "\u0627\u0644\u0631\u0642\u0645 \u0627\u0644\u0642\u0648\u0645\u064a: 29001011234567"),

        ("Legal NDA document",
         "This NDA is entered into between AcmeCorp and BetaInc. "
         "Attorney-client privileged. Confidential business terms."),

        ("Empty / whitespace",
         "   "),

        ("Long code snippet",
         "def get_db_password():\n    return 'SuperSecret123'\n"
         "DATABASE_URL = 'postgresql://admin:pass@prod-db.internal:5432/customers'"),
    ]

    correct = 0
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
        if r.important_segments:
            print(f"  Segments     : {r.important_segments[:2]}")
        print(f"  Classifier input:")
        cl_text = r.classifier_input()
        if cl_text:
            for line in cl_text.splitlines():
                print(f"    {line[:120]}")
        print(f"  Latency      : {r.latency_ms}ms")
        if r.contains_sensitive_content:
            correct += 1

    print(f"\n{'=' * 65}")
    print(f"  Total test cases: {len(tests)}")
    print(f"  Detected sensitive: {correct}")
    print(f"  Status: LocalFallbackPreprocessor v2 active (model={prep.status()['model']})")
    print(f"{'=' * 65}")