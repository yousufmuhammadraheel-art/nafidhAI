"""
NafidhAI — Arabic Intent Classification LLM Prompt
Used inside the agent workflow to classify incoming user messages before routing.

Supports: MSA (Modern Standard Arabic), Gulf dialect (Khaleeji),
Levantine dialect, English, and code-switched Gulf Arabic/English.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


# ─────────────────────────────────────────────
# OUTPUT SCHEMA
# ─────────────────────────────────────────────

class ExtractedEntity(BaseModel):
    """A single entity extracted from the user message."""
    entity_type: str = Field(
        ...,
        description=(
            "Type of entity: PERSON | ORGANIZATION | DATE | AMOUNT | "
            "NATIONAL_ID | DEPARTMENT | LEAVE_TYPE | INVOICE_NUMBER | LOCATION | OTHER"
        ),
    )
    value: str = Field(..., description="The extracted entity value as it appeared in the input")
    normalized_value: str | None = Field(
        None,
        description="Normalized form (e.g. date → ISO 8601, amount → numeric string)"
    )
    confidence: float = Field(..., ge=0.0, le=1.0)


class IntentClassificationResult(BaseModel):
    """
    Strict output schema for the Arabic Intent Classification LLM call.
    The model must return JSON matching this schema exactly.
    """

    intent: str = Field(
        ...,
        description=(
            "Primary intent. One of: "
            "LEAVE_REQUEST | PAYROLL_INQUIRY | EMPLOYEE_LOOKUP | INVOICE_APPROVAL | "
            "REPORT_REQUEST | SYSTEM_STATUS | ESCALATE_TO_HUMAN | GREETING | "
            "COMPLAINT | OTHER"
        ),
    )

    sub_intent: str | None = Field(
        None,
        description=(
            "More specific intent when applicable. Examples: "
            "LEAVE_REQUEST.ANNUAL | LEAVE_REQUEST.SICK | PAYROLL_INQUIRY.SALARY_SLIP | "
            "PAYROLL_INQUIRY.ALLOWANCE"
        ),
    )

    language: Literal["arabic", "english", "mixed"] = Field(
        ...,
        description="Detected primary language of the input",
    )

    dialect: Literal["msa", "gulf", "levantine", "egyptian", "maghrebi", "unknown"] = Field(
        ...,
        description=(
            "Arabic dialect detected. 'msa' = Modern Standard Arabic (فصحى). "
            "'gulf' = Khaleeji/Gulf Arabic. 'unknown' when language is English."
        ),
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model's confidence in the intent classification (0.0 = no confidence, 1.0 = certain)",
    )

    extracted_entities: list[ExtractedEntity] = Field(
        default_factory=list,
        description="Structured entities extracted from the input",
    )

    requires_human_escalation: bool = Field(
        ...,
        description=(
            "True if the message contains: legal threats, harassment, medical emergency, "
            "expressions of distress, requests outside agent scope, or explicit escalation request. "
            "When True, intent should be ESCALATE_TO_HUMAN."
        ),
    )

    raw_input_language_detected: str = Field(
        ...,
        description=(
            "ISO 639-1 language code of the detected primary language. "
            "Examples: 'ar', 'en', 'ar-AE', 'ar-SA'"
        ),
    )

    sentiment: Literal["positive", "neutral", "negative", "urgent"] = Field(
        ...,
        description="Detected sentiment/tone of the message",
    )

    @field_validator("confidence")
    @classmethod
    def round_confidence(cls, v: float) -> float:
        return round(v, 3)


# ─────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────

INTENT_CLASSIFICATION_SYSTEM_PROMPT = """\
You are an Arabic-first intent classification engine for NafidhAI, an enterprise AI platform \
serving Gulf (KSA, UAE) and broader MENA region enterprises.

## YOUR TASK
Analyze the user message provided in <user_input> tags. Classify its intent, detect language \
and dialect, extract structured entities, and determine if human escalation is required.

## LANGUAGE HANDLING
- Input may be: Modern Standard Arabic (فصحى/MSA), Gulf dialect (Khaleeji — spoken in KSA, UAE, \
Kuwait, Bahrain, Qatar), Levantine Arabic, English, or code-switched Arabic/English (common in Gulf).
- Detect the primary language and dialect accurately. Do not confuse Gulf dialect markers \
(e.g. "وش", "كيفك", "إيش", "ودي") with MSA.
- For code-switched input (e.g. "أبي أعمل leave request"), set language = "mixed".

## INTENT CATEGORIES
Classify into exactly one of:
- LEAVE_REQUEST: Employee requesting time off (annual, sick, emergency, maternity, etc.)
- PAYROLL_INQUIRY: Questions about salary, payslip, allowances, deductions, end-of-service
- EMPLOYEE_LOOKUP: Searching for employee information
- INVOICE_APPROVAL: Approving or querying invoices, purchase orders, vendor payments
- REPORT_REQUEST: Requesting reports, dashboards, analytics
- SYSTEM_STATUS: Asking about system availability, errors, integrations
- ESCALATE_TO_HUMAN: User explicitly wants a human, or message triggers escalation rules
- GREETING: Pure greeting with no actionable intent
- COMPLAINT: User expressing a complaint or problem
- OTHER: Does not fit any above category

## ENTITY EXTRACTION
Extract all relevant entities: names, dates (convert Hijri to Gregorian if possible), \
amounts (in SAR/AED/USD), national IDs (mask all but last 4 digits in normalized_value), \
department names, leave types, invoice numbers, and locations.

## HUMAN ESCALATION TRIGGERS — set requires_human_escalation = true if:
1. User expresses legal threat or mentions "محامي" (lawyer) / "قانوني" (legal)
2. User expresses significant distress or emergency
3. Request is explicitly outside enterprise HR/Finance/Operations scope
4. User says "تحدث مع إنسان" / "I want a human" / "بشر" / "موظف"
5. Confidence of intent classification is below 0.5

## OUTPUT FORMAT
Return ONLY a valid JSON object. No preamble. No explanation. No markdown.
The JSON must exactly match this schema:
{
  "intent": string,
  "sub_intent": string | null,
  "language": "arabic" | "english" | "mixed",
  "dialect": "msa" | "gulf" | "levantine" | "egyptian" | "maghrebi" | "unknown",
  "confidence": float (0.0–1.0),
  "extracted_entities": [
    {
      "entity_type": string,
      "value": string,
      "normalized_value": string | null,
      "confidence": float
    }
  ],
  "requires_human_escalation": boolean,
  "raw_input_language_detected": string (ISO 639-1),
  "sentiment": "positive" | "neutral" | "negative" | "urgent"
}
"""

# ─────────────────────────────────────────────
# FEW-SHOT EXAMPLES
# ─────────────────────────────────────────────

FEW_SHOT_EXAMPLES = [
    # ── EXAMPLE 1: MSA (Modern Standard Arabic) — Leave Request ──
    {
        "description": "MSA formal leave request with specific dates",
        "input": "أرغب في تقديم طلب إجازة سنوية للفترة من 15 يوليو 2025 حتى 25 يوليو 2025",
        "output": {
            "intent": "LEAVE_REQUEST",
            "sub_intent": "LEAVE_REQUEST.ANNUAL",
            "language": "arabic",
            "dialect": "msa",
            "confidence": 0.97,
            "extracted_entities": [
                {
                    "entity_type": "LEAVE_TYPE",
                    "value": "إجازة سنوية",
                    "normalized_value": "annual_leave",
                    "confidence": 0.99,
                },
                {
                    "entity_type": "DATE",
                    "value": "15 يوليو 2025",
                    "normalized_value": "2025-07-15",
                    "confidence": 0.99,
                },
                {
                    "entity_type": "DATE",
                    "value": "25 يوليو 2025",
                    "normalized_value": "2025-07-25",
                    "confidence": 0.99,
                },
            ],
            "requires_human_escalation": False,
            "raw_input_language_detected": "ar",
            "sentiment": "neutral",
        },
    },

    # ── EXAMPLE 2: Gulf Dialect (Khaleeji) — Payroll Inquiry ──
    {
        "description": "Gulf Khaleeji dialect salary inquiry with urgency marker",
        "input": "وش صار براتبي هالشهر؟ ما وصل الراتب وابي أعرف في إيش المشكلة",
        "output": {
            "intent": "PAYROLL_INQUIRY",
            "sub_intent": "PAYROLL_INQUIRY.SALARY_NOT_RECEIVED",
            "language": "arabic",
            "dialect": "gulf",
            "confidence": 0.94,
            "extracted_entities": [
                {
                    "entity_type": "DATE",
                    "value": "هالشهر",
                    "normalized_value": "current_month",
                    "confidence": 0.88,
                }
            ],
            "requires_human_escalation": False,
            "raw_input_language_detected": "ar-SA",
            "sentiment": "urgent",
        },
    },

    # ── EXAMPLE 3: Code-switched Gulf Arabic/English — Invoice Approval ──
    {
        "description": "Mixed Gulf Arabic and English — invoice approval request",
        "input": "أبي أعمل approve على invoice رقم INV-2025-4821 من سابك، المبلغ 45,000 ريال",
        "output": {
            "intent": "INVOICE_APPROVAL",
            "sub_intent": None,
            "language": "mixed",
            "dialect": "gulf",
            "confidence": 0.96,
            "extracted_entities": [
                {
                    "entity_type": "INVOICE_NUMBER",
                    "value": "INV-2025-4821",
                    "normalized_value": "INV-2025-4821",
                    "confidence": 0.99,
                },
                {
                    "entity_type": "ORGANIZATION",
                    "value": "سابك",
                    "normalized_value": "SABIC",
                    "confidence": 0.93,
                },
                {
                    "entity_type": "AMOUNT",
                    "value": "45,000 ريال",
                    "normalized_value": "45000 SAR",
                    "confidence": 0.99,
                },
            ],
            "requires_human_escalation": False,
            "raw_input_language_detected": "ar-SA",
            "sentiment": "neutral",
        },
    },
]


# ─────────────────────────────────────────────
# LANGCHAIN PROMPT BUILDER
# ─────────────────────────────────────────────

def build_intent_classification_messages(user_input: str) -> list[dict]:
    """
    Build the complete messages list for the intent classification LLM call.
    
    SECURITY: user_input is wrapped in XML delimiters — NEVER interpolated
    into the system prompt string. This prevents prompt injection.
    
    Args:
        user_input: Raw user message (Arabic, English, or mixed). Untrusted.
    
    Returns:
        List of message dicts for LangChain ainvoke().
    """
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

    messages = [SystemMessage(content=INTENT_CLASSIFICATION_SYSTEM_PROMPT)]

    # Inject few-shot examples as alternating Human/AI turns
    import json
    for example in FEW_SHOT_EXAMPLES:
        messages.append(
            HumanMessage(content=f"<user_input>{example['input']}</user_input>")
        )
        messages.append(
            AIMessage(content=json.dumps(example["output"], ensure_ascii=False, indent=2))
        )

    # The actual user input — wrapped in XML delimiters, never raw interpolation
    # Max length enforced to prevent token flooding
    sanitized_input = user_input[:2048]  # Hard cap: 2048 chars
    messages.append(
        HumanMessage(content=f"<user_input>{sanitized_input}</user_input>")
    )

    return messages


# ─────────────────────────────────────────────
# LLM CALL CONFIG
# ─────────────────────────────────────────────

INTENT_CLASSIFICATION_LLM_CONFIG = {
    "model_provider": "anthropic",
    "model_name": "claude-sonnet-4-20250514",
    "temperature": 0.0,       # Zero temperature: deterministic classification
    "max_tokens": 1000,       # Classification output is compact JSON
    "timeout_seconds": 15,    # Fast timeout: classification must be near-realtime
}
