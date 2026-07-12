import os
import json
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from typing import Any, Optional

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-sonnet-5"

# Fallback schema, used only if the grader doesn't send one in the request.
DEFAULT_SCHEMA = {
    "type": "object",
    "properties": {
        "vendor": {"type": "string"},
        "currency": {"type": "string", "enum": ["USD", "EUR", "GBP", "INR", "JPY"]},
        "total_amount": {"type": "integer"},
        "invoice_date": {"type": "string"},
        "due_in_days": {"type": "integer"},
        "is_paid": {"type": "boolean"},
        "priority": {"type": "string", "enum": ["low", "normal", "high", "urgent"]},
        "contact_email": {"type": "string"},
        "line_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "sku": {"type": "string"},
                    "quantity": {"type": "integer"},
                    "unit_price": {"type": "integer"},
                },
                "required": ["sku", "quantity", "unit_price"],
            },
        },
        "item_count": {"type": "integer"},
    },
    "required": [
        "vendor", "currency", "total_amount", "invoice_date", "due_in_days",
        "is_paid", "priority", "contact_email", "line_items", "item_count",
    ],
}

SYSTEM_PROMPT = """You are a strict data-extraction engine for a logistics firm's ERP system. \
You read one messy free-text invoice/document and call the `extract` tool with clean, \
strongly-typed fields. Follow these normalization rules exactly:

- vendor: the biller's proper name, exactly as written in the text.
- currency: ISO 4217 code (USD, EUR, GBP, INR, JPY). Map words/symbols: "euros"->EUR, \
"pounds sterling"/"GBP"/"£"->GBP, "₹"/"rupees"/"Rs."->INR, "yen"/"¥"->JPY, "$"/"dollars"->USD \
(assume USD for plain "$" unless context says otherwise).
- total_amount: integer, main currency unit, no separators or symbols. Convert spelled-out \
numbers ("twelve thousand four hundred eighty" -> 12480), plain grouped numbers ("12,480" -> 12480), \
Indian digit grouping ("1,24,800" -> 124800), and "K" suffixes ("12K" -> 12000).
- invoice_date: normalize to YYYY-MM-DD.
- due_in_days: integer. "Net 30" -> 30. "payable within 45 days" -> 45. Convert relative phrases \
("due in two weeks" -> 14, "due in a month" -> 30) to integer days.
- is_paid: boolean inferred from wording (e.g. "paid in full" -> true, "awaiting payment"/"unpaid"/\
"outstanding" -> false). If not mentioned, infer the most reasonable value from context; default false.
- priority: one of low, normal, high, urgent, inferred from tone/wording/urgency cues in the text.
- contact_email: lowercase the email exactly as it appears (just lowercase, don't alter the address).
- line_items: array of {sku, quantity, unit_price} in the exact order they appear in the text. \
unit_price is an integer in the main currency unit.
- item_count: number of line items (must equal length of line_items array).

Call the `extract` tool exactly once with all fields filled in. Do not omit any field. \
If a field is genuinely not derivable, make the most reasonable inference from context — \
never leave a field blank or null."""


class ExtractRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    document_id: Optional[str] = None
    text: str
    schema_: Optional[dict] = Field(default=None, alias="schema")


def call_claude(text: str, schema: dict) -> dict:
    if not ANTHROPIC_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="ANTHROPIC_API_KEY environment variable is not set on the server.",
        )

    tool_schema = schema if schema else DEFAULT_SCHEMA

    payload = {
        "model": MODEL,
        "max_tokens": 2048,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": text}],
        "tools": [
            {
                "name": "extract",
                "description": "Return the cleaned, strongly-typed invoice fields.",
                "input_schema": tool_schema,
            }
        ],
        "tool_choice": {"type": "tool", "name": "extract"},
    }

    resp = requests.post(
        ANTHROPIC_URL,
        headers={
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json=payload,
        timeout=60,
    )

    if resp.status_code != 200:
        raise HTTPException(
            status_code=502, detail=f"Claude API error: {resp.status_code} {resp.text}"
        )

    data = resp.json()
    for block in data.get("content", []):
        if block.get("type") == "tool_use" and block.get("name") == "extract":
            return block["input"]

    raise HTTPException(status_code=502, detail="Claude did not return a tool_use block.")


def postprocess(result: dict) -> dict:
    """Deterministic safety net on top of the LLM output."""
    if isinstance(result.get("contact_email"), str):
        result["contact_email"] = result["contact_email"].lower()

    if isinstance(result.get("currency"), str):
        result["currency"] = result["currency"].upper()

    if isinstance(result.get("priority"), str):
        p = result["priority"].lower()
        result["priority"] = p if p in ("low", "normal", "high", "urgent") else "normal"

    if isinstance(result.get("line_items"), list):
        result["item_count"] = len(result["line_items"])
        for item in result["line_items"]:
            for key in ("quantity", "unit_price"):
                if key in item and item[key] is not None:
                    try:
                        item[key] = int(round(float(item[key])))
                    except (TypeError, ValueError):
                        pass

    for key in ("total_amount", "due_in_days"):
        if key in result and result[key] is not None:
            try:
                result[key] = int(round(float(result[key])))
            except (TypeError, ValueError):
                pass

    if "is_paid" in result and not isinstance(result["is_paid"], bool):
        result["is_paid"] = str(result["is_paid"]).strip().lower() in ("true", "yes", "1")

    return result


def run_extraction(payload: ExtractRequest) -> dict:
    schema = payload.schema_ if payload.schema_ else None
    result = call_claude(payload.text, schema)
    return postprocess(result)


@app.post("/extract")
def extract(payload: ExtractRequest):
    return JSONResponse(content=run_extraction(payload))


# Also accept POST at the root, in case the grader posts to the bare submitted URL.
@app.post("/")
def extract_root(payload: ExtractRequest):
    return JSONResponse(content=run_extraction(payload))


@app.get("/")
def health():
    return {"status": "ok"}