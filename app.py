import os
import json

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from openai import OpenAI

app = FastAPI(
    title="Invoice Extraction API"
)

# AI Pipe (OpenAI-compatible)
client = OpenAI(
    api_key=os.environ["AIPIPE_TOKEN"],
    base_url="https://aipipe.org/openai/v1"
)


class InvoiceRequest(BaseModel):
    document_id: str
    text: str
    json_schema: dict = Field(alias="schema")

    model_config = ConfigDict(populate_by_name=True)


SYSTEM_PROMPT = """
You are an expert invoice information extraction engine.

Your task is to extract information ONLY from the invoice text.

Rules:

- Return ONLY valid JSON.
- No markdown.
- No explanations.
- No extra keys.
- Follow the provided JSON Schema EXACTLY.
- Preserve array order.
- Numbers must be integers.
- Dates must be YYYY-MM-DD.
- currency must be one of:
USD
EUR
GBP
INR
JPY

Extraction Rules

vendor:
Use the biller's proper name exactly as written.

currency:
Return ISO 4217 code.

total_amount:
Return integer only.

invoice_date:
Normalize to YYYY-MM-DD.

due_in_days:
Return integer.

is_paid:
Return boolean.

priority:
Must be one of
low
normal
high
urgent

contact_email:
Lowercase.

line_items:
Preserve order.

item_count:
Number of line_items.

Return ONLY JSON.
"""


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/extract")
def extract(req: InvoiceRequest):

    try:

        response = client.chat.completions.create(
            model="gpt-4.1",
            temperature=0,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "invoice_schema",
                    "schema": req.json_schema,
                    "strict": True
                }
            },
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": req.text
                }
            ]
        )
        result = json.loads(response.choices[0].message.content)

        # ---------- Normalization ----------

        # Vendor
        if isinstance(result.get("vendor"), str):
            result["vendor"] = result["vendor"].strip().rstrip(".")

        # Email
        if isinstance(result.get("contact_email"), str):
            result["contact_email"] = result["contact_email"].strip().lower()

        # Currency
        if isinstance(result.get("currency"), str):
            result["currency"] = result["currency"].strip().upper()

        # Priority
        if isinstance(result.get("priority"), str):
            result["priority"] = result["priority"].strip().lower()

        # Integer fields
        for key in ["total_amount", "due_in_days", "item_count"]:
            if key in result and result[key] is not None:
                result[key] = int(result[key])

        # Line items
        if isinstance(result.get("line_items"), list):
            for item in result["line_items"]:
                if "sku" in item and isinstance(item["sku"], str):
                    item["sku"] = item["sku"].strip()

                if "quantity" in item:
                    item["quantity"] = int(item["quantity"])

                if "unit_price" in item:
                    item["unit_price"] = int(item["unit_price"])

            # Recalculate item count to guarantee correctness
            result["item_count"] = len(result["line_items"])

        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.post("/")
def extract_root(req: InvoiceRequest):
    return extract(req)