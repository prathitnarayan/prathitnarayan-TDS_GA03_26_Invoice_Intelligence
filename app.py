import os
import json

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI(title="Invoice Extraction API")

client = OpenAI(
    api_key=os.environ["AIPIPE_TOKEN"],
    base_url="https://aipipe.org/openai/v1"
)


class InvoiceRequest(BaseModel):
    document_id: str
    text: str
    schema: dict


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/")
def extract(req: InvoiceRequest):

    system_prompt = """
You are an invoice information extraction engine.

Rules:
- Extract only the requested fields.
- Follow the provided JSON schema exactly.
- No markdown.
- No explanations.
- Return ONLY valid JSON.
- Preserve line item order.
- Numbers must be integers.
- Dates must be YYYY-MM-DD.
- currency must be one of:
USD EUR GBP INR JPY
"""

    try:

        response = client.chat.completions.create(
            model="gpt-4.1",
            temperature=0,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "invoice",
                    "schema": req.schema
                }
            },
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": req.text
                }
            ]
        )

        return json.loads(
            response.choices[0].message.content
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )