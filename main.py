import json
import streamlit as st
from openai import OpenAI

MODEL_NAME = "gpt-5-mini"

st.set_page_config(page_title="MVR Hiring Decision", layout="centered")

api_key = st.secrets.get("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY secret topilmadi. Streamlit Secrets ga qo‘ying.")
    st.stop()

client = OpenAI(api_key=api_key)

NORMATIVE_RULES = """
You are an MVR (Motor Vehicle Record) hiring decision agent.
Compare the provided MVR (in PDF) against the rules below and return ONLY JSON.

A) AUTOMATIC REJECT (Major):
- Speeding > 15 mph
- Following too closely / Tailgating
- Texting while driving / handheld cellphone
- Reckless or careless driving
- DUI / DWI / alcohol / drugs
- Hit and run
- Felony convictions
- Refusal to take drug/alcohol test
- Failed DOT drug test (unless SAP RTD completed)
- Suspended license (not reinstated)

B) MODERATE:
- Preventable accidents
- Insurance claims
→ ACCEPT but reason must say "Safety review required"

C) MINOR (thresholds):
- Speeding <= 10 mph (max 2)
- Failure to yield / stop sign (1 only)
- Improper lane / restricted lane (max 2)
- Seatbelt (once only)
- Expired medical → REJECT
- Missing CDL endorsements → REJECT

Parsing:
- "ARREST" without "CONVICTION" ≠ felony
- License VALID = OK
- If PDF unreadable → REJECT (insufficient data)
"""

TEXT_FORMAT_JSON_SCHEMA = {
    "type": "json_schema",
    "name": "mvr_hiring_decision",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "decision": {"type": "string", "enum": ["ACCEPT", "REJECT"]},
            "reason": {"type": "string"},
            "flags": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["decision", "reason", "flags"]
    }
}

def decide_from_uploaded_pdf(uploaded_file):
    pdf_bytes = uploaded_file.getvalue()
    if not pdf_bytes:
        raise ValueError("PDF bo‘sh")

    uploaded = client.files.create(
        file=(uploaded_file.name, pdf_bytes, "application/pdf"),
        purpose="user_data"
    )

    resp = client.responses.create(
        model=MODEL_NAME,
        input=[
            {"role": "system", "content": NORMATIVE_RULES},
            {"role": "user", "content": [
                {"type": "input_text", "text": "Analyze this MVR PDF and return ACCEPT or REJECT."},
                {"type": "input_file", "file_id": uploaded.id}
            ]}
        ],
        text={"format": TEXT_FORMAT_JSON_SCHEMA}
    )

    return json.loads(resp.output_text)

st.title("🚛 MVR Hiring Decision Agent")
st.caption("PDF yuklang → normativ bilan tekshiradi → ACCEPT yoki REJECT")

uploaded_file = st.file_uploader("MVR PDF yuklang", type=["pdf"])

if uploaded_file and st.button("🔍 Tahlil qilish", type="primary"):
    try:
        with st.spinner("PDF o‘qilmoqda va tahlil qilinmoqda..."):
            result = decide_from_uploaded_pdf(uploaded_file)

        decision = result["decision"]
        reason = result["reason"]
        flags = result["flags"]

        (st.success if decision == "ACCEPT" else st.error)(f"DECISION: {decision}")
        st.write("**REASON:**", reason)
        st.write("**FLAGS:**", ", ".join(flags) if flags else "None")

    except Exception as e:
        st.error("Xatolik yuz berdi")
        st.exception(e)
