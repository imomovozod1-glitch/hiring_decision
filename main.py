import json
import streamlit as st
from openai import OpenAI

# =============================
# CONFIG
# =============================
MODEL = "gpt-5.2"

st.set_page_config(page_title="MVR Checker – Multi Agent", layout="centered")

api_key = st.secrets.get("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY topilmadi")
    st.stop()

client = OpenAI(api_key=api_key)

# =============================
# AGENT 1 – FACT EXTRACTOR
# =============================
EXTRACTOR_PROMPT = """
You are Agent 1: MVR FACT EXTRACTOR.

Read the MVR PDF and extract ONLY objective facts.
Do NOT make hiring decisions.
Do NOT interpret severity.

Extract:
- accident_count (integer)
- violations (list of strings exactly as written)
- cdl_status (VALID / DISQUALIFIED / SUSPENDED / OTHER)
- medical_status (VALID / EXPIRED / UNKNOWN)
- clearinghouse_issue (true/false)
- unlicensed_driver (true/false)

If anything is unclear, set conservative values (UNKNOWN / true).

Return ONLY JSON.
"""

EXTRACT_SCHEMA = {
    "type": "json_schema",
    "name": "mvr_extract",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "accident_count": {"type": "integer"},
            "violations": {"type": "array", "items": {"type": "string"}},
            "cdl_status": {"type": "string"},
            "medical_status": {"type": "string"},
            "clearinghouse_issue": {"type": "boolean"},
            "unlicensed_driver": {"type": "boolean"}
        },
        "required": [
            "accident_count",
            "violations",
            "cdl_status",
            "medical_status",
            "clearinghouse_issue",
            "unlicensed_driver"
        ]
    }
}

# =============================
# AGENT 2 – VALIDATOR
# =============================
VALIDATOR_PROMPT = """
You are Agent 2: FACT VALIDATOR.

Review extracted facts.
Your job:
- Assume risk if ambiguity exists
- If accident_count >= 1 → confirm
- If words like DISQUALIFIED / PROHIBITED / CLEARINGHOUSE appear → confirm issue
- If unsure → escalate risk (true)

Return corrected JSON only.
"""

# =============================
# AGENT 3 – DECISION MAKER
# =============================
DECISION_PROMPT = """
You are Agent 3: FINAL DECISION MAKER.

COMPANY POLICY (EXTREMELY STRICT):

REJECT if ANY of the following:
- accident_count >= 1
- any violation exists
- cdl_status != VALID
- medical_status != VALID
- clearinghouse_issue == true
- unlicensed_driver == true
- any doubt or ambiguity

ACCEPT only if record is perfectly clean.

Return ONLY JSON.
"""

DECISION_SCHEMA = {
    "type": "json_schema",
    "name": "mvr_decision",
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

# =============================
# CORE PIPELINE
# =============================
def run_multi_agent_pipeline(uploaded_file):
    pdf_bytes = uploaded_file.getvalue()

    uploaded = client.files.create(
        file=(uploaded_file.name, pdf_bytes, "application/pdf"),
        purpose="user_data"
    )

    # -------- Agent 1 --------
    extract_resp = client.responses.create(
        model=MODEL,
        reasoning={"effort": "high"},
        input=[
            {"role": "system", "content": EXTRACTOR_PROMPT},
            {"role": "user", "content": [
                {"type": "input_text", "text": "Extract facts from this MVR PDF."},
                {"type": "input_file", "file_id": uploaded.id}
            ]}
        ],
        text={"format": EXTRACT_SCHEMA}
    )

    extracted = json.loads(extract_resp.output_text)

    # -------- Agent 2 --------
    validate_resp = client.responses.create(
        model=MODEL,
        reasoning={"effort": "medium"},
        input=[
            {"role": "system", "content": VALIDATOR_PROMPT},
            {"role": "user", "content": json.dumps(extracted)}
        ],
        text={"format": EXTRACT_SCHEMA}
    )

    validated = json.loads(validate_resp.output_text)

    # -------- Agent 3 --------
    decision_resp = client.responses.create(
        model=MODEL,
        reasoning={"effort": "xhigh"},
        input=[
            {"role": "system", "content": DECISION_PROMPT},
            {"role": "user", "content": json.dumps(validated)}
        ],
        text={"format": DECISION_SCHEMA}
    )

    return validated, json.loads(decision_resp.output_text)

# =============================
# UI
# =============================
st.title("🚛 MVR Checker – Multi Agent AI")
st.caption("3 Agent Pipeline • GPT-5.2 Thinking • Default = REJECT")

uploaded_file = st.file_uploader("MVR PDF yuklang", type=["pdf"])
debug = st.checkbox("Debug: Agent outputlarni ko‘rsatish")

if uploaded_file and st.button("🔍 Tahlil qilish", type="primary"):
    try:
        with st.spinner("3 ta AI agent ishlayapti..."):
            facts, decision = run_multi_agent_pipeline(uploaded_file)

        if decision["decision"] == "REJECT":
            st.error(f"DECISION: {decision['decision']}")
        else:
            st.success(f"DECISION: {decision['decision']}")

        st.write("**REASON:**", decision["reason"])
        st.write("**FLAGS:**", ", ".join(decision["flags"]))

        if debug:
            st.divider()
            st.subheader("Validated Facts (Agent 2)")
            st.json(facts)

    except Exception as e:
        st.error("Xatolik yuz berdi")
        st.exception(e)
