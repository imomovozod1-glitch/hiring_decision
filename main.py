import json
import streamlit as st
from openai import OpenAI

# =============================
# CONFIG
# =============================
MODEL_NAME = "gpt-5.2"

st.set_page_config(
    page_title="MVR Hiring Decision – AI 5.2",
    layout="centered"
)

api_key = st.secrets.get("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY topilmadi. Streamlit Secrets ga qo‘ying.")
    st.stop()

client = OpenAI(api_key=api_key)

# =============================
# VERY STRICT NORMATIVE
# =============================
NORMATIVE_RULES = """
You are an MVR (Motor Vehicle Record) hiring decision agent.

THIS COMPANY POLICY IS EXTREMELY STRICT.

DEFAULT:
- IF THERE IS ANY DOUBT, AMBIGUITY, OR RISK → REJECT.
- ACCEPT is allowed ONLY if the record is perfectly clean.

AUTOMATIC REJECT (ANY ONE = REJECT):
- Any accident history (1 or more accidents, regardless of fault)
- Any insurance claim
- Any equipment or inspection violation (including "Operate Uninspected Vehicle")
- Any CDL status other than VALID (including DISQUALIFIED, SUSPENDED, EXPIRED)
- Any reference to Drug & Alcohol Clearinghouse issues
- Failed or refused DOT drug/alcohol test (even if old)
- DUI / DWI / alcohol / drugs
- Reckless or careless driving
- Following too closely / tailgating
- Texting or handheld phone use
- Speeding more than 10 mph over limit
- Hit and run
- Felony conviction
- Unlicensed driver
- Missing or outdated CDL endorsements
- Expired or missing medical certificate

INTERPRETATION RULES:
- "ACCIDENT" presence alone = REJECT (do NOT evaluate fault).
- "DISQUALIFIED" anywhere = REJECT.
- "PROHIBITED" + "CLEARINGHOUSE" = REJECT.
- Administrative wording does NOT reduce severity.
- If PDF text is incomplete or unclear → REJECT.

OUTPUT:
- Return ONLY valid JSON.
- No explanations outside JSON.
"""

# =============================
# JSON SCHEMA (STRICT)
# =============================
TEXT_FORMAT_JSON_SCHEMA = {
    "type": "json_schema",
    "name": "mvr_hiring_decision",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "decision": {
                "type": "string",
                "enum": ["ACCEPT", "REJECT"]
            },
            "reason": {
                "type": "string"
            },
            "flags": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["decision", "reason", "flags"]
    }
}

# =============================
# CORE FUNCTION
# =============================
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
        reasoning={"effort": "xhigh"},
        input=[
            {
                "role": "system",
                "content": NORMATIVE_RULES
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Analyze the attached MVR PDF using the company policy. "
                            "Decide ACCEPT or REJECT. Be extremely strict."
                        )
                    },
                    {
                        "type": "input_file",
                        "file_id": uploaded.id
                    }
                ]
            }
        ],
        text={"format": TEXT_FORMAT_JSON_SCHEMA}
    )

    return json.loads(resp.output_text)

# =============================
# UI
# =============================
st.title("🚛 MVR Hiring Decision")
st.caption("AI 5.2 (Thinking) • Extremely Strict Policy • Default = REJECT")

uploaded_file = st.file_uploader("MVR PDF yuklang", type=["pdf"])

if uploaded_file and st.button("🔍 AI bilan tahlil qilish", type="primary"):
    try:
        with st.spinner("AI hujjatni tahlil qilmoqda..."):
            result = decide_from_uploaded_pdf(uploaded_file)

        decision = result["decision"]
        reason = result["reason"]
        flags = result["flags"]

        if decision == "REJECT":
            st.error(f"DECISION: {decision}")
        else:
            st.success(f"DECISION: {decision}")

        st.write("**REASON:**", reason)
        st.write("**FLAGS:**", ", ".join(flags) if flags else "None")

    except Exception as e:
        st.error("Xatolik yuz berdi")
        st.exception(e)

