import json
import re
import streamlit as st
from openai import OpenAI

MODEL_NAME = "gpt-5.2"

st.set_page_config(page_title="MVR Hiring Decision (Deterministic)", layout="centered")

api_key = st.secrets.get("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY secret topilmadi. Streamlit Secrets ga qo‘ying.")
    st.stop()

client = OpenAI(api_key=api_key)

# -----------------------------
# 1) GPT faqat EXTRACT qiladi
# -----------------------------
EXTRACTION_SYSTEM = """
You are an MVR (Motor Vehicle Record) extraction agent.
Read the provided MVR PDF and output ONLY JSON matching the schema.

Rules:
- Do NOT decide ACCEPT/REJECT.
- Do NOT guess missing data. If something isn't explicitly present, set it to null/false/empty.
- Count accidents from the "Accidents" section (each incident date = 1 accident).
- Extract violations descriptions as written.
- If the PDF is unreadable or you cannot extract data confidently, set `unreadable_pdf` = true.
"""

EXTRACTION_JSON_SCHEMA = {
    "type": "json_schema",
    "name": "mvr_extraction",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "unreadable_pdf": {"type": "boolean"},
            "license_status": {"type": ["string", "null"]},  # e.g. VALID / SUSPENDED / REVOKED / etc.
            "cdl_class": {"type": ["string", "null"]},       # e.g. A / B / C / null
            "missing_cdl_endorsements": {"type": "boolean"}, # explicit missing endorsements only
            "medical_cert_expired": {"type": "boolean"},      # true only if clearly expired / expired medical
            "accident_count": {"type": "integer", "minimum": 0},
            "violations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "incident_date": {"type": ["string", "null"]},
                        "conviction_date": {"type": ["string", "null"]},
                        "description": {"type": "string"}
                    },
                    "required": ["incident_date", "conviction_date", "description"]
                }
            },
            "major_indicators": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "speeding_over_15": {"type": "boolean"},
                    "tailgating_following_too_closely": {"type": "boolean"},
                    "texting_handheld_phone": {"type": "boolean"},
                    "reckless_careless": {"type": "boolean"},
                    "dui_dwi_alcohol_drugs": {"type": "boolean"},
                    "hit_and_run": {"type": "boolean"},
                    "felony_conviction": {"type": "boolean"},
                    "refused_test": {"type": "boolean"},
                    "failed_dot_drug_test_no_sap": {"type": "boolean"},
                    "license_suspended_not_reinstated": {"type": "boolean"}
                },
                "required": [
                    "speeding_over_15",
                    "tailgating_following_too_closely",
                    "texting_handheld_phone",
                    "reckless_careless",
                    "dui_dwi_alcohol_drugs",
                    "hit_and_run",
                    "felony_conviction",
                    "refused_test",
                    "failed_dot_drug_test_no_sap",
                    "license_suspended_not_reinstated"
                ]
            }
        },
        "required": [
            "unreadable_pdf",
            "license_status",
            "cdl_class",
            "missing_cdl_endorsements",
            "medical_cert_expired",
            "accident_count",
            "violations",
            "major_indicators"
        ]
    }
}

# -----------------------------
# 2) Deterministic policy
# -----------------------------
def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()

def count_uninspected_vehicle(violations) -> int:
    # match "operate uninspected vehicle" in any casing/spacing
    c = 0
    for v in violations or []:
        desc = normalize(v.get("description", ""))
        if "operate uninspected vehicle" in desc or "uninspected vehicle" in desc:
            c += 1
    return c

def decide_policy(extracted: dict) -> dict:
    flags = []
    reasons = []

    if extracted.get("unreadable_pdf") is True:
        return {
            "decision": "REJECT",
            "reason": "PDF unreadable / insufficient data.",
            "flags": ["INSUFFICIENT_DATA"]
        }

    # Expired medical → REJECT
    if extracted.get("medical_cert_expired") is True:
        flags.append("EXPIRED_MEDICAL")
        reasons.append("Expired medical certificate.")

    # Missing endorsements → REJECT
    if extracted.get("missing_cdl_endorsements") is True:
        flags.append("MISSING_ENDORSEMENTS")
        reasons.append("Missing required CDL endorsements.")

    # License suspended not reinstated → REJECT
    mi = extracted.get("major_indicators", {}) or {}
    if mi.get("license_suspended_not_reinstated") is True:
        flags.append("LICENSE_SUSPENDED")
        reasons.append("License suspended (not reinstated).")

    # AUTOMATIC REJECT majors (your list)
    major_map = [
        ("speeding_over_15", "SPEEDING_OVER_15"),
        ("tailgating_following_too_closely", "FOLLOWING_TOO_CLOSELY"),
        ("texting_handheld_phone", "TEXTING_HANDHELD_PHONE"),
        ("reckless_careless", "RECKLESS_CARELESS"),
        ("dui_dwi_alcohol_drugs", "DUI_DWI"),
        ("hit_and_run", "HIT_AND_RUN"),
        ("felony_conviction", "FELONY_CONVICTION"),
        ("refused_test", "REFUSED_TEST"),
        ("failed_dot_drug_test_no_sap", "FAILED_DOT_DRUG_TEST_NO_SAP"),
    ]
    for key, flag in major_map:
        if mi.get(key) is True:
            flags.append(flag)
            reasons.append(f"Major violation indicator: {flag}.")

    # STRICT POLICY ADDITIONS (as you requested)
    # Accident count >= 2 → REJECT (regardless of fault wording)
    accident_count = int(extracted.get("accident_count") or 0)
    if accident_count >= 2:
        flags.append("ACCIDENTS_2_PLUS")
        reasons.append(f"Accident count is {accident_count} (>=2).")

    # Operate uninspected vehicle >=2 → REJECT
    uninspected_count = count_uninspected_vehicle(extracted.get("violations"))
    if uninspected_count >= 2:
        flags.append("UNINSPECTED_VEHICLE_2_PLUS")
        reasons.append(f"Operate uninspected vehicle count is {uninspected_count} (>=2).")

    # Final decision
    if reasons:
        return {
            "decision": "REJECT",
            "reason": " | ".join(reasons),
            "flags": sorted(list(set(flags)))
        }

    return {
        "decision": "ACCEPT",
        "reason": "No reject triggers found under current policy.",
        "flags": []
    }

# -----------------------------
# 3) OpenAI extraction call
# -----------------------------
def extract_from_uploaded_pdf(uploaded_file) -> dict:
    pdf_bytes = uploaded_file.getvalue()
    if not pdf_bytes:
        return {
            "unreadable_pdf": True,
            "license_status": None,
            "cdl_class": None,
            "missing_cdl_endorsements": False,
            "medical_cert_expired": False,
            "accident_count": 0,
            "violations": [],
            "major_indicators": {
                "speeding_over_15": False,
                "tailgating_following_too_closely": False,
                "texting_handheld_phone": False,
                "reckless_careless": False,
                "dui_dwi_alcohol_drugs": False,
                "hit_and_run": False,
                "felony_conviction": False,
                "refused_test": False,
                "failed_dot_drug_test_no_sap": False,
                "license_suspended_not_reinstated": False
            }
        }

    uploaded = client.files.create(
        file=(uploaded_file.name, pdf_bytes, "application/pdf"),
        purpose="user_data"
    )

    resp = client.responses.create(
        model=MODEL_NAME,
        input=[
            {"role": "system", "content": EXTRACTION_SYSTEM},
            {"role": "user", "content": [
                {"type": "input_text", "text": "Extract MVR facts into the required JSON schema."},
                {"type": "input_file", "file_id": uploaded.id}
            ]}
        ],
        text={"format": EXTRACTION_JSON_SCHEMA}
    )

    return json.loads(resp.output_text)

# -----------------------------
# UI
# -----------------------------
st.title("🚛 MVR Hiring Decision (Deterministic Policy)")
st.caption("PDF → GPT faqat faktlarni chiqaradi → qaror Python qoidalari bilan beriladi")

uploaded_file = st.file_uploader("MVR PDF yuklang", type=["pdf"])

show_debug = st.checkbox("Debug: extracted JSON ni ko‘rsatish", value=False)

if uploaded_file and st.button("🔍 Tahlil qilish", type="primary"):
    try:
        with st.spinner("PDF o‘qilmoqda (extract) va policy qo‘llanmoqda..."):
            extracted = extract_from_uploaded_pdf(uploaded_file)
            result = decide_policy(extracted)

        decision = result["decision"]
        reason = result["reason"]
        flags = result["flags"]

        (st.success if decision == "ACCEPT" else st.error)(f"DECISION: {decision}")
        st.write("**REASON:**", reason)
        st.write("**FLAGS:**", ", ".join(flags) if flags else "None")

        if show_debug:
            st.divider()
            st.subheader("Extracted JSON (debug)")
            st.json(extracted)

    except Exception as e:
        st.error("Xatolik yuz berdi")
        st.exception(e)
