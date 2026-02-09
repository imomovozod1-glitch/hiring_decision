import streamlit as st
import fitz  # PyMuPDF
from openai import OpenAI
import yaml
import json
import re

# --- SETTINGS ---
# Put the API key in Streamlit Secrets
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)


# --- FUNCTIONS ---

def load_rules(filepath="hiring_rules.yaml"):
    """Loads rules from a YAML file"""
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        st.error(f"Error: '{filepath}' file not found. Please create the rules file.")
        st.stop()
    except Exception as e:
        st.error(f"There is an error in the YAML file: {e}")
        st.stop()


def clean_pdf_text(text):
    """Cleans disclaimers and unnecessary legal text"""
    # Disclaimer patterns (can be extended if needed)
    patterns = [
        r"Disclaimer: The information contained herein.*?binding in all 50 states\.",
        r"The User agrees to release.*?dba StarPoint Screening",
        r"Per the signed Membership Agreement.*?"
    ]

    cleaned_text = text
    for pattern in patterns:
        cleaned_text = re.sub(pattern, "", cleaned_text, flags=re.DOTALL | re.IGNORECASE)

    return re.sub(r'\s+', ' ', cleaned_text).strip()


def extract_text_from_pdf(uploaded_file):
    """Extracts cleaned text from an uploaded PDF file"""
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return clean_pdf_text(text)


# --- MAIN UI ---

st.set_page_config(page_title="Driver Safety AI", page_icon="🚛", layout="centered")

# Load rules
rules_data = load_rules()
company_name = rules_data['meta']['carrier_name']

st.title(f"🚛 {company_name}")
st.caption("AI-Powered Driver Qualification System (YAML Configured)")

# Upload file
uploaded_file = st.file_uploader("Upload the driver report (PDF)", type=["pdf"])

if uploaded_file:
    # 1. Extract text
    pdf_text = extract_text_from_pdf(uploaded_file)

    # 2. Analyze button
    if st.button("🚀 Start Analysis", type="primary"):

        with st.status("AI Agent is running...", expanded=True) as status:
            st.write("Checking MVR...")

            # Convert YAML to string (for the prompt)
            rules_str = yaml.dump(rules_data)

            # --- PROMPT ---
            system_prompt = """
            You are a Safety Manager assistant. Your tasks:
            1. Validate the Candidate information against the provided YAML Rules.
            2. Return the output strictly in JSON format.

            IMPORTANT:
            - If any violation matches the "Hard Stops" list -> REJECT.
            - If Age does not match the rule -> REJECT.
            - If total "Minor" violations exceed the threshold -> REJECT.

            JSON STRUCTURE:
            {
                "status": "APPROVE" | "REJECT" | "DELAY" | "REVIEW",
                "final_score": "Short summary (e.g., 2 Minor, 1 Major)",
                "violations_found": [
                    {"violation": "Name", "severity": "Major/Minor", "rule_matched": "Rule from YAML"}
                ],
                "missing_docs": ["Document name" (if Delay)],
                "recruiter_action": "Clear instruction for recruiter (plain English)"
            }
            """

            user_prompt = f"""
            --- RULES (YAML) ---
            {rules_str}

            --- CANDIDATE DATA (PDF) ---
            {pdf_text}
            """

            try:
                # AI Call
                response = client.chat.completions.create(
                    model="gpt-5-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"}
                )

                # Get result
                result = json.loads(response.choices[0].message.content)
                status.update(label="Analysis completed!", state="complete", expanded=False)

                # --- VISUALIZATION ---
                decision = result.get("status", "UNKNOWN").upper()
                action = result.get("recruiter_action", "")
                violations = result.get("violations_found", [])

                st.divider()

                # Color-coded statuses
                if decision == "REJECT":
                    st.error(f"⛔ DECISION: {decision}")
                elif decision == "APPROVE":
                    st.success(f"✅ DECISION: {decision}")
                elif decision == "DELAY":
                    st.warning(f"⚠️ DECISION: {decision} (Missing documents)")
                elif decision == "REVIEW":
                    st.info(f"👀 DECISION: {decision} (Additional review required)")
                else:
                    st.write(f"DECISION: {decision}")

                # Details section
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("⚠️ Issues Found")
                    if violations:
                        for v in violations:
                            severity_icon = "🔴" if v['severity'] == "Major" else "🟡"
                            st.markdown(f"{severity_icon} **{v['violation']}**")
                            st.caption(f"Rule: {v['rule_matched']} | Severity: {v['severity']}")
                    else:
                        st.success("Clean! No violations found.")

                with col2:
                    st.subheader("📝 Recruiter Instructions")
                    st.info(action)
                    if result.get("missing_docs"):
                        st.markdown("**Missing documents:**")
                        for doc in result["missing_docs"]:
                            st.markdown(f"- {doc}")

            except Exception as e:
                status.update(label="Error!", state="error")
                st.error(f"System error: {e}")
