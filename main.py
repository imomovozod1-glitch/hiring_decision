import streamlit as st
import fitz  # PyMuPDF
from openai import OpenAI
import yaml
import json
import re

# --- SOZLAMALAR ---
# API Kalitni shu yerga qo'ying
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)


# --- FUNKSIYALAR ---

def load_rules(filepath="hiring_rules.yaml"):
    """YAML faylidan qoidalarni o'qiydi"""
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        st.error(f"Xatolik: '{filepath}' fayli topilmadi. Iltimos, qoidalar faylini yarating.")
        st.stop()
    except Exception as e:
        st.error(f"YAML faylida xatolik bor: {e}")
        st.stop()


def clean_pdf_text(text):
    """Disclaimer va ortiqcha huquqiy matnlarni tozalash"""
    # Disclaimer shablonlari (kerak bo'lsa yanada kuchaytirish mumkin)
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
    """PDF fayldan tozalangan matnni oladi"""
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return clean_pdf_text(text)


# --- ASOSIY INTERFEYS ---

st.set_page_config(page_title="Driver Safety AI", page_icon="🚛", layout="centered")

# Qoidalarni yuklash
rules_data = load_rules()
company_name = rules_data['meta']['carrier_name']

st.title(f"🚛 {company_name}")
st.caption("AI-Powered Driver Qualification System (YAML Configured)")

# Fayl yuklash
uploaded_file = st.file_uploader("Haydovchi hisobotini yuklang (PDF)", type=["pdf"])

if uploaded_file:
    # 1. Matnni olish
    pdf_text = extract_text_from_pdf(uploaded_file)

    # 2. Tahlil tugmasi
    if st.button("🚀 Tahlilni Boshlash", type="primary"):

        with st.status("AI Agent ishlamoqda...", expanded=True) as status:
            st.write("YAML qoidalari yuklandi...")
            st.write("PDF ma'lumotlari o'qilmoqda...")
            st.write("Qoidabuzarliklar solishtirilmoqda...")

            # YAMLni string ko'rinishiga o'tkazish (Prompt uchun)
            rules_str = yaml.dump(rules_data)

            # --- PROMPT ---
            system_prompt = """
            Siz Safety Manager yordamchisiz. Vazifangiz:
            1. Berilgan YAML qoidalari (Rules) asosida Nomzod (Candidate) ma'lumotlarini tekshirish.
            2. Javobni qat'iy JSON formatida qaytarish.

            DIQQAT:
            - Agar "Hard Stops" ro'yxatidagi birorta qoidabuzarlik bo'lsa -> REJECT.
            - Agar Yosh (Age) to'g'ri kelmasa -> REJECT.
            - Agar jami "Minor" qoidabuzarliklar chegaradan (threshold) oshsa -> REJECT.

            JSON STRUKTURASI:
            {
                "status": "APPROVE" | "REJECT" | "DELAY" | "REVIEW",
                "final_score": "Qisqa xulosa (masalan: 2 ta Minor, 1 ta Major)",
                "violations_found": [
                    {"violation": "Nomi", "severity": "Major/Minor", "rule_matched": "YAMLdagi qoida"}
                ],
                "missing_docs": ["Hujjat nomi" (agar Delay bo'lsa)],
                "recruiter_action": "Recruiterga aniq ko'rsatma/Inson tilida"
            }
            """

            user_prompt = f"""
            --- QOIDALAR (YAML) ---
            {rules_str}

            --- NOMZOD MA'LUMOTI (PDF) ---
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

                # Natijani olish
                result = json.loads(response.choices[0].message.content)
                status.update(label="Tahlil yakunlandi!", state="complete", expanded=False)

                # --- VIZUALIZATSIYA ---
                decision = result.get("status", "UNKNOWN").upper()
                action = result.get("recruiter_action", "")
                violations = result.get("violations_found", [])

                st.divider()

                # Rangli Statuslar
                if decision == "REJECT":
                    st.error(f"⛔ QAROR: {decision}")
                elif decision == "APPROVE":
                    st.success(f"✅ QAROR: {decision}")
                elif decision == "DELAY":
                    st.warning(f"⚠️ QAROR: {decision} (Hujjatlar yetishmaydi)")
                elif decision == "REVIEW":
                    st.info(f"👀 QAROR: {decision} (Qo'shimcha tekshiruv kerak)")
                else:
                    st.write(f"QAROR: {decision}")

                # Tafsilotlar bo'limi
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("⚠️ Topilgan muammolar")
                    if violations:
                        for v in violations:
                            severity_icon = "🔴" if v['severity'] == "Major" else "🟡"
                            st.markdown(f"{severity_icon} **{v['violation']}**")
                            st.caption(f"Qoida: {v['rule_matched']} | Daraja: {v['severity']}")
                    else:
                        st.success("Toza! Qoidabuzarliklar topilmadi.")

                with col2:
                    st.subheader("📝 Ko'rsatma")
                    st.info(action)
                    if result.get("missing_docs"):
                        st.markdown("**Yetishmayotgan hujjatlar:**")
                        for doc in result["missing_docs"]:
                            st.markdown(f"- {doc}")

            except Exception as e:
                status.update(label="Xatolik!", state="error")
                st.error(f"Tizim xatosi: {e}")

