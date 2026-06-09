# 🚛 Driver Safety AI — AI-Powered Driver Qualification System

An intelligent hiring decision tool for trucking companies. Upload a driver's MVR (Motor Vehicle Record) PDF, and the AI instantly analyzes it against your company's hiring rules — returning a clear **APPROVE / REJECT / DELAY / REVIEW** decision with full reasoning.

---

## What It Does

- **Instant MVR Analysis** — Upload a driver's PDF report and get a hiring decision in seconds
- **Rule-Based Compliance** — All decisions follow FMCSA regulations and your custom insurance underwriting rules defined in `hiring_rules.yaml`
- **Clear Violation Breakdown** — Shows every violation found, its severity (Major / Minor), and which rule was triggered
- **Actionable Recruiter Instructions** — Tells your recruiter exactly what to do next
- **Interactive Chat** — Ask follow-up questions about the report (e.g., "How many years of experience does this driver have?")

---

## Decision Outcomes

| Status | Meaning |
|--------|---------|
| ✅ **APPROVE** | Driver meets all requirements — eligible to hire |
| ⛔ **REJECT** | Hard stop violation found — do not hire |
| ⚠️ **DELAY** | Missing documents (expired med card, endorsement) — request and re-review |
| 👀 **REVIEW** | Borderline case — Safety Director must make final call |

---

## Hiring Rules Summary

Rules live in `hiring_rules.yaml` and are fully customizable.

**Age Policy**
- Under 21 → **REJECT** (FMCSA interstate prohibition)
- 21–24 → **REJECT** (insurance minimum age is 25)
- 25–65 → **APPROVED** age range
- Over 65 → **REVIEW** (requires recent medical certificate + insurance approval)

**Hard Stop Violations (auto-REJECT)**
- DUI / DWI / OWI
- Reckless or careless driving
- Speeding 11+ mph over the limit
- Hit and Run
- ELD / log falsification
- HOS core violations (11h, 14h, 60h, 70h)
- Handheld cellphone use while driving
- Following too closely / tailgating

**Minor Violations**
- Speeding 1–10 mph over → allowed up to **2** within 3-year lookback; 3+ → REJECT

**Experience Policy**
- Under 6 months CDL experience → **REJECT**
- 6–11 months → **REVIEW** (zero-tolerance window; any moving violation = REJECT)
- 12–23 months → **Acceptable** (max 1 minor violation)
- 24+ months → **Preferred** (standard thresholds apply)

---

## Project Structure

```
hiring_decision_project/
├── main.py               # Streamlit app — UI + AI analysis + chat
├── hiring_rules.yaml     # All hiring rules (editable, no code required)
├── requirements.txt      # Python dependencies
└── README.md
```

---

## Setup & Installation

### 1. Clone or download the project

```bash
git clone <repo-url>
cd hiring_decision_project
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Alternatively, install manually:

```bash
pip install streamlit openai PyMuPDF pyyaml
```

### 3. Set your OpenAI API key

Open `main.py` and replace the API key value on line 9–10 with your own key.

> **Security tip:** For production use, store the key in an environment variable or `.env` file instead of hardcoding it.

### 4. Run the app

```bash
streamlit run main.py
```

The app will open at `http://localhost:8501`.

---

## How to Use

1. **Upload** a driver's MVR PDF using the file uploader
2. Click **"Start Analysis"** — the AI reads the report and checks it against all rules
3. View the **decision** (APPROVE / REJECT / DELAY / REVIEW) and the list of violations found
4. Read the **Recruiter Instructions** — a plain-English action item telling you exactly what to do
5. Use the **chat box** to ask specific questions about the driver's record

---

## Customizing the Rules

You do not need to touch any code to update hiring rules. Simply edit `hiring_rules.yaml`:

- **Add a new violation keyword** → add it under `keywords_mapping`
- **Change the minor violation threshold** → edit `thresholds.minor_moving_max`
- **Update the lookback period** → edit `thresholds.lookback_years`
- **Change age requirements** → edit `age_policy.rules`
- **Update experience thresholds** → edit `experience_policy.thresholds`

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| UI / Frontend | [Streamlit](https://streamlit.io) |
| AI Engine | OpenAI GPT (via `openai` Python SDK) |
| PDF Parsing | PyMuPDF (`fitz`) |
| Rules Config | YAML (`pyyaml`) |
| Language | Python 3.9+ |

---

## Requirements

```
streamlit
openai
PyMuPDF
pyyaml
```

---

## Company Context

Built for **Bipolar Bear Enterprises LLC** — Owner-Operator hiring for Dry Van / Power Only operations.  
Compliance basis: FMCSA regulations + insurance underwriting requirements.

---

## Notes

- The AI reads raw PDF text and matches it against YAML rules. Make sure the uploaded PDF is text-based, not a scanned image. For scanned PDFs, OCR pre-processing is required.
- The chat feature is context-aware — it only answers questions based on the uploaded report and analysis result.
- All decisions should be reviewed by a qualified Safety Manager before final action. This system is a decision-support tool, not a replacement for human judgment.
