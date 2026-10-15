import hashlib
import io
import os
import time

import streamlit as st
from google import genai
from google.genai import errors, types

try:
    from pypdf import PdfReader
except ImportError:
    from PyPDF2 import PdfReader

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
MODELS = ["gemini-3.8-flash", "gemini-3.5-flash", "gemini-3.1-flash-lite"]
MAX_CHARS = 30_000

PERSONAS = [
    "Director of Technical Account Management (TAM)",
    "Global Head / Director of Customer Success (CS)",
    "Director of Technical Support Operations & Escalation Engineering",
]

GROUND_RULE = (
    "Use ONLY facts present in the text provided. Never invent employers, titles, dates, "
    "metrics, tools or certifications. Where something would strengthen the story but is "
    "missing, write it as [ADD IF TRUE: ...] instead of making it up."
)

st.set_page_config(page_title="Zycus Growth Suite", layout="wide")
st.title("🎯 Zycus Executive Growth Suite & AI Telemetry Engine")
st.caption("Tiered matrix, Thomas PPA-style aligner & Source-to-Pay churn tracker")

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def secret_key() -> str:
    try:
        return st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        return ""

@st.cache_resource
def get_client(key: str):
    return genai.Client(api_key=key)

def build_config():
    try:
        return types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="low")
        )
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def extract_pdf_text(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    return "\n".join((page.extract_text() or "") for page in reader.pages).strip()

def stream_ai(client, prompt: str):
    last_error = "unknown error"
    for model in MODELS:
        started = False
        try:
            for chunk in client.models.generate_content_stream(
                model=model, contents=prompt, config=build_config()
            ):
                if chunk.text:
                    started = True
                    yield chunk.text
            if started:
                return
            last_error = f"{model}: empty or blocked response"
        except errors.APIError as e:
            msg = str(e)
            if e.code in (401, 403) or "API key" in msg:
                raise RuntimeError("Gemini rejected the API key. Re-check it in the sidebar.")
            if started:
                yield "\n\n⚠️ _Connection dropped mid-answer. Click the button again._"
                return
            last_error = f"{model}: {e.code} {getattr(e, 'message', msg)}"
            time.sleep(0.5)
        except Exception as e:
            if started:
                yield "\n\n⚠️ _Connection dropped mid-answer. Click the button again._"
                return
            last_error = f"{model}: {e}"
    raise RuntimeError(f"All models failed. Last error: {last_error}")

def render_task(client, key: str, label: str, prompt: str, download_name=None):
    state_key = f"result_{key}"
    fingerprint = hashlib.md5(prompt.encode()).hexdigest()

    if st.button(label, key=f"btn_{key}", use_container_width=True):
        status = st.empty()
        status.caption("⏳ Generating...")
        try:
            result = st.write_stream(stream_ai(client, prompt))
            st.session_state[state_key] = {"fp": fingerprint, "text": result}
        except RuntimeError as e:
            st.error(str(e))
        finally:
            status.empty()
    elif state_key in st.session_state:
        saved = st.session_state[state_key]
        if saved["fp"] != fingerprint:
            st.warning("Inputs changed since this was generated. Click the button to refresh.")
        st.markdown(saved["text"])

    saved = st.session_state.get(state_key)
    if download_name and saved:
        st.download_button(
            "💾 Download Framework Data",
            data=saved["text"],
            file_name=download_name,
            mime="text/markdown",
            key=f"dl_{key}",
            use_container_width=True,
        )

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("🔑 Control Panel")
    api_key = (
        st.text_input("Gemini API key:", type="password")
        or secret_key()
        or os.getenv("GEMINI_API_KEY", "")
    )
    target_persona = st.selectbox("Target profile track:", PERSONAS)
    candidate_name = st.text_input("Candidate name:", value="Madhu Poonacha")

if not api_key:
    st.info("⚠️ Enter your Gemini API key in the sidebar to activate the dashboard.")
    st.stop()

client = get_client(api_key)

# ----------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------
st.subheader("📋 Profile baseline")
col_a, col_b = st.columns(2)
uploaded_file = col_a.file_uploader("Upload profile PDF:", type=["pdf"])
raw_text = col_b.text_area("OR paste profile text / metrics:", height=170)
target_company = st.text_input(
    "Target company / framework:", value="Zycus (Enterprise SaaS/AI Procurement)"
)

text = ""
if uploaded_file:
    try:
        text = extract_pdf_text(uploaded_file.getvalue())
        if not text:
            st.warning("No text found in the PDF. Paste the text instead.")
    except Exception as e:
        st.error(f"Could not read the PDF: {e}")
if not text:
    text = raw_text.strip()
if len(text) > MAX_CHARS:
    st.caption(f"Input trimmed to the first {MAX_CHARS:,} characters for speed.")
    text = text[:MAX_CHARS]
if text:
    st.caption(f"✅ {len(text):,} characters loaded")

# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🚀 Optimizer & Exporter",
        "🧠 Thomas PPA-style Aligner",
        "🎯 Interview Simulator",
        "🕵️ Zycus Silent Churn EWS Tracker",
    ]
)

NEED_TEXT = "💡 Upload or paste profile text above to use this tab."

with tab1:
    st.markdown(f"### Optimization focus: {target_persona}")
    if not text:
        st.info(NEED_TEXT)
    else:
        c1, c2 = st.columns(2)
        with c1:
            render_task(
                client,
                "analyze",
                "Analyze Structure",
                f"Recruiter check for '{target_persona}' at '{target_company}'. Score out of 100, with a Markdown table checking metrics and tracks. {GROUND_RULE}\n\nText:\n{text}",
            )
        with c2:
            render_task(
                client,
                "rewrite",
                "✨ Auto-Rewrite Profile",
                f"Rewrite this text for '{target_persona}' at '{target_company}'. Name: {candidate_name}. Location: Bengaluru, India. Fix timelines, maximize enterprise systems architecture (SAP, Oracle, NetSuite), separate strategic value from support labor, and include compliance (SOC2, GDPR). {GROUND_RULE}\n\nText:\n{text}",
                download_name="Optimized_Profile.md"
            )

with tab2:
    st.markdown("### 🧠 Thomas PPA Behavioral Assessment")
    if not text:
        st.info(NEED_TEXT)
    else:
        render_task(
            client,
            "ppa",
            "Run Behavioral Alignment Check",
            f"Run a Thomas PPA psychometric audit on this text. Target: High D, High I executive profile. 1. Estimate D, I, S, C percentages. 2. Verdict on if it passes 'D-I Heavy'. 3. List weak terms. 4. Give 3 High-Dominance conversions. {GROUND_RULE}\n\nText:\n{text}"
        )

with tab3:
    st.markdown("### 🎯 Executive Live Interview Simulator")
    if not text:
        st.info(NEED_TEXT)
    else:
        render_task(
            client,
            "interview",
            "Generate Interview Simulation Prep",
            f"You are the VP of Global Post-Sales Strategy at '{target_company}'. Generate 3 tough executive behavioral questions for a '{target_persona}' candidate based on their background. Include the core question, the hidden trait, and a High-Dominance Answer Blueprint. {GROUND_RULE}\n\nText:\n{text}"
        )

with tab4:
    st.markdown("### 🕵️ Specialized Zycus Merlin AI & Source-to-Pay Early Warning System")
    st.caption("Strategic Telemetry Playbook Designed to Isolate Unseen Technical and Cognitive Drift Across Procurement Landscapes")
    if not text:
        st.info(NEED_TEXT)
    else:
        render_task(
            client,
            "zycus_ews",
            "Generate Zycus Proactive Account Recovery Playbook",
            f"You are the VP of Global Post-Sales Architecture at Zycus. Develop a comprehensive technical strategy blueprint and playbook to track, flag, and eradicate 'Silent Churn' across key enterprise accounts. Structure output into: 1. ZYCUS COGNITIVE DRIFT AUDIT (Merlin AI platform). 2. CORE ERP INTEGRATION METRICS TELEMETRY (SAP S/4HANA, Oracle Cloud, NetSuite). 3. ZYCUS SPECIFIC EARLY WARNING CRITERIA GRID (Markdown table). 4. EXEC C-SUITE ALIGNMENT RETENTION PLAYBOOK (CPOs/CIOs alignment). {GROUND_RULE}\n\nText:\n{text}",
            download_name="Zycus_EWS_Playbook.md"
        )

