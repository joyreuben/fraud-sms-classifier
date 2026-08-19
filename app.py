import html
import math
import re
from pathlib import Path

import joblib
import streamlit as st
from scipy.sparse import csr_matrix, hstack

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="Fraud SMS Classifier", page_icon="🛡️", layout="centered")


@st.cache_resource
def load_pipeline():
    vectorizer = joblib.load(BASE_DIR / "tfidf_vectorizer.joblib")
    model = joblib.load(BASE_DIR / "final_svm_model.joblib")
    return vectorizer, model


vectorizer, model = load_pipeline()


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " urltoken ", text)
    text = re.sub(r"[£$€₦]|(\bn\d{2,}\b)", " moneytoken ", text)
    text = re.sub(r"\b\d{7,}\b", " phonetoken ", text)
    text = re.sub(r"\b\d+\b", " numtoken ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ----------------------------------------------------------------
# Styling
# ----------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=IBM+Plex+Sans:wght@400;500&family=IBM+Plex+Mono:wght@400;500&display=swap');

    :root {
        --bg-panel: #FFFFFF;
        --bg-input: #F6F5FC;
        --accent-primary: #5B4FE9;
        --accent-primary-dark: #4638C9;
        --accent-alert: #FF5A5F;
        --accent-safe: #16C79A;
        --text-primary: #1B1A2E;
        --text-muted: #6B6B85;
        --border-subtle: #E1E0F2;
    }

    .stApp {
        background: linear-gradient(180deg, #F5F3FF 0%, #ECEFFB 100%);
        font-family: 'IBM Plex Sans', sans-serif;
        color: var(--text-primary);
    }

    header[data-testid="stHeader"] {
        background-color: transparent !important;
    }

    .eyebrow {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        letter-spacing: 0.15em;
        color: var(--accent-primary);
        text-transform: uppercase;
        margin-bottom: 0.4rem;
        font-weight: 500;
    }
    .page-title {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 2.2rem;
        margin: 0 0 0.3rem 0;
        color: var(--text-primary);
    }
    .page-subtitle {
        color: var(--text-muted);
        font-size: 0.95rem;
        margin-bottom: 1.8rem;
    }

    /* Scan panel */
    .scan-panel {
        background: var(--bg-panel);
        border-radius: 16px;
        overflow: hidden;
        margin-bottom: 1.2rem;
        box-shadow: 0 8px 24px rgba(91, 79, 233, 0.08);
        border: 1px solid var(--border-subtle);
    }
    .scan-panel-strip {
        height: 5px;
        background: linear-gradient(90deg, var(--accent-primary) 0%, var(--accent-safe) 100%);
    }
    .scan-panel-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.9rem 1.3rem;
        border-bottom: 1px solid var(--border-subtle);
    }
    .scan-panel-label {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 0.9rem;
        color: var(--text-primary);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .pulse-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--accent-safe);
        display: inline-block;
        box-shadow: 0 0 0 rgba(22, 199, 154, 0.5);
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(22, 199, 154, 0.5); }
        70% { box-shadow: 0 0 0 8px rgba(22, 199, 154, 0); }
        100% { box-shadow: 0 0 0 0 rgba(22, 199, 154, 0); }
    }
    .scan-panel-tag {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        color: var(--accent-primary);
        background: var(--bg-input);
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
    }
    .scan-panel-body {
        padding: 1.2rem 1.3rem 1.4rem 1.3rem;
    }

    [data-testid="stTextArea"] textarea {
        background-color: var(--bg-input) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 12px !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-size: 0.95rem !important;
        padding: 0.9rem !important;
    }
    [data-testid="stTextArea"] textarea:focus {
        border-color: var(--accent-primary) !important;
        box-shadow: 0 0 0 2px rgba(91, 79, 233, 0.15) !important;
    }
    [data-testid="stTextArea"] label {
        display: none;
    }

    .stButton > button {
        background: var(--accent-primary) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 10px !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 700 !important;
        padding: 0.6rem 1.5rem !important;
        letter-spacing: 0.02em;
    }
    .stButton > button:hover {
        background: var(--accent-primary-dark) !important;
        color: #FFFFFF !important;
    }

    /* Result card */
    .result-card {
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        margin-top: 1rem;
        background: var(--bg-panel);
        border: 1px solid var(--border-subtle);
        box-shadow: 0 8px 24px rgba(27, 26, 46, 0.06);
    }
    .result-verdict {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 1.15rem;
        margin-bottom: 0.7rem;
    }
    .result-verdict.safe { color: var(--accent-safe); }
    .result-verdict.alert { color: var(--accent-alert); }

    .risk-meter-track {
        width: 100%;
        height: 10px;
        background: var(--bg-input);
        border-radius: 6px;
        overflow: hidden;
        margin-bottom: 0.5rem;
    }
    .risk-meter-fill {
        height: 100%;
        border-radius: 6px;
    }
    .risk-meter-fill.safe { background: var(--accent-safe); }
    .risk-meter-fill.alert { background: var(--accent-alert); }

    .result-meta {
        display: flex;
        justify-content: space-between;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        color: var(--text-muted);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------
# Header
# ----------------------------------------------------------------
st.markdown('<div class="eyebrow">Fraud Detection System · 3MTT AI/ML Project</div>', unsafe_allow_html=True)
st.markdown('<div class="page-title">Fraud SMS Classifier</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-subtitle">Paste an incoming SMS message to scan it for fraud patterns.</div>',
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------
# Scan panel
# ----------------------------------------------------------------
st.markdown(
    """
    <div class="scan-panel">
        <div class="scan-panel-strip"></div>
        <div class="scan-panel-header">
            <span class="scan-panel-label"><span class="pulse-dot"></span> Scan panel</span>
            <span class="scan-panel-tag">Linear SVM · TF-IDF</span>
        </div>
        <div class="scan-panel-body">
    """,
    unsafe_allow_html=True,
)

message = st.text_area("SMS message", height=110, label_visibility="collapsed", placeholder="Paste the SMS text here...")
scan_clicked = st.button("Scan message")

st.markdown("</div></div>", unsafe_allow_html=True)


# ----------------------------------------------------------------
# Result
# ----------------------------------------------------------------
if scan_clicked:
    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        cleaned = clean_text(message)
        tfidf_features = vectorizer.transform([cleaned])
        length_feature = csr_matrix([[len(message)]])
        features = hstack([tfidf_features, length_feature])

        prediction = model.predict(features)[0]
        score = model.decision_function(features)[0]

        # Squash the raw SVM margin score into a 0-100 display range
        # purely for the visual meter. This is NOT a calibrated
        # probability, it's a rough visual read on confidence.
        risk_pct = 1 / (1 + math.exp(-score)) * 100
        risk_pct = max(2, min(98, risk_pct))  # keep the bar visible at extremes

        verdict_class = "alert" if prediction == 1 else "safe"
        verdict_text = "⚠️ Flagged as fraud / spam" if prediction == 1 else "✅ Looks like a legitimate message"

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-verdict {verdict_class}">{verdict_text}</div>
                <div class="risk-meter-track">
                    <div class="risk-meter-fill {verdict_class}" style="width:{risk_pct:.0f}%;"></div>
                </div>
                <div class="result-meta">
                    <span>risk score: {risk_pct:.0f}%</span>
                    <span>raw margin: {score:.2f}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
