# app.py – Streamlit app for translating any language to Turkish and evaluating with BLEU, BERTScore, COMET
# ------------------------------------------------------------------------------
# Requirements (add these lines in a requirements.txt file in the same repo):
# streamlit>=1.33.0
# transformers>=4.40.0
# evaluate>=0.4.1
# torch  # or accelerate if using CPU-only runtime
# sentencepiece  # tokenizer dependency
# sacrebleu
# bert_score
# comet-ml  # for COMET metric model download (GPU recommended)
# ------------------------------------------------------------------------------
# On Streamlit Community Cloud, set an environment variable HF_HOME="/tmp/huggingface" 
# to ensure model weights are cached in ephemeral storage.

import streamlit as st
from transformers import pipeline
import evaluate

# ---------------------------------------------------------------
# Cached model & metric loaders – run only once per session
# ---------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_components():
    """Download/instantiate translator + evaluation metrics once."""
    # Multilingual → Turkish MarianMT model (compact 85 MB)
    translator = pipeline(
        "translation",
        model="Helsinki-NLP/opus-mt-mul-tr",
        tokenizer="Helsinki-NLP/opus-mt-mul-tr",
        max_length=512
    )

    bleu = evaluate.load("sacrebleu")  # faster & accurate BLEU
    bertscore = evaluate.load("bertscore")
    comet = evaluate.load("comet")  # downloads ~500 MB model the first time

    return translator, bleu, bertscore, comet

translator, bleu_metric, bert_metric, comet_metric = load_components()

# ---------------------------------------------------------------
# Streamlit page config & custom CSS for a sleek, minimal look
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Türkçe Çeviri + Değerlendirme",
    page_icon="🇹🇷",
    layout="centered"
)

st.markdown(
    """
    <style>
        html, body {font-family: 'Inter', sans-serif; background:#ffffff; color:#111827;}
        .stTextArea textarea {font-size:0.9rem; line-height:1.4;}
        div.stButton > button:first-child {
            background-color:#1f2937; color:white; border:none; border-radius:8px;
            padding:0.6em 1.2em; font-weight:600; transition: background 0.3s ease;
        }
        div.stButton > button:first-child:hover {background:#374151;}
        div[data-testid="stMetric"] {background:#f9fafb; border-radius:8px; padding:1em; box-shadow:0 1px 3px rgba(0,0,0,0.06);} 
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------------
# App UI
# ---------------------------------------------------------------
st.title("Translate to Turkish & Auto‑Evaluate")

st.write("Paste any sentence below, click **Translate**, and (optionally) supply a reference Turkish translation to get BLEU, BERTScore, and COMET quality scores.")

source_text = st.text_area("🌐 Source text (any language)", height=140, placeholder="Type or paste here…")
reference_text = st.text_area("🎯 Reference translation in Turkish (optional)", height=140, placeholder="If you have a gold translation, paste it here for scoring…")

if st.button("Translate"):
    if not source_text.strip():
        st.warning("Please enter some text to translate.")
    else:
        with st.spinner("Translating…"):
            translation = translator(source_text.strip())[0]["translation_text"]
        st.text_area("🚀 Model translation", value=translation, height=140)

        # Evaluation block only if reference is given
        if reference_text.strip():
            with st.spinner("Calculating scores…"):
                bleu_score = bleu_metric.compute(predictions=[translation], references=[[reference_text.strip()]])
                bert_score = bert_metric.compute(predictions=[translation], references=[reference_text.strip()], lang="tr")
                comet_score = comet_metric.compute(predictions=[translation], references=[reference_text.strip()], sources=[source_text.strip()])

            col1, col2, col3 = st.columns(3)
            col1.metric("BLEU", f"{bleu_score['score']:.2f}")
            col2.metric("BERTScore F1", f"{bert_score['f1'][0]*100:.2f}")
            col3.metric("COMET", f"{comet_score['scores'][0]:.2f}")
        else:
            st.info("Add a reference translation to compute quality metrics.")

st.markdown("— Made with ❤️ using Streamlit, Hugging Face Transformers & Evaluate —")
