from __future__ import annotations
import os, functools
import gradio as gr
from transformers import pipeline
import evaluate
from langdetect import detect

# Writable cache dirs inside the HF sandbox
os.environ["HOME"] = "/tmp"
os.environ["HF_HOME"] = "/tmp/.cache/hf"
os.makedirs("/tmp/.cache/hf", exist_ok=True)

MODEL_ID   = "facebook/nllb-200-distilled-600M"   # use 418M for lighter demo
TARGET_TAG = "tur_Latn"

NLLB_MAP = {
    "af": "afr_Latn", "ar": "arb_Arab", "de": "deu_Latn", "en": "eng_Latn",
    "es": "spa_Latn", "fr": "fra_Latn", "hi": "hin_Deva", "ja": "jpn_Jpan",
    "ko": "kor_Hang", "pt": "por_Latn", "ru": "rus_Cyrl", "tr": "tur_Latn",
    "zh-cn": "zho_Hans",
}

@functools.lru_cache(maxsize=1)
def load_components():
    translator = pipeline("translation", model=MODEL_ID, tokenizer=MODEL_ID, max_length=512)
    bleu       = evaluate.load("sacrebleu")
    bertscore  = evaluate.load("bertscore")
    try:
        comet = evaluate.load("comet")
    except Exception:
        comet = None
    return translator, bleu, bertscore, comet

translator, bleu_metric, bert_metric, comet_metric = load_components()

def translate_and_score(source: str, reference: str | None):
    if not source.strip():
        return "", "–", "–", "–"

    try:
        lang_iso = detect(source[:200])
    except Exception:
        lang_iso = "en"
    src_tag = NLLB_MAP.get(lang_iso, "eng_Latn")

    translation = translator(source.strip(), src_lang=src_tag, tgt_lang=TARGET_TAG)[0]["translation_text"]

    if reference and reference.strip():
        bleu  = bleu_metric.compute(predictions=[translation], references=[[reference.strip()]])["score"]
        bert  = bert_metric.compute(
            predictions=[translation],
            references=[reference.strip()],
            lang="tr",
            model_type="dbmdz/bert-base-turkish-cased",
        )["f1"][0] * 100
        comet = "N/A"
        if comet_metric is not None:
            try:
                comet = comet_metric.compute(
                    predictions=[translation],
                    references=[reference.strip()],
                    sources=[source.strip()],
                )["scores"][0]
            except Exception:
                comet = "N/A"
        return translation, f"{bleu:.2f}", f"{bert:.2f}", f"{comet}" if isinstance(comet, str) else f"{comet:.2f}"
    else:
        return translation, "–", "–", "–"

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# Evaluator  \n"
        "Enter text in **any language**, get a Turkish translation plus automatic "
        "**BLEU**, **BERTScore (BERTurk)** and **COMET** scores."
    )

    with gr.Row():
        src_box = gr.Textbox(label="Source Text", lines=6, placeholder="Paste or type any text…")
        ref_box = gr.Textbox(
            label="Reference Turkish Translation (optional)",
            lines=6,
            placeholder="If you have a gold translation, paste it here…",
        )

    translate_btn = gr.Button("Translate & Evaluate", variant="primary")

    out_translation = gr.Textbox(label="Model Translation", lines=6)
    with gr.Row():
        bleu_out  = gr.Textbox(label="BLEU", max_lines=1)
        bert_out  = gr.Textbox(label="BERTScore F1", max_lines=1)
        comet_out = gr.Textbox(label="COMET", max_lines=1)

    translate_btn.click(
        translate_and_score,
        inputs=[src_box, ref_box],
        outputs=[out_translation, bleu_out, bert_out, comet_out],
    )

    gr.Markdown(
        "<small>Translation model: <b>facebook/nllb‑200‑distilled‑600M</b>  ·  "
        "BERTScore embeddings: <b>BERTurk</b></small>",
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    demo.launch()
