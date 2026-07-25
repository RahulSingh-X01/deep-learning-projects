import base64
from io import BytesIO
from datetime import datetime

import streamlit as st
from PIL import Image

import sys
import os

# Get the project root (one level up from this file's folder)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from inference import load_pipeline, CLASS_NAMES
from src.predict import predict

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🔬",
    layout="centered",
)

# ----------------------------------------------------------------------
# Load model once, cache across reruns (Streamlit reruns the whole
# script on every interaction, so this avoids reloading the model
# every time someone uploads a new image)
# ----------------------------------------------------------------------
@st.cache_resource
def get_pipeline():
    return load_pipeline()


def image_to_base64(img):
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


# ----------------------------------------------------------------------
# Styling — "Field Specimen Log" theme
#   parchment background, typewriter display font, mono data font,
#   taped photograph, rubber-stamp verdict, ruler-style confidence gauge
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Special+Elite&family=IBM+Plex+Mono:wght@400;600&family=Source+Sans+3:wght@400;600&display=swap');

    :root {
        --paper: #EFE6D5;
        --paper-line: #DCCFB4;
        --ink: #24313D;
        --ink-soft: #5B6672;
        --cat-color: #4C7A72;
        --dog-color: #A05C36;
        --stamp-red: #9C3D3D;
    }

    .stApp {
        background-color: var(--paper);
        background-image:
            repeating-linear-gradient(
                0deg, var(--paper-line) 0px, var(--paper-line) 1px,
                transparent 1px, transparent 32px
            );
        font-family: 'Source Sans 3', sans-serif;
        color: var(--ink);
    }

    /* Hide default streamlit chrome we don't want */
    #MainMenu, footer, header {visibility: hidden;}

    .eyebrow {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: var(--ink-soft);
        border-bottom: 1px dashed var(--ink-soft);
        display: inline-block;
        padding-bottom: 4px;
        margin-bottom: 6px;
    }

    .journal-title {
        font-family: 'Special Elite', monospace;
        font-size: 2.4rem;
        color: var(--ink);
        margin: 0 0 0.2rem 0;
        line-height: 1.1;
    }

    .journal-subtitle {
        font-size: 0.95rem;
        color: var(--ink-soft);
        margin-bottom: 1.8rem;
    }

    /* Uploader box */
    [data-testid="stFileUploaderDropzone"] {
        background-color: #F7F1E4;
        border: 2px dashed var(--ink-soft) !important;
        border-radius: 2px;
    }

    /* Photo card - taped polaroid look */
    .photo-frame {
        background: #FBF8F1;
        padding: 14px 14px 46px 14px;
        box-shadow: 0 6px 16px rgba(36,49,61,0.25);
        transform: rotate(-2deg);
        position: relative;
        margin: 20px auto 10px auto;
        max-width: 340px;
    }
    .photo-frame img {
        width: 100%;
        display: block;
        filter: sepia(8%) contrast(1.02);
    }
    .washi-tape {
        position: absolute;
        top: -14px;
        left: 50%;
        transform: translateX(-50%) rotate(-3deg);
        width: 90px;
        height: 26px;
        background: rgba(160, 92, 54, 0.55);
        border: 1px solid rgba(160, 92, 54, 0.3);
    }
    .photo-caption {
        text-align: center;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        color: var(--ink-soft);
        margin-top: 8px;
    }

    /* Notes card */
    .notes-card {
        background: #FBF8F1;
        border: 1px solid var(--paper-line);
        box-shadow: 3px 3px 0 rgba(36,49,61,0.1);
        padding: 24px 26px 28px 26px;
        margin-top: 22px;
        position: relative;
        overflow: hidden;
    }

    .notes-heading {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--ink-soft);
        margin-bottom: 14px;
    }

    /* Stamp */
    @keyframes stampDown {
        0%   { transform: scale(2.2) rotate(-10deg); opacity: 0; }
        60%  { transform: scale(0.92) rotate(-8deg); opacity: 1; }
        80%  { transform: scale(1.06) rotate(-8deg); }
        100% { transform: scale(1) rotate(-8deg); opacity: 1; }
    }
    .stamp {
        display: inline-block;
        font-family: 'Special Elite', monospace;
        font-size: 2.6rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        padding: 6px 22px;
        border: 5px solid currentColor;
        border-radius: 6px;
        transform: rotate(-8deg);
        animation: stampDown 0.6s ease-out;
        margin: 6px 0 18px 0;
    }
    .stamp.cat { color: var(--cat-color); }
    .stamp.dog { color: var(--dog-color); }
    .stamp::after {
        content: "";
        position: absolute;
    }

    /* Ruler-style confidence gauge */
    .ruler-label {
        display: flex;
        justify-content: space-between;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        color: var(--ink-soft);
        margin-bottom: 4px;
    }
    .ruler-track {
        position: relative;
        height: 26px;
        background:
            repeating-linear-gradient(
                90deg,
                var(--ink-soft) 0px, var(--ink-soft) 1px,
                transparent 1px, transparent 10%
            ),
            #E7DEC9;
        border: 1px solid var(--ink-soft);
        overflow: hidden;
    }
    @keyframes growFill {
        from { width: 0%; }
        to   { width: var(--final-width); }
    }
    .ruler-fill {
        height: 100%;
        animation: growFill 0.9s ease-out forwards;
        opacity: 0.85;
    }
    .ruler-fill.cat { background: var(--cat-color); }
    .ruler-fill.dog { background: var(--dog-color); }

    .confidence-number {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 10px;
    }
    .confidence-number.cat { color: var(--cat-color); }
    .confidence-number.dog { color: var(--dog-color); }

    .log-footer {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        color: var(--ink-soft);
        margin-top: 14px;
        border-top: 1px dashed var(--paper-line);
        padding-top: 8px;
    }

    .empty-state {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
        color: var(--ink-soft);
        text-align: center;
        padding: 40px 20px;
        border: 1px dashed var(--ink-soft);
        margin-top: 20px;
        background: rgba(255,255,255,0.3);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.markdown('<div class="eyebrow">Field Log · Entry 001</div>', unsafe_allow_html=True)
st.markdown('<div class="journal-title">Specimen Classifier</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="journal-subtitle">Upload a photograph. The model examines it and logs its verdict below.</div>',
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Upload
# ----------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Drop a photograph here",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

if uploaded_file is None:
    st.markdown(
        '<div class="empty-state">No specimen loaded yet.<br>Upload a photograph above to begin the examination.</div>',
        unsafe_allow_html=True,
    )
else:
    image = Image.open(uploaded_file).convert("RGB")
    img_b64 = image_to_base64(image)

    # --- Photo card ---
    st.markdown(
        f"""
        <div class="photo-frame">
            <div class="washi-tape"></div>
            <img src="data:image/png;base64,{img_b64}" />
            <div class="photo-caption">{uploaded_file.name}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model, transform, device = get_pipeline()
    with st.spinner("Examining specimen..."):
        label, confidence = predict(model, image, transform, CLASS_NAMES, device)

    css_class = "cat" if label == "Cat" else "dog"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # --- Notes card with stamp + ruler gauge ---
    st.markdown(
        f"""
        <div class="notes-card">
            <div class="notes-heading">Examiner's Verdict</div>
            <div class="stamp {css_class}">{label.upper()}</div>
            <div class="ruler-label">
                <span>Confidence</span>
                <span>{confidence:.1f}%</span>
            </div>
            <div class="ruler-track">
                <div class="ruler-fill {css_class}" style="--final-width: {confidence:.1f}%; width: {confidence:.1f}%;"></div>
            </div>
            <div class="log-footer">Logged {timestamp} · model: cat-vs-dog-classifier</div>
        </div>
        """,
        unsafe_allow_html=True,
    )