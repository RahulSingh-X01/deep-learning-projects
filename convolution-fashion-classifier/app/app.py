import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

import streamlit as st
from PIL import Image
from inference import Inference

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title = "Fashion Classifier",
    page_icon  = "👗",
    layout     = "centered"
)

# ── Title ─────────────────────────────────────────────────────
st.title("👗 Fashion MNIST Classifier")
st.write("Upload an image and the model will predict what fashion item it is.")

# ── Load Model — cached so it loads only once ─────────────────
@st.cache_resource
def load_model():
    return Inference()

inferencer = load_model()

# ── File Uploader ──────────────────────────────────────────────
uploaded_file = st.file_uploader(
    label      = "Choose an image",
    type       = ["jpg", "jpeg", "png"]
)

# ── Prediction ────────────────────────────────────────────────
if uploaded_file is not None:

    # Show uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)

    # Save temporarily so Inference can read it from path
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Run prediction
    with st.spinner("Classifying..."):
        result = inferencer.predict(temp_path)

    # Show results
    st.success("Done!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            label = "Predicted Class",
            value = result['class_name']
        )
    
    with col2:
        st.metric(
            label = "Confidence",
            value = result['confidence']
        )

    # Clean up temp file
    import os
    os.remove(temp_path)