import os
import requests
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =========================================================
# Page
# =========================================================
st.set_page_config(page_title="COVID-19 CXR Detection", layout="centered")
st.title("CNN-Based COVID-19 Detection System (Chest X-ray)")
st.write("Upload a chest X-ray image. The model predicts: COVID or Normal.")

# =========================================================
# Model download settings (Google Drive)
# =========================================================
DRIVE_FILE_ID = "18zNPnB62-DvJAddb7mcVcV_A2WT0Awr7"   # ✅ your model file id
MODEL_PATH = "covid_cnn_model.keras"

# ⚠️ IMPORTANT: Must match your training order (train_ds.class_names)
# If your Colab printed: ['COVID','Normal'] keep it as below.
CLASS_NAMES = ["COVID", "Normal"]   # label0, label1

IMG_SIZE = (224, 224)

# =========================================================
# Google Drive downloader (handles big-file warning token)
# =========================================================
def download_from_gdrive(file_id: str, dest_path: str):
    url = "https://drive.google.com/uc?export=download"
    session = requests.Session()

    r = session.get(url, params={"id": file_id}, stream=True)
    r.raise_for_status()

    # Check for confirmation token (large file warning)
    token = None
    for k, v in r.cookies.items():
        if k.startswith("download_warning"):
            token = v
            break

    if token:
        r = session.get(url, params={"id": file_id, "confirm": token}, stream=True)
        r.raise_for_status()

    with open(dest_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

def ensure_model():
    # if model already downloaded and looks large enough, skip
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 5_000_000:
        return

    st.info("Downloading model from Google Drive...")
    download_from_gdrive(DRIVE_FILE_ID, MODEL_PATH)

    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1_000_000:
        st.error("❌ Model download failed or file too small. Check Drive sharing settings.")
        st.stop()

    st.success("Model downloaded ✅")

@st.cache_resource
def load_model_cached():
    ensure_model()
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model_cached()

# =========================================================
# UI
# =========================================================
st.caption(f"Class mapping: 0 → {CLASS_NAMES[0]} | 1 → {CLASS_NAMES[1]}")
min_conf = st.slider("Minimum confidence to accept prediction (%)", 50, 99, 80, 1)

uploaded = st.file_uploader("Upload chest X-ray image", type=["png", "jpg", "jpeg"])

# =========================================================
# Prediction
# =========================================================
if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    img_resized = img.resize(IMG_SIZE)
    x = np.array(img_resized)
    x = np.expand_dims(x, axis=0)   # ✅ NO /255 because the model has Rescaling inside

    # Model output: probability of class 1 (CLASS_NAMES[1])
    p1 = float(model.predict(x, verbose=0)[0][0])
    p0 = 1.0 - p1

    # Choose the higher probability (more stable than threshold on p1 only)
    pred_idx = 1 if p1 > p0 else 0
    pred_name = CLASS_NAMES[pred_idx]
    conf = (p1 if pred_idx == 1 else p0) * 100

    st.subheader("Prediction")
    st.write(f"Result: **{pred_name}**")
    st.write(f"Confidence: **{conf:.2f}%**")

    st.write("Probabilities")
    st.write(f"- P({CLASS_NAMES[0]}): {p0:.4f}")
    st.write(f"- P({CLASS_NAMES[1]}): {p1:.4f}")

    if conf < min_conf:
        st.warning("⚠️ Low confidence prediction. Try a clearer X-ray image or different view.")

st.warning("Educational use only — not a clinical diagnosis tool.")

