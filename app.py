import os
import requests
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="COVID-19 CXR Detection", layout="centered")
st.title("CNN-Based COVID-19 Detection System (Chest X-ray)")
st.write("Upload a chest X-ray image. The model predicts: COVID or Normal.")
st.caption("⚠️ Educational use only — not a clinical diagnosis tool.")

# ====== SETTINGS ======
DRIVE_FILE_ID = "1ffyT6-lb19VUCbGCal0eN_qeNGZz9KR4"  # or your other file id
MODEL_PATH = "/tmp/covid_model.keras"
CLASS_NAMES = ["COVID", "Normal"]
IMG_SIZE = (224, 224)
MIN_BYTES = 50 * 1024 * 1024  # 50MB

def download_from_gdrive(file_id: str, dest_path: str):
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()

    r = session.get(URL, params={"id": file_id}, stream=True)
    r.raise_for_status()

    # confirm token (for large files)
    token = None
    for k, v in r.cookies.items():
        if k.startswith("download_warning"):
            token = v
            break

    if token:
        r = session.get(URL, params={"id": file_id, "confirm": token}, stream=True)
        r.raise_for_status()

    with open(dest_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

def ensure_model_exists():
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > MIN_BYTES:
        return

    st.info("Downloading model from Google Drive (first time only)...")
    try:
        download_from_gdrive(DRIVE_FILE_ID, MODEL_PATH)
    except Exception as e:
        st.error(f"❌ Download failed: {e}")
        st.stop()

    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < MIN_BYTES:
        size = os.path.getsize(MODEL_PATH) if os.path.exists(MODEL_PATH) else 0
        st.error(f"❌ Model file too small after download ({size} bytes). Check Drive sharing settings.")
        st.stop()

    st.success(f"✅ Model ready ({os.path.getsize(MODEL_PATH)/(1024*1024):.2f} MB)")

@st.cache_resource(show_spinner="Loading model (first time only)...")
def load_model():
    ensure_model_exists()
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

def preprocess_image(pil_img: Image.Image):
    pil_img = pil_img.convert("RGB").resize(IMG_SIZE)
    x = np.array(pil_img, dtype=np.float32)
    x = np.expand_dims(x, axis=0)
    return x  # no /255 if model has Rescaling inside

threshold = st.slider("Decision threshold (for class 1)", 0.10, 0.90, 0.50, 0.01)
uploaded = st.file_uploader("Upload chest X-ray image", type=["png", "jpg", "jpeg"])

if uploaded is None:
    st.info("⬆️ Upload an image to start prediction (model loads after upload).")
else:
    model = load_model()
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    x = preprocess_image(img)
    p1 = float(model.predict(x, verbose=0)[0][0])  # P(label=1)
    p0 = 1.0 - p1

    pred_idx = 1 if p1 >= threshold else 0
    pred_name = CLASS_NAMES[pred_idx]
    conf = (p1 if pred_idx == 1 else p0) * 100.0

    st.subheader("Prediction")
    st.write(f"Result: **{pred_name}**")
    st.write(f"Confidence: **{conf:.2f}%**")
    st.write(f"P({CLASS_NAMES[0]}): {p0:.4f}")
    st.write(f"P({CLASS_NAMES[1]}): {p1:.4f}")
