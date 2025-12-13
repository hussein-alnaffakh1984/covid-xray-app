import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown

st.set_page_config(page_title="COVID-19 CXR Detection", layout="centered")
st.title("CNN-Based COVID-19 Detection System (Chest X-ray)")
st.write("Upload a chest X-ray image. The model predicts: COVID or Normal.")
st.caption("⚠️ Educational use only — not a clinical diagnosis tool.")

# =========================
# SETTINGS
# =========================
DRIVE_FILE_ID = "1ffyT6-lb19VUCbGCal0eN_qeNGZz9KR4"
MODEL_PATH = "/tmp/covid_mobilenetv2_infer.keras"   # ✅ faster + stable on Streamlit Cloud
CLASS_NAMES = ["COVID", "Normal"]
IMG_SIZE = (224, 224)
MIN_BYTES = 50 * 1024 * 1024  # 50MB


def ensure_model_exists():
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > MIN_BYTES:
        return

    st.info("Downloading model from Google Drive (first time only)...")
    url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
    out = gdown.download(url, MODEL_PATH, quiet=False)

    if out is None or (not os.path.exists(MODEL_PATH)):
        st.error("❌ Model download failed. Check Drive sharing: Anyone with the link -> Viewer.")
        st.stop()

    size = os.path.getsize(MODEL_PATH)
    if size < MIN_BYTES:
        st.error(
            f"❌ Downloaded file too small ({size/(1024*1024):.2f} MB). "
            "Drive may have returned an HTML/virus-check page. Re-share and try again."
        )
        st.stop()

    st.success(f"✅ Model ready ({size/(1024*1024):.2f} MB).")


@st.cache_resource(show_spinner="Loading model (first time only)...")
def load_model():
    ensure_model_exists()
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


def preprocess_image(pil_img: Image.Image):
    pil_img = pil_img.convert("RGB").resize(IMG_SIZE)
    x = np.array(pil_img, dtype=np.float32)
    x = np.expand_dims(x, axis=0)
    # ✅ no /255 because model contains Rescaling(1./255)
    return x


st.caption(f"Class mapping: 0 → {CLASS_NAMES[0]} | 1 → {CLASS_NAMES[1]}")

threshold = st.slider("Decision threshold (for class 1)", 0.10, 0.90, 0.50, 0.01)
show_probs = st.checkbox("Show probabilities", value=True)

uploaded = st.file_uploader("Upload chest X-ray image", type=["png", "jpg", "jpeg"])

# ✅ IMPORTANT: model loads ONLY after user uploads an image
if uploaded is not None:
    model = load_model()

    img = Image.open(uploaded)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    x = preprocess_image(img)
    p1 = float(model.predict(x, verbose=0)[0][0])  # P(label=1) = Normal
    p0 = 1.0 - p1

    pred_idx = 1 if p1 >= threshold else 0
    pred_name = CLASS_NAMES[pred_idx]
    conf = (p1 if pred_idx == 1 else p0) * 100.0

    st.subheader("Prediction")
    st.write(f"Result: **{pred_name}**")
    st.write(f"Confidence: **{conf:.2f}%**")

    if show_probs:
        st.write("Probabilities")
        st.write(f"- P({CLASS_NAMES[0]}): {p0:.4f}")
        st.write(f"- P({CLASS_NAMES[1]}): {p1:.4f}")
else:
    st.info("⬆️ Upload an image to start prediction (model will load only when needed).")
