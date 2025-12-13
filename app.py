import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="COVID-19 CXR Detection", layout="centered")
st.title("CNN-Based COVID-19 Detection System (Chest X-ray)")
st.write("Upload a chest X-ray image. The model predicts: COVID or Normal.")
st.caption("⚠️ Educational use only — not a clinical diagnosis tool.")

# =========================
# SETTINGS (EDIT ONLY THESE)
# =========================
DRIVE_FILE_ID = "1ffyT6-lb19VUCbGCal0eN_qeNGZz9KR4"   # <- ID from your Drive link
MODEL_PATH = "covid_mobilenetv2_infer.keras"          # <- local filename to save
CLASS_NAMES = ["COVID", "Normal"]                     # label 0, label 1
IMG_SIZE = (224, 224)

# If your model is ~100MB+ keep 50MB minimum check
MIN_BYTES = 50 * 1024 * 1024  # 50MB


# =========================
# DOWNLOAD MODEL (ONLY ONCE)
# =========================
def ensure_model_exists():
    # If already downloaded and size OK, skip downloading
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > MIN_BYTES:
        return

    st.info("Downloading model from Google Drive (first time only)...")
    url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
    out = gdown.download(url, MODEL_PATH, quiet=False)

    # Validate
    if out is None or (not os.path.exists(MODEL_PATH)):
        st.error("❌ Model download failed. Please check Drive sharing: Anyone with the link -> Viewer.")
        st.stop()

    size = os.path.getsize(MODEL_PATH)
    if size < MIN_BYTES:
        st.error(
            f"❌ Model downloaded but file size is too small ({size/(1024*1024):.2f} MB).\n"
            "This usually happens if Drive returned an HTML/virus-check page.\n"
            "✅ Fix: Re-share file (Anyone with link) and try again."
        )
        st.stop()

    st.success(f"✅ Model downloaded successfully ({size/(1024*1024):.2f} MB).")


# =========================
# LOAD MODEL (CACHED)
# =========================
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    ensure_model_exists()
    # compile=False -> faster + avoids issues
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


model = load_model()

st.caption(f"Class mapping: 0 → {CLASS_NAMES[0]} | 1 → {CLASS_NAMES[1]}")

# =========================
# UI CONTROLS
# =========================
threshold = st.slider("Decision threshold (for class 1)", 0.10, 0.90, 0.50, 0.01)
show_probs = st.checkbox("Show probabilities", value=True)

uploaded = st.file_uploader("Upload chest X-ray image", type=["png", "jpg", "jpeg"])


# =========================
# PREDICTION
# =========================
def preprocess_image(pil_img: Image.Image):
    pil_img = pil_img.convert("RGB")
    pil_img = pil_img.resize(IMG_SIZE)
    x = np.array(pil_img, dtype=np.float32)
    x = np.expand_dims(x, axis=0)  # (1,224,224,3)
    # ✅ DO NOT /255 هنا إذا كان الموديل يحتوي Rescaling(1./255)
    return x


if uploaded is not None:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    x = preprocess_image(img)

    # Model outputs sigmoid probability for label 1
    p1 = float(model.predict(x, verbose=0)[0][0])
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

    st.info("Tip: If results look biased, adjust the threshold slider and test multiple images.")
