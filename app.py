import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown

# =========================
# Page UI
# =========================
st.set_page_config(page_title="COVID-19 CXR Detection", layout="centered")
st.title("CNN-Based COVID-19 Detection System (Chest X-ray)")
st.write("Upload a chest X-ray image. The model predicts: COVID or Normal.")

# =========================
# Model settings
# =========================
DRIVE_FILE_ID = "18zNPnB62-DvJAddb7mcVcV_A2WT0Awr7"   # Google Drive file id
MODEL_PATH = "covid_cnn_model.keras"                  # local filename in Streamlit Cloud

# IMPORTANT: must match your training order (train_ds.class_names)
CLASS_NAMES = ["COVID", "Normal"]   # label 0, label 1
IMG_SIZE = (224, 224)

# =========================
# Download model from Google Drive
# =========================
def ensure_model():
    # if already present and looks valid, skip download
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 5_000_000:
        return

    st.info("Downloading model from Google Drive (gdown)...")
    url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
    out = gdown.download(url, MODEL_PATH, quiet=False)

    # validate download
    if out is None or (not os.path.exists(MODEL_PATH)) or os.path.getsize(MODEL_PATH) < 5_000_000:
        size = os.path.getsize(MODEL_PATH) if os.path.exists(MODEL_PATH) else 0
        st.error(
            f"❌ Model download failed (downloaded size={size} bytes). "
            f"Check Drive sharing, quota limits, or re-upload the model with a new link."
        )
        st.stop()

    st.success(f"Model downloaded ✅ (size={os.path.getsize(MODEL_PATH)//(1024*1024)} MB)")

@st.cache_resource
def load_model_cached():
    ensure_model()
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model_cached()

# =========================
# Controls
# =========================
st.caption(f"Class mapping: 0 → {CLASS_NAMES[0]} | 1 → {CLASS_NAMES[1]}")
min_conf = st.slider("Minimum confidence to accept prediction (%)", 50, 99, 80, 1)

uploaded = st.file_uploader(
    "Upload chest X-ray image",
    type=["png", "jpg", "jpeg"]
)

# =========================
# Prediction
# =========================
if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # preprocess
    img_resized = img.resize(IMG_SIZE)
    x = np.array(img_resized, dtype=np.float32)
    x = np.expand_dims(x, axis=0)  # (1, 224, 224, 3)

    # predict
    # sigmoid output = P(label=1) = CLASS_NAMES[1] which is "Normal" in this mapping
    p1 = float(model.predict(x, verbose=0)[0][0])   # P(label=1)
    p0 = 1.0 - p1                                   # P(label=0)

    # choose higher probability
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
        st.warning("⚠️ Low confidence prediction. Try a clearer chest X-ray image or different view.")

# =========================
# Disclaimer
# =========================
st.warning("Educational use only — not a clinical diagnosis tool.")
