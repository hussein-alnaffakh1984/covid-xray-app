import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="COVID-19 CXR Detection",
    layout="centered"
)

st.title("CNN-Based COVID-19 Detection System (Chest X-ray)")
st.write("Upload a chest X-ray image. The model predicts: COVID or Normal.")

# -------------------------------------------------
# MODEL SETTINGS (مطابقة 100% لـ Colab)
# -------------------------------------------------
DRIVE_FILE_ID = "15GpcUSNtMf1Hy83nbzrptL-WyBboyy_l"
MODEL_PATH = "covid_mobilenetv2_model.keras"

# ⚠️ هذا الترتيب يجب أن يطابق train_ds.class_names
CLASS_NAMES = ["COVID", "Normal"]  # label 0, label 1
IMG_SIZE = (224, 224)

# -------------------------------------------------
# Download model if not exists
# -------------------------------------------------
def ensure_model():
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 50_000_000:
        return

    st.info("Downloading model from Google Drive...")
    url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model download failed.")
        st.stop()

    size_mb = os.path.getsize(MODEL_PATH) // (1024 * 1024)
    st.success(f"✅ Model downloaded (size = {size_mb} MB)")

# -------------------------------------------------
# Load model (cached)
# -------------------------------------------------
@st.cache_resource
def load_model():
    ensure_model()
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

st.caption(f"Class mapping: 0 → {CLASS_NAMES[0]} | 1 → {CLASS_NAMES[1]}")

# -------------------------------------------------
# Threshold control
# -------------------------------------------------
threshold = st.slider(
    "Decision threshold for class 1 (Normal)",
    0.10, 0.90, 0.50, 0.01
)

# -------------------------------------------------
# Upload image
# -------------------------------------------------
uploaded = st.file_uploader(
    "Upload chest X-ray image",
    type=["png", "jpg", "jpeg"]
)

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # preprocessing مطابق للتدريب
    img = img.resize(IMG_SIZE)
    x = np.array(img)
    x = np.expand_dims(x, axis=0)  # ❗ لا /255 (Rescaling داخل الموديل)

    # prediction
    p_label1 = float(model.predict(x, verbose=0)[0][0])  # Normal
    p_label0 = 1 - p_label1                              # COVID

    if p_label1 >= threshold:
        pred_idx = 1
        confidence = p_label1 * 100
    else:
        pred_idx = 0
        confidence = p_label0 * 100

    st.subheader("Prediction Result")
    st.write(f"**Prediction:** {CLASS_NAMES[pred_idx]}")
    st.write(f"**Confidence:** {confidence:.2f}%")

    st.write("Probabilities:")
    st.write(f"- P(COVID): {p_label0:.4f}")
    st.write(f"- P(Normal): {p_label1:.4f}")

st.warning("⚠️ Educational use only — not a clinical diagnostic tool.")
