import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown

st.set_page_config(page_title="COVID-19 CXR Detection", layout="centered")
st.title("CNN-Based COVID-19 Detection System (Chest X-ray)")
st.write("Upload a chest X-ray image. The model predicts: COVID or Normal.")

# ✅ MobileNetV2 model (Drive)
DRIVE_FILE_ID = "1Avkei-8mc1rwN0PCeAUT0V1XHChl3X_J"
MODEL_PATH = "covid_mobilenetv2_model.keras"

# ✅ MUST match your training order (Colab showed: ['COVID','Normal'])
CLASS_NAMES = ["COVID", "Normal"]   # label 0, label 1
IMG_SIZE = (224, 224)

def ensure_model():
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 5_000_000:
        return

    st.info("Downloading MobileNetV2 model from Google Drive (gdown)...")
    url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
    out = gdown.download(url, MODEL_PATH, quiet=False)

    if out is None or (not os.path.exists(MODEL_PATH)) or os.path.getsize(MODEL_PATH) < 5_000_000:
        size = os.path.getsize(MODEL_PATH) if os.path.exists(MODEL_PATH) else 0
        st.error(
            f"❌ Model download failed (downloaded size={size} bytes). "
            f"Check Drive sharing/quota or re-upload the model and update DRIVE_FILE_ID."
        )
        st.stop()

    st.success(f"Model downloaded ✅ (size={os.path.getsize(MODEL_PATH)//(1024*1024)} MB)")

@st.cache_resource
def load_model_cached():
    ensure_model()
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model_cached()

st.caption(f"Class mapping: 0 → {CLASS_NAMES[0]} | 1 → {CLASS_NAMES[1]}")
threshold = st.slider("Threshold (same as Colab)", 0.10, 0.90, 0.50, 0.01)

uploaded = st.file_uploader("Upload chest X-ray image", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    img_resized = img.resize(IMG_SIZE)
    x = np.array(img_resized, dtype=np.float32)
    x = np.expand_dims(x, axis=0)  # ✅ NO /255 because model has Rescaling internally

    # sigmoid output = P(label=1) = P(CLASS_NAMES[1]) = Normal
    p_label1 = float(model.predict(x, verbose=0)[0][0])
    p_label0 = 1.0 - p_label1

    # Colab decision
    if p_label1 >= threshold:
        pred_idx = 1
        confidence = p_label1 * 100
    else:
        pred_idx = 0
        confidence = p_label0 * 100

    st.subheader("Result (Colab-matched)")
    st.write(f"Raw sigmoid P(label=1) = **{p_label1:.4f}**")
    st.write(f"Threshold = **{threshold:.2f}**")
    st.write(f"Prediction = **{CLASS_NAMES[pred_idx]}**")
    st.write(f"Confidence = **{confidence:.2f}%**")

    st.write("Probabilities")
    st.write(f"- P({CLASS_NAMES[0]}) = {p_label0:.4f}")
    st.write(f"- P({CLASS_NAMES[1]}) = {p_label1:.4f}")

st.warning("Educational use only — not a clinical diagnosis tool.")
