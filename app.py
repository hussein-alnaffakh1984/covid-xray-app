import os
import sys
import time
import streamlit as st
import numpy as np
from PIL import Image

st.set_page_config(page_title="COVID-19 CXR Detection", layout="centered")
st.title("CNN-Based COVID-19 Detection System (Chest X-ray)")
st.caption("⚠️ Educational use only — not a clinical diagnosis tool.")

# =========================
# SETTINGS
# =========================
DRIVE_FILE_ID = "15GpcUSNtMf1Hy83nbzrptL-WyBboyy_l"  # <-- ضع ID ملف الموديل هنا (من رابطك الجديد)
MODEL_PATH = "covid_mobilenetv2_model.keras"
CLASS_NAMES = ["COVID", "Normal"]   # تأكد أنها نفس ترتيب التدريب
IMG_SIZE = (224, 224)
MIN_BYTES = 30 * 1024 * 1024        # 30MB حد أدنى للتأكد أنه مو HTML

# =========================
# DEBUG INFO (مهم)
# =========================
with st.expander("Debug info"):
    st.write("Python:", sys.version)
    st.write("Working dir:", os.getcwd())
    st.write("Model exists:", os.path.exists(MODEL_PATH))
    if os.path.exists(MODEL_PATH):
        st.write("Model size (MB):", round(os.path.getsize(MODEL_PATH)/(1024*1024), 2))

# =========================
# Download helper (بدون gdown)
# =========================
def download_from_gdrive(file_id: str, dest_path: str):
    """
    Download public Drive file using requests-like approach via urllib (works on Streamlit Cloud).
    Avoids gdown issues when Drive blocks confirmation tokens.
    """
    import urllib.request

    url = f"https://drive.google.com/uc?export=download&id={file_id}"

    # Try direct download
    urllib.request.urlretrieve(url, dest_path)

    # Validate size
    if (not os.path.exists(dest_path)) or os.path.getsize(dest_path) < MIN_BYTES:
        # Sometimes Drive returns an HTML page. Force a clearer error.
        raise RuntimeError(
            "Downloaded file is too small. "
            "Likely Drive permission/quota or HTML interstitial page."
        )

# =========================
# Lazy model load
# =========================
@st.cache_resource(show_spinner=False)
def load_tf_model(path: str):
    import tensorflow as tf
    return tf.keras.models.load_model(path, compile=False)

# =========================
# UI - Model control
# =========================
st.subheader("1) Load model")

col1, col2 = st.columns(2)

with col1:
    if st.button("⬇️ Download model"):
        try:
            st.info("Downloading model from Google Drive...")
            # حذف أي ملف قديم صغير/HTML
            if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) < MIN_BYTES:
                os.remove(MODEL_PATH)

            download_from_gdrive(DRIVE_FILE_ID, MODEL_PATH)
            st.success(f"Model downloaded ✅ ({os.path.getsize(MODEL_PATH)//(1024*1024)} MB)")
        except Exception as e:
            st.error(f"Download failed: {e}")

with col2:
    if st.button("🧠 Load model into memory"):
        try:
            if not os.path.exists(MODEL_PATH):
                st.error("Model file not found. Please download it first.")
            elif os.path.getsize(MODEL_PATH) < MIN_BYTES:
                st.error("Model file looks invalid (too small). Re-download with correct permissions.")
            else:
                st.info("Loading model (this can take a bit on first load)...")
                model = load_tf_model(MODEL_PATH)
                st.session_state["model_ready"] = True
                st.success("Model loaded ✅")
        except Exception as e:
            st.error(f"Load failed: {e}")

st.caption(f"Class mapping: 0 → {CLASS_NAMES[0]} | 1 → {CLASS_NAMES[1]}")

# =========================
# Prediction UI
# =========================
st.subheader("2) Predict")
threshold = st.slider("Decision threshold (for class 1)", 0.10, 0.90, 0.50, 0.01)
uploaded = st.file_uploader("Upload chest X-ray image", type=["png", "jpg", "jpeg"])

def preprocess(pil_img: Image.Image):
    pil_img = pil_img.convert("RGB").resize(IMG_SIZE)
    x = np.array(pil_img, dtype=np.float32)
    x = np.expand_dims(x, axis=0)
    # IMPORTANT:
    # إذا كان الموديل يحتوي Rescaling(1./255) داخل الشبكة => لا تقسم هنا.
    return x

if uploaded is not None:
    if not st.session_state.get("model_ready", False):
        st.warning("Please load the model first (Step 1).")
    else:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded Image", use_container_width=True)

        model = load_tf_model(MODEL_PATH)
        x = preprocess(img)

        p1 = float(model.predict(x, verbose=0)[0][0])  # P(class 1)
        p0 = 1.0 - p1

        pred_idx = 1 if p1 >= threshold else 0
        pred_name = CLASS_NAMES[pred_idx]
        conf = (p1 if pred_idx == 1 else p0) * 100.0

        st.success(f"Result: {pred_name} | Confidence: {conf:.2f}%")
        st.write(f"P({CLASS_NAMES[0]}) = {p0:.4f}")
        st.write(f"P({CLASS_NAMES[1]}) = {p1:.4f}")
