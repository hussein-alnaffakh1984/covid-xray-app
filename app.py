import os
import requests
import streamlit as st
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

# =========================
# UI
# =========================
st.set_page_config(page_title="COVID-19 CXR Detection (TFLite)", layout="centered")
st.title("COVID-19 Detection System (Chest X-ray) — TFLite")
st.write("Upload a chest X-ray image. The model predicts: COVID or Normal.")
st.caption("⚠️ Educational use only — not a clinical diagnosis tool.")

# =========================
# SETTINGS
# =========================
DRIVE_FILE_ID = "1fHpCW_JwF8sTLtSx1pBvsdUnggfDQxxW"
MODEL_PATH = "/tmp/covid_model.tflite"
CLASS_NAMES = ["COVID", "Normal"]   # label 0, label 1
IMG_SIZE = (224, 224)

# =========================
# Download from Google Drive (stable)
# =========================
def download_from_gdrive(file_id: str, dest_path: str):
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()

    r = session.get(URL, params={"id": file_id}, stream=True)
    r.raise_for_status()

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
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1_000_000:
        return
    st.info("Downloading TFLite model (first time only)...")
    try:
        download_from_gdrive(DRIVE_FILE_ID, MODEL_PATH)
    except Exception as e:
        st.error(f"❌ Download failed: {e}")
        st.stop()

    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1_000_000:
        size = os.path.getsize(MODEL_PATH) if os.path.exists(MODEL_PATH) else 0
        st.error(f"❌ Model download failed or file too small ({size} bytes). Check Drive sharing.")
        st.stop()

    st.success(f"✅ Model ready ({os.path.getsize(MODEL_PATH)/(1024*1024):.2f} MB)")

@st.cache_resource(show_spinner="Loading model (first time only)...")
def load_interpreter():
    ensure_model_exists()
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

def preprocess(pil_img: Image.Image):
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    x = np.array(img, dtype=np.float32)
    x = np.expand_dims(x, axis=0)   # (1,224,224,3)
    # ✅ TFLite غالباً يحتاج /255
    x = x / 255.0
    return x

# =========================
# Controls
# =========================
threshold = st.slider("Decision threshold (for class 1)", 0.10, 0.90, 0.50, 0.01)
show_probs = st.checkbox("Show probabilities", value=True)

uploaded = st.file_uploader("Upload chest X-ray image", type=["png", "jpg", "jpeg"])

if uploaded is None:
    st.info("⬆️ Upload an image to start prediction.")
else:
    interpreter = load_interpreter()

    img = Image.open(uploaded)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    x = preprocess(img)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], x.astype(np.float32))
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]["index"])

    # out is sigmoid probability for label 1
    p1 = float(out.reshape(-1)[0])
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
