import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown

st.set_page_config(page_title="COVID-19 CXR Detection", layout="centered")
st.title("CNN-Based COVID-19 Detection System (Chest X-ray)")
st.write("Upload a chest X-ray image. The model predicts: COVID or Normal.")

# ✅ نفس الـ File ID من رابط Drive
DRIVE_FILE_ID = "18zNPnB62-DvJAddb7mcVcV_A2WT0Awr7"

# ✅ اسم محلي فقط (أي اسم تريده) — المهم يكون .keras
MODEL_PATH = "covid_cnn_model.keras"

# ⚠️ تأكد أن هذا نفس ترتيب التدريب لديك (train_ds.class_names)
CLASS_NAMES = ["COVID", "Normal"]  # label 0, label 1
IMG_SIZE = (224, 224)

def ensure_model():
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 5_000_000:
        return

    st.info("Downloading model from Google Drive (gdown)...")
    url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
    out = gdown.download(url, MODEL_PATH, quiet=False)

    if out is None or (not os.path.exists(MODEL_PATH)) or os.path.getsize(MODEL_PATH) < 5_000_000:
        size = os.path.getsize(MODEL_PATH) if os.path.exists(MODEL_PATH) else 0
        st.error(f"❌ Model download failed (downloaded size={size} bytes). "
                 f"Possible causes: Drive quota/virus-scan page. Try again or re-upload the model to Drive with a new link.")
        st.stop()

    st.success(f"Model downloaded ✅ (size={os.path.getsize(MODEL_PATH)//(1024*1024)} MB)")

@st.cache_resource
def load_model_cached():
    ensure_model()
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model_cached()

st.caption(f"Class mapping: 0 → {CLASS_NAMES[0]} | 1 → {CLASS_NAMES[1]}")
min_conf = st.slider("Minimum confidence to accept prediction (%)", 50, 99, 80, 1)

uploaded = st.file_uploader(
    "Upload chest X-ray image",
    type=["png", "jpg", "jpeg"]
)


