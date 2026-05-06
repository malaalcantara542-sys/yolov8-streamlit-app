import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO

# =========================
# PAGE CONFIG (DAPAT UNA)
# =========================
st.set_page_config(
    page_title="AI Vision SaaS Pro",
    page_icon="🎥",
    layout="wide"
)

# =========================
# MODEL LOAD
# =========================
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# =========================
# UI DESIGN
# =========================
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}

[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        rgba(200, 180, 255, 0.85),
        rgba(170, 150, 255, 0.85)
    );
    backdrop-filter: blur(10px);
    border-right: 1px solid rgba(255,255,255,0.15);
}

[data-testid="stSidebar"] * {
    color: #2e2e2e !important;
    font-weight: 500;
}

[data-testid="stSidebar"] h1, 
[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] h3 {
    color: #1a1a1a !important;
    font-weight: 700;
}

.stButton>button {
    background: linear-gradient(90deg, #a18cd1, #c2a5ff);
    color: white;
    border-radius: 10px;
    border: none;
    padding: 0.5em 1em;
    font-weight: 600;
    width: 100%;
}

.stButton>button:hover {
    transform: scale(1.03);
}
h1, h2, h3 {
    color: white;
}
</style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================
st.title("🎥 Live Object Detection & Tracing")
st.write("Point your camera at objects to identify them in real-time.")

# =========================
# DETECTION FUNCTION
# =========================
def detect(frame):
    results = model.predict(frame, conf=0.3, verbose=False)
    annotated = results[0].plot()
    count = len(results[0].boxes) if results[0].boxes is not None else 0
    return annotated, count

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("⚙️ Control Panel")
    mode = st.selectbox("Select Mode", ["Live Camera", "Upload Image"])

    st.markdown("---")
    st.info("Object Detection System")
    st.markdown("---")
    st.markdown("👩‍💻 **Developer:**")
    st.markdown("Ma. Arlyn L. Alcantara")
    st.markdown("BSCS - 3A")

# =========================
# MAIN APP
# =========================
if mode == "Live Camera":
    st.subheader("📷 Live Camera Detection")

    img_file = st.camera_input("Open Camera")

    if img_file:
        image = Image.open(img_file).convert("RGB")
        image = np.array(image)

        processed, count = detect(image)

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original")
        with col2:
            st.image(processed, caption=f"Detected Objects: {count}")

elif mode == "Upload Image":
    st.subheader("🖼️ Image Detection")

    uploaded = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        image = np.array(image)

        processed, count = detect(image)

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original")
        with col2:
            st.image(processed, caption=f"Detected Objects: {count}")
