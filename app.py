"""
1:1 Identity Verification Gate - Interactive Dashboard
Streamlit-based UI for KYC biometric photo matching and metric threshold analysis.
"""

from pathlib import Path
import streamlit as st
from PIL import Image
from engine import FaceVerificationEngine

# Page configuration
st.set_page_config(
    page_title="1:1 Identity Verification Gate",
    page_icon="🛡️",
    layout="wide",
)


@st.cache_resource
def load_engine():
    """Initializes and caches the deep learning models in memory."""
    return FaceVerificationEngine(device="cpu")


engine = load_engine()

# --- Sidebar Controls ---
st.sidebar.title("Gate Parameters")
threshold = st.sidebar.slider(
    "Decision Threshold (Cosine Sim)",
    min_value=0.40,
    max_value=0.95,
    value=0.65,
    step=0.01,
    help="Higher threshold lowers False Acceptance Rate (stricter security). Lower threshold reduces False Rejection Rate.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("Biometric Metrics Guide")
st.sidebar.caption(
    """
    * **Cosine Similarity**: Measures angular similarity of the 512-D vectors in $[-1, 1]$. Values above `0.65` strongly suggest identical identity.
    * **Euclidean Distance**: Geometric distance in vector space. Identical vectors yield distance $\\approx 0.0$.
    """
)

# --- Header ---
st.title("🛡️ 1:1 Identity Verification Gate")
st.markdown(
    "Pairwise facial biometric verification system comparing a **Reference Image** against a **Probe Image** using MTCNN and Inception-ResNet-v1."
)

# Demo Buttons Row
st.markdown("##### Quick Demo Presets")
col_d1, col_d2, _ = st.columns([1.5, 1.5, 3])

demo_dir = Path("demo_images")
ref_img = None
probe_img = None

if "ref_img" not in st.session_state:
    st.session_state.ref_img = None
if "probe_img" not in st.session_state:
    st.session_state.probe_img = None

with col_d1:
    if st.button("👥 Load Genuine Pair (Match)"):
        st.session_state.ref_img = Image.open(demo_dir / "personA_reference.jpg")
        st.session_state.probe_img = Image.open(demo_dir / "personA_probe.jpg")

with col_d2:
    if st.button("🚫 Load Imposter Pair (Mismatch)"):
        st.session_state.ref_img = Image.open(demo_dir / "personA_reference.jpg")
        st.session_state.probe_img = Image.open(demo_dir / "personB_probe.jpg")

st.markdown("---")

# Image Input Columns
col_ref, col_probe = st.columns(2)

with col_ref:
    st.subheader("1. Reference Image")
    st.caption("Enrollment portrait or ID document")
    uploaded_ref = st.file_uploader(
        "Upload reference photo", type=["jpg", "jpeg", "png"], key="ref_upload"
    )
    if uploaded_ref:
        st.session_state.ref_img = Image.open(uploaded_ref)

    if st.session_state.ref_img:
        st.image(st.session_state.ref_img, use_container_width=True)

with col_probe:
    st.subheader("2. Probe Image")
    st.caption("Verification query or live photo")
    uploaded_probe = st.file_uploader(
        "Upload probe photo", type=["jpg", "jpeg", "png"], key="probe_upload"
    )
    if uploaded_probe:
        st.session_state.probe_img = Image.open(uploaded_probe)

    if st.session_state.probe_img:
        st.image(st.session_state.probe_img, use_container_width=True)

# Verification Execution
st.markdown("---")
if st.button("Run Verification Check", type="primary", use_container_width=True):
    if st.session_state.ref_img is None or st.session_state.probe_img is None:
        st.warning("Please upload or load both Reference and Probe images to run verification.")
    else:
        with st.spinner("Aligning faces and computing 512-D metric embeddings..."):
            result = engine.verify(
                st.session_state.ref_img,
                st.session_state.probe_img,
                threshold=threshold,
            )

        if not result["success"]:
            st.error(f"⚠️ Verification Halted: {result['error']}")
        else:
            sim = result["cosine_similarity"]
            dist = result["euclidean_distance"]
            matched = result["match"]

            # Visual Result Card
            st.markdown("### Verification Result")
            if matched:
                st.success(
                    f"✅ **IDENTITY VERIFIED** — Confidence: {sim:.4f} (Threshold: {threshold:.2f})"
                )
            else:
                st.error(
                    f"❌ **IDENTITY MISMATCH** — Similarity {sim:.4f} below threshold ({threshold:.2f})"
                )

            # Metrics Row
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Cosine Similarity", f"{sim:.4f}")
            m_col2.metric("Euclidean Distance", f"{dist:.4f}")
            m_col3.metric("Decision Verdict", "MATCH" if matched else "REJECT")

            # Aligned Face Inspection
            st.markdown("#### Aligned Facial Crops (MTCNN Output)")
            crop_col1, crop_col2 = st.columns(2)
            with crop_col1:
                st.image(result["ref_crop"], caption="Reference Face", width=180)
            with crop_col2:
                st.image(result["probe_crop"], caption="Probe Face", width=180)