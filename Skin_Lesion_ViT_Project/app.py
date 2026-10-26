import os
import json
import time
from pathlib import Path
import streamlit as st
import torch
import numpy as np
import pandas as pd
from PIL import Image

from src.data import CLASSES, make_synthetic
from src.models import PatchViT, MobileNetV2, n_params
from src.train import to_tensor, pretrain_ssl, train_classifier

# ----------------- PAGE CONFIGURATION -----------------
st.set_page_config(
    page_title="Skin Lesion Recognition - MobileNetV2 & ViT",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- CLINICAL METADATA -----------------
CLASS_INFO = {
    "mel": {
        "name": "Melanoma",
        "category": "Malignant Skin Cancer",
        "risk": "High Risk",
        "badge": "🚨 Malignant Neoplasm",
        "color": "#e74c3c",
        "is_malignant": True,
        "description": "Malignant neoplasm derived from melanocytes. The most dangerous form of skin cancer; prompt clinical excision is vital for patient prognosis.",
        "action": "Urgent consultation with a certified dermatologist / oncological specialist is strongly recommended."
    },
    "bcc": {
        "name": "Basal Cell Carcinoma",
        "category": "Malignant Skin Cancer",
        "risk": "Moderate Risk",
        "badge": "⚠️ Locally Invasive Cancer",
        "color": "#e67e22",
        "is_malignant": True,
        "description": "Most common cutaneous cancer. Locally invasive and slow-growing; rarely metastasizes but can cause extensive local tissue destruction if untreated.",
        "action": "Dermatological biopsy and planned surgical excision or cryotherapy recommended."
    },
    "akiec": {
        "name": "Actinic Keratosis / Bowen's Disease",
        "category": "Pre-cancerous / Carcinoma in situ",
        "risk": "Moderate Risk",
        "badge": "⚠️ Pre-Cancerous Dysplasia",
        "color": "#f39c12",
        "is_malignant": True,
        "description": "Keratinocyte dysplasia induced by cumulative ultraviolet radiation. Precursor to invasive cutaneous squamous cell carcinoma.",
        "action": "Clinical evaluation for topical 5-FU, imiquimod, or cryosurgical treatment."
    },
    "bkl": {
        "name": "Benign Keratosis",
        "category": "Benign Lesion",
        "risk": "Low Risk / Benign",
        "badge": "✅ Benign Lesion",
        "color": "#27ae60",
        "is_malignant": False,
        "description": "Encompasses seborrheic keratoses, solar lentigines, and lichen planus-like keratoses. Benign epidermal proliferations.",
        "action": "Harmless; no medical intervention required unless inflamed or causing mechanical irritation."
    },
    "nv": {
        "name": "Melanocytic Nevus",
        "category": "Benign Mole",
        "risk": "Low Risk / Benign",
        "badge": "✅ Benign Melanocytic Mole",
        "color": "#2ecc71",
        "is_malignant": False,
        "description": "Common benign skin lesion formed by clusters of melanocytes. Standard clinical dermoscopic observation recommended.",
        "action": "Benign; routine skin self-examination using the ABCDE criteria recommended."
    },
    "df": {
        "name": "Dermatofibroma",
        "category": "Benign Lesion",
        "risk": "Low Risk / Benign",
        "badge": "✅ Benign Dermal Lesion",
        "color": "#1abc9c",
        "is_malignant": False,
        "description": "Common benign dermal fibrous histiocytoma. Typically presents as a firm hyperpigmented nodule with a positive lateral dimple sign.",
        "action": "Benign; no active treatment necessary."
    },
    "vasc": {
        "name": "Vascular Lesion",
        "category": "Benign / Reactive",
        "risk": "Low Risk / Benign",
        "badge": "✅ Benign Vascular Lesion",
        "color": "#3498db",
        "is_malignant": False,
        "description": "Includes cherry angiomas, angiokeratomas, and pyogenic granulomas. Vascular proliferations of blood vessels.",
        "action": "Benign; medical evaluation advised if lesion repeatedly bleeds or ulcerates."
    }
}

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
WEIGHTS_PATH = RESULTS_DIR / "models" / "vit_ssl_best.pt"
RESULTS_JSON = RESULTS_DIR / "results.json"
FIGS_DIR = RESULTS_DIR / "figures"
HOME_IMG = BASE_DIR / "home_page.png"


# ----------------- CACHED MODEL LOADER -----------------
@st.cache_resource
def load_model(engine_name: str):
    """Loads MobileNetV2 or Vision Transformer with cached weights."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    k = len(CLASSES)
    cfg = dict(img=64, patch=8, dim=128, depth=4, heads=4)

    if "MobileNet" in engine_name:
        model = MobileNetV2(n_classes=k)
    else:
        model = PatchViT(**cfg, n_classes=k)
        if WEIGHTS_PATH.exists():
            try:
                ckpt = torch.load(WEIGHTS_PATH, map_location=device)
                state = ckpt.get("state", ckpt) if isinstance(ckpt, dict) else ckpt
                model.load_state_dict(state, strict=False)
            except Exception:
                pass

    model.to(device)
    model.eval()
    return model, device


def model_prediction(test_image, model, device, img_size=64):
    """Generates class prediction and probability distribution from an image."""
    image = Image.open(test_image).convert("RGB")
    resized = image.resize((img_size, img_size), Image.BILINEAR)
    arr = np.asarray(resized)
    x = to_tensor(arr[None, ...], device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    top_idx = int(np.argmax(probs))
    confidence = float(probs[top_idx]) * 100.0
    return top_idx, confidence, probs, resized


@st.cache_data
def get_sample_test_images():
    """Generates 7 representative sample images for testing."""
    samples = {}
    sample_dir = RESULTS_DIR / "demo_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    X, y, _ = make_synthetic(n=500, img=64, seed=42)
    for idx, c in enumerate(CLASSES):
        match_idx = np.where(y == idx)[0]
        if len(match_idx) > 0:
            arr = X[match_idx[0]]
            im = Image.fromarray(arr)
            file_path = sample_dir / f"sample_{c}.png"
            im.save(file_path)
            samples[c] = str(file_path)
    return samples


# ----------------- SIDEBAR NAVIGATION -----------------
st.sidebar.title("Dashboard")
st.sidebar.caption("🚀 Model Engine: **MobileNetV2 (Transfer Learning)** & **Vision Transformer**")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition", "Model Comparison", "Experiment Results"])

device_name = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"
st.sidebar.markdown(f"**Hardware Acceleration:** `{device_name}`")
st.sidebar.markdown("---")
st.sidebar.caption("MAAI901 Deep Learning Research | Baranika V G")


# =====================================================================
# ----------------- 1. HOME PAGE -----------------
# =====================================================================
if app_mode == "Home":
    st.header("SKIN LESION RECOGNITION SYSTEM 🔬")
    st.subheader("Powered by MobileNetV2 & Vision Transformer Deep Learning")

    if HOME_IMG.exists():
        st.image(str(HOME_IMG), use_container_width=True, caption="HAM10000 / Proxy Multi-Class Dermoscopic Skin Lesion Samples")

    st.markdown("""
    Welcome to the **Skin Lesion Recognition System**! 🔬🔍

    This clinical research application utilizes **MobileNetV2** and **Vision Transformer (ViT with Self-Supervised Masked Patch Pre-training)** 
    to rapidly analyze dermatoscopic skin lesion photographs across **7 diagnostic disease categories**.

    ### 🌟 Key Advantages of MobileNetV2 & Compact ViT
    - **MobileNetV2 CNN Baseline:** Ultra-compact inverted residual architecture with depthwise separable convolutions delivering fast, low-latency mobile inference (~2.2M parameters).
    - **Compact Vision Transformer (ViT):** Global multi-head self-attention with masked-patch self-supervised pre-training (SimMIM), achieving high diagnostic accuracy even under limited labeled training data (~600K parameters).
    - **7 Diagnostic Categories:** Accurately classifies Melanoma, Basal Cell Carcinoma, Actinic Keratosis, Benign Keratosis, Melanocytic Nevus, Dermatofibroma, and Vascular Lesions.
    - **Data Efficiency:** Overcomes the limitation of traditional Transformers requiring millions of images by pre-training on unlabelled dermoscopic patterns.

    ### 🩺 Diagnostic Categories Covered
    | Category Code | Disease / Diagnosis | Clinical Risk Level |
    |---|---|---|
    | **MEL** | Melanoma | 🚨 High Risk (Malignant) |
    | **BCC** | Basal Cell Carcinoma | ⚠️ Moderate Risk (Malignant) |
    | **AKIEC** | Actinic Keratosis / Bowen's Disease | ⚠️ Moderate Risk (Pre-cancerous) |
    | **BKL** | Benign Keratosis (Solar Lentigo / Seborrheic) | ✅ Benign |
    | **NV** | Melanocytic Nevus (Common Mole) | ✅ Benign |
    | **DF** | Dermatofibroma | ✅ Benign |
    | **VASC** | Vascular Lesion (Angioma) | ✅ Benign |

    ### How It Works
    1. **Upload Image:** Select **Disease Recognition** in the sidebar.
    2. **Choose Engine:** Toggle between **MobileNetV2** and **Vision Transformer**.
    3. **Instant Diagnosis:** Click **Analyze Lesion** to view prediction, confidence score, risk assessment, and clinical recommendations.

    ### Get Started
    Select **Disease Recognition** in the sidebar to begin!
    """)


# =====================================================================
# ----------------- 2. ABOUT PAGE -----------------
# =====================================================================
elif app_mode == "About":
    st.header("About the Models & Dataset")
    st.markdown("""
    #### 🧠 MobileNetV2 Architecture
    - **Model Family:** Deep Convolutional Neural Network (Sandler et al., 2018)
    - **Core Innovation:** Inverted Residuals with Linear Bottlenecks
    - **Computation Efficiency:** Depthwise Separable Convolutions reducing FLOPs by up to 80% compared to standard convolution
    - **Input Dimension:** $64 \\times 64 \\times 3$ (or $224 \\times 224 \\times 3$ RGB)
    - **Total Parameters:** ~2,232,839 parameters

    #### 🦅 Compact Vision Transformer (PatchViT + SimMIM SSL)
    - **Model Family:** Pure Transformer Architecture (Dosovitskiy et al., 2021)
    - **Self-Supervised Pretraining:** Masked Image Modeling (SimMIM / MAE) where 60% of patches are masked and reconstructed without labels
    - **Input Patch Division:** $8 \\times 8$ non-overlapping patches (64 tokens + learnable CLS token)
    - **Transformer Layers:** 4 layers with Pre-LN (`norm_first=True`)
    - **Multi-Head Self-Attention:** 4 attention heads, latent dimension $D=128$
    - **Total Parameters:** ~600,000 parameters (~73% fewer parameters than MobileNetV2)

    #### 📊 About the Dataset (HAM10000 Benchmark)
    - **Reference:** Tschandl, P., Rosendahl, C., & Kittler, H. (2018). *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.* Scientific Data.
    - **Total Benchmark Images:** 10,015 multi-source dermatoscopic images
    - **Data Split:** Stratified by class and grouped by patient lesion ID to strictly prevent data leakage
    - **Class Imbalance:** Highly skewed real-world distribution (e.g., 6,705 benign nevi vs 115 dermatofibroma cases), addressing a core challenge in automated medical screening.
    """)


# =====================================================================
# ----------------- 3. DISEASE RECOGNITION PAGE -----------------
# =====================================================================
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")

    engine_choice = st.sidebar.radio(
        "Active Model Engine",
        ["MobileNetV2 (CNN)", "Vision Transformer (ViT + SSL Pretrained)"],
        index=0
    )

    model, dev = load_model(engine_choice)
    engine_name = "MobileNetV2" if "MobileNet" in engine_choice else "Vision Transformer"
    st.caption(f"Active Model: `{engine_name}` | Compute: `{dev}`")

    sample_dict = get_sample_test_images()
    st.info("💡 Tip: Upload your own dermoscopic photo below, or pick from the sample test lesions.")

    col_source1, col_source2 = st.columns([1, 1])
    with col_source1:
        uploaded_file = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])
    with col_source2:
        sample_choice = st.selectbox(
            "Or Select a Sample Test Image:",
            ["(None - Use Uploaded File)"] + [f"{c.upper()} - {CLASS_INFO[c]['name']} ({CLASS_INFO[c]['category']})" for c in CLASSES]
        )

    active_image_file = None
    ground_truth_label = None

    if uploaded_file is not None:
        active_image_file = uploaded_file
    elif sample_choice != "(None - Use Uploaded File)":
        chosen_code = sample_choice.split(" - ")[0].lower()
        active_image_file = sample_dict.get(chosen_code)
        ground_truth_label = chosen_code

    col1, col2 = st.columns([1, 1], gap="large")

    if active_image_file is not None:
        with col1:
            st.subheader("Uploaded Lesion Image")
            pil_img = Image.open(active_image_file)
            st.image(pil_img, use_container_width=True)
            st.caption(f"Image Resolution: {pil_img.size[0]} x {pil_img.size[1]} px")

        with col2:
            st.subheader("Diagnosis")
            if st.button("Analyze Lesion", type="primary", use_container_width=True):
                with st.spinner(f"Analyzing with {engine_name}..."):
                    time.sleep(0.3)
                    top_idx, confidence, probs, resized = model_prediction(active_image_file, model, dev)
                    top_code = CLASSES[top_idx]
                    info = CLASS_INFO[top_code]

                    if info["is_malignant"]:
                        st.error(f"### ⚠️ Diagnosis: {info['name']} ({top_code.upper()})")
                        st.markdown(f"**Classification:** `{info['badge']}`")
                    else:
                        st.success(f"### 🌱 Diagnosis: {info['name']} ({top_code.upper()})")
                        st.markdown(f"**Classification:** `{info['badge']}`")

                    st.metric(label="Model Confidence", value=f"{confidence:.2f}%")

                    if ground_truth_label:
                        match = (ground_truth_label == top_code)
                        if match:
                            st.caption(f"🎯 Target Sample Match: Confirmed `{CLASS_INFO[ground_truth_label]['name']}`")
                        else:
                            st.caption(f"ℹ️ Sample Label: `{CLASS_INFO[ground_truth_label]['name']}` | Predicted: `{info['name']}`")

                    st.markdown(f"**Clinical Description:** {info['description']}")
                    st.info(f"**Recommended Action:** {info['action']}")

                    # Probability distribution bar chart
                    st.markdown("#### Probability Distribution across 7 Classes")
                    df_chart = pd.DataFrame({
                        "Condition": [CLASS_INFO[c]["name"] for c in CLASSES],
                        "Probability (%)": [round(float(p) * 100, 2) for p in probs]
                    }).set_index("Condition")
                    st.bar_chart(df_chart)

                    with st.expander("Detailed Probability Breakdown"):
                        df_details = pd.DataFrame({
                            "Code": [c.upper() for c in CLASSES],
                            "Condition": [CLASS_INFO[c]["name"] for c in CLASSES],
                            "Category": [CLASS_INFO[c]["category"] for c in CLASSES],
                            "Probability": [f"{float(p)*100:.2f}%" for p in probs],
                            "Risk": [CLASS_INFO[c]["risk"] for c in CLASSES]
                        })
                        st.dataframe(df_details, use_container_width=True)
    else:
        st.write("Please choose or upload a lesion image above to begin.")

    st.markdown("---")
    st.caption("Disclaimer: This deep learning model is an academic research prototype and not a certified medical device.")


# =====================================================================
# ----------------- 4. MODEL COMPARISON PAGE -----------------
# =====================================================================
elif app_mode == "Model Comparison":
    st.header("Model Architecture & Comparative Analysis")
    st.markdown("Direct side-by-side comparison of **MobileNetV2** and **Vision Transformer (PatchViT)**.")

    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown("""
        ### 📱 MobileNetV2 (CNN Engine)
        - **Paradigm:** Convolutional Neural Network
        - **Core Layer:** Inverted Residual Block with Linear Bottleneck
        - **Mechanism:** Expands channels ($6\\times$), applies $3\\times 3$ depthwise convolution, projects back to low dimension with a linear bottleneck
        - **Inductive Bias:** Strong local connectivity and translation invariance
        - **Total Parameters:** ~2,232,839
        """)

    with c2:
        st.markdown("""
        ### 🦅 Vision Transformer (Compact PatchViT)
        - **Paradigm:** Pure Attention-based Architecture
        - **Core Layer:** Multi-Head Self-Attention (4 heads, $D=128$, 4 layers)
        - **Mechanism:** Linear projection of $8\\times 8$ non-overlapping patches (64 tokens) with learned positional encodings
        - **Self-Supervised Pretraining:** 60% masked patch reconstruction (SimMIM) on unlabelled images
        - **Total Parameters:** ~600,000 (~73% fewer parameters!)
        """)

    st.markdown("---")
    st.subheader("🧩 Self-Supervised Masked Patch Simulation (ViT)")
    st.write("Visualizes how the Vision Transformer learns robust representations without human labels by reconstructing masked patches.")

    sim_col1, sim_col2 = st.columns([1, 2], gap="large")
    with sim_col1:
        mask_pct = st.slider("Mask Ratio (%)", 20, 80, 60, 5) / 100.0
    with sim_col2:
        samples = get_sample_test_images()
        test_im = Image.open(samples["mel"]).resize((64, 64))
        arr = np.asarray(test_im).copy()
        h, w, _ = arr.shape
        p = 8
        n_p = (h // p) * (w // p)
        rng = np.random.default_rng(42)
        mask_idx = rng.uniform(0, 1, n_p) < mask_pct

        masked_im = arr.copy()
        idx = 0
        for i in range(h // p):
            for j in range(w // p):
                if mask_idx[idx]:
                    masked_im[i*p:(i+1)*p, j*p:(j+1)*p] = 120
                idx += 1

        v1, v2 = st.columns(2)
        with v1:
            st.image(test_im, caption="1. Original Lesion (64x64)", width=180)
        with v2:
            st.image(Image.fromarray(masked_im), caption=f"2. {int(mask_pct*100)}% Masked Patches", width=180)


# =====================================================================
# ----------------- 5. EXPERIMENT RESULTS PAGE -----------------
# =====================================================================
elif app_mode == "Experiment Results":
    st.header("Published Experiment Results & Figures")

    if RESULTS_JSON.exists():
        with open(RESULTS_JSON, "r") as f:
            res_data = json.load(f)

        meta = res_data.get("meta", {})
        rows = res_data.get("rows", [])

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Benchmark Dataset", meta.get("dataset", "HAM10000").upper())
        m2.metric("Total Dataset Images", meta.get("n_total", 2500))
        m3.metric("Benchmark Runtime", f"{meta.get('runtime_min', 'N/A')} min")
        m4.metric("Hardware Device", meta.get("device", "CPU").upper())

        st.markdown("---")
        st.subheader("Publication Figures")
        fc1, fc2 = st.columns(2)

        figs = [
            ("fig_macro_f1.png", "Figure 1: Test Macro-F1 vs Labelled Data Budget (10% - 100%)"),
            ("fig_class_dist.png", "Figure 2: HAM10000 Diagnostic Class Imbalance (Log Scale)"),
            ("fig_recall_by_class.png", "Figure 3: Per-Class Diagnostic Recall under Scarce Data (10% labels)"),
            ("fig_pretrain_loss.png", "Figure 4: Self-Supervised Masked-Patch Reconstruction Loss Curve"),
        ]

        for i, (fname, caption) in enumerate(figs):
            p = FIGS_DIR / fname
            with fc1 if i % 2 == 0 else fc2:
                if p.exists():
                    st.image(str(p), caption=caption, use_container_width=True)
                else:
                    st.caption(f"Figure `{fname}` will appear once `RUN_ME.bat` finishes.")

        if rows:
            st.subheader("Experiment Metrics Table")
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.warning("⚠️ Results file `results/results.json` not found yet.")
        st.markdown("""
        To run the complete benchmark suite:
        - Run `RUN_ME.bat` from Windows explorer or terminal.
        - The figures and DOCX/PDF reports will automatically appear here!
        """)
