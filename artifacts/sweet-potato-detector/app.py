"""
Sweet Potato Leaf Health Detection System
Image-based CNN classification using MobileNetV2 with GradCAM explainability.
Billante & Querioso, LNU 2024
"""

import io
import os
import numpy as np
import streamlit as st
from PIL import Image
import plotly.graph_objects as go
import plotly.figure_factory as ff
import pandas as pd

from model import load_model, predict, CLASS_NAMES
from gradcam import compute_gradcam, overlay_heatmap, create_comparison_figure
from recommendations import get_disease_info, get_severity_badge

# ─── Page configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sweet Potato Leaf Health Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Session state ─────────────────────────────────────────────────────────────
if "model" not in st.session_state:
    with st.spinner("Loading MobileNetV2 model…"):
        st.session_state.model, st.session_state.is_trained = load_model()

model     = st.session_state.model
is_trained = st.session_state.is_trained

DATASET_DIR = os.path.join(os.path.dirname(__file__), "sorted_dataset_jpeg", "sorted_dataset")
FOLDER_MAP  = {
    "Healthy":              "Healthy",
    "Leaf_Curl_Virus":      "Sweet Potato Leaf Curl Virus",
    "Fusarium_Wilt":        "Fusarium Wilt",
    "Cercospora_Leaf_Spot": "Cercospora Leaf Spot",
}
CLASS_COLORS = ["#27AE60", "#E67E22", "#C0392B", "#8E44AD"]

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d2/Ipomoea_batatas_006.jpg/320px-Ipomoea_batatas_006.jpg",
        caption="Ipomoea batatas",
        use_container_width=True,
    )
    st.markdown("## 🌿 About This System")
    st.markdown(
        """
        An image-based leaf health detection system using **MobileNetV2** CNN
        with **GradCAM** explainability — classifying sweet potato leaves into:

        - ✅ Healthy
        - ⚠️ Sweet Potato Leaf Curl Virus
        - 🔴 Fusarium Wilt
        - 🟣 Cercospora Leaf Spot
        """
    )
    st.divider()
    st.markdown("### ⚙️ GradCAM Settings")
    heatmap_alpha = st.slider("Overlay intensity", 0.2, 0.8, 0.45, 0.05)
    show_raw_heatmap = st.checkbox("Show raw heatmap", value=False)
    colormap = st.selectbox("Colormap", ["jet", "inferno", "plasma", "hot", "YlOrRd"])
    st.divider()
    if not is_trained:
        st.warning("⚠️ **Demo Mode** — no trained weights found.", icon="🧪")
    else:
        st.success("✅ Trained model loaded.", icon="🤖")
    st.divider()
    st.caption(
        "**Reference:** Billante & Querioso, *Development of an Image-Based System "
        "for Detecting Health Conditions of Sweet Potato Leaves Using CNNs*, LNU 2024."
    )

# ─── Page title ────────────────────────────────────────────────────────────────
st.title("🌿 Sweet Potato Leaf Health Detection System")
st.markdown(
    "An image-based CNN diagnostic system for classifying *Ipomoea batatas* leaf health conditions "
    "with GradCAM explainability and evidence-based prescriptive recommendations."
)

# ─── Navigation tabs ───────────────────────────────────────────────────────────
tab_detect, tab_eval = st.tabs(["🔬 Detection & Diagnosis", "📊 Model Evaluation & Results"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DETECTION
# ══════════════════════════════════════════════════════════════════════════════
with tab_detect:
    st.divider()
    col_upload, col_preview = st.columns([1, 1], gap="large")

    with col_upload:
        st.subheader("📤 Upload Leaf Image")
        uploaded_file = st.file_uploader(
            "Choose a sweet potato leaf image (JPG / PNG / WEBP)",
            type=["jpg", "jpeg", "png", "webp"],
            help="For best results, upload a clear, well-lit image of a single leaf.",
        )
        if uploaded_file:
            pil_img = Image.open(uploaded_file).convert("RGB")
            st.markdown(f"**File:** `{uploaded_file.name}` — {pil_img.width}×{pil_img.height} px")

    with col_preview:
        if uploaded_file:
            st.subheader("🖼️ Image Preview")
            st.image(pil_img, use_container_width=True)

    if uploaded_file:
        img_array = np.array(pil_img)

        with st.spinner("🔬 Analyzing leaf image…"):
            probs         = predict(model, img_array, is_trained)
            predicted_idx = int(np.argmax(probs))
            predicted_class = CLASS_NAMES[predicted_idx]
            confidence    = float(probs[predicted_idx])

            heatmap = compute_gradcam(model, img_array, predicted_idx, is_trained)
            overlay = overlay_heatmap(img_array, heatmap, alpha=heatmap_alpha, colormap=colormap)
            fig_buf = create_comparison_figure(img_array, heatmap, overlay, predicted_class, confidence)

        disease_info = get_disease_info(predicted_class)
        st.divider()

        # ── Classification result ──────────────────────────────────────────
        st.subheader("📊 Classification Result")
        r1, r2, r3 = st.columns(3)
        r1.metric("Detected Condition", f"{disease_info['icon']} {predicted_class}")
        r2.metric("Confidence Score",   f"{confidence:.1%}")
        r3.metric("Severity",           get_severity_badge(disease_info["severity"]))

        st.markdown("#### Class Probability Distribution")
        fig = go.Figure(go.Bar(
            x=CLASS_NAMES,
            y=[float(p) for p in probs],
            marker_color=CLASS_COLORS,
            text=[f"{float(p):.1%}" for p in probs],
            textposition="outside",
        ))
        fig.update_layout(
            yaxis=dict(range=[0, 1.1], tickformat=".0%", title="Probability"),
            xaxis_title="Health Condition",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"), height=300, margin=dict(t=10, b=10),
        )
        fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)")
        fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
        st.plotly_chart(fig, use_container_width=True)
        st.divider()

        # ── Descriptive analytics ──────────────────────────────────────────
        st.subheader("🔍 Descriptive Analytics — Observed Symptoms")
        st.markdown(f"**{disease_info['icon']} {predicted_class}** — {disease_info['description']}")
        sym_cols = st.columns(2)
        for i, symptom in enumerate(disease_info["symptoms"]):
            sym_cols[i % 2].markdown(f"• {symptom}")
        st.divider()

        # ── GradCAM diagnostic ─────────────────────────────────────────────
        st.subheader("🧠 Diagnostic Analytics — GradCAM Visual Explanation")
        st.markdown(
            "GradCAM highlights the **leaf regions** most influential in the model's decision. "
            "Warm colours (red/yellow) = high attention; cool colours = low attention."
        )
        st.image(fig_buf, caption=f"GradCAM — {predicted_class} ({confidence:.1%} confidence)", use_container_width=True)
        if show_raw_heatmap:
            st.image(
                overlay_heatmap(np.zeros_like(img_array), heatmap, alpha=1.0, colormap=colormap),
                caption="Raw GradCAM heatmap",
                use_container_width=True,
            )
        st.divider()

        # ── Predictive analytics ───────────────────────────────────────────
        st.subheader("📈 Predictive Analytics — Spread Risk Assessment")
        risk_map = {
            "Healthy":                        ("Low",       "No active pathogen detected. Risk of crop loss is minimal.",                                                                                              "#27AE60"),
            "Sweet Potato Leaf Curl Virus":   ("Very High", "Viral disease spreads rapidly via whitefly vectors. Immediate removal and vector control is critical.",                                                   "#C0392B"),
            "Fusarium Wilt":                  ("High",      "Soilborne pathogen persists for years. Crop rotation and soil treatment are essential to prevent re-infection.",                                           "#E74C3C"),
            "Cercospora Leaf Spot":           ("Moderate",  "Spread accelerates under warm, humid conditions. Early fungicide intervention can prevent defoliation and significant yield loss.",                        "#E67E22"),
        }
        risk_level, risk_text, risk_color = risk_map[predicted_class]
        st.markdown(
            f"""<div style="border-left:4px solid {risk_color};padding:12px 16px;
            border-radius:4px;background:rgba(255,255,255,0.04);margin-bottom:8px;">
            <strong style="color:{risk_color}">Spread Risk: {risk_level}</strong><br>{risk_text}
            </div>""",
            unsafe_allow_html=True,
        )
        st.divider()

        # ── Prescriptive recommendations ──────────────────────────────────
        st.subheader("📋 Prescriptive Analytics — Management Recommendations")
        st.markdown(
            "Evidence-based interventions from **Philippine Bureau of Plant Industry (BPI)** "
            "and **Department of Agriculture (DA)** crop protection guidelines."
        )
        if predicted_class == "Healthy":
            st.success("✅ No disease detected. Continue current agronomic practices and regular monitoring.", icon="🌱")
        for rec_group in disease_info["recommendations"]:
            with st.expander(f"📌 {rec_group['category']}", expanded=True):
                for action in rec_group["actions"]:
                    st.markdown(f"▸ {action}")
        st.divider()
        st.caption(
            "⚠️ This system is a **decision-support tool**. Final diagnosis should be confirmed "
            "by a licensed agriculturist. Follow local BPI/DA guidelines for pesticide application."
        )

    else:
        st.info("👆 Upload a sweet potato leaf image above to begin analysis.", icon="🌿")
        st.markdown("### 📚 Detectable Health Conditions")
        cols = st.columns(4)
        cards = [
            ("✅", "Healthy",                       "#27AE60", "Normal leaf — no disease signs detected."),
            ("⚠️", "Sweet Potato\nLeaf Curl Virus", "#E67E22", "Viral disease spread by whiteflies causing leaf curling and mosaic patterns."),
            ("🔴", "Fusarium Wilt",                 "#C0392B", "Soilborne fungal disease causing yellowing, wilting, and vascular browning."),
            ("🟣", "Cercospora\nLeaf Spot",          "#8E44AD", "Fungal disease producing circular spots with dark margins and defoliation."),
        ]
        for col, (icon, name, color, desc) in zip(cols, cards):
            col.markdown(
                f"""<div style="border:1px solid {color};border-radius:8px;padding:16px;
                text-align:center;min-height:160px;">
                <div style="font-size:2rem">{icon}</div>
                <div style="font-weight:700;color:{color};margin:6px 0">{name}</div>
                <div style="font-size:0.85rem;color:#aaa">{desc}</div></div>""",
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.divider()
    st.subheader("📊 Model Evaluation & Performance Results")
    st.markdown(
        "Performance metrics of the **MobileNetV2 CNN** trained on the sweet potato leaf dataset. "
        "Evaluated using stratified 80/20 train-validation split with data augmentation."
    )

    dataset_exists = os.path.isdir(DATASET_DIR)

    # ── Dataset overview ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📁 Dataset Overview")

    if dataset_exists:
        folders   = list(FOLDER_MAP.keys())
        counts    = [len(os.listdir(os.path.join(DATASET_DIR, f)))
                     for f in folders if os.path.isdir(os.path.join(DATASET_DIR, f))]
        labels    = [FOLDER_MAP[f] for f in folders]
        total     = sum(counts)

        d1, d2, d3, d4 = st.columns(4)
        for col, lbl, cnt, clr in zip([d1, d2, d3, d4], labels, counts, CLASS_COLORS):
            col.metric(lbl, f"{cnt} images", f"{cnt/total*100:.1f}% of dataset")

        st.metric("Total Dataset Size", f"{total} images", "Sweet potato leaf photos collected in Leyte, Philippines")

        fig_pie = go.Figure(go.Pie(
            labels=labels,
            values=counts,
            marker_colors=CLASS_COLORS,
            hole=0.4,
            textinfo="label+percent+value",
            textfont_size=12,
        ))
        fig_pie.update_layout(
            title="Dataset Distribution by Class",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            height=350,
            margin=dict(t=40, b=10),
            showlegend=False,
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("Dataset folder not found. Showing pre-computed metrics from training.")

    # ── Compute metrics on validation set ─────────────────────────────────
    st.markdown("---")
    st.markdown("### 🎯 Classification Performance")

    @st.cache_data(show_spinner="Computing evaluation metrics on validation set…")
    def compute_metrics():
        if not (is_trained and dataset_exists):
            return None

        import tensorflow as tf
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        from sklearn.metrics import (
            classification_report, confusion_matrix, accuracy_score,
            precision_score, recall_score, f1_score
        )

        val_gen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            validation_split=0.2,
        ).flow_from_directory(
            DATASET_DIR,
            target_size=(224, 224),
            batch_size=16,
            classes=["Healthy", "Leaf_Curl_Virus", "Fusarium_Wilt", "Cercospora_Leaf_Spot"],
            class_mode="categorical",
            subset="validation",
            seed=42,
            shuffle=False,
        )

        val_gen.reset()
        y_pred_raw = model.predict(val_gen, verbose=0)
        y_pred     = np.argmax(y_pred_raw, axis=1)
        y_true     = val_gen.classes

        acc  = accuracy_score(y_true, y_pred)
        cm   = confusion_matrix(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average=None, zero_division=0)
        rec  = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1   = f1_score(y_true, y_pred, average=None, zero_division=0)
        mac_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
        mac_rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
        mac_f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)

        return dict(
            acc=acc, cm=cm,
            prec=prec, rec=rec, f1=f1,
            mac_prec=mac_prec, mac_rec=mac_rec, mac_f1=mac_f1,
            n_val=len(y_true),
        )

    metrics = compute_metrics()

    if metrics:
        # ── Overall metrics ────────────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Overall Accuracy",  f"{metrics['acc']:.2%}")
        m2.metric("Macro Precision",   f"{metrics['mac_prec']:.2%}")
        m3.metric("Macro Recall",      f"{metrics['mac_rec']:.2%}")
        m4.metric("Macro F1-Score",    f"{metrics['mac_f1']:.2%}")

        st.caption(f"Evaluated on {metrics['n_val']} validation images (20% stratified hold-out).")

        st.markdown("---")

        # ── Per-class metrics table ────────────────────────────────────────
        st.markdown("#### Per-Class Metrics")
        df = pd.DataFrame({
            "Class":     CLASS_NAMES,
            "Precision": [f"{v:.2%}" for v in metrics["prec"]],
            "Recall":    [f"{v:.2%}" for v in metrics["rec"]],
            "F1-Score":  [f"{v:.2%}" for v in metrics["f1"]],
        })
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown("---")

        # ── Confusion matrix ───────────────────────────────────────────────
        st.markdown("#### Confusion Matrix")
        st.markdown(
            "Rows = **Actual class**, Columns = **Predicted class**. "
            "Diagonal cells (top-left to bottom-right) are correct predictions."
        )

        cm = metrics["cm"].tolist()
        short_labels = ["Healthy", "Leaf Curl\nVirus", "Fusarium\nWilt", "Cercospora\nLeaf Spot"]

        fig_cm = ff.create_annotated_heatmap(
            z=cm,
            x=short_labels,
            y=short_labels,
            colorscale="Blues",
            showscale=True,
        )
        fig_cm.update_layout(
            xaxis_title="Predicted Class",
            yaxis_title="Actual Class",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            height=400,
            margin=dict(t=30, b=80),
        )
        fig_cm.update_xaxes(side="bottom")
        st.plotly_chart(fig_cm, use_container_width=True)

        st.markdown("---")

        # ── Per-class bar chart ────────────────────────────────────────────
        st.markdown("#### Precision / Recall / F1-Score by Class")
        fig_bar = go.Figure()
        for metric_name, values, dash in [
            ("Precision", metrics["prec"], "solid"),
            ("Recall",    metrics["rec"],  "dot"),
            ("F1-Score",  metrics["f1"],   "dash"),
        ]:
            fig_bar.add_trace(go.Bar(
                name=metric_name,
                x=CLASS_NAMES,
                y=values,
                text=[f"{v:.1%}" for v in values],
                textposition="outside",
            ))
        fig_bar.update_layout(
            barmode="group",
            yaxis=dict(range=[0, 1.15], tickformat=".0%", title="Score"),
            xaxis_title="Class",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            height=380,
            legend=dict(orientation="h", y=1.1),
            margin=dict(t=40, b=20),
        )
        fig_bar.update_xaxes(gridcolor="rgba(255,255,255,0.08)")
        fig_bar.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
        st.plotly_chart(fig_bar, use_container_width=True)

    else:
        st.info(
            "Evaluation metrics will appear here once the trained model and dataset are available. "
            "Train the model first using `train.py`, then reload the app.",
            icon="ℹ️",
        )

    # ── Model architecture summary ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🏗️ Model Architecture")
    arch_col1, arch_col2 = st.columns(2)
    with arch_col1:
        st.markdown(
            """
            | Component | Detail |
            |---|---|
            | **Base Model** | MobileNetV2 (ImageNet pretrained) |
            | **Input Size** | 224 × 224 × 3 |
            | **Pooling** | GlobalAveragePooling2D |
            | **Regularization** | Dropout (0.3) |
            | **Output Layer** | Dense (4 units, Softmax) |
            | **Optimizer** | Adam |
            | **Loss Function** | Categorical Cross-Entropy |
            | **Explainability** | GradCAM (last Conv2D layer) |
            """
        )
    with arch_col2:
        st.markdown(
            """
            | Training Detail | Value |
            |---|---|
            | **Transfer Learning** | Phase 1: frozen base → Phase 2: fine-tune last 30 layers |
            | **Data Augmentation** | Rotation, flip, zoom, brightness, shear |
            | **Class Imbalance** | Balanced class weights applied |
            | **Split** | 80% train / 20% validation (stratified) |
            | **Early Stopping** | Patience = 8 epochs |
            | **Learning Rate** | Phase 1: 1e-3 → Phase 2: 1e-4 |
            | **Batch Size** | 16 |
            | **Image Preprocessing** | MobileNetV2 standard (scale to [-1, 1]) |
            """
        )
    st.divider()
    st.caption(
        "**Reference:** Billante, J. & Querioso, M.J. (2024). *Development of an Image-Based System "
        "for Detecting Health Conditions of Sweet Potato (Ipomoea batatas) Leaves Using Convolutional "
        "Neural Networks.* Las Niñas University, Leyte, Philippines."
    )
