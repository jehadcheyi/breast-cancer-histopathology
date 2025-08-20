#!/usr/bin/env python3
"""
app.py — Rewritten application based on the working 'app short.py'.
- Robust image preprocessing (handles grayscale/RGBA)
- Quiet TensorFlow logging and CPU-forced model loading
- Stable Grad-CAM generation with safe normalization
- Clear error handling and fallbacks
- Gradio UI for upload / analyze / clear
"""

import os
import warnings
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import img_to_array
import gradio as gr
from huggingface_hub import hf_hub_download

# Suppress non-critical warnings and TF logs for clean output
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
tf.get_logger().setLevel("ERROR")

# Configuration
REPO_ID = "jehadcheyi/bc_models"
HF_TOKEN = os.getenv("jehad")  # ensure this env var is set
IMG_SIZE = (128, 128)
TARGET_SIZE = (400, 400)

CLASSES = {
    "binary": ["benign", "malignant"],
    "benign": ["A", "F", "PT", "TA"],
    "malignant": ["DC", "LC", "MC", "PC"],
    "benign_names": {
        "A": "Adenosis",
        "F": "Fibroadenoma",
        "PT": "Phyllodes Tumor",
        "TA": "Tubular Adenoma",
    },
    "malignant_names": {
        "DC": "Ductal Carcinoma",
        "LC": "Lobular Carcinoma",
        "MC": "Mucinous Carcinoma",
        "PC": "Papillary Carcinoma",
    },
}

MODEL_PATHS = {"binary": "binary.h5", "benign": "benign.h5", "malignant": "malignant.h5"}

# ------------------------------------------------------------------------------
# Model loading
# ------------------------------------------------------------------------------
def load_models():
    """Download models from HF and load them (CPU to avoid GPU-specific logs)."""
    if HF_TOKEN is None:
        raise EnvironmentError("Hugging Face token not found. Set environment variable 'jehad'.")

    models = {}
    print("🚀 Initializing AI Models...")

    for key, fname in MODEL_PATHS.items():
        try:
            print(f"📥 Downloading {key} model...")
            local_path = hf_hub_download(repo_id=REPO_ID, filename=fname, token=HF_TOKEN, cache_dir="model_cache")
            print(f"✅ Found {key} at {local_path}. Loading...")
            # Force CPU device while loading to reduce GPU-related warnings in many environments
            with tf.device("/CPU:0"):
                models[key] = load_model(local_path, compile=False)
            print(f"🎯 {key} model loaded")
        except Exception as e:
            print(f"❌ Error loading {key} model: {e}")
            raise

    return models


models = load_models()

# ------------------------------------------------------------------------------
# Grad-CAM setup
# ------------------------------------------------------------------------------
def setup_gradcam(binary_model):
    """Return a model that outputs last conv layer and predictions (for Grad-CAM)."""
    last_conv_name = None
    for layer in reversed(binary_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_name = layer.name
            break
    if last_conv_name is None:
        raise ValueError("No Conv2D layer found in binary model for Grad-CAM.")
    grad_model = Model(inputs=binary_model.inputs, outputs=[binary_model.get_layer(last_conv_name).output, binary_model.output])
    return grad_model


grad_model = setup_gradcam(models["binary"])
print("🎯 Grad-CAM ready — All systems operational")

# ------------------------------------------------------------------------------
# Image preprocessing and prediction
# ------------------------------------------------------------------------------
def preprocess_image(img):
    """
    Convert input numpy image to an RGB, resized, normalized array suitable for model.predict.
    Handles grayscale and RGBA images.
    """
    if img is None:
        raise ValueError("No image provided to preprocess_image()")

    arr = img.copy()

    # If grayscale, stack channels
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    # If single-channel shaped (H, W, 1)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.concatenate([arr] * 3, axis=2)
    # If RGBA, drop alpha
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[..., :3]

    # Convert to PIL to safely ensure RGB and resize with good interpolation
    pil = Image.fromarray(arr.astype("uint8"))
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    pil = pil.resize(IMG_SIZE, Image.LANCZOS)
    arr = img_to_array(pil).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict_image(img):
    """Run binary + subtype prediction; returns index, confidences and subtype probabilities."""
    arr = preprocess_image(img)

    # Binary prediction
    with tf.device("/CPU:0"):
        binary_pred = models["binary"].predict(arr, verbose=0)

    pred_idx = int(np.argmax(binary_pred[0]))
    binary_conf = float(np.max(binary_pred[0]) * 100.0)
    label = CLASSES["binary"][pred_idx]  # 'benign' or 'malignant'

    # Subtype prediction
    with tf.device("/CPU:0"):
        subtype_pred = models[label].predict(arr, verbose=0)

    subtype_idx = int(np.argmax(subtype_pred[0]))
    subtype_conf = float(np.max(subtype_pred[0]) * 100.0)
    subtype_code = CLASSES[label][subtype_idx]
    subtype_name = CLASSES[f"{label}_names"][subtype_code]

    # Build subtype probabilities dict (human names)
    subtype_probs = {
        CLASSES[f"{label}_names"][code]: float(prob * 100.0)
        for code, prob in zip(CLASSES[label], subtype_pred[0])
    }

    return pred_idx, binary_conf, subtype_name, subtype_conf, subtype_probs

# ------------------------------------------------------------------------------
# Grad-CAM generation
# ------------------------------------------------------------------------------
def generate_gradcam(img, pred_idx):
    """
    Generate a colored heatmap (in RGB) for the predicted class index.
    Returns a uint8 (H,W,3) heatmap in BGR (OpenCV) format.
    """
    try:
        arr = preprocess_image(img)  # shape (1, H, W, 3)

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(arr)
            loss = predictions[:, pred_idx]

        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            # If gradient flow failed, return empty map
            return np.zeros((*TARGET_SIZE, 3), dtype=np.uint8)

        grads = grads[0]  # (h, w, channels)
        conv_out = conv_outputs[0]  # (h, w, channels)

        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1))  # (channels,)

        # Weighted sum of conv feature maps
        heatmap = tf.reduce_sum(conv_out * pooled_grads, axis=-1)

        # ReLU and normalize
        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.reduce_max(heatmap)
        if max_val.numpy() > 0:
            heatmap = heatmap / max_val

        heatmap = heatmap.numpy()
        heatmap_resized = cv2.resize(heatmap, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)

        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)  # BGR
        return colored

    except Exception as e:
        print(f"Grad-CAM generation error: {e}")
        return np.zeros((*TARGET_SIZE, 3), dtype=np.uint8)

# ------------------------------------------------------------------------------
# Visualization
# ------------------------------------------------------------------------------
def create_visualization(img, pred_idx):
    """Return a matplotlib Figure showing original, heatmap and overlay."""
    try:
        # Ensure image is RGB uint8 and resized for display
        if img.ndim == 2:
            img_rgb = np.stack([img] * 3, axis=-1)
        elif img.shape[2] == 4:
            img_rgb = img[..., :3]
        else:
            img_rgb = img.copy()

        # Resize for visualization
        disp_img = cv2.resize(img_rgb, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
        # Gradio images often are RGB; OpenCV uses BGR for blending/colormap
        disp_bgr = cv2.cvtColor(disp_img.astype("uint8"), cv2.COLOR_RGB2BGR)

        heatmap_bgr = generate_gradcam(img, pred_idx)  # BGR uint8

        if heatmap_bgr.sum() > 0:
            overlay_bgr = cv2.addWeighted(disp_bgr, 0.6, heatmap_bgr, 0.4, 0)
        else:
            overlay_bgr = disp_bgr

        # Convert to RGB for Matplotlib
        disp_rgb = cv2.cvtColor(disp_bgr, cv2.COLOR_BGR2RGB)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=100)
        fig.patch.set_facecolor("white")

        axes[0].imshow(disp_rgb)
        axes[0].set_title("🔬 Original Image", fontsize=12, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(heatmap_rgb)
        axes[1].set_title("🌡️ Attention Map", fontsize=12, fontweight="bold")
        axes[1].axis("off")

        axes[2].imshow(overlay_rgb)
        axes[2].set_title("🎯 Overlay", fontsize=12, fontweight="bold")
        axes[2].axis("off")

        plt.tight_layout()
        return fig

    except Exception as e:
        print(f"Visualization creation error: {e}")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, f"Visualization Error\n{str(e)}", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return fig

# ------------------------------------------------------------------------------
# Main analysis function (used by Gradio)
# ------------------------------------------------------------------------------
def create_waiting_message():
    return """
    <div style='background: linear-gradient(135deg, #e3f2fd, #bbdefb); padding: 20px; border-radius: 12px;'>
        <h3 style='color:#1976d2;'>AI Diagnostic System Ready</h3>
        <p>Upload a histopathology image to receive analysis (binary + subtype + attention map).</p>
    </div>
    """


def analyze_image(input_img):
    """Primary function called from front-end."""
    if input_img is None:
        return create_waiting_message(), None, "🟡 Status: Waiting for image"

    try:
        # Gradio provides numpy arrays in RGB order, keep that consistent
        img = input_img.astype("uint8")

        pred_idx, binary_conf, subtype, subtype_conf, subtype_probs = predict_image(img)

        diagnosis_label = CLASSES["binary"][pred_idx].upper()
        color = "#4CAF50" if diagnosis_label == "BENIGN" else "#F44336"
        emoji = "🟢" if diagnosis_label == "BENIGN" else "🔴"
        risk_level = "LOW RISK" if diagnosis_label == "BENIGN" else "HIGH RISK"

        # Build subtype rows HTML
        subtype_rows = ""
        for name, prob in sorted(subtype_probs.items(), key=lambda x: x[1], reverse=True):
            pc = min(100, float(prob))
            bar_color = "#4CAF50" if pc < 40 else "#FF9800" if pc < 70 else "#F44336"
            subtype_rows += f"""
            <tr style="border-bottom:1px solid #eee;">
                <td style="padding:8px;">{name}</td>
                <td style="padding:8px; text-align:center; color:{bar_color}; font-weight:bold;">{pc:.1f}%</td>
                <td style="padding:8px; width:120px;">
                    <div style="background:#f0f0f0; border-radius:8px; height:10px;">
                        <div style="width:{pc}%; background:{bar_color}; height:100%; border-radius:8px;"></div>
                    </div>
                </td>
            </tr>
            """

        results_html = f"""
        <div style="border:3px solid {color}; border-radius:12px; padding:18px; background:linear-gradient(135deg, {color}10, {color}05);">
            <div style="text-align:center;">
                <h2 style="color:{color}; margin:0;">{emoji} {diagnosis_label}</h2>
                <p style="margin:6px 0; font-weight:bold;">{binary_conf:.1f}% • {risk_level}</p>
            </div>
            <div style="display:flex; gap:12px; margin-top:12px;">
                <div style="flex:1; background:white; border-radius:8px; padding:10px;">
                    <h4 style="margin-top:0;">Primary subtype</h4>
                    <div style="font-weight:bold;">{subtype}</div>
                    <div style="margin-top:6px; background:#3498db; color:white; display:inline-block; padding:4px 10px; border-radius:12px;">{subtype_conf:.1f}%</div>
                </div>
                <div style="flex:2; background:white; border-radius:8px; padding:10px;">
                    <h4 style="margin-top:0;">All subtypes</h4>
                    <table style="width:100%; border-collapse:collapse; font-size:0.9em;">{subtype_rows}</table>
                </div>
            </div>
            <div style="margin-top:12px; background:#fff3cd; padding:8px; border-radius:6px;">
                <strong>Disclaimer:</strong> AI output for research/educational purposes. Consult clinicians for decisions.
            </div>
        </div>
        """

        fig = create_visualization(img, pred_idx)
        status = f"{emoji} Analysis complete — {diagnosis_label} ({binary_conf:.1f}%)"
        return results_html, fig, status

    except Exception as e:
        print(f"Analyze error: {e}")
        err_html = f"""
        <div style='background:linear-gradient(45deg,#e74c3c,#c0392b); color:white; padding:12px; border-radius:8px;'>
            <h3 style="margin:0;">❌ Analysis Error</h3>
            <p style="margin-top:8px;">{str(e)}</p>
        </div>
        """
        return err_html, None, "🔴 Analysis failed"

# ------------------------------------------------------------------------------
# Gradio UI
# ------------------------------------------------------------------------------
custom_css = """
.gradio-container { background: linear-gradient(135deg,#667eea,#764ba2); font-family:Segoe UI, Arial; }
.main-header { background: rgba(255,255,255,0.95); border-radius:12px; padding:16px; margin-bottom:12px; text-align:center; }
.analysis-section { background: rgba(255,255,255,0.95); padding:12px; border-radius:10px; }
.btn-primary { background: linear-gradient(45deg,#4CAF50,#45a049) !important; }
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft(), title="Breast Cancer AI Analyzer") as demo:
    gr.HTML("""
    <div class="main-header">
        <h1>🏥 Breast Cancer AI Analyzer</h1>
        <p>Binary + subtype detection with Grad-CAM attention visualization</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                gr.HTML('<div class="analysis-section">')
                gr.Markdown("### Upload histopathology image")
                image_input = gr.Image(label="Histopathology Image", type="numpy", sources=["upload", "clipboard"])
                with gr.Row():
                    analyze_btn = gr.Button("🔍 Analyze", variant="primary", elem_classes=["btn-primary"])
                    clear_btn = gr.Button("🧹 Clear")
                status_output = gr.Markdown("🟡 Status: Ready")
                gr.HTML('</div>')

        with gr.Column(scale=2):
            with gr.Group():
                gr.HTML('<div class="analysis-section">')
                gr.Markdown("### Results")
                results_output = gr.HTML(create_waiting_message())
                gr.Markdown("### Visual Attention")
                plot_output = gr.Plot(label="Grad-CAM Visualization")
                gr.HTML('</div>')

    def clear_interface():
        return None, create_waiting_message(), None, "🟡 Status: Cleared"

    analyze_btn.click(fn=analyze_image, inputs=image_input, outputs=[results_output, plot_output, status_output])
    clear_btn.click(fn=clear_interface, outputs=[image_input, results_output, plot_output, status_output])

# ------------------------------------------------------------------------------
# Launch
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    print("🚀 Starting application...")
    demo.launch(share=True, show_error=True, server_name="0.0.0.0", server_port=7860)