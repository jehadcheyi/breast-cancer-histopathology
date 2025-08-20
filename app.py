import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import gradio as gr
import warnings

from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model as keras_load_model

# Suppress noisy warnings (keeps output cleaner)
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.get_logger().setLevel('ERROR')

DEBUG = False

REPO_ID = "jehadcheyi/bc_models"
HF_TOKEN = os.getenv("jehad")

model_paths = {
    'binary': 'binary.h5',
    'benign': 'benign.h5',
    'malignant': 'malignant.h5'
}

img_size = (128, 128)
target_size = (500, 500)

classes = {
    'binary': ['benign', 'malignant'],
    'benign': ['A', 'F', 'PT', 'TA'],
    'malignant': ['DC', 'LC', 'MC', 'PC'],
    'benign_names': {'A': 'Adenosis', 'F': 'Fibroadenoma', 'PT': 'Phyllodes Tumor', 'TA': 'Tubular Adenoma'},
    'malignant_names': {'DC': 'Ductal Carcinoma', 'LC': 'Lobular Carcinoma', 'MC': 'Mucinous Carcinoma', 'PC': 'Papillary Carcinoma'}
}

models = {}

print("🔄 Initializing AI Models...")
print("📋 Verifying model files...")
for name, filename in model_paths.items():
    try:
        local_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            token=HF_TOKEN,
            cache_dir="model_cache"
        )
        print(f"✅ Found {name} model at: {local_path}")
    except Exception as e:
        print(f"❌ Error verifying {name} model: {str(e)}")
        raise

print("\n🧠 Loading AI models into memory...")
try:
    for name, filename in model_paths.items():
        local_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            token=HF_TOKEN,
            cache_dir="model_cache"
        )
        with tf.device('/CPU:0'):
            models[name] = keras_load_model(local_path, compile=False)
        print(f"🎯 Successfully loaded {name} model")
except Exception as e:
    print(f"❌ Model loading failed: {str(e)}")
    raise

print("\n🔬 Setting up Grad-CAM visualization...")
try:
    binary_model = models['binary']
    last_conv = None
    for layer in reversed(binary_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer.name
            break
    if last_conv is None:
        raise ValueError("No convolutional layer found in binary model")
    grad_model = Model(
        inputs=binary_model.inputs,
        outputs=[binary_model.get_layer(last_conv).output, binary_model.output]
    )
    print("✅ Grad-CAM setup complete")
except Exception as e:
    print(f"❌ Grad-CAM setup failed: {str(e)}")
    raise

print("\n🚀 All AI systems loaded successfully!")

def prep_img(img):
    if img is None:
        raise ValueError("No image input provided to prep_img()")
    arr = img.copy()
    if arr.dtype != np.uint8:
        if arr.dtype == np.float32 or arr.dtype == np.float64:
            if arr.max() <= 1.0:
                arr = (arr * 255).astype('uint8')
            else:
                arr = arr.astype('uint8')
        else:
            arr = arr.astype('uint8')
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.concatenate([arr] * 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[..., :3]
    pil = Image.fromarray(arr)
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    pil = pil.resize(img_size, Image.LANCZOS)
    arr = img_to_array(pil).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    if DEBUG:
        print("prep_img -> shape:", arr.shape, "dtype:", arr.dtype, "min/max:", arr.min(), arr.max())
    return arr

def predict(img):
    try:
        arr = prep_img(img)
        with tf.device('/CPU:0'):
            binary_pred = models['binary'].predict(arr, verbose=0)
        pred_idx = int(np.argmax(binary_pred[0]))
        binary_conf = float(100.0 * np.max(binary_pred[0]))
        label = classes['binary'][pred_idx]
        subtype_probs = {}
        with tf.device('/CPU:0'):
            subtype_pred = models[label].predict(arr, verbose=0)
        subtype_idx = int(np.argmax(subtype_pred[0]))
        subtype_conf = float(100.0 * np.max(subtype_pred[0]))
        if label == 'benign':
            subtype_code = classes['benign'][subtype_idx]
            subtype_name = classes['benign_names'][subtype_code]
            for code, prob in zip(classes['benign'], subtype_pred[0]):
                subtype_probs[classes['benign_names'][code]] = float(prob * 100.0)
        else:
            subtype_code = classes['malignant'][subtype_idx]
            subtype_name = classes['malignant_names'][subtype_code]
            for code, prob in zip(classes['malignant'], subtype_pred[0]):
                subtype_probs[classes['malignant_names'][code]] = float(prob * 100.0)
        if DEBUG:
            print("predict -> pred_idx:", pred_idx, "binary_conf:", binary_conf, "subtype:", subtype_name, "subtype_conf:", subtype_conf)
            print("subtype_probs:", subtype_probs)
        return pred_idx, binary_conf, subtype_name, subtype_conf, subtype_probs
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise

def make_heatmap(img, pred_idx):
    """Robust Grad-CAM, works for softmax and sigmoid outputs."""
    # Resize and normalize input for model
    resized = cv2.resize(img, img_size)
    img_in = resized.astype(np.float32) / 255.0
    img_in = np.expand_dims(img_in, axis=0)
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_in)
        if preds.shape[-1] == 1:
            target_label = classes["binary"][pred_idx]
            if target_label == "malignant":
                loss = preds[:, 0]
            else:
                loss = 1.0 - preds[:, 0]
        else:
            loss = preds[:, pred_idx]
    grads = tape.gradient(loss, conv_out)[0]
    pooled = tf.reduce_mean(grads, axis=(0, 1))
    conv_out = conv_out[0]
    heatmap = np.dot(conv_out, pooled.numpy())
    heatmap = np.maximum(heatmap, 0)
    maxval = np.max(heatmap)
    if maxval > 0:
        heatmap = heatmap / maxval
    else:
        heatmap = np.zeros_like(heatmap)
    heatmap = cv2.resize(heatmap, target_size)
    heatmap_uint8 = np.uint8(255 * heatmap)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    return colored

def get_crop_box(shape, percent):
    h, w = shape[:2]
    crop_h = int(h * percent / 100)
    crop_w = int(w * percent / 100)
    left = (w - crop_w) // 2
    top = (h - crop_h) // 2
    return left, top, left + crop_w, top + crop_h

def create_visualization(img, pred_idx, binary_conf, label, subtype_conf, zooms=[75, 50, 25]):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    levels = 1 + len(zooms)
    fig, axs = plt.subplots(levels, 3, figsize=(18, 6 * levels), dpi=100)
    if levels == 1:
        axs = [axs]
    resized = cv2.resize(img, target_size)
    heatmap = make_heatmap(img, pred_idx)
    overlay = cv2.addWeighted(resized, 0.6, heatmap, 0.4, 0)
    resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    axs[0][0].imshow(resized_rgb)
    axs[0][0].set_title("Original Tissue Sample\n(Full View)", fontsize=12, pad=10)
    axs[0][0].axis('off')
    axs[0][1].imshow(heatmap_rgb)
    axs[0][1].set_title("Model Attention Heatmap\n(Red = High Importance)", fontsize=12, pad=10)
    axs[0][1].axis('off')
    axs[0][2].imshow(overlay_rgb)
    axs[0][2].set_title("Attention Overlay\n(Where the model focuses)", fontsize=12, pad=10)
    axs[0][2].axis('off')
    for i, zoom in enumerate(zooms):
        x1, y1, x2, y2 = get_crop_box(img.shape, zoom)
        crop = img[y1:y2, x1:x2]
        resized_crop = cv2.resize(crop, target_size)
        crop_heat = make_heatmap(crop, pred_idx)
        crop_over = cv2.addWeighted(resized_crop, 0.6, crop_heat, 0.4, 0)
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 1
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        highlight = np.where(mask[..., None], img, gray_rgb)
        cv2.rectangle(highlight, (x1, y1), (x2, y2), (255, 0, 0), 3)
        highlight_rgb = cv2.cvtColor(highlight, cv2.COLOR_BGR2RGB)
        resized_crop_rgb = cv2.cvtColor(resized_crop, cv2.COLOR_BGR2RGB)
        crop_heat_rgb = cv2.cvtColor(crop_heat, cv2.COLOR_BGR2RGB)
        crop_over_rgb = cv2.cvtColor(crop_over, cv2.COLOR_BGR2RGB)
        axs[i+1][0].imshow(highlight_rgb)
        axs[i+1][0].set_title(f"Analysis Area: {zoom}% of Image\n(rectangle shows zoom region)", fontsize=12, pad=10)
        axs[i+1][0].axis('off')
        axs[i+1][1].imshow(crop_heat_rgb)
        axs[i+1][1].set_title(f"Zoomed Attention Heatmap\n(Detailed view at {zoom}% scale)", fontsize=12, pad=10)
        axs[i+1][1].axis('off')
        axs[i+1][2].imshow(crop_over_rgb)
        axs[i+1][2].set_title(f"Zoomed Overlay\n(Combined view at {zoom}% scale)", fontsize=12, pad=10)
        axs[i+1][2].axis('off')
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    return fig

def analyze_image(input_img):
    if input_img is None:
        return """
        <div style="background: linear-gradient(45deg, #FFA726, #FF7043); color: white; padding: 25px; border-radius: 15px; text-align: center; margin: 20px 0;">
            <h2 style="margin: 0; font-size: 1.5em;">⚠️ No Image Provided</h2>
            <p style="margin: 10px 0; font-size: 1.1em;">Please upload a histopathology image to begin AI analysis</p>
            <div style="background: rgba(255,255,255,0.2); padding: 10px; border-radius: 8px; margin-top: 15px;">
                <strong>Supported formats:</strong> JPG, PNG, JPEG, TIF
            </div>
        </div>
        """, None, "🟡 **Status**: Waiting for Image Upload"
    try:
        img = input_img.astype('uint8')
        pred_idx, binary_conf, subtype, subtype_conf, subtype_probs = predict(img)
        label = classes['binary'][pred_idx]
        diagnosis = label.upper()
        diagnosis_color = "#4CAF50" if diagnosis == "BENIGN" else "#F44336"
        diagnosis_emoji = "🟢" if diagnosis == "BENIGN" else "🔴"
        if diagnosis == "BENIGN":
            risk_level = "LOW RISK"
            risk_color = "#4CAF50"
            recommendations = """
            • Continue routine screening schedule
            • Monitor for any changes in symptoms
            • Maintain healthy lifestyle practices
            • Follow-up as recommended by physician
            """
        else:
            risk_level = "HIGH RISK - REQUIRES ATTENTION"
            risk_color = "#F44336"
            recommendations = """
            • IMMEDIATE consultation with oncologist required
            • Consider additional diagnostic procedures
            • Discuss treatment options with medical team
            • Seek second opinion if needed
            """
        subtype_table = """
        <div style='background: #f8f9fa; padding: 20px; border-radius: 12px; margin: 15px 0;'>
            <h4 style='color: #2c3e50; margin-bottom: 15px; text-align: center;'>🔬 Detailed Subtype Analysis</h4>
            <table style='width:100%; border-collapse: collapse; font-family: Arial, sans-serif;'>
                <tr style='background: linear-gradient(45deg, #667eea, #764ba2); color: white;'>
                    <th style='padding: 12px; text-align: left; border-radius: 8px 0 0 0;'>Cancer Subtype</th>
                    <th style='padding: 12px; text-align: center;'>Confidence Level</th>
                    <th style='padding: 12px; text-align: center; border-radius: 0 8px 0 0;'>Visual Indicator</th>
                </tr>
        """
        for name, prob in sorted(subtype_probs.items(), key=lambda x: x[1], reverse=True):
            prob_clamped = max(0.0, min(100.0, float(prob)))
            bar_width = int(prob_clamped)
            bar_color = "#4CAF50" if prob_clamped < 30 else "#FF9800" if prob_clamped < 70 else "#F44336"
            subtype_table += f"""
            <tr style='border-bottom: 1px solid #e0e0e0;'>
                <td style='padding: 12px; font-weight: bold;'>{name}</td>
                <td style='padding: 12px; text-align: center; font-weight: bold; color: {bar_color};'>{prob_clamped:.1f}%</td>
                <td style='padding: 12px; text-align: center;'>
                    <div style='background: #e0e0e0; border-radius: 10px; height: 20px; width: 100px; margin: 0 auto; position: relative;'>
                        <div style='background: {bar_color}; height: 100%; width: {bar_width}%; border-radius: 10px; transition: all 0.3s ease;'></div>
                    </div>
                </td>
            </tr>
            """
        subtype_table += "</table></div>"
        diagnosis_html = f"""
        <div style="background: linear-gradient(135deg, {diagnosis_color}15, {diagnosis_color}25); border: 3px solid {diagnosis_color}; border-radius: 20px; padding: 30px; margin: 20px 0; font-family: 'Segoe UI', Arial, sans-serif;">
            <div style="text-align: center; margin-bottom: 25px;">
                <h1 style="color: {diagnosis_color}; margin: 0; font-size: 2.2em;">{diagnosis_emoji} AI DIAGNOSTIC REPORT</h1>
                <div style="background: {diagnosis_color}; color: white; padding: 8px 20px; border-radius: 25px; display: inline-block; margin: 15px 0; font-size: 1.1em; font-weight: bold;">
                    Primary Diagnosis: {diagnosis}
                </div>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 25px;">
                <div style="background: rgba(255,255,255,0.9); padding: 20px; border-radius: 15px; text-align: center;">
                    <h3 style="color: {diagnosis_color}; margin-top: 0;">🎯 Primary Confidence</h3>
                    <div style="font-size: 2.5em; font-weight: bold; color: {diagnosis_color};">{binary_conf:.1f}%</div>
                    <div style="background: {risk_color}; color: white; padding: 5px 15px; border-radius: 20px; margin-top: 10px; font-weight: bold;">
                        {risk_level}
                    </div>
                </div>
                <div style="background: rgba(255,255,255,0.9); padding: 20px; border-radius: 15px; text-align: center;">
                    <h3 style="color: #2980b9; margin-top: 0;">🔬 Most Likely Subtype</h3>
                    <div style="font-size: 1.3em; font-weight: bold; color: #2c3e50; margin: 10px 0;">{subtype}</div>
                    <div style="background: #3498db; color: white; padding: 5px 15px; border-radius: 20px; font-weight: bold;">
                        {subtype_conf:.1f}% Confidence
                    </div>
                </div>
            </div>
            {subtype_table}
            <div style="background: rgba(255,255,255,0.95); padding: 20px; border-radius: 15px; margin-top: 20px;">
                <h3 style="color: #2c3e50; margin-top: 0;">📋 Clinical Recommendations</h3>
                <div style="color: #34495e; line-height: 1.6; font-size: 1.1em;">
                    {recommendations}
                </div>
            </div>
            <div style="background: linear-gradient(45deg, #ff6b6b, #ffa500); color: white; padding: 15px; border-radius: 12px; margin-top: 20px; text-align: center;">
                <strong>⚠️ Medical Disclaimer:</strong> This AI analysis is for professional assistance only. 
                Always consult qualified healthcare professionals for definitive diagnosis and treatment decisions.
            </div>
        </div>
        """
        status_msg = f"{diagnosis_emoji} **Status**: Analysis Complete - {diagnosis} detected with {binary_conf:.1f}% confidence"
        fig = create_visualization(img, pred_idx, binary_conf, label, subtype_conf)
        return diagnosis_html, fig, status_msg
    except Exception as e:
        print(f"Analyze error: {e}")
        error_html = f"""
        <div style="background: linear-gradient(45deg, #e74c3c, #c0392b); color: white; padding: 25px; border-radius: 15px; margin: 20px 0;">
            <h2 style="color: white; margin: 0; text-align: center;">❌ ANALYSIS ERROR</h2>
            <div style="background: rgba(255,255,255,0.2); padding: 15px; border-radius: 10px; margin: 15px 0;">
                <strong>Error Details:</strong> {str(e)}
            </div>
            <div style="text-align: center; margin-top: 20px;">
                <h4>🔧 Troubleshooting Steps:</h4>
                <ul style="text-align: left; max-width: 400px; margin: 0 auto;">
                    <li>Ensure image is a valid histopathology scan</li>
                    <li>Check file format (PNG, JPG, JPEG, TIF supported)</li>
                    <li>Try uploading a different image</li>
                    <li>Contact technical support if issue persists</li>
                </ul>
            </div>
        </div>
        """
        return error_html, None, "🔴 **Status**: Analysis Failed - System Error"

def get_sample_images():
    sample_images = []
    sample_folder = "example"
    if os.path.exists(sample_folder):
        for file in os.listdir(sample_folder):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                try:
                    img_path = os.path.join(sample_folder, file)
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        sample_images.append((img, file))
                except Exception as e:
                    print(f"Error loading sample image {file}: {e}")
    return sample_images

sample_images = get_sample_images()

custom_css = """
/* Global Styles */
.gradio-container { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important; min-height: 100vh; }
/* Header Styling */
.main-header { background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(15px); border-radius: 25px; padding: 40px; margin-bottom: 30px; box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1); border: 1px solid rgba(255, 255, 255, 0.2); text-align: center; }
/* Card Styling */
.analysis-card { background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(15px); border-radius: 20px; padding: 25px; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1); border: 1px solid rgba(255, 255, 255, 0.2); margin: 15px 0; transition: transform 0.3s ease, box-shadow 0.3s ease; }
.analysis-card:hover { transform: translateY(-5px); box-shadow: 0 15px 40px rgba(0, 0, 0, 0.15); }
/* Button Styling */
.analyze-button { background: linear-gradient(45deg, #4CAF50, #45a049) !important; border: none !important; border-radius: 25px !important; padding: 15px 30px !important; font-size: 16px !important; font-weight: bold !important; color: white !important; box-shadow: 0 6px 20px rgba(76, 175, 80, 0.3) !important; transition: all 0.3s ease !important; }
.analyze-button:hover { transform: translateY(-3px) !important; box-shadow: 0 8px 25px rgba(76, 175, 80, 0.4) !important; }
.clear-button { background: linear-gradient(45deg, #FF7043, #FF5722) !important; border: none !important; border-radius: 25px !important; padding: 15px 30px !important; font-size: 16px !important; font-weight: bold !important; color: white !important; box-shadow: 0 6px 20px rgba(255, 87, 34, 0.3) !important; transition: all 0.3s ease !important; }
.clear-button:hover { transform: translateY(-3px) !important; box-shadow: 0 8px 25px rgba(255, 87, 34, 0.4) !important; }
/* Image Containers */
.image-container { border-radius: 15px; overflow: hidden; box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1); transition: transform 0.3s ease; }
.image-container:hover { transform: scale(1.02); }
/* Gallery Styling */
.example-gallery { border-radius: 15px; overflow: hidden; box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1); }
/* Status Indicators */
.status-display { background: rgba(255, 255, 255, 0.9); padding: 15px; border-radius: 12px; margin: 10px 0; text-align: center; font-weight: bold; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1); }
/* Animation */
@keyframes pulse { 0% { transform: scale(1); } 50% { transform: scale(1.05); } 100% { transform: scale(1); } }
.pulse-animation { animation: pulse 2s infinite; }
/* Responsive Design */
@media (max-width: 768px) { .analysis-card { margin: 10px 5px; padding: 20px; } .main-header { padding: 25px; margin-bottom: 20px; } }
"""

title = "🏥 Advanced Histopathology Analysis System"
description = """...""" # Omitted for brevity, unchanged
instructions = """...""" # Omitted for brevity, unchanged
disclaimer = """...""" # Omitted for brevity, unchanged

with gr.Blocks(css=custom_css, theme=gr.themes.Soft(), title="BREAST AI PRO - Advanced Cancer Detection") as demo:
    with gr.Row():
        with gr.Column():
            gr.HTML(f"""
                <div class="main-header">
                    <h1 style="color: #2c3e50; margin-bottom: 15px; font-size: 2.5em;">
                        {title}
                    </h1>
                    {description}
                    {instructions}
                    {disclaimer}
                </div>
            """)
    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            with gr.Group():
                gr.HTML('<div class="analysis-card">')
                gr.Markdown("### 📤 **AI-Powered Image Analysis Center**")
                gr.Markdown("*Upload your histopathology image for comprehensive AI analysis*")
                image_input = gr.Image(
                    label="🔬 Upload Histopathology Image | حمّل صورة | داگرتن  ",
                    type="numpy",
                    height=400,
                    elem_classes=["image-container"]
                )
                gr.Markdown("""
                <div style="background: #e3f2fd; padding: 15px; border-radius: 10px; margin: 10px 0;">
                    <strong>📋 Supported Formats:</strong> JPG, JPEG, PNG, TIF<br>
                    <strong>🎯 Optimal Quality:</strong> High-resolution images (>500x500 pixels)<br>
                    <strong>🔬 Image Types:</strong> H&E stained tissue sections
                </div>
                """)
                with gr.Row():
                    submit_btn = gr.Button(
                        "🔍 Analyze with AI | حلل |  شلوڤەکرن ",
                        variant="primary",
                        size="lg",
                        scale=2,
                        elem_classes=["analyze-button"]
                    )
                    clear_btn = gr.Button(
                        "🧹 Clear ",
                        variant="secondary",
                        size="lg",
                        scale=1,
                        elem_classes=["clear-button"]
                    )
                status_output = gr.Markdown(
                    "🟡 **Status**: System Ready - Upload an image to begin analysis",
                    elem_classes=["status-display"]
                )
                gr.HTML('</div>')
            if sample_images:
                with gr.Group():
                    gr.HTML('<div class="analysis-card">')
                    gr.Markdown("### 🖼️ **Sample Histopathology Images**")
                    gr.Markdown("*Click on any image below to load it for analysis*")
                    gallery_items = [(img, f"Sample: {name}") for img, name in sample_images]
                    example_gallery = gr.Gallery(
                        value=gallery_items,
                        label="Curated histopathology samples from medical database",
                        columns=2,
                        rows=3,
                        height=350,
                        object_fit="cover",
                        elem_classes=["example-gallery"],
                        show_share_button=False,
                        allow_preview=True
                    )
                    gr.HTML('</div>')
        with gr.Column(scale=2):
            with gr.Group():
                gr.HTML('<div class="analysis-card">')
                gr.Markdown("### 📊 **Comprehensive AI Diagnostic Report**")
                diagnosis_output = gr.HTML("""
                <div style='background: linear-gradient(135deg, #e3f2fd, #bbdefb); padding: 30px; border-radius: 20px; text-align: center; margin: 20px 0; border: 2px dashed #2196f3;'>
                    <div style='font-size: 3em; margin-bottom: 15px;'>🔬</div>
                    <h2 style='color: #1976d2; margin: 0; font-size: 1.8em;'>AI Diagnostic System Ready</h2>
                    <p style='color: #424242; font-size: 1.2em; margin: 15px 0;'>
                        Upload a histopathology image to receive comprehensive AI-powered analysis
                    </p>
                    <div style='background: rgba(255,255,255,0.8); padding: 15px; border-radius: 12px; margin-top: 20px;'>
                        <strong>🎯 What You'll Get:</strong><br>
                        • Binary classification (Benign/Malignant)<br>
                        • Detailed subtype identification<br>
                        • Confidence scoring system<br>
                        • Visual attention heatmaps<br>
                        • Clinical recommendations
                    </div>
                </div>
                """)
                gr.Markdown("### 🌡️ **Multi-Scale Attention Visualization**")
                gr.Markdown("*AI attention heatmaps showing where the model focuses during analysis*")
                plot_output = gr.Plot(
                    label="Advanced Grad-CAM Visualization System",
                    elem_classes=["plot-container"]
                )
                gr.HTML("""
                <div style='background: linear-gradient(135deg, #f5f5f5, #e0e0e0); padding: 20px; border-radius: 15px; margin: 15px 0;'>
                    <h3 style='color: #2c3e50; text-align: center; margin-bottom: 20px;'>🎯 Heatmap Interpretation Guide</h3>
                    <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; text-align: center;'>
                        <div style='background: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);'>
                            <div style='display: inline-block; width: 20px; height: 20px; background: #ff0000; border-radius: 50%; margin-bottom: 8px;'></div>
                            <div><strong>Critical Areas</strong></div>
                            <div style='color: #666; font-size: 0.9em;'>High malignancy indicators</div>
                        </div>
                        <div style='background: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);'>
                            <div style='display: inline-block; width: 20px; height: 20px; background: #ff6600; border-radius: 50%; margin-bottom: 8px;'></div>
                            <div><strong>Moderate Risk</strong></div>
                            <div style='color: #666; font-size: 0.9em;'>Suspicious patterns</div>
                        </div>
                        <div style='background: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);'>
                            <div style='display: inline-block; width: 20px; height: 20px; background: #ffcc00; border-radius: 50%; margin-bottom: 8px;'></div>
                            <div><strong>Low Risk</strong></div>
                            <div style='color: #666; font-size: 0.9em;'>Benign characteristics</div>
                        </div>
                        <div style='background: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);'>
                            <div style='display: inline-block; width: 20px; height: 20px; background: #00cc00; border-radius: 50%; margin-bottom: 8px;'></div>
                            <div><strong>Normal Tissue</strong></div>
                            <div style='color: #666; font-size: 0.9em;'>Healthy patterns</div>
                        </div>
                    </div>
                    <div style='background: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 10px; margin-top: 15px; text-align: center;'>
                        <strong>💡 Note:</strong> Heatmaps show AI attention at multiple zoom levels (100%, 75%, 50%, 25%) for comprehensive analysis
                    </div>
                </div>
                """)
                gr.HTML('</div>')
    def select_example(evt: gr.SelectData):
        return sample_images[evt.index][0]
    def clear_interface():
        return None, """
        <div style='background: linear-gradient(135deg, #e3f2fd, #bbdefb); padding: 30px; border-radius: 20px; text-align: center; margin: 20px 0; border: 2px dashed #2196f3;'>
            <div style='font-size: 3em; margin-bottom: 15px;'>🔬</div>
            <h2 style='color: #1976d2; margin: 0; font-size: 1.8em;'>AI Diagnostic System Ready</h2>
            <p style='color: #424242; font-size: 1.2em; margin: 15px 0;'>
                Upload a histopathology image to receive comprehensive AI-powered analysis
            </p>
            <div style='background: rgba(255,255,255,0.8); padding: 15px; border-radius: 12px; margin-top: 20px;'>
                <strong>🎯 What You'll Get:</strong><br>
                • Binary classification (Benign/Malignant)<br>
                • Detailed subtype identification<br>
                • Confidence scoring system<br>
                • Visual attention heatmaps<br>
                • Clinical recommendations
            </div>
        </div>
        """, None, "🟡 **Status**: Interface Cleared - Ready for new analysis"
    if sample_images:
        example_gallery.select(select_example, outputs=image_input)
    submit_btn.click(
        fn=analyze_image,
        inputs=image_input,
        outputs=[diagnosis_output, plot_output, status_output]
    )
    clear_btn.click(
        fn=clear_interface,
        outputs=[image_input, diagnosis_output, plot_output, status_output]
    )

if __name__ == "__main__":
    demo.launch(
        share=True,
        favicon_path=None,
        ssl_verify=False
    )