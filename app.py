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
import logging

# Suppress TensorFlow warnings and configure logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Configure matplotlib for better performance
plt.ioff()  # Turn off interactive mode
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model as keras_load_model

# Hugging Face Hub configuration
REPO_ID = "jehadcheyi/bc_models"
HF_TOKEN = os.getenv("jehad")

# Model paths
model_paths = {
    'binary': 'binary.h5',
    'benign': 'benign.h5', 
    'malignant': 'malignant.h5'
}

# Image and class definitions
img_size = (128, 128)
target_size = (500, 500)

classes = {
    'binary': ['benign', 'malignant'],
    'benign': ['A', 'F', 'PT', 'TA'],
    'malignant': ['DC', 'LC', 'MC', 'PC'],
    'benign_names': {'A': 'Adenosis', 'F': 'Fibroadenoma', 'PT': 'Phyllodes Tumor', 'TA': 'Tubular Adenoma'},
    'malignant_names': {'DC': 'Ductal Carcinoma', 'LC': 'Lobular Carcinoma', 'MC': 'Mucinous Carcinoma', 'PC': 'Papillary Carcinoma'}
}

# Initialize models dictionary
models = {}
grad_model = None

def safe_model_load():
    """Safely load models with proper error handling"""
    global models, grad_model
    
    print("🔄 Initializing AI Models...")
    
    try:
        # 1. Download and verify model files
        print("📋 Downloading model files...")
        model_files = {}
        
        for name, filename in model_paths.items():
            try:
                local_path = hf_hub_download(
                    repo_id=REPO_ID,
                    filename=filename,
                    token=HF_TOKEN,
                    cache_dir="model_cache"
                )
                model_files[name] = local_path
                print(f"✅ Downloaded {name} model")
            except Exception as e:
                print(f"❌ Error downloading {name} model: {str(e)}")
                raise Exception(f"Failed to download {name} model")
        
        # 2. Load models with CPU optimization
        print("\n🧠 Loading AI models...")
        
        # Configure TensorFlow for CPU usage
        tf.config.set_visible_devices([], 'GPU')  # Force CPU usage
        
        for name, local_path in model_files.items():
            try:
                # Load model with custom options
                model = keras_load_model(
                    local_path,
                    compile=False  # Don't compile to avoid metric warnings
                )
                
                # Recompile with basic settings
                model.compile(
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                models[name] = model
                print(f"🎯 Successfully loaded {name} model")
                
            except Exception as e:
                print(f"❌ Error loading {name} model: {str(e)}")
                raise Exception(f"Failed to load {name} model")
        
        # 3. Setup Grad-CAM with error handling
        print("\n🔬 Setting up Grad-CAM visualization...")
        
        try:
            binary_model = models['binary']
            last_conv = None
            
            # Find last convolutional layer
            for layer in reversed(binary_model.layers):
                if 'conv' in layer.name.lower() and hasattr(layer, 'filters'):
                    last_conv = layer.name
                    break
            
            if last_conv is None:
                # Fallback: find any layer with 'conv' in name
                for layer in binary_model.layers:
                    if 'conv' in layer.name.lower():
                        last_conv = layer.name
                        break
            
            if last_conv is None:
                print("⚠️ No convolutional layer found, Grad-CAM will be disabled")
                grad_model = None
            else:
                grad_model = Model(
                    inputs=binary_model.inputs,
                    outputs=[binary_model.get_layer(last_conv).output, binary_model.output]
                )
                print(f"✅ Grad-CAM setup complete using layer: {last_conv}")
                
        except Exception as e:
            print(f"⚠️ Grad-CAM setup failed: {str(e)}")
            print("📊 Continuing without visualization (predictions will still work)")
            grad_model = None
        
        print("\n🚀 All AI systems loaded successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Critical error during model initialization: {str(e)}")
        return False

def prep_img(img):
    """Prepare image for model prediction"""
    try:
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img.astype('uint8'))
        img = img.resize(img_size)
        arr = img_to_array(img) / 255.0
        return np.expand_dims(arr, axis=0)
    except Exception as e:
        raise Exception(f"Image preprocessing failed: {str(e)}")

def predict(img):
    """Make predictions with error handling"""
    try:
        arr = prep_img(img)
        
        # Binary Classification
        binary_pred = models['binary'].predict(arr, verbose=0)
        pred_idx = np.argmax(binary_pred[0])
        binary_conf = 100 * np.max(binary_pred[0])
        label = classes['binary'][pred_idx]

        # Subtype Classification
        subtype_probs = {}
        if label == 'benign':
            subtype_pred = models['benign'].predict(arr, verbose=0)
            subtype_idx = np.argmax(subtype_pred[0])
            subtype_conf = 100 * np.max(subtype_pred[0])
            subtype_code = classes['benign'][subtype_idx]
            subtype_name = classes['benign_names'][subtype_code]
            
            for code, prob in zip(classes['benign'], subtype_pred[0]):
                subtype_probs[classes['benign_names'][code]] = prob*100
        else:
            subtype_pred = models['malignant'].predict(arr, verbose=0)
            subtype_idx = np.argmax(subtype_pred[0])
            subtype_conf = 100 * np.max(subtype_pred[0])
            subtype_code = classes['malignant'][subtype_idx]
            subtype_name = classes['malignant_names'][subtype_code]
            
            for code, prob in zip(classes['malignant'], subtype_pred[0]):
                subtype_probs[classes['malignant_names'][code]] = prob*100

        return pred_idx, binary_conf, subtype_name, subtype_conf, subtype_probs
        
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        raise Exception(f"Prediction failed: {str(e)}")

def make_heatmap(img, pred_idx):
    """Generate Grad-CAM heatmap with error handling"""
    if grad_model is None:
        # Return a placeholder heatmap if Grad-CAM is not available
        placeholder = np.zeros(target_size + (3,), dtype=np.uint8)
        cv2.putText(placeholder, "Visualization Unavailable", (50, 250), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return placeholder
    
    try:
        resized = cv2.resize(img, img_size)
        img_in = resized.astype(np.float32) / 255.0
        img_in = np.expand_dims(img_in, axis=0)

        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(img_in)
            loss = preds[:, pred_idx]

        grads = tape.gradient(loss, conv_out)[0]
        pooled = tf.reduce_mean(grads, axis=(0, 1))
        conv_out = conv_out[0]
        heatmap = conv_out @ pooled[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()

        heatmap = cv2.resize(heatmap, target_size)
        return cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        
    except Exception as e:
        print(f"⚠️ Heatmap generation failed: {str(e)}")
        # Return a placeholder heatmap
        placeholder = np.zeros(target_size + (3,), dtype=np.uint8)
        cv2.putText(placeholder, "Heatmap Error", (150, 250), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return placeholder

def get_crop_box(shape, percent):
    """Calculate centered crop box coordinates"""
    h, w = shape[:2]
    crop_h = int(h * percent / 100)
    crop_w = int(w * percent / 100)
    left = (w - crop_w) // 2
    top = (h - crop_h) // 2
    return left, top, left + crop_w, top + crop_h

def create_visualization(img, pred_idx, binary_conf, label, subtype_conf, zooms=[75, 50, 25]):
    """Create visualization with error handling"""
    try:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]
        levels = 1 + len(zooms)

        fig, axs = plt.subplots(levels, 3, figsize=(18, 6 * levels), dpi=80)
        if levels == 1:
            axs = [axs]

        # Full image analysis
        resized = cv2.resize(img, target_size)
        heatmap = make_heatmap(img, pred_idx)
        overlay = cv2.addWeighted(resized, 0.6, heatmap, 0.4, 0)

        # Convert back to RGB for display
        resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

        axs[0][0].imshow(resized_rgb)
        axs[0][0].set_title("🔬 Original Tissue Sample\n(Full View)", fontsize=12, pad=10, fontweight='bold')
        axs[0][0].axis('off')

        axs[0][1].imshow(heatmap_rgb)
        axs[0][1].set_title("🌡️ AI Attention Heatmap\n(Red = High Importance)", fontsize=12, pad=10, fontweight='bold')
        axs[0][1].axis('off')

        axs[0][2].imshow(overlay_rgb)
        axs[0][2].set_title("🎯 Combined Analysis\n(AI Focus Areas)", fontsize=12, pad=10, fontweight='bold')
        axs[0][2].axis('off')

        # Zoomed analysis
        for i, zoom in enumerate(zooms):
            try:
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

                # Convert to RGB for display
                highlight_rgb = cv2.cvtColor(highlight, cv2.COLOR_BGR2RGB)
                resized_crop_rgb = cv2.cvtColor(resized_crop, cv2.COLOR_BGR2RGB)
                crop_over_rgb = cv2.cvtColor(crop_over, cv2.COLOR_BGR2RGB)

                axs[i+1][0].imshow(highlight_rgb)
                axs[i+1][0].set_title(f"🔍 Analysis Region: {zoom}%", fontsize=10, pad=8)
                axs[i+1][0].axis('off')

                axs[i+1][1].imshow(crop_heat)
                axs[i+1][1].set_title(f"🌡️ Detailed Heatmap ({zoom}%)", fontsize=10, pad=8)
                axs[i+1][1].axis('off')

                axs[i+1][2].imshow(crop_over_rgb)
                axs[i+1][2].set_title(f"🎯 Focused Analysis ({zoom}%)", fontsize=10, pad=8)
                axs[i+1][2].axis('off')
                
            except Exception as e:
                print(f"⚠️ Error creating zoom level {zoom}%: {str(e)}")
                # Fill with placeholder
                for j in range(3):
                    axs[i+1][j].text(0.5, 0.5, f"Zoom {zoom}%\nUnavailable", 
                                   ha='center', va='center', transform=axs[i+1][j].transAxes)
                    axs[i+1][j].axis('off')

        # Main title
        diagnosis = classes['binary'][pred_idx].upper()
        color = '#4CAF50' if diagnosis == 'BENIGN' else '#F44336'

        plt.suptitle(
            f"🏥 AI HISTOPATHOLOGY ANALYSIS\n"
            f"Diagnosis: {diagnosis} ({binary_conf:.1f}% confidence)",
            fontsize=16,
            y=0.98,
            color=color,
            fontweight='bold'
        )

        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        return fig
        
    except Exception as e:
        print(f"❌ Visualization creation failed: {str(e)}")
        # Return a simple error plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.text(0.5, 0.5, f"Visualization Error\n{str(e)}", 
               ha='center', va='center', fontsize=14, color='red')
        ax.axis('off')
        return fig

def analyze_image(input_img):
    """Main analysis function with comprehensive error handling"""
    if input_img is None:
        return """
        <div style="background: linear-gradient(45deg, #FFA726, #FF7043); color: white; padding: 25px; border-radius: 15px; text-align: center; margin: 20px 0;">
            <h2 style="margin: 0; font-size: 1.5em;">⚠️ No Image Provided</h2>
            <p style="margin: 10px 0; font-size: 1.1em;">Please upload a histopathology image to begin AI analysis</p>
        </div>
        """, None, "🟡 **Status**: Waiting for Image Upload"

    try:
        # Validate models are loaded
        if not models:
            return """
            <div style="background: #f44336; color: white; padding: 25px; border-radius: 15px; text-align: center;">
                <h2>❌ SYSTEM ERROR</h2>
                <p>AI models are not loaded. Please refresh the page and try again.</p>
            </div>
            """, None, "🔴 **Status**: System Error - Models Not Loaded"

        img = input_img.astype('uint8')
        pred_idx, binary_conf, subtype, subtype_conf, subtype_probs = predict(img)

        label = classes['binary'][pred_idx]
        diagnosis = label.upper()
        diagnosis_color = "#4CAF50" if diagnosis == "BENIGN" else "#F44336"
        diagnosis_emoji = "🟢" if diagnosis == "BENIGN" else "🔴"
        
        # Risk assessment
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

        # Create subtype analysis table
        subtype_table = """
        <div style='background: #f8f9fa; padding: 20px; border-radius: 12px; margin: 15px 0;'>
            <h4 style='color: #2c3e50; margin-bottom: 15px; text-align: center;'>🔬 Detailed Subtype Analysis</h4>
            <table style='width:100%; border-collapse: collapse; font-family: Arial, sans-serif;'>
                <tr style='background: linear-gradient(45deg, #667eea, #764ba2); color: white;'>
                    <th style='padding: 12px; text-align: left;'>Cancer Subtype</th>
                    <th style='padding: 12px; text-align: center;'>Confidence Level</th>
                    <th style='padding: 12px; text-align: center;'>Visual Indicator</th>
                </tr>
        """
        
        for name, prob in sorted(subtype_probs.items(), key=lambda x: x[1], reverse=True):
            bar_width = max(5, int(prob))  # Minimum 5% width for visibility
            bar_color = "#4CAF50" if prob < 30 else "#FF9800" if prob < 70 else "#F44336"
            
            subtype_table += f"""
            <tr style='border-bottom: 1px solid #e0e0e0;'>
                <td style='padding: 12px; font-weight: bold;'>{name}</td>
                <td style='padding: 12px; text-align: center; font-weight: bold; color: {bar_color};'>{prob:.1f}%</td>
                <td style='padding: 12px; text-align: center;'>
                    <div style='background: #e0e0e0; border-radius: 10px; height: 20px; width: 100px; margin: 0 auto; position: relative;'>
                        <div style='background: {bar_color}; height: 100%; width: {bar_width}%; border-radius: 10px;'></div>
                    </div>
                </td>
            </tr>
            """
        subtype_table += "</table></div>"

        # Main diagnostic report
        diagnosis_html = f"""
        <div style="background: linear-gradient(135deg, {diagnosis_color}15, {diagnosis_color}25); border: 3px solid {diagnosis_color}; border-radius: 20px; padding: 30px; margin: 20px 0;">
            
            <div style="text-align: center; margin-bottom: 25px;">
                <h1 style="color: {diagnosis_color}; margin: 0; font-size: 2.2em;">{diagnosis_emoji} AI DIAGNOSTIC REPORT</h1>
                <div style="background: {diagnosis_color}; color: white; padding: 8px 20px; border-radius: 25px; display: inline-block; margin: 15px 0;">
                    Primary Diagnosis: {diagnosis}
                </div>
            </div>

            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 25px;">
                <div style="background: rgba(255,255,255,0.9); padding: 20px; border-radius: 15px; text-align: center;">
                    <h3 style="color: {diagnosis_color}; margin-top: 0;">🎯 Primary Confidence</h3>
                    <div style="font-size: 2.5em; font-weight: bold; color: {diagnosis_color};">{binary_conf:.1f}%</div>
                    <div style="background: {risk_color}; color: white; padding: 5px 15px; border-radius: 20px; margin-top: 10px;">
                        {risk_level}
                    </div>
                </div>
                
                <div style="background: rgba(255,255,255,0.9); padding: 20px; border-radius: 15px; text-align: center;">
                    <h3 style="color: #2980b9; margin-top: 0;">🔬 Most Likely Subtype</h3>
                    <div style="font-size: 1.3em; font-weight: bold; color: #2c3e50; margin: 10px 0;">{subtype}</div>
                    <div style="background: #3498db; color: white; padding: 5px 15px; border-radius: 20px;">
                        {subtype_conf:.1f}% Confidence
                    </div>
                </div>
            </div>

            {subtype_table}

            <div style="background: rgba(255,255,255,0.95); padding: 20px; border-radius: 15px; margin-top: 20px;">
                <h3 style="color: #2c3e50; margin-top: 0;">📋 Clinical Recommendations</h3>
                <div style="color: #34495e; line-height: 1.6;">{recommendations}</div>
            </div>

            <div style="background: linear-gradient(45deg, #ff6b6b, #ffa500); color: white; padding: 15px; border-radius: 12px; margin-top: 20px; text-align: center;">
                <strong>⚠️ Medical Disclaimer:</strong> This AI analysis is for professional assistance only. 
                Always consult qualified healthcare professionals for definitive diagnosis and treatment decisions.
            </div>
        </div>
        """

        status_msg = f"{diagnosis_emoji} **Status**: Analysis Complete - {diagnosis} detected with {binary_conf:.1f}% confidence"

        # Create visualization
        try:
            fig = create_visualization(img, pred_idx, binary_conf, label, subtype_conf)
        except Exception as viz_error:
            print(f"⚠️ Visualization failed: {str(viz_error)}")
            fig = None

        return diagnosis_html, fig, status_msg

    except Exception as e:
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
                    <li>Check file format (PNG, JPG, JPEG supported)</li>
                    <li>Try uploading a different image</li>
                    <li>Refresh the page if issue persists</li>
                </ul>
            </div>
        </div>
        """
        return error_html, None, "🔴 **Status**: Analysis Failed - System Error"

def get_sample_images():
    """Load sample images with error handling"""
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
                    print(f"⚠️ Error loading sample image {file}: {e}")
    
    return sample_images

# Initialize the system
print("🚀 Starting Breast Cancer AI Analysis System...")

# Load models
model_load_success = safe_model_load()

if not model_load_success:
    print("❌ Failed to initialize AI models. Please check your configuration.")
    # You might want to exit here or show an error interface

# Load sample images
sample_images = get_sample_images()
print(f"📸 Loaded {len(sample_images)} sample images")

# Enhanced CSS with better performance
custom_css = """
.gradio-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    font-family: 'Segoe UI', Arial, sans-serif !important;
}

.main-header {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 20px;
    padding: 30px;
    margin-bottom: 20px;
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
    text-align: center;
}

.analysis-card {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
    margin: 10px 0;
}

.analyze-button {
    background: linear-gradient(45deg, #4CAF50, #45a049) !important;
    border: none !important;
    border-radius: 20px !important;
    padding: 12px 25px !important;
    font-weight: bold !important;
    color: white !important;
}

.clear-button {
    background: linear-gradient(45deg, #FF7043, #FF5722) !important;
    border: none !important;
    border-radius: 20px !important;
    padding: 12px 25px !important;
    font-weight: bold !important;
    color: white !important;
}
"""

# Create the interface
title = "🏥 Advanced Histopathology Analysis System"

description = """
<div style="text-align: center; padding: 20px;">
    <h1 style="color: #2980b9; margin-bottom: 15px;">🔬 AI-Powered Breast Cancer Detection</h1>
    <p style="font-size: 1.2em; color: #34495e; max-width: 800px; margin: 0 auto;">
        Advanced deep learning system for histopathological analysis with real-time diagnosis and visualization
    </p>
</div>
"""

# Create the interface
with gr.Blocks(css=custom_css, theme=gr.themes.Soft(), title="Breast Cancer AI Analysis") as demo:
    
    # Header
    with gr.Row():
        with gr.Column():
            gr.HTML(f"""
                <div class="main-header">
                    <h1 style="color: #2c3e50; margin-bottom: 15px;">{title}</h1>
                    {description}
                </div>
            """)
    
    # Main interface
    with gr.Row():
        # Left column - Input
        with gr.Column(scale=1):
            with gr.Group():
                gr.HTML('<div class="analysis-card">')
                gr.Markdown("### 📤 **Upload & Analyze**")
                
                image_input = gr.Image(
                    label="🔬 Upload Histopathology Image",
                    type="numpy",
                    height=350
                )
                
                with gr.Row():
                    submit_btn = gr.Button(
                        "🔍 Analyze with AI",
                        variant="primary",
                        size="lg",
                        scale=2,
                        elem_classes=["analyze-button"]
                    )
                    clear_btn = gr.Button(
                        "🧹 Clear",
                        variant="secondary",
                        size="lg",
                        scale=1,
                        elem_classes=["clear-button"]
                    )
                
                status_output = gr.Markdown("🟡 **Status**: System Ready")
                gr.HTML('</div>')
            
            # Sample gallery
            if sample_images:
                with gr.Group():
                    gr.HTML('<div class="analysis-card">')
                    gr.Markdown("### 🖼️ **Sample Images**")
                    
                    gallery_items = [(img, f"Sample: {name}") for img, name in sample_images]
                    example_gallery = gr.Gallery(
                        value=gallery_items,
                        label="Click to load sample",
                        columns=2,
                        rows=2,
                        height=300,
                        object_fit="cover"
                    )
                    gr.HTML('</div>')
        
        # Right column - Results
        with gr.Column
