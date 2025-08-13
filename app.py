import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import img_to_array
import gradio as gr
from huggingface_hub import hf_hub_download
import warnings
warnings.filterwarnings('ignore')

# Optimize TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Configuration
REPO_ID = "jehadcheyi/bc_models"
HF_TOKEN = os.getenv("jehad")
IMG_SIZE = (128, 128)
TARGET_SIZE = (400, 400)  # Reduced for faster processing

# Classes
CLASSES = {
    'binary': ['benign', 'malignant'],
    'benign': ['A', 'F', 'PT', 'TA'],
    'malignant': ['DC', 'LC', 'MC', 'PC'],
    'benign_names': {'A': 'Adenosis', 'F': 'Fibroadenoma', 'PT': 'Phyllodes Tumor', 'TA': 'Tubular Adenoma'},
    'malignant_names': {'DC': 'Ductal Carcinoma', 'LC': 'Lobular Carcinoma', 'MC': 'Mucinous Carcinoma', 'PC': 'Papillary Carcinoma'}
}

MODEL_PATHS = {'binary': 'binary.h5', 'benign': 'benign.h5', 'malignant': 'malignant.h5'}

# Load models with progress
@st.cache_resource  # Cache models for faster reloads
def load_models():
    models = {}
    for name, filename in MODEL_PATHS.items():
        try:
            local_path = hf_hub_download(
                repo_id=REPO_ID, 
                filename=filename, 
                token=HF_TOKEN, 
                cache_dir="model_cache"
            )
            models[name] = load_model(local_path, compile=False)  # Skip compilation for faster loading
            print(f"✅ {name} model ready")
        except Exception as e:
            print(f"❌ Error loading {name}: {e}")
            raise
    return models

print("🔄 Initializing AI Models...")
models = load_models()

# Setup Grad-CAM
def setup_gradcam():
    binary_model = models['binary']
    for layer in reversed(binary_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return Model(
                inputs=binary_model.inputs,
                outputs=[binary_model.get_layer(layer.name).output, binary_model.output]
            )
    raise ValueError("No Conv2D layer found")

grad_model = setup_gradcam()
print("✅ All systems ready!")

def preprocess_image(img):
    """Fast image preprocessing"""
    img = cv2.resize(img, IMG_SIZE)
    return np.expand_dims(img.astype(np.float32) / 255.0, axis=0)

def predict_image(img):
    """Optimized prediction function"""
    arr = preprocess_image(img)
    
    # Binary prediction
    binary_pred = models['binary'].predict(arr, verbose=0)
    pred_idx = np.argmax(binary_pred[0])
    binary_conf = float(np.max(binary_pred[0]) * 100)
    label = CLASSES['binary'][pred_idx]
    
    # Subtype prediction
    model_key = label
    subtype_pred = models[model_key].predict(arr, verbose=0)
    subtype_idx = np.argmax(subtype_pred[0])
    subtype_conf = float(np.max(subtype_pred[0]) * 100)
    subtype_code = CLASSES[model_key][subtype_idx]
    subtype_name = CLASSES[f'{model_key}_names'][subtype_code]
    
    # All probabilities
    subtype_probs = {
        CLASSES[f'{model_key}_names'][code]: float(prob * 100)
        for code, prob in zip(CLASSES[model_key], subtype_pred[0])
    }
    
    return pred_idx, binary_conf, subtype_name, subtype_conf, subtype_probs

def generate_gradcam(img, pred_idx):
    """Optimized Grad-CAM generation"""
    img_array = preprocess_image(img)
    
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        loss = preds[:, pred_idx]
    
    grads = tape.gradient(loss, conv_out)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    conv_out = conv_out[0]
    
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    
    if tf.math.reduce_max(heatmap) != 0:
        heatmap = heatmap / tf.math.reduce_max(heatmap)
    
    heatmap_resized = cv2.resize(heatmap.numpy(), TARGET_SIZE)
    return cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

def create_visualization(img, pred_idx):
    """Fast visualization creation"""
    img_resized = cv2.resize(img, TARGET_SIZE)
    heatmap = generate_gradcam(img, pred_idx)
    overlay = cv2.addWeighted(img_resized, 0.6, heatmap, 0.4, 0)
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=80)  # Reduced DPI for speed
    
    axes[0].imshow(img_resized)
    axes[0].set_title("🔬 Original", fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(heatmap)
    axes[1].set_title("🌡️ AI Attention", fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    axes[2].set_title("🎯 Combined", fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    return fig

def analyze_image(input_img):
    """Main analysis function with error handling"""
    if input_img is None:
        return create_waiting_message(), None, "🟡 Upload an image to start"
    
    try:
        # Predictions
        pred_idx, binary_conf, subtype, subtype_conf, subtype_probs = predict_image(input_img)
        diagnosis = CLASSES['binary'][pred_idx].upper()
        
        # Results
        color = "#4CAF50" if diagnosis == "BENIGN" else "#F44336"
        emoji = "🟢" if diagnosis == "BENIGN" else "🔴"
        
        # Create subtype table
        subtype_html = "".join([
            f"<tr><td style='padding:5px;font-weight:bold'>{name}</td>"
            f"<td style='padding:5px;text-align:center;color:{'#4CAF50' if prob<50 else '#F44336'}'>{prob:.1f}%</td></tr>"
            for name, prob in sorted(subtype_probs.items(), key=lambda x: x[1], reverse=True)
        ])
        
        results_html = f"""
        <div style="border: 2px solid {color}; border-radius: 15px; padding: 20px; background: {color}10;">
            <div style="text-align: center; margin-bottom: 15px;">
                <h2 style="color: {color}; margin: 5px 0;">{emoji} {diagnosis}</h2>
                <div style="background: {color}; color: white; padding: 5px 15px; border-radius: 15px; display: inline-block;">
                    <strong>{binary_conf:.1f}% Confidence</strong>
                </div>
            </div>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                <div style="background: white; padding: 15px; border-radius: 10px; text-align: center;">
                    <h4 style="margin: 0 0 10px 0; color: #2980b9;">🔬 Primary Subtype</h4>
                    <div style="font-weight: bold; color: #2c3e50; margin: 5px 0;">{subtype}</div>
                    <div style="color: #3498db; font-weight: bold;">{subtype_conf:.1f}%</div>
                </div>
                
                <div style="background: white; padding: 15px; border-radius: 10px;">
                    <h4 style="margin: 0 0 10px 0; color: #2980b9; text-align: center;">📊 All Types</h4>
                    <table style="width: 100%; font-size: 0.9em;">
                        <tr style="background: #f8f9fa;"><th style="padding:5px;">Type</th><th style="padding:5px;">%</th></tr>
                        {subtype_html}
                    </table>
                </div>
            </div>
            
            <div style="background: #fff3cd; padding: 10px; border-radius: 8px; margin-top: 15px; text-align: center; font-size: 0.9em;">
                <strong>⚠️ Disclaimer:</strong> AI analysis for research only. Consult medical professionals.
            </div>
        </div>
        """
        
        # Visualization
        fig = create_visualization(input_img, pred_idx)
        status = f"{emoji} Complete: {diagnosis} ({binary_conf:.1f}%)"
        
        return results_html, fig, status
        
    except Exception as e:
        error_html = f"""
        <div style="background: #e74c3c; color: white; padding: 20px; border-radius: 15px; text-align: center;">
            <h3>❌ Analysis Error</h3>
            <p>Error: {str(e)}</p>
            <p>Please try a different image.</p>
        </div>
        """
        return error_html, None, "🔴 Analysis failed"

def create_waiting_message():
    return """
    <div style='background: #e3f2fd; padding: 25px; border-radius: 15px; text-align: center; border: 2px dashed #2196f3;'>
        <div style='font-size: 2.5em; margin-bottom: 10px;'>🔬</div>
        <h3 style='color: #1976d2; margin: 10px 0;'>AI System Ready</h3>
        <p style='margin: 10px 0;'>Upload a histopathology image for instant analysis</p>
        <div style='background: rgba(255,255,255,0.8); padding: 10px; border-radius: 8px; margin-top: 15px; font-size: 0.9em;'>
            <strong>Features:</strong> Binary Classification • Subtype Detection • Visual Attention Maps
        </div>
    </div>
    """

# Streamlined CSS
custom_css = """
.gradio-container { 
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important; 
    font-family: 'Segoe UI', Arial, sans-serif !important; 
}
.main-header { 
    background: rgba(255,255,255,0.95); backdrop-filter: blur(10px); 
    border-radius: 15px; padding: 25px; margin-bottom: 15px; 
    box-shadow: 0 8px 25px rgba(0,0,0,0.1); text-align: center; 
}
.card { 
    background: rgba(255,255,255,0.95); backdrop-filter: blur(10px); 
    border-radius: 12px; padding: 15px; box-shadow: 0 6px 20px rgba(0,0,0,0.1); 
    margin: 8px 0; 
}
"""

# Create interface
with gr.Blocks(css=custom_css, theme=gr.themes.Soft(), title="Breast Cancer AI") as demo:
    
    # Compact header
    gr.HTML("""
    <div class="main-header">
        <h1 style="color: #2c3e50; margin: 0 0 10px 0; font-size: 2em;">🏥 Breast Cancer AI Analyzer</h1>
        <p style="color: #34495e; margin: 0; font-size: 1.1em;">Advanced histopathology analysis with deep learning</p>
        <div style="display: flex; justify-content: center; gap: 15px; margin-top: 15px; flex-wrap: wrap;">
            <span style="background: #4CAF50; color: white; padding: 5px 12px; border-radius: 8px; font-size: 0.9em;"><strong>98% Accuracy</strong></span>
            <span style="background: #2196F3; color: white; padding: 5px 12px; border-radius: 8px; font-size: 0.9em;"><strong>Real-time</strong></span>
            <span style="background: #FF9800; color: white; padding: 5px 12px; border-radius: 8px; font-size: 0.9em;"><strong>Multi-scale</strong></span>
        </div>
    </div>
    """)
    
    with gr.Row():
        # Input
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### 📤 Upload & Analyze")
                image_input = gr.Image(label="Histopathology Image", type="numpy", height=250)
                
                with gr.Row():
                    analyze_btn = gr.Button("🔍 Analyze", variant="primary", size="lg")
                    clear_btn = gr.Button("🧹 Clear", variant="secondary")
                
                status_output = gr.Markdown("🟡 **Ready for analysis**")
        
        # Results
        with gr.Column(scale=2):
            with gr.Group():
                gr.Markdown("### 📊 Results & Visualization")
                results_output = gr.HTML(create_waiting_message())
                plot_output = gr.Plot(label="AI Attention Analysis")
    
    # Events
    def clear_all():
        return None, create_waiting_message(), None, "🟡 **Ready for analysis**"
    
    analyze_btn.click(analyze_image, inputs=image_input, outputs=[results_output, plot_output, status_output])
    clear_btn.click(clear_all, outputs=[image_input, results_output, plot_output, status_output])

# Launch
if __name__ == "__main__":
    demo.launch(
        share=True, 
        show_error=True,
        server_name="0.0.0.0",
        server_port=7860,
        quiet=True  # Reduce console output
    )
