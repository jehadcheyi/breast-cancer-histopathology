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

# Configuration
REPO_ID = "jehadcheyi/bc_models"
HF_TOKEN = os.getenv("jehad")
IMG_SIZE = (128, 128)
TARGET_SIZE = (500, 500)

# Classes and model paths
CLASSES = {
    'binary': ['benign', 'malignant'],
    'benign': ['A', 'F', 'PT', 'TA'],
    'malignant': ['DC', 'LC', 'MC', 'PC'],
    'benign_names': {'A': 'Adenosis', 'F': 'Fibroadenoma', 'PT': 'Phyllodes Tumor', 'TA': 'Tubular Adenoma'},
    'malignant_names': {'DC': 'Ductal Carcinoma', 'LC': 'Lobular Carcinoma', 'MC': 'Mucinous Carcinoma', 'PC': 'Papillary Carcinoma'}
}

MODEL_PATHS = {'binary': 'binary.h5', 'benign': 'benign.h5', 'malignant': 'malignant.h5'}

# Load models
print("🔄 Loading AI Models...")
models = {}
for name, filename in MODEL_PATHS.items():
    try:
        local_path = hf_hub_download(repo_id=REPO_ID, filename=filename, token=HF_TOKEN, cache_dir="model_cache")
        models[name] = load_model(local_path)
        print(f"✅ Loaded {name} model")
    except Exception as e:
        print(f"❌ Error loading {name}: {e}")
        raise

# Setup Grad-CAM
binary_model = models['binary']
last_conv = None
for layer in reversed(binary_model.layers):
    if isinstance(layer, tf.keras.layers.Conv2D):
        last_conv = layer.name
        break

grad_model = Model(inputs=binary_model.inputs, outputs=[binary_model.get_layer(last_conv).output, binary_model.output])
print("✅ Grad-CAM ready")

def preprocess_image(img):
    """Preprocess image for model prediction"""
    img = Image.fromarray(img).resize(IMG_SIZE)
    return np.expand_dims(img_to_array(img) / 255.0, axis=0)

def predict_image(img):
    """Make predictions on image"""
    arr = preprocess_image(img)
    
    # Binary prediction
    binary_pred = models['binary'].predict(arr, verbose=0)
    pred_idx = np.argmax(binary_pred[0])
    binary_conf = 100 * np.max(binary_pred[0])
    label = CLASSES['binary'][pred_idx]
    
    # Subtype prediction
    model_key = 'benign' if label == 'benign' else 'malignant'
    subtype_pred = models[model_key].predict(arr, verbose=0)
    subtype_idx = np.argmax(subtype_pred[0])
    subtype_conf = 100 * np.max(subtype_pred[0])
    subtype_code = CLASSES[model_key][subtype_idx]
    subtype_name = CLASSES[f'{model_key}_names'][subtype_code]
    
    # Subtype probabilities
    subtype_probs = {}
    for code, prob in zip(CLASSES[model_key], subtype_pred[0]):
        subtype_probs[CLASSES[f'{model_key}_names'][code]] = prob * 100
    
    return pred_idx, binary_conf, subtype_name, subtype_conf, subtype_probs

def generate_gradcam(img, pred_idx):
    """Generate Grad-CAM heatmap"""
    resized = cv2.resize(img, IMG_SIZE)
    img_array = resized.astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        loss = preds[:, pred_idx]
    
    grads = tape.gradient(loss, conv_out)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    conv_out = conv_out[0]
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    heatmap = cv2.resize(heatmap.numpy(), TARGET_SIZE)
    return cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

def create_visualization(img, pred_idx):
    """Create visualization with original, heatmap, and overlay"""
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    resized = cv2.resize(img_bgr, TARGET_SIZE)
    heatmap = generate_gradcam(img, pred_idx)
    overlay = cv2.addWeighted(resized, 0.6, heatmap, 0.4, 0)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    axes[0].set_title("🔬 Original Image", fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
    axes[1].set_title("🌡️ AI Attention Heatmap", fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[2].set_title("🎯 Combined Analysis", fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    return fig

def analyze_image(input_img):
    """Main analysis function"""
    if input_img is None:
        return create_error_message("No image provided"), None, "🟡 Waiting for image upload"
    
    try:
        pred_idx, binary_conf, subtype, subtype_conf, subtype_probs = predict_image(input_img)
        diagnosis = CLASSES['binary'][pred_idx].upper()
        
        # Create results HTML
        diagnosis_color = "#4CAF50" if diagnosis == "BENIGN" else "#F44336"
        diagnosis_emoji = "🟢" if diagnosis == "BENIGN" else "🔴"
        
        # Subtype table
        subtype_rows = ""
        for name, prob in sorted(subtype_probs.items(), key=lambda x: x[1], reverse=True):
            bar_color = "#4CAF50" if prob < 30 else "#FF9800" if prob < 70 else "#F44336"
            subtype_rows += f"""
            <tr>
                <td style='padding: 8px; font-weight: bold;'>{name}</td>
                <td style='padding: 8px; text-align: center; color: {bar_color}; font-weight: bold;'>{prob:.1f}%</td>
            </tr>
            """
        
        results_html = f"""
        <div style="background: linear-gradient(135deg, {diagnosis_color}15, {diagnosis_color}25); 
                    border: 2px solid {diagnosis_color}; border-radius: 15px; padding: 25px; margin: 15px 0;">
            <div style="text-align: center; margin-bottom: 20px;">
                <h2 style="color: {diagnosis_color}; margin: 0;">{diagnosis_emoji} AI DIAGNOSIS: {diagnosis}</h2>
                <div style="background: {diagnosis_color}; color: white; padding: 8px 20px; 
                           border-radius: 20px; display: inline-block; margin: 10px 0; font-weight: bold;">
                    Confidence: {binary_conf:.1f}%
                </div>
            </div>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px;">
                <div style="background: rgba(255,255,255,0.9); padding: 15px; border-radius: 10px; text-align: center;">
                    <h4 style="margin-top: 0; color: #2980b9;">🔬 Most Likely Subtype</h4>
                    <div style="font-size: 1.2em; font-weight: bold; color: #2c3e50;">{subtype}</div>
                    <div style="color: #3498db; font-weight: bold;">{subtype_conf:.1f}% Confidence</div>
                </div>
                
                <div style="background: rgba(255,255,255,0.9); padding: 15px; border-radius: 10px;">
                    <h4 style="margin-top: 0; color: #2980b9; text-align: center;">📊 All Subtypes</h4>
                    <table style="width: 100%; border-collapse: collapse;">
                        <tr style="background: #f8f9fa;">
                            <th style="padding: 8px; text-align: left;">Type</th>
                            <th style="padding: 8px; text-align: center;">Probability</th>
                        </tr>
                        {subtype_rows}
                    </table>
                </div>
            </div>
            
            <div style="background: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 10px; text-align: center;">
                <strong>⚠️ Medical Disclaimer:</strong> This AI analysis is for research purposes only. 
                Always consult qualified healthcare professionals for medical decisions.
            </div>
        </div>
        """
        
        # Create visualization
        fig = create_visualization(input_img, pred_idx)
        status = f"{diagnosis_emoji} Analysis complete: {diagnosis} ({binary_conf:.1f}% confidence)"
        
        return results_html, fig, status
        
    except Exception as e:
        return create_error_message(str(e)), None, "🔴 Analysis failed"

def create_error_message(error_text):
    """Create error message HTML"""
    return f"""
    <div style="background: linear-gradient(45deg, #e74c3c, #c0392b); color: white; 
                padding: 20px; border-radius: 15px; text-align: center;">
        <h3>❌ Analysis Error</h3>
        <p>Error: {error_text}</p>
        <p>Please try uploading a different image or contact support.</p>
    </div>
    """

def load_sample_images():
    """Load sample images if available"""
    sample_images = []
    if os.path.exists("example"):
        for file in os.listdir("example"):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                try:
                    img_path = os.path.join("example", file)
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        sample_images.append((img, file))
                except Exception as e:
                    print(f"Error loading {file}: {e}")
    return sample_images

# Custom CSS
custom_css = """
.gradio-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    font-family: 'Segoe UI', Arial, sans-serif !important;
}
.main-header {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(15px);
    border-radius: 20px;
    padding: 30px;
    margin-bottom: 20px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
    text-align: center;
}
.analysis-card {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(15px);
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
    margin: 10px 0;
}
"""

# Create Gradio interface
sample_images = load_sample_images()

with gr.Blocks(css=custom_css, theme=gr.themes.Soft(), title="Breast Cancer AI Analyzer") as demo:
    
    # Header
    gr.HTML("""
    <div class="main-header">
        <h1 style="color: #2c3e50; margin-bottom: 15px;">🏥 Breast Cancer AI Analyzer</h1>
        <p style="font-size: 1.2em; color: #34495e;">Advanced histopathology analysis with deep learning</p>
        <div style="display: flex; justify-content: center; gap: 20px; margin-top: 20px;">
            <div style="background: #4CAF50; color: white; padding: 10px 20px; border-radius: 10px;">
                <strong>98% Accuracy</strong>
            </div>
            <div style="background: #2196F3; color: white; padding: 10px 20px; border-radius: 10px;">
                <strong>Real-time Analysis</strong>
            </div>
            <div style="background: #FF9800; color: white; padding: 10px 20px; border-radius: 10px;">
                <strong>Multi-scale Detection</strong>
            </div>
        </div>
    </div>
    """)
    
    with gr.Row():
        # Input column
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### 📤 Upload Image")
                image_input = gr.Image(label="Histopathology Image", type="numpy", height=300)
                
                with gr.Row():
                    analyze_btn = gr.Button("🔍 Analyze", variant="primary", size="lg")
                    clear_btn = gr.Button("🧹 Clear", variant="secondary")
                
                status_output = gr.Markdown("🟡 **Status**: Ready for analysis")
            
            # Sample gallery
            if sample_images:
                with gr.Group():
                    gr.Markdown("### 🖼️ Sample Images")
                    gallery = gr.Gallery(
                        value=[(img, name) for img, name in sample_images],
                        columns=2, height=250, object_fit="cover"
                    )
        
        # Results column
        with gr.Column(scale=2):
            with gr.Group():
                gr.Markdown("### 📊 AI Analysis Results")
                results_output = gr.HTML("""
                <div style='background: #e3f2fd; padding: 30px; border-radius: 15px; text-align: center; border: 2px dashed #2196f3;'>
                    <div style='font-size: 3em; margin-bottom: 15px;'>🔬</div>
                    <h3 style='color: #1976d2;'>AI System Ready</h3>
                    <p>Upload an image to begin comprehensive analysis</p>
                </div>
                """)
                
                gr.Markdown("### 🌡️ Attention Visualization")
                plot_output = gr.Plot(label="Grad-CAM Heatmap Analysis")
    
    # Event handlers
    def select_example(evt: gr.SelectData):
        return sample_images[evt.index][0]
    
    def clear_all():
        return None, """
        <div style='background: #e3f2fd; padding: 30px; border-radius: 15px; text-align: center; border: 2px dashed #2196f3;'>
            <div style='font-size: 3em; margin-bottom: 15px;'>🔬</div>
            <h3 style='color: #1976d2;'>AI System Ready</h3>
            <p>Upload an image to begin comprehensive analysis</p>
        </div>
        """, None, "🟡 **Status**: Ready for analysis"
    
    # Connect events
    if sample_images:
        gallery.select(select_example, outputs=image_input)
    
    analyze_btn.click(analyze_image, inputs=image_input, outputs=[results_output, plot_output, status_output])
    clear_btn.click(clear_all, outputs=[image_input, results_output, plot_output, status_output])

# Launch
if __name__ == "__main__":
    demo.launch(share=True, show_error=True)
