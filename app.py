import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from huggingface_hub import hf_hub_download
import gradio as gr

# Configuration
REPO_ID = "jehadcheyi/bc_models"
MODEL_PATHS = {
    'binary': 'binary.h5',
    'benign': 'benign.h5', 
    'malignant': 'malignant.h5'
}

IMG_SIZE = (128, 128)
TARGET_SIZE = (500, 500)

CLASSES = {
    'binary': ['benign', 'malignant'],
    'benign': ['A', 'F', 'PT', 'TA'],
    'malignant': ['DC', 'LC', 'MC', 'PC'],
    'benign_names': {
        'A': 'Adenosis', 
        'F': 'Fibroadenoma', 
        'PT': 'Phyllodes Tumor', 
        'TA': 'Tubular Adenoma'
    },
    'malignant_names': {
        'DC': 'Ductal Carcinoma', 
        'LC': 'Lobular Carcinoma', 
        'MC': 'Mucinous Carcinoma', 
        'PC': 'Papillary Carcinoma'
    }
}

# Initialize models
models = {}

def load_models():
    """Load all required models from Hugging Face Hub"""
    print("🔄 Loading models...")
    try:
        for name, filename in MODEL_PATHS.items():
            model_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                cache_dir="model_cache"
            )
            models[name] = load_model(model_path)
            print(f"✅ Loaded {name} model from {model_path}")
        
        # Setup Grad-CAM
        setup_gradcam()
        return True
    except Exception as e:
        print(f"❌ Model loading failed: {str(e)}")
        return False

def setup_gradcam():
    """Configure Grad-CAM visualization model"""
    binary_model = models['binary']
    last_conv = None
    
    for layer in reversed(binary_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer.name
            break
    
    if last_conv is None:
        raise ValueError("No convolutional layer found in binary model")
    
    global grad_model
    grad_model = Model(
        inputs=binary_model.inputs,
        outputs=[binary_model.get_layer(last_conv).output, binary_model.output]
    )
    print(f"🎯 Grad-CAM configured using layer: {last_conv}")

def preprocess_image(img):
    """Convert and resize input image to model format"""
    if isinstance(img, Image.Image):
        img = np.array(img)
    elif not isinstance(img, np.ndarray):
        raise ValueError("Input must be PIL Image or numpy array")
    
    if len(img.shape) == 2:  # Grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:  # RGBA
        img = img[..., :3]
    
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype('float32') / 255.0
    return np.expand_dims(img, axis=0)

def predict(image):
    """Run model prediction pipeline"""
    try:
        # Preprocess
        img_array = preprocess_image(image)
        
        # Binary prediction
        binary_pred = models['binary'].predict(img_array, verbose=0)[0]
        pred_idx = int(np.argmax(binary_pred))
        binary_conf = float(np.max(binary_pred)) * 100
        label = CLASSES['binary'][pred_idx]
        
        # Subtype prediction
        subtype_model = 'benign' if label == 'benign' else 'malignant'
        subtype_pred = models[subtype_model].predict(img_array, verbose=0)[0]
        subtype_idx = int(np.argmax(subtype_pred))
        subtype_conf = float(np.max(subtype_pred)) * 100
        subtype_code = CLASSES[subtype_model][subtype_idx]
        subtype_name = CLASSES[f"{subtype_model}_names"][subtype_code]
        
        # Prepare probabilities
        subtype_probs = {}
        for code, prob in zip(CLASSES[subtype_model], subtype_pred):
            name = CLASSES[f"{subtype_model}_names"][code]
            subtype_probs[name] = float(prob) * 100
        
        return {
            'binary_label': label,
            'binary_confidence': binary_conf,
            'subtype_name': subtype_name,
            'subtype_confidence': subtype_conf,
            'subtype_probs': subtype_probs,
            'pred_idx': pred_idx
        }
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        raise

def generate_heatmap(img, pred_idx):
    """Generate Grad-CAM heatmap"""
    img_array = preprocess_image(img)[0]  # Remove batch dim
    img_array = img_array.astype('float32')
    
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(np.expand_dims(img_array, axis=0))
        loss = predictions[:, pred_idx]
    
    grads = tape.gradient(loss, conv_output)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    conv_output = conv_output[0]
    heatmap = tf.matmul(conv_output, pooled_grads[..., tf.newaxis])
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    
    # Resize and colorize
    heatmap = cv2.resize(heatmap, TARGET_SIZE)
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

def create_visualization(img, pred_idx, binary_conf, subtype_conf):
    """Create multi-scale visualization"""
    img = np.array(img)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    # Full image analysis
    resized = cv2.resize(img, TARGET_SIZE)
    heatmap = generate_heatmap(img, pred_idx)
    overlay = cv2.addWeighted(resized, 0.6, heatmap, 0.4, 0)
    
    # Create figure
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    ax[0].set_title("Original Image", fontweight='bold')
    ax[0].axis('off')
    
    ax[1].imshow(heatmap)
    ax[1].set_title("AI Attention Heatmap", fontweight='bold')
    ax[1].axis('off')
    
    ax[2].imshow(overlay)
    ax[2].set_title("Overlay Visualization", fontweight='bold')
    ax[2].axis('off')
    
    plt.suptitle(
        f"Diagnosis: {CLASSES['binary'][pred_idx].upper()} ({binary_conf:.1f}%)\n"
        f"Subtype: {subtype_conf:.1f}% Confidence",
        fontsize=16,
        y=1.05
    )
    plt.tight_layout()
    return fig

def analyze_image(input_img):
    """Gradio interface handler"""
    if input_img is None:
        return (
            "<div style='color: red; text-align: center;'>"
            "⚠️ Please upload an image first</div>",
            None,
            "🟡 Status: Waiting for input"
        )
    
    try:
        # Convert to numpy if PIL Image
        if isinstance(input_img, Image.Image):
            input_img = np.array(input_img)
        
        # Run prediction
        results = predict(input_img)
        
        # Generate visualization
        fig = create_visualization(
            input_img,
            results['pred_idx'],
            results['binary_confidence'],
            results['subtype_confidence']
        )
        
        # Create HTML report
        diagnosis_color = "#4CAF50" if results['binary_label'] == "benign" else "#F44336"
        html_report = f"""
        <div style="border: 2px solid {diagnosis_color}; border-radius: 10px; padding: 20px;">
            <h2 style="color: {diagnosis_color}; text-align: center;">
                {'🔴 MALIGNANT' if results['binary_label'] == 'malignant' else '🟢 BENIGN'} DETECTION
            </h2>
            <p><strong>Confidence:</strong> {results['binary_confidence']:.1f}%</p>
            <p><strong>Most Likely Subtype:</strong> {results['subtype_name']} ({results['subtype_confidence']:.1f}%)</p>
            
            <h3>Subtype Probabilities:</h3>
            <ul>
                {"".join(f"<li>{name}: {prob:.1f}%</li>" for name, prob in results['subtype_probs'].items())}
            </ul>
        </div>
        """
        
        return html_report, fig, "🟢 Analysis Complete"
    
    except Exception as e:
        error_html = f"""
        <div style="color: white; background: #F44336; padding: 20px; border-radius: 10px;">
            <h2>❌ Analysis Failed</h2>
            <p><strong>Error:</strong> {str(e)}</p>
            <p>Please try another image or check console for details.</p>
        </div>
        """
        return error_html, None, "🔴 Error Occurred"

# Load models (must happen before Gradio interface)
if not load_models():
    raise RuntimeError("Failed to load required models")

# Gradio Interface
with gr.Blocks(title="Histopathology AI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🔬 Breast Cancer Histopathology Analysis
    *AI-powered detection of malignant/benign tumors with subtype classification*
    """)
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Upload Histopathology Slide", type="pil")
            analyze_btn = gr.Button("Analyze", variant="primary")
            clear_btn = gr.Button("Clear")
            
        with gr.Column():
            html_output = gr.HTML()
            plot_output = gr.Plot()
            status_output = gr.Markdown("🟡 Status: Ready")
    
    # Examples
    gr.Examples(
        examples=[
            os.path.join("examples", f) 
            for f in os.listdir("examples") 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ],
        inputs=image_input,
        label="Sample Images (Click to Load)"
    )
    
    # Event handlers
    analyze_btn.click(
        fn=analyze_image,
        inputs=image_input,
        outputs=[html_output, plot_output, status_output]
    )
    
    clear_btn.click(
        fn=lambda: [None, None, "🟡 Status: Ready"],
        outputs=[html_output, plot_output, status_output]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False  # Set to True for temporary public link
    )