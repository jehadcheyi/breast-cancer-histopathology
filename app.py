import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import img_to_array
import gradio as gr
from huggingface_hub import hf_hub_download

# Configure TensorFlow
tf.get_logger().setLevel('ERROR')

# Configuration
REPO_ID = "jehadcheyi/bc_models"
HF_TOKEN = os.getenv("jehad")
IMG_SIZE = (128, 128)
TARGET_SIZE = (500, 500)

# Classes
classes = {
    'binary': ['benign', 'malignant'],
    'benign': ['A', 'F', 'PT', 'TA'],
    'malignant': ['DC', 'LC', 'MC', 'PC'],
    'benign_names': {'A': 'Adenosis', 'F': 'Fibroadenoma', 'PT': 'Phyllodes Tumor', 'TA': 'Tubular Adenoma'},
    'malignant_names': {'DC': 'Ductal Carcinoma', 'LC': 'Lobular Carcinoma', 'MC': 'Mucinous Carcinoma', 'PC': 'Papillary Carcinoma'}
}

model_paths = {
    'binary': 'binary.h5',
    'benign': 'benign.h5', 
    'malignant': 'malignant.h5'
}

# Initialize models
models = {}
print("🔄 Loading AI Models...")

def load_models():
    """Load all models with proper error handling"""
    global models
    try:
        for name, filename in model_paths.items():
            print(f"📥 Loading {name} model...")
            local_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                token=HF_TOKEN,
                cache_dir="model_cache"
            )
            models[name] = load_model(local_path, compile=False)
            print(f"✅ {name} model loaded successfully")
        
        print("🚀 All models loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Model loading failed: {str(e)}")
        return False

# Load models
if not load_models():
    raise Exception("Failed to load models")

# Setup Grad-CAM
def setup_gradcam():
    """Setup Grad-CAM model"""
    try:
        binary_model = models['binary']
        last_conv = None
        
        # Find last convolutional layer
        for layer in reversed(binary_model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv = layer.name
                break
        
        if last_conv is None:
            raise ValueError("No convolutional layer found")
        
        grad_model = Model(
            inputs=binary_model.inputs,
            outputs=[binary_model.get_layer(last_conv).output, binary_model.output]
        )
        
        print("🎯 Grad-CAM setup complete")
        return grad_model
    except Exception as e:
        print(f"❌ Grad-CAM setup failed: {str(e)}")
        return None

grad_model = setup_gradcam()

def preprocess_image(img):
    """
    Properly preprocess image for model input
    """
    try:
        print(f"📸 Input image shape: {img.shape}, dtype: {img.dtype}")
        
        # Ensure image is in correct format
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
        
        # Convert to RGB if needed
        if len(img.shape) == 3 and img.shape[2] == 3:
            # Assume input is RGB
            img_rgb = img
        elif len(img.shape) == 3 and img.shape[2] == 4:
            # Convert RGBA to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        else:
            # Convert grayscale to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        # Resize to model input size
        img_resized = cv2.resize(img_rgb, IMG_SIZE)
        
        # Normalize to [0, 1]
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        print(f"✅ Preprocessed shape: {img_batch.shape}, range: [{img_batch.min():.3f}, {img_batch.max():.3f}]")
        return img_batch
        
    except Exception as e:
        print(f"❌ Preprocessing error: {str(e)}")
        raise

def predict_image(img):
    """
    Make predictions with detailed logging
    """
    try:
        print("🔍 Starting prediction...")
        
        # Preprocess image
        img_processed = preprocess_image(img)
        
        # Binary classification
        print("🎯 Running binary classification...")
        binary_pred = models['binary'].predict(img_processed, verbose=0)
        print(f"Binary prediction shape: {binary_pred.shape}")
        print(f"Binary prediction values: {binary_pred[0]}")
        
        pred_idx = int(np.argmax(binary_pred[0]))
        binary_conf = float(np.max(binary_pred[0]) * 100)
        label = classes['binary'][pred_idx]
        
        print(f"✅ Binary result: {label} ({binary_conf:.1f}%)")
        
        # Subtype classification
        print(f"🔬 Running {label} subtype classification...")
        subtype_pred = models[label].predict(img_processed, verbose=0)
        print(f"Subtype prediction shape: {subtype_pred.shape}")
        print(f"Subtype prediction values: {subtype_pred[0]}")
        
        subtype_idx = int(np.argmax(subtype_pred[0]))
        subtype_conf = float(np.max(subtype_pred[0]) * 100)
        subtype_code = classes[label][subtype_idx]
        subtype_name = classes[f'{label}_names'][subtype_code]
        
        print(f"✅ Subtype result: {subtype_name} ({subtype_conf:.1f}%)")
        
        # All subtype probabilities
        subtype_probs = {}
        for code, prob in zip(classes[label], subtype_pred[0]):
            subtype_probs[classes[f'{label}_names'][code]] = float(prob * 100)
        
        print("🎉 Prediction completed successfully!")
        return pred_idx, binary_conf, subtype_name, subtype_conf, subtype_probs
        
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def generate_gradcam(img, pred_idx):
    """
    Generate Grad-CAM heatmap with proper error handling
    """
    try:
        if grad_model is None:
            print("⚠️ Grad-CAM not available")
            return np.zeros((*TARGET_SIZE, 3), dtype=np.uint8)
        
        print("🌡️ Generating Grad-CAM...")
        
        # Preprocess image for Grad-CAM
        img_processed = preprocess_image(img)
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_processed)
            loss = predictions[:, pred_idx]
        
        # Calculate gradients
        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            print("⚠️ No gradients computed")
            return np.zeros((*TARGET_SIZE, 3), dtype=np.uint8)
        
        grads = grads[0]
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
        conv_outputs = conv_outputs[0]
        
        # Generate heatmap
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0)
        
        # Normalize
        max_val = tf.math.reduce_max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val
        
        # Resize and apply colormap
        heatmap_resized = cv2.resize(heatmap.numpy(), TARGET_SIZE)
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        
        print("✅ Grad-CAM generated successfully")
        return heatmap_colored
        
    except Exception as e:
        print(f"❌ Grad-CAM error: {str(e)}")
        return np.zeros((*TARGET_SIZE, 3), dtype=np.uint8)

def create_visualization(img, pred_idx, binary_conf, label, subtype_conf, zooms=[75, 50, 25]):
    """
    Create visualization with proper image handling
    """
    try:
        print("🎨 Creating visualization...")
        
        # Ensure image is in correct format
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
        
        # Convert to RGB if needed
        if len(img.shape) == 3 and img.shape[2] == 3:
            img_rgb = img
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        h, w = img_rgb.shape[:2]
        levels = 1 + len(zooms)
        
        fig, axs = plt.subplots(levels, 3, figsize=(18, 6 * levels), dpi=100)
        if levels == 1:
            axs = [axs]
        
        # Full image analysis
        img_resized = cv2.resize(img_rgb, TARGET_SIZE)
        heatmap = generate_gradcam(img, pred_idx)
        
        # Create overlay
        if heatmap.sum() > 0:
            overlay = cv2.addWeighted(img_resized, 0.6, heatmap, 0.4, 0)
        else:
            overlay = img_resized
        
        # Plot full image
        axs[0][0].imshow(img_resized)
        axs[0][0].set_title("🔬 Original Tissue Sample\n(Full View)", fontsize=14, pad=15, fontweight='bold')
        axs[0][0].axis('off')
        
        axs[0][1].imshow(heatmap)
        axs[0][1].set_title("🌡️ AI Attention Heatmap\n(Red = High Importance)", fontsize=14, pad=15, fontweight='bold')
        axs[0][1].axis('off')
        
        axs[0][2].imshow(overlay)
        axs[0][2].set_title("🎯 Combined Analysis\n(AI Focus Areas)", fontsize=14, pad=15, fontweight='bold')
        axs[0][2].axis('off')
        
        # Zoomed analysis
        for i, zoom in enumerate(zooms):
            try:
                # Calculate crop box
                crop_h = int(h * zoom / 100)
                crop_w = int(w * zoom / 100)
                left = (w - crop_w) // 2
                top = (h - crop_h) // 2
                right = left + crop_w
                bottom = top + crop_h
                
                # Ensure bounds are valid
                left = max(0, left)
                top = max(0, top)
                right = min(w, right)
                bottom = min(h, bottom)
                
                crop = img_rgb[top:bottom, left:right]
                
                if crop.size > 0:
                    crop_resized = cv2.resize(crop, TARGET_SIZE)
                    crop_heatmap = generate_gradcam(crop, pred_idx)
                    
                    if crop_heatmap.sum() > 0:
                        crop_overlay = cv2.addWeighted(crop_resized, 0.6, crop_heatmap, 0.4, 0)
                    else:
                        crop_overlay = crop_resized
                    
                    # Create highlight image
                    highlight = img_rgb.copy()
                    cv2.rectangle(highlight, (left, top), (right, bottom), (255, 0, 0), 3)
                    
                    axs[i+1][0].imshow(highlight)
                    axs[i+1][0].set_title(f"🔍 Analysis Region: {zoom}%\n(Red box shows zoom area)", fontsize=12, pad=10)
                    axs[i+1][0].axis('off')
                    
                    axs[i+1][1].imshow(crop_heatmap)
                    axs[i+1][1].set_title(f"🌡️ Detailed Heatmap\n({zoom}% scale analysis)", fontsize=12, pad=10)
                    axs[i+1][1].axis('off')
                    
                    axs[i+1][2].imshow(crop_overlay)
                    axs[i+1][2].set_title(f"🎯 Focused Analysis\n({zoom}% scale overlay)", fontsize=12, pad=10)
                    axs[i+1][2].axis('off')
                else:
                    # Handle empty crop
                    for j in range(3):
                        axs[i+1][j].text(0.5, 0.5, f"Invalid crop at {zoom}%", 
                                       ha='center', va='center', transform=axs[i+1][j].transAxes)
                        axs[i+1][j].axis('off')
                        
            except Exception as e:
                print(f"⚠️ Error creating zoom level {zoom}%: {str(e)}")
                for j in range(3):
                    axs[i+1][j].text(0.5, 0.5, f"Error at {zoom}%", 
                                   ha='center', va='center', transform=axs[i+1][j].transAxes)
                    axs[i+1][j].axis('off')
        
        # Main title
        diagnosis = classes['binary'][pred_idx].upper()
        color = '#4CAF50' if diagnosis == 'BENIGN' else '#F44336'
        
        plt.suptitle(
            f"🏥 ADVANCED HISTOPATHOLOGY AI ANALYSIS\n\n"
            f"🎯 Primary Diagnosis: {diagnosis} ({binary_conf:.1f}% confidence)\n"
            f"🔬 Multi-Scale Attention Visualization System",
            fontsize=18,
            y=0.98,
            color=color,
            fontweight='bold'
        )
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        
        print("✅ Visualization created successfully")
        return fig
        
    except Exception as e:
        print(f"❌ Visualization error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Return simple error figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.text(0.5, 0.5, f"Visualization Error\n{str(e)}", 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.axis('off')
        return fig

def analyze_image(input_img):
    """
    Main analysis function with comprehensive error handling
    """
    print("\n" + "="*50)
    print("🚀 STARTING IMAGE ANALYSIS")
    print("="*50)
    
    if input_img is None:
        print("❌ No image provided")
        return """
        <div style="background: linear-gradient(45deg, #FFA726, #FF7043); color: white; padding: 25px; border-radius: 15px; text-align: center; margin: 20px 0;">
            <h2 style="margin: 0; font-size: 1.5em;">⚠️ No Image Provided</h2>
            <p style="margin: 10px 0; font-size: 1.1em;">Please upload a histopathology image to begin AI analysis</p>
        </div>
        """, None, "🟡 **Status**: Waiting for Image Upload"
    
    try:
        print(f"📊 Analyzing image with shape: {input_img.shape}, dtype: {input_img.dtype}")
        
        # Run prediction
        pred_idx, binary_conf, subtype, subtype_conf, subtype_probs = predict_image(input_img)
        
        label = classes['binary'][pred_idx]
        diagnosis = label.upper()
        diagnosis_color = "#4CAF50" if diagnosis == "BENIGN" else "#F44336"
        diagnosis_emoji = "🟢" if diagnosis == "BENIGN" else "🔴"
        
        print(f"🎯 Final diagnosis: {diagnosis} ({binary_conf:.1f}%)")
        print(f"🔬 Subtype: {subtype} ({subtype_conf:.1f}%)")
        
        # Create detailed HTML report
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
            bar_width = int(prob)
            bar_color = "#4CAF50" if prob < 30 else "#FF9800" if prob < 70 else "#F44336"
            
            subtype_table += f"""
            <tr style='border-bottom: 1px solid #e0e0e0;'>
                <td style='padding: 12px; font-weight: bold;'>{name}</td>
                <td style='padding: 12px; text-align: center; font-weight: bold; color: {bar_color};'>{prob:.1f}%</td>
                <td style='padding: 12px; text-align: center;'>
                    <div style='background: #e0e0e0; border-radius: 10px; height: 20px; width: 100px; margin: 0 auto;'>
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
                <div style="background: {diagnosis_color}; color: white; padding: 8px 20px; border-radius: 25px; display: inline-block; margin: 15px 0; font-weight: bold;">
                    Primary Diagnosis: {diagnosis} ({binary_conf:.1f}% confidence)
                </div>
            </div>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 25px;">
                <div style="background: rgba(255,255,255,0.9); padding: 20px; border-radius: 15px; text-align: center;">
                    <h3 style="color: {diagnosis_color}; margin-top: 0;">🎯 Primary Classification</h3>
                    <div style="font-size: 2.5em; font-weight: bold; color: {diagnosis_color};">{binary_conf:.1f}%</div>
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
            
            <div style="background: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 12px; margin-top: 20px; text-align: center;">
                <strong>⚠️ Medical Disclaimer:</strong> This AI analysis is for research purposes only. Always consult healthcare professionals.
            </div>
        </div>
        """
        
        # Create visualization
        fig = create_visualization(input_img, pred_idx, binary_conf, label, subtype_conf)
        
        status_msg = f"{diagnosis_emoji} **Status**: Analysis Complete - {diagnosis} detected with {binary_conf:.1f}% confidence"
        
        print("🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*50)
        
        return diagnosis_html, fig, status_msg
        
    except Exception as e:
        print(f"❌ ANALYSIS FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        print("="*50)
        
        error_html = f"""
        <div style="background: linear-gradient(45deg, #e74c3c, #c0392b); color: white; padding: 25px; border-radius: 15px; margin: 20px 0;">
            <h2 style="margin: 0; text-align: center;">❌ ANALYSIS ERROR</h2>
            <div style="background: rgba(255,255,255,0.2); padding: 15px; border-radius: 10px; margin: 15px 0;">
                <strong>Error Details:</strong> {str(e)}
            </div>
            <p style="text-align: center;">Please try uploading a different image or contact support.</p>
        </div>
        """
        return error_html, None, "🔴 **Status**: Analysis Failed - System Error"

# Rest of your Gradio interface code remains the same...
# [Include all your existing Gradio interface code here]

# Create the interface (keeping your existing interface code)
with gr.Blocks(css="""
.gradio-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
}
""", theme=gr.themes.Soft(), title="Breast Cancer AI Analyzer") as demo:
    
    # Your existing interface code here...
    gr.HTML("""
    <div style="background: rgba(255, 255, 255, 0.95); padding: 30px; border-radius: 20px; text-align: center; margin-bottom: 20px;">
        <h1 style="color: #2c3e50; margin-bottom: 15px;">🏥 Advanced Breast Cancer AI Analyzer</h1>
        <p style="font-size: 1.2em; color: #34495e;">Professional-grade histopathology analysis powered by deep learning</p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                label="🔬 Upload Histopathology Image",
                type="numpy",
                height=400
            )
            
            with gr.Row():
                analyze_btn = gr.Button("🔍 Analyze with AI", variant="primary", size="lg")
                clear_btn = gr.Button("🧹 Clear", variant="secondary", size="lg")
            
            status_output = gr.Markdown("🟡 **Status**: Ready for analysis")
        
        with gr.Column(scale=2):
            results_output = gr.HTML("""
            <div style='background: #e3f2fd; padding: 30px; border-radius: 20px; text-align: center;'>
                <h2>🔬 AI Diagnostic System Ready</h2>
                <p>Upload a histopathology image to begin analysis</p>
            </div>
            """)
            
            plot_output = gr.Plot(label="Multi-Scale Visualization")
    
    # Event handlers
    def clear_interface():
        return None, """
        <div style='background: #e3f2fd; padding: 30px; border-radius: 20px; text-align: center;'>
            <h2>🔬 AI Diagnostic System Ready</h2>
            <p>Upload a histopathology image to begin analysis</p>
        </div>
        """, None, "🟡 **Status**: Interface cleared"
    
    analyze_btn.click(
        fn=analyze_image,
        inputs=image_input,
        outputs=[results_output, plot_output, status_output]
    )
    
    clear_btn.click(
        fn=clear_interface,
        outputs=[image_input, results_output, plot_output, status_output]
    )

# Launch
if __name__ == "__main__":
    print("\n🚀 Starting Breast Cancer AI Analyzer...")
    demo.launch(
        share=True,
        show_error=True,
        debug=True  # Enable debug mode to see more error details
    )
