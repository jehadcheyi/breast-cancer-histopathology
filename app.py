import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
import gradio as gr

# Model paths - ensure these files are uploaded to your Hugging Face Space
model_paths = {
    'binary': 'BC/binary.h5',
    'benign': 'BC/benign93.h5',
    'malignant': 'BC/malignant.h5'
}

# Verify model files exist
print("Checking model files...")
for name, path in model_paths.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    print(f"Found {name} model at: {path}")

# Load models
print("Loading models...")
try:
    models = {name: load_model(path) for name, path in model_paths.items()}
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    raise

def predict(img):
    """Process image and make predictions"""
    img = Image.fromarray(img).resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Binary prediction
    binary_pred = models['binary'].predict(img_array, verbose=0)[0]
    diagnosis = 'benign' if np.argmax(binary_pred) == 0 else 'malignant'
    confidence = round(100 * np.max(binary_pred), 1)
    
    # Subtype prediction
    subtype_model = models['benign'] if diagnosis == 'benign' else models['malignant']
    subtype_pred = subtype_model.predict(img_array, verbose=0)[0]
    subtype_idx = np.argmax(subtype_pred)
    subtype_conf = round(100 * np.max(subtype_pred), 1)
    
    return diagnosis, confidence, subtype_idx, subtype_conf

def create_visualization(img, diagnosis, confidence):
    """Create heatmap visualization"""
    img_array = np.array(Image.fromarray(img).resize((128, 128))) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Create Grad-CAM heatmap
    model = models['binary']
    last_conv_layer = next(layer for layer in reversed(model.layers) 
                          if isinstance(layer, tf.keras.layers.Conv2D))
    grad_model = Model(inputs=model.inputs, 
                      outputs=[last_conv_layer.output, model.output])
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0 if diagnosis == 'benign' else 1]
    
    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs[0]), axis=-1)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    heatmap = cv2.resize(heatmap.numpy(), (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.imshow(heatmap, alpha=0.5)
    ax.set_title(f"Diagnosis: {diagnosis.capitalize()} ({confidence}% confidence)")
    ax.axis('off')
    return fig

def analyze_image(img):
    """Main analysis function"""
    try:
        diagnosis, confidence, subtype_idx, subtype_conf = predict(img)
        fig = create_visualization(img, diagnosis, confidence)
        
        # Create results display
        result_html = f"""
        <div style="font-family: Arial; max-width: 600px; margin: auto;">
            <div style="background: {'#4CAF50' if diagnosis == 'benign' else '#F44336'}; 
                 color: white; padding: 20px; border-radius: 10px; text-align: center;">
                <h2>Diagnosis: {diagnosis.capitalize()}</h2>
                <h3>Confidence: {confidence}%</h3>
            </div>
            <div style="background: #f5f5f5; padding: 15px; border-radius: 10px; margin-top: 20px;">
                <h3>Subtype Information</h3>
                <p>Most likely subtype with {subtype_conf}% confidence</p>
            </div>
        </div>
        """
        return result_html, fig
    except Exception as e:
        return f"Error processing image: {str(e)}", None

# Gradio Interface
with gr.Blocks(title="Breast Cancer Analysis") as demo:
    gr.Markdown("# Breast Cancer Histopathology Analysis")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Upload Tissue Image", type="numpy")
            analyze_btn = gr.Button("Analyze", variant="primary")
        
        with gr.Column():
            results_html = gr.HTML()
            plot_output = gr.Plot()
    
    analyze_btn.click(
        fn=analyze_image,
        inputs=image_input,
        outputs=[results_html, plot_output]
    )

demo.launch()