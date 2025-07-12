import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import gradio as gr

# Load Models
model_paths = {
    'binary': 'BC/binary.h5',
    'benign': 'BC/benign93.h5',
    'malignant': 'BC/malignant.h5'
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

# Verify model files exist before loading
print("Checking model files...")
for name, path in model_paths.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    print(f"Found {name} model at: {path}")

# Load models
print("Loading models...")
try:
    models = {name: load_model(path) for name, path in model_paths.items()}

    # Grad-CAM setup
    binary_model = models['binary']
    last_conv = None
    for layer in reversed(binary_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer.name
            break

    grad_model = Model(inputs=binary_model.inputs,
                      outputs=[binary_model.get_layer(last_conv).output, binary_model.output])

    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    raise

def prep_img(img):
    """Prepare image for model input from uploaded file"""
    img = Image.fromarray(img)
    img = img.resize(img_size)
    arr = img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)

def predict(img):
    """Make predictions using the models"""
    try:
        arr = prep_img(img)
        binary_pred = models['binary'].predict(arr, verbose=0)
        pred_idx = np.argmax(binary_pred[0])
        binary_conf = 100 * np.max(binary_pred[0])
        label = classes['binary'][pred_idx]

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

        if subtype_conf < 60:
            subtype_name = f"Uncertain (most likely: {subtype_name})"

        return pred_idx, binary_conf, subtype_name, subtype_conf, subtype_probs

    except Exception as e:
        print(f"\nERROR during prediction: {str(e)}")
        raise

def make_heatmap(img, pred_idx):
    """Generate Grad-CAM heatmap"""
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

def create_visualization(img, pred_idx, binary_conf, subtype, subtype_conf):
    """Create visualization with detailed explanations"""
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    fig, axs = plt.subplots(2, 3, figsize=(18, 12), dpi=100)

    # Full image analysis
    resized = cv2.resize(img, target_size)
    heatmap = make_heatmap(img, pred_idx)
    overlay = cv2.addWeighted(resized, 0.6, heatmap, 0.4, 0)

    resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    axs[0,0].imshow(resized_rgb)
    axs[0,0].set_title("Original Tissue Sample", fontsize=12)
    axs[0,1].imshow(heatmap_rgb)
    axs[0,1].set_title("Model Attention Heatmap", fontsize=12)
    axs[0,2].imshow(overlay_rgb)
    axs[0,2].set_title("Attention Overlay", fontsize=12)

    diagnosis = classes['binary'][pred_idx].upper()
    color = 'green' if diagnosis == 'BENIGN' else 'red'
    plt.suptitle(f"Diagnosis: {diagnosis} ({binary_conf:.1f}% confidence)", fontsize=14, color=color)
    plt.tight_layout()
    return fig

def analyze_image(input_img):
    """Main analysis function"""
    try:
        img = input_img.astype('uint8')
        pred_idx, binary_conf, subtype, subtype_conf, subtype_probs = predict(img)
        fig = create_visualization(img, pred_idx, binary_conf, subtype, subtype_conf)

        diagnosis = classes['binary'][pred_idx].upper()
        diagnosis_color = "#4CAF50" if diagnosis == "BENIGN" else "#F44336"
        
        subtype_table = "<table style='width:100%;'><tr><th>Subtype</th><th>Confidence</th></tr>"
        for name, prob in subtype_probs.items():
            subtype_table += f"<tr><td>{name}</td><td>{prob:.1f}%</td></tr>"
        subtype_table += "</table>"

        diagnosis_html = f"""
        <div style="border: 2px solid {diagnosis_color}; border-radius: 8px; padding: 20px; margin: 20px 0;">
            <h2 style="color: {diagnosis_color};">Diagnosis: {diagnosis} ({binary_conf:.1f}% confidence)</h2>
            <h3>Most Likely: {subtype} ({subtype_conf:.1f}% confidence)</h3>
            {subtype_table}
        </div>
        """

        return diagnosis_html, fig

    except Exception as e:
        error_html = f"<div style='color: red;'>Error: {str(e)}</div>"
        return error_html, None

# Example images
example_images = [
    ("example/SOB_B_A-14-22549AB-40-001.png", "Adenosis (Benign)"),
    ("example/SOB_B_F-14-9133-100-001.png", "Fibroadenoma (Benign)"),
    ("example/SOB_M_DC-14-2523-40-010.png", "Ductal Carcinoma (Malignant)")
]

# Custom CSS
custom_css = """
:root {
    --primary-text: #2c3e50;
    --secondary-text: #7f8c8d;
    --diagnosis-text: #4a6fa5;
}
.example-btn {
    height: 100px;
    width: 100px;
    background-size: cover;
    background-position: center;
    border: 2px solid transparent;
    transition: all 0.3s ease;
    margin: 0 auto;
}
.example-btn:hover {
    transform: scale(1.05);
    border-color: var(--diagnosis-text);
}
.example-label {
    text-align: center;
    margin-top: 5px;
    color: var(--secondary-text);
}
"""

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("# Breast Cancer Histopathology Analysis")
    
    with gr.Row():
        with gr.Column(scale=1):
            # Upload section
            image_input = gr.Image(label="Upload Image", type="numpy")
            submit_btn = gr.Button("Analyze", variant="primary")
            
            # Example images
            with gr.Accordion("Example Images", open=True):
                gr.Markdown("Click an example to load it:")
                with gr.Row():
                    for img_path, label in example_images:
                        if os.path.exists(img_path):
                            with gr.Column():
                                example_btn = gr.Button(label="", elem_classes=["example-btn"])
                                gr.Markdown(f"<div class='example-label'>{label}</div>")
                                # Set background image
                                example_btn.click(
                                    None,
                                    None,
                                    None,
                                    _js=f"""
                                    () => {{
                                        document.querySelector('#{example_btn.elem_id}').style.backgroundImage = "url('{img_path}')";
                                        return [];
                                    }}
                                    """
                                )
                                # Load image on click
                                example_btn.click(
                                    lambda x: x,
                                    inputs=[gr.Image(value=img_path, visible=False)],
                                    outputs=[image_input]
                                )
            
        with gr.Column(scale=2):
            # Results
            diagnosis_output = gr.HTML()
            plot_output = gr.Plot()
    
    submit_btn.click(
        fn=analyze_image,
        inputs=[image_input],
        outputs=[diagnosis_output, plot_output]
    )

demo.launch()