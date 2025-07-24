import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model, Model
import gradio as gr
from huggingface_hub import hf_hub_download

# ========== MODEL LOADING ==========
model_paths = {
    'binary': 'binary.h5',
    'benign': 'benign.h5', 
    'malignant': 'malignant.h5'
}

models = {}

# Load models from Hugging Face Hub
for name, filename in model_paths.items():
    try:
        model_path = hf_hub_download(
            repo_id="jehadcheyi/bc_models",
            filename=filename,
            token=os.environ["jehas"]
        )
        models[name] = load_model(model_path)
        print(f"{name} model loaded ✅")
    except Exception as e:
        print(f"❌ Error loading {name}: {e}")
        models[name] = None  # Ensure the key exists even if model fails to load

# Set up Grad-CAM
if models.get('binary'):
    binary_model = models['binary']
    last_conv = None
    for layer in reversed(binary_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer.name
            break

    grad_model = Model(inputs=binary_model.inputs,
                     outputs=[binary_model.get_layer(last_conv).output, 
                             binary_model.output])
    print("Grad-CAM model ready!")
else:
    print("⚠️ Binary model not available for Grad-CAM")

# ========== CONFIGURATION ==========
img_size = (128, 128)
target_size = (500, 500)

classes = {
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

# ========== IMAGE PROCESSING ==========
def prep_img(img):
    img = Image.fromarray(img)
    img = img.resize(img_size)
    arr = img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)

def predict(img):
    try:
        arr = prep_img(img)
        
        # Binary Classification
        if not models.get('binary'):
            raise ValueError("Binary model not loaded")
        
        binary_pred = models['binary'].predict(arr, verbose=0)
        pred_idx = np.argmax(binary_pred[0])
        binary_conf = 100 * np.max(binary_pred[0])
        label = classes['binary'][pred_idx]

        # Subtype Classification
        subtype_probs = {}
        if label == 'benign':
            if not models.get('benign'):
                raise ValueError("Benign model not loaded")
                
            subtype_pred = models['benign'].predict(arr, verbose=0)
            subtype_idx = np.argmax(subtype_pred[0])
            subtype_conf = 100 * np.max(subtype_pred[0])
            subtype_code = classes['benign'][subtype_idx]
            subtype_name = classes['benign_names'][subtype_code]
            
            for code, prob in zip(classes['benign'], subtype_pred[0]):
                subtype_probs[classes['benign_names'][code]] = prob*100
        else:
            if not models.get('malignant'):
                raise ValueError("Malignant model not loaded")
                
            subtype_pred = models['malignant'].predict(arr, verbose=0)
            subtype_idx = np.argmax(subtype_pred[0])
            subtype_conf = 100 * np.max(subtype_pred[0])
            subtype_code = classes['malignant'][subtype_idx]
            subtype_name = classes['malignant_names'][subtype_code]
            
            for code, prob in zip(classes['malignant'], subtype_pred[0]):
                subtype_probs[classes['malignant_names'][code]] = prob*100

        return pred_idx, binary_conf, subtype_name, subtype_conf, subtype_probs
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        raise

# ========== VISUALIZATION ==========
def make_heatmap(img, pred_idx):
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
    return cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET

def create_visualization(img, pred_idx):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    resized = cv2.resize(img, target_size)
    heatmap = make_heatmap(img, pred_idx)
    overlay = cv2.addWeighted(resized, 0.6, heatmap, 0.4, 0)

    resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(resized_rgb)
    ax[0].set_title("Original Image")
    ax[0].axis('off')

    ax[1].imshow(heatmap_rgb)
    ax[1].set_title("Attention Heatmap")
    ax[1].axis('off')

    ax[2].imshow(overlay_rgb)
    ax[2].set_title("Attention Overlay")
    ax[2].axis('off')

    plt.tight_layout()
    return fig

# ========== SAMPLE IMAGES ==========
def get_sample_images():
    sample_images = []
    sample_folder = "examples"  # Create this folder with sample images
    if os.path.exists(sample_folder):
        for file in os.listdir(sample_folder):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                try:
                    img_path = os.path.join(sample_folder, file)
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        sample_images.append(img)
                except Exception as e:
                    print(f"Error loading sample image {file}: {e}")
    return sample_images

sample_images = get_sample_images()

# ========== GRADIO INTERFACE ==========
def analyze_image(input_img):
    try:
        if input_img is None:
            raise ValueError("No image provided")
            
        img = input_img.astype('uint8')
        pred_idx, binary_conf, subtype, subtype_conf, subtype_probs = predict(img)

        label = classes['binary'][pred_idx].upper()
        diagnosis_color = "#4CAF50" if label == "BENIGN" else "#F44336"

        # Create diagnosis HTML
        subtype_table = "<table style='width:100%; border-collapse: collapse; margin: 15px 0;'>"
        subtype_table += """
            <tr style='background-color: #f2f2f2;'>
                <th style='padding: 8px; text-align: left;'>Subtype</th>
                <th style='padding: 8px; text-align: left;'>Confidence</th>
            </tr>
        """
        
        for name, prob in subtype_probs.items():
            subtype_table += f"""
            <tr>
                <td style='padding: 8px;'>{name}</td>
                <td style='padding: 8px; font-weight: bold;'>{prob:.1f}%</td>
            </tr>
            """
        subtype_table += "</table>"

        diagnosis_html = f"""
        <div style="border: 2px solid {diagnosis_color}; border-radius: 8px; padding: 20px; margin: 20px 0;">
            <h2 style="color: {diagnosis_color}; margin: 0;">Diagnosis: {label}</h2>
            <div style="background-color: {diagnosis_color}; color: white; padding: 5px 15px; border-radius: 20px; display: inline-block; margin: 10px 0;">
                {binary_conf:.1f}% Confidence
            </div>
            <h3 style="margin-top: 15px;">Most Likely Subtype: {subtype}</h3>
            <div style="background-color: #e3f2fd; color: #1976d2; padding: 3px 10px; border-radius: 12px; display: inline-block;">
                {subtype_conf:.1f}% Confidence
            </div>
            <div style="margin-top: 15px;">
                <h4>All Subtype Probabilities:</h4>
                {subtype_table}
            </div>
        </div>
        """

        fig = create_visualization(img, pred_idx)
        return diagnosis_html, fig

    except Exception as e:
        error_html = f"""
        <div style="border: 2px solid #FF5722; border-radius: 8px; padding: 20px; margin: 20px 0; background-color: #fff3e0;">
            <h2 style="color: #FF5722; margin: 0;">Error</h2>
            <p>{str(e)}</p>
        </div>
        """
        return error_html, None

# ========== APP SETUP ==========
with gr.Blocks(
    title="BREAST AI - Advanced Cancer Detection",
    theme="soft",
    css="""
    .header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 40px;
        width: 90%;
        margin: 0 auto;
        padding: 20px;
        background: #f8f9fa;
        border-radius: 15px;
    }
    .diagnosis-box { 
        margin: 20px 0; 
        padding: 20px; 
        border-radius: 8px; 
    }
    .benign { 
        border: 2px solid #4CAF50; 
        background-color: #e8f5e9; 
    }
    .malignant { 
        border: 2px solid #F44336; 
        background-color: #ffebee; 
    }
    """
) as demo:
    
    # Header
    with gr.Row():
        gr.Markdown("""
        <div class="header-container">
            <div style="text-align: center; flex: 1;">
                <h1>BREAST AI PRO</h1>
                <h3>Advanced Breast Cancer Detection System</h3>
            </div>
        </div>
        """)

    # Main Content
    with gr.Row():
        # Input Column
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("## 🔬 Upload Image")
                image_input = gr.Image(type="numpy", label="Histopathology Image")
                
                with gr.Row():
                    submit_btn = gr.Button("Analyze", variant="primary")
                    clear_btn = gr.Button("Clear", variant="secondary")

            # Sample Gallery
            if sample_images:
                with gr.Group():
                    gr.Markdown("## 📚 Examples")
                    gallery = gr.Gallery(
                        value=[(img, f"Sample {i+1}") for i, img in enumerate(sample_images)],
                        columns=3,
                        height="auto"
                    )
                    gallery.select(
                        fn=lambda evt: sample_images[evt.index],
                        outputs=image_input
                    )

        # Output Column
        with gr.Column(scale=2):
            with gr.Tab("Diagnostic Report"):
                diagnosis_output = gr.HTML()
                gr.Markdown("## 🌡️ Visualization")
                plot_output = gr.Plot()

    # Footer
    gr.Markdown("""
    <div style='text-align: center; margin-top: 20px;'>
        <p>© 2025 BREAST AI | For research use only</p>
    </div>
    """)

    # Event Handling
    submit_btn.click(
        fn=analyze_image,
        inputs=image_input,
        outputs=[diagnosis_output, plot_output]
    )

    clear_btn.click(
        fn=lambda: [None, None],
        outputs=[image_input, diagnosis_output]
    )

# Launch
if __name__ == "__main__":
    demo.launch(share=True)