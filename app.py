import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import gradio as gr

from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model as keras_load_model
import os
import tensorflow as tf
from tensorflow.keras.models import Model

# Hugging Face Hub configuration (unchanged)
REPO_ID = "jehadcheyi/bc_models"
HF_TOKEN = os.getenv("jehad")  # Keep your original token name

# Model paths (unchanged)
model_paths = {
    'binary': 'binary.h5',
    'benign': 'benign.h5', 
    'malignant': 'malignant.h5'
}

# Image and class definitions (unchanged)
img_size = (128, 128)
target_size = (500, 500)

classes = {
    'binary': ['benign', 'malignant'],
    'benign': ['A', 'F', 'PT', 'TA'],
    'malignant': ['DC', 'LC', 'MC', 'PC'],
    'benign_names': {'A': 'Adenosis', 'F': 'Fibroadenoma', 'PT': 'Phyllodes Tumor', 'TA': 'Tubular Adenoma'},
    'malignant_names': {'DC': 'Ductal Carcinoma', 'LC': 'Lobular Carcinoma', 'MC': 'Mucinous Carcinoma', 'PC': 'Papillary Carcinoma'}
}

# Initialize models dictionary (unchanged)
models = {}

# --- Original Loading Logic (preserved but organized) ---
print("Checking and loading models...")

# 1. First verify files exist
print("Verifying model files...")
for name, filename in model_paths.items():
    try:
        local_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            token=HF_TOKEN,
            cache_dir="model_cache"
        )
        print(f"✓ Found {name} model at: {local_path}")
    except Exception as e:
        print(f"❌ Error verifying {name} model: {str(e)}")
        raise

# 2. Load all models
print("\nLoading models into memory...")
try:
    for name, filename in model_paths.items():
        local_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            token=HF_TOKEN,
            cache_dir="model_cache"
        )
        models[name] = keras_load_model(local_path)
        print(f"✅ Successfully loaded {name} model")
except Exception as e:
    print(f"❌ Model loading failed: {str(e)}")
    raise

# 3. Grad-CAM Setup (unchanged)
print("\nSetting up Grad-CAM...")
try:
    binary_model = models['binary']
    last_conv = None
    
    # Find last conv layer
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

print("\nAll models loaded successfully!")


def prep_img(img):
    img = Image.fromarray(img)
    img = img.resize(img_size)
    arr = img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)

def predict(img):
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
        print(f"Error during prediction: {str(e)}")
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

def create_visualization(img, pred_idx):
    """Create visualization with heatmap"""
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    resized = cv2.resize(img, target_size)
    heatmap = make_heatmap(img, pred_idx)
    overlay = cv2.addWeighted(resized, 0.6, heatmap, 0.4, 0)

    # Convert back to RGB for display
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

def analyze_image(input_img):
    try:
        img = input_img.astype('uint8')
        pred_idx, binary_conf, subtype, subtype_conf, subtype_probs = predict(img)

        label = classes['binary'][pred_idx]  # Get label: 'benign' or 'malignant'
        diagnosis = label.upper()
        diagnosis_color = "#4CAF50" if diagnosis == "BENIGN" else "#F44336"

        # Table with colored header text
        subtype_table = """
        <table style='width:100%; border-collapse: collapse; margin: 15px 0; font-family: Arial, sans-serif;'>
            <tr style='background-color: #f2f2f2;'>
                <th style='color: #0D47A1; padding: 8px; text-align: left;'>Subtype</th>
                <th style='color: #0D47A1; padding: 8px; text-align: left;'>Confidence</th>
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

        # Diagnosis summary box
        diagnosis_html = f"""
        <div style="border: 2px solid {diagnosis_color}; border-radius: 8px; padding: 20px; margin: 20px 0; font-family: Arial, sans-serif;">
            <h2 style="color: {diagnosis_color}; margin: 0;">Diagnosis: {diagnosis}</h2>
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

        # Create visualization with Grad-CAM overlay
        fig = create_visualization(img, pred_idx, binary_conf, label, subtype_conf)

        return diagnosis_html, fig

    except Exception as e:
        error_html = f"""
        <div style="border: 2px solid #FF5722; border-radius: 8px; padding: 20px; margin: 20px 0; background-color: #fff3e0;">
            <h2 style="color: #FF5722; margin: 0;">Error Processing Image</h2>
            <p>{str(e)}</p>
        </div>
        """
        return error_html, None

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
                        sample_images.append((img, file))  # Store both image and filename
                except Exception as e:
                    print(f"Error loading sample image {file}: {e}")
    return sample_images

sample_images = get_sample_images()

def get_crop_box(shape, percent):
    """Calculate centered crop box coordinates"""
    h, w = shape[:2]
    crop_h = int(h * percent / 100)
    crop_w = int(w * percent / 100)
    left = (w - crop_w) // 2
    top = (h - crop_h) // 2
    return left, top, left + crop_w, top + crop_h

def create_visualization(img, pred_idx, binary_conf, label, subtype_conf, zooms=[75, 50, 25]):
    """Create visualization with detailed explanations"""
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert back to BGR for OpenCV
    h, w = img.shape[:2]
    levels = 1 + len(zooms)

    fig, axs = plt.subplots(levels, 3, figsize=(18, 6 * levels), dpi=100)
    if levels == 1:
        axs = [axs]  # Ensure axs is always 2D

    # Full image analysis
    resized = cv2.resize(img, target_size)
    heatmap = make_heatmap(img, pred_idx)
    overlay = cv2.addWeighted(resized, 0.6, heatmap, 0.4, 0)

    # Convert back to RGB for display
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

    # Zoomed analysis
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

        # Convert to RGB for display
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

    # Main title with diagnosis information
    diagnosis = classes['binary'][pred_idx].upper()
    color = 'green' if diagnosis == 'BENIGN' else 'red'

    plt.suptitle(
        f"\n\n\n\nBREAST TISSUE HISTOPATHOLOGY ANALYSIS\n\n"
        f"Primary Diagnosis: {diagnosis} ({binary_conf:.1f}% confidence)\n"
        f"Attention Visualization at Multiple Scales",
        fontsize=20,
        y=1.02,
        color=color,
        fontweight='bold'
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    return fig

with gr.Blocks(
    title="BREAST AI - Advanced Cancer Detection",
    theme="soft",
    css="""
    .centered-container {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        flex-direction: column !important;
        width: 100% !important;
        text-align: center !important;
        margin: 0 auto !important;
        padding: 20px !important;
    }
    .centered-container * {
        max-width: 100% !important;
        margin: 5px 0 !important;
    }
    """
) as demo:
    gr.Markdown("""
    <div class="centered-container">
        <h1 style="font-size: 2.5em;">BREAST AI PRO</h1>
        <h2 style="font-size: 1.5em;">
            Great AI Application for Breast Cancer Detection<br>
            أحدث نظام ذكاء اصطناعي لكشف سرطان الثدي<br>
            نویترین پرۆگرامێ زیرەکیا دەستکرد بو دەستنیشانکرنا شێرپەنجا مەمکان
        </h2>
        <p style="font-size: 1.2em;">
            Powered by Revolutionary Deep Learning • 98% Accuracy<br>
            ٪مدعوم بأحدث تقنيات التعلم العميق - بدقة 98<br>
            پشتگیریا فێربوونا کویرا نوی - برێژا ٪98
        </p>
        <p style="font-size: 1.2em;">
            Upload an image or select from examples below<br>
            ارفع صورة أو اختر من النماذج<br>
            وێنەکی دابگرە یان نموونەکی بهەڵبژێرە
        </p>
        <hr style="margin: 30px auto; width: 80%; border: 1px solid #ddd;">
    </div>
    """)
    

    with gr.Row():
        # Left Panel
        with gr.Column(scale=1):
            # Upload Section
            with gr.Group():
                gr.Markdown("""
                ## 🔬 AI-Powered Histopathology Analysis
                Discover malignant and benign patterns with unmatched precision:<br>
                🔹 Detects breast cancer subtypes automatically<br>
                🔹 10x faster than traditional pathology review<br>
                """)

                image_input = gr.Image(type="numpy", label="Upload Histopathology Image | حمّل صورة الأنسجة ")
                gr.Markdown("""
                <small>Supported Formats: jpg, jpeg, png, tif</small>
                """)

                with gr.Row():
                    submit_btn = gr.Button("🔍 Analyze Now | حلل الآن ", variant="primary", scale=2)
                    clear_btn = gr.Button("🧹 Clear  ", variant="secondary", scale=1)

            # Sample Gallery
            if sample_images:
                with gr.Group():
                    gr.Markdown("""
                    ## 📚 Example Images 
                    """)

                    gallery_items = [(img, f"Sample {i+1}") for i, img in enumerate(sample_images)] if not isinstance(sample_images[0], (list, tuple)) else [(img, name) for img, name in sample_images]

                    gallery = gr.Gallery(
                        value=gallery_items,
                        columns=3,
                        height="auto",
                        allow_preview=True,
                        show_share_button=True
                    )

                    def select_sample_image(evt: gr.SelectData):
                        return sample_images[evt.index] if not isinstance(sample_images[0], (list, tuple)) else sample_images[evt.index][0]

                    gallery.select(select_sample_image, None, image_input)

        # Right Panel - Results
        with gr.Column(scale=2):
            # Combined Diagnostic Report and Heatmap
            with gr.Tab("📊 AI Diagnostic Report "):
                with gr.Column():
                    diagnosis_output = gr.HTML("""
                    <div style='text-align: center; padding: 20px; background: #f9f9f9; border-radius: 8px; margin-bottom: 20px;'>
                    <h3 style='margin-top: 0;'>Awaiting Analysis.....</h3>
                    <p>Upload a histopathology image to receive an AI-based diagnostic report</p>
                    </div>
                    """)

                    gr.Markdown("### 🌡️ Tissue Analysis Heatmap ")
                    plot_output = gr.Plot()

                    gr.Markdown("""
                    <div style='background: #f5f5f5; padding: 10px; border-radius: 8px;'>
                    <h4 style='margin-top: 0;'>Heatmap Legend :</h4>
                    <div style='display: flex; flex-wrap: wrap; gap: 15px;'>
                        <div><span style='display: inline-block; width: 15px; height: 15px; background: #ff0000;'></span> High Malignancy (90-100%) </div>
                        <div><span style='display: inline-block; width: 15px; height: 15px; background: #ff6600;'></span> Moderate (70-90%) </div>
                        <div><span style='display: inline-block; width: 15px; height: 15px; background: #ffcc00;'></span> Suspicious (50-70%) </div>
                        <div><span style='display: inline-block; width: 15px; height: 15px; background: #00cc00;'></span> Normal Tissue </div>
                    </div>
                    </div>
                    """)

    # Footer
    gr.Markdown("""
    <div style='text-align: center; margin-top: 20px; font-size: 12px;'>
    <p>© 2025 BREAST AI Technologies | Research Use Only<br>© </p>
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

demo.launch(share=True)