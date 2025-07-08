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
    """
    Make predictions using the models with robust subtype classification
    Returns: pred_idx, binary_conf, subtype_name, subtype_conf, subtype_probs
    """
    try:
        # Prepare image
        arr = prep_img(img)

        # --- Binary Classification ---
        binary_pred = models['binary'].predict(arr, verbose=0)
        pred_idx = np.argmax(binary_pred[0])
        binary_conf = 100 * np.max(binary_pred[0])
        label = classes['binary'][pred_idx]

        print("\n" + "="*50)
        print(f"BINARY CLASSIFICATION RESULT: {label.upper()} ({binary_conf:.1f}% confidence)")
        print(f"Raw output: {binary_pred[0]}")
        print("="*50)

        # --- Subtype Classification ---
        subtype_probs = {}
        if label == 'benign':
            subtype_pred = models['benign'].predict(arr, verbose=0)
            subtype_idx = np.argmax(subtype_pred[0])
            subtype_conf = 100 * np.max(subtype_pred[0])
            subtype_code = classes['benign'][subtype_idx]
            subtype_name = classes['benign_names'][subtype_code]

            # Store all subtype probabilities
            for code, prob in zip(classes['benign'], subtype_pred[0]):
                subtype_probs[classes['benign_names'][code]] = prob*100

            print("\nBENIGN SUBTYPE ANALYSIS:")
            print(f"Predicted: {subtype_name} ({subtype_conf:.1f}% confidence)")
            print("All subtype probabilities:")
            for name, prob in subtype_probs.items():
                print(f"  {name}: {prob:.1f}%")

        else:  # malignant
            subtype_pred = models['malignant'].predict(arr, verbose=0)
            subtype_idx = np.argmax(subtype_pred[0])
            subtype_conf = 100 * np.max(subtype_pred[0])
            subtype_code = classes['malignant'][subtype_idx]
            subtype_name = classes['malignant_names'][subtype_code]

            # Store all subtype probabilities
            for code, prob in zip(classes['malignant'], subtype_pred[0]):
                subtype_probs[classes['malignant_names'][code]] = prob*100

            print("\nMALIGNANT SUBTYPE ANALYSIS:")
            print(f"Predicted: {subtype_name} ({subtype_conf:.1f}% confidence)")
            print("All subtype probabilities:")
            for name, prob in subtype_probs.items():
                print(f"  {name}: {prob:.1f}%")

        # Confidence threshold check
        MIN_CONFIDENCE = 60
        if subtype_conf < MIN_CONFIDENCE:
            subtype_name = f"Uncertain (most likely: {subtype_name})"
            print(f"\nWARNING: Low confidence in subtype classification ({subtype_conf:.1f}%)")

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
        f"\n\n"
        f"\n\n"
        f"\n\n"
        f"BREAST TISSUE HISTOPATHOLOGY ANALYSIS\n\n"
        f"Primary Diagnosis: {diagnosis} ({binary_conf:.1f}% confidence)\n"
        f"Attention Visualization at Multiple Scales",
        fontsize=14,
        y=1.02,
        color=color,
        fontweight='bold'
    )

    # Add footer with interpretation guide
    plt.figtext(
        0.5, 0.01,
        "INTERPRETATION GUIDE:\n"
        "• Red areas in heatmaps show where the model focused most for its decision\n"
        "• Warmer colors indicate higher importance regions\n"
        "• Multiple zoom levels help verify consistent attention patterns",
        ha="center",
        fontsize=11,
        bbox={"facecolor":"lightgray", "alpha":0.3, "pad":5}
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.12)
    return fig

def get_subtype_description(subtype):
    """Return detailed description for each subtype"""
    descriptions = {
        "Adenosis": "Adenosis is a benign condition involving enlargement of breast lobules. It's common and usually doesn't increase cancer risk.",
        "Fibroadenoma": "Fibroadenomas are common benign solid breast tumors made of glandular and stromal tissue. They're most common in young women.",
        "Phyllodes Tumor": "Phyllodes tumors are rare breast tumors that grow in the stroma. Most are benign but some can be borderline or malignant.",
        "Tubular Adenoma": "Tubular adenomas are benign breast tumors composed of closely packed tubules. They resemble fibroadenomas microscopically.",
        "Ductal Carcinoma": "Invasive ductal carcinoma is the most common type of breast cancer, starting in the milk ducts and invading nearby tissue.",
        "Lobular Carcinoma": "Invasive lobular carcinoma begins in the milk-producing lobules and tends to spread more diffusely than ductal carcinoma.",
        "Mucinous Carcinoma": "Mucinous (or colloid) carcinoma is a rare type where cancer cells produce mucus. It tends to have a better prognosis.",
        "Papillary Carcinoma": "Papillary carcinoma is a rare breast cancer with finger-like projections. It's often treatable when caught early."
    }

    # Extract the base subtype name if it's in "Uncertain (most likely: X)" format
    base_subtype = subtype.split(": ")[-1].replace(")", "") if "Uncertain" in subtype else subtype
    return descriptions.get(base_subtype, "No additional information available for this subtype.")

def analyze_image(input_img):
    """Main analysis function for Gradio interface"""
    try:
        # Convert Gradio numpy array to proper format
        img = input_img.astype('uint8')

        # Make predictions
        pred_idx, binary_conf, subtype, subtype_conf, subtype_probs = predict(img)

        # Create visualization
        fig = create_visualization(img, pred_idx, binary_conf, subtype, subtype_conf)

        # Prepare diagnosis text
        diagnosis = classes['binary'][pred_idx].upper()
        diagnosis_color = "#4CAF50" if diagnosis == "BENIGN" else "#F44336"

        # Create subtype probabilities table
        subtype_table = "<table style='width:100%; border-collapse: collapse; margin: 15px 0;'>"
        subtype_table += "<tr style='background-color: #f2f2f2;'><th style='padding: 10px; text-align: left; border: 1px solid #ddd; color: black;'>Subtype</th><th style='padding: 10px; text-align: left; border: 1px solid #ddd; color: black;'>Confidence</th></tr>"
        for name, prob in subtype_probs.items():
            subtype_table += f"<tr><td style='padding: 10px; border: 1px solid #ddd; color: black;'>{name}</td><td style='padding: 10px; border: 1px solid #ddd; color: black; font-weight: bold;'>{prob:.1f}%</td></tr>"
        subtype_table += "</table>"

        diagnosis_html = f"""
        <div style="border: 2px solid {diagnosis_color}; border-radius: 8px; padding: 20px; margin: 20px 0; background-color: #f8f9fa; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <h2 style="color: {diagnosis_color}; margin: 0; font-size: 24px;">Diagnosis: {diagnosis}</h2>
                <div style="background-color: {diagnosis_color}; color: white; padding: 5px 15px; border-radius: 20px; font-weight: bold;">
                    {binary_conf:.1f}% Confidence
                </div>
            </div>

            <div style="background-color: white; border-radius: 6px; padding: 15px; margin: 15px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                <h3 style="margin-top: 0; color: #333; border-bottom: 1px solid #eee; padding-bottom: 10px;">Subtype Analysis</h3>
                <div style="display: flex; align-items: center; margin-bottom: 10px;">
                    <div style="font-weight: bold; margin-right: 10px; color: var(--primary-text);">Most Likely:</div>
                    <div style="font-size: 18px; font-weight: bold; color: {diagnosis_color};">{subtype}</div>
                    <div style="margin-left: auto; background-color: #e3f2fd; color: #1976d2; padding: 3px 10px; border-radius: 12px; font-weight: bold;">
                        {subtype_conf:.1f}% Confidence
                    </div>
                </div>

                <div style="margin-top: 15px;">
                    <h4 style="margin-bottom: 10px; color: #555;">All Subtype Probabilities:</h4>
                    {subtype_table}
                </div>
            </div>

            <div style="font-size: 14px; color: #666; line-height: 1.6;">
                <p><strong>Note:</strong> This analysis should be reviewed by a qualified pathologist. The confidence levels indicate the model's certainty in its predictions, with values above 70% considered high confidence.</p>
            </div>
        </div>
        """

        # Prepare detailed explanation based on diagnosis
        if diagnosis == "BENIGN":
            explanation = f"""
            <div style="background-color: #e8f5e9; padding: 20px; border-radius: 8px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                <h3 style="color: #2e7d32; margin-top: 0; border-bottom: 1px solid #c8e6c9; padding-bottom: 10px;">About Benign Breast Tissue</h3>

                <div style="display: flex; margin: 15px 0;">
                    <div style="flex: 1; padding-right: 15px;">
                        <h4 style="color: #388e3c; margin-bottom: 8px;">What This Means</h4>
                        <p style="margin-top: 0; color: var(--primary-text);">The analysis suggests the tissue shows characteristics of <strong>non-cancerous (benign)</strong> breast tissue. The most likely subtype is <strong>{subtype}</strong> with {subtype_conf:.1f}% confidence.</p>
                        <p style="color: var(--primary-text);">Benign conditions are common and often don't require treatment, but some may need monitoring or intervention.</p>
                    </div>
                    <div style="flex: 1; padding-left: 15px; border-left: 1px solid #c8e6c9;">
                        <h4 style="color: #388e3c; margin-bottom: 8px;">Recommended Next Steps</h4>
                        <ul style="margin-top: 0; padding-left: 20px; color: var(--primary-text);">
                            <li>Consult with a pathologist to confirm these findings</li>
                            <li>Discuss any necessary follow-up with your healthcare provider</li>
                            <li>Regular monitoring as recommended</li>
                            <li>Maintain routine breast health screenings</li>
                        </ul>
                    </div>
                </div>

                <div style="background-color: #c8e6c9; padding: 12px; border-radius: 6px; margin-top: 15px;">
                    <h4 style="margin: 0 0 8px 0; color: #1b5e20;">About {subtype}</h4>
                    <p style="margin: 0; color: var(--primary-text);">{get_subtype_description(subtype)}</p>
                </div>
            </div>
            """
        else:
            explanation = f"""
            <div style="background-color: #ffebee; padding: 20px; border-radius: 8px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                <h3 style="color: #c62828; margin-top: 0; border-bottom: 1px solid #ffcdd2; padding-bottom: 10px;">About Malignant Breast Tissue</h3>

                <div style="display: flex; margin: 15px 0;">
                    <div style="flex: 1; padding-right: 15px;">
                        <h4 style="color: #d32f2f; margin-bottom: 8px;">What This Means</h4>
                        <p style="margin-top: 0; color: var(--primary-text);">The analysis suggests the tissue shows characteristics of <strong>cancerous (malignant)</strong> breast tissue. The most likely subtype is <strong>{subtype}</strong> with {subtype_conf:.1f}% confidence.</p>
                        <p style="color: var(--primary-text);">Malignant findings require prompt medical attention but many treatment options are available.</p>
                    </div>
                    <div style="flex: 1; padding-left: 15px; border-left: 1px solid #ffcdd2;">
                        <h4 style="color: #d32f2f; margin-bottom: 8px;">Immediate Actions</h4>
                        <ul style="margin-top: 0; padding-left: 20px; color: var(--primary-text);">
                            <li>Schedule an appointment with an oncologist immediately</li>
                            <li>Consult with a pathologist to confirm these findings</li>
                            <li>Additional diagnostic tests may be required</li>
                            <li>Discuss treatment options with your healthcare team</li>
                        </ul>
                    </div>
                </div>

                <div style="background-color: #ffcdd2; padding: 12px; border-radius: 6px; margin-top: 15px;">
                    <h4 style="margin: 0 0 8px 0; color: #b71c1c;">About {subtype}</h4>
                    <p style="margin: 0; color: var(--primary-text);">{get_subtype_description(subtype)}</p>
                </div>

                <div style="margin-top: 15px; padding: 15px; background-color: #f5f5f5; border-radius: 6px;">
                    <h4 style="margin: 0 0 10px 0; color: #333;">Support Resources</h4>
                    <p style="margin: 0 0 10px 0; color: var(--primary-text);">A cancer diagnosis can be overwhelming. Consider these resources:</p>
                    <ul style="margin: 0; padding-left: 20px; color: var(--primary-text);">
                        <li>American Cancer Society: <a href="https://www.cancer.org" target="_blank" style="color: #1976d2;">www.cancer.org</a></li>
                        <li>National Breast Cancer Foundation: <a href="https://www.nationalbreastcancer.org" target="_blank" style="color: #1976d2;">www.nationalbreastcancer.org</a></li>
                        <li>BreastCancer.org: <a href="https://www.breastcancer.org" target="_blank" style="color: #1976d2;">www.breastcancer.org</a></li>
                    </ul>
                </div>
            </div>
            """

        return diagnosis_html + explanation, fig

    except Exception as e:
        error_html = f"""
        <div style="border: 2px solid #FF5722; border-radius: 8px; padding: 20px; margin: 20px 0; background-color: #fff3e0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <div style="display: flex; align-items: center; margin-bottom: 15px;">
                <svg style="width: 24px; height: 24px; margin-right: 10px;" fill="#FF5722" viewBox="0 0 24 24">
                    <path d="M12,2L1,21H23M12,6L19.53,19H4.47M11,10V14H13V10M11,16V18H13V16"/>
                </svg>
                <h2 style="color: #FF5722; margin: 0;">Error Processing Image</h2>
            </div>
            <p style="color: #6d4c41; margin-bottom: 15px;">{str(e)}</p>
            <div style="background-color: #ffecb3; padding: 10px; border-radius: 6px;">
                <p style="margin: 0; font-size: 14px; color: var(--primary-text);">
                    <strong>Troubleshooting:</strong> Ensure you've uploaded a valid histopathology image in JPG, PNG, or TIFF format.
                    The image should be clear and show breast tissue at sufficient magnification.
                </p>
            </div>
        </div>
        """
        return error_html, None
        # Prepare metrics content
        metrics_content = f"""
        <div class="card">
            <h3 style="color: var(--primary-text);">Subtype Confidence Metrics</h3>
            {subtype_table}
            <div style="margin-top: 20px;">
                <h4 style="color: var(--primary-text);">Confidence Interpretation</h4>
                <ul style="color: var(--primary-text);">
                    <li><span style="color: #4CAF50; font-weight: bold;">High Confidence</span>: >70%</li>
                    <li><span style="color: #FFC107; font-weight: bold;">Medium Confidence</span>: 50-70%</li>
                    <li><span style="color: #F44336; font-weight: bold;">Low Confidence</span>: <50%</li>
                </ul>
            </div>
        </div>
        """

        return diagnosis_html + explanation, fig, metrics_content

    except Exception as e:
        error_html = f"""..."""  # Keep your existing error HTML
        return error_html, None, "<div style='color: var(--danger-text);'>Error generating metrics</div>"

# Custom CSS with enhanced color scheme
custom_css = """
:root {
    --primary-text: #2c3e50;        /* Dark blue for main text */
    --secondary-text: #7f8c8d;      /* Gray for secondary text */
    --light-text: #f8f9fa;          /* Light color for dark backgrounds */
    --diagnosis-text: #4a6fa5;      /* Blue for diagnosis text */
    --warning-text: #d35400;        /* Orange for warnings */
    --success-text: #27ae60;        /* Green for success messages */
    --danger-text: #e74c3c;         /* Red for danger/error messages */
    --link-color: #1976d2;          /* Blue for links */
    --border-color: #e0e0e0;        /* Light gray for borders */
    --card-bg: #ffffff;             /* White for cards */
    --shadow-color: rgba(0,0,0,0.1); /* Shadow color */
}

.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
    line-height: 1.6;
    color: var(--primary-text);
}

.dark .gradio-container {
    color: var(--light-text);
}

/* Headings */
h1, h2, h3, h4 {
    color: var(--primary-text);
    font-weight: 600;
}

.dark h1, .dark h2, .dark h3, .dark h4 {
    color: var(--light-text);
}

/* Text elements */
p, li {
    color: var(--primary-text);
}

.dark p, .dark li {
    color: var(--light-text);
}

/* Secondary text */
.secondary-text {
    color: var(--secondary-text);
}

/* Diagnosis specific colors */
.diagnosis-text {
    color: var(--diagnosis-text);
}

.warning-text {
    color: var(--warning-text);
}

.success-text {
    color: var(--success-text);
}

.danger-text {
    color: var(--danger-text);
}

.link-text {
    color: var(--link-color);
}

/* Cards and containers */
.card {
    background-color: var(--card-bg);
    border-radius: 8px;
    padding: 20px;
    margin: 15px 0;
    box-shadow: 0 2px 4px var(--shadow-color);
    border: 1px solid var(--border-color);
}

.dark .card {
    background-color: #2d3748;
    border-color: #4a5568;
}

/* Upload box */
.upload-box {
    border: 2px dashed var(--diagnosis-text) !important;
    border-radius: 8px !important;
    padding: 2rem !important;
    background-color: rgba(74, 111, 165, 0.05) !important;
    transition: all 0.3s ease !important;
}

.dark .upload-box {
    background-color: rgba(160, 196, 255, 0.05) !important;
    border-color: #a0c4ff !important;
}

.upload-box:hover {
    background-color: rgba(74, 111, 165, 0.1) !important;
}

.dark .upload-box:hover {
    background-color: rgba(160, 196, 255, 0.1) !important;
}

/* Buttons */
.primary-button {
    background-color: var(--diagnosis-text) !important;
    border: none !important;
    color: white !important;
    padding: 0.75rem 1.5rem !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 6px var(--shadow-color) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}

.primary-button:hover {
    background-color: #3a5a80 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15) !important;
}

.dark .primary-button {
    background-color: #5a86c2 !important;
}

.dark .primary-button:hover {
    background-color: #4a76b2 !important;
}

/* Tabs */
.tab-button {
    border-radius: 8px !important;
    padding: 0.5rem 1rem !important;
    transition: all 0.3s ease !important;
    color: var(--primary-text) !important;
}

.tab-button.selected {
    background-color: var(--diagnosis-text) !important;
    color: white !important;
}

.dark .tab-button.selected {
    background-color: #5a86c2 !important;
}

/* Footer */
.footer {
    font-size: 0.85rem;
    color: var(--secondary-text);
    text-align: center;
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border-color);
}

.dark .footer {
    color: #b0b0b0;
    border-top-color: #4a5568;
}

/* Custom scrollbar */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 10px;
}

.dark ::-webkit-scrollbar-track {
    background: #2d3748;
}

::-webkit-scrollbar-thumb {
    background: var(--diagnosis-text);
    border-radius: 10px;
}

.dark ::-webkit-scrollbar-thumb {
    background: #5a86c2;
}

::-webkit-scrollbar-thumb:hover {
    background: #3a5a80;
}

.dark ::-webkit-scrollbar-thumb:hover {
    background: #4a76b2;
}

/* Tables */
table {
    width: 100%;
    border-collapse: collapse;
    margin: 15px 0;
}

th {
    background-color: #f2f2f2;
    padding: 10px;
    text-align: left;
    border: 1px solid #ddd;
}

.dark th {
    background-color: #4a5568;
    color: var(--light-text);
}

td {
    padding: 10px;
    border: 1px solid #ddd;
}

.dark td {
    border-color: #4a5568;
}

/* Accordions */
.accordion {
    background-color: var(--card-bg) !important;
    border-radius: 8px !important;
    box-shadow: 0 2px 4px var(--shadow-color) !important;
    margin-bottom: 1.5rem !important;
    border: 1px solid var(--border-color) !important;
}

.dark .accordion {
    background-color: #2d3748 !important;
    border-color: #4a5568 !important;
}

/* Checkbox group */
.checkbox-group {
    padding: 1rem !important;
}

/* Plot container */
.plot-container {
    border-radius: 8px !important;
    overflow: hidden !important;
    box-shadow: 0 2px 4px var(--shadow-color) !important;
}
"""

# Gradio UI with modern design and color scheme
with gr.Blocks(
    title="Breast Cancer Histopathology Analysis",
    css=custom_css,
    theme=gr.themes.Default(
        primary_hue="blue",
        secondary_hue="gray",
        neutral_hue="gray",
        radius_size="md",
        text_size="md",
    )
) as demo:
    # Header with logo and title
    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("""
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <svg style="width: 40px; height: 40px; margin-right: 15px;" fill="#4a6fa5" viewBox="0 0 24 24">
                    <path d="M12,3L2,12H5V20H19V12H22M12,7.7C14.1,7.7 15.8,9.4 15.8,11.5C15.8,13.6 14.1,15.3 12,15.3C9.9,15.3 8.2,13.6 8.2,11.5C8.2,9.4 9.9,7.7 12,7.7M7,18V10H17V18H7Z"/>
                </svg>
                <h1 style="margin: 0; color: var(--primary-text);">Breast Cancer Histopathology Analysis</h1>
            </div>
            <p class="description">
                <span style="color: var(--secondary-text);">AI-powered diagnostic assistance for breast tissue histopathology images</span><br>
                <span style="color: var(--secondary-text);">Combining deep learning with explainable AI for transparent results</span>
            </p>
            """)

    # Main content
    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            # Upload section
            with gr.Group():
                gr.Markdown("### 1. Upload Histopathology Image")
                image_input = gr.Image(
                    label="",
                    type="numpy",
                    elem_classes="upload-box",
                    interactive=True
                )
                submit_btn = gr.Button(
                    "Analyze Image",
                    variant="primary",
                    elem_classes="primary-button"
                )

            # Options section
            with gr.Accordion("⚙️ Analysis Options", open=False):
                zoom_levels = gr.CheckboxGroup(
                    choices=["Full (100%)", "Zoom 75%", "Zoom 50%", "Zoom 25%"],
                    value=["Full (100%)", "Zoom 75%", "Zoom 50%"],
                    label="Select Zoom Levels to Display",
                    interactive=True
                )

                confidence_threshold = gr.Slider(
                    minimum=50,
                    maximum=95,
                    value=70,
                    step=5,
                    label="Confidence Threshold for Subtype Classification",
                    interactive=True
                )

            # Information section
            with gr.Accordion("ℹ️ About This Tool", open=False):
                gr.Markdown("""
                <div class="card">
                    <h3 style="color: var(--primary-text);">How It Works</h3>
                    <p style="color: var(--primary-text);">This AI tool analyzes breast tissue histopathology images using deep learning models:</p>
                    <ol style="color: var(--primary-text);">
                        <li><strong>Binary Classification</strong>: Determines if tissue is benign or malignant</li>
                        <li><strong>Subtype Classification</strong>: Identifies specific tissue subtypes</li>
                        <li><strong>Attention Visualization</strong>: Shows which image regions influenced the decision</li>
                    </ol>

                    <h3 style="color: var(--primary-text); margin-top: 20px;">Interpretation Guide</h3>
                    <ul style="color: var(--primary-text);">
                        <li><strong>Heatmaps</strong>: Red areas indicate regions most important for the diagnosis</li>
                        <li><strong>Confidence Scores</strong>: Higher values (70%+) indicate more reliable predictions</li>
                        <li><strong>Subtype Analysis</strong>: Shows probabilities for all possible subtypes</li>
                    </ul>

                    <h3 style="color: var(--primary-text); margin-top: 20px;">Important Notes</h3>
                    <ul style="color: var(--primary-text);">
                        <li>This tool is for <strong>research and educational purposes only</strong></li>
                        <li>Always consult a qualified pathologist for clinical diagnosis</li>
                        <li>Results should not be used as sole basis for medical decisions</li>
                    </ul>
                </div>
                """)

            # Instructions
            gr.Markdown(f"""
            <div class="card">
                <h3 style="color: var(--primary-text); margin-top: 0;">📝 Instructions</h3>
                <ol style="color: var(--primary-text); padding-left: 20px;">
                    <li>Upload a high-quality histopathology image</li>
                    <li>Click "Analyze Image" to process</li>
                    <li>Review the diagnosis and visualization</li>
                    <li>Consult the detailed subtype analysis</li>
                    <li>Share results with your healthcare provider</li>
                </ol>
                <p style="font-size: 13px; color: var(--secondary-text); margin-bottom: 0;">
                <strong>Note:</strong> For best results, use images at 40x-400x magnification with clear tissue structures.
                </p>
            </div>
            """)

        with gr.Column(scale=2):
            # Results tabs
            with gr.Tabs():
                with gr.TabItem("📋 Diagnosis Summary", id="diagnosis"):
                    diagnosis_output = gr.HTML(
                        label="Diagnosis Results",
                        value="<div style='text-align: center; padding: 40px; color: var(--secondary-text);'>Analysis results will appear here after processing an image.</div>"
                    )

                with gr.TabItem("🔍 Visualization", id="visualization"):
                    plot_output = gr.Plot(
                        label="Model Attention Analysis",
                        visible=True
                    )

                with gr.TabItem("📊 Confidence Metrics", id="metrics"):
                    gr.Markdown("""
                    <div style="text-align: center; padding: 40px; color: var(--secondary-text);">
                        Confidence metrics and detailed probabilities will appear here after analysis.
                    </div>
                    """)

            # Detailed explanation
            with gr.Accordion("📚 Detailed Explanation", open=True):
                gr.Markdown("""
                <div class="card">
                    <h3 style="color: var(--primary-text); margin-top: 0;">Understanding Your Results</h3>
                    <p style="color: var(--primary-text);">After analysis, this section will provide detailed information about:</p>
                    <ul style="color: var(--primary-text);">
                        <li>The specific characteristics of your tissue sample</li>
                        <li>What the diagnosis means clinically</li>
                        <li>Recommended next steps based on the findings</li>
                        <li>Additional resources for more information</li>
                    </ul>
                </div>
                """)

            # Footer
            gr.Markdown("""
            <div class="footer">
                <p style="color: var(--secondary-text);">This AI tool was developed for research purposes using deep learning models trained on histopathology images.</p>
                <p style="color: var(--secondary-text);">For clinical concerns, please consult with a qualified medical professional.</p>
                <p style="font-size: 0.8rem; color: var(--secondary-text);">v2.1.0 | Last updated: July 2023</p>
            </div>
            """)

    # Event handling
    submit_btn.click(
        fn=analyze_image,
        inputs=image_input,
        outputs=[diagnosis_output, plot_output]
    )
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(shape=(500, 500)),
    outputs=gr.Label(num_top_classes=2),
    title="Breast Cancer Histopathology Classifier"
)
iface.launch()
