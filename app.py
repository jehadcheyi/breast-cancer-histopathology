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

# Verify and load models
print("Checking model files...")
for name, path in model_paths.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    print(f"Found {name} model at: {path}")

print("Loading models...")
try:
    models = {name: load_model(path) for name, path in model_paths.items()}
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    raise

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

def analyze_image(input_img):
    try:
        img = input_img.astype('uint8')
        pred_idx, binary_conf, subtype, subtype_conf, subtype_probs = predict(img)

        # Create diagnosis HTML
        diagnosis = classes['binary'][pred_idx].upper()
        diagnosis_color = "#4CAF50" if diagnosis == "BENIGN" else "#F44336"
        
        subtype_table = "<table style='width:100%; border-collapse: collapse; margin: 15px 0;'>"
        subtype_table += "<tr style='background-color: #f2f2f2;'><th>Subtype</th><th>Confidence</th></tr>"
        for name, prob in subtype_probs.items():
            subtype_table += f"<tr><td>{name}</td><td style='font-weight: bold;'>{prob:.1f}%</td></tr>"
        subtype_table += "</table>"

        diagnosis_html = f"""
        <div style="border: 2px solid {diagnosis_color}; border-radius: 8px; padding: 20px; margin: 20px 0;">
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

        return diagnosis_html

    except Exception as e:
        error_html = f"""
        <div style="border: 2px solid #FF5722; border-radius: 8px; padding: 20px; margin: 20px 0; background-color: #fff3e0;">
            <h2 style="color: #FF5722; margin: 0;">Error Processing Image</h2>
            <p>{str(e)}</p>
        </div>
        """
        return error_html

# Get sample images from 'sample' folder
sample_images = []
sample_folder = "sample"
if os.path.exists(sample_folder):
    for file in os.listdir(sample_folder):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            sample_images.append(os.path.join(sample_folder, file))

# Gradio UI
with gr.Blocks(title="Breast Cancer Histopathology Analysis") as demo:
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## Upload Histopathology Image")
            image_input = gr.Image(type="numpy")
            submit_btn = gr.Button("Analyze Image", variant="primary")
            
            # Gallery of sample images
            if sample_images:
                gr.Markdown("## Sample Images")
                gallery = gr.Gallery(
                    value=sample_images,
                    label="Click on a sample image to load it",
                    columns=3,
                    height="auto"
                )
                
                def select_sample_image(evt: gr.SelectData):
                    return evt.value
                
                gallery.select(select_sample_image, None, image_input)
        
        with gr.Column(scale=2):
            diagnosis_output = gr.HTML(
                value="<div style='text-align: center; padding: 40px;'>Analysis results will appear here after processing an image.</div>"
            )
    
    submit_btn.click(
        fn=analyze_image,
        inputs=image_input,
        outputs=diagnosis_output
    )

demo.launch(share=True)