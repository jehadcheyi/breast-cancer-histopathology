---
title: Breast AI PRO - Histopathology Classifier
emoji: 🧠
colorFrom: pink
colorTo: red
sdk: gradio
sdk_version: "4.20.0"
app_file: app.py
pinned: true
---


# 🧠 BREAST AI PRO - Deep Learning System for Breast Cancer Detection

## 🔍 Overview

**BREAST AI PRO** is a state-of-the-art deep learning system designed to detect and classify **breast cancer** subtypes from **histopathology images** using **Convolutional Neural Networks (CNNs)**. This powerful application leverages a multi-stage classification pipeline to first determine whether a sample is *benign* or *malignant*, and then further classifies the sample into a more detailed subtype using dedicated models. A built-in **Grad-CAM visualization** system enables users to see where the model is focusing its attention during predictions.

This application is powered by **TensorFlow**, **Gradio**, and **Hugging Face Hub**, making it efficient, scalable, and user-friendly.

---

## 🌟 Features

- ✅ **Multi-Stage Classifier**: First performs binary classification, then detailed subtype classification.
- 📊 **Subtype Prediction**:
  - Benign: Adenosis, Fibroadenoma, Phyllodes Tumor, Tubular Adenoma
  - Malignant: Ductal Carcinoma, Lobular Carcinoma, Mucinous Carcinoma, Papillary Carcinoma
- 🎯 **High Accuracy**: Utilizes pretrained and optimized deep learning models with >98% classification accuracy.
- 🔥 **Grad-CAM Visualization**: Displays attention heatmaps to interpret model decisions.
- 🧪 **Sample Image Gallery**: Users can test the system with built-in sample images.
- 🌐 **Multilingual UI**: Interface labels and guidance provided in **English**, **Arabic**, and **Kurdish**.
- 📦 **Easy to Use**: Web-based application built with Gradio for ease of access and deployment.
- 💻 **Lightweight and Efficient**: Supports `.jpg`, `.png`, `.jpeg`, `.tif` image formats.

---

## 🏗️ Project Architecture

The BREAST AI system is composed of several components:

1. **Model Loading & Caching**
   - Three Keras models are downloaded from Hugging Face Hub and loaded into memory:
     - `binary.h5` for benign vs. malignant
     - `benign.h5` for benign subtypes
     - `malignant.h5` for malignant subtypes

2. **Prediction Pipeline**
   - Input image is resized, normalized, and passed to the binary classifier.
   - Based on the result, a corresponding subtype model is selected.
   - Predictions are displayed along with probabilities.

3. **Grad-CAM Attention**
   - Visualizes which areas of the image the model used for prediction.
   - Multiple zoom levels (100%, 75%, 50%, 25%) are displayed for better interpretability.

4. **Interactive Frontend with Gradio**
   - Upload images or select from examples.
   - View diagnostic summary and Grad-CAM visualizations.
   - Interactive design with stylized components for clear communication.

---

## 🧠 How It Works

### Step 1: Binary Classification
- Determines whether the tissue is **benign** or **malignant** using a CNN-based model.

### Step 2: Subtype Classification
- If **benign**, it identifies:
  - **Adenosis**, **Fibroadenoma**, **Phyllodes Tumor**, **Tubular Adenoma**
- If **malignant**, it identifies:
  - **Ductal Carcinoma**, **Lobular Carcinoma**, **Mucinous Carcinoma**, **Papillary Carcinoma**

### Step 3: Grad-CAM Heatmap Generation
- Grad-CAM is used to extract feature maps from the final convolutional layers.
- Highlights where the model "looked" during its decision.
- Multiple zoomed-in crops are processed for enhanced analysis.

---

## 🧬 Example Subtypes

| Category  | Code | Full Name              |
|----------|------|------------------------|
| Benign   | A    | Adenosis               |
| Benign   | F    | Fibroadenoma           |
| Benign   | PT   | Phyllodes Tumor        |
| Benign   | TA   | Tubular Adenoma        |
| Malignant| DC   | Ductal Carcinoma       |
| Malignant| LC   | Lobular Carcinoma      |
| Malignant| MC   | Mucinous Carcinoma     |
| Malignant| PC   | Papillary Carcinoma    |

---

## 🧪 Sample Image Gallery

The app provides built-in example images from a folder named `example/`. Users can:
- Upload custom pathology images for evaluation.
- Click on sample images to run instant predictions.

---

## 📦 Installation and Setup

### ✅ Requirements

- Python 3.7+
- TensorFlow >= 2.4
- Gradio
- OpenCV
- NumPy
- Matplotlib
- Pillow
- Hugging Face Hub

### 🔧 Install Dependencies

```bash
pip install tensorflow gradio opencv-python matplotlib numpy pillow huggingface-hub
🔐 Set Hugging Face Access Token
bash
Copy
Edit
export jehad="your_hf_token_here"
Replace "your_hf_token_here" with your actual Hugging Face token.

🚀 Run the App
bash
Copy
Edit
python app.py
or if using Jupyter/Colab:

python
Copy
Edit
demo.launch(share=True)
The app will start in your browser and allow local or remote access (if share=True is used).

🔬 Grad-CAM Visualization
Grad-CAM is a visualization technique that reveals which areas of the input image were important for the model’s decision. In BREAST AI PRO:

Red areas: High model attention (suspicious/malignant regions)

Green areas: Low model attention (normal tissue)

Overlay: Heatmap blended with the original image

Multi-level zoom: Helps pathologists analyze patterns in different scales

🌍 Multi-Language Interface
This application includes labels and guidance in:

English

Arabic

Kurdish

Making it accessible to a wide range of users in international medical and academic communities.

📁 Folder Structure
bash
Copy
Edit
.
├── example/             # Sample histopathology images
├── model_cache/         # Cached models from Hugging Face
├── app.py               # Main application script
├── README.md            # This file
💬 Future Improvements
📈 Real-time performance monitoring

🔍 Segmentation of lesion areas

🌐 Web deployment with Docker or FastAPI

📱 Mobile App Integration

🧪 Integration with electronic health records (EHR)

❗ Disclaimer
BREAST AI PRO is intended for research and educational purposes only. It should not be used for clinical diagnosis or treatment without the supervision of qualified medical professionals.

📜 License
This project is licensed under MIT License. Feel free to use and modify it for academic or research purposes.

🙌 Acknowledgments
Built using TensorFlow, Gradio, and Hugging Face Hub

Sample histopathology images adapted from public datasets

Special thanks to researchers and medical experts for annotations and insights

📬 Contact
For collaboration, questions, or suggestions, feel free to reach out:

Developer/Researcher: Jehad Cheyi
Institution: Breast AI Technologies | 2025