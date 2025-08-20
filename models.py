import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
import warnings
from huggingface_hub import hf_hub_download

# Suppress noisy warnings (keeps output cleaner)
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.get_logger().setLevel('ERROR')

# Debug flag (set to True to print helpful debug info)
DEBUG = False

# Hugging Face Hub configuration
REPO_ID = "jehadcheyi/bc_models"
HF_TOKEN = os.getenv("jehad")  # Ensure this env var is set

# Model paths
model_paths = {
    'binary': 'binary.h5',
    'benign': 'benign.h5',
    'malignant': 'malignant.h5'
}

# Image and class definitions
img_size = (128, 128)
target_size = (500, 500)

classes = {
    'binary': ['benign', 'malignant'],
    'benign': ['A', 'F', 'PT', 'TA'],
    'malignant': ['DC', 'LC', 'MC', 'PC'],
}

# Initialize models dictionary
models = {}

# Analysis history for statistics
analysis_history = []

# Grad-CAM model
grad_model = None

def load_all_models():
    """Load all models from Hugging Face Hub"""
    global models, grad_model
    
    # Model Loading
    print("🔄 Initializing AI Models...")
    
    # 1. First verify files exist on HF hub (download to cache)
    print("📋 Verifying model files...")
    for name, filename in model_paths.items():
        try:
            local_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                token=HF_TOKEN,
                cache_dir="model_cache"
            )
            print(f"✅ Found {name} model at: {local_path}")
        except Exception as e:
            print(f"❌ Error verifying {name} model: {str(e)}")
            raise
    
    # 2. Load all models
    print("\n🧠 Loading AI models into memory...")
    try:
        for name, filename in model_paths.items():
            local_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                token=HF_TOKEN,
                cache_dir="model_cache"
            )
            # Use CPU device to avoid environment GPU-related issues during load
            with tf.device('/CPU:0'):
                models[name] = load_model(local_path, compile=False)
            print(f"🎯 Successfully loaded {name} model")
    except Exception as e:
        print(f"❌ Model loading failed: {str(e)}")
        raise
    
    # 3. Grad-CAM Setup
    print("\n🔬 Setting up Grad-CAM visualization...")
    try:
        binary_model = models['binary']
        last_conv = None
        
        # Find last conv layer (most reliable approach)
        for layer in reversed(binary_model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv = layer.name
                break
        
        if last_conv is None:
            raise ValueError("No convolutional layer found in binary model")
        
        grad_model = Model(
            inputs=binary_model.inputs,
            outputs=[
                binary_model.get_layer(last_conv).output,
                binary_model.output
            ]
        )
    except Exception as e:
        print(f"❌ Grad-CAM setup failed: {str(e)}")
        raise
    
    print("\n🚀 All AI models loaded successfully!")
    return models, grad_model

def predict_with_uncertainty(img, n_samples=10):
    """
    Perform Monte Carlo prediction with dropout enabled for uncertainty estimation
    Returns predictions with uncertainty metrics
    """
    try:
        # Prepare image
        resized = tf.image.resize(img, img_size)
        img_array = np.expand_dims(resized, axis=0) / 255.0
        
        # Binary classification with uncertainty
        binary_preds = []
        for _ in range(n_samples):
            # Enable dropout during inference
            binary_pred = models['binary'](img_array, training=True)
            binary_preds.append(binary_pred.numpy())
        
        binary_preds = np.vstack(binary_preds)
        binary_mean = np.mean(binary_preds, axis=0)[0]
        binary_std = np.std(binary_preds, axis=0)[0]
        
        # Get binary prediction
        binary_class = np.argmax(binary_mean)
        binary_conf = binary_mean[binary_class] * 100
        binary_uncertainty = binary_std[binary_class] * 100
        
        # Determine which subtype model to use based on binary prediction
        subtype_model_name = classes['binary'][binary_class]
        subtype_model = models[subtype_model_name]
        
        # Subtype classification with uncertainty
        subtype_preds = []
        for _ in range(n_samples):
            subtype_pred = subtype_model(img_array, training=True)
            subtype_preds.append(subtype_pred.numpy())
        
        subtype_preds = np.vstack(subtype_preds)
        subtype_mean = np.mean(subtype_preds, axis=0)[0]
        subtype_std = np.std(subtype_preds, axis=0)[0]
        
        # Get subtype prediction
        subtype_class = np.argmax(subtype_mean)
        subtype_conf = subtype_mean[subtype_class] * 100
        subtype_uncertainty = subtype_std[subtype_class] * 100
        
        # Get class names
        binary_label = classes['binary'][binary_class]
        subtype_label = classes[binary_label][subtype_class]
        
        # Create dictionary of all subtype probabilities with uncertainties
        subtype_probs = {}
        for i, class_name in enumerate(classes[binary_label]):
            subtype_probs[class_name] = {
                'prob': subtype_mean[i] * 100,
                'uncertainty': subtype_std[i] * 100
            }
        
        # Return comprehensive results
        return {
            'binary_class': binary_class,
            'binary_label': binary_label,
            'binary_conf': binary_conf,
            'binary_uncertainty': binary_uncertainty,
            'subtype_class': subtype_class,
            'subtype_label': subtype_label,
            'subtype_conf': subtype_conf,
            'subtype_uncertainty': subtype_uncertainty,
            'subtype_probs': subtype_probs
        }
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise