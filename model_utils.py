import tensorflow as tf
from tensorflow.keras.models import Model
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model as keras_load_model
import numpy as np
import cv2

from config import model_paths, REPO_ID, HF_TOKEN, models, classes, img_size, target_size

# ---------- Model Loading ----------
def load_models():
    print("🔄 Initializing AI Models...")
    print("📋 Verifying model files...")
    for name, filename in model_paths.items():
        local_path = hf_hub_download(repo_id=REPO_ID, filename=filename, token=HF_TOKEN, cache_dir="model_cache")
        print(f"✅ Found {name} model at: {local_path}")

    print("\n🧠 Loading AI models into memory...")
    for name, filename in model_paths.items():
        local_path = hf_hub_download(repo_id=REPO_ID, filename=filename, token=HF_TOKEN, cache_dir="model_cache")
        with tf.device('/CPU:0'):
            models[name] = keras_load_model(local_path, compile=False)
        print(f"🎯 Successfully loaded {name} model")
    print("\n🚀 All AI systems loaded successfully!")
    return models

# ---------- Grad-CAM Setup ----------
def get_grad_model():
    binary_model = models['binary']
    last_conv = None
    for layer in reversed(binary_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer.name
            break
    if last_conv is None:
        raise ValueError("No convolutional layer found in binary model")
    grad_model = Model(inputs=binary_model.inputs,
                       outputs=[binary_model.get_layer(last_conv).output, binary_model.output])
    return grad_model

# ---------- Prediction with Uncertainty ----------
def predict_with_uncertainty(img, n_samples=10):
    from image_utils import prep_img
    arr = prep_img(img)
    binary_predictions = []
    for _ in range(n_samples):
        with tf.device('/CPU:0'):
            pred = models['binary'].predict(arr, verbose=0)
            binary_predictions.append(pred[0])
    binary_predictions = np.array(binary_predictions)
    binary_mean = np.mean(binary_predictions, axis=0)
    binary_std = np.std(binary_predictions, axis=0)
    pred_idx = int(np.argmax(binary_mean))
    binary_conf = float(100.0 * np.max(binary_mean))
    binary_uncertainty = float(100.0 * binary_std[pred_idx])
    label = classes['binary'][pred_idx]
    subtype_predictions = []
    for _ in range(n_samples):
        with tf.device('/CPU:0'):
            pred = models[label].predict(arr, verbose=0)
            subtype_predictions.append(pred[0])
    subtype_predictions = np.array(subtype_predictions)
    subtype_mean = np.mean(subtype_predictions, axis=0)
    subtype_std = np.std(subtype_predictions, axis=0)
    subtype_idx = int(np.argmax(subtype_mean))
    subtype_conf = float(100.0 * np.max(subtype_mean))
    subtype_uncertainty = float(100.0 * subtype_std[subtype_idx])
    subtype_probs = {}
    if label == 'benign':
        subtype_code = classes['benign'][subtype_idx]
        subtype_name = classes['benign_names'][subtype_code]
        for code, prob, std in zip(classes['benign'], subtype_mean, subtype_std):
            subtype_probs[classes['benign_names'][code]] = {'mean': float(prob * 100.0), 'std': float(std * 100.0)}
    else:
        subtype_code = classes['malignant'][subtype_idx]
        subtype_name = classes['malignant_names'][subtype_code]
        for code, prob, std in zip(classes['malignant'], subtype_mean, subtype_std):
            subtype_probs[classes['malignant_names'][code]] = {'mean': float(prob * 100.0), 'std': float(std * 100.0)}
    return (pred_idx, binary_conf, binary_uncertainty, subtype_name, subtype_conf, subtype_uncertainty, subtype_probs)

# ---------- Cropping helper ----------
def get_crop_box(shape, percent):
    h, w = shape[:2]
    crop_h = int(h * percent / 100)
    crop_w = int(w * percent / 100)
    left = (w - crop_w) // 2
    top = (h - crop_h) // 2
    return left, top, left + crop_w, top + crop_h
