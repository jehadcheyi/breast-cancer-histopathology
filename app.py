
import os
import warnings
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import img_to_array
import gradio as gr
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model as keras_load_model

# -----------------------------------------------------------------------------
# Environment / logging
# -----------------------------------------------------------------------------
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
tf.get_logger().setLevel("ERROR")

# Toggle for verbose debugging (prints numeric diagnostics)
DEBUG = False

# -----------------------------------------------------------------------------
# Configuration and classes (kept from your original)
# -----------------------------------------------------------------------------
REPO_ID = "jehadcheyi/bc_models"
HF_TOKEN = os.getenv("jehad")  # must be set in environment
if HF_TOKEN is None:
    raise EnvironmentError("Hugging Face token not found. Set environment variable 'jehad'")

model_paths = {
    "binary": "binary.h5",
    "benign": "benign.h5",
    "malignant": "malignant.h5",
}

img_size = (128, 128)
target_size = (500, 500)

classes = {
    "binary": ["benign", "malignant"],
    "benign": ["A", "F", "PT", "TA"],
    "malignant": ["DC", "LC", "MC", "PC"],
    "benign_names": {"A": "Adenosis", "F": "Fibroadenoma", "PT": "Phyllodes Tumor", "TA": "Tubular Adenoma"},
    "malignant_names": {"DC": "Ductal Carcinoma", "LC": "Lobular Carcinoma", "MC": "Mucinous Carcinoma", "PC": "Papillary Carcinoma"},
}

# Containers for models & gradcam model
models = {}
grad_model = None  # will be set after load

# -----------------------------------------------------------------------------
# Model loading (download from HF and load with CPU)
# -----------------------------------------------------------------------------
def load_models():
    print("🔄 Initializing AI Models...")
    for name, filename in model_paths.items():
        try:
            local_path = hf_hub_download(repo_id=REPO_ID, filename=filename, token=HF_TOKEN, cache_dir="model_cache")
            print(f"📥 Found {name} model at: {local_path} - loading...")
            with tf.device("/CPU:0"):
                models[name] = keras_load_model(local_path, compile=False)
            print(f"✅ Loaded {name}")
        except Exception as e:
            print(f"❌ Failed to load {name}: {e}")
            raise
    print("🧠 All models loaded.")
    return models

load_models()

# -----------------------------------------------------------------------------
# Select best conv layer and build grad_model
# -----------------------------------------------------------------------------
def find_best_conv_layer(model):
    """Return name of Conv2D layer with the largest spatial size (h*w)."""
    best_name = None
    best_size = 0
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            try:
                shape = layer.output_shape  # (None, h, w, c)
                if shape and len(shape) >= 3 and shape[1] is not None and shape[2] is not None:
                    size = int(shape[1]) * int(shape[2])
                    if size > best_size:
                        best_size = size
                        best_name = layer.name
            except Exception:
                continue
    return best_name

def build_grad_model(layer_name=None):
    """Set global grad_model using the chosen conv layer."""
    global grad_model
    binary_model = models["binary"]
    if layer_name is None:
        layer_name = find_best_conv_layer(binary_model)
        if DEBUG:
            print("Auto-selected conv layer:", layer_name)
    if layer_name is None:
        # fallback: pick any Conv2D by scanning reversed layers (original approach)
        for layer in reversed(binary_model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                layer_name = layer.name
                break
    if layer_name is None:
        raise ValueError("No Conv2D layer found in binary model to construct Grad-CAM.")
    grad_model = Model(inputs=binary_model.inputs, outputs=[binary_model.get_layer(layer_name).output, binary_model.output])
    if DEBUG:
        print("Grad-CAM model built with conv layer:", layer_name)
    return layer_name

# build grad model now
selected_conv = build_grad_model()

# -----------------------------------------------------------------------------
# Image preprocessing (robust)
# -----------------------------------------------------------------------------
def prep_img(img):
    """Return batch (1, H, W, 3) float32 normalized array for model inference."""
    if img is None:
        raise ValueError("No image provided")

    arr = img.copy()

    # Convert floats in [0,1] to uint8
    if arr.dtype in (np.float32, np.float64):
        if arr.max() <= 1.0:
            arr = (arr * 255).astype("uint8")
        else:
            arr = arr.astype("uint8")
    else:
        arr = arr.astype("uint8")

    # Expand channels if grayscale or single-channel
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.concatenate([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[..., :3]

    pil = Image.fromarray(arr)
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    pil = pil.resize(img_size, Image.LANCZOS)
    arr = img_to_array(pil).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    if DEBUG:
        print("prep_img ->", arr.shape, arr.dtype, arr.min(), arr.max())
    return arr

# -----------------------------------------------------------------------------
# Prediction helpers
# -----------------------------------------------------------------------------
def binary_predict_prob(arr_batch):
    """Return malignant probability for each sample in batch (float in [0,1])."""
    # arr_batch: shape (N,H,W,3)
    with tf.device("/CPU:0"):
        preds = models["binary"].predict(arr_batch, verbose=0)
    preds = np.asarray(preds)
    if preds.ndim == 2 and preds.shape[1] > 1:
        # softmax multi-d output: assume index 1 = malignant
        probs = preds[:, 1]
    elif preds.ndim == 2 and preds.shape[1] == 1:
        # single sigmoid output
        probs = preds[:, 0]
    elif preds.ndim == 1:
        # rare flattened output
        probs = preds
    else:
        # fallback
        probs = np.squeeze(preds)
    return probs

def predict_image(img):
    """Run binary + subtype predictions and return readable values."""
    arr = prep_img(img)
    probs = binary_predict_prob(arr)
    prob = float(probs[0])
    # Determine predicted index and confidence (match original UI)
    if prob >= 0.5:
        pred_idx = 1
        binary_conf = prob * 100.0
    else:
        pred_idx = 0
        binary_conf = (1.0 - prob) * 100.0

    label = classes["binary"][pred_idx]  # 'benign' or 'malignant'

    # Subtype prediction
    with tf.device("/CPU:0"):
        subtype_pred = models[label].predict(arr, verbose=0)
    subtype_pred = np.asarray(subtype_pred)
    subtype_idx = int(np.argmax(subtype_pred[0]))
    subtype_conf = float(np.max(subtype_pred[0]) * 100.0)
    if label == "benign":
        subtype_code = classes["benign"][subtype_idx]
        subtype_name = classes["benign_names"][subtype_code]
        subtype_probs = {classes["benign_names"][code]: float(prob * 100.0) for code, prob in zip(classes["benign"], subtype_pred[0])}
    else:
        subtype_code = classes["malignant"][subtype_idx]
        subtype_name = classes["malignant_names"][subtype_code]
        subtype_probs = {classes["malignant_names"][code]: float(prob * 100.0) for code, prob in zip(classes["malignant"], subtype_pred[0])}

    if DEBUG:
        print("predict_image -> pred_idx", pred_idx, "binary_conf", binary_conf, "subtype", subtype_name, "subtype_conf", subtype_conf)
    return pred_idx, binary_conf, subtype_name, subtype_conf, subtype_probs

# -----------------------------------------------------------------------------
# Grad-CAM implementation (raw heatmap float in [0,1])
# -----------------------------------------------------------------------------
def gradcam_raw(img, pred_idx):
    """
    Compute Grad-CAM heatmap (float, resized to target_size) in [0,1].
    Returns heatmap_resized (H,W) float32.
    """
    arr = prep_img(img)  # (1,H,W,3)
    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(arr)
        # handle single-logit (sigmoid) or multi-class outputs
        if preds.shape[-1] == 1:
            target_label = classes["binary"][pred_idx]
            if target_label == "malignant":
                loss = preds[:, 0]
            else:
                loss = 1.0 - preds[:, 0]
        else:
            loss = preds[:, pred_idx]

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        if DEBUG:
            print("gradcam_raw: grads is None")
        return np.zeros(target_size, dtype=np.float32)

    grads = grads[0]        # (h, w, channels)
    conv_out = conv_outputs[0]  # (h, w, channels)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))  # (channels,)
    # Weighted combination
    heatmap = tf.reduce_sum(conv_out * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)

    max_val = tf.reduce_max(heatmap)
    if max_val is None or float(max_val.numpy()) <= 1e-8:
        if DEBUG:
            print("gradcam_raw: heatmap max too small:", None if max_val is None else float(max_val.numpy()))
        return np.zeros(target_size, dtype=np.float32)

    heatmap = heatmap / max_val
    heatmap_np = heatmap.numpy().astype(np.float32)
    heatmap_resized = cv2.resize(heatmap_np, target_size, interpolation=cv2.INTER_LINEAR)
    # Smooth for visibility
    heatmap_resized = cv2.GaussianBlur(heatmap_resized, (7, 7), 0)
    # Clip and normalize again to [0,1]
    heatmap_resized = np.clip(heatmap_resized, 0.0, 1.0)
    minv, maxv = float(np.min(heatmap_resized)), float(np.max(heatmap_resized))
    if maxv > minv:
        heatmap_resized = (heatmap_resized - minv) / (maxv - minv)
    return heatmap_resized

def gradcam_colormap_from_raw(raw_heatmap):
    """Given raw heatmap (H,W float [0,1]) return colored BGR uint8 heatmap."""
    heat_uint8 = np.uint8(255 * raw_heatmap)
    heat_uint8 = cv2.normalize(heat_uint8, None, 0, 255, cv2.NORM_MINMAX)
    colored = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)  # BGR
    # boost intensity a bit
    colored = cv2.convertScaleAbs(colored, alpha=1.2, beta=0)
    return colored

# SmoothGrad: average raw heatmaps on noisy inputs
def smooth_gradcam(img, pred_idx, n_samples=12, noise_level=0.12):
    """Return averaged raw heatmap (float [0,1])"""
    arr0 = prep_img(img)[0]  # shape (H,W,3)
    heatmaps = []
    for i in range(n_samples):
        noise = np.random.normal(0, noise_level, arr0.shape).astype("float32")
        noisy = np.clip(arr0 + noise, 0.0, 1.0)
        # convert noisy back to uint8 RGB input for gradcam_raw by resizing from model input size to original? Simpler:
        # Create pseudo image: convert noisy back to uint8 RGB  - we need the same size as original img for prep_img to work.
        # So convert noisy to uint8 and scale up to original image size used earlier:
        noisy_img = (noisy * 255).astype("uint8")
        # Now upsample to original image approximate size - we can reuse resized image approach:
        # Build a temporary PIL image from noisy_img and scale to original input's size (we will pass to prep_img again)
        pil = Image.fromarray(noisy_img)
        pil = pil.resize((img.shape[1], img.shape[0]), Image.BILINEAR)
        noisy_np = np.array(pil)
        raw = gradcam_raw(noisy_np, pred_idx)
        heatmaps.append(raw)
    avg = np.mean(heatmaps, axis=0)
    # Normalize
    minv, maxv = float(np.min(avg)), float(np.max(avg))
    if maxv > minv:
        avg = (avg - minv) / (maxv - minv)
    return avg

# -----------------------------------------------------------------------------
# Tile-based probability heatmap and top-k tiles
# -----------------------------------------------------------------------------
def tile_probability_map(img, tile_size=128, stride=None, batch_size=16, top_k=6):
    """
    Slide non-overlapping or strided windows over the input RGB image (numpy).
    Returns:
      - heatmap_up (H, W) float in [0,1] of malignant probabilities per pixel (upsampled)
      - overlay_bgr (H, W, 3) BGR uint8 overlay of tile heatmap on image
      - top_tiles: list of (thumbnail_rgb_uint8, (x,y), prob)
    """
    if stride is None:
        stride = tile_size

    h, w = img.shape[:2]
    grid_y = list(range(0, max(1, h - tile_size + 1), stride))
    grid_x = list(range(0, max(1, w - tile_size + 1), stride))
    if len(grid_y) == 0:
        grid_y = [0]
    if len(grid_x) == 0:
        grid_x = [0]

    coords = []
    tiles = []
    for y in grid_y:
        for x in grid_x:
            tile = img[y : y + tile_size, x : x + tile_size]
            if tile.shape[0] == 0 or tile.shape[1] == 0:
                continue
            coords.append((x, y))
            # if tile smaller than tile_size, pad with zeros (or resize) - we'll resize in prep_img
            tiles.append(tile)

    if len(tiles) == 0:
        # fallback: single whole image
        tiles = [img]
        coords = [(0, 0)]

    # Batch predict
    probs = []
    for i in range(0, len(tiles), batch_size):
        batch_tiles = tiles[i : i + batch_size]
        batch_arrs = np.vstack([prep_img(t) for t in batch_tiles])  # shape (N,H,W,3)
        batch_probs = binary_predict_prob(batch_arrs)
        probs.extend(batch_probs.tolist())

    probs = np.array(probs, dtype=np.float32)
    # Build grid map
    heat_grid = np.zeros((len(grid_y), len(grid_x)), dtype=np.float32)
    idx = 0
    for yi, y in enumerate(grid_y):
        for xi, x in enumerate(grid_x):
            if idx < len(probs):
                heat_grid[yi, xi] = probs[idx]
            idx += 1

    # Upsample to image size
    heat_up = cv2.resize(heat_grid, (w, h), interpolation=cv2.INTER_LINEAR)
    # Normalize
    minv, maxv = float(np.min(heat_up)), float(np.max(heat_up))
    if maxv > minv:
        heat_up = (heat_up - minv) / (maxv - minv)
    else:
        heat_up = np.zeros_like(heat_up)

    # Colorize
    heat_uint8 = np.uint8(255 * heat_up)
    colored = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)  # BGR

    # Overlay on original (convert input to BGR)
    base_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    overlay_bgr = cv2.addWeighted(base_bgr, 0.6, colored, 0.4, 0)

    # Build top-k tiles
    tile_info = []
    for (x, y), p, tile in zip(coords, probs, tiles):
        thumb = cv2.resize(tile, (128, 128), interpolation=cv2.INTER_LINEAR)
        thumb_rgb = thumb.copy()
        tile_info.append((thumb_rgb, (x, y), float(p)))

    tile_info_sorted = sorted(tile_info, key=lambda x: x[2], reverse=True)
    top_tiles = tile_info_sorted[:top_k]
    # For gallery, return list of RGB numpy images
    top_imgs = [t[0] for t in top_tiles]
    top_meta = [(t[1], t[2]) for t in top_tiles]

    return heat_up, overlay_bgr, top_imgs, top_meta

# -----------------------------------------------------------------------------
# Visualization helpers (matplotlib figure)
# -----------------------------------------------------------------------------
def create_visual_plot(img_rgb, raw_heatmap_float, tile_overlay_bgr=None, show_tile=False):
    """
    Create Matplotlib figure with original, Grad-CAM heatmap and overlay.
    raw_heatmap_float: (H,W) in [0,1] resized to target_size
    tile_overlay_bgr: overlay from tile heatmap (can be original resolution)
    """
    try:
        # prepare images (convert to display size target_size)
        if img_rgb.dtype != np.uint8:
            img_disp = (np.clip(img_rgb, 0.0, 1.0) * 255).astype("uint8") if img_rgb.max() <= 1.0 else img_rgb.astype("uint8")
        else:
            img_disp = img_rgb.copy()

        img_resized = cv2.resize(img_disp, target_size, interpolation=cv2.INTER_LINEAR)
        heat_col = gradcam_colormap_from_raw(raw_heatmap_float)
        # overlay gradcam onto image (image BGR vs heat_col BGR)
        img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
        overlay_bgr = cv2.addWeighted(img_bgr, 0.6, heat_col, 0.4, 0)
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

        heat_rgb = cv2.cvtColor(heat_col, cv2.COLOR_BGR2RGB)
        img_rgb_disp = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Create figure (if tile overlay requested, add extra column)
        fig_cols = 3 if not show_tile else 4
        fig, axs = plt.subplots(1, fig_cols, figsize=(4 * fig_cols, 4), dpi=100)
        if fig_cols == 3:
            axs[0].imshow(img_rgb_disp); axs[0].set_title("🔬 Original"); axs[0].axis("off")
            axs[1].imshow(heat_rgb); axs[1].set_title("🌡️ Grad-CAM"); axs[1].axis("off")
            axs[2].imshow(overlay_rgb); axs[2].set_title("🎯 Overlay"); axs[2].axis("off")
        else:
            axs[0].imshow(img_rgb_disp); axs[0].set_title("🔬 Original"); axs[0].axis("off")
            axs[1].imshow(heat_rgb); axs[1].set_title("🌡️ Grad-CAM"); axs[1].axis("off")
            axs[2].imshow(overlay_rgb); axs[2].set_title("🎯 Overlay"); axs[2].axis("off")
            # tile overlay (if provided) - convert BGR->RGB
            if tile_overlay_bgr is not None:
                tile_rgb = cv2.cvtColor(tile_overlay_bgr, cv2.COLOR_BGR2RGB)
                tile_rgb_resized = cv2.resize(tile_rgb, target_size, interpolation=cv2.INTER_LINEAR)
                axs[3].imshow(tile_rgb_resized); axs[3].set_title("🗺️ Tile Heatmap"); axs[3].axis("off")
            else:
                axs[3].text(0.5, 0.5, "No tile heatmap", ha="center", va="center"); axs[3].axis("off")

        plt.tight_layout()
        return fig
    except Exception as e:
        if DEBUG:
            import traceback; traceback.print_exc()
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.text(0.5, 0.5, f"Visualization Error\n{e}", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return fig

# -----------------------------------------------------------------------------
# Sample images loader (unchanged)
# -----------------------------------------------------------------------------
def get_sample_images():
    sample_images = []
    sample_folder = "example"
    if os.path.exists(sample_folder):
        for file in os.listdir(sample_folder):
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".tif")):
                try:
                    img_path = os.path.join(sample_folder, file)
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        sample_images.append((img, file))
                except Exception as e:
                    print(f"Error loading sample image {file}: {e}")
    return sample_images

sample_images = get_sample_images()

# -----------------------------------------------------------------------------
# Main analyze function used by Gradio
# -----------------------------------------------------------------------------
def analyze_image_ui(input_img, use_smoothgrad, smooth_samples, smooth_noise,
                     use_tilemap, tile_size, tile_stride, top_k):
    """
    Returns: diagnosis_html (str), figure (matplotlib), tile_overlay (RGB numpy or None), top_tiles (list of RGB numpy), status (str)
    """
    if input_img is None:
        return create_waiting_message(), None, None, [], "🟡 Status: Waiting for Image"

    try:
        img = input_img.astype("uint8")  # Gradio supplies RGB numpy

        # Predictions
        pred_idx, binary_conf, subtype, subtype_conf, subtype_probs = predict_image(img)
        label = classes["binary"][pred_idx]
        diagnosis = label.upper()
        diag_color = "#4CAF50" if diagnosis == "BENIGN" else "#F44336"
        diag_emoji = "🟢" if diagnosis == "BENIGN" else "🔴"

        # Grad-CAM raw
        if use_smoothgrad:
            raw = smooth_gradcam(img, pred_idx, n_samples=max(4, int(smooth_samples)), noise_level=float(smooth_noise))
        else:
            raw = gradcam_raw(img, pred_idx)

        # Colored grad-cam & overlay will be created in plot function
        # Tile map optionally
        tile_overlay = None
        top_tiles = []
        top_meta = []
        if use_tilemap:
            _, tile_overlay_bgr, top_imgs, top_meta = tile_probability_map(img, tile_size=int(tile_size), stride=int(tile_stride), top_k=int(top_k))
            tile_overlay = cv2.cvtColor(tile_overlay_bgr, cv2.COLOR_BGR2RGB)
            top_tiles = top_imgs  # list of RGB numpy thumbs
        else:
            tile_overlay_bgr = None

        # Create HTML diagnosis report (preserve original style)
        # Risk & recs
        if diagnosis == "BENIGN":
            risk_level = "LOW RISK"
            risk_color = "#4CAF50"
            recommendations = """
            • Continue routine screening schedule
            • Monitor for any changes in symptoms
            • Maintain healthy lifestyle practices
            • Follow-up as recommended by physician
            """
        else:
            risk_level = "HIGH RISK - REQUIRES ATTENTION"
            risk_color = "#F44336"
            recommendations = """
            • IMMEDIATE consultation with oncologist required
            • Consider additional diagnostic procedures
            • Discuss treatment options with medical team
            • Seek second opinion if needed
            """

        # Build subtype table HTML
        subtype_table = "<div style='background:#f8f9fa;padding:12px;border-radius:8px;'><table style='width:100%;'>"
        for name, prob in sorted(subtype_probs.items(), key=lambda x: x[1], reverse=True):
            p = max(0.0, min(100.0, float(prob)))
            color = "#4CAF50" if p < 30 else "#FF9800" if p < 70 else "#F44336"
            subtype_table += f"<tr><td style='padding:6px;font-weight:bold'>{name}</td><td style='padding:6px;text-align:center;color:{color};font-weight:bold'>{p:.1f}%</td></tr>"
        subtype_table += "</table></div>"

        diagnosis_html = f"""
        <div style='border:3px solid {diag_color}; border-radius:12px; padding:16px; background: linear-gradient(135deg, {diag_color}15, {diag_color}05);'>
          <div style='text-align:center;'>
            <h2 style='color:{diag_color}; margin:0;'>{diag_emoji} {diagnosis}</h2>
            <p style='margin:4px 0; font-weight:bold;'>{binary_conf:.1f}% • {risk_level}</p>
          </div>
          <div style='display:flex; gap:12px; margin-top:12px;'>
            <div style='flex:1; background:white; padding:10px; border-radius:8px;'>
              <h4 style='margin:4px 0;'>Primary subtype</h4>
              <div style='font-weight:bold'>{subtype}</div>
              <div style='margin-top:6px; background:#3498db; color:white; display:inline-block; padding:4px 8px; border-radius:12px;'>{subtype_conf:.1f}%</div>
            </div>
            <div style='flex:2; background:white; padding:10px; border-radius:8px;'>
              <h4 style='margin:4px 0;'>All subtypes</h4>
              {subtype_table}
            </div>
          </div>
          <div style='margin-top:12px; background:#fff3cd; padding:8px; border-radius:6px;'>
            <strong>Disclaimer:</strong> AI output for research/educational purposes. Consult clinicians for decisions.
          </div>
        </div>
        """

        # Create plot figure, include tile overlay if requested
        fig = create_visual_plot(img, raw, tile_overlay_bgr if use_tilemap else None, show_tile=use_tilemap)

        status = f"{diag_emoji} Analysis complete — {diagnosis} ({binary_conf:.1f}%)"
        return diagnosis_html, fig, tile_overlay, top_tiles, status

    except Exception as e:
        if DEBUG:
            import traceback; traceback.print_exc()
        err_html = f"<div style='background:linear-gradient(45deg,#e74c3c,#c0392b);color:white;padding:12px;border-radius:8px;'><h3>Error</h3><p>{str(e)}</p></div>"
        return err_html, None, None, [], "🔴 Analysis failed"

# -----------------------------------------------------------------------------
# Waiting message helper
# -----------------------------------------------------------------------------
def create_waiting_message():
    return """
    <div style='background: linear-gradient(135deg, #e3f2fd, #bbdefb); padding: 20px; border-radius: 12px;'>
        <h3 style='color:#1976d2;'>AI Diagnostic System Ready</h3>
        <p>Upload a histopathology image to receive analysis (binary + subtype + attention map).</p>
    </div>
    """

# -----------------------------------------------------------------------------
# Gradio UI - keep original styling and layout, add new controls & outputs
# -----------------------------------------------------------------------------
custom_css = """
/* minimal retain of original styling, keep consistent */
.gradio-container { background: linear-gradient(135deg,#667eea,#764ba2); font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important; }
.main-header { background: rgba(255,255,255,0.95); border-radius:12px; padding:16px; margin-bottom:12px; text-align:center; }
.analysis-card { background: rgba(255,255,255,0.95); padding:12px; border-radius:12px; }
"""

title = "🏥 Advanced Histopathology Analysis System (with Grad-CAM + Tiles)"

with gr.Blocks(css=custom_css, theme=gr.themes.Soft(), title=title) as demo:
    # Header
    gr.HTML(f"""
    <div class="main-header">
        <h1 style="margin:0">{title}</h1>
        <p style="margin:6px 0 0 0;">Binary + subtype classification with Grad-CAM, SmoothGrad and tile heatmaps</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                gr.HTML('<div class="analysis-card">')
                gr.Markdown("### 📤 Upload Histopathology Image")
                image_input = gr.Image(label="Histopathology Image (RGB)", type="numpy", height=360, source="upload")
                gr.Markdown("### ⚙️ Visualization Options")
                with gr.Row():
                    use_sg = gr.Checkbox(label="Use SmoothGrad (smoother Grad-CAM)", value=False)
                    sg_samples = gr.Slider(4, 40, value=12, step=2, label="SmoothGrad samples")
                with gr.Row():
                    sg_noise = gr.Slider(0.01, 0.3, value=0.12, step=0.01, label="SmoothGrad noise level (std)")
                with gr.Row():
                    use_tile = gr.Checkbox(label="Generate Tile Probability Heatmap", value=True)
                with gr.Row():
                    tile_size = gr.Slider(64, 512, value=128, step=32, label="Tile size")
                    tile_stride = gr.Slider(32, 512, value=128, step=32, label="Tile stride")
                topk = gr.Slider(1, 12, value=6, step=1, label="Top-K suspicious tiles")
                with gr.Row():
                    analyze_btn = gr.Button("🔍 Analyze")
                    clear_btn = gr.Button("🧹 Clear")
                status_output = gr.Markdown("🟡 Status: Ready")
                gr.HTML('</div>')

        with gr.Column(scale=2):
            with gr.Group():
                gr.HTML('<div class="analysis-card">')
                gr.Markdown("### 📊 Diagnostic Report")
                diagnosis_output = gr.HTML(create_waiting_message())
                gr.Markdown("### 🌡️ Attention & Tile Visualizations")
                plot_output = gr.Plot(label="Grad-CAM + Overlays")
                gr.Markdown("### 🗺️ Tile Heatmap Overlay")
                tile_overlay_output = gr.Image(label="Tile Heatmap Overlay (RGB)", type="numpy")
                gr.Markdown("### 🔎 Top Suspicious Tiles")
                top_tiles_gallery = gr.Gallery(label="Top tiles (highest malignant probability)").style(grid=[3], height="auto")
                gr.HTML('</div>')

    # Event handlers
    def clear_all():
        return None, create_waiting_message(), None, None, [], "🟡 Status: Cleared"

    analyze_btn.click(
        fn=analyze_image_ui,
        inputs=[image_input, use_sg, sg_samples, sg_noise, use_tile, tile_size, tile_stride, topk],
        outputs=[diagnosis_output, plot_output, tile_overlay_output, top_tiles_gallery, status_output],
    )

    clear_btn.click(
        fn=clear_all,
        outputs=[image_input, diagnosis_output, plot_output, tile_overlay_output, top_tiles_gallery, status_output],
    )

# -----------------------------------------------------------------------------
# Launch
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("🚀 Starting application...")
    demo.launch(share=True)