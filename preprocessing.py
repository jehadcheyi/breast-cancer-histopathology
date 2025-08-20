import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os

from models import models, grad_model, classes, img_size, analysis_history

def prep_img(img):
    """
    Robust preprocessing: handle grayscale, single-channel, RGBA, different dtypes.
    Returns batch array shape (1, H, W, 3) float32 in [0,1].
    """
    if img is None:
        raise ValueError("No image input provided to prep_img()")

    arr = img.copy()

    # Convert possible float images to uint8 range if necessary (common in Gradio)
    if arr.dtype != np.uint8:
        # If in [0,1] floats, scale
        if arr.dtype == np.float32 or arr.dtype == np.float64:
            if np.max(arr) <= 1.0:
                arr = (arr * 255).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)

    # Handle grayscale (2D) images
    if len(arr.shape) == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    
    # Handle single-channel images
    elif len(arr.shape) == 3 and arr.shape[2] == 1:
        arr = np.concatenate([arr, arr, arr], axis=-1)
    
    # Handle RGBA images
    elif len(arr.shape) == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    
    # Ensure we have 3 channels
    if len(arr.shape) != 3 or arr.shape[2] != 3:
        raise ValueError(f"Invalid image shape after preprocessing: {arr.shape}")
    
    # Resize to model input size
    arr_resized = cv2.resize(arr, img_size)
    
    # Convert to float32 in [0,1] range
    arr_norm = arr_resized.astype(np.float32) / 255.0
    
    # Add batch dimension
    arr_batch = np.expand_dims(arr_norm, axis=0)
    
    return arr_batch

def analyze_image_properties(img):
    """
    Analyze image properties including color distribution, contrast, and texture metrics
    """
    # Convert to RGB if needed
    if len(img.shape) == 2:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:  # RGBA
        img_rgb = img[:, :, :3]
    else:
        img_rgb = img.copy()
    
    # Convert to different color spaces
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    
    # Calculate basic statistics
    stats = {
        'rgb_mean': np.mean(img_rgb, axis=(0, 1)).tolist(),
        'rgb_std': np.std(img_rgb, axis=(0, 1)).tolist(),
        'hsv_mean': np.mean(img_hsv, axis=(0, 1)).tolist(),
        'hsv_std': np.std(img_hsv, axis=(0, 1)).tolist(),
        'lab_mean': np.mean(img_lab, axis=(0, 1)).tolist(),
        'contrast': np.std(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)),
        'brightness': np.mean(img_hsv[:, :, 2]),
        'saturation': np.mean(img_hsv[:, :, 1]),
    }
    
    # Calculate texture metrics (GLCM)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    # Simple texture metrics
    stats['entropy'] = -np.sum(np.histogram(gray, bins=256, range=[0, 256])[0] / gray.size * 
                              np.log2(np.histogram(gray, bins=256, range=[0, 256])[0] / gray.size + 1e-10))
    
    # Edge detection for texture complexity
    edges = cv2.Canny(gray, 100, 200)
    stats['edge_density'] = np.sum(edges > 0) / edges.size
    
    return stats

def create_image_statistics_plot(img):
    """
    Create a comprehensive visualization of image statistics
    """
    # Get image properties
    stats = analyze_image_properties(img)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(12, 8))
    
    # Define grid layout
    gs = fig.add_gridspec(3, 3)
    
    # 1. RGB Histogram
    ax1 = fig.add_subplot(gs[0, 0])
    for i, c in enumerate(['r', 'g', 'b']):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        ax1.plot(hist, color=c)
    ax1.set_title('RGB Histogram')
    ax1.set_xlim([0, 256])
    ax1.grid(alpha=0.3)
    
    # 2. HSV Distribution
    ax2 = fig.add_subplot(gs[0, 1])
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    for i, c, l in zip([0, 1, 2], ['r', 'g', 'b'], ['Hue', 'Saturation', 'Value']):
        hist = cv2.calcHist([img_hsv], [i], None, [256], [0, 256])
        ax2.plot(hist, color=c, label=l)
    ax2.set_title('HSV Distribution')
    ax2.set_xlim([0, 256])
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 3. Texture Metrics
    ax3 = fig.add_subplot(gs[0, 2])
    metrics = {
        'Contrast': stats['contrast'],
        'Brightness': stats['brightness'],
        'Saturation': stats['saturation'],
        'Entropy': stats['entropy'],
        'Edge Density': stats['edge_density'] * 100
    }
    bars = ax3.bar(metrics.keys(), metrics.values(), color='skyblue')
    ax3.set_title('Texture Metrics')
    ax3.set_ylim([0, max(metrics.values()) * 1.2])
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Color Distribution
    ax4 = fig.add_subplot(gs[1, :])
    # Create color patches
    img_resized = cv2.resize(img, (100, 100))
    h, w = img_resized.shape[:2]
    color_sort = img_resized.reshape(-1, 3)
    color_sort = color_sort[np.argsort(np.sum(color_sort, axis=1))]
    color_sort = color_sort.reshape(h, w, 3)
    ax4.imshow(color_sort)
    ax4.set_title('Color Distribution (Sorted)')
    ax4.axis('off')
    
    # 5. Edge Detection
    ax5 = fig.add_subplot(gs[2, 0])
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    ax5.imshow(edges, cmap='gray')
    ax5.set_title('Edge Detection')
    ax5.axis('off')
    
    # 6. Grayscale Histogram
    ax6 = fig.add_subplot(gs[2, 1])
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    ax6.plot(hist, color='black')
    ax6.set_title('Grayscale Histogram')
    ax6.set_xlim([0, 256])
    ax6.grid(alpha=0.3)
    
    # 7. Color Channels
    ax7 = fig.add_subplot(gs[2, 2])
    channel_means = [np.mean(img[:,:,i]) for i in range(3)]
    channel_std = [np.std(img[:,:,i]) for i in range(3)]
    ax7.bar(['Red', 'Green', 'Blue'], channel_means, yerr=channel_std, 
            color=['red', 'green', 'blue'], alpha=0.7)
    ax7.set_title('RGB Channel Statistics')
    ax7.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig

def get_crop_box(shape, percent):
    """Calculate centered crop box coordinates"""
    h, w = shape[:2]
    crop_h = int(h * percent / 100)
    crop_w = int(w * percent / 100)
    left = (w - crop_w) // 2
    top = (h - crop_h) // 2
    return left, top, left + crop_w, top + crop_h

def get_analysis_statistics():
    """Generate statistics from analysis history"""
    if not analysis_history:
        return """
        <div style="background: linear-gradient(45deg, #3498db, #2980b9); color: white; padding: 25px; border-radius: 15px; margin: 20px 0;">
            <h2 style="color: white; margin: 0; text-align: center;">📊 ANALYSIS STATISTICS</h2>
            <div style="background: rgba(255,255,255,0.2); padding: 15px; border-radius: 10px; margin: 15px 0;">
                <p style="text-align: center; font-size: 18px;">No analysis data available yet.</p>
                <p style="text-align: center;">Upload and analyze images to generate statistics.</p>
            </div>
        </div>
        """
    
    # Count diagnoses
    total = len(analysis_history)
    benign_count = sum(1 for item in analysis_history if item['binary_label'] == 'benign')
    malignant_count = total - benign_count
    
    # Calculate percentages
    benign_percent = (benign_count / total) * 100 if total > 0 else 0
    malignant_percent = (malignant_count / total) * 100 if total > 0 else 0
    
    # Count subtypes
    subtype_counts = {}
    for binary_class in classes.keys():
        if binary_class != 'binary':
            subtype_counts[binary_class] = {subtype: 0 for subtype in classes[binary_class]}
    
    for item in analysis_history:
        binary_label = item['binary_label']
        subtype_label = item['subtype_label']
        if binary_label in subtype_counts and subtype_label in subtype_counts[binary_label]:
            subtype_counts[binary_label][subtype_label] += 1
    
    # Calculate average confidence
    avg_binary_conf = sum(item['binary_conf'] for item in analysis_history) / total if total > 0 else 0
    avg_subtype_conf = sum(item['subtype_conf'] for item in analysis_history) / total if total > 0 else 0
    
    # Calculate average uncertainty
    avg_binary_uncertainty = sum(item['binary_uncertainty'] for item in analysis_history) / total if total > 0 else 0
    avg_subtype_uncertainty = sum(item['subtype_uncertainty'] for item in analysis_history) / total if total > 0 else 0
    
    # Format statistics HTML
    stats_html = f"""
    <div style="background: linear-gradient(45deg, #3498db, #2980b9); color: white; padding: 25px; border-radius: 15px; margin: 20px 0;">
        <h2 style="color: white; margin: 0; text-align: center;">📊 ANALYSIS STATISTICS</h2>
        
        <div style="display: flex; justify-content: space-between; margin: 20px 0;">
            <div style="background: rgba(255,255,255,0.2); padding: 15px; border-radius: 10px; width: 48%;">
                <h3 style="margin-top: 0;">Summary</h3>
                <p>Total Analyses: <strong>{total}</strong></p>
                <p>Benign Diagnoses: <strong>{benign_count}</strong> ({benign_percent:.1f}%)</p>
                <p>Malignant Diagnoses: <strong>{malignant_count}</strong> ({malignant_percent:.1f}%)</p>
                <p>Average Binary Confidence: <strong>{avg_binary_conf:.1f}%</strong></p>
                <p>Average Subtype Confidence: <strong>{avg_subtype_conf:.1f}%</strong></p>
            </div>
            
            <div style="background: rgba(255,255,255,0.2); padding: 15px; border-radius: 10px; width: 48%;">
                <h3 style="margin-top: 0;">Uncertainty Metrics</h3>
                <p>Average Binary Uncertainty: <strong>{avg_binary_uncertainty:.1f}%</strong></p>
                <p>Average Subtype Uncertainty: <strong>{avg_subtype_uncertainty:.1f}%</strong></p>
                <p>Reliability Score: <strong>{100 - avg_binary_uncertainty:.1f}%</strong></p>
                <p>Last Analysis: <strong>{analysis_history[-1]['timestamp'] if analysis_history else 'N/A'}</strong></p>
            </div>
        </div>
        
        <div style="background: rgba(255,255,255,0.2); padding: 15px; border-radius: 10px; margin: 20px 0;">
            <h3 style="margin-top: 0;">Subtype Distribution</h3>
            <div style="display: flex; justify-content: space-between;">
                <div style="width: 48%;">
                    <h4>Benign Subtypes</h4>
                    <ul>
    """
    
    for subtype, count in subtype_counts['benign'].items():
        percent = (count / benign_count) * 100 if benign_count > 0 else 0
        stats_html += f"<li>{subtype}: <strong>{count}</strong> ({percent:.1f}%)</li>"
    
    stats_html += """
                    </ul>
                </div>
                <div style="width: 48%;">
                    <h4>Malignant Subtypes</h4>
                    <ul>
    """
    
    for subtype, count in subtype_counts['malignant'].items():
        percent = (count / malignant_count) * 100 if malignant_count > 0 else 0
        stats_html += f"<li>{subtype}: <strong>{count}</strong> ({percent:.1f}%)</li>"
    
    stats_html += """
                    </ul>
                </div>
            </div>
        </div>
    </div>
    """
    
    return stats_html

def get_sample_images():
    """Load sample images from the samples directory"""
    sample_images = []
    sample_dir = "samples"
    
    # Create samples directory if it doesn't exist
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
        print(f"Created samples directory: {sample_dir}")
        return []
    
    # Look for sample images
    for file in os.listdir(sample_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
            try:
                img_path = os.path.join(sample_dir, file)
                # Just store the path, we'll load the image when selected
                sample_images.append((file, img_path))
                print(f"Found sample image: {file}")
            except Exception as e:
                print(f"Error loading sample image {file}: {e}")
    return sample_images