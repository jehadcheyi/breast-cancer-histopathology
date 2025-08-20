import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import tensorflow as tf

from models import models, grad_model, classes, img_size
from preprocessing import get_crop_box

def make_enhanced_heatmap(img, pred_idx, layer_names=None):
    """Generate enhanced Grad-CAM heatmaps from multiple layers"""
    resized = cv2.resize(img, img_size)
    img_in = resized.astype(np.float32) / 255.0
    img_in = np.expand_dims(img_in, axis=0)
    
    if layer_names is None:
        # Find multiple conv layers
        conv_layers = []
        for layer in reversed(models['binary'].layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                conv_layers.append(layer.name)
                if len(conv_layers) >= 3:  # Get top 3 conv layers
                    break
        layer_names = conv_layers
    
    heatmaps = {}
    
    for layer_name in layer_names:
        try:
            # Create Grad-CAM model for this layer
            grad_model = tf.keras.models.Model(
                inputs=models['binary'].inputs,
                outputs=[
                    models['binary'].get_layer(layer_name).output,
                    models['binary'].output
                ]
            )
            
            # Compute gradients
            with tf.GradientTape() as tape:
                conv_output, predictions = grad_model(img_in)
                loss = predictions[:, pred_idx]
            
            # Extract gradients
            grads = tape.gradient(loss, conv_output)
            
            # Global average pooling
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            # Weight the channels by gradients
            conv_output = conv_output[0]
            for i in range(pooled_grads.shape[0]):
                conv_output[:, :, i] *= pooled_grads[i]
            
            # Average over channels
            heatmap = tf.reduce_mean(conv_output, axis=-1)
            
            # Normalize heatmap
            heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-10)
            heatmap = np.uint8(255 * heatmap)
            
            # Resize to original image size
            heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
            
            # Store in dictionary
            heatmaps[layer_name] = heatmap
        except Exception as e:
            print(f"Error generating heatmap for layer {layer_name}: {e}")
            continue
    
    return heatmaps

def create_confidence_radar_chart(binary_conf, binary_uncertainty, subtype_probs):
    """Create an interactive radar chart for confidence visualization"""
    
    # Prepare data for radar chart
    categories = ['Binary Confidence', 'Certainty Score', 'Model Stability']
    values = [
        binary_conf,
        100 - binary_uncertainty,
        min(100, binary_conf - binary_uncertainty)
    ]
    
    # Add subtype confidences
    for name, prob_data in list(subtype_probs.items())[:3]:  # Top 3 subtypes
        categories.append(f"{name[:15]}...")
        values.append(prob_data['prob'])
    
    # Create radar chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(67, 160, 71, 0.5)',
        line=dict(color='rgb(67, 160, 71)', width=2),
        name='Confidence Metrics'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=False,
        title="Confidence Analysis",
        height=400,
        margin=dict(l=80, r=80, t=40, b=40)
    )
    
    return fig

def create_uncertainty_visualization(subtype_probs):
    """Create an interactive visualization of prediction uncertainty"""
    
    # Prepare data
    subtypes = list(subtype_probs.keys())
    probs = [subtype_probs[s]['prob'] for s in subtypes]
    uncertainties = [subtype_probs[s]['uncertainty'] for s in subtypes]
    
    # Create figure
    fig = go.Figure()
    
    # Add bar chart for probabilities
    fig.add_trace(go.Bar(
        x=subtypes,
        y=probs,
        name='Probability',
        marker_color='rgb(55, 83, 109)',
        error_y=dict(
            type='data',
            array=uncertainties,
            visible=True,
            color='red'
        )
    ))
    
    # Update layout
    fig.update_layout(
        title="Prediction Uncertainty",
        xaxis_title="Subtype",
        yaxis_title="Probability (%)",
        yaxis=dict(range=[0, 100]),
        height=400,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    return fig

def create_risk_assessment_gauge(binary_conf, diagnosis):
    """Create a risk assessment gauge visualization"""
    
    # Determine risk level based on diagnosis and confidence
    if diagnosis == 'malignant':
        risk_value = min(100, binary_conf + 20)  # Higher risk for malignant
        risk_color = 'red'
        risk_text = 'High Risk'
    else:
        risk_value = max(0, 100 - binary_conf)  # Lower risk for benign
        risk_color = 'green'
        risk_text = 'Low Risk'
    
    # Create gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_value,
        domain=dict(x=[0, 1], y=[0, 1]),
        title=dict(text=f"Risk Assessment: {risk_text}", font=dict(size=16)),
        delta=dict(reference=50),
        gauge=dict(
            axis=dict(range=[0, 100]),
            bar=dict(color=risk_color),
            steps=[
                dict(range=[0, 30], color="rgba(0, 200, 0, 0.3)"),
                dict(range=[30, 70], color="rgba(255, 165, 0, 0.3)"),
                dict(range=[70, 100], color="rgba(255, 0, 0, 0.3)")
            ],
            threshold=dict(
                line=dict(color="black", width=4),
                thickness=0.75,
                value=risk_value
            )
        )
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def create_enhanced_visualization(img, pred_idx, binary_conf, label, subtype_conf, zooms=[75, 50, 25]):
    """Create an enhanced visualization with multiple zoom levels and heatmaps"""
    try:
        # Convert image to BGR for OpenCV operations
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if len(img.shape) == 3 and img.shape[2] == 3 else img
        
        # Get image dimensions
        h, w = img_bgr.shape[:2]
        
        # Generate heatmaps from multiple layers
        heatmaps = make_enhanced_heatmap(img, pred_idx)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 12))
        gs = fig.add_gridspec(3, 4)
        
        # 1. Original Image
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        ax1.set_title("Original Image", fontsize=12)
        ax1.axis('off')
        
        # 2. Main Heatmap Overlay
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Get the first heatmap (usually the most important layer)
        if heatmaps:
            main_heatmap = list(heatmaps.values())[0]
            
            # Apply colormap
            heatmap_colored = cv2.applyColorMap(main_heatmap, cv2.COLORMAP_JET)
            
            # Overlay heatmap on original image
            overlay = cv2.addWeighted(
                img_bgr, 0.7,
                heatmap_colored, 0.3,
                0
            )
            
            ax2.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            ax2.set_title(f"Activation Heatmap\n{label} ({binary_conf:.1f}%)", fontsize=12)
        else:
            ax2.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            ax2.set_title(f"Heatmap Unavailable\n{label} ({binary_conf:.1f}%)", fontsize=12)
        ax2.axis('off')
        
        # 3. Multiple Zoom Levels with Heatmaps
        for i, zoom in enumerate(zooms):
            ax = fig.add_subplot(gs[0, 2 + i] if i < 2 else gs[1, i - 2])
            
            # Calculate crop box
            x1, y1, x2, y2 = get_crop_box(img_bgr.shape, zoom)
            
            # Crop image
            cropped = img_bgr[y1:y2, x1:x2]
            
            # If we have heatmaps, overlay them
            if heatmaps:
                # Get a different heatmap for each zoom if available
                heatmap_idx = min(i, len(heatmaps) - 1)
                heatmap = list(heatmaps.values())[heatmap_idx]
                
                # Crop heatmap
                heatmap_cropped = heatmap[y1:y2, x1:x2]
                
                # Apply colormap
                heatmap_colored = cv2.applyColorMap(heatmap_cropped, cv2.COLORMAP_JET)
                
                # Overlay heatmap on cropped image
                overlay = cv2.addWeighted(
                    cropped, 0.7,
                    heatmap_colored, 0.3,
                    0
                )
                
                ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            else:
                ax.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
            
            ax.set_title(f"{zoom}% Zoom", fontsize=12)
            ax.axis('off')
        
        # 4. Additional Visualizations
        # 4.1 Edge Detection
        ax_edge = fig.add_subplot(gs[1, 2])
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        ax_edge.imshow(edges, cmap='gray')
        ax_edge.set_title("Edge Detection", fontsize=12)
        ax_edge.axis('off')
        
        # 4.2 Color Channels
        ax_color = fig.add_subplot(gs[1, 3])
        # Split into RGB channels
        r, g, b = cv2.split(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        # Create composite image
        zeros = np.zeros_like(r)
        r_channel = cv2.merge([r, zeros, zeros])
        g_channel = cv2.merge([zeros, g, zeros])
        b_channel = cv2.merge([zeros, zeros, b])
        
        # Stack channels horizontally
        color_channels = np.hstack([r_channel, g_channel, b_channel])
        ax_color.imshow(color_channels)
        ax_color.set_title("RGB Channels", fontsize=12)
        ax_color.axis('off')
        
        # 5. Diagnosis Summary
        ax_summary = fig.add_subplot(gs[2, :])
        ax_summary.axis('off')
        
        # Create summary text
        summary_text = f"""
        DIAGNOSIS SUMMARY:
        
        Primary Classification: {label.upper()} (Confidence: {binary_conf:.1f}%)
        Subtype Classification: {subtype_conf:.1f}% confidence
        
        ANALYSIS NOTES:
        - AI analysis completed successfully
        - Multiple visualization layers generated
        - Heatmap shows regions of interest for diagnosis
        - Edge detection highlights tissue boundaries
        - Color channel analysis shows distribution of RGB components
        """
        
        ax_summary.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=12,
                       bbox=dict(facecolor='lightblue', alpha=0.5, boxstyle='round,pad=1'))
        
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"Enhanced visualization error: {e}")
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.text(0.5, 0.5, f"Visualization Error\n{str(e)}", ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
        return fig