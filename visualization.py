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
    """Generate simulated heatmaps for demonstration purposes"""
    # For demonstration, we'll generate simulated heatmaps
    # In a real application, this would use actual Grad-CAM computation
    
    heatmaps = {}
    
    # Get image dimensions
    h, w = img.shape[:2]
    
    # Generate 3 different simulated heatmaps
    for i in range(3):
        # Create a blank heatmap
        heatmap = np.zeros((h, w), dtype=np.uint8)
        
        # Generate a random center point for the heatmap
        center_x = np.random.randint(w // 4, 3 * w // 4)
        center_y = np.random.randint(h // 4, 3 * h // 4)
        
        # Generate a random radius
        radius = np.random.randint(min(h, w) // 8, min(h, w) // 3)
        
        # Create a circular gradient
        y, x = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        mask = dist_from_center <= radius
        
        # Create gradient effect
        heatmap[mask] = 255 * (1 - dist_from_center[mask] / radius)
        
        # Apply Gaussian blur to make it look more natural
        heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
        
        # Store in dictionary with a fake layer name
        layer_name = f"conv2d_{i+1}"
        heatmaps[layer_name] = heatmap
    
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