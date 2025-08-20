import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
import json

from models import analysis_history, classes

# Custom CSS for the Gradio interface
custom_css = """
/* Global Styles */
.gradio-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
    min-height: 100vh;
}

/* Header Styling */
.main-header {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(15px);
    border-radius: 25px;
    padding: 40px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
    margin-bottom: 30px;
}

.main-header h1 {
    font-size: 2.5rem;
    background: linear-gradient(45deg, #4776E6, #8E54E9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 10px;
}

.main-header p {
    font-size: 1.2rem;
    color: #555;
    margin-bottom: 0;
}

/* Tab Styling */
.tabs {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 25px;
    padding: 20px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
}

/* Analysis Card */
.analysis-card {
    background: white;
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
    margin-bottom: 20px;
}

/* Button Styling */
button.primary {
    background: linear-gradient(45deg, #4776E6, #8E54E9) !important;
    border: none !important;
    border-radius: 50px !important;
    padding: 12px 25px !important;
    font-weight: bold !important;
    color: white !important;
    box-shadow: 0 5px 15px rgba(71, 118, 230, 0.3) !important;
    transition: all 0.3s ease !important;
}

button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(71, 118, 230, 0.4) !important;
}

button.secondary {
    background: linear-gradient(45deg, #e6e6e6, #f5f5f5) !important;
    border: none !important;
    border-radius: 50px !important;
    padding: 12px 25px !important;
    font-weight: bold !important;
    color: #555 !important;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05) !important;
    transition: all 0.3s ease !important;
}

button.secondary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1) !important;
}

/* Results Styling */
.results-panel {
    background: white;
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
}

.diagnosis-result {
    background: linear-gradient(45deg, #4776E6, #8E54E9);
    color: white;
    border-radius: 15px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 5px 15px rgba(71, 118, 230, 0.3);
}

.diagnosis-result h2 {
    margin-top: 0;
    font-size: 1.8rem;
}

/* Plot Styling */
.plot-container {
    background: white;
    border-radius: 15px;
    padding: 15px;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
    margin-bottom: 20px;
}

/* Status Message */
.status-message {
    background: rgba(255, 255, 255, 0.8);
    border-radius: 10px;
    padding: 10px 15px;
    margin-top: 20px;
    font-style: italic;
    color: #555;
}

/* Gallery */
.gallery-item {
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
    transition: all 0.3s ease;
}

.gallery-item:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
}

/* Statistics Tab */
.statistics-container {
    background: white;
    border-radius: 15px;
    padding: 25px;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
}

/* Responsive Adjustments */
@media (max-width: 768px) {
    .main-header {
        padding: 20px;
    }
    
    .main-header h1 {
        font-size: 2rem;
    }
    
    .tabs {
        padding: 15px;
    }
}
"""

def analyze_image(input_img):
    if input_img is None:
        return """
        <div style="background: linear-gradient(45deg, #e74c3c, #c0392b); color: white; padding: 25px; border-radius: 15px; margin: 20px 0;">
            <h2 style="color: white; margin: 0; text-align: center;">❌ ERROR</h2>
            <div style="background: rgba(255,255,255,0.2); padding: 15px; border-radius: 10px; margin: 15px 0;">
                <p style="text-align: center; font-size: 18px;">No image provided!</p>
                <p style="text-align: center;">Please upload an image or select from the gallery.</p>
            </div>
        </div>
        """, None, None, None, None, None, "🔴 **Status**: No image provided"
    
    try:
        # Import here to avoid circular imports
        from preprocessing import prep_img, create_image_statistics_plot
        from models import predict_with_uncertainty
        from visualization import (
            create_confidence_radar_chart, 
            create_uncertainty_visualization,
            create_risk_assessment_gauge,
            create_enhanced_visualization
        )
        
        # Process the image
        img_array = np.array(input_img)
        
        # Make prediction with uncertainty
        results = predict_with_uncertainty(img_array)
        
        # Extract results
        binary_class = results['binary_class']
        binary_label = results['binary_label']
        binary_conf = results['binary_conf']
        binary_uncertainty = results['binary_uncertainty']
        subtype_class = results['subtype_class']
        subtype_label = results['subtype_label']
        subtype_conf = results['subtype_conf']
        subtype_uncertainty = results['subtype_uncertainty']
        subtype_probs = results['subtype_probs']
        
        # Store in analysis history
        analysis_history.append({
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'binary_class': binary_class,
            'binary_label': binary_label,
            'binary_conf': binary_conf,
            'binary_uncertainty': binary_uncertainty,
            'subtype_class': subtype_class,
            'subtype_label': subtype_label,
            'subtype_conf': subtype_conf,
            'subtype_uncertainty': subtype_uncertainty
        })
        
        # Create diagnosis HTML
        if binary_label == 'benign':
            color_gradient = "linear-gradient(45deg, #2ecc71, #27ae60)"
            emoji = "✅"
        else:
            color_gradient = "linear-gradient(45deg, #e74c3c, #c0392b)"
            emoji = "⚠️"
        
        diagnosis_html = f"""
        <div style="background: {color_gradient}; color: white; padding: 25px; border-radius: 15px; margin: 20px 0;">
            <h2 style="color: white; margin: 0; text-align: center;">{emoji} {binary_label.upper()} ({binary_conf:.1f}%)</h2>
            <div style="background: rgba(255,255,255,0.2); padding: 15px; border-radius: 10px; margin: 15px 0;">
                <p style="text-align: center; font-size: 18px;">Subtype: <strong>{subtype_label}</strong> ({subtype_conf:.1f}%)</p>
                <p style="text-align: center;">Uncertainty: ±{binary_uncertainty:.1f}%</p>
            </div>
            <div style="display: flex; justify-content: space-between; margin-top: 15px;">
                <div style="background: rgba(255,255,255,0.2); padding: 10px; border-radius: 10px; width: 48%; text-align: center;">
                    <strong>Confidence Score</strong><br>{binary_conf:.1f}%
                </div>
                <div style="background: rgba(255,255,255,0.2); padding: 10px; border-radius: 10px; width: 48%; text-align: center;">
                    <strong>Certainty Level</strong><br>{100-binary_uncertainty:.1f}%
                </div>
            </div>
        </div>
        """
        
        # Create enhanced visualization
        enhanced_plot = create_enhanced_visualization(
            img_array, binary_class, binary_conf, binary_label, subtype_conf
        )
        
        # Create image statistics plot
        stats_plot = create_image_statistics_plot(img_array)
        
        # Create confidence radar chart
        radar_plot = create_confidence_radar_chart(
            binary_conf, binary_uncertainty, subtype_probs
        )
        
        # Create uncertainty visualization
        uncertainty_plot = create_uncertainty_visualization(subtype_probs)
        
        # Create risk assessment gauge
        risk_gauge = create_risk_assessment_gauge(binary_conf, binary_label)
        
        # Return all results
        return (
            diagnosis_html,
            enhanced_plot,
            stats_plot,
            radar_plot,
            uncertainty_plot,
            risk_gauge,
            f"🟢 **Status**: Analysis completed successfully - {binary_label.upper()} detected"
        )
    except Exception as e:
        print(f"Enhanced analyze error: {e}")
        error_html = f"""
        <div style="background: linear-gradient(45deg, #e74c3c, #c0392b); color: white; padding: 25px; border-radius: 15px; margin: 20px 0;">
            <h2 style="color: white; margin: 0; text-align: center;">❌ ANALYSIS ERROR</h2>
            <div style="background: rgba(255,255,255,0.2); padding: 15px; border-radius: 10px; margin: 15px 0;">
                <strong>Error Details:</strong> {str(e)}
            </div>
            <div style="text-align: center; margin-top: 20px;">
                <h4>🔧 Troubleshooting Steps:</h4>
                <ul style="text-align: left; max-width: 400px; margin: 0 auto;">
                    <li>Ensure image is a valid histopathology scan</li>
                    <li>Check file format (PNG, JPG, JPEG, TIF supported)</li>
                    <li>Try uploading a different image</li>
                    <li>Contact technical support if issue persists</li>
                </ul>
            </div>
        </div>
        """
        return error_html, None, None, None, None, None, "🔴 **Status**: Analysis Failed - System Error"