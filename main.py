import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import gradio as gr
import warnings
import pandas as pd
from datetime import datetime

# Import from our modules
from models import load_all_models, models, analysis_history, classes
from preprocessing import prep_img, get_sample_images, get_analysis_statistics
from visualization import (
    create_confidence_radar_chart,
    create_uncertainty_visualization,
    create_risk_assessment_gauge,
    create_enhanced_visualization
)
from utils import analyze_image, custom_css

# Suppress noisy warnings (keeps output cleaner)
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.get_logger().setLevel('ERROR')

# Load models at startup
models, grad_model = load_all_models()

# Get sample images
sample_images = get_sample_images()

# ---------- Gradio Interface ----------
with gr.Blocks(css=custom_css, theme=gr.themes.Soft(), title="BREAST AI PRO - Enhanced Cancer Detection") as demo:
    # Header
    with gr.Row():
        with gr.Column():
            gr.HTML(f"""
            <div class="main-header">
                <h1>🔬 BREAST AI PRO</h1>
                <p>Enhanced Histopathology Analysis System with Uncertainty Quantification</p>
                <p style="font-size: 0.9rem; color: #666;">Version 2.0 | Advanced Diagnostic AI</p>
            </div>
            """)
    
    # Tabs
    with gr.Tabs() as tabs:
        # Main Analysis Tab
        with gr.TabItem("🔬 AI Analysis", elem_id="main-analysis"):
            with gr.Row(equal_height=True):
                # Left Column - Input
                with gr.Column(scale=1):
                    # Input Card
                    with gr.Group():
                        gr.HTML('<div class="analysis-card">')
                        gr.Markdown("### 📤 **Enhanced AI Analysis Center**")
                        gr.Markdown("*Upload histopathology images for comprehensive AI analysis with uncertainty quantification*")
                        
                        image_input = gr.Image(
                            label="Upload Histopathology Image",
                            type="numpy",
                            tool="editor",
                            height=400,
                            shape=(500, 500),
                            sources=["upload", "clipboard", "webcam", "select"]
                        )
                        
                        gr.Markdown("""
                        **Supported formats:** PNG, JPG, JPEG, TIF
                        **Recommended size:** 500x500 pixels or larger
                        """)
                        
                        with gr.Row():
                            submit_btn = gr.Button(
                                "🔍 Analyze Image", 
                                variant="primary",
                                elem_classes=["primary"]
                            )
                            
                            clear_btn = gr.Button(
                                "🗑️ Clear", 
                                variant="secondary",
                                elem_classes=["secondary"]
                            )
                        
                        # Status output
                        status_output = gr.Markdown(
                            "⚪ **Status**: Ready - Upload an image to begin analysis",
                            elem_classes=["status-message"]
                        )
                        gr.HTML('</div>')
                    
                    # Sample Gallery
                    if sample_images:
                        with gr.Group():
                            gr.HTML('<div class="analysis-card">')
                            gr.Markdown("### 🖼️ **Sample Images**")
                            gr.Markdown("*Click on any sample image to analyze*")
                            
                            example_gallery = gr.Gallery(
                                value=[(path, name) for name, path in sample_images],
                                columns=4,
                                object_fit="contain",
                                height="auto",
                                elem_classes=["gallery-item"]
                            )
                            gr.HTML('</div>')
                
                # Right Column - Results
                with gr.Column(scale=2):
                    # Results Card
                    with gr.Group():
                        gr.HTML('<div class="results-panel">')
                        gr.Markdown("### 📊 **Analysis Results**")
                        
                        # Diagnosis Output
                        diagnosis_output = gr.HTML()
                        
                        # Visualization Tabs
                        with gr.Tabs():
                            with gr.TabItem("Enhanced Visualization"):
                                enhanced_plot_output = gr.Plot()
                            
                            with gr.TabItem("Image Statistics"):
                                stats_plot_output = gr.Plot()
                            
                            with gr.TabItem("Confidence Analysis"):
                                radar_plot_output = gr.Plot()
                            
                            with gr.TabItem("Uncertainty Metrics"):
                                uncertainty_plot_output = gr.Plot()
                            
                            with gr.TabItem("Risk Assessment"):
                                risk_gauge_output = gr.Plot()
                        gr.HTML('</div>')
        
        # Statistics Tab
        with gr.TabItem("📈 Analysis Statistics", elem_id="statistics"):
            with gr.Group():
                gr.HTML('<div class="statistics-container">')
                gr.Markdown("### 📊 **Comprehensive Analysis Statistics**")
                gr.Markdown("*View statistics from all analyses performed in this session*")
                
                statistics_output = gr.HTML()
                
                stats_btn = gr.Button(
                    "🔄 Refresh Statistics", 
                    variant="primary",
                    elem_classes=["primary"]
                )
                gr.HTML('</div>')
    
    # ---------- Event Handlers ----------
    def handle_submit(image):
        if image is None:
            return (
                """<div style="text-align: center; padding: 20px; background-color: #f8d7da; border-radius: 10px;">
                <h3 style="color: #721c24;">No image provided!</h3>
                <p>Please upload an image or select from the gallery.</p>
                </div>""",
                None, None, None, None, None,
                "🔴 **Status**: No image provided"
            )
        
        return analyze_image(image)
    
    def handle_clear():
        return (
            None,
            """<div style="text-align: center; padding: 20px;">
            <p>Upload an image to begin analysis</p>
            </div>""",
            None, None, None, None, None,
            "⚪ **Status**: Ready - Upload an image to begin analysis"
        )
    
    def handle_gallery_select(evt: gr.SelectData):
        return sample_images[evt.index][1]
    
    def handle_statistics():
        return get_analysis_statistics()
    
    # ---------- Event Bindings ----------
    submit_btn.click(
        fn=handle_submit,
        inputs=[image_input],
        outputs=[
            diagnosis_output,
            enhanced_plot_output,
            stats_plot_output,
            radar_plot_output,
            uncertainty_plot_output,
            risk_gauge_output,
            status_output
        ]
    )

    clear_btn.click(
        fn=handle_clear,
        outputs=[
            image_input,
            diagnosis_output,
            enhanced_plot_output,
            stats_plot_output,
            radar_plot_output,
            uncertainty_plot_output,
            risk_gauge_output,
            status_output
        ]
    )

    if sample_images:
        example_gallery.select(
            fn=handle_gallery_select,
            outputs=[image_input]
        )

    stats_btn.click(
        fn=handle_statistics,
        outputs=[statistics_output]
    )

# ---------- Launch Configuration ----------
if __name__ == "__main__":
    print("🚀 Launching Enhanced Histopathology Analysis System Pro...")
    print("🔬 Features: Uncertainty Quantification, Multi-layer Visualization, Interactive Analytics")
    print("📊 New: Statistical Dashboards, Risk Assessment, Comprehensive Analysis")
    
    # Launch with enhanced configuration
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
        max_threads=10,
        favicon_path=None,
        ssl_verify=False,
        quiet=False,
        show_api=True
    )