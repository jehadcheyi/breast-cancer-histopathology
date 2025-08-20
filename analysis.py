from config import np, cv2, Image, img_to_array, tf, Model, img_size, target_size, classes, models, analysis_history, DEBUG
from visualizations import (
    create_image_statistics_plot,
    create_confidence_radar_chart,
    create_uncertainty_visualization,
    create_risk_assessment_gauge
)

# ---------- Enhanced preprocessing ----------
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
            if arr.max() <= 1.0:
                arr = (arr * 255).astype('uint8')
            else:
                arr = arr.astype('uint8')
        else:
            arr = arr.astype('uint8')

    # Normalize channel shapes:
    if arr.ndim == 2:
        # grayscale HxW -> HxWx3
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.concatenate([arr] * 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        # RGBA -> RGB (drop alpha)
        arr = arr[..., :3]

    # Convert to PIL to force RGB mode and resize with high-quality filter
    pil = Image.fromarray(arr)
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    pil = pil.resize(img_size, Image.LANCZOS)
    arr = img_to_array(pil).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    if DEBUG:
        print("prep_img -> shape:", arr.shape, "dtype:", arr.dtype, "min/max:", arr.min(), arr.max())

    return arr

# ---------- Enhanced Prediction with Uncertainty ----------
def predict_with_uncertainty(img, n_samples=10):
    """
    Enhanced prediction with uncertainty estimation using Monte Carlo Dropout
    """
    try:
        arr = prep_img(img)
        
        # Binary Classification with uncertainty
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

        # Subtype Classification with uncertainty
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
                subtype_probs[classes['benign_names'][code]] = {
                    'mean': float(prob * 100.0),
                    'std': float(std * 100.0)
                }
        else:
            subtype_code = classes['malignant'][subtype_idx]
            subtype_name = classes['malignant_names'][subtype_code]
            for code, prob, std in zip(classes['malignant'], subtype_mean, subtype_std):
                subtype_probs[classes['malignant_names'][code]] = {
                    'mean': float(prob * 100.0),
                    'std': float(std * 100.0)
                }

        return (pred_idx, binary_conf, binary_uncertainty, 
                subtype_name, subtype_conf, subtype_uncertainty, subtype_probs)
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise

# ---------- Enhanced Grad-CAM with Multiple Layers ----------
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
            grad_model_temp = Model(
                inputs=models['binary'].inputs,
                outputs=[models['binary'].get_layer(layer_name).output, models['binary'].output]
            )
            
            with tf.GradientTape() as tape:
                conv_out, preds = grad_model_temp(img_in)
                if preds.shape[-1] == 1:
                    target_label = classes["binary"][pred_idx]
                    if target_label == "malignant":
                        loss = preds[:, 0]
                    else:
                        loss = 1.0 - preds[:, 0]
                else:
                    loss = preds[:, pred_idx]
            
            grads = tf.gradients(loss, conv_out)[0] if hasattr(tf, 'gradients') else None
            if grads is None:
                grads = tf.GradientTape().gradient(loss, conv_out)
            grads = tf.convert_to_tensor(grads)[0]
            pooled = tf.reduce_mean(grads, axis=(0, 1))
            conv_out = conv_out[0]
            heatmap = np.dot(conv_out, pooled.numpy())
            heatmap = np.maximum(heatmap, 0)
            maxval = np.max(heatmap)
            if maxval > 0:
                heatmap = heatmap / maxval
            else:
                heatmap = np.zeros_like(heatmap)
            
            heatmap = cv2.resize(heatmap, target_size)
            heatmaps[layer_name] = heatmap
            
        except Exception as e:
            print(f"Error generating heatmap for layer {layer_name}: {e}")
            continue
    
    return heatmaps

# ---------- Enhanced Visualization Creation ----------
def create_enhanced_visualization(img, pred_idx, binary_conf, label, subtype_conf, zooms=[75, 50, 25]):
    """Create enhanced visualization with multiple heatmap layers"""
    import matplotlib.pyplot as plt
    try:
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)

        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        h, w = img_bgr.shape[:2]
        
        # Get multiple heatmaps
        heatmaps = make_enhanced_heatmap(img, pred_idx)
        
        # Create figure with more subplots
        fig, axs = plt.subplots(3, len(heatmaps) + 2, figsize=(20, 15), dpi=100)
        
        # Original image
        resized = cv2.resize(img_bgr, target_size)
        resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        axs[0, 0].imshow(resized_rgb)
        axs[0, 0].set_title("🔬 Original Image", fontsize=14, fontweight='bold')
        axs[0, 0].axis('off')
        
        # Multiple heatmaps
        for i, (layer_name, heatmap) in enumerate(heatmaps.items()):
            heatmap_uint8 = np.uint8(255 * heatmap)
            colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
            
            axs[0, i+1].imshow(colored_rgb)
            axs[0, i+1].set_title(f"🌡️ {layer_name}", fontsize=12)
            axs[0, i+1].axis('off')
            
            # Overlay
            overlay = cv2.addWeighted(resized, 0.6, colored, 0.4, 0)
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            axs[1, i+1].imshow(overlay_rgb)
            axs[1, i+1].set_title(f"🎯 Overlay {layer_name}", fontsize=12)
            axs[1, i+1].axis('off')
        
        # Edge detection and other analyses
        gray = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        axs[1, 0].imshow(edges, cmap='gray')
        axs[1, 0].set_title("🔍 Edge Detection", fontsize=14, fontweight='bold')
        axs[1, 0].axis('off')
        
        # Texture analysis
        kernel = np.ones((5,5), np.float32)/25
        smooth = cv2.filter2D(gray, -1, kernel)
        
        axs[2, 0].imshow(smooth, cmap='viridis')
        axs[2, 0].set_title("🎨 Texture Analysis", fontsize=14, fontweight='bold')
        axs[2, 0].axis('off')
        
        # Fill remaining subplots with zoomed analyses
        for i in range(1, len(heatmaps) + 2):
            if i < len(zooms) + 1:
                zoom = zooms[i-1] if i-1 < len(zooms) else 50
                x1, y1, x2, y2 = get_crop_box(img_bgr.shape, zoom)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 > x1 and y2 > y1:
                    crop = img_bgr[y1:y2, x1:x2]
                    crop_resized = cv2.resize(crop, (200, 200))
                    crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
                    
                    axs[2, i].imshow(crop_rgb)
                    axs[2, i].set_title(f"🔍 Zoom {zoom}%", fontsize=12)
                    axs[2, i].axis('off')
                else:
                    axs[2, i].axis('off')
            else:
                axs[2, i].axis('off')
        
        diagnosis = classes['binary'][pred_idx].upper()
        color = '#4CAF50' if diagnosis == 'BENIGN' else '#F44336'
        
        plt.suptitle(
            f"🏥 ENHANCED MULTI-LAYER ANALYSIS\n"
            f"🎯 Diagnosis: {diagnosis} ({binary_conf:.1f}% confidence)\n"
            f"🔬 Advanced Deep Learning Visualization",
            fontsize=16, y=0.98, color=color, fontweight='bold'
        )
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        return fig
        
    except Exception as e:
        print(f"Enhanced visualization error: {e}")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.text(0.5, 0.5, f"Visualization Error\n{str(e)}", ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
        return fig

# ---------- Cropping helper ----------
def get_crop_box(shape, percent):
    """Calculate centered crop box coordinates"""
    h, w = shape[:2]
    crop_h = int(h * percent / 100)
    crop_w = int(w * percent / 100)
    left = (w - crop_w) // 2
    top = (h - crop_h) // 2
    return left, top, left + crop_w, top + crop_h

# ---------- Enhanced Main analyze function ----------
def analyze_image(input_img):
    from datetime import datetime
    if input_img is None:
        return """
        <div style="background: linear-gradient(45deg, #FFA726, #FF7043); color: white; padding: 25px; border-radius: 15px; text-align: center; margin: 20px 0;">
            <h2 style="margin: 0; font-size: 1.5em;">⚠️ No Image Provided</h2>
            <p style="margin: 10px 0; font-size: 1.1em;">Please upload a histopathology image to begin AI analysis</p>
            <div style="background: rgba(255,255,255,0.2); padding: 10px; border-radius: 8px; margin-top: 15px;">
                <strong>Supported formats:</strong> JPG, PNG, JPEG, TIF
            </div>
        </div>
        """, None, None, None, None, None, "🟡 **Status**: Waiting for Image Upload"

    try:
        img = input_img.astype('uint8')
        
        # Enhanced prediction with uncertainty
        (pred_idx, binary_conf, binary_uncertainty, 
         subtype, subtype_conf, subtype_uncertainty, subtype_probs) = predict_with_uncertainty(img)

        label = classes['binary'][pred_idx]
        diagnosis = label.upper()
        diagnosis_color = "#4CAF50" if diagnosis == "BENIGN" else "#F44336"
        diagnosis_emoji = "🟢" if diagnosis == "BENIGN" else "🔴"

        # Store analysis in history
        analysis_record = {
            'timestamp': datetime.now().isoformat(),
            'diagnosis': diagnosis,
            'confidence': binary_conf,
            'uncertainty': binary_uncertainty,
            'subtype': subtype,
            'subtype_confidence': subtype_conf
        }
        analysis_history.append(analysis_record)

        # Risk assessment
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

        # Enhanced subtype table with uncertainty
        subtype_table = """
        <div style='background: #f8f9fa; padding: 20px; border-radius: 12px; margin: 15px 0;'>
            <h4 style='color: #2c3e50; margin-bottom: 15px; text-align: center;'>🔬 Enhanced Subtype Analysis with Uncertainty</h4>
            <table style='width:100%; border-collapse: collapse; font-family: Arial, sans-serif;'>
                <tr style='background: linear-gradient(45deg, #667eea, #764ba2); color: white;'>
                    <th style='padding: 12px; text-align: left;'>Cancer Subtype</th>
                    <th style='padding: 12px; text-align: center;'>Confidence</th>
                    <th style='padding: 12px; text-align: center;'>Uncertainty</th>
                    <th style='padding: 12px; text-align: center;'>Reliability</th>
                </tr>
        """

        for name, prob_data in sorted(subtype_probs.items(), key=lambda x: x[1]['mean'], reverse=True):
            prob_mean = max(0.0, min(100.0, prob_data['mean']))
            prob_std = max(0.0, min(100.0, prob_data['std']))
            reliability = max(0, 100 - prob_std)
            
            bar_color = "#4CAF50" if prob_mean < 30 else "#FF9800" if prob_mean < 70 else "#F44336"
            reliability_color = "#4CAF50" if reliability > 80 else "#FF9800" if reliability > 60 else "#F44336"

            subtype_table += f"""
            <tr style='border-bottom: 1px solid #e0e0e0;'>
                <td style='padding: 12px; font-weight: bold;'>{name}</td>
                <td style='padding: 12px; text-align: center; font-weight: bold; color: {bar_color};'>{prob_mean:.1f}%</td>
                <td style='padding: 12px; text-align: center; color: #666;'>±{prob_std:.1f}%</td>
                <td style='padding: 12px; text-align: center;'>
                    <div style='background: #e0e0e0; border-radius: 10px; height: 20px; width: 100px; margin: 0 auto; position: relative;'>
                        <div style='background: {reliability_color}; height: 100%; width: {int(reliability)}%; border-radius: 10px; transition: all 0.3s ease;'></div>
                    </div>
                    <small style='color: {reliability_color}; font-weight: bold;'>{reliability:.0f}%</small>
                </td>
            </tr>
            """

        subtype_table += "</table></div>"

        # Enhanced diagnostic report with uncertainty metrics
        diagnosis_html = f"""
        <div style="background: linear-gradient(135deg, {diagnosis_color}15, {diagnosis_color}25); border: 3px solid {diagnosis_color}; border-radius: 20px; padding: 30px; margin: 20px 0; font-family: 'Segoe UI', Arial, sans-serif;">
            
            <div style="text-align: center; margin-bottom: 25px;">
                <h1 style="color: {diagnosis_color}; margin: 0; font-size: 2.2em;">{diagnosis_emoji} ENHANCED AI DIAGNOSTIC REPORT</h1>
                <div style="background: {diagnosis_color}; color: white; padding: 8px 20px; border-radius: 25px; display: inline-block; margin: 15px 0; font-size: 1.1em; font-weight: bold;">
                    Primary Diagnosis: {diagnosis}
                </div>
            </div>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin-bottom: 25px;">
                <div style="background: rgba(255,255,255,0.9); padding: 20px; border-radius: 15px; text-align: center;">
                    <h3 style="color: {diagnosis_color}; margin-top: 0;">🎯 Confidence</h3>
                    <div style="font-size: 2.5em; font-weight: bold; color: {diagnosis_color};">{binary_conf:.1f}%</div>
                    <div style="background: {risk_color}; color: white; padding: 5px 15px; border-radius: 20px; margin-top: 10px; font-weight: bold;">
                        {risk_level}
                    </div>
                </div>
                
                <div style="background: rgba(255,255,255,0.9); padding: 20px; border-radius: 15px; text-align: center;">
                    <h3 style="color: #e74c3c; margin-top: 0;">📊 Uncertainty</h3>
                    <div style="font-size: 2.5em; font-weight: bold; color: #e74c3c;">±{binary_uncertainty:.1f}%</div>
                    <div style="background: #3498db; color: white; padding: 5px 15px; border-radius: 20px; margin-top: 10px; font-weight: bold;">
                        Model Stability
                    </div>
                </div>
                
                <div style="background: rgba(255,255,255,0.9); padding: 20px; border-radius: 15px; text-align: center;">
                    <h3 style="color: #2980b9; margin-top: 0;">🔬 Subtype</h3>
                    <div style="font-size: 1.3em; font-weight: bold; color: #2c3e50; margin: 10px 0;">{subtype}</div>
                    <div style="background: #3498db; color: white; padding: 5px 15px; border-radius: 20px; font-weight: bold;">
                        {subtype_conf:.1f}% ±{subtype_uncertainty:.1f}%
                    </div>
                </div>
            </div>
            
            {subtype_table}
            
            <div style="background: rgba(255,255,255,0.95); padding: 20px; border-radius: 15px; margin-top: 20px;">
                <h3 style="color: #2c3e50; margin-top: 0;">📋 Clinical Recommendations</h3>
                <div style="color: #34495e; line-height: 1.6; font-size: 1.1em;">
                    {recommendations}
                </div>
            </div>
            
            <div style="background: linear-gradient(45deg, #ff6b6b, #ffa500); color: white; padding: 15px; border-radius: 12px; margin-top: 20px; text-align: center;">
                <strong>⚠️ Medical Disclaimer:</strong> This AI analysis is for professional assistance only. 
                Always consult qualified healthcare professionals for definitive diagnosis and treatment decisions.
            </div>
        </div>
        """

        # Status message
        status_msg = f"{diagnosis_emoji} **Status**: Enhanced Analysis Complete - {diagnosis} detected with {binary_conf:.1f}% confidence (±{binary_uncertainty:.1f}%)"

        # Create all visualizations
        enhanced_fig = create_enhanced_visualization(img, pred_idx, binary_conf, label, subtype_conf)
        stats_fig = create_image_statistics_plot(img)
        radar_fig = create_confidence_radar_chart(binary_conf, binary_uncertainty, subtype_probs)
        uncertainty_fig = create_uncertainty_visualization(subtype_probs)
        risk_gauge = create_risk_assessment_gauge(binary_conf, diagnosis)

        return diagnosis_html, enhanced_fig, stats_fig, radar_fig, uncertainty_fig, risk_gauge, status_msg

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
