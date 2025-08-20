from config import np, cv2, go, make_subplots, target_size

def analyze_image_properties(img):
    """Analyze basic image properties"""
    properties = {}
    
    # Basic properties
    properties['height'], properties['width'] = img.shape[:2]
    properties['channels'] = img.shape[2] if len(img.shape) > 2 else 1
    properties['total_pixels'] = properties['height'] * properties['width']
    
    # Color analysis
    if len(img.shape) == 3:
        properties['mean_rgb'] = np.mean(img, axis=(0, 1))
        properties['std_rgb'] = np.std(img, axis=(0, 1))
    
    # Intensity analysis
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
    properties['mean_intensity'] = np.mean(gray)
    properties['std_intensity'] = np.std(gray)
    properties['min_intensity'] = np.min(gray)
    properties['max_intensity'] = np.max(gray)
    
    # Texture analysis (simple)
    properties['contrast'] = np.std(gray)
    
    return properties

def create_image_statistics_plot(img):
    """Create comprehensive image statistics visualization"""
    properties = analyze_image_properties(img)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=('RGB Histogram', 'Intensity Distribution', 'Color Channels',
                       'Image Properties', 'Texture Analysis', 'Quality Metrics'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
               [{"type": "table"}, {"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # RGB Histogram
    colors = ['red', 'green', 'blue']
    for i, color in enumerate(colors):
        hist, bins = np.histogram(img[:,:,i].flatten(), bins=50, range=[0, 255])
        fig.add_trace(
            go.Scatter(x=bins[:-1], y=hist, mode='lines', name=f'{color.upper()}', 
                      line=dict(color=color, width=2)),
            row=1, col=1
        )
    
    # Intensity Distribution
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hist, bins = np.histogram(gray.flatten(), bins=50, range=[0, 255])
    fig.add_trace(
        go.Scatter(x=bins[:-1], y=hist, mode='lines', fill='tonexty',
                  name='Intensity', line=dict(color='purple', width=2)),
        row=1, col=2
    )
    
    # Color Channels Bar Chart
    mean_values = [np.mean(img[:,:,i]) for i in range(3)]
    fig.add_trace(
        go.Bar(x=['Red', 'Green', 'Blue'], y=mean_values, 
               marker_color=['red', 'green', 'blue'], name='Mean Values'),
        row=1, col=3
    )
    
    # Image Properties Table
    props_data = [
        ['Dimensions', f"{properties['width']} × {properties['height']}"],
        ['Total Pixels', f"{properties['total_pixels']:,}"],
        ['Mean Intensity', f"{properties['mean_intensity']:.2f}"],
        ['Contrast', f"{properties['contrast']:.2f}"],
        ['Dynamic Range', f"{properties['max_intensity'] - properties['min_intensity']}"]
    ]
    
    fig.add_trace(
        go.Table(
            header=dict(values=['Property', 'Value'], fill_color='lightblue'),
            cells=dict(values=list(zip(*props_data)), fill_color='white')
        ),
        row=2, col=1
    )
    
    # Texture Analysis (Edge Detection)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    texture_metrics = ['Edge Density', 'Contrast', 'Homogeneity']
    texture_values = [edge_density, properties['contrast']/255, 1 - properties['std_intensity']/255]
    
    fig.add_trace(
        go.Bar(x=texture_metrics, y=texture_values, 
               marker_color=['orange', 'purple', 'green'], name='Texture'),
        row=2, col=2
    )
    
    # Quality Metrics
    quality_score = min(100, (properties['contrast'] / 50) * 100)  # Simple quality metric
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    fig.add_trace(
        go.Scatter(x=['Quality Score', 'Sharpness'], y=[quality_score, min(100, sharpness/10)],
                  mode='markers+lines', marker=dict(size=15, color='red'),
                  name='Quality Metrics'),
        row=2, col=3
    )
    
    fig.update_layout(height=800, showlegend=True, title_text="📊 Comprehensive Image Analysis")
    return fig

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
        values.append(prob_data['mean'])
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Confidence Scores',
        line_color='rgb(255, 100, 100)',
        fillcolor='rgba(255, 100, 100, 0.3)'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title="🎯 AI Confidence Radar Chart",
        height=500
    )
    return fig

def create_uncertainty_visualization(subtype_probs):
    """Create uncertainty visualization with error bars"""
    names = list(subtype_probs.keys())
    means = [prob_data['mean'] for prob_data in subtype_probs.values()]
    stds = [prob_data['std'] for prob_data in subtype_probs.values()]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names,
        y=means,
        error_y=dict(type='data', array=stds, visible=True),
        marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F'],
        name='Subtype Probabilities'
    ))
    fig.update_layout(
        title="🔬 Subtype Classification with Uncertainty",
        xaxis_title="Cancer Subtypes",
        yaxis_title="Probability (%)",
        height=400,
        showlegend=False
    )
    return fig

def create_risk_assessment_gauge(binary_conf, diagnosis):
    """Create a risk assessment gauge"""
    # Calculate risk score
    if diagnosis.lower() == 'malignant':
        risk_score = binary_conf
        risk_color = 'red'
        risk_text = 'HIGH RISK'
    else:
        risk_score = 100 - binary_conf
        risk_color = 'green'
        risk_text = 'LOW RISK'
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = risk_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"🚨 Risk Assessment: {risk_text}"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': risk_color},
            'steps': [
                {'range': [0, 25], 'color': "lightgreen"},
                {'range': [25, 50], 'color': "yellow"},
                {'range': [50, 75], 'color': "orange"},
                {'range': [75, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=400)
    return fig
