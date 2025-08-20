import cv2
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from tensorflow.keras.models import Model
from config import models, classes, target_size, img_size
from model_utils import get_crop_box

def make_enhanced_heatmap(img, pred_idx, layer_names=None):
    resized = cv2.resize(img, img_size)
    img_in = resized.astype(np.float32) / 255.0
    img_in = np.expand_dims(img_in, axis=0)
    if layer_names is None:
        conv_layers = []
        for layer in reversed(models['binary'].layers):
            if isinstance(layer, type(models['binary'].layers[0])) and 'conv' in layer.name.lower():
                conv_layers.append(layer.name)
                if len(conv_layers) >= 3:
                    break
        layer_names = conv_layers
    heatmaps = {}
    for layer_name in layer_names:
        grad_model_temp = Model(inputs=models['binary'].inputs, outputs=[models['binary'].get_layer(layer_name).output, models['binary'].output])
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model_temp(img_in)
            loss = preds[:, pred_idx]
        grads = tape.gradient(loss, conv_out)[0]
        pooled = np.mean(grads, axis=(0, 1))
        conv_out = conv_out[0]
        heatmap = np.dot(conv_out, pooled)
        heatmap = np.maximum(heatmap, 0)
        heatmap = heatmap / np.max(heatmap) if np.max(heatmap)>0 else np.zeros_like(heatmap)
        heatmap = cv2.resize(heatmap, target_size)
        heatmaps[layer_name] = heatmap
    return heatmaps

def create_confidence_radar_chart(binary_conf, binary_uncertainty, subtype_probs):
    categories = ['Binary Confidence','Certainty Score','Model Stability']
    values = [binary_conf, 100-binary_uncertainty, min(100, binary_conf-binary_uncertainty)]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=values, theta=categories, fill='toself', name='Confidence Scores'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,100])))
    return fig
