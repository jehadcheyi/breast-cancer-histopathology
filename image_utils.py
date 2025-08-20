import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from config import img_size, DEBUG
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def prep_img(img):
    if img is None:
        raise ValueError("No image input provided to prep_img()")
    arr = img.copy()
    if arr.dtype != np.uint8:
        if arr.dtype in [np.float32, np.float64]:
            if arr.max() <= 1.0:
                arr = (arr * 255).astype('uint8')
            else:
                arr = arr.astype('uint8')
        else:
            arr = arr.astype('uint8')
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.concatenate([arr] * 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[..., :3]
    pil = Image.fromarray(arr)
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    pil = pil.resize(img_size, Image.LANCZOS)
    arr = img_to_array(pil).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    if DEBUG:
        print("prep_img -> shape:", arr.shape, "dtype:", arr.dtype)
    return arr

def analyze_image_properties(img):
    props = {}
    props['height'], props['width'] = img.shape[:2]
    props['channels'] = img.shape[2] if len(img.shape) > 2 else 1
    props['total_pixels'] = props['height'] * props['width']
    if len(img.shape) == 3:
        props['mean_rgb'] = np.mean(img, axis=(0, 1))
        props['std_rgb'] = np.std(img, axis=(0, 1))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
    props['mean_intensity'] = np.mean(gray)
    props['std_intensity'] = np.std(gray)
    props['min_intensity'] = np.min(gray)
    props['max_intensity'] = np.max(gray)
    props['contrast'] = np.std(gray)
    return props

def create_image_statistics_plot(img):
    props = analyze_image_properties(img)
    fig = make_subplots(rows=2, cols=3,
                        subplot_titles=('RGB Histogram', 'Intensity Distribution', 'Color Channels',
                                       'Image Properties', 'Texture Analysis', 'Quality Metrics'))
    colors = ['red', 'green', 'blue']
    for i, color in enumerate(colors):
        hist, bins = np.histogram(img[:,:,i].flatten(), bins=50, range=[0, 255])
        fig.add_trace(go.Scatter(x=bins[:-1], y=hist, mode='lines', name=f'{color.upper()}', line=dict(color=color, width=2)), row=1, col=1)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hist, bins = np.histogram(gray.flatten(), bins=50, range=[0, 255])
    fig.add_trace(go.Scatter(x=bins[:-1], y=hist, mode='lines', fill='tonexty', name='Intensity', line=dict(color='purple', width=2)), row=1, col=2)
    mean_values = [np.mean(img[:,:,i]) for i in range(3)]
    fig.add_trace(go.Bar(x=['Red','Green','Blue'], y=mean_values, marker_color=['red','green','blue']), row=1, col=3)
    props_data = [['Dimensions', f"{props['width']} × {props['height']}"],
                  ['Total Pixels', f"{props['total_pixels']:,}"],
                  ['Mean Intensity', f"{props['mean_intensity']:.2f}"],
                  ['Contrast', f"{props['contrast']:.2f}"],
                  ['Dynamic Range', f"{props['max_intensity'] - props['min_intensity']}"]]
    fig.add_trace(go.Table(header=dict(values=['Property','Value']), cells=dict(values=list(zip(*props_data)))), row=2, col=1)
    return fig
