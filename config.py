import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import gradio as gr
import warnings
import seaborn as sns
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime
import json

from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model as keras_load_model

# Suppress noisy warnings (keeps output cleaner)
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.get_logger().setLevel('ERROR')

# Debug flag (set to True to print helpful debug info)
DEBUG = False

# Hugging Face Hub configuration
REPO_ID = "jehadcheyi/bc_models"
HF_TOKEN = os.getenv("jehad1")  # Ensure this env var is set

# Model paths
model_paths = {
    'binary': 'binary.h5',
    'benign': 'benign.h5',
    'malignant': 'malignant.h5'
}

# Image and class definitions
img_size = (128, 128)
target_size = (500, 500)

classes = {
    'binary': ['benign', 'malignant'],
    'benign': ['A', 'F', 'PT', 'TA'],
    'malignant': ['DC', 'LC', 'MC', 'PC'],
    'benign_names': {'A': 'Adenosis', 'F': 'Fibroadenoma', 'PT': 'Phyllodes Tumor', 'TA': 'Tubular Adenoma'},
    'malignant_names': {'DC': 'Ductal Carcinoma', 'LC': 'Lobular Carcinoma', 'MC': 'Mucinous Carcinoma', 'PC': 'Papillary Carcinoma'}
}

# Initialize models dictionary
models = {}

# Analysis history for statistics
analysis_history = []
