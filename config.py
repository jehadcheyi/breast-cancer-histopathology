import os
import warnings
import tensorflow as tf

# Suppress warnings and TensorFlow logs
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.get_logger().setLevel('ERROR')

# Debug flag
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

# Image sizes
img_size = (128, 128)
target_size = (500, 500)

# Class definitions
classes = {
    'binary': ['benign', 'malignant'],
    'benign': ['A', 'F', 'PT', 'TA'],
    'malignant': ['DC', 'LC', 'MC', 'PC'],
    'benign_names': {'A': 'Adenosis', 'F': 'Fibroadenoma', 'PT': 'Phyllodes Tumor', 'TA': 'Tubular Adenoma'},
    'malignant_names': {'DC': 'Ductal Carcinoma', 'LC': 'Lobular Carcinoma', 'MC': 'Mucinous Carcinoma', 'PC': 'Papillary Carcinoma'}
}

# Global variables
models = {}
analysis_history = []
