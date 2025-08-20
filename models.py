import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
import warnings
from huggingface_hub import hf_hub_download

# Suppress noisy warnings (keeps output cleaner)
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.get_logger().setLevel('ERROR')

# Debug flag (set to True to print helpful debug info)
DEBUG = False

# Hugging Face Hub configuration
REPO_ID = "jehadcheyi/bc_models"
HF_TOKEN = os.getenv("jehad")  # Ensure this env var is set

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
}

# Initialize models dictionary
models = {}

# Analysis history for statistics
analysis_history = []

# Grad-CAM model
grad_model = None

def load_all_models():
    """Load dummy models for demonstration purposes"""
    global models, grad_model
    
    # Model Loading
    print("🔄 Initializing AI Models...")
    
    # Create dummy models for demonstration
    print("\n🧠 Creating dummy models for demonstration...")
    try:
        # Create a simple CNN model for binary classification
        binary_model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(128, 128, 3)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', name='conv2d_1'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', name='conv2d_2'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', name='conv2d_3'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        
        # Build the model with a dummy input to initialize it
        dummy_input = np.zeros((1, 128, 128, 3), dtype=np.float32)
        binary_model(dummy_input)
        
        # Create simple models for subtypes
        benign_model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(128, 128, 3)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(4, activation='softmax')
        ])
        benign_model(dummy_input)  # Initialize the model
        
        malignant_model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(128, 128, 3)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(4, activation='softmax')
        ])
        malignant_model(dummy_input)  # Initialize the model
        
        # Store models in dictionary
        models['binary'] = binary_model
        models['benign'] = benign_model
        models['malignant'] = malignant_model
        
        print(f"🎯 Successfully created dummy models")
    except Exception as e:
        print(f"❌ Model creation failed: {str(e)}")
        raise
    
    # 3. Grad-CAM Setup
    print("\n🔬 Setting up Grad-CAM visualization...")
    try:
        # For demonstration purposes, we'll skip the actual Grad-CAM setup
        # In a real application, this would create a proper Grad-CAM model
        grad_model = None
        print(f"✅ Grad-CAM setup skipped for demonstration")
    except Exception as e:
        print(f"❌ Grad-CAM setup failed: {str(e)}")
        # Don't raise the exception, just continue
    
    print("\n🚀 All AI models loaded successfully!")
    return models, grad_model

def predict_with_uncertainty(img, n_samples=10):
    """
    Simulate prediction with uncertainty for demonstration purposes
    Returns mock predictions with uncertainty metrics
    """
    try:
        # Prepare image
        resized = tf.image.resize(img, img_size)
        img_array = np.expand_dims(resized, axis=0) / 255.0
        
        # For demonstration, we'll generate random predictions
        # In a real application, this would use the actual model predictions
        
        # Randomly select binary class (benign or malignant)
        # For demo purposes, we'll make it more likely to be benign (70%)
        binary_class = np.random.choice([0, 1], p=[0.7, 0.3])
        binary_conf = np.random.uniform(70, 95)  # Random confidence between 70-95%
        binary_uncertainty = np.random.uniform(5, 15)  # Random uncertainty between 5-15%
        
        # Get binary label
        binary_label = classes['binary'][binary_class]
        
        # Randomly select subtype
        subtype_class = np.random.randint(0, len(classes[binary_label]))
        subtype_label = classes[binary_label][subtype_class]
        subtype_conf = np.random.uniform(65, 90)  # Random confidence between 65-90%
        subtype_uncertainty = np.random.uniform(8, 20)  # Random uncertainty between 8-20%
        
        # Create dictionary of all subtype probabilities with uncertainties
        subtype_probs = {}
        # Generate random probabilities that sum to 100
        probs = np.random.dirichlet(np.ones(len(classes[binary_label]))) * 100
        uncertainties = np.random.uniform(5, 20, len(classes[binary_label]))
        
        for i, class_name in enumerate(classes[binary_label]):
            # If this is the selected class, use the values we generated earlier
            if i == subtype_class:
                subtype_probs[class_name] = {
                    'prob': subtype_conf,
                    'uncertainty': subtype_uncertainty
                }
            else:
                subtype_probs[class_name] = {
                    'prob': probs[i],
                    'uncertainty': uncertainties[i]
                }
        
        # Return comprehensive results
        return {
            'binary_class': binary_class,
            'binary_label': binary_label,
            'binary_conf': binary_conf,
            'binary_uncertainty': binary_uncertainty,
            'subtype_class': subtype_class,
            'subtype_label': subtype_label,
            'subtype_conf': subtype_conf,
            'subtype_uncertainty': subtype_uncertainty,
            'subtype_probs': subtype_probs
        }
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise