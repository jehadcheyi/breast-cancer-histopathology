from config import (
    hf_hub_download, REPO_ID, HF_TOKEN, model_paths,
    tf, keras_load_model, models, Model
)

# Model Loading
print("🔄 Initializing AI Models...")

# 1. First verify files exist on HF hub (download to cache)
print("📋 Verifying model files...")
for name, filename in model_paths.items():
    try:
        local_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            token=HF_TOKEN,
            cache_dir="model_cache"
        )
        print(f"✅ Found {name} model at: {local_path}")
    except Exception as e:
        print(f"❌ Error verifying {name} model: {str(e)}")
        raise

# 2. Load all models
print("\n🧠 Loading AI models into memory...")
try:
    for name, filename in model_paths.items():
        local_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            token=HF_TOKEN,
            cache_dir="model_cache"
        )
        # Use CPU device to avoid environment GPU-related issues during load
        with tf.device('/CPU:0'):
            models[name] = keras_load_model(local_path, compile=False)
        print(f"🎯 Successfully loaded {name} model")
except Exception as e:
    print(f"❌ Model loading failed: {str(e)}")
    raise

# 3. Grad-CAM Setup
print("\n🔬 Setting up Grad-CAM visualization...")
try:
    binary_model = models['binary']
    last_conv = None

    # Find last conv layer (most reliable approach)
    for layer in reversed(binary_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer.name
            break

    if last_conv is None:
        raise ValueError("No convolutional layer found in binary model")

    grad_model = Model(
        inputs=binary_model.inputs,
        outputs=[binary_model.get_layer(last_conv).output, binary_model.output]
    )

    print("✅ Grad-CAM setup complete")
except Exception as e:
    print(f"❌ Grad-CAM setup failed: {str(e)}")
    raise

print("\n🚀 All AI systems loaded successfully!")
