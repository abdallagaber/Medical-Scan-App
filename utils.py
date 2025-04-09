import os
import kagglehub
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np


def load_model_from_kaggle(user: str, model_slug: str, variation_slug: str, filename: str):
    """Load model from Kaggle Hub with error handling."""
    try:
        # Construct the model handle dynamically
        model_handle = f"{user}/{model_slug}/keras/{variation_slug}"
        MODEL_PATH = kagglehub.model_download(model_handle)

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

        model = load_model(os.path.join(MODEL_PATH, filename), compile=False)
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise


def preprocess_image(img):
    """Preprocesses the uploaded image for ResNet50."""
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    # Normalize using ResNet50's preprocessing
    img_array = preprocess_input(img_array)
    return img_array
