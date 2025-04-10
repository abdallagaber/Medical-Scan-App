from fastapi import APIRouter, UploadFile, File, HTTPException
import numpy as np
from io import BytesIO
from PIL import Image
import logging
import os
from utils import preprocess_image, load_model_from_kaggle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter()

# Add a root endpoint for health check


@router.get("/")
async def root():
    return {"status": "healthy", "message": "TB Detection API is running"}

# Global model variable
model = None


@router.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    global model
    try:
        logger.info("Starting TB model initialization...")
        model = load_model_from_kaggle(
            "khalednabawi", "tb-chest-prediction", "v1", 'tb_resnet.h5')
        logger.info("TB model initialized successfully!")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise HTTPException(
            status_code=500, detail="Model initialization failed")

# Define class labels
CLASS_LABELS = {0: "Normal", 1: "Tuberculosis"}


@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Receives an image file, preprocesses it, and returns TB classification."""
    try:
        logger.info(f"Receiving prediction request for file: {file.filename}")
        contents = await file.read()
        img = Image.open(BytesIO(contents)).convert("RGB")  # Ensure RGB format
        img_array = preprocess_image(img)

        if model is None:
            raise ValueError("Model not initialized")

        # Make prediction
        prediction = model.predict(img_array)
        # Convert sigmoid output to 0 or 1
        predicted_class = int(prediction[0][0] > 0.5)
        confidence = float(prediction[0][0])  # Confidence score

        logger.info(f"Prediction complete: {CLASS_LABELS[predicted_class]}")
        return {
            "success": True,
            "filename": file.filename,
            "prediction": CLASS_LABELS[predicted_class],
            "confidence": confidence
        }
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }
