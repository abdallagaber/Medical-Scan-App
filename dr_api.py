import logging
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from PIL import Image
from io import BytesIO
import tensorflow as tf
import numpy as np
import json
from similarity import check_similarity

from utils import preprocess_image
# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Define class labels
DR_CLASSES = {
    0: "DR",
    1: "No_DR"
}


@router.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Diabetic Retinopathy Detection API is running"}


@router.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    """
    Endpoint to predict diabetic retinopathy from uploaded retinal images
    Args:
        file: Uploaded image file
        request: FastAPI request object to access app state
    Returns:
        dict: Prediction results including class and confidence
    """
    try:
        logger.info(f"Receiving prediction request for file: {file.filename}")

        # Get model from app state
        dr_model = request.app.state.dr_model
        if dr_model is None:
            raise ValueError("DR model not initialized")

        # Validate image file
        if not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="File must be an image"
            )

        # Read and preprocess image
        contents = await file.read()
        # Parse similarity check response
        similar = check_similarity(contents)
        similarity_data = similar.body.decode()  # Convert bytes to string
        similarity_result = json.loads(similarity_data)  # Parse JSON string

        if similarity_result.get("prediction") == "not-medical":
            raise HTTPException(
                status_code=400,
                detail="File is not a valid medical image"
            )

        # Add logging for debugging
        logger.info(f"Similarity check result: {similarity_data}")

        img_bytes = BytesIO(contents)

        # Process image for prediction
        img = Image.open(img_bytes).convert("RGB")
        img_array = preprocess_image(img)

        # Make prediction
        prediction = dr_model.predict(img_array, verbose=0)
        predicted_class = int(prediction[0][0] > 0.5)
        confidence = float(prediction[0][0])

        result = DR_CLASSES[predicted_class]
        logger.info(f"Prediction complete: {result}")

        return {
            "success": True,
            "filename": file.filename,
            "prediction": result,
            "confidence": confidence,
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
