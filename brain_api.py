from fastapi import APIRouter, UploadFile, File, HTTPException
from utils import preprocess_image
import logging
from io import BytesIO
from PIL import Image
from utils import ModelSingleton

# Initialize router
router = APIRouter()


@router.get("/")
async def root():
    return {"status": "healthy", "message": "Brain Tumor Detection API is running"}


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model variable
model = None
class_indices = {'Brain Tumor': 0, 'Healthy': 1}


@router.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    global model
    try:
        logger.info("Starting brain model initialization...")
        model = ModelSingleton.get_brain_model()
        logger.info("Brain model initialized successfully!")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise HTTPException(
            status_code=500, detail="Model initialization failed")


@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Receives an image file, preprocesses it, and returns brain tumor classification."""
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

        result = "Brain Tumor" if predicted_class == 0 else "Healthy"
        logger.info(f"Prediction complete: {result}")

        return {
            "success": True,
            "filename": file.filename,
            "prediction": result,
            "confidence": confidence,
        }
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Prediction failed: {str(e)}")
