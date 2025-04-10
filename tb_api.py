from fastapi import APIRouter, UploadFile, File, HTTPException
from utils import preprocess_image, load_model_from_kaggle
import logging
from io import BytesIO
from PIL import Image
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter()

# Global model variable
model = None
CLASS_LABELS = {0: "Normal", 1: "Tuberculosis"}


@router.on_event("startup")
async def startup_event():
    """Initialize model on startup with error handling"""
    global model
    try:
        logger.info("Starting TB model initialization...")
        # Disable GPU and set memory growth
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

        model = load_model_from_kaggle(
            "khalednabawi",
            'tb-chest-prediction',
            "v1",
            'tb_resnet.h5'
        )
        logger.info("TB model initialized successfully!")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Model initialization failed: {str(e)}"
        )


@router.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "TB Detection API is running"
    }


@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint to predict tuberculosis from uploaded chest X-ray
    Args:
        file: Uploaded image file
    Returns:
        dict: Prediction results including class and confidence
    """
    try:
        logger.info(f"Receiving prediction request for file: {file.filename}")

        # Validate image file
        if not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="File must be an image"
            )

        # Read and preprocess image
        contents = await file.read()
        img = Image.open(BytesIO(contents)).convert("RGB")
        img_array = preprocess_image(img)

        if model is None:
            raise ValueError("Model not initialized")

        # Make prediction
        prediction = model.predict(img_array, verbose=0)
        predicted_class = int(prediction[0][0] > 0.5)
        confidence = float(prediction[0][0])

        result = CLASS_LABELS[predicted_class]
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
