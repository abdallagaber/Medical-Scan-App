import logging
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from brain_api import router as brain_router
from tb_api import router as tb_router
from fastapi import FastAPI, UploadFile, File
from utils import load_model_from_kaggle
import os

# Configure environment variables
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Disable tensorflow logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Initialize FastAPI app
app = FastAPI(title="Medical Analysis API")


@app.on_event("startup")
async def startup_event():
    """Initialize all models at startup"""
    try:
        logging.info("Loading all models at startup...")

        # Load Brain Tumor model
        app.state.brain_model = load_model_from_kaggle(
            "khalednabawi",
            "brain-tumor-resnet",
            "v2",
            "resnet_brain_model.h5"
        )
        logging.info("Brain Tumor model loaded successfully")

        # Load TB model
        app.state.tb_model = load_model_from_kaggle(
            "khalednabawi",
            "tb-chest-prediction",
            "v1",
            "tb_resnet.h5"
        )
        logging.info("TB model loaded successfully")

    except Exception as e:
        logging.error(f"Error loading models: {str(e)}")
        raise


@app.get("/")
async def root():
    return {
        "status": "healthy",
        "message": "Medical Scan Detection API is running",
        "models_loaded": {
            "brain_model": app.state.brain_model is not None,
            "tb_model": app.state.tb_model is not None
        }
    }

# Include the API routers with specific paths
app.include_router(brain_router, prefix="/Brain-Tumor",
                   tags=["Brain Tumor Detection"])
app.include_router(tb_router, prefix="/Tuberculosis", tags=["TB Detection"])
