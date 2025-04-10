import logging
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from brain_api import router as brain_router
from tb_api import router as tb_router
from fastapi import FastAPI, UploadFile, File
import os
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


@app.get("/")
async def root():
    return {"status": "healthy", "message": "Medical Scan Detection API is running"}


# Include the TB Detection API router with a specific path
app.include_router(tb_router, prefix="/Tuberculosis",
                   tags=["TB Detection"])
app.include_router(brain_router, prefix="/Brain-Tumor",
                   tags=["Brain Tumor Detection"])
