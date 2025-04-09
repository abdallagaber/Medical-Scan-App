# Disable GPU and force TensorFlow to use CPU
from tb_api import router as tb_router
from brain_api import router as brain_router
import logging
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
from PIL import Image
from io import BytesIO
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


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
