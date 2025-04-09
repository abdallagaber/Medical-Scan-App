import uvicorn
from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
from io import BytesIO
from PIL import Image
import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import logging
from brain_api import router as brain_router
from tb_api import router as tb_router

# Initialize FastAPI app
app = FastAPI(title="Medical Analysis API")


@app.get("/")
async def root():
    return {"status": "healthy", "message": "Medical Scan Detection API is running"}


# Include the TB Detection API router with a specific path
app.include_router(brain_router, prefix="/Brain-Tumor",
                   tags=["Brain Tumor Detection"])
app.include_router(tb_router, prefix="/Tuberculosis",
                   tags=["TB Detection"])
