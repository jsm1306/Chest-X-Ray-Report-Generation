"""
Main Application - FastAPI medical report generation service
"""
import os
import logging
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import modules
from model_loader import load_models, unload_models
from routes import router
from utils import create_upload_dir


# ============ LIFESPAN MANAGEMENT ============
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    
    Startup:
    - Create upload directory
    - Load all models (encoder, full model, tokenizer)
    
    Shutdown:
    - Unload models from memory
    """
    logger.info("=" * 60)
    logger.info("STARTING APPLICATION...")
    logger.info("=" * 60)
    
    try:
        # Create upload directory
        create_upload_dir()
        
        # Load models
        logger.info("Loading models...")
        load_models(
            encoder_path="encoder_model.keras",
            full_model_path="full_model.keras",
            tokenizer_path="tokenizer.pkl"
        )
        logger.info("✓ Models loaded successfully at startup")
        
    except FileNotFoundError as e:
        logger.error(f"❌ Startup failed - Model files not found: {e}")
        logger.error("Please ensure encoder_model.keras, full_model.keras, and tokenizer.pkl are in the project directory")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        sys.exit(1)
    
    yield  # Application runs here
    
    logger.info("=" * 60)
    logger.info("SHUTTING DOWN APPLICATION...")
    logger.info("=" * 60)
    
    # Cleanup
    try:
        unload_models()
        logger.info("✓ Models unloaded")
    except Exception as e:
        logger.warning(f"Error during shutdown: {e}")


# ============ CREATE FASTAPI APP ============
app = FastAPI(
    title="Medical Diagnosis Report API",
    description="Vision-Language Model backend for generating medical diagnostic reports from X-ray images",
    version="1.0.0",
    lifespan=lifespan
)


# ============ CORS MIDDLEWARE ============
# Enable CORS for frontend integration
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:5173",  # Vite dev server
    "http://127.0.0.1",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "*",  # Allow all origins (restrict in production)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("✓ CORS enabled for frontend integration")


# ============ INCLUDE ROUTES ============
app.include_router(router, tags=["medical-reports"])

logger.info("✓ API routes registered")


# ============ STARTUP/SHUTDOWN EVENTS ============
@app.on_event("startup")
async def startup_event():
    """Additional startup logging"""
    logger.info("FastAPI application started successfully")
    logger.info("API Documentation available at: http://localhost:8000/docs")


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown logging"""
    logger.info("FastAPI application shutting down")


# ============ RUN SERVER ============
if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting development server...")
    logger.info("Open browser to: http://localhost:8000")
    logger.info("View API docs at: http://localhost:8000/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
