import logging
import os
import sys
import platform
import torch
from contextlib import asynccontextmanager
# Add the parent directory to sys.path to allow imports from api_spz
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



# -------------LOW VRAM TESTING -------------
#
# # only used for debugging, to emulate low-vram graphics cards:
#
# torch.cuda.set_per_process_memory_fraction(0.5)  # Limit to 43% of my available VRAM, for testing.
# And/or set maximum split size (in MB)
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,garbage_collection_threshold:0.8'


import argparse
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


# Import the StableProjectorz API routes
from api_spz.routes.generation import router as generation_router
from api_spz.core.state_manage import state

# Set up logging
os.makedirs("temp", exist_ok=True)
os.makedirs("temp/current_generation", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join("temp", "api.log"))
    ]
)
logger = logging.getLogger("hunyuan3d_api")

# Print system information
print(
    f"\n[System Info] Python: {platform.python_version():<8} | "
    f"PyTorch: {torch.__version__:<8} | "
    f"CUDA: {'not available' if not torch.cuda.is_available() else torch.version.cuda}\n"
)

# Parse command line arguments
parser = argparse.ArgumentParser(description="Run Hunyuan3D-StableProjectorz API server")
parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server to")
parser.add_argument("--port", type=int, default=7960, help="Port to bind the server to")
parser.add_argument("--model-path", type=str, default="tencent/Hunyuan3D-2mini", help="Hunyuan3D model path")
parser.add_argument("--subfolder", type=str, default="hunyuan3d-dit-v2-mini-turbo", help="Model subfolder")
parser.add_argument("--texgen-model-path", type=str, default="tencent/Hunyuan3D-2", help="Texture model path")
parser.add_argument("--device", type=str, default=None, help="Device to use (cuda or cpu)")
parser.add_argument("--disable-flashvdm", action="store_true", help="Disable FlashVDM acceleration")
parser.add_argument("--low-vram-mode", action="store_true", help="Enable low VRAM mode with CPU offloading")

args = parser.parse_args()

# Print startup information
print("\n" + "="*50)
print("Hunyuan3D-StableProjectorz API Server is starting up:")
print("Touching this window will pause it. If it happens, click inside it and press 'Enter' to unpause")
print("="*50 + "\n")

# Define lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize models and resources
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    state.initialize_pipeline(
        model_path=args.model_path,
        subfolder=args.subfolder,
        texgen_model_path=args.texgen_model_path,
        device=device,
        enable_flashvdm=not args.disable_flashvdm,
        low_vram_mode=args.low_vram_mode
    )
    logger.info(f"Initialized Hunyuan3D with model {args.model_path}/{args.subfolder} on {device}")
    
    print("\n" + "="*50)
    print(f"Hunyuan3D-StableProjectorz API Server v1.0.0")
    print(f"Server is active and listening on {args.host}:{args.port}")
    print("="*50 + "\n")
    
    yield  # This is where the FastAPI application runs
    
    # Shutdown: Clean up resources
    state.cleanup()
    logger.info("Server shutting down, resources cleaned up")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Hunyuan3D-StableProjectorz API",
    description="API for Hunyuan3D 3D generation compatible with StableProjectorz",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the generation routes
app.include_router(generation_router)

# Mount static files for downloads
app.mount("/downloads", StaticFiles(directory=Path("temp/current_generation")), name="downloads")


@app.get("/")
async def root():
    return {
        "message": "Hunyuan3D-StableProjectorz API is running",
        "status": "ready",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")