from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import httpx
import io
from PIL import Image
import os
from dotenv import load_dotenv
import torch
from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground
import numpy as np
import logging
from functools import lru_cache
from pathlib import Path
import threading
import asyncio
import cache_utils

load_dotenv()

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

app = FastAPI(title="3D Model Generator API")

API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"
FLUX_API_KEY = os.getenv('FLUX_API')
if not FLUX_API_KEY:
    raise ValueError("FLUX_API environment variable is not set")

headers = {"Authorization": f"Bearer {FLUX_API_KEY}"}
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_SERVER = cache_utils.create_index(os.getenv('REDIS_HOST', 'localhost'), os.getenv('REDIS_PORT', '6379'))

class ModelService:
    @staticmethod
    @lru_cache(maxsize=1)
    def get_model() -> TSR:
        """Initialize and cache the TSR model."""
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading model on device: {device}")
            model = TSR.from_pretrained(
                "stabilityai/TripoSR",
                config_name="config.yaml",
                weight_name="model.ckpt"
            )
            return model.to(device)
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Model initialization failed: {str(e)}")

async def query(payload: dict) -> bytes:
    """Make an async request to the Hugging Face API."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                API_URL,
                headers=headers,
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            return response.content
    except httpx.HTTPError as e:
        logger.error(f"HTTP error occurred: {str(e)}")
        raise HTTPException(status_code=502, detail="Failed to generate image")
    except Exception as e:
        logger.error(f"Error during API query: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

async def process_image(object_name: str) -> None:
    """Process the image generation and 3D conversion pipeline."""
    try:
        # Generate image
        prompt = f"a {object_name} against a plain gray background presented at slight angle to make it look like a 3d asset"
        image_bytes = await query({"inputs": prompt})
        
        # Save initial image
        image = Image.open(io.BytesIO(image_bytes))
        image_path = OUTPUT_DIR / f"{object_name}.png"
        image.save(image_path, format="PNG")
        logger.info(f"Saved initial image to {image_path}")
        
        # Convert to 3D
        await convert_to_3d(image_path, object_name)
        logger.info(f"Completed 3D conversion for {object_name}")
    
    except Exception as e:
        logger.error(f"Error processing image for {object_name}: {str(e)}")
        raise

async def convert_to_3d(image_path: Path, object_name: str) -> None:
    """Convert 2D image to 3D model."""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = ModelService.get_model()

        # Process image
        image = Image.open(image_path).convert("RGB")
        image = remove_background(image)
        image = resize_foreground(image, 0.85)
        logger.info("image turning to 3d!!")
        
        # Convert to numpy array and normalize
        image = np.array(image).astype(np.float32) / 255.0
        image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
        image = Image.fromarray((image * 255.0).astype(np.uint8))
        logger.info("image being extrated to mesh!!")

        with torch.no_grad():
            scene_codes = model([image], device=device)
        meshes = model.extract_mesh(scene_codes, has_vertex_color=True, resolution=256)
        
        mesh_path = OUTPUT_DIR / f"{object_name}.obj"
        meshes[0].export(str(mesh_path))
        logger.info(f"Saved 3D model to {mesh_path}")

    except Exception as e:
        logger.error(f"Error in 3D conversion: {str(e)}")
        raise

def run_in_thread(object_name: str) -> None:
    """Run the process_image function in a separate thread."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(process_image(object_name))
    loop.close()

@app.get("/generate/{object_name}")
async def text_to_image(object_name: str) -> JSONResponse:
    """
    Generate a 3D model from a text description.
    
    Args:
        object_name: Name of the object to generate
    
    Returns:
        JSONResponse with status message
    """
    try:
        # Validate input
        if not object_name.strip():
            raise HTTPException(status_code=400, detail="Object name cannot be empty")
        
        embedding = CACHE_SERVER.getEmbedding(object_name)

        # Check if object exists in cache
        cacheRes = CACHE_SERVER.get(embedding)
        if cacheRes:
            return JSONResponse(
                status_code=200,
                content={
                    "url": cacheRes,
                }
            )
        
        # Start a new thread for processing
        thread = threading.Thread(target=run_in_thread, args=(object_name,))
        thread.start()
        thread.join()

        # Add object to cache
        url = CACHE_SERVER.post("/backend/output/" + object_name + ".obj", embedding)
    
        return JSONResponse(
            status_code=200,
            content={
                "url": url
            }
        )
    
    except Exception as e:
        logger.error(f"Error initiating generation for {object_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)