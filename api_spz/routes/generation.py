
import logging
import time
import traceback
from typing import Dict, Optional, List
import asyncio
import io
import base64
import os
from fastapi import APIRouter, File, Response, UploadFile, Form, HTTPException, Query, Depends
from fastapi.responses import FileResponse
from PIL import Image
import torch

from hy3dgen.shapegen.pipelines import export_to_trimesh

from api_spz.core.exceptions import CancelledException
from api_spz.core.files_manage import file_manager
from api_spz.core.state_manage import state
from api_spz.core.models_pydantic import (
    GenerationArgForm,
    GenerationResponse, 
    TaskStatus,
    StatusResponse
)


router = APIRouter()

logger = logging.getLogger("trellis") #was already setup earlier, during main.

cancel_event = asyncio.Event() # This event will be set by the endpoint /interrupt

# A single lock to ensure only one generation at a time
generation_lock = asyncio.Lock()
def is_generation_in_progress() -> bool:
    return generation_lock.locked()


# A single dictionary holding "current generation" metadata
current_generation = {
    "status": TaskStatus.FAILED, # default
    "progress": 0,
    "message": "",
    "outputs": None,       # pipeline outputs if we did partial gen.
    "model_url": None      # final model path if relevant.
}


# Helper to reset the "current_generation" dictionary
# (useful to start fresh each time we begin generating)
def reset_current_generation():
    cancel_event.clear()
    current_generation["status"] = TaskStatus.PROCESSING
    current_generation["progress"] = 0
    current_generation["message"] = ""
    current_generation["outputs"] = None
    current_generation["model_url"] = None


# Helper to update the "current_generation" dictionary
def update_current_generation(
    status: Optional[TaskStatus] = None,
    progress: Optional[int] = None,
    message: Optional[str] = None,
    outputs=None
):
    if status is not None:
        current_generation["status"] = status
    if progress is not None:
        current_generation["progress"] = progress
    if message is not None:
        current_generation["message"] = message
    if outputs is not None:
        current_generation["outputs"] = outputs


# Cleanup files in "current_generation" folder
async def cleanup_generation_files(keep_videos: bool = False, keep_model: bool = False):
    file_manager.cleanup_generation_files(keep_videos=keep_videos, keep_model=keep_model)


# Validate input
def _gen_3d_validate_params(file_or_files, b64_or_b64list, arg: GenerationArgForm):
    """Validate incoming parameters before generation."""
    if (not file_or_files or len(file_or_files) == 0) and (not b64_or_b64list or len(b64_or_b64list) == 0):
        raise HTTPException(400, "No input images provided")
    # Range checks:
    if not (0 < arg.guidance_scale <= 10):
        raise HTTPException(status_code=400, detail="Guidance scale must be above 0 and <= 10")
    if not (0 < arg.num_inference_steps <= 50):
        raise HTTPException(status_code=400, detail="Inference steps must be above 0 and <= 50")
    if not (128 <= arg.octree_resolution <= 512):
        raise HTTPException(status_code=400, detail="Octree resolution must be between 128 and 512")
    if not (1000 <= arg.num_chunks <= 200000):
        raise HTTPException(status_code=400, detail="Number of chunks must be between 1000 and 200000")
    if not (0 < arg.mesh_simplify_ratio <= 100): 
        raise HTTPException(status_code=400, detail="mesh_simplify_ratio must be between 0 and 1, or between 0 and 100")
    if not (512 <= arg.texture_size <= 4096):
        raise HTTPException(status_code=400, detail="Texture size must be between 512 and 4096")
    if arg.output_format not in ["glb", "obj"]:
        raise HTTPException(status_code=400, detail="Unsupported output format")



async def _gen_3d_get_image(file: Optional[UploadFile], image_base64: Optional[str]) -> Image.Image:
    if image_base64:
        try:
            # base64 branch
            if "base64," in image_base64:
                image_base64 = image_base64.split("base64,")[1]
            data = base64.b64decode(image_base64)
            pil_image = Image.open(io.BytesIO(data)).convert("RGBA")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 data: {str(e)}")
    else:
        try:
            content = await file.read()
            pil_image = Image.open(io.BytesIO(content)).convert("RGBA")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"get_image - issue opening the image file: {str(e)}")
    return pil_image



# Reads multiple images from either UploadFile objects or base64 strings,
# and returns a list of PIL.Image.
async def _load_images_into_list( files: Optional[List[UploadFile]] = None,
                                  images_base64: Optional[List[str]] = None) -> List[Image.Image]:
    all_images = []
    files = files or []
    images_base64 = images_base64 or []
    # 1) Base64-encoded:
    for b64_str in images_base64:
        if "base64," in b64_str:
            b64_str = b64_str.split("base64,")[1]
        img = await _gen_3d_get_image(file=None, image_base64=b64_str)
        all_images.append(img)
    # 2) Files:
    for f in files:
        img = await _gen_3d_get_image(file=f, image_base64=None)
        all_images.append(img)

    if not all_images:
        raise HTTPException(status_code=400, detail="No images provided (files or base64).")

    return all_images



# If `pil_images` is a single PIL.Image, calls pipeline.run(...).
# If `pil_images` is a list of multiple PIL.Images, calls pipeline.run_multi_image(...).
# Returns the pipeline outputs dictionary.
async def _run_pipeline_generate_3d(pil_images, arg):
    """Generate 3D structure using Hunyuan3D pipeline"""
    def worker():
        try:
            state.ensure_shape_model_on_gpu()
            pipeline = state.pipeline

            rembg = state.rembg
            
            # Clear CUDA cache before starting
            torch.cuda.empty_cache()
            
            # Set up generator for reproducible results
            generator = torch.Generator(device="cuda").manual_seed(arg.seed)
            
            # Common parameters for all pipeline calls
            pipeline_params = {
                "num_inference_steps": arg.num_inference_steps,
                "guidance_scale": arg.guidance_scale,
                "generator": generator,
                "octree_resolution": arg.octree_resolution,
                "num_chunks": arg.num_chunks,
                "output_type": 'mesh',
            }
            
            # Process input based on type (single image or multi-view)
            if isinstance(pil_images, list):
                # Check if we're using MV model
                if "mv" in state.model_path.lower():
                    # MV models always require a dictionary with view names
                    processed_images = {}
                    view_names = ['front', 'back', 'left', 'right']
                    
                    # Process available images
                    for i, img in enumerate(pil_images):
                        if i < len(view_names):
                            processed_image = rembg(img.convert('RGB'))
                            processed_images[view_names[i]] = processed_image
                    
                    # MV model always requires at least the front view
                    if 'front' not in processed_images and len(pil_images) > 0:
                        processed_images['front'] = rembg(pil_images[0].convert('RGB'))
                        
                    # Pass the dictionary directly to the pipeline
                    raw_outputs = pipeline(
                        image=processed_images,
                        **pipeline_params
                    )
                else:
                    # Standard model - just use first image
                    if len(pil_images) > 1:
                        logger.warning("Using multiple images but not using multiview model. Using only the first image.")
                    processed_image = rembg(pil_images[0].convert('RGB'))
                    raw_outputs = pipeline(image=processed_image, **pipeline_params)
            else:
                # Single image directly (not in a list)
                if "mv" in state.model_path.lower():
                    # MV model needs dictionary format
                    processed_images = {
                        'front': rembg(pil_images.convert('RGB'))
                    }
                    raw_outputs = pipeline(
                        image=processed_images,
                        **pipeline_params
                    )
                else:
                    # Standard model
                    processed_image = rembg(pil_images.convert('RGB'))
                    raw_outputs = pipeline(image=processed_image, **pipeline_params)
            
            # Convert to trimesh and apply post-processing
            mesh = export_to_trimesh(raw_outputs)[0]
            mesh = state.floater_remover(mesh)
            mesh = state.degenerate_face_remover(mesh)
            
            # Store the original image for texture generation
            if not isinstance(pil_images, list):
                original_image = pil_images
            else:
                original_image = pil_images[0] if len(pil_images) > 0 else None
            
            # Create a structure that mimics what Trellis would return
            # so the rest of StableProjectorz code can work with it
            outputs = {
                "mesh": [mesh],
                "gaussian": [mesh],  # Use mesh as stand-in for gaussian format
                "original_image": original_image
            }
            
            return outputs
            
        except Exception as e:
            logger.error(f"Error in 3D generation: {str(e)}")
            raise e
        finally:
            torch.cuda.empty_cache()
    
    return await asyncio.to_thread(worker)



def normalize_meshSimplify_ratio(ratio: float) -> float:
    """Normalize mesh_simplify_ratio to [0,1] range"""
    if ratio > 1.0:  # Detect [0,100] range
        return ratio / 100.0
    return ratio


async def _run_pipeline_generate_glb(outputs, mesh_simplify_ratio:float, texture_size:int, apply_texture:bool=False):
    """Generate final GLB model with optional texturing"""
    mesh_simplify_ratio = normalize_meshSimplify_ratio(mesh_simplify_ratio)

    def worker():
        try:
            start_time = time.time()
            
            # Clear CUDA cache
            torch.cuda.empty_cache()
            
            mesh = outputs["mesh"][0]
            original_image = outputs.get("original_image")
            
            # Apply mesh preprocessing
            logger.info(f"Processing mesh (faces: {len(mesh.faces)})")
            mesh = state.floater_remover(mesh)
            mesh = state.degenerate_face_remover(mesh)
            
            # Apply mesh simplification if requested
            if mesh_simplify_ratio < 1.0:
                target_faces = int(len(mesh.faces) * mesh_simplify_ratio)
                logger.info(f"Reducing faces to {target_faces}")
                mesh = state.face_reducer(mesh, max_facenum=target_faces)
            
            # Only apply texturing if requested for this generation
            if apply_texture and hasattr(state, 'texture_pipeline') and state.texture_pipeline is not None and original_image is not None:
                try:
                    state.unload_shape_model()  # CRITICAL: Unload shape model to free ~6GB VRAM for texture generation
                    state.optimize_texture_pipeline() #memory optimization

                    # Update texture size in existing pipeline config
                    if texture_size != state.texture_pipeline.config.texture_size:
                        logger.info(f"Changing texture resolution from {state.texture_pipeline.config.texture_size} to {texture_size}")
                        state.texture_pipeline.config.texture_size = texture_size
                        # Also update the renderer's texture size
                        state.texture_pipeline.render.set_default_texture_resolution(texture_size)

                    logger.info("Starting texture generation...")
                    textured_mesh = state.texture_pipeline(mesh, image=original_image)
                    mesh = textured_mesh
                    logger.info("Applied texture to mesh successfully")
                except Exception as tex_err:
                    logger.error(f"Texture application failed: {str(tex_err)}")
                    # Continue with untextured mesh
            else:
                logger.info("Skipping texture generation for this request")
            
            # Export to GLB
            model_path = file_manager.get_temp_path("model.glb")
            mesh.export(str(model_path))
            logger.info(f"Exported GLB model to {model_path} in {time.time() - start_time:.2f} seconds")
            
            return model_path
            
        except Exception as e:
            logger.error(f"Error generating GLB: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise e
        finally:
            torch.cuda.empty_cache()
    
    return await asyncio.to_thread(worker)

# --------------------------------------------------
# Routes
# --------------------------------------------------

@router.get("/ping")
async def ping():
    """Root endpoint to check server status."""
    busy = is_generation_in_progress()
    return {
        "status": "running",
        "message": "Trellis API is operational",
        "busy": busy
    }


@router.get("/status", response_model=StatusResponse)
async def get_status():
    """
    Get status of the single current/last generation.
    """
    return StatusResponse(
        status=current_generation["status"],
        progress=current_generation["progress"],
        message=current_generation["message"],
        busy=is_generation_in_progress(),
    )


@router.post("/generate_no_preview", response_model=GenerationResponse)
async def generate_no_preview(
    file: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None),
    arg: GenerationArgForm = Depends()
):
    """Generate a 3D model directly (no preview)."""
    print()#empty line, for an easier read.
    logger.info("Client asked to generate")
    # Acquire the lock (non-blocking)
    try:
        await asyncio.wait_for(generation_lock.acquire(), timeout=0.001)
    except asyncio.TimeoutError:
        raise HTTPException( status_code=503, detail="Server is busy with another generation")
    
    start_time = time.time() 
    # We have the lock => let's reset the "current_generation"
    reset_current_generation()
    try:
        _gen_3d_validate_params(file, image_base64, arg)

        #construct an image
        image = await _gen_3d_get_image(file, image_base64)

        update_current_generation( status=TaskStatus.PROCESSING, progress=10, message="Generating 3D structure...")
        outputs = await _run_pipeline_generate_3d(image, arg)
        update_current_generation( progress=50, message="3D structure generated", outputs=outputs)

        # Generate final GLB
        update_current_generation( progress=70, message="Generating GLB file..." )
        await _run_pipeline_generate_glb(outputs, arg.mesh_simplify_ratio, arg.texture_size, apply_texture=arg.apply_texture)

        # Done
        update_current_generation( status=TaskStatus.COMPLETE, progress=100, message="Generation complete")

        # Clean up intermediate files, keep final model
        await cleanup_generation_files(keep_model=True)

        duration = time.time() - start_time  # Calculate duration before return
        logger.info(f"Generation completed in {duration:.2f} seconds")
        return GenerationResponse(
            status=TaskStatus.COMPLETE,
            progress=100,
            message="Generation complete",
            model_url="/download/model"  # single endpoint
        )
    except CancelledException as cex:
        # Cancel was triggered mid-generation
        update_current_generation( status=TaskStatus.FAILED, progress=0, message="Cancelled by user")
        await cleanup_generation_files()
        raise HTTPException(status_code=499, detail="Generation cancelled by user")
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(error_trace)
        update_current_generation( status=TaskStatus.FAILED, progress=0, message=str(e))
        await cleanup_generation_files()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        generation_lock.release()



@router.post("/generate_multi_no_preview", response_model=GenerationResponse)
async def generate_multi_no_preview(
    file_list: Optional[List[UploadFile]] = File(None),
    image_list_base64: Optional[List[str]] = Form(None),
    arg: GenerationArgForm = Depends(),
):
    """
    Generate a 3D model using multiple images, directly (no preview).
    The pipeline will receive [img1, img2, ...] as input.
    """
    print()#empty line, for an easier read.
    logger.info("Client asked to multi-view-generate")
    try:
        await asyncio.wait_for(generation_lock.acquire(), timeout=0.001)
    except asyncio.TimeoutError:
        raise HTTPException( status_code=503, detail="Server is busy with another generation")
    
    start_time = time.time() 
    # We have the lock => let's reset the "current_generation"
    reset_current_generation()

    try:
        _gen_3d_validate_params(file_list, image_list_base64, arg)
        # construct images:
        images = await _load_images_into_list(file_list, image_list_base64)

        update_current_generation(status=TaskStatus.PROCESSING, progress=10, message="Generating 3D structure...")
        outputs = await _run_pipeline_generate_3d(images, arg)
        update_current_generation(progress=50, message="3D structure generated", outputs=outputs)

        # 2) Generate final GLB
        update_current_generation(progress=70, message="Generating GLB file...")
        await _run_pipeline_generate_glb(outputs, arg.mesh_simplify_ratio, arg.texture_size, apply_texture=arg.apply_texture)

        # Done
        update_current_generation(status=TaskStatus.COMPLETE, progress=100, message="Generation complete")

        # Clean up intermediate files, keep final model
        await cleanup_generation_files(keep_model=True)

        duration = time.time() - start_time  # Calculate duration before return
        logger.info(f"Generation completed in {duration:.2f} seconds")
        return GenerationResponse(
            status=TaskStatus.COMPLETE,
            progress=100,
            message="Generation complete",
            model_url="/download/model"  # single endpoint
        )
    except CancelledException as cex:
        # Cancel was triggered mid-generation
        update_current_generation( status=TaskStatus.FAILED, progress=0, message="Cancelled by user")
        await cleanup_generation_files()
        raise HTTPException(status_code=499, detail="Generation cancelled by user")
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(error_trace)
        update_current_generation(status=TaskStatus.FAILED, progress=0, message=str(e))
        await cleanup_generation_files()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        generation_lock.release()



@router.post("/generate", response_model=GenerationResponse)
async def process_ui_generation_request(
    data: Dict
):
    """Process generation request from the UI panel and redirect to appropriate endpoint."""
    try:
        # Extract values from simplified input format
        arg = GenerationArgForm(
            seed = int(data.get("seed", 1234)),
            guidance_scale = data.get("guidance_scale", 5.0),
            num_inference_steps = int(data.get("num_inference_steps", 20)),
            octree_resolution = int(data.get("octree_resolution", 256)),
            num_chunks = int(data.get("num_chunks", 8000)),
            mesh_simplify_ratio = data.get("mesh_simplify", 0.95),
            apply_texture = data.get("apply_texture", False),
            texture_size = int(data.get("texture_size", 1024)),
            output_format = "glb"
        )
        # Get images from input
        images_base64 = data.get("single_multi_img_input", [])
        if not images_base64:
            raise HTTPException(status_code=400, detail="No images provided")

        # Always skip videos for now
        response = await generate_multi_no_preview(
            file_list=None,
            image_list_base64=images_base64,
            arg=arg
        )
        return response

    except Exception as e:
        logger.error(f"Error processing UI generation request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# for example:
#   "make_meshes_and_tex",
#   "retexture",
#   "retexture_via_masks"  etc (see api-documentation.html)
@router.get("/info/supported_operations")
async def get_supported_operation_types():
   return ["make_meshes_and_tex"]



@router.post("/interrupt")
async def interrupt_generation():
    """Interrupt the current generation if one is in progress."""
    logger.info("Client cancelled the generation.")
    if not is_generation_in_progress():
        return {"status": "no_generation_in_progress"}
    cancel_event.set()  # <-- Signal cancellation
    return {"status": "interrupt_requested"}
    


@router.get("/download/model")
async def download_model():
    """Download final 3D model (GLB)."""
    logger.info("Client is downloading a model.")
    model_path = file_manager.get_temp_path("model.glb")
    if not model_path.exists():
        logger.error(f"mesh not found")
        raise HTTPException(status_code=404, detail="Mesh not found")
    return FileResponse(
        str(model_path),
        media_type="model/gltf-binary",
        filename="model.glb"
    )

@router.get("/download/spz-ui-layout/generation-3d-panel")
async def get_generation_panel_layout():
    """Return the UI layout for the generation panel."""
    try:
        file_path = os.path.join(os.path.dirname(__file__), 'layout_generation_3d_panel.txt')
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # By using Response(content=content, media_type="text/plain; charset=utf-8")
            # - Bypass the automatic JSON encoding
            # - Explicitly tell the client this is plain text (not JSON)
            # - Ensure proper UTF-8 encoding is maintained
            # This way Unity receives the layout text exactly as it appears in the file.
            # It keeps proper line breaks and formatting intact, with special characters not being escaped:
            return Response(content=content,  media_type="text/plain; charset=utf-8")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Layout file not found")