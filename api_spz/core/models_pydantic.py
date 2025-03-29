from enum import Enum
from typing import Optional, Dict
from fastapi import Form
from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    PROCESSING = "PROCESSING"
    PREVIEW_READY = "PREVIEW_READY"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


class GenerationArgForm:
    def __init__(
        self,
        seed: int = Form(1234),
        guidance_scale: float = Form(5.0),
        num_inference_steps: int = Form(20),
        octree_resolution: int = Form(256),
        num_chunks: int = Form(8000),
        mesh_simplify_ratio: float = Form(0.95),
        apply_texture: bool = Form(False),
        texture_size: int = Form(1024),
        output_format: str = Form("glb"),
    ):
        self.seed = seed
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.octree_resolution = octree_resolution
        self.num_chunks = num_chunks
        self.mesh_simplify_ratio = mesh_simplify_ratio
        self.apply_texture = apply_texture
        self.texture_size = texture_size
        self.output_format = output_format


class GenerationResponse(BaseModel):
    # No task_id anymore, we focus on a single generation
    status: TaskStatus
    progress: int = 0
    message: str = ""
    # Only used if we did "generate_preview"
    preview_urls: Optional[Dict[str, str]] = None
    # Only used if generation is complete
    model_url: Optional[str] = None


class StatusResponse(BaseModel):
    status: TaskStatus
    progress: int
    message: str
    busy: bool