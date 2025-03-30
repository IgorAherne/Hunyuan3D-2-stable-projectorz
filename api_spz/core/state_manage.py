import os
import sys
from contextlib import contextmanager
import logging
import torch
from pathlib import Path
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline, FloaterRemover, DegenerateFaceRemover, FaceReducer
from hy3dgen.texgen import Hunyuan3DPaintPipeline
from hy3dgen.rembg import BackgroundRemover


@contextmanager
def suppress_float16_cpu_warnings():
    """Selectively suppress only float16/CPU compatibility warnings"""
    import warnings
    import logging
    
    # Save original state
    diffusers_logger = logging.getLogger("diffusers")
    original_diffusers_level = diffusers_logger.level
    original_filters = warnings.filters.copy()
    
    try:
        # Add specific warning filter for float16/CPU warnings
        warnings.filterwarnings("ignore", 
                               message="Pipelines loaded with `dtype=torch.float16` cannot run with `cpu` device.*")
        
        # Increase diffusers logger level to suppress the same warning from their side
        diffusers_logger.setLevel(logging.ERROR)
        
        yield
    finally:
        # Restore original state
        warnings.filters = original_filters
        diffusers_logger.setLevel(original_diffusers_level)


logger = logging.getLogger("trellis")

class HunyuanState:
    def __init__(self):
        self.temp_dir = Path("temp")
        self.temp_dir.mkdir(exist_ok=True)
        self.pipeline = None
        self.texture_pipeline = None
        self.rembg = None
        self.floater_remover = None
        self.degenerate_face_remover = None
        self.face_reducer = None

    def cleanup(self):
        if self.temp_dir.exists():
            from shutil import rmtree
            rmtree(self.temp_dir)

    def _initialize_texture_pipeline(self, texgen_model_path, low_vram_mode=False):
        """Initialize texture pipeline with CPU device and memory optimizations"""
        try:
            # Import the modules we need to modify
            from hy3dgen.texgen import pipelines
            from hy3dgen.texgen.pipelines import Hunyuan3DTexGenConfig
            
            # Create a modified version with CPU device
            class CPUHunyuan3DTexGenConfig(Hunyuan3DTexGenConfig):
                def __init__(self, light_remover_ckpt_path, multiview_ckpt_path):
                    # Call parent init
                    super().__init__(light_remover_ckpt_path, multiview_ckpt_path)
                    # Override device to CPU
                    self.device = 'cpu'
            
            # Store original class for restoration
            original_config_class = pipelines.Hunyuan3DTexGenConfig
            
            # Monkey patch with our CPU version
            pipelines.Hunyuan3DTexGenConfig = CPUHunyuan3DTexGenConfig
            
            # Load pipeline with selective warning suppression
            with suppress_float16_cpu_warnings():
                self.texture_pipeline = Hunyuan3DPaintPipeline.from_pretrained(texgen_model_path)
            
            # Restore the original class
            pipelines.Hunyuan3DTexGenConfig = original_config_class
            
            # Apply CPU offloading for selective GPU usage
            if low_vram_mode and hasattr(self.texture_pipeline, 'enable_model_cpu_offload'):
                self.texture_pipeline.enable_model_cpu_offload()
                logger.info("Texture pipeline loaded with CPU offloading (low VRAM mode)")
                print("Texture pipeline loaded with CPU offloading (low VRAM mode)")
            else:
                logger.info("Texture pipeline loaded successfully")
                print("Texture pipeline loaded successfully")
            return True
        except Exception as e:
            # Ensure we restore the original class in case of error
            if 'original_config_class' in locals():
                pipelines.Hunyuan3DTexGenConfig = original_config_class
                
            logger.error(f"Failed to load texture pipeline: {e}")
            print(f"Failed to load texture pipeline: {e}")
            self.texture_pipeline = None
            return False
        
        
    def initialize_pipeline(self, 
                           model_path='tencent/Hunyuan3D-2mini', 
                           subfolder='hunyuan3d-dit-v2-mini-turbo',
                           texgen_model_path='tencent/Hunyuan3D-2',
                           device=None,
                           enable_flashvdm=True,
                           mc_algo='mc',
                           low_vram_mode=False):
        print(f"Initializing Hunyuan3D models from {model_path}/{subfolder}")
        
        # Store model path for reference
        self.model_path = model_path
        self.subfolder = subfolder
        
        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        
        # Store config values without computation
        self._enable_flashvdm = enable_flashvdm
        self._mc_algo = mc_algo
        self._low_vram_mode = low_vram_mode
        self._texgen_model_path = texgen_model_path
        # Initialize components
        self.rembg = BackgroundRemover()
        # Load shape model
        self.pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            model_path,
            subfolder=subfolder,
            use_safetensors=True,
            device="cpu" if low_vram_mode else device,  # Start on CPU when in low VRAM mode
        )
        if enable_flashvdm:
            self.pipeline.enable_flashvdm(mc_algo=mc_algo)
        
        # Post-processing utilities
        self.floater_remover = FloaterRemover()
        self.degenerate_face_remover = DegenerateFaceRemover()
        self.face_reducer = FaceReducer()
        
        # Initialize texture pipeline with extracted method
        self._initialize_texture_pipeline(texgen_model_path, low_vram_mode)
        
        # Basic memory logging
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 * 1024)
            logger.info(f"VRAM allocated at startup: {allocated:.1f}MB")
            print(f"VRAM allocated at startup: {allocated:.1f}MB")


    def _is_shape_model_on_cpu(self):
        """Check if the shape model exists but is currently on CPU"""
        if not hasattr(self, 'pipeline') or self.pipeline is None:
            return False
            
        # Check if pipeline is on CPU
        if hasattr(self.pipeline, 'device'):
            return str(self.pipeline.device) == 'cpu'
        else:
            # Try to detect if pipeline is on CPU by checking components
            for attr in ['model', 'vae', 'conditioner']:
                if hasattr(self.pipeline, attr):
                    component = getattr(self.pipeline, attr)
                    if hasattr(component, 'device') and 'cuda' in str(component.device):
                        return False
            return True  # Assume CPU if we can't detect GPU
        

    def _reload_shape_model(self):
        """Move shape generation model back to GPU for inference"""
        if hasattr(self, 'pipeline') and self.pipeline is not None:
            # Get the original device
            device = getattr(self, '_shape_model_device', 
                            "cuda" if torch.cuda.is_available() else "cpu")
            
            if device != 'cpu':  # Only move back if original device wasn't CPU
                logger.info(f"Moving shape generation model back to {device}")
                
                # Move model back to original device
                self.pipeline.to(device)
                
                logger.info(f"Shape generation model moved to {device}")
        else:
            # Fall back to full reload if model was completely unloaded
            logger.info("Shape model was fully unloaded, performing full reload")
            # Use the existing reload logic for this case
            if hasattr(self, '_pipeline_checkpoint'):
                # ... existing reload code ...
                self.pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                    self._pipeline_checkpoint['model_path'],
                    subfolder=self._pipeline_checkpoint['subfolder'],
                    use_safetensors=True,
                    device=self._pipeline_checkpoint.get('device', "cuda" if torch.cuda.is_available() else "cpu"),
                )
                
                # Restore settings
                if self._pipeline_checkpoint.get('enable_flashvdm', False):
                    self.pipeline.enable_flashvdm(
                        mc_algo=self._pipeline_checkpoint.get('mc_algo', 'mc')
                    )

    def ensure_shape_model_on_gpu(self):
        """Make sure the shape model is loaded and on GPU"""
        if not hasattr(self, 'pipeline') or self.pipeline is None:
            logger.info("Shape generation model needs to be reloaded")
            self._reload_shape_model()
        elif self._is_shape_model_on_cpu():
            logger.info("Shape generation model is on CPU, moving back to GPU")
            self._reload_shape_model()


    def unload_shape_model(self):
        """Move shape generation model to CPU to free GPU VRAM"""
        if hasattr(self, 'pipeline') and self.pipeline is not None:
            logger.info("Moving shape generation model to CPU to free ~6GB VRAM")
            
            # Store current device for later
            if hasattr(self.pipeline, 'device'):
                self._shape_model_device = self.pipeline.device
            else:
                self._shape_model_device = "cuda" if torch.cuda.is_available() else "cpu"
            # Move model to CPU instead of completely unloading
            self.pipeline.to('cpu')
            torch.cuda.empty_cache()
            
            
    # Helper functions need to be defined as proper methods with self
    @staticmethod
    @torch.no_grad()
    def _chunked_vae_decode(original_fn, latents, chunk_size=3, *args, **kwargs):
        """Memory-efficient VAE decoding that processes in chunks."""
        logger.info("Using chunked VAE decoding")
        
        # Clear cache before operation
        torch.cuda.empty_cache()
        
        # Single item - process normally
        batch_size = latents.shape[0]
        if batch_size <= chunk_size:
            return original_fn(latents, *args, **kwargs)
        
        # Process in chunks
        logger.info(f"Processing VAE decode in {batch_size} items with chunk size {chunk_size}")
        
        results = []
        for i in range(0, batch_size, chunk_size):
            # Get chunk of items
            end_idx = min(i + chunk_size, batch_size)
            chunk = latents[i:end_idx]
            
            # Process chunk
            chunk_result = original_fn(chunk, *args, **kwargs)
            
            # Handle different return types
            if isinstance(chunk_result, tuple):
                if i == 0:
                    # Initialize results list for tuple return
                    results = [[] for _ in range(len(chunk_result))]
                # Add each part of the tuple to its respective list
                for j, part in enumerate(chunk_result):
                    results[j].append(part)
            else:
                results.append(chunk_result)
            
            # Clear cache after each chunk
            torch.cuda.empty_cache()
        
        # Combine results
        if isinstance(results[0], list):
            # For tuple returns, concatenate each part separately
            return tuple(torch.cat(part_list) for part_list in results)
        else:
            # For single tensor returns
            return torch.cat(results)


    @staticmethod
    @torch.no_grad()
    def _chunked_encode_images(original_fn, images, chunk_size=2):
        """Memory-efficient image encoding that processes in chunks."""
        logger.info("Using chunked image encoding")
        
        # Clear cache before operation
        torch.cuda.empty_cache()
        
        # Single batch or small batch - process normally
        batch_size = images.shape[0]
        if batch_size <= chunk_size:
            return original_fn(images)
        
        # Process in chunks
        logger.info(f"Processing image encoding with chunk size {chunk_size}")
        
        results = []
        for i in range(0, batch_size, chunk_size):
            # Get chunk of images
            end_idx = min(i + chunk_size, batch_size)
            chunk = images[i:end_idx]
            
            # Encode chunk
            chunk_result = original_fn(chunk)
            results.append(chunk_result)
            # Clear cache
            torch.cuda.empty_cache()
        
        # Combine results
        return torch.cat(results)

    def optimize_texture_pipeline(self, chunk_size=3):
        """Ultra-high quality memory optimization for texture pipeline with adjustable chunk size.
        
        Args:
            chunk_size: Number of items to process at once. Higher values = faster but more memory.
                      Default is 3 which balances speed and memory usage.
        """
        if not hasattr(self, 'texture_pipeline') or self.texture_pipeline is None:
            return
            
        # Only apply once
        if getattr(self.texture_pipeline, '_memory_optimized_marker', False):
            logger.info("Optimization already applied - skipping")
            return
            
        logger.info(f"Applying ultra-high quality memory optimization (chunk size: {chunk_size})")
        
        # Access the multiview model
        if 'multiview_model' not in self.texture_pipeline.models:
            logger.warning("No multiview_model found - optimization not applied")
            return
            
        multiview_model = self.texture_pipeline.models['multiview_model']
        
        # Check if the pipeline exists
        if not hasattr(multiview_model, 'pipeline'):
            logger.warning("multiview_model has no pipeline - optimization not applied")
            return
            
        pipeline = multiview_model.pipeline
        
        # 1. Memory-efficient VAE decode
        if hasattr(pipeline.vae, 'decode'):
            original_vae_decode = pipeline.vae.decode
            
            # Create a method that references our helper function
            def optimized_vae_decode(self, *args, **kwargs):
                return HunyuanState._chunked_vae_decode(original_vae_decode, *args, chunk_size=chunk_size, **kwargs)
            
            # Apply the patch
            import types
            pipeline.vae.decode = types.MethodType(optimized_vae_decode, pipeline.vae)
            logger.info(f"Applied chunked processing to VAE decoder (chunk size: {chunk_size})")
        
        # 2. Memory-efficient image encoding
        if hasattr(pipeline, 'encode_images'):
            original_encode_images = pipeline.encode_images
            
            # Create a method that references our helper function
            def optimized_encode_images(self, images):
                return HunyuanState._chunked_encode_images(original_encode_images, images, chunk_size=chunk_size)
            
            # Apply the patch
            pipeline.encode_images = types.MethodType(optimized_encode_images, pipeline)
            logger.info(f"Applied chunked processing to image encoding (chunk size: {chunk_size})")
        
        # Mark as optimized
        self.texture_pipeline._memory_optimized_marker = True
        
        logger.info("Ultra-high quality memory optimization applied")

# Global state instance
state = HunyuanState()