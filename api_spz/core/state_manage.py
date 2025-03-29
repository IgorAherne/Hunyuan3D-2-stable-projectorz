import torch
from pathlib import Path
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline, FloaterRemover, DegenerateFaceRemover, FaceReducer
from hy3dgen.texgen import Hunyuan3DPaintPipeline
from hy3dgen.rembg import BackgroundRemover

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
        
        # Initialize components
        self.rembg = BackgroundRemover()
        
        self.pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            model_path,
            subfolder=subfolder,
            use_safetensors=True,
            device=device,
        )
        if enable_flashvdm:
            self.pipeline.enable_flashvdm(mc_algo=mc_algo)
        
        # Post-processing utilities
        self.floater_remover = FloaterRemover()
        self.degenerate_face_remover = DegenerateFaceRemover()
        self.face_reducer = FaceReducer()
        
        # Always try to load texture pipeline, but with CPU offloading in low VRAM mode
        try:
            self.texture_pipeline = Hunyuan3DPaintPipeline.from_pretrained(texgen_model_path)
            if low_vram_mode:
                self.texture_pipeline.enable_model_cpu_offload()
                print("Texture pipeline loaded with CPU offloading (low VRAM mode)")
            else:
                print("Texture pipeline loaded successfully")
        except Exception as e:
            print(f"Failed to load texture pipeline: {e}")
            self.texture_pipeline = None


# Global state instance
state = HunyuanState()