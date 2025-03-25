[中文阅读](README_zh_cn.md)
[日本語で読む](README_ja_jp.md)

# Hunyuan3D-2GP: 3D Generation for the GPU Poor
*GPU Poor version by **DeepBeepMeep**. This great video generator can now run smoothly with less than 6 GB of VRAM.*
<BR>

This is another integration of the *mmgp* module that allows easy to setup advanced and fast offloading.\
https://github.com/deepbeepmeep/mmgp





## 🔥 News
- Mar 18, 2025: 💬 Hunyuan3D-2.0GP by Deepbeepmeep: Support for Hunyuan3D turbo models
- Mar 18, 2025: 💬 Hunyuan3D-2.0GP by Deepbeepmeep: Support for Hunyuan3D-2mv and  Hunyuan3D-2mini
- Jan 25, 2025: 💬 Hunyuan3D-2.0GP by Deepbeepmeep: Synced code with original repo.Many thanks to YanWenKun for the work.
- Jan 23, 2025: 💬 Hunyuan3D-2.0GP by Deepbeepmeep: added lighning fix in rendering window
- Jan 23, 2025: 💬 Hunyuan3D-2.0GP by Deepbeepmeep: added Windows support thanks to MrForExample and sdbds + omitted optimization that keeps under VRAM 6GB with profile 4 or 5
- Jan 22, 2025: 💬 Hunyuan3D-2.0GP by Deepbeepmeep: low VRAM support and unlocked text to 3D generator
- Jan 21, 2025: 💬 Release [Hunyuan3D 2.0](https://huggingface.co/spaces/tencent/Hunyuan3D-2). Please give it a try!

- Mar 19, 2025: 🤗 Release turbo model [Hunyuan3D-2-Turbo](https://huggingface.co/tencent/Hunyuan3D-2/), [Hunyuan3D-2mini-Turbo](https://huggingface.co/tencent/Hunyuan3D-2mini/) and [FlashVDM](https://github.com/Tencent/FlashVDM).
- Mar 18, 2025: 🤗 Release multiview shape model [Hunyuan3D-2mv](https://huggingface.co/tencent/Hunyuan3D-2mv) and 0.6B
  shape model [Hunyuan3D-2mini](https://huggingface.co/tencent/Hunyuan3D-2mini).
- Feb 14, 2025: 🛠️ Release texture enhancement module, please obtain high-definition textures
  via [here](minimal_demo.py)!
- Feb 3, 2025: 🐎
  Release [Hunyuan3D-DiT-v2-0-Fast](https://huggingface.co/tencent/Hunyuan3D-2/tree/main/hunyuan3d-dit-v2-0-fast), our
  guidance distillation model that could half the dit inference time, see [here](minimal_demo.py) for usage.
- Jan 27, 2025: 🛠️ Release Blender addon for Hunyuan3D 2.0, Check it out [here](#blender-addon).
- Jan 23, 2025: 💬 We thank community members for
  creating [Windows installation tool](https://github.com/YanWenKun/Hunyuan3D-2-WinPortable), ComfyUI support
  with [ComfyUI-Hunyuan3DWrapper](https://github.com/kijai/ComfyUI-Hunyuan3DWrapper)
  and [ComfyUI-3D-Pack](https://github.com/MrForExample/ComfyUI-3D-Pack) and other
  awesome [extensions](#community-resources).
- Jan 21, 2025: 💬 Enjoy exciting 3D generation on our website [Hunyuan3D Studio](https://3d.hunyuan.tencent.com)!
- Jan 21, 2025: 🤗 Release inference code and pretrained models
  of [Hunyuan3D 2.0](https://huggingface.co/tencent/Hunyuan3D-2). Please give it a try
  via [huggingface space](https://huggingface.co/spaces/tencent/Hunyuan3D-2) and
  our [official site](https://3d.hunyuan.tencent.com)!

## **Abstract**

We present Hunyuan3D 2.0, an advanced large-scale 3D synthesis system for generating high-resolution textured 3D assets.
This system includes two foundation components: a large-scale shape generation model - Hunyuan3D-DiT, and a large-scale
texture synthesis model - Hunyuan3D-Paint.
The shape generative model, built on a scalable flow-based diffusion transformer, aims to create geometry that properly
aligns with a given condition image, laying a solid foundation for downstream applications.
The texture synthesis model, benefiting from strong geometric and diffusion priors, produces high-resolution and vibrant
texture maps for either generated or hand-crafted meshes.
Furthermore, we build Hunyuan3D-Studio - a versatile, user-friendly production platform that simplifies the re-creation
process of 3D assets. It allows both professional and amateur users to manipulate or even animate their meshes
efficiently.
We systematically evaluate our models, showing that Hunyuan3D 2.0 outperforms previous state-of-the-art models,
including the open-source models and closed-source models in geometry details, condition alignment, texture quality, and
e.t.c.

## How to run the Gradio app
1) Follow the installation instructions below

2) Enter either one of the commande lines in bash session
 
To run the Hunyuan3D-2mini (low VRAM) image to 3D generator:
```bash
python gradio_app.py 
```

To run the Hunyuan3D-2mv (multi views) image to 3D generator:
```bash
python gradio_app.py --mv
```

To run the text to 3D generator (an extension of the mini generator):
```bash
python gradio_app.py --enable_t23d
```

To run the original Hunyuan3D-2 image to 3D generator:
```bash
python gradio_app.py --h2
```

To use the Turbo version of one specific model, add *--turbo*. For instance to run the turbo Hunyuan3D-2mv (multi views) image to 3D generator:
```bash
python gradio_app.py --mv --turbo
```



By default the memory profile assumes 9 GB of VRAM *(profile 3)*. If you have less but at least 6 GB of VRAM add *--profile 4*

To run the image to 3D generator with optimized memory management:
```bash
python gradio_app.py --profile 3

```
To run the text to 3D generator with optimized memory management:
```bash
python gradio_app.py --enable_t23d --profile 4

```

You can choose between 5 profiles depending on your hardware:
- HighRAM_HighVRAM  (1) 
- HighRAM_LowVRAM  (2)
- LowRAM_HighVRAM  (3)
- LowRAM_LowVRAM  (4)
- VerylowRAM_LowVRAM  (5) 

Each profile's name describes the targeted level of RAM and VRAM consumptions.\
Usualy the lower the profile number the faster the generation.

## Other GPU Poor Applications

- Wan2GP: https://github.com/deepbeepmeep/Wan2GP :\
Another great 3D Image to Video and Text to Video generator. It can run on very low config as one its models is only 1.5 B parameters

- HuanyuanVideoGP: https://github.com/deepbeepmeep/HunyuanVideoGP :\
One of the best open source Text to Video generator

- FluxFillGP: https://github.com/deepbeepmeep/FluxFillGP :\
One of the best inpainting / outpainting tools based on Flux that can run with less than 12 GB of VRAM.

- Cosmos1GP: https://github.com/deepbeepmeep/Cosmos1GP :\
This application include two models: a text to world generator and a image / video to world (probably the best open source image to video generator).

- OminiControlGP: https://github.com/deepbeepmeep/OminiControlGP :\
A Flux derived application very powerful that can be used to transfer an object of your choice in a prompted scene. With mmgp you can run it with only 6 GB of VRAM.

- YuE GP: https://github.com/deepbeepmeep/YuEGP :\
A great song generator (instruments + singer's voice) based on prompted Lyrics and a genre description. Thanks to mmgp you can run it with less than 10 GB of VRAM without waiting forever.



### Architecture

Hunyuan3D 2.0 features a two-stage generation pipeline, starting with the creation of a bare mesh, followed by the
synthesis of a texture map for that mesh. This strategy is effective for decoupling the difficulties of shape and
texture generation and also provides flexibility for texturing either generated or handcrafted meshes.

<p align="left">
  <img src="assets/images/arch.jpg">
</p>

### Performance

We have evaluated Hunyuan3D 2.0 with other open-source as well as close-source 3d-generation methods.
The numerical results indicate that Hunyuan3D 2.0 surpasses all baselines in the quality of generated textured 3D assets
and the condition following ability.

| Model                   | CMMD(⬇)   | FID_CLIP(⬇) | FID(⬇)      | CLIP-score(⬆) |
|-------------------------|-----------|-------------|-------------|---------------|
| Top Open-source Model1  | 3.591     | 54.639      | 289.287     | 0.787         |
| Top Close-source Model1 | 3.600     | 55.866      | 305.922     | 0.779         |
| Top Close-source Model2 | 3.368     | 49.744      | 294.628     | 0.806         |
| Top Close-source Model3 | 3.218     | 51.574      | 295.691     | 0.799         |
| Hunyuan3D 2.0           | **3.193** | **49.165**  | **282.429** | **0.809**     |

Generation results of Hunyuan3D 2.0:
<p align="left">
  <img src="assets/images/e2e-1.gif"  height=300>
  <img src="assets/images/e2e-2.gif"  height=300>
</p>

## 🎁 Models Zoo

It takes 6 GB VRAM for shape generation and 16 GB for shape and texture generation in total.

Hunyuan3D-2mini Series

| Model                       | Description                   | Date       | Size | Huggingface                                                                                      |
|-----------------------------|-------------------------------|------------|------|--------------------------------------------------------------------------------------------------|
| Hunyuan3D-DiT-v2-mini-Turbo | Step Distillation Version     | 2025-03-19 | 0.6B | [Download](https://huggingface.co/tencent/Hunyuan3D-2mini/tree/main/hunyuan3d-dit-v2-mini-turbo) |
| Hunyuan3D-DiT-v2-mini-Fast  | Guidance Distillation Version | 2025-03-18 | 0.6B | [Download](https://huggingface.co/tencent/Hunyuan3D-2mini/tree/main/hunyuan3d-dit-v2-mini-fast)  |
| Hunyuan3D-DiT-v2-mini       | Mini Image to Shape Model     | 2025-03-18 | 0.6B | [Download](https://huggingface.co/tencent/Hunyuan3D-2mini/tree/main/hunyuan3d-dit-v2-mini)       |


Hunyuan3D-2mv Series

| Model                     | Description                    | Date       | Size | Huggingface                                                                                  |
|---------------------------|--------------------------------|------------|------|----------------------------------------------------------------------------------------------| 
| Hunyuan3D-DiT-v2-mv-Turbo | Step Distillation Version      | 2025-03-19 | 1.1B | [Download](https://huggingface.co/tencent/Hunyuan3D-2mv/tree/main/hunyuan3d-dit-v2-mv-turbo) |
| Hunyuan3D-DiT-v2-mv-Fast  | Guidance Distillation Version  | 2025-03-18 | 1.1B | [Download](https://huggingface.co/tencent/Hunyuan3D-2mv/tree/main/hunyuan3d-dit-v2-mv-fast)  |
| Hunyuan3D-DiT-v2-mv       | Multiview Image to Shape Model | 2025-03-18 | 1.1B | [Download](https://huggingface.co/tencent/Hunyuan3D-2mv/tree/main/hunyuan3d-dit-v2-mv)       |

Hunyuan3D-2 Series

| Model                    | Description                 | Date       | Size | Huggingface                                                                               |
|--------------------------|-----------------------------|------------|------|-------------------------------------------------------------------------------------------| 
| Hunyuan3D-DiT-v2-0-Turbo | Step Distillation Model     | 2025-03-19 | 1.1B | [Download](https://huggingface.co/tencent/Hunyuan3D-2/tree/main/hunyuan3d-dit-v2-0-turbo) |
| Hunyuan3D-DiT-v2-0-Fast  | Guidance Distillation Model | 2025-02-03 | 1.1B | [Download](https://huggingface.co/tencent/Hunyuan3D-2/tree/main/hunyuan3d-dit-v2-0-fast)  |
| Hunyuan3D-DiT-v2-0       | Image to Shape Model        | 2025-01-21 | 1.1B | [Download](https://huggingface.co/tencent/Hunyuan3D-2/tree/main/hunyuan3d-dit-v2-0)       |
| Hunyuan3D-Paint-v2-0     | Texture Generation Model    | 2025-01-21 | 1.3B | [Download](https://huggingface.co/tencent/Hunyuan3D-2/tree/main/hunyuan3d-paint-v2-0)     |
| Hunyuan3D-Delight-v2-0   | Image Delight Model         | 2025-01-21 | 1.3B | [Download](https://huggingface.co/tencent/Hunyuan3D-2/tree/main/hunyuan3d-delight-v2-0)   | 

## 🤗 Get Started with Hunyuan3D 2.0

Hunyuan3D 2.0 supports Macos, Windows, Linux. You may follow the next steps to use Hunyuan3D 2.0 via:

- [Code](#code-usage)
- [Gradio App](#gradio-app)
- [API Server](#api-server)
- [Blender Addon](#blender-addon)
- [Official Site](#official-site)

### Install Requirements
To use the application on Windows (without WSL) you will need to install Microsoft Visual Studio 2022 or later. If you get an error during the execution of onr of the *python setup.py* below you will need to set the path to the C++ compiler by running the following script (once you have located the installation path of VS Studio which may differ):
```bash
"C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd" -arch=x64 
```


In any case please make sure you have Python 3.10 installed, you may create a conda environnment: 
```bash
conda create -n Hunyuan3D-2GP python==3.10.9 
```
Then install the required libraries:
```bash
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cu124
pip install -r requirements.txt
# for texture
cd hy3dgen/texgen/custom_rasterizer
python3 setup.py install
cd ../../..
cd hy3dgen/texgen/differentiable_renderer
python3 setup.py install
```

### Code Usage

We designed a diffusers-like API to use our shape generation model - Hunyuan3D-DiT and texture synthesis model -
Hunyuan3D-Paint.

You could assess **Hunyuan3D-DiT** via:

```python
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2')
mesh = pipeline(image='assets/demo.png')[0]
```

The output mesh is a [trimesh object](https://trimesh.org/trimesh.html), which you could save to glb/obj (or other
format) file.

For **Hunyuan3D-Paint**, do the following:

```python
from hy3dgen.texgen import Hunyuan3DPaintPipeline
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

# let's generate a mesh first
pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2')
mesh = pipeline(image='assets/demo.png')[0]

pipeline = Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2')
mesh = pipeline(mesh, image='assets/demo.png')
```

Please visit [examples](examples) folder for more advanced usage, such as **multiview image to 3D generation** and *
*texture generation
for handcrafted mesh**.



### API Server

You could launch an API server locally, which you could post web request for Image/Text to 3D, Texturing existing mesh,
and e.t.c.

```bash
python api_server.py --host 0.0.0.0 --port 8080
```

A demo post request for image to 3D without texture.

```bash
img_b64_str=$(base64 -i assets/demo.png)
curl -X POST "http://localhost:8080/generate" \
     -H "Content-Type: application/json" \
     -d '{
           "image": "'"$img_b64_str"'",
         }' \
     -o test2.glb
```

### Blender Addon

With an API server launched, you could also directly use Hunyuan3D 2.0 in your blender with
our [Blender Addon](blender_addon.py). Please follow our tutorial to install and use.

https://github.com/user-attachments/assets/8230bfb5-32b1-4e48-91f4-a977c54a4f3e

### Official Site

Don't forget to visit [Hunyuan3D](https://3d.hunyuan.tencent.com) for quick use, if you don't want to host yourself.

## 📑 Open-Source Plan

- [x] Inference Code
- [x] Model Checkpoints
- [ ] ComfyUI
- [ ] TensorRT Version

## 🔗 BibTeX

If you found this repository helpful, please cite our report:

```bibtex
@misc{hunyuan3d22025tencent,
    title={Hunyuan3D 2.0: Scaling Diffusion Models for High Resolution Textured 3D Assets Generation},
    author={Tencent Hunyuan3D Team},
    year={2025},
    eprint={2501.12202},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}

@misc{yang2024hunyuan3d,
    title={Hunyuan3D 1.0: A Unified Framework for Text-to-3D and Image-to-3D Generation},
    author={Tencent Hunyuan3D Team},
    year={2024},
    eprint={2411.02293},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## Community Resources

Thanks for the contributions of community members, here we have these great extensions of Hunyuan3D 2.0:

- [ComfyUI-3D-Pack](https://github.com/MrForExample/ComfyUI-3D-Pack)
- [ComfyUI-Hunyuan3DWrapper](https://github.com/kijai/ComfyUI-Hunyuan3DWrapper)
- [Hunyuan3D-2-for-windows](https://github.com/sdbds/Hunyuan3D-2-for-windows)
- [📦 A bundle for running on Windows | 整合包](https://github.com/YanWenKun/Hunyuan3D-2-WinPortable)
- [Hunyuan3D-2GP](https://github.com/deepbeepmeep/Hunyuan3D-2GP)
- [Kaggle Notebook](https://github.com/darkon12/Hunyuan3D-2GP_Kaggle)

## Acknowledgements

We would like to thank the contributors to
the [Trellis](https://github.com/microsoft/TRELLIS),  [DINOv2](https://github.com/facebookresearch/dinov2), [Stable Diffusion](https://github.com/Stability-AI/stablediffusion), [FLUX](https://github.com/black-forest-labs/flux), [diffusers](https://github.com/huggingface/diffusers), [HuggingFace](https://huggingface.co), [CraftsMan3D](https://github.com/wyysf-98/CraftsMan3D),
and [Michelangelo](https://github.com/NeuralCarver/Michelangelo/tree/main) repositories, for their open research and
exploration.

## Star History

<a href="https://star-history.com/#Tencent/Hunyuan3D-2&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=Tencent/Hunyuan3D-2&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=Tencent/Hunyuan3D-2&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=Tencent/Hunyuan3D-2&type=Date" />
 </picture>
</a>
