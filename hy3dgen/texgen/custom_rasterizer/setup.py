from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch
import os
import sysconfig

# Add proper include directories
include_dirs = [
    os.path.join(os.path.dirname(torch.__file__), 'include'),
    os.path.join(os.path.dirname(torch.__file__), 'include', 'torch', 'csrc', 'api', 'include'),
    os.path.join(os.path.dirname(torch.__file__), 'include', 'TH'),
    os.path.join(os.path.dirname(torch.__file__), 'include', 'THC'),
]

# Add Python include path
python_include = sysconfig.get_path('include')
include_dirs.append(python_include)

# Multiple GPU architectures support
nvcc_flags = [
    '--allow-unsupported-compiler',
    '--expt-relaxed-constexpr',
    '-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH',
    '-Xcompiler', '/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH',
    # Pascal
    '-gencode=arch=compute_61,code=sm_61',  # GTX 1080, etc.
    # Volta
    '-gencode=arch=compute_70,code=sm_70',  # V100, Titan V
    # Turing
    '-gencode=arch=compute_75,code=sm_75',  # RTX 2080, etc.
    # Ampere
    '-gencode=arch=compute_80,code=sm_80',  # A100, etc.
    '-gencode=arch=compute_86,code=sm_86',  # RTX 3090, etc.
    # Ada Lovelace
    '-gencode=arch=compute_89,code=sm_89',  # RTX 4090, etc.
    # Blackwell
    '-gencode=arch=compute_90,code=sm_90',  # RTX 5000 series, etc.
    # Update PTX for forward compatibility
    '-gencode=arch=compute_90,code=compute_90',
    # PTX for forward compatibility
    '-gencode=arch=compute_90,code=compute_90',
]

# build custom rasterizer
custom_rasterizer_module = CUDAExtension(
    'custom_rasterizer_kernel', 
    [
        'lib/custom_rasterizer_kernel/rasterizer.cpp',
        'lib/custom_rasterizer_kernel/grid_neighbor.cpp',
        'lib/custom_rasterizer_kernel/rasterizer_gpu.cu',
    ],
    include_dirs=include_dirs,
    extra_compile_args={
        'cxx': ['/bigobj', '/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH'] if os.name == 'nt' else [],
        'nvcc': nvcc_flags
    }
)

setup(
    packages=find_packages(),
    version='0.1',
    name='custom_rasterizer',
    author="Modified from Tencent Hunyuan3D",
    description="Custom rasterizer for Hunyuan3D",
    include_package_data=True,
    package_dir={'': '.'},
    ext_modules=[
        custom_rasterizer_module,
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)