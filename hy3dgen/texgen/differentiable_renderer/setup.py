from setuptools import setup, Extension
import pybind11
import sys
import platform
import sysconfig
import os

def get_platform_specific_args():
    system = platform.system().lower()
    cpp_std = 'c++14'
    
    if sys.platform == 'win32':
        compile_args = ['/O2', f'/std:{cpp_std}', '/EHsc', '/MP', '/DWIN32_LEAN_AND_MEAN', '/bigobj', '/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH']
        link_args = []
        extra_includes = []
    elif system == 'linux':
        compile_args = ['-O3', f'-std={cpp_std}', '-fPIC', '-Wall', '-Wextra', '-pthread']
        link_args = ['-fPIC', '-pthread']
        extra_includes = []
    elif sys.platform == 'darwin':
        compile_args = ['-O3', f'-std={cpp_std}', '-fPIC', '-Wall', '-Wextra',
                       '-stdlib=libc++', '-mmacosx-version-min=10.14']
        link_args = ['-fPIC', '-stdlib=libc++', '-mmacosx-version-min=10.14', '-dynamiclib']
        extra_includes = []
    else:
        raise RuntimeError(f"Unsupported platform: {system}")
    
    return compile_args, link_args, extra_includes

extra_compile_args, extra_link_args, platform_includes = get_platform_specific_args()

# Add Python include path explicitly
python_include = sysconfig.get_path('include')
include_dirs = [
    pybind11.get_include(), 
    pybind11.get_include(user=True),
    python_include
]
include_dirs.extend(platform_includes)

ext_modules = [
    Extension(
        "mesh_processor",
        ["mesh_processor.cpp"],
        include_dirs=include_dirs,
        language='c++',
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

setup(
    name="mesh_processor",
    version="0.1.0",
    author="Modified from Tencent Hunyuan3D",
    description="Mesh processor for Hunyuan3D",
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.6.0'],
    python_requires='>=3.6',
)