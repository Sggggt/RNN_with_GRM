from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME


ROOT = Path(__file__).parent.resolve()


def _has_nvcc() -> bool:
    if not CUDA_HOME:
        return False
    nvcc_path = Path(CUDA_HOME) / "bin" / "nvcc"
    return nvcc_path.exists()


def _nvcc_version() -> str:
    if not _has_nvcc():
        return ""
    nvcc_path = Path(CUDA_HOME) / "bin" / "nvcc"
    try:
        output = subprocess.check_output([str(nvcc_path), "--version"], text=True)
    except Exception:
        return ""
    match = re.search(r"release\s+(\d+\.\d+)", output)
    return match.group(1) if match else ""


def _torch_cuda_version() -> str:
    return str(torch.version.cuda or "")


def _has_compatible_cuda_toolchain() -> bool:
    nvcc_version = _nvcc_version()
    torch_cuda_version = _torch_cuda_version()
    if not nvcc_version or not torch_cuda_version:
        return False
    nvcc_major_minor = ".".join(nvcc_version.split(".")[:2])
    torch_major_minor = ".".join(torch_cuda_version.split(".")[:2])
    return nvcc_major_minor == torch_major_minor


def _build_extension():
    use_cuda = _has_compatible_cuda_toolchain()
    sources = [
        str(ROOT / "grm" / "csrc" / "grm_cuda.cpp"),
        str(ROOT / "grm" / "csrc" / "bindings.cpp"),
    ]
    extra_compile_args = {"cxx": ["-O3"]}
    define_macros = [("GRM_CUDA_EXT_WITH_KERNELS", "1" if use_cuda else "0")]
    extension_cls = CppExtension

    if use_cuda:
        extension_cls = CUDAExtension
        sources.extend(
            [
                str(ROOT / "grm" / "csrc" / "chunk_update.cu"),
                str(ROOT / "grm" / "csrc" / "apply_query.cu"),
                str(ROOT / "grm" / "csrc" / "batched_gather.cu"),
            ]
        )
        extra_compile_args["nvcc"] = ["-O3", "--use_fast_math"]

    return extension_cls(
        name="grm_cuda_ext",
        sources=sources,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
    )


setup(
    name="grm_cuda_ext",
    version="0.1.0",
    description="GRM custom op extension scaffold",
    ext_modules=[_build_extension()],
    cmdclass={"build_ext": BuildExtension},
)
