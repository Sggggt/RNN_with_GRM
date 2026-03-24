"""Segment-memory models, CUDA kernels, and experiment utilities."""

from .core import (
    ParameterMixtureFusion,
    RetrievedStateFusion,
    SegmentRecurrentMemoryModel,
    SegmentRecurrentMemoryLayer,
    LinearMemoryCell,
    HierarchicalSegmentMemoryModel,
    TopKMemoryRetriever,
    cuda_apply_query,
    cuda_batched_memory_gather,
    cuda_chunk_update,
    build_retrieved_state_fusion,
    build_topk_memory_retriever,
    build_segment_memory_model,
    build_hierarchical_segment_memory_model,
    get_grm_cuda_runtime_status,
    is_grm_cpp_extension_available,
    is_grm_cuda_available,
    load_grm_cuda_extension,
)
from .utils import (
    MemoryArchitectureConfig,
    ExperimentTrainer,
    estimate_memory_footprint,
    build_context_length_config,
)

__version__ = "3.0.0"

__all__ = [
    "TopKMemoryRetriever",
    "RetrievedStateFusion",
    "ParameterMixtureFusion",
    "LinearMemoryCell",
    "SegmentRecurrentMemoryModel",
    "SegmentRecurrentMemoryLayer",
    "HierarchicalSegmentMemoryModel",
    "cuda_chunk_update",
    "cuda_apply_query",
    "cuda_batched_memory_gather",
    "load_grm_cuda_extension",
    "is_grm_cpp_extension_available",
    "is_grm_cuda_available",
    "get_grm_cuda_runtime_status",
    "build_topk_memory_retriever",
    "build_retrieved_state_fusion",
    "build_segment_memory_model",
    "build_hierarchical_segment_memory_model",
    "MemoryArchitectureConfig",
    "build_context_length_config",
    "estimate_memory_footprint",
    "ExperimentTrainer",
]
