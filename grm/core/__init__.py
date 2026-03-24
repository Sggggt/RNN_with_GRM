"""Public core API for the active segment-memory recurrent implementation."""

from .aggregator import ParameterMixtureFusion, RetrievedStateFusion, build_retrieved_state_fusion
from .cuda_ops import (
    cuda_apply_query,
    cuda_batched_memory_gather,
    cuda_chunk_update,
    get_grm_cuda_runtime_status,
    is_grm_cpp_extension_available,
    is_grm_cuda_available,
    load_grm_cuda_extension,
)
from .gating_unit import TopKMemoryRetriever, build_topk_memory_retriever
from .linear_attention import LinearMemoryCell
from .multilayer_grm import SegmentRecurrentMemoryLayer, HierarchicalSegmentMemoryModel, build_hierarchical_segment_memory_model
from .paper_grm import SegmentRecurrentMemoryModel, build_segment_memory_model

__all__ = [
    "RetrievedStateFusion",
    "ParameterMixtureFusion",
    "build_retrieved_state_fusion",
    "cuda_chunk_update",
    "cuda_apply_query",
    "cuda_batched_memory_gather",
    "load_grm_cuda_extension",
    "is_grm_cpp_extension_available",
    "is_grm_cuda_available",
    "get_grm_cuda_runtime_status",
    "TopKMemoryRetriever",
    "build_topk_memory_retriever",
    "LinearMemoryCell",
    "SegmentRecurrentMemoryModel",
    "build_segment_memory_model",
    "SegmentRecurrentMemoryLayer",
    "HierarchicalSegmentMemoryModel",
    "build_hierarchical_segment_memory_model",
]
