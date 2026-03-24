"""Lazy-loaded GRM C++/CUDA op wrappers with PyTorch fallbacks."""

from __future__ import annotations

import importlib
import math
import os
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor


_IMPORT_CANDIDATES = ("grm_cuda_ext", "grm.grm_cuda_ext")
_REGISTERED_OPS = (
    "chunk_update_forward",
    "chunk_update_backward",
    "apply_query_forward",
    "apply_query_backward",
    "batched_memory_gather",
)

_EXTENSION_MODULE: Optional[Any] = None
_EXTENSION_LOAD_ATTEMPTED = False
_EXTENSION_LOAD_ERROR = ""
_KERNEL_POLICY_ENV = "GRM_CUDA_KERNEL_POLICY"
_KERNEL_POLICY_OP_ENVS = {
    "chunk_update": "GRM_CUDA_CHUNK_UPDATE_POLICY",
    "apply_query": "GRM_CUDA_APPLY_QUERY_POLICY",
    "batched_memory_gather": "GRM_CUDA_BATCHED_GATHER_POLICY",
}
_VALID_KERNEL_POLICIES = {"auto", "native", "fallback"}


def _has_registered_ops() -> bool:
    namespace = getattr(torch.ops, "grm_cuda", None)
    if namespace is None:
        return False
    return all(hasattr(namespace, op_name) for op_name in _REGISTERED_OPS)


def _has_dispatch_kernel(op_name: str, dispatch_key: str) -> bool:
    checker = getattr(torch._C, "_dispatch_has_kernel_for_dispatch_key", None)
    if checker is None:
        return False
    try:
        return bool(checker(f"grm_cuda::{op_name}", dispatch_key))
    except Exception:
        return False


def _get_dispatch_kernel_status() -> Dict[str, Dict[str, bool]]:
    return {
        op_name: {
            "cuda": _has_dispatch_kernel(op_name, "CUDA"),
            "composite_explicit_autograd": _has_dispatch_kernel(op_name, "CompositeExplicitAutograd"),
        }
        for op_name in _REGISTERED_OPS
    }


def load_grm_cuda_extension(force: bool = False) -> bool:
    global _EXTENSION_MODULE, _EXTENSION_LOAD_ATTEMPTED, _EXTENSION_LOAD_ERROR

    if _EXTENSION_MODULE is not None and not force:
        return True
    if _EXTENSION_LOAD_ATTEMPTED and not force:
        return False

    _EXTENSION_LOAD_ATTEMPTED = True
    _EXTENSION_MODULE = None
    _EXTENSION_LOAD_ERROR = ""

    errors = []
    for module_name in _IMPORT_CANDIDATES:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - exercised only when extension import fails
            errors.append(f"{module_name}: {type(exc).__name__}: {exc}")
            continue
        if not _has_registered_ops():
            errors.append(f"{module_name}: torch.ops.grm_cuda registration missing")
            continue
        _EXTENSION_MODULE = module
        _EXTENSION_LOAD_ERROR = ""
        return True

    _EXTENSION_LOAD_ERROR = "; ".join(errors)
    return False


def is_grm_cpp_extension_available() -> bool:
    return load_grm_cuda_extension()


def _get_kernel_policy(op_name: str) -> str:
    env_names = (_KERNEL_POLICY_OP_ENVS.get(op_name, ""), _KERNEL_POLICY_ENV)
    for env_name in env_names:
        if not env_name:
            continue
        value = os.environ.get(env_name, "").strip().lower()
        if not value:
            continue
        if value in _VALID_KERNEL_POLICIES:
            return value
    return "auto"


def _base_extension_available(device_type: str, *, enabled: bool, debug_fallback: bool) -> bool:
    if not enabled or debug_fallback:
        return False
    if device_type == "cuda":
        return is_grm_cuda_available()
    return load_grm_cuda_extension()


def _should_use_chunk_update_extension(
    keys: Tensor,
    values: Tensor,
    queries: Tensor,
    memory: Tensor,
    *,
    enabled: bool,
    debug_fallback: bool,
) -> bool:
    if not _base_extension_available(memory.device.type, enabled=enabled, debug_fallback=debug_fallback):
        return False

    policy = _get_kernel_policy("chunk_update")
    if policy == "fallback":
        return False
    if policy == "native":
        return True

    return memory.device.type == "cuda"


def _should_use_apply_query_extension(
    memories: Tensor,
    queries: Tensor,
    *,
    enabled: bool,
    debug_fallback: bool,
) -> bool:
    if not _base_extension_available(memories.device.type, enabled=enabled, debug_fallback=debug_fallback):
        return False

    policy = _get_kernel_policy("apply_query")
    if policy == "fallback":
        return False
    if policy == "native":
        return True

    # Rank-4 training paths benefit modestly from the native backward path.
    return memories.ndim == 4 and (memories.requires_grad or queries.requires_grad)


def _should_use_batched_memory_gather_extension(
    memories_per_batch: Tensor,
    *,
    enabled: bool,
    debug_fallback: bool,
) -> bool:
    if not _base_extension_available(
        memories_per_batch.device.type,
        enabled=enabled,
        debug_fallback=debug_fallback,
    ):
        return False

    policy = _get_kernel_policy("batched_memory_gather")
    if policy == "fallback":
        return False
    return True


def is_grm_cuda_available() -> bool:
    if not load_grm_cuda_extension():
        return False
    dispatch_status = _get_dispatch_kernel_status()
    has_cuda_dispatch = all(status["cuda"] for status in dispatch_status.values())
    return has_cuda_dispatch and torch.cuda.is_available()


def get_grm_cuda_runtime_status() -> Dict[str, Any]:
    extension_loaded = load_grm_cuda_extension()
    namespace = getattr(torch.ops, "grm_cuda", None)
    registered_ops = {op_name: bool(namespace and hasattr(namespace, op_name)) for op_name in _REGISTERED_OPS}
    compiled_with_cuda_sources = bool(
        getattr(_EXTENSION_MODULE, "__grm_cuda_with_kernels__", False) if _EXTENSION_MODULE is not None else False
    )
    dispatch_kernels = _get_dispatch_kernel_status() if extension_loaded else {
        op_name: {"cuda": False, "composite_explicit_autograd": False} for op_name in _REGISTERED_OPS
    }
    compiled_with_cuda_kernels = compiled_with_cuda_sources and all(
        dispatch_status["cuda"] for dispatch_status in dispatch_kernels.values()
    )
    return {
        "load_attempted": _EXTENSION_LOAD_ATTEMPTED,
        "extension_loaded": extension_loaded,
        "compiled_with_cuda_sources": compiled_with_cuda_sources,
        "compiled_with_cuda_kernels": compiled_with_cuda_kernels,
        "torch_cuda_available": torch.cuda.is_available(),
        "cuda_backend_available": compiled_with_cuda_kernels and torch.cuda.is_available(),
        "load_error": _EXTENSION_LOAD_ERROR,
        "module_name": getattr(_EXTENSION_MODULE, "__name__", "") if _EXTENSION_MODULE is not None else "",
        "registered_ops": registered_ops,
        "dispatch_kernels": dispatch_kernels,
        "kernel_policy": {
            "global": _get_kernel_policy(""),
            "chunk_update": _get_kernel_policy("chunk_update"),
            "apply_query": _get_kernel_policy("apply_query"),
            "batched_memory_gather": _get_kernel_policy("batched_memory_gather"),
        },
    }


def _vectorized_chunk_update(
    keys: Tensor,
    values: Tensor,
    queries: Tensor,
    memory: Tensor,
    memory_decay: float,
) -> Tuple[Tensor, Tensor]:
    batch_size, chunk_len, memory_key_dim = keys.shape
    hidden_size = values.size(-1)

    if chunk_len == 0:
        return values.new_zeros(batch_size, 0, hidden_size), memory

    decay_base = torch.as_tensor(memory_decay, device=memory.device, dtype=memory.dtype)
    beta = 1.0 - decay_base
    positions = torch.arange(chunk_len, device=memory.device, dtype=memory.dtype)

    start_scale = torch.pow(decay_base, positions + 1.0).view(1, chunk_len, 1)
    start_hidden = torch.matmul(memory, queries.transpose(1, 2)).transpose(1, 2) * start_scale

    scores = torch.matmul(queries, keys.transpose(1, 2)) / math.sqrt(float(memory_key_dim))
    rel = positions.unsqueeze(1) - positions.unsqueeze(0)
    causal_mask = rel >= 0
    decay_mask = torch.pow(decay_base, rel.clamp_min(0))
    weighted_scores = scores * decay_mask.unsqueeze(0) * causal_mask.unsqueeze(0).to(scores.dtype)
    local_hidden = beta * torch.matmul(weighted_scores, values)
    hidden = start_hidden + local_hidden

    end_scales = torch.pow(decay_base, (chunk_len - 1) - positions)
    weighted_v = values * (beta * end_scales / math.sqrt(float(memory_key_dim))).view(1, chunk_len, 1)
    memory_delta = torch.einsum("bch,bck->bhk", weighted_v, keys)
    memory_out = torch.pow(
        decay_base,
        torch.tensor(float(chunk_len), device=memory.device, dtype=memory.dtype),
    ) * memory + memory_delta

    return hidden, memory_out


def _fallback_apply_query(memories: Tensor, queries: Tensor) -> Tensor:
    if memories.ndim == 5:
        return torch.einsum("bqrhk,bqk->bqrh", memories, queries)
    if memories.ndim == 4:
        return torch.einsum("brhk,bk->brh", memories, queries)
    raise ValueError(f"Unsupported memory rank for apply-query: {memories.ndim}")


def _fallback_batched_memory_gather(memories_per_batch: Tensor, topk_indices: Tensor) -> Tensor:
    if memories_per_batch.ndim != 4:
        raise ValueError(f"Expected memories_per_batch rank 4, got {memories_per_batch.ndim}")
    if topk_indices.ndim not in {2, 3}:
        raise ValueError(f"Expected topk_indices rank 2 or 3, got {topk_indices.ndim}")

    output_shape = (
        memories_per_batch.size(0),
        *topk_indices.shape[1:],
        *memories_per_batch.shape[2:],
    )
    selected = memories_per_batch.new_empty(output_shape)

    for batch_idx in range(memories_per_batch.size(0)):
        memory_batch = memories_per_batch[batch_idx]
        indices = topk_indices[batch_idx].reshape(-1)
        gathered = memory_batch.index_select(0, indices)
        selected[batch_idx].copy_(gathered.reshape(selected[batch_idx].shape))

    return selected.contiguous()


class ChunkUpdateCudaFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, keys: Tensor, values: Tensor, queries: Tensor, memory: Tensor, memory_decay: float):
        hidden, memory_out, aux = torch.ops.grm_cuda.chunk_update_forward(
            keys,
            values,
            queries,
            memory,
            float(memory_decay),
        )
        ctx.memory_decay = float(memory_decay)
        ctx.save_for_backward(keys, values, queries, memory, aux)
        return hidden, memory_out

    @staticmethod
    def backward(ctx, grad_hidden: Tensor, grad_memory_out: Tensor):
        keys, values, queries, memory, aux = ctx.saved_tensors
        with torch.enable_grad():
            grad_k, grad_v, grad_q, grad_memory = torch.ops.grm_cuda.chunk_update_backward(
                grad_hidden.contiguous(),
                grad_memory_out.contiguous(),
                keys,
                values,
                queries,
                memory,
                aux,
                float(ctx.memory_decay),
            )
        return grad_k, grad_v, grad_q, grad_memory, None


class ApplyQueryCudaFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, memories: Tensor, queries: Tensor):
        hidden = torch.ops.grm_cuda.apply_query_forward(memories, queries)
        ctx.save_for_backward(memories, queries)
        return hidden

    @staticmethod
    def backward(ctx, grad_hidden: Tensor):
        memories, queries = ctx.saved_tensors
        with torch.enable_grad():
            grad_memories, grad_queries = torch.ops.grm_cuda.apply_query_backward(
                grad_hidden.contiguous(),
                memories,
                queries,
            )
        return grad_memories, grad_queries


def cuda_chunk_update(
    keys: Tensor,
    values: Tensor,
    queries: Tensor,
    memory: Tensor,
    memory_decay: float,
    *,
    enabled: bool = True,
    debug_fallback: bool = False,
) -> Tuple[Tensor, Tensor]:
    use_extension = _should_use_chunk_update_extension(
        keys,
        values,
        queries,
        memory,
        enabled=enabled,
        debug_fallback=debug_fallback,
    )
    if use_extension:
        return ChunkUpdateCudaFn.apply(keys, values, queries, memory, float(memory_decay))
    return _vectorized_chunk_update(keys, values, queries, memory, memory_decay)


def cuda_apply_query(
    memories: Tensor,
    queries: Tensor,
    *,
    enabled: bool = True,
    debug_fallback: bool = False,
) -> Tensor:
    use_extension = _should_use_apply_query_extension(
        memories,
        queries,
        enabled=enabled,
        debug_fallback=debug_fallback,
    )
    if use_extension:
        return ApplyQueryCudaFn.apply(memories, queries)
    return _fallback_apply_query(memories, queries)


def cuda_batched_memory_gather(
    memories_per_batch: Tensor,
    topk_indices: Tensor,
    *,
    enabled: bool = True,
    debug_fallback: bool = False,
) -> Tensor:
    use_extension = _should_use_batched_memory_gather_extension(
        memories_per_batch,
        enabled=enabled,
        debug_fallback=debug_fallback,
    )
    if use_extension:
        return torch.ops.grm_cuda.batched_memory_gather(memories_per_batch, topk_indices)
    return _fallback_batched_memory_gather(memories_per_batch, topk_indices)


__all__ = [
    "cuda_apply_query",
    "cuda_batched_memory_gather",
    "cuda_chunk_update",
    "get_grm_cuda_runtime_status",
    "is_grm_cpp_extension_available",
    "is_grm_cuda_available",
    "load_grm_cuda_extension",
]
