"""Segment-memory recurrent model built on cached linear memory matrices."""

from collections import deque
import math
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from .aggregator import RetrievedStateFusion
from .cuda_ops import (
    cuda_apply_query,
    cuda_batched_memory_gather,
    cuda_chunk_update,
    get_grm_cuda_runtime_status,
)
from .gating_unit import TopKMemoryRetriever
from .linear_attention import LinearMemoryCell


@dataclass
class SegmentReconstructionState:
    inputs: Tensor
    memory_start: Tensor


class _BatchMemoryArchive:
    def __init__(self, max_segments: int):
        self.max_segments = max_segments
        self.memories: Deque[Tensor] = deque(maxlen=max_segments)
        self.summaries: Deque[Tensor] = deque(maxlen=max_segments)
        self.checkpoints: Deque[SegmentReconstructionState] = deque(maxlen=max_segments)

    def append(self, memory: Tensor, summary: Tensor, checkpoint: SegmentReconstructionState) -> None:
        self.memories.append(memory)
        self.summaries.append(summary)
        self.checkpoints.append(checkpoint)

    def __len__(self) -> int:
        return len(self.memories)


class SegmentMemoryArchive(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        memory_key_dim: int,
        summary_size: int,
        max_segments: int,
        memory_storage_dtype: torch.dtype,
        recomputation_ratio: float,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_key_dim = memory_key_dim
        self.summary_size = summary_size
        self.max_segments = max_segments
        self.memory_storage_dtype = memory_storage_dtype
        self.recomputation_ratio = recomputation_ratio
        self._groups: Dict[int, _BatchMemoryArchive] = {}
        self._batch_groups = self._groups
        self.register_buffer("total_pushed", torch.tensor(0, dtype=torch.long))
        self.register_buffer("total_recomputed", torch.tensor(0, dtype=torch.long))
        self.register_buffer("total_retrieved", torch.tensor(0, dtype=torch.long))

    def _group(self, batch_size: int) -> _BatchMemoryArchive:
        if batch_size not in self._groups:
            self._groups[batch_size] = _BatchMemoryArchive(self.max_segments)
        return self._groups[batch_size]

    def clear(self) -> None:
        self._groups.clear()
        self.total_pushed.zero_()
        self.total_recomputed.zero_()
        self.total_retrieved.zero_()

    def _apply(self, fn):
        super()._apply(fn)
        for group in self._groups.values():
            group.memories = deque((fn(memory) for memory in group.memories), maxlen=self.max_segments)
            group.summaries = deque((fn(summary) for summary in group.summaries), maxlen=self.max_segments)
            group.checkpoints = deque(
                (
                    SegmentReconstructionState(
                        inputs=fn(checkpoint.inputs),
                        memory_start=fn(checkpoint.memory_start),
                    )
                    for checkpoint in group.checkpoints
                ),
                maxlen=self.max_segments,
            )
        return self

    def get_extra_state(self):
        return {
            "groups": {
                batch_size: {
                    "memories": list(group.memories),
                    "summaries": list(group.summaries),
                    "checkpoints": [
                        {
                            "inputs": checkpoint.inputs,
                            "memory_start": checkpoint.memory_start,
                        }
                        for checkpoint in group.checkpoints
                    ],
                }
                for batch_size, group in self._groups.items()
            }
        }

    def set_extra_state(self, state):
        self._groups = {}
        groups_state = {} if state is None else state.get("groups", {})
        for batch_size, group_state in groups_state.items():
            group = _BatchMemoryArchive(self.max_segments)
            for memory in group_state.get("memories", []):
                group.memories.append(memory)
            for summary in group_state.get("summaries", []):
                group.summaries.append(summary)
            for checkpoint in group_state.get("checkpoints", []):
                group.checkpoints.append(
                    SegmentReconstructionState(
                        inputs=checkpoint["inputs"],
                        memory_start=checkpoint["memory_start"],
                    )
                )
            self._groups[int(batch_size)] = group
        self._batch_groups = self._groups

    def push(self, memory: Tensor, summary: Tensor, checkpoint: SegmentReconstructionState) -> None:
        batch_size = memory.size(0)
        group = self._group(batch_size)
        group.append(
            memory.detach().to(self.memory_storage_dtype).clone(),
            summary.detach().to(self.memory_storage_dtype).clone(),
            SegmentReconstructionState(
                inputs=checkpoint.inputs.detach().clone(),
                memory_start=checkpoint.memory_start.detach().to(self.memory_storage_dtype).clone(),
            ),
        )
        self.total_pushed.add_(1)

    def get_num_segments(self, batch_size: Optional[int] = None) -> int:
        if batch_size is not None:
            return len(self._groups.get(batch_size, []))
        if not self._groups:
            return 0
        return max(len(group) for group in self._groups.values())

    def get_segment_summaries(
        self,
        query_batch_size: int,
        device: torch.device,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        group = self._groups.get(query_batch_size)
        target_dtype = dtype or torch.float32
        if group is None or len(group) == 0:
            return torch.zeros(query_batch_size, 0, self.summary_size, device=device, dtype=target_dtype)

        summaries = torch.stack(list(group.summaries), dim=0).to(device=device, dtype=target_dtype)
        return summaries.clone().permute(1, 0, 2).contiguous()

    def _build_recompute_mask(self, gates: Tensor) -> Tensor:
        if self.recomputation_ratio <= 0.0 or gates.numel() == 0:
            return torch.zeros_like(gates, dtype=torch.bool)
        flat_gates = gates.reshape(-1)
        num_recompute = int(flat_gates.numel() * self.recomputation_ratio)
        if num_recompute <= 0:
            return torch.zeros_like(gates, dtype=torch.bool)
        num_recompute = min(num_recompute, flat_gates.numel())
        _, order = torch.topk(flat_gates, k=num_recompute, dim=0)
        flat_mask = torch.zeros_like(flat_gates, dtype=torch.bool)
        flat_mask.scatter_(0, order, True)
        return flat_mask.view_as(gates)

    def _recompute_memory(
        self,
        checkpoint: SegmentReconstructionState,
        memory_cell: LinearMemoryCell,
        device: torch.device,
        memory_decay: float,
        batch_index: int,
    ) -> Tensor:
        memory = checkpoint.memory_start[batch_index : batch_index + 1].to(device=device, dtype=torch.float32)
        inputs = checkpoint.inputs[:, batch_index : batch_index + 1].to(device=device, dtype=torch.float32)
        scale = math.sqrt(memory_cell.memory_key_dim)

        for t in range(inputs.size(0)):
            x_t = inputs[t]
            k_t = F.normalize(memory_cell.key_proj(x_t), p=2, dim=-1, eps=1e-6)
            v_t = memory_cell.value_proj(x_t)
            update = torch.bmm(v_t.unsqueeze(2), k_t.unsqueeze(1)) / scale
            memory = memory_decay * memory + (1.0 - memory_decay) * update

        return memory.squeeze(0)

    def retrieve_memories(
        self,
        topk_indices: Tensor,
        gates: Tensor,
        memory_cell: LinearMemoryCell,
        device: torch.device,
        memory_decay: float,
        cuda_cpp_debug_fallback: bool = False,
    ) -> Tensor:
        batch_size = topk_indices.size(0)
        group = self._groups.get(batch_size)
        if group is None or topk_indices.numel() == 0 or len(group) == 0:
            if topk_indices.ndim == 3:
                return torch.zeros(
                    batch_size,
                    topk_indices.size(1),
                    0,
                    self.hidden_size,
                    self.memory_key_dim,
                    device=device,
                    dtype=torch.float32,
                )
            return torch.zeros(batch_size, 0, self.hidden_size, self.memory_key_dim, device=device, dtype=torch.float32)

        all_memories = torch.stack(list(group.memories), dim=0).to(device=device, dtype=torch.float32)
        memories_per_batch = all_memories.permute(1, 0, 2, 3).contiguous()
        selected = cuda_batched_memory_gather(
            memories_per_batch=memories_per_batch,
            topk_indices=topk_indices,
            enabled=True,
            debug_fallback=cuda_cpp_debug_fallback,
        )

        self.total_retrieved.add_(int(topk_indices.numel()))

        recompute_mask = self._build_recompute_mask(gates)
        if recompute_mask.any():
            self.total_recomputed.add_(int(recompute_mask.sum().item()))
            recompute_cache: Dict[Tuple[int, int], Tensor] = {}
            if topk_indices.ndim == 3:
                for b_idx, q_idx, k_idx in recompute_mask.nonzero(as_tuple=False).tolist():
                    seg_idx = int(topk_indices[b_idx, q_idx, k_idx].item())
                    if 0 <= seg_idx < len(group):
                        cache_key = (seg_idx, b_idx)
                        if cache_key not in recompute_cache:
                            recompute_cache[cache_key] = self._recompute_memory(
                                checkpoint=group.checkpoints[seg_idx],
                                memory_cell=memory_cell,
                                device=device,
                                memory_decay=memory_decay,
                                batch_index=b_idx,
                            )
                        selected[b_idx, q_idx, k_idx] = recompute_cache[cache_key]
            else:
                for b_idx, k_idx in recompute_mask.nonzero(as_tuple=False).tolist():
                    seg_idx = int(topk_indices[b_idx, k_idx].item())
                    if 0 <= seg_idx < len(group):
                        cache_key = (seg_idx, b_idx)
                        if cache_key not in recompute_cache:
                            recompute_cache[cache_key] = self._recompute_memory(
                                checkpoint=group.checkpoints[seg_idx],
                                memory_cell=memory_cell,
                                device=device,
                                memory_decay=memory_decay,
                                batch_index=b_idx,
                            )
                        selected[b_idx, k_idx] = recompute_cache[cache_key]

        return selected

    def retrieve_from_indices(
        self,
        query_inputs: Tensor,
        topk_indices: Tensor,
        gates: Tensor,
        memory_cell: LinearMemoryCell,
        device: torch.device,
        memory_decay: float,
        cuda_cpp_debug_fallback: bool = False,
    ) -> Tensor:
        memories = self.retrieve_memories(
            topk_indices=topk_indices,
            gates=gates,
            memory_cell=memory_cell,
            device=device,
            memory_decay=memory_decay,
            cuda_cpp_debug_fallback=cuda_cpp_debug_fallback,
        )
        if topk_indices.ndim == 3:
            if memories.size(2) == 0:
                return torch.zeros(
                    query_inputs.size(0),
                    query_inputs.size(1),
                    0,
                    self.hidden_size,
                    device=device,
                    dtype=torch.float32,
                )
            q_t = F.normalize(memory_cell.query_proj(query_inputs), p=2, dim=-1, eps=1e-6).to(memories.dtype)
            return cuda_apply_query(
                memories,
                q_t,
                enabled=True,
                debug_fallback=cuda_cpp_debug_fallback,
            )

        if memories.size(1) == 0:
            return torch.zeros(query_inputs.size(0), 0, self.hidden_size, device=device, dtype=torch.float32)

        q_t = F.normalize(memory_cell.query_proj(query_inputs), p=2, dim=-1, eps=1e-6).to(memories.dtype)
        return cuda_apply_query(
            memories,
            q_t,
            enabled=True,
            debug_fallback=cuda_cpp_debug_fallback,
        )

    def get_recompute_rate(self) -> float:
        total_retrieved = int(self.total_retrieved.item())
        if total_retrieved == 0:
            return 0.0
        return float(self.total_recomputed.item()) / total_retrieved

    def get_memory_info(self) -> Dict[str, float]:
        memory_bytes = 0
        summary_bytes = 0
        checkpoint_bytes = 0
        num_segments = 0
        for group in self._groups.values():
            num_segments += len(group)
            for memory in group.memories:
                memory_bytes += memory.numel() * memory.element_size()
            for summary in group.summaries:
                summary_bytes += summary.numel() * summary.element_size()
            for checkpoint in group.checkpoints:
                checkpoint_bytes += checkpoint.inputs.numel() * checkpoint.inputs.element_size()
                checkpoint_bytes += checkpoint.memory_start.numel() * checkpoint.memory_start.element_size()
        total_mb = (memory_bytes + summary_bytes + checkpoint_bytes) / (1024 ** 2)
        return {
            "cached_states_mb": round(memory_bytes / (1024 ** 2), 2),
            "summaries_mb": round(summary_bytes / (1024 ** 2), 2),
            "checkpoints_mb": round(checkpoint_bytes / (1024 ** 2), 2),
            "total_mb": round(total_mb, 2),
            "num_segments": num_segments,
            "num_groups": len(self._groups),
            "recompute_rate": self.get_recompute_rate(),
        }


class SegmentRecurrentMemoryModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        rnn_type: Literal["linear"] = "linear",
        segment_length: int = 64,
        memory_capacity_segments: int = 64,
        retrieval_top_k: int = 4,
        memory_storage_dtype: torch.dtype = torch.float16,
        recomputation_ratio: float = 0.02,
        segment_pooling_mode: Literal["mean", "max", "attention", "weighted"] = "mean",
        retrieval_fusion_mode: Literal["residual", "concat", "gate", "linear", "memory_soup"] = "residual",
        use_layer_norm: bool = True,
        dropout: float = 0.0,
        batch_first: bool = True,
        memory_key_dim: Optional[int] = None,
        memory_initialization_mode: Literal["checkpoint", "independent"] = "checkpoint",
        retrieval_query_source: Literal["input"] = "input",
        segment_summary_source: Literal["input_mean"] = "input_mean",
        memory_decay: float = 0.97,
        **kwargs,
    ):
        super().__init__()
        if rnn_type != "linear":
            raise ValueError(
                "The active GRM architecture only supports rnn_type='linear'. "
                "Legacy GRU/LSTM/RNN backends have been retired."
            )
        if retrieval_query_source != "input" or segment_summary_source != "input_mean":
            raise ValueError("The active paper backend supports only retrieval_query_source='input' and segment_summary_source='input_mean'.")
        if segment_pooling_mode != "mean":
            raise ValueError("The active paper backend supports only mean segment summaries.")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.rnn_type = rnn_type
        self.segment_length = segment_length
        self.memory_capacity_segments = memory_capacity_segments
        self.retrieval_top_k = retrieval_top_k
        self.batch_first = batch_first
        self.memory_key_dim = memory_key_dim or max(16, hidden_size // 8)
        self.memory_initialization_mode = memory_initialization_mode
        self.retrieval_query_source = retrieval_query_source
        self.segment_summary_source = segment_summary_source
        self.memory_decay = memory_decay
        self.retrieval_fusion_mode = retrieval_fusion_mode
        self.enable_activation_checkpointing = bool(kwargs.pop("enable_activation_checkpointing", False))
        removed_runtime_args = [
            key
            for key in ("use_torch_compile", "torch_compile_mode", "use_triton_kernels", "use_cuda_cpp_kernels")
            if key in kwargs
        ]
        if removed_runtime_args:
            removed_args = ", ".join(sorted(removed_runtime_args))
            raise TypeError(
                f"Removed runtime arguments: {removed_args}. "
                "The active GRM runtime always uses the CUDA C++ op wrapper path."
            )
        self.cuda_cpp_debug_fallback = bool(kwargs.pop("cuda_cpp_debug_fallback", False))

        self.memory_cell = LinearMemoryCell(
            input_size=input_size,
            hidden_size=hidden_size,
            memory_key_dim=self.memory_key_dim,
            memory_init="zeros",
        )
        self.gating_unit = TopKMemoryRetriever(
            query_dim=input_size,
            key_dim=input_size,
            hidden_dim=input_size,
            retrieval_top_k=retrieval_top_k,
            use_causal_mask=True,
            dropout=dropout,
        )
        self.aggregator = RetrievedStateFusion(
            hidden_size=hidden_size,
            retrieval_fusion_mode=retrieval_fusion_mode if retrieval_fusion_mode != "memory_soup" else "residual",
            use_layer_norm=use_layer_norm,
            dropout=dropout,
        )
        self.memory_soup_norm = nn.LayerNorm(hidden_size) if retrieval_fusion_mode == "memory_soup" and use_layer_norm else nn.Identity()
        self.output_proj = nn.Linear(hidden_size, output_size)
        self.memory_bank = SegmentMemoryArchive(
            hidden_size=hidden_size,
            memory_key_dim=self.memory_key_dim,
            summary_size=input_size,
            max_segments=memory_capacity_segments,
            memory_storage_dtype=memory_storage_dtype,
            recomputation_ratio=recomputation_ratio,
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def _build_runtime_optimization_status(self, device_type: Optional[str] = None) -> Dict[str, Any]:
        cuda_cpp_status = get_grm_cuda_runtime_status()
        return {
            "runtime_path": "cuda_cpp",
            "cuda_cpp_enabled": True,
            "cuda_cpp_debug_fallback": self.cuda_cpp_debug_fallback,
            "cuda_cpp_runtime": cuda_cpp_status,
            "device_type": device_type or str(self.output_proj.weight.device.type),
        }

    def get_runtime_optimization_status(self, device_type: Optional[str] = None) -> Dict[str, Any]:
        return self._build_runtime_optimization_status(device_type=device_type)

    def _init_memory(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        return self.memory_cell.init_memory(batch_size, device, dtype).memory

    def clear_memory(self) -> None:
        self.memory_bank.clear()

    def get_num_segments(self) -> int:
        return self.memory_bank.get_num_segments()

    def get_memory_info(self) -> Dict[str, float]:
        return self.memory_bank.get_memory_info()

    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[Tensor, Tensor]:
        h_init = torch.zeros(batch_size, self.hidden_size, device=device)
        memory_init = self._init_memory(batch_size, device, h_init.dtype)
        return h_init, memory_init

    def _step_memory(self, x_t: Tensor, memory: Tensor) -> Tuple[Tensor, Tensor]:
        k_t = F.normalize(self.memory_cell.key_proj(x_t), p=2, dim=-1, eps=1e-6)
        v_t = self.memory_cell.value_proj(x_t)
        q_t = F.normalize(self.memory_cell.query_proj(x_t), p=2, dim=-1, eps=1e-6)
        update = torch.bmm(v_t.unsqueeze(2), k_t.unsqueeze(1)) / math.sqrt(self.memory_key_dim)
        memory = self.memory_decay * memory + (1.0 - self.memory_decay) * update
        hidden = torch.bmm(memory, q_t.unsqueeze(2)).squeeze(2)
        return self.dropout(hidden), memory

    def _forward_memory_chunk_math(self, batch_inputs: Tensor, memory: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size, chunk_len, _ = batch_inputs.shape
        if chunk_len == 0:
            return batch_inputs.new_zeros(batch_size, 0, self.hidden_size), memory

        k = F.normalize(self.memory_cell.key_proj(batch_inputs), p=2, dim=-1, eps=1e-6)
        v = self.memory_cell.value_proj(batch_inputs)
        q = F.normalize(self.memory_cell.query_proj(batch_inputs), p=2, dim=-1, eps=1e-6)

        h_online, memory_end = cuda_chunk_update(
            keys=k,
            values=v,
            queries=q,
            memory=memory,
            memory_decay=self.memory_decay,
            enabled=True,
            debug_fallback=self.cuda_cpp_debug_fallback,
        )
        return self.dropout(h_online.clone()), memory_end.clone()

    def _forward_memory_chunk(self, segment_inputs: Tensor, memory: Tensor) -> Tuple[Tensor, Tensor]:
        batch_inputs = segment_inputs.transpose(0, 1).contiguous()
        if (
            self.enable_activation_checkpointing
            and self.training
            and torch.is_grad_enabled()
            and batch_inputs.requires_grad
        ):
            return activation_checkpoint(
                self._forward_memory_chunk_math,
                batch_inputs,
                memory,
                use_reentrant=False,
            )
        return self._forward_memory_chunk_math(batch_inputs, memory)

    def _segment_summary(self, segment_inputs: Tensor) -> Tensor:
        return segment_inputs.mean(dim=0)

    def _retrieve_and_enhance(self, x_t: Tensor, h_online: Tensor, batch_size: int) -> Tensor:
        segment_keys = self.memory_bank.get_segment_summaries(
            query_batch_size=batch_size,
            device=x_t.device,
            dtype=x_t.dtype,
        )
        if segment_keys.size(1) == 0:
            return h_online

        gates, indices, _ = self.gating_unit(
            query=x_t,
            keys=segment_keys,
            segment_offset=self.memory_bank.get_num_segments(batch_size),
        )

        if self.retrieval_fusion_mode == "memory_soup":
            selected_memories = self.memory_bank.retrieve_memories(
                topk_indices=indices,
                gates=gates,
                memory_cell=self.memory_cell,
                device=x_t.device,
                memory_decay=self.memory_decay,
                cuda_cpp_debug_fallback=self.cuda_cpp_debug_fallback,
            )
            q_t = F.normalize(self.memory_cell.query_proj(x_t), p=2, dim=-1, eps=1e-6).to(selected_memories.dtype)
            soup_memory = (selected_memories * gates.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
            soup_hidden = torch.matmul(soup_memory, q_t.unsqueeze(-1)).squeeze(-1)
            enhanced = self.memory_soup_norm(h_online + soup_hidden)
            return self.dropout(enhanced)

        retrieved = self.memory_bank.retrieve_from_indices(
            query_inputs=x_t,
            topk_indices=indices,
            gates=gates,
            memory_cell=self.memory_cell,
            device=x_t.device,
            memory_decay=self.memory_decay,
            cuda_cpp_debug_fallback=self.cuda_cpp_debug_fallback,
        )
        return self.aggregator(h_online=h_online, cached_states=retrieved, gates=gates)

    def _route_chunk(self, batch_inputs: Tensor, segment_keys: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size, chunk_len, _ = batch_inputs.shape
        num_keys = segment_keys.size(1)

        q_proj = self.gating_unit.query_proj(batch_inputs.reshape(batch_size * chunk_len, -1))
        q_proj = q_proj.view(batch_size, chunk_len, -1)
        k_proj = self.gating_unit.key_proj(segment_keys.reshape(batch_size * num_keys, -1))
        k_proj = k_proj.view(batch_size, num_keys, -1)
        scores = torch.matmul(q_proj, k_proj.transpose(1, 2)) * self.gating_unit.scale

        k_actual = min(self.retrieval_top_k, num_keys)
        if k_actual < num_keys:
            topk_scores, topk_indices = torch.topk(scores, k=k_actual, dim=-1)
        else:
            topk_scores = scores
            topk_indices = (
                torch.arange(num_keys, device=batch_inputs.device)
                .view(1, 1, num_keys)
                .expand(batch_size, chunk_len, -1)
            )

        gates = F.softmax(topk_scores / self.gating_unit.temperature, dim=-1)
        gates = self.gating_unit.dropout(gates)
        return gates.clone(), topk_indices.clone()

    def _memory_soup_chunk(
        self,
        h_online: Tensor,
        batch_inputs: Tensor,
        selected_memories: Tensor,
        gates: Tensor,
    ) -> Tensor:
        q_t = F.normalize(self.memory_cell.query_proj(batch_inputs), p=2, dim=-1, eps=1e-6).to(selected_memories.dtype)
        soup_memory = (selected_memories * gates.unsqueeze(-1).unsqueeze(-1)).sum(dim=2)
        soup_hidden = torch.einsum("bchk,bck->bch", soup_memory, q_t)
        enhanced = self.memory_soup_norm(h_online + soup_hidden)
        return self.dropout(enhanced).clone()

    def _fuse_retrieved_chunk(self, h_online: Tensor, retrieved: Tensor, gates: Tensor) -> Tensor:
        batch_size, chunk_len, k_actual, _ = retrieved.shape
        enhanced = self.aggregator(
            h_online=h_online.reshape(batch_size * chunk_len, self.hidden_size),
            cached_states=retrieved.reshape(batch_size * chunk_len, k_actual, self.hidden_size),
            gates=gates.reshape(batch_size * chunk_len, k_actual),
        )
        return enhanced.view(batch_size, chunk_len, self.hidden_size).clone()

    def _retrieve_and_enhance_chunk(self, segment_inputs: Tensor, h_online: Tensor, batch_size: int) -> Tensor:
        batch_inputs = segment_inputs.transpose(0, 1).contiguous()
        segment_keys = self.memory_bank.get_segment_summaries(
            query_batch_size=batch_size,
            device=batch_inputs.device,
            dtype=batch_inputs.dtype,
        )
        if segment_keys.size(1) == 0:
            return h_online

        gates, topk_indices = self._route_chunk(batch_inputs, segment_keys)

        if self.retrieval_fusion_mode == "memory_soup":
            selected_memories = self.memory_bank.retrieve_memories(
                topk_indices=topk_indices,
                gates=gates,
                memory_cell=self.memory_cell,
                device=batch_inputs.device,
                memory_decay=self.memory_decay,
                cuda_cpp_debug_fallback=self.cuda_cpp_debug_fallback,
            )
            return self._memory_soup_chunk(h_online, batch_inputs, selected_memories, gates)

        retrieved = self.memory_bank.retrieve_from_indices(
            query_inputs=batch_inputs,
            topk_indices=topk_indices,
            gates=gates,
            memory_cell=self.memory_cell,
            device=batch_inputs.device,
            memory_decay=self.memory_decay,
            cuda_cpp_debug_fallback=self.cuda_cpp_debug_fallback,
        )
        return self._fuse_retrieved_chunk(h_online, retrieved, gates)

    def forward(
        self,
        x_seq: Tensor,
        h_init: Optional[Tensor] = None,
        c_init: Optional[Tensor] = None,
        return_all_outputs: bool = True,
        return_hidden_states: bool = False,
        reset_memory: bool = False,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Dict[str, Any]]]:
        if x_seq.ndim == 2:
            x_seq = x_seq.unsqueeze(1 if self.batch_first else 0)
        if self.batch_first:
            x_seq = x_seq.transpose(0, 1)

        seq_len, batch_size, _ = x_seq.shape
        device = x_seq.device
        dtype = x_seq.dtype

        if reset_memory:
            self.clear_memory()

        memory = c_init if (c_init is not None and c_init.ndim == 3) else self._init_memory(batch_size, device, dtype)
        last_hidden = (
            h_init
            if (h_init is not None and h_init.ndim == 2)
            else torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        )
        output_segments: List[Tensor] = []
        hidden_segments: List[Tensor] = []
        last_output: Optional[Tensor] = None
        hidden_states: Optional[List[Tensor]] = [] if return_hidden_states else None

        if seq_len == 0:
            if return_all_outputs:
                if self.batch_first:
                    outputs_tensor = x_seq.new_zeros(batch_size, 0, self.output_size)
                else:
                    outputs_tensor = x_seq.new_zeros(0, batch_size, self.output_size)
            else:
                outputs_tensor = torch.zeros(batch_size, self.output_size, device=device, dtype=dtype)

            aux = {
                "memory_info": self.memory_bank.get_memory_info(),
                "num_segments": self.memory_bank.get_num_segments(batch_size),
                "recompute_rate": self.memory_bank.get_recompute_rate(),
            }
            if hidden_states is not None:
                hidden_tensor = (
                    x_seq.new_zeros(batch_size, 0, self.hidden_size)
                    if self.batch_first
                    else x_seq.new_zeros(0, batch_size, self.hidden_size)
                )
                aux["hidden_states"] = hidden_tensor
            return outputs_tensor, last_hidden, memory, aux

        for segment_start in range(0, seq_len, self.segment_length):
            segment_end = min(seq_len, segment_start + self.segment_length)
            segment_inputs = x_seq[segment_start:segment_end]
            segment_start_memory = memory.detach().clone()

            h_online, memory = self._forward_memory_chunk(segment_inputs, memory)
            h_enhanced = self._retrieve_and_enhance_chunk(segment_inputs, h_online, batch_size)
            y_chunk = self.output_proj(h_enhanced)

            if return_all_outputs:
                output_segments.append(y_chunk.transpose(0, 1))
            else:
                last_output = y_chunk[:, -1]

            if hidden_states is not None:
                hidden_segments.append(h_enhanced.transpose(0, 1).clone())

            last_hidden = h_enhanced[:, -1]

            self.memory_bank.push(
                memory=memory,
                summary=self._segment_summary(segment_inputs),
                checkpoint=SegmentReconstructionState(inputs=segment_inputs, memory_start=segment_start_memory),
            )

            if self.memory_initialization_mode == "independent":
                memory = self._init_memory(batch_size, device, dtype)

        if return_all_outputs:
            outputs_tensor = torch.cat(output_segments, dim=0)
            if self.batch_first:
                outputs_tensor = outputs_tensor.transpose(0, 1).contiguous()
        else:
            outputs_tensor = last_output if last_output is not None else torch.zeros(batch_size, self.output_size, device=device, dtype=dtype)

        aux = {
            "memory_info": self.memory_bank.get_memory_info(),
            "num_segments": self.memory_bank.get_num_segments(batch_size),
            "recompute_rate": self.memory_bank.get_recompute_rate(),
        }
        if hidden_states is not None:
            hidden_tensor = torch.cat(hidden_segments, dim=0)
            if self.batch_first:
                hidden_tensor = hidden_tensor.transpose(0, 1).contiguous()
            aux["hidden_states"] = hidden_tensor

        return outputs_tensor, last_hidden, memory, aux

    def forward_incremental(
        self,
        x_seq: Tensor,
        incremental_state: Optional[Dict[str, Any]] = None,
        h_init: Optional[Tensor] = None,
        c_init: Optional[Tensor] = None,
        return_all_outputs: bool = True,
        reset_memory: bool = False,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Dict[str, Any]]:
        outputs, h_final, memory_final, _ = self.forward(
            x_seq=x_seq,
            h_init=h_init,
            c_init=c_init,
            return_all_outputs=return_all_outputs,
            return_hidden_states=False,
            reset_memory=reset_memory,
        )
        new_state = dict(incremental_state or {})
        new_state["last_memory"] = memory_final
        return outputs, h_final, memory_final, new_state

    def extra_repr(self) -> str:
        return (
            f"input_size={self.input_size}, hidden_size={self.hidden_size}, "
            f"output_size={self.output_size}, rnn_type='{self.rnn_type}', "
            f"memory_key_dim={self.memory_key_dim}, segment_length={self.segment_length}, "
            f"memory_capacity_segments={self.memory_capacity_segments}, retrieval_top_k={self.retrieval_top_k}, "
            f"memory_initialization_mode='{self.memory_initialization_mode}', memory_decay={self.memory_decay}, "
            f"cuda_cpp_debug_fallback={self.cuda_cpp_debug_fallback}"
        )


def build_segment_memory_model(input_size: int, hidden_size: int, output_size: int, **kwargs) -> SegmentRecurrentMemoryModel:
    return SegmentRecurrentMemoryModel(input_size=input_size, hidden_size=hidden_size, output_size=output_size, **kwargs)
