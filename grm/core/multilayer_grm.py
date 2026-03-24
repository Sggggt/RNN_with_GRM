"""Hierarchical segment-memory recurrent blocks."""

from typing import Dict, Literal, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .paper_grm import SegmentRecurrentMemoryModel


class SegmentRecurrentMemoryLayer(nn.Module):
    """One segment-memory block with residual projection and normalization."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        rnn_type: Literal["linear"] = "linear",
        segment_length: int = 64,
        memory_capacity_segments: int = 64,
        retrieval_top_k: int = 8,
        memory_storage_dtype: torch.dtype = torch.float16,
        recomputation_ratio: float = 0.0,
        retrieval_fusion_mode: str = "residual",
        use_layer_norm: bool = True,
        dropout: float = 0.1,
        batch_first: bool = True,
        memory_key_dim: Optional[int] = None,
        memory_initialization_mode: str = "checkpoint",
        retrieval_query_source: str = "input",
        segment_summary_source: str = "input_mean",
        memory_decay: float = 0.97,
        enable_activation_checkpointing: bool = False,
        cuda_cpp_debug_fallback: bool = False,
    ):
        super().__init__()
        if rnn_type != "linear":
            raise ValueError("SegmentRecurrentMemoryLayer currently supports only the linear-memory backend.")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.segment_length = segment_length
        self.batch_first = batch_first

        self.grm = SegmentRecurrentMemoryModel(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=input_size,
            rnn_type="linear",
            segment_length=segment_length,
            memory_capacity_segments=memory_capacity_segments,
            retrieval_top_k=retrieval_top_k,
            memory_storage_dtype=memory_storage_dtype,
            recomputation_ratio=recomputation_ratio,
            retrieval_fusion_mode=retrieval_fusion_mode,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
            batch_first=batch_first,
            memory_key_dim=memory_key_dim,
            memory_initialization_mode=memory_initialization_mode,
            retrieval_query_source=retrieval_query_source,
            segment_summary_source=segment_summary_source,
            memory_decay=memory_decay,
            enable_activation_checkpointing=enable_activation_checkpointing,
            cuda_cpp_debug_fallback=cuda_cpp_debug_fallback,
        )
        self.layer_norm = nn.LayerNorm(input_size) if use_layer_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        x: Tensor,
        h_init: Optional[Tensor] = None,
        c_init: Optional[Tensor] = None,
        return_all_outputs: bool = False,
        return_hidden_states: bool = False,
        reset_memory: bool = False,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Dict]]:
        if x.ndim == 2:
            x = x.unsqueeze(1 if self.batch_first else 0)

        outputs, h_final, memory_final, aux = self.grm(
            x,
            h_init=h_init,
            c_init=c_init,
            return_all_outputs=return_all_outputs,
            return_hidden_states=return_hidden_states,
            reset_memory=reset_memory,
        )

        if return_all_outputs:
            outputs = self.layer_norm(outputs + x)
        else:
            residual = x[:, -1, :] if self.batch_first and x.ndim == 3 else x[-1, :, :] if x.ndim == 3 else x
            outputs = self.layer_norm(outputs + residual)
        outputs = self.dropout(outputs)
        return outputs, h_final, memory_final, aux

    @property
    def memory_bank(self):
        return self.grm.memory_bank


class HierarchicalSegmentMemoryModel(nn.Module):
    """Stacked segment-memory blocks for hierarchical recurrent processing."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 6,
        rnn_type: Literal["linear"] = "linear",
        segment_length: int = 64,
        memory_capacity_segments: int = 64,
        retrieval_top_k: int = 8,
        memory_storage_dtype: torch.dtype = torch.float16,
        recomputation_ratio: float = 0.0,
        retrieval_fusion_mode: str = "residual",
        use_layer_norm: bool = True,
        dropout: float = 0.1,
        batch_first: bool = True,
        memory_key_dim: Optional[int] = None,
        memory_initialization_mode: str = "checkpoint",
        retrieval_query_source: str = "input",
        segment_summary_source: str = "input_mean",
        memory_decay: float = 0.97,
        enable_activation_checkpointing: bool = False,
        cuda_cpp_debug_fallback: bool = False,
    ):
        super().__init__()
        if rnn_type != "linear":
            raise ValueError("HierarchicalSegmentMemoryModel currently supports only the linear-memory backend.")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.batch_first = batch_first

        self.layers = nn.ModuleList(
            [
                SegmentRecurrentMemoryLayer(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    rnn_type="linear",
                    segment_length=segment_length,
                    memory_capacity_segments=memory_capacity_segments,
                    retrieval_top_k=retrieval_top_k,
                    memory_storage_dtype=memory_storage_dtype,
                    recomputation_ratio=recomputation_ratio,
                    retrieval_fusion_mode=retrieval_fusion_mode,
                    use_layer_norm=use_layer_norm,
                    dropout=dropout,
                    batch_first=batch_first,
                    memory_key_dim=memory_key_dim,
                    memory_initialization_mode=memory_initialization_mode,
                    retrieval_query_source=retrieval_query_source,
                    segment_summary_source=segment_summary_source,
                    memory_decay=memory_decay,
                    enable_activation_checkpointing=enable_activation_checkpointing,
                    cuda_cpp_debug_fallback=cuda_cpp_debug_fallback,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_proj = nn.Linear(input_size, output_size)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        x: Tensor,
        h_init: Optional[Tensor] = None,
        c_init: Optional[Tensor] = None,
        return_all_outputs: bool = True,
        return_hidden_states: bool = False,
        reset_memory: bool = False,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Dict]]:
        batch_size = x.size(0) if self.batch_first else x.size(1)
        device = x.device

        if reset_memory:
            self.clear_memory()

        if h_init is None:
            h_init = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device, dtype=x.dtype)
        if c_init is None:
            memory_key_dim = self.layers[0].grm.memory_key_dim
            c_init = torch.zeros(
                self.num_layers,
                batch_size,
                self.hidden_size,
                memory_key_dim,
                device=device,
                dtype=x.dtype,
            )

        h_final_list = []
        c_final_list = []
        current_input = x
        aux = None

        for idx, layer in enumerate(self.layers):
            current_input, h_final, memory_final, aux = layer(
                current_input,
                h_init=h_init[idx],
                c_init=c_init[idx],
                return_all_outputs=return_all_outputs,
                return_hidden_states=return_hidden_states,
                reset_memory=False,
            )
            h_final_list.append(h_final)
            c_final_list.append(memory_final)

        outputs = self.dropout(self.output_proj(current_input))
        h_final = torch.stack(h_final_list, dim=0)
        c_final = torch.stack(c_final_list, dim=0)
        return outputs, h_final, c_final, aux

    def clear_memory(self) -> None:
        for layer in self.layers:
            layer.memory_bank.clear()

    def get_memory_info(self):
        return [layer.memory_bank.get_memory_info() for layer in self.layers]

    def get_runtime_optimization_status(self, device_type: Optional[str] = None) -> Dict[str, object]:
        return self.layers[0].grm.get_runtime_optimization_status(device_type=device_type)

    @property
    def memory_bank(self):
        return self.layers[0].memory_bank

    def extra_repr(self) -> str:
        return (
            f"input_size={self.input_size}, hidden_size={self.hidden_size}, "
            f"output_size={self.output_size}, num_layers={self.num_layers}, "
            f"rnn_type='{self.rnn_type}'"
        )


def build_hierarchical_segment_memory_model(
    input_size: int,
    hidden_size: int,
    output_size: int,
    num_layers: int = 6,
    **kwargs,
) -> HierarchicalSegmentMemoryModel:
    return HierarchicalSegmentMemoryModel(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        **kwargs,
    )
