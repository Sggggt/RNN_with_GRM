"""Fusion operators for online states and retrieved memory states."""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn
from torch import Tensor


class RetrievedStateFusion(nn.Module):
    """Fuse the online recurrent state with retrieved segment states."""

    def __init__(
        self,
        hidden_size: int,
        retrieval_fusion_mode: Literal["residual", "concat", "gate", "linear", "memory_soup"] = "residual",
        use_layer_norm: bool = True,
        use_output_projection: bool = False,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.retrieval_fusion_mode = retrieval_fusion_mode
        self.use_layer_norm = use_layer_norm

        self.layer_norm = nn.LayerNorm(hidden_size) if use_layer_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.output_projection: Optional[nn.Linear] = None
        self.gate_projection: Optional[nn.Linear] = None
        self.online_projection: Optional[nn.Linear] = None
        self.memory_projection: Optional[nn.Linear] = None
        self.soup_projection: Optional[nn.Linear] = None

        if retrieval_fusion_mode == "residual":
            if use_output_projection:
                self.output_projection = nn.Linear(hidden_size, hidden_size)
        elif retrieval_fusion_mode == "concat":
            self.output_projection = nn.Linear(hidden_size * 2, hidden_size)
        elif retrieval_fusion_mode == "gate":
            self.gate_projection = nn.Linear(hidden_size * 2, hidden_size)
        elif retrieval_fusion_mode == "linear":
            self.online_projection = nn.Linear(hidden_size, hidden_size, bias=False)
            self.memory_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        elif retrieval_fusion_mode == "memory_soup":
            self.soup_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        else:
            raise ValueError(f"Unknown retrieval_fusion_mode: {retrieval_fusion_mode}")

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in (
            self.output_projection,
            self.gate_projection,
            self.online_projection,
            self.memory_projection,
            self.soup_projection,
        ):
            if module is None:
                continue
            nn.init.xavier_uniform_(module.weight)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def _combine_retrieved_states(
        self,
        online_state: Tensor,
        retrieved_states: Tensor,
        routing_weights: Optional[Tensor] = None,
    ) -> tuple[Tensor, Optional[Tensor]]:
        if retrieved_states.size(1) == 0:
            return torch.zeros_like(online_state), None

        batch_size, num_routes, _ = retrieved_states.shape
        device = online_state.device

        if routing_weights is None:
            routing_weights = torch.full(
                (batch_size, num_routes),
                fill_value=1.0 / num_routes,
                device=device,
                dtype=online_state.dtype,
            )

        if routing_weights.dim() == 1:
            routing_weights = routing_weights.unsqueeze(0)
        if routing_weights.size(0) == 1 and batch_size > 1:
            routing_weights = routing_weights.expand(batch_size, -1)
        if routing_weights.size(1) != num_routes:
            if routing_weights.size(1) < num_routes:
                padding = torch.zeros(
                    batch_size,
                    num_routes - routing_weights.size(1),
                    device=device,
                    dtype=routing_weights.dtype,
                )
                routing_weights = torch.cat([routing_weights, padding], dim=-1)
            else:
                routing_weights = routing_weights[:, :num_routes]
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True).clamp(min=1e-9)

        weighted_states = retrieved_states * routing_weights.unsqueeze(-1)
        return weighted_states.sum(dim=1), routing_weights

    def _apply_fusion_rule(
        self,
        online_state: Tensor,
        retrieved_contribution: Tensor,
        retrieved_states: Tensor,
        routing_weights: Optional[Tensor],
    ) -> Tensor:
        mode = self.retrieval_fusion_mode

        if mode == "residual":
            fused_state = online_state + retrieved_contribution
            if self.output_projection is not None:
                fused_state = self.output_projection(fused_state)
        elif mode == "concat":
            fused_state = self.output_projection(torch.cat([online_state, retrieved_contribution], dim=-1))
        elif mode == "gate":
            gate = torch.sigmoid(self.gate_projection(torch.cat([online_state, retrieved_contribution], dim=-1)))
            fused_state = gate * online_state + (1.0 - gate) * retrieved_contribution
        elif mode == "linear":
            fused_state = self.online_projection(online_state) + self.memory_projection(retrieved_contribution)
        elif mode == "memory_soup":
            if retrieved_states.size(1) == 0:
                soup_input = online_state
            else:
                weights = routing_weights
                if weights is None:
                    weights = torch.full(
                        (retrieved_states.size(0), retrieved_states.size(1)),
                        fill_value=1.0 / max(1, retrieved_states.size(1)),
                        device=online_state.device,
                        dtype=online_state.dtype,
                    )
                soup_input = online_state + (retrieved_states * weights.unsqueeze(-1)).sum(dim=1)
            fused_state = self.soup_projection(soup_input)
        else:
            raise ValueError(f"Unknown retrieval_fusion_mode: {mode}")

        fused_state = self.layer_norm(fused_state)
        return self.dropout(fused_state)

    def forward(
        self,
        online_state: Optional[Tensor] = None,
        retrieved_states: Optional[Tensor] = None,
        routing_weights: Optional[Tensor] = None,
        **legacy_kwargs,
    ) -> Tensor:
        """Fuse online and retrieved states, while accepting legacy keyword aliases."""
        if online_state is None:
            online_state = legacy_kwargs.pop("h_online", None)
        if retrieved_states is None:
            retrieved_states = legacy_kwargs.pop("cached_states", None)
        if routing_weights is None:
            routing_weights = legacy_kwargs.pop("gates", None)
        if legacy_kwargs:
            unexpected = ", ".join(sorted(legacy_kwargs))
            raise TypeError(f"Unexpected keyword arguments: {unexpected}")
        if online_state is None or retrieved_states is None:
            raise TypeError("`online_state` and `retrieved_states` must be provided.")

        retrieved_contribution, routing_weights = self._combine_retrieved_states(
            online_state,
            retrieved_states,
            routing_weights,
        )
        return self._apply_fusion_rule(
            online_state,
            retrieved_contribution,
            retrieved_states,
            routing_weights,
        )

    def forward_with_auxiliary_statistics(
        self,
        online_state: Optional[Tensor] = None,
        retrieved_states: Optional[Tensor] = None,
        routing_weights: Optional[Tensor] = None,
        **legacy_kwargs,
    ) -> tuple[Tensor, dict[str, float]]:
        """Return fused states together with simple routing diagnostics."""
        if online_state is None:
            online_state = legacy_kwargs.pop("h_online", None)
        if retrieved_states is None:
            retrieved_states = legacy_kwargs.pop("cached_states", None)
        if routing_weights is None:
            routing_weights = legacy_kwargs.pop("gates", None)
        if legacy_kwargs:
            unexpected = ", ".join(sorted(legacy_kwargs))
            raise TypeError(f"Unexpected keyword arguments: {unexpected}")
        if online_state is None or retrieved_states is None:
            raise TypeError("`online_state` and `retrieved_states` must be provided.")

        retrieved_contribution, routing_weights = self._combine_retrieved_states(
            online_state,
            retrieved_states,
            routing_weights,
        )
        fused_state = self._apply_fusion_rule(
            online_state,
            retrieved_contribution,
            retrieved_states,
            routing_weights,
        )

        if routing_weights is None:
            entropy = 0.0
        else:
            entropy = float(
                (-(routing_weights * torch.log(routing_weights.clamp(min=1e-10))).sum(dim=-1).mean()).item()
            )

        auxiliary_statistics = {
            "online_norm": float(online_state.norm().item()),
            "retrieved_norm": float(retrieved_contribution.norm().item()),
            "num_retrieved": int(retrieved_states.size(1)),
            "routing_entropy": entropy,
        }
        return fused_state, auxiliary_statistics

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, "
            f"retrieval_fusion_mode='{self.retrieval_fusion_mode}', "
            f"use_layer_norm={self.use_layer_norm}"
        )


class ParameterMixtureFusion(nn.Module):
    """Approximate the paper's parameter-mixture memory-soup variant."""

    def __init__(
        self,
        hidden_size: int,
        num_memory_layers: int = 2,
        segment_pooling_mode: str = "mean",
        use_layer_norm: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_memory_layers = num_memory_layers
        self.segment_pooling_mode = segment_pooling_mode
        self.segment_weights = nn.ParameterList(
            [nn.Parameter(torch.full((hidden_size,), 0.1)) for _ in range(num_memory_layers)]
        )
        self.layer_norm = nn.LayerNorm(hidden_size) if use_layer_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        online_state: Tensor,
        retrieved_states: Tensor,
        routing_weights: Optional[Tensor] = None,
    ) -> Tensor:
        if retrieved_states.size(1) == 0:
            return self.dropout(self.layer_norm(online_state))

        batch_size, num_routes, _ = retrieved_states.shape
        if routing_weights is None:
            routing_weights = torch.full(
                (batch_size, num_routes),
                fill_value=1.0 / num_routes,
                device=online_state.device,
                dtype=online_state.dtype,
            )

        mixed_states = []
        for weight_vector in self.segment_weights:
            projected_states = retrieved_states * weight_vector.view(1, 1, -1)
            mixed_states.append((projected_states * routing_weights.unsqueeze(-1)).sum(dim=1))

        mixture = mixed_states[0] if len(mixed_states) == 1 else torch.stack(mixed_states, dim=0).mean(dim=0)
        return self.dropout(self.layer_norm(online_state + mixture))


def build_retrieved_state_fusion(
    hidden_size: int,
    retrieval_fusion_mode: str = "residual",
    **kwargs,
) -> RetrievedStateFusion:
    return RetrievedStateFusion(
        hidden_size=hidden_size,
        retrieval_fusion_mode=retrieval_fusion_mode,
        **kwargs,
    )


__all__ = [
    "ParameterMixtureFusion",
    "RetrievedStateFusion",
    "build_retrieved_state_fusion",
]
