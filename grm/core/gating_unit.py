"""Sparse retrieval modules for segment-level memory routing."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TopKMemoryRetriever(nn.Module):
    """Top-k sparse retriever over segment summary keys."""

    def __init__(
        self,
        query_dim: int,
        key_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        retrieval_top_k: int = 16,
        temperature: float = 1.0,
        use_causal_mask: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.query_dim = query_dim
        self.key_dim = key_dim or query_dim
        self.hidden_dim = hidden_dim or query_dim
        self.retrieval_top_k = retrieval_top_k
        self.temperature = temperature
        self.use_causal_mask = use_causal_mask

        self.query_projection = nn.Linear(self.query_dim, self.hidden_dim, bias=False)
        self.key_projection = nn.Linear(self.key_dim, self.hidden_dim, bias=False)
        self.query_proj = self.query_projection
        self.key_proj = self.key_projection
        self.output_projection = (
            nn.Linear(self.hidden_dim, self.query_dim, bias=False)
            if self.hidden_dim != self.query_dim
            else None
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.scale = 1.0 / math.sqrt(self.hidden_dim)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.query_projection.weight)
        nn.init.xavier_uniform_(self.key_projection.weight)
        if self.output_projection is not None:
            nn.init.xavier_uniform_(self.output_projection.weight)

    def forward(
        self,
        query: Tensor,
        keys: Tensor,
        segment_offset: int = 0,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        batch_size = query.size(0)
        device = query.device

        if keys.ndim == 2:
            keys = keys.unsqueeze(0).expand(batch_size, -1, -1)

        projected_query = self.query_projection(query)
        projected_keys = self.key_projection(keys)
        similarity_scores = torch.matmul(
            projected_query.unsqueeze(1),
            projected_keys.transpose(-2, -1),
        ).squeeze(1) * self.scale

        if attention_mask is not None:
            similarity_scores = similarity_scores.masked_fill(~attention_mask.bool(), float("-inf"))

        num_keys = keys.size(1)
        if self.use_causal_mask and 0 < segment_offset < num_keys:
            causal_mask = torch.arange(num_keys, device=device).expand(batch_size, -1) >= segment_offset
            similarity_scores = similarity_scores.masked_fill(causal_mask, float("-inf"))

        active_routes = min(self.retrieval_top_k, num_keys)
        if active_routes < num_keys:
            top_scores, top_indices = torch.topk(similarity_scores, k=active_routes, dim=-1)
        else:
            top_scores = similarity_scores
            top_indices = torch.arange(num_keys, device=device).unsqueeze(0).expand(batch_size, -1)

        routing_weights = F.softmax(top_scores / self.temperature, dim=-1)
        routing_weights = self.dropout(routing_weights)
        return routing_weights, top_indices, similarity_scores

    def compute_dense_routing_weights(
        self,
        query: Tensor,
        keys: Tensor,
        segment_offset: int = 0,
    ) -> Tensor:
        batch_size = query.size(0)
        if keys.ndim == 2:
            keys = keys.unsqueeze(0).expand(batch_size, -1, -1)

        projected_query = self.query_projection(query)
        projected_keys = self.key_projection(keys)
        similarity_scores = torch.matmul(
            projected_query.unsqueeze(1),
            projected_keys.transpose(-2, -1),
        ).squeeze(1) * self.scale

        num_keys = keys.size(1)
        if self.use_causal_mask and 0 < segment_offset < num_keys:
            device = query.device
            causal_mask = torch.arange(num_keys, device=device).expand(batch_size, -1) >= segment_offset
            similarity_scores = similarity_scores.masked_fill(causal_mask, float("-inf"))

        return F.softmax(similarity_scores / self.temperature, dim=-1)

    def extra_repr(self) -> str:
        return (
            f"query_dim={self.query_dim}, "
            f"key_dim={self.key_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"retrieval_top_k={self.retrieval_top_k}, "
            f"temperature={self.temperature}, "
            f"use_causal_mask={self.use_causal_mask}"
        )


def build_topk_memory_retriever(
    query_dim: int,
    key_dim: Optional[int] = None,
    retrieval_top_k: int = 16,
    **kwargs,
) -> TopKMemoryRetriever:
    return TopKMemoryRetriever(
        query_dim=query_dim,
        key_dim=key_dim,
        retrieval_top_k=retrieval_top_k,
        **kwargs,
    )


__all__ = ["TopKMemoryRetriever", "build_topk_memory_retriever"]
