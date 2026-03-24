"""Linear attention memory modules used by legacy GRM backends."""

import math
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class LinearMemoryState:
    """State for the linear-attention memory cell."""

    memory: Tensor
    h: Optional[Tensor] = None


class LinearMemoryCell(nn.Module):
    """Linear-attention memory cell with decayed rank-1 updates."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        memory_key_dim: Optional[int] = None,
        use_bias: bool = True,
        memory_init: Literal["zeros", "eye", "random"] = "zeros",
        memory_decay: float = 0.97,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.memory_key_dim = memory_key_dim or hidden_size
        self.use_bias = use_bias
        self.memory_init = memory_init
        self.memory_decay = memory_decay

        self.key_proj = nn.Linear(input_size, self.memory_key_dim, bias=use_bias)
        self.value_proj = nn.Linear(input_size, hidden_size, bias=use_bias)
        self.query_proj = nn.Linear(input_size, self.memory_key_dim, bias=use_bias)
        self.output_proj = nn.Linear(hidden_size, input_size, bias=use_bias)

        self._init_weights()

    def _init_weights(self) -> None:
        for proj in [self.key_proj, self.value_proj, self.query_proj]:
            nn.init.xavier_uniform_(proj.weight)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)

        nn.init.xavier_uniform_(self.output_proj.weight)
        if self.output_proj.bias is not None:
            nn.init.zeros_(self.output_proj.bias)

    def init_memory(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> LinearMemoryState:
        if self.memory_init == "zeros":
            memory = torch.zeros(
                batch_size,
                self.hidden_size,
                self.memory_key_dim,
                device=device,
                dtype=dtype,
            )
        elif self.memory_init == "eye":
            eye = torch.eye(
                self.hidden_size,
                self.memory_key_dim,
                device=device,
                dtype=dtype,
            )
            memory = eye.unsqueeze(0).expand(batch_size, -1, -1).clone()
        elif self.memory_init == "random":
            memory = (
                torch.randn(
                    batch_size,
                    self.hidden_size,
                    self.memory_key_dim,
                    device=device,
                    dtype=dtype,
                )
                * 0.02
            )
        else:
            raise ValueError(f"Unknown memory_init: {self.memory_init}")

        return LinearMemoryState(memory=memory)

    def forward(
        self,
        x_t: Tensor,
        state: Optional[LinearMemoryState] = None,
    ) -> Tuple[Tensor, LinearMemoryState]:
        batch_size = x_t.size(0)
        device = x_t.device

        if state is None:
            state = self.init_memory(batch_size, device, x_t.dtype)

        memory = state.memory
        k_t = F.normalize(self.key_proj(x_t), p=2, dim=-1, eps=1e-6)
        v_t = self.value_proj(x_t)
        q_t = F.normalize(self.query_proj(x_t), p=2, dim=-1, eps=1e-6)

        memory_update = torch.bmm(v_t.unsqueeze(2), k_t.unsqueeze(1))
        memory_update = memory_update / math.sqrt(self.memory_key_dim)
        new_memory = self.memory_decay * memory + (1.0 - self.memory_decay) * memory_update

        hidden = torch.bmm(new_memory, q_t.unsqueeze(2)).squeeze(2)
        output = self.output_proj(hidden)
        new_state = LinearMemoryState(memory=new_memory, h=hidden)

        return output, new_state

    def forward_sequence(
        self,
        x_seq: Tensor,
        state: Optional[LinearMemoryState] = None,
        return_all_outputs: bool = True,
    ) -> Tuple[Tensor, LinearMemoryState]:
        batch_size, seq_len, _ = x_seq.shape
        device = x_seq.device

        if state is None:
            state = self.init_memory(batch_size, device, x_seq.dtype)

        memory = state.memory
        outputs = []
        hidden = torch.zeros(batch_size, self.hidden_size, device=device, dtype=x_seq.dtype)
        output = self.output_proj(hidden)

        if seq_len == 0:
            if return_all_outputs:
                outputs_tensor = x_seq.new_zeros(batch_size, 0, self.input_size)
            else:
                outputs_tensor = output
            return outputs_tensor, LinearMemoryState(memory=memory, h=hidden)

        for t in range(seq_len):
            x_t = x_seq[:, t]
            k_t = F.normalize(self.key_proj(x_t), p=2, dim=-1, eps=1e-6)
            v_t = self.value_proj(x_t)
            q_t = F.normalize(self.query_proj(x_t), p=2, dim=-1, eps=1e-6)

            memory_update = torch.bmm(v_t.unsqueeze(2), k_t.unsqueeze(1))
            memory_update = memory_update / math.sqrt(self.memory_key_dim)
            memory = self.memory_decay * memory + (1.0 - self.memory_decay) * memory_update

            hidden = torch.bmm(memory, q_t.unsqueeze(2)).squeeze(2)
            output = self.output_proj(hidden)

            if return_all_outputs:
                outputs.append(output)

        if return_all_outputs:
            outputs_tensor = torch.stack(outputs, dim=1)
        else:
            outputs_tensor = output

        return outputs_tensor, LinearMemoryState(memory=memory, h=hidden)


class DeepLinearMemoryCell(nn.Module):
    """Deep linear-attention memory cell with per-sample fast weights."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        memory_key_dim: Optional[int] = None,
        num_layers: int = 2,
        expansion: int = 4,
        activation: Literal["gelu", "relu"] = "gelu",
        lr: float = 0.01,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.memory_key_dim = memory_key_dim or hidden_size
        self.num_layers = num_layers
        self.expansion = expansion
        self.activation = activation
        self.lr = lr

        self.key_proj = nn.Linear(input_size, self.memory_key_dim, bias=False)

        layers = []
        current_dim = self.memory_key_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, current_dim * expansion, bias=False))
            layers.append(nn.GELU() if activation == "gelu" else nn.ReLU())
            current_dim = current_dim * expansion
        layers.append(nn.Linear(current_dim, hidden_size, bias=False))
        self.memory_mlp = nn.Sequential(*layers)

        self.query_proj = nn.Linear(input_size, self.memory_key_dim, bias=False)
        self.value_proj = nn.Linear(input_size, hidden_size, bias=False)
        self.output_proj = nn.Linear(hidden_size, input_size, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        for layer in self.memory_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

        for proj in [self.key_proj, self.query_proj, self.value_proj, self.output_proj]:
            nn.init.xavier_uniform_(proj.weight)

    def forward(
        self,
        x_t: Tensor,
        memory_params: Optional[dict] = None,
    ) -> Tuple[Tensor, dict]:
        batch_size = x_t.size(0)
        k_t = self.key_proj(x_t)
        v_t = self.value_proj(x_t)

        working_params = self._prepare_memory_params(
            memory_params=memory_params,
            batch_size=batch_size,
        )
        hidden = self._forward_with_params(k_t, working_params)

        if batch_size == 0:
            output = self.output_proj(hidden)
            return output, working_params

        updated_chunks = {name: [] for name in working_params}
        for batch_idx in range(batch_size):
            sample_params = {
                name: param[batch_idx : batch_idx + 1]
                for name, param in working_params.items()
            }
            sample_hidden = self._forward_with_params(
                k_t[batch_idx : batch_idx + 1],
                sample_params,
            )
            sample_loss = 0.5 * F.mse_loss(
                sample_hidden,
                v_t[batch_idx : batch_idx + 1],
                reduction="mean",
            )
            grads = torch.autograd.grad(
                sample_loss,
                tuple(sample_params.values()),
                allow_unused=True,
                create_graph=torch.is_grad_enabled(),
            )
            for (name, param), grad in zip(sample_params.items(), grads):
                updated_chunks[name].append(param if grad is None else param - self.lr * grad)

        new_memory_params = {
            name: torch.cat(chunks, dim=0)
            for name, chunks in updated_chunks.items()
        }

        output = self.output_proj(hidden)
        return output, new_memory_params

    def _forward_with_params(self, x: Tensor, params: dict) -> Tensor:
        layers = list(self.memory_mlp)
        i = 0

        while i < len(layers):
            layer = layers[i]
            if isinstance(layer, nn.Linear):
                weight = params.get(f"{i}.weight", layer.weight)
                bias = params.get(f"{i}.bias", layer.bias)
                x = self._linear_with_params(x, weight, bias)
                i += 1
            elif isinstance(layer, (nn.GELU, nn.ReLU)):
                x = layer(x)
                i += 1
            else:
                x = layer(x)
                i += 1

        return x

    def _prepare_memory_params(
        self,
        memory_params: Optional[dict],
        batch_size: int,
    ) -> dict:
        raw_params = (
            memory_params
            if memory_params is not None
            else {name: param for name, param in self.memory_mlp.named_parameters()}
        )

        prepared_params = {}
        for name, param in raw_params.items():
            if param.dim() >= 3:
                if param.size(0) != batch_size:
                    raise ValueError(
                        f"memory_params[{name!r}] batch dimension mismatch: "
                        f"expected {batch_size}, got {param.size(0)}"
                    )
                prepared = param
            else:
                prepared = param.unsqueeze(0).expand(batch_size, *param.shape).clone()

            if not prepared.requires_grad:
                prepared = prepared.requires_grad_(True)
            prepared_params[name] = prepared

        return prepared_params

    def _linear_with_params(
        self,
        x: Tensor,
        weight: Tensor,
        bias: Optional[Tensor] = None,
    ) -> Tensor:
        if weight.dim() == 2:
            return F.linear(x, weight, bias)

        output = torch.bmm(weight, x.unsqueeze(-1)).squeeze(-1)
        if bias is not None:
            output = output + bias
        return output

    def init_memory_params(self, batch_size: int = 1) -> dict:
        return {
            name: p.unsqueeze(0).expand(batch_size, *p.shape).clone().requires_grad_(True)
            for name, p in self.memory_mlp.named_parameters()
        }


class LinearSegmentMemoryWrapper(nn.Module):
    """Adapter that exposes linear-attention cells through the GRM RNN API."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        memory_type: Literal["linear", "deep_linear"] = "linear",
        **kwargs,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.memory_type = memory_type

        if memory_type == "linear":
            self.memory_cell = LinearMemoryCell(
                input_size=input_size,
                hidden_size=hidden_size,
                **kwargs,
            )
        elif memory_type == "deep_linear":
            self.memory_cell = DeepLinearMemoryCell(
                input_size=input_size,
                hidden_size=hidden_size,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown memory_type: {memory_type}")

    def init_hidden(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Tuple:
        if self.memory_type == "linear":
            state = self.memory_cell.init_memory(batch_size, device)
            return state.memory, None

        params = self.memory_cell.init_memory_params(batch_size=batch_size)
        placeholder = torch.zeros(batch_size, self.hidden_size, device=device)
        return placeholder, params

    def forward(
        self,
        x_t: Tensor,
        h_prev: Optional[Tensor] = None,
        c_prev: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if self.memory_type == "linear":
            state = LinearMemoryState(memory=h_prev) if h_prev is not None else None
            _, new_state = self.memory_cell(x_t, state)
            return new_state.h, None

        output, new_params = self.memory_cell(x_t, c_prev)
        return output, new_params


def create_linear_memory(
    input_size: int,
    hidden_size: int,
    memory_type: str = "linear",
    **kwargs,
) -> LinearSegmentMemoryWrapper:
    """Factory for the legacy linear-attention wrappers."""

    return LinearSegmentMemoryWrapper(
        input_size=input_size,
        hidden_size=hidden_size,
        memory_type=memory_type,
        **kwargs,
    )
