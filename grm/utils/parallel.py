"""Parallel and distributed training helpers for segment-memory experiments."""

from __future__ import annotations

import copy
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP


ParallelMode = Literal["none", "data", "segment", "hybrid"]
SyncStrategy = Literal["full", "async", "lazy"]


class ReplicatedTrainingEngine(nn.Module):
    """Wrap a model in `DataParallel` or `DistributedDataParallel` when available."""

    def __init__(
        self,
        module: nn.Module,
        device_ids: Optional[List[int]] = None,
        output_device: Optional[int] = None,
        use_ddp: bool = False,
        find_unused_parameters: bool = False,
    ) -> None:
        super().__init__()

        self.original_module = module
        self.device_ids = device_ids or []
        self.output_device = output_device
        self.use_ddp = use_ddp
        self.find_unused_parameters = find_unused_parameters

        self.num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.available_devices = list(range(self.num_devices))
        self.parallel_module: nn.Module
        self.is_parallel = False

        if self.num_devices <= 1 or len(self.device_ids) <= 1:
            self.parallel_module = module
            return

        valid_ids = [device_id for device_id in self.device_ids if device_id in self.available_devices]
        if len(valid_ids) <= 1:
            self.parallel_module = module
            return

        primary_device = torch.device(f"cuda:{valid_ids[0]}")
        module = module.to(primary_device)
        self.original_module = module
        self.is_parallel = True

        if use_ddp and dist.is_available() and dist.is_initialized():
            self.parallel_module = DDP(
                module,
                device_ids=[valid_ids[0]],
                output_device=valid_ids[0],
                find_unused_parameters=find_unused_parameters,
            )
        else:
            self.parallel_module = nn.DataParallel(
                module,
                device_ids=valid_ids,
                output_device=output_device or valid_ids[0],
            )

    def forward(
        self,
        x_seq: Tensor,
        h_init: Optional[Tensor] = None,
        c_init: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Dict[str, Any]]]:
        """Execute a forward pass through the wrapped parallel module."""
        return self.parallel_module(x_seq, h_init=h_init, c_init=c_init, **kwargs)

    def backward(self, loss: Tensor) -> None:
        """Backpropagate the provided scalar loss."""
        loss.backward()

    def train(self, mode: bool = True) -> "ReplicatedTrainingEngine":
        """Set training mode on the wrapped module."""
        self.parallel_module.train(mode)
        return self

    def eval(self) -> "ReplicatedTrainingEngine":
        """Set evaluation mode on the wrapped module."""
        self.parallel_module.eval()
        return self

    def to(self, device: torch.device) -> "ReplicatedTrainingEngine":
        """Move the non-parallel path to a target device."""
        if not self.is_parallel:
            self.parallel_module = self.parallel_module.to(device)
            self.original_module = self.parallel_module
        return self

    @contextmanager
    def no_sync(self):
        """Temporarily disable gradient synchronization for DDP accumulation steps."""
        if isinstance(self.parallel_module, DDP):
            with self.parallel_module.no_sync():
                yield
            return
        yield

    def extra_repr(self) -> str:
        """Return a concise summary of the parallel wrapper state."""
        return (
            f"num_devices={self.num_devices}, "
            f"is_parallel={self.is_parallel}, "
            f"use_ddp={self.use_ddp}"
        )


class SegmentParallelExecutor(nn.Module):
    """Distribute sequence segments across replicated worker models."""

    def __init__(
        self,
        module: nn.Module,
        num_workers: int = 4,
        segment_length: int = 64,
        sync_strategy: SyncStrategy = "async",
    ) -> None:
        super().__init__()

        self.original_module = module
        self.num_workers = num_workers
        self.segment_length = segment_length
        self.sync_strategy = sync_strategy

        self.num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.is_parallel = self.num_devices > 1 and num_workers > 1
        self.worker_devices: List[int]

        if not self.is_parallel:
            self.worker_devices = [0] if self.num_devices == 1 else []
            self.worker_models = nn.ModuleList()
            return

        devices_per_worker = max(1, self.num_devices // num_workers)
        self.worker_devices = [
            min(worker_index * devices_per_worker, self.num_devices - 1)
            for worker_index in range(num_workers)
        ]
        self.worker_models = nn.ModuleList()
        for device_id in self.worker_devices:
            worker_model = copy.deepcopy(module).to(torch.device(f"cuda:{device_id}"))
            self.worker_models.append(worker_model)

    def forward(
        self,
        x_seq: Tensor,
        h_init: Optional[Tensor] = None,
        c_init: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Dict[str, Any]]]:
        """Process a sequence by dispatching consecutive segments to worker replicas."""
        if not self.is_parallel or not self.worker_devices:
            return self.original_module(x_seq, h_init=h_init, c_init=c_init, **kwargs)

        batch_first = getattr(self.original_module, "batch_first", True)
        if x_seq.ndim == 2:
            x_seq = x_seq.unsqueeze(1 if batch_first else 0)

        if batch_first:
            batch_size = x_seq.size(0)
            sequence_length = x_seq.size(1)
            concat_dim = 1
        else:
            sequence_length = x_seq.size(0)
            batch_size = x_seq.size(1)
            concat_dim = 0
        device = x_seq.device

        default_h, default_c = self._build_default_state(batch_size=batch_size, device=device, dtype=x_seq.dtype)
        h_t = default_h if h_init is None else h_init
        c_t = default_c if c_init is None else c_init

        outputs: List[Tensor] = []
        num_segments = (sequence_length + self.segment_length - 1) // self.segment_length

        for segment_index in range(num_segments):
            start_index = segment_index * self.segment_length
            end_index = min(start_index + self.segment_length, sequence_length)
            segment_input = (
                x_seq[:, start_index:end_index, :]
                if batch_first
                else x_seq[start_index:end_index, :, :]
            )

            worker_index = segment_index % len(self.worker_models)
            worker_device = torch.device(f"cuda:{self.worker_devices[worker_index]}")
            worker_model = self.worker_models[worker_index]

            segment_input = segment_input.to(worker_device)
            h_worker = h_t.to(worker_device)
            c_worker = c_t.to(worker_device) if c_t is not None else None

            segment_output, h_worker, c_worker, _ = worker_model(
                segment_input,
                h_init=h_worker,
                c_init=c_worker,
                return_all_outputs=True,
                **kwargs,
            )

            outputs.append(segment_output.to(device))
            h_t = h_worker.to(device)
            if c_worker is not None:
                c_t = c_worker.to(device)

        return torch.cat(outputs, dim=concat_dim), h_t, c_t, None

    def _build_default_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Construct default hidden and memory states for the wrapped recurrent model."""
        if hasattr(self.original_module, "init_hidden"):
            hidden, memory = self.original_module.init_hidden(batch_size, device)
            hidden = hidden.to(device=device, dtype=dtype)
            if memory is not None:
                memory = memory.to(device=device, dtype=dtype)
            return hidden, memory

        hidden = torch.zeros(batch_size, self.original_module.hidden_size, device=device, dtype=dtype)
        memory = None
        return hidden, memory

    def extra_repr(self) -> str:
        """Return a concise summary of the executor layout."""
        return (
            f"num_workers={self.num_workers}, "
            f"segment_length={self.segment_length}, "
            f"sync_strategy='{self.sync_strategy}', "
            f"is_parallel={self.is_parallel}"
        )


class ParallelExperimentTrainer:
    """Lightweight training helper for optional multi-GPU experiment execution."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        parallel_mode: ParallelMode = "none",
        num_workers: int = 4,
        gradient_accumulation_factor: int = 1,
        mixed_precision: bool = False,
    ) -> None:
        self.device = device
        self.parallel_mode = parallel_mode
        self.num_workers = num_workers
        self.gradient_accumulation_factor = max(1, gradient_accumulation_factor)
        self.mixed_precision = mixed_precision
        self.optimizer = optimizer
        self.criterion = criterion
        self.scaler = torch.amp.GradScaler("cuda", enabled=(mixed_precision and torch.cuda.is_available()))

        self.parallel_model = self._build_parallel_model(model)
        self.current_step = 0
        self.total_steps = 0

    def _build_parallel_model(self, model: nn.Module) -> nn.Module:
        """Instantiate the requested parallel execution strategy."""
        if self.parallel_mode == "none":
            return model.to(self.device)
        if self.parallel_mode == "data":
            return ReplicatedTrainingEngine(
                model,
                device_ids=list(range(self.num_workers)) if self.num_workers > 1 else None,
            )
        if self.parallel_mode == "segment":
            return SegmentParallelExecutor(model, num_workers=self.num_workers)
        if self.parallel_mode == "hybrid":
            return ReplicatedTrainingEngine(
                model,
                device_ids=list(range(self.num_workers)) if self.num_workers > 1 else None,
            )
        raise ValueError(f"Unknown parallel_mode: {self.parallel_mode}")

    def train_epoch(
        self,
        dataloader: Iterable,
        epoch: int = 0,
        gradient_clip_norm: float = 1.0,
    ) -> Dict[str, float]:
        """Run one training epoch for the configured parallel strategy."""
        self.parallel_model.train()
        self.optimizer.zero_grad()

        total_loss = 0.0
        num_batches = 0

        for batch_index, batch in enumerate(dataloader):
            use_no_sync = hasattr(self.parallel_model, "no_sync") and (
                (batch_index + 1) % self.gradient_accumulation_factor != 0
            )
            if use_no_sync:
                with self.parallel_model.no_sync():
                    loss = self._train_step(batch)
            else:
                loss = self._train_step(batch)

            if (batch_index + 1) % self.gradient_accumulation_factor == 0:
                self._optimizer_step(gradient_clip_norm)

            total_loss += float(loss.item())
            num_batches += 1

        return {
            "loss": total_loss / max(1, num_batches),
            "epoch": epoch,
            "step": self.current_step,
        }

    def _train_step(self, batch: Any) -> Tensor:
        """Execute a single gradient-accumulation step."""
        inputs, targets = self._move_batch_to_device(batch)
        if self.mixed_precision:
            with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda")):
                outputs, _, _, _ = self.parallel_model(inputs)
                loss = self.criterion(outputs, targets)
            self.scaler.scale(loss / self.gradient_accumulation_factor).backward()
        else:
            outputs, _, _, _ = self.parallel_model(inputs)
            loss = self.criterion(outputs, targets)
            (loss / self.gradient_accumulation_factor).backward()
        return loss

    def _optimizer_step(self, gradient_clip_norm: float) -> None:
        """Apply one optimizer step after optional gradient clipping."""
        if self.mixed_precision:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.parallel_model.parameters(), gradient_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(self.parallel_model.parameters(), gradient_clip_norm)
            self.optimizer.step()
        self.optimizer.zero_grad()
        self.current_step += 1

    def _move_batch_to_device(self, batch: Any) -> Tuple[Tensor, Tensor]:
        """Move a supervised or self-supervised batch to the target device."""
        if isinstance(batch, (tuple, list)):
            inputs = batch[0].to(self.device)
            targets = batch[1].to(self.device)
            return inputs, targets

        inputs = batch.to(self.device)
        return inputs, inputs

    @torch.no_grad()
    def evaluate(self, dataloader: Iterable) -> Dict[str, float]:
        """Evaluate the current model under the configured parallel strategy."""
        self.parallel_model.eval()

        total_loss = 0.0
        num_batches = 0
        for batch in dataloader:
            inputs, targets = self._move_batch_to_device(batch)
            outputs, _, _, _ = self.parallel_model(inputs)
            loss = self.criterion(outputs, targets)
            total_loss += float(loss.item())
            num_batches += 1

        return {"loss": total_loss / max(1, num_batches)}

    def save_checkpoint(
        self,
        filepath: str,
        epoch: int,
        additional_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Serialize model, optimizer, and mixed-precision state."""
        checkpoint = {
            "epoch": epoch,
            "step": self.current_step,
            "model_state_dict": self._get_model_state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "parallel_mode": self.parallel_mode,
        }
        if self.mixed_precision:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        if additional_info:
            checkpoint.update(additional_info)
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str) -> Dict[str, Any]:
        """Restore trainer state from a serialized checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self._load_model_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_step = checkpoint.get("step", 0)
        if self.mixed_precision and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        return checkpoint

    def _get_model_state_dict(self) -> Dict[str, Tensor]:
        """Return the underlying model weights regardless of wrapper type."""
        if isinstance(self.parallel_model, ReplicatedTrainingEngine):
            return self.parallel_model.original_module.state_dict()
        if isinstance(self.parallel_model, SegmentParallelExecutor):
            return self.parallel_model.original_module.state_dict()
        return self.parallel_model.state_dict()

    def _load_model_state_dict(self, state_dict: Dict[str, Tensor]) -> None:
        """Load weights into the underlying model regardless of wrapper type."""
        if isinstance(self.parallel_model, ReplicatedTrainingEngine):
            self.parallel_model.original_module.load_state_dict(state_dict)
            return
        if isinstance(self.parallel_model, SegmentParallelExecutor):
            self.parallel_model.original_module.load_state_dict(state_dict)
            for worker_model in self.parallel_model.worker_models:
                worker_model.load_state_dict(state_dict)
            return
        self.parallel_model.load_state_dict(state_dict)

    def get_model(self) -> nn.Module:
        """Expose the underlying model without its parallel wrapper."""
        if isinstance(self.parallel_model, ReplicatedTrainingEngine):
            return self.parallel_model.original_module
        if isinstance(self.parallel_model, SegmentParallelExecutor):
            return self.parallel_model.original_module
        return self.parallel_model


def build_parallel_config(
    parallel_mode: ParallelMode = "none",
    num_gpus: int = 0,
) -> Dict[str, Any]:
    """Build a minimal runtime configuration for optional multi-GPU execution."""
    if num_gpus == 0:
        num_gpus = torch.cuda.device_count()
    if num_gpus <= 1:
        parallel_mode = "none"
    return {
        "parallel_mode": parallel_mode,
        "num_workers": max(1, num_gpus),
        "device": torch.device("cuda:0" if num_gpus > 0 else "cpu"),
    }


def initialize_distributed_training(
    backend: Literal["nccl", "gloo", "mpi"] = "nccl",
) -> Tuple[int, int]:
    """Initialize a distributed process group and return `(rank, world_size)`."""
    if not dist.is_available():
        return 0, 1
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    return dist.get_rank(), dist.get_world_size()


def finalize_distributed_training() -> None:
    """Destroy the active distributed process group when one exists."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
