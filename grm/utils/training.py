"""Lightweight training helpers for standalone experiments."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn
from tqdm import tqdm


class SupervisedExperimentTrainer:
    """Minimal trainer for supervised experiments outside the main CLI."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        gradient_clip_norm: float = 1.0,
        scheduler: Optional[Callable] = None,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.gradient_clip_norm = gradient_clip_norm
        self.scheduler = scheduler
        self.global_step = 0

    def train_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        epoch: int = 0,
        collect_outputs: bool = False,
    ) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        cached_outputs = []
        cached_targets = []

        for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
            if isinstance(batch, (tuple, list)):
                inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
            elif isinstance(batch, dict):
                inputs = batch["input"].to(self.device)
                targets = batch["target"].to(self.device)
            else:
                inputs = batch.to(self.device)
                targets = None

            self.optimizer.zero_grad()
            outputs, _, _, _ = self._forward(inputs)

            loss = self._compute_loss(outputs, targets) if targets is not None else outputs
            loss_value = float(loss.item() if isinstance(loss, torch.Tensor) else loss)

            if isinstance(loss, torch.Tensor):
                loss.backward()
                if self.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                self.optimizer.step()

            total_loss += loss_value
            num_batches += 1
            self.global_step += 1

            if collect_outputs:
                cached_outputs.append(outputs.detach().cpu())
                if targets is not None:
                    cached_targets.append(targets.detach().cpu())

        if self.scheduler is not None:
            self.scheduler.step()

        metrics: Dict[str, float] = {
            "loss": total_loss / max(1, num_batches),
            "num_batches": float(num_batches),
            "global_step": float(self.global_step),
        }
        if hasattr(self.model, "get_memory_info"):
            metrics.update(self.model.get_memory_info())
        return metrics

    def _forward(self, inputs: torch.Tensor):
        if inputs.ndim == 2:
            inputs = inputs.unsqueeze(0)
        model_output = self.model(inputs)
        if isinstance(model_output, tuple) and len(model_output) == 4:
            return model_output
        return model_output, None, None, None

    def _compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if outputs.ndim == 3 and not getattr(self.model, "batch_first", False):
            outputs = outputs.transpose(0, 1)
        if targets.ndim == 3 and not getattr(self.model, "batch_first", False):
            targets = targets.transpose(0, 1)
        if outputs.ndim == 3:
            outputs = outputs.reshape(-1, outputs.size(-1))
            targets = targets.reshape(-1, targets.size(-1))
        return self.criterion(outputs, targets)

    @torch.no_grad()
    def evaluate(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(dataloader, desc="Evaluating"):
            if isinstance(batch, (tuple, list)):
                inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
            else:
                inputs = batch.to(self.device)
                targets = None

            outputs, _, _, _ = self._forward(inputs)
            if targets is None:
                continue
            total_loss += float(self._compute_loss(outputs, targets).item())
            num_batches += 1

        return {
            "loss": total_loss / max(1, num_batches),
            "num_batches": float(num_batches),
        }

    def save_checkpoint(self, filepath: str, epoch: int, **extra_state: Any) -> None:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
        }
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        checkpoint.update(extra_state)
        torch.save(checkpoint, filepath)

    def load_checkpoint(
        self,
        filepath: str,
        load_optimizer: bool = True,
        load_scheduler: bool = True,
    ) -> Dict[str, Any]:
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if load_scheduler and self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.global_step = checkpoint.get("global_step", 0)
        return checkpoint


def create_trainer(
    model: nn.Module,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    gradient_clip_norm: float = 1.0,
    device: Optional[torch.device] = None,
    **kwargs: Any,
) -> SupervisedExperimentTrainer:
    runtime_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    criterion = nn.MSELoss()
    return SupervisedExperimentTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=runtime_device,
        gradient_clip_norm=gradient_clip_norm,
        **kwargs,
    )


__all__ = ["SupervisedExperimentTrainer", "create_trainer"]
