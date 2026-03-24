"""Evaluate a saved checkpoint on one of the in-tree benchmarks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from grm.core import SegmentRecurrentMemoryModel
from grm.data import AddingProblemDataset, CopyingMemoryDataset, SequentialMNIST


def take_last_step(outputs: torch.Tensor, batch_first: bool) -> torch.Tensor:
    return outputs[:, -1] if batch_first else outputs[-1]


def take_output_window(outputs: torch.Tensor, num_steps: int, batch_first: bool) -> torch.Tensor:
    if batch_first:
        return outputs[:, -num_steps:, :]
    return outputs[-num_steps:, :, :].transpose(0, 1)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a segment-memory checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["adding_problem", "copying_memory", "sequential_mnist"],
        help="Benchmark to evaluate",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--device", type=str, default="auto", help="Runtime device: auto, cuda, or cpu")
    parser.add_argument("--seq_len", type=int, default=None, help="Optional evaluation sequence length override")
    return parser.parse_args()


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})

    input_size = config.get("input_size", 2)
    hidden_size = config.get("hidden_size", 128)
    output_size = config.get("output_size", 1)
    rnn_type = config.get("rnn_type", "linear")
    memory_key_dim = config.get("memory_key_dim", config.get("key_size"))
    segment_length = config.get("segment_length", config.get("segment_size", 25))
    memory_capacity_segments = config.get("memory_capacity_segments", config.get("max_cached_segments", 64))
    retrieval_top_k = config.get("retrieval_top_k", config.get("top_k", 8))
    retrieval_fusion_mode = config.get("retrieval_fusion_mode", config.get("fusion_mode", "residual"))

    model = SegmentRecurrentMemoryModel(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        rnn_type=rnn_type,
        memory_key_dim=memory_key_dim,
        segment_length=segment_length,
        memory_capacity_segments=memory_capacity_segments,
        retrieval_top_k=retrieval_top_k,
        memory_storage_dtype=torch.float32,
        recomputation_ratio=0.0,
        segment_pooling_mode="mean",
        retrieval_fusion_mode=retrieval_fusion_mode,
        dropout=0.0,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def evaluate_adding_problem(model, device, batch_size, seq_len=None):
    seq_len = seq_len or 100
    _, dataloader = AddingProblemDataset.build_dataloaders(
        batch_size=batch_size,
        seq_len=seq_len,
        train_samples=batch_size,
        val_samples=5000,
    )
    criterion = nn.MSELoss()
    total_loss = 0.0
    total_mae = 0.0
    predictions = []
    targets = []

    with torch.no_grad():
        for inputs, target_values in dataloader:
            inputs = inputs.to(device)
            target_values = target_values.to(device)
            if not getattr(model, "batch_first", False):
                inputs = inputs.transpose(0, 1)
            outputs, _, _, _ = model(inputs, return_hidden_states=True)
            predicted_values = take_last_step(outputs, getattr(model, "batch_first", False))
            loss = criterion(predicted_values, target_values)

            total_loss += loss.item()
            total_mae += torch.abs(predicted_values - target_values).mean().item()
            predictions.extend(predicted_values.cpu().tolist())
            targets.extend(target_values.cpu().tolist())

    predictions_tensor = torch.tensor(predictions)
    targets_tensor = torch.tensor(targets)
    rmse = torch.sqrt(((predictions_tensor - targets_tensor) ** 2).mean()).item()
    return {
        "mse": total_loss / max(1, len(dataloader)),
        "mae": total_mae / max(1, len(dataloader)),
        "rmse": rmse,
    }


def evaluate_copying_memory(model, device, batch_size, seq_len=None):
    seq_len = seq_len or 100
    dataset = CopyingMemoryDataset(num_samples=5000, seq_len=seq_len, num_copy=10)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs, _, _, _ = model(inputs, return_hidden_states=True)

            predicted_window = take_output_window(outputs, targets.size(1), getattr(model, "batch_first", False))
            loss = criterion(predicted_window.reshape(-1, predicted_window.size(-1)), targets.reshape(-1))
            total_loss += loss.item()

            recall_length = dataset.num_copy
            predicted_tokens = predicted_window.argmax(dim=-1)[:, -recall_length:]
            target_tokens = targets[:, -recall_length:]
            correct += (predicted_tokens == target_tokens).sum().item()
            total += target_tokens.numel()

    return {
        "loss": total_loss / max(1, len(dataloader)),
        "accuracy": correct / max(1, total),
    }


def evaluate_sequential_mnist(model, device, batch_size):
    _, dataloader = SequentialMNIST.build_dataloaders(batch_size=batch_size, root="./data", pixel_level=False)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs, _, _, _ = model(inputs, return_hidden_states=True)
            logits = take_last_step(outputs, getattr(model, "batch_first", False))
            total_loss += criterion(logits, targets).item()
            predictions = logits.argmax(dim=1)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)

    return {
        "loss": total_loss / max(1, len(dataloader)),
        "accuracy": correct / max(1, total),
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        alternative = Path("checkpoints") / args.checkpoint
        if alternative.exists():
            checkpoint_path = alternative
        else:
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    model, checkpoint = load_model(checkpoint_path, device)
    if args.dataset == "adding_problem":
        results = evaluate_adding_problem(model, device, args.batch_size, args.seq_len)
    elif args.dataset == "copying_memory":
        results = evaluate_copying_memory(model, device, args.batch_size, args.seq_len)
    else:
        results = evaluate_sequential_mnist(model, device, args.batch_size)

    print(f"Dataset: {args.dataset}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Epoch: {checkpoint.get('epoch', 'unknown')}")
    for key, value in results.items():
        print(f"{key}: {value:.6f}" if isinstance(value, float) else f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
