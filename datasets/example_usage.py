"""Example scripts for the in-tree benchmark datasets."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from grm.core import SegmentRecurrentMemoryModel
from grm.data import AddingProblemDataset, SequentialMNIST, TimeSeriesDataset
from grm.utils.config import MemoryArchitectureConfig


def take_last_step(outputs: torch.Tensor, batch_first: bool) -> torch.Tensor:
    return outputs[:, -1] if batch_first else outputs[-1]


def train_on_adding_problem():
    print("\n" + "=" * 60)
    print("Example 1: Adding Problem")
    print("=" * 60)

    train_loader, val_loader = AddingProblemDataset.build_dataloaders(
        batch_size=32,
        seq_len=100,
        train_samples=10000,
        val_samples=2000,
    )

    config = MemoryArchitectureConfig.load_preset_config("minimal")
    config.input_size = 2
    config.hidden_size = 128
    config.output_size = 1
    config.segment_length = 25
    config.memory_capacity_segments = 64
    config.retrieval_top_k = 8

    model = SegmentRecurrentMemoryModel(
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        output_size=config.output_size,
        rnn_type="linear",
        segment_length=config.segment_length,
        memory_capacity_segments=config.memory_capacity_segments,
        retrieval_top_k=config.retrieval_top_k,
        memory_storage_dtype=torch.float32,
        recomputation_ratio=0.0,
        retrieval_fusion_mode="residual",
        dropout=0.0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    total_loss = 0.0
    num_batches = 0
    for batch_index, (inputs, targets) in enumerate(train_loader):
        if batch_index >= 100:
            break

        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs, _, _, _ = model(inputs)
        predictions = take_last_step(outputs, model.batch_first)
        loss = criterion(predictions, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        if batch_index % 20 == 0:
            print(f"  Batch {batch_index:3d}: loss={loss.item():.6f}")

    print(f"Average loss: {total_loss / max(1, num_batches):.6f}")

    model.eval()
    validation_loss = 0.0
    with torch.no_grad():
        for inputs, targets in list(val_loader)[:20]:
            outputs, _, _, _ = model(inputs.to(device))
            predictions = take_last_step(outputs, model.batch_first)
            validation_loss += criterion(predictions, targets.to(device)).item()

    print(f"Validation loss: {validation_loss / 20:.6f}")
    return model


def train_on_sequential_mnist():
    print("\n" + "=" * 60)
    print("Example 2: Sequential MNIST")
    print("=" * 60)

    train_loader, _ = SequentialMNIST.build_dataloaders(
        batch_size=64,
        root="./data",
        pixel_level=False,
    )

    model = SegmentRecurrentMemoryModel(
        input_size=28,
        hidden_size=128,
        output_size=10,
        rnn_type="linear",
        segment_length=7,
        memory_capacity_segments=32,
        retrieval_top_k=4,
        memory_storage_dtype=torch.float32,
        recomputation_ratio=0.0,
        retrieval_fusion_mode="residual",
        dropout=0.1,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for batch_index, (inputs, targets) in enumerate(train_loader):
        if batch_index >= 50:
            break

        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs, _, _, _ = model(inputs)
        logits = take_last_step(outputs, model.batch_first)
        loss = criterion(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_examples += targets.size(0)
        if batch_index % 10 == 0:
            accuracy = 100.0 * total_correct / max(1, total_examples)
            print(f"  Batch {batch_index:3d}: loss={loss.item():.6f}, acc={accuracy:.2f}%")

    print(f"Average loss: {total_loss / 50:.6f}")
    return model


def train_on_timeseries():
    print("\n" + "=" * 60)
    print("Example 3: Time Series Forecasting")
    print("=" * 60)

    candidate_paths = [
        Path("./data/synthetic/synthetic_timeseries.csv"),
        Path("./data/timeseries/synthetic_timeseries.csv"),
        Path("./data/adding_problem/synthetic_timeseries.csv"),
    ]
    data_path = next((path for path in candidate_paths if path.exists()), candidate_paths[0])

    if data_path.exists():
        import pandas as pd

        values = pd.read_csv(data_path).values
    else:
        length = 2000
        num_features = 7
        time_index = np.linspace(0, 100, length)
        values = np.zeros((length, num_features))
        for feature_index in range(num_features):
            frequency = 0.1 + feature_index * 0.05
            phase = feature_index * 0.5
            values[:, feature_index] = (
                np.sin(2 * np.pi * frequency * time_index + phase)
                + 0.1 * np.random.randn(length)
            )
        values += np.linspace(0, 1, length).reshape(-1, 1) * 0.3

    dataset = TimeSeriesDataset(data=values, seq_len=100, pred_len=20, target_col=0, stride=10)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, _ = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = SegmentRecurrentMemoryModel(
        input_size=7,
        hidden_size=64,
        output_size=20,
        rnn_type="linear",
        segment_length=25,
        memory_capacity_segments=32,
        retrieval_top_k=4,
        memory_storage_dtype=torch.float32,
        recomputation_ratio=0.0,
        retrieval_fusion_mode="residual",
        dropout=0.0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for batch_index, (inputs, targets) in enumerate(train_loader):
        if batch_index >= 50:
            break
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs, _, _, _ = model(inputs)
        predictions = take_last_step(outputs, model.batch_first)
        loss = criterion(predictions, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_index % 10 == 0:
            print(f"  Batch {batch_index:3d}: loss={loss.item():.6f}")
    return model


def main():
    for example in (train_on_adding_problem, train_on_sequential_mnist, train_on_timeseries):
        try:
            example()
        except Exception as exc:  # pragma: no cover - example harness
            print(f"[ERROR] {example.__name__} failed: {exc}")
            raise


if __name__ == "__main__":
    main()
