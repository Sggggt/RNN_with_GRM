"""Sanity checks for the benchmark datasets shipped with the repository."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from grm.data import (
    AddingProblemDataset,
    CopyingMemoryDataset,
    SequentialMNIST,
    TimeSeriesDataset,
    build_benchmark_dataloader,
)


def test_adding_problem() -> bool:
    dataset = AddingProblemDataset(num_samples=100, seq_len=50, seed=42)
    inputs, targets = dataset[0]
    print("Adding Problem:", inputs.shape, targets.shape)

    train_loader, _ = AddingProblemDataset.build_dataloaders(
        batch_size=16,
        seq_len=50,
        train_samples=200,
        val_samples=50,
    )
    batch_inputs, batch_targets = next(iter(train_loader))
    print("Adding batch:", batch_inputs.shape, batch_targets.shape)
    return True


def test_copying_memory() -> bool:
    dataset = CopyingMemoryDataset(num_samples=100, seq_len=50, num_copy=5, seed=42)
    inputs, targets = dataset[0]
    print("Copying memory:", inputs.shape, targets.shape)

    loader = torch.utils.data.DataLoader(dataset, batch_size=16)
    batch_inputs, batch_targets = next(iter(loader))
    print("Copying batch:", batch_inputs.shape, batch_targets.shape)
    return True


def test_sequential_mnist() -> bool:
    dataset = SequentialMNIST(root="./data", train=True, pixel_level=False, download=True)
    inputs, target = dataset[0]
    print("Sequential MNIST:", inputs.shape, int(target))

    train_loader, _ = SequentialMNIST.build_dataloaders(batch_size=32, pixel_level=False)
    batch_inputs, batch_targets = next(iter(train_loader))
    print("Sequential batch:", batch_inputs.shape, batch_targets[:5])
    return True


def test_time_series() -> bool:
    import numpy as np

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
    values += np.linspace(0, 1, length).reshape(-1, 1) * 0.5

    dataset = TimeSeriesDataset(data=values, seq_len=100, pred_len=20, target_col=0, stride=10)
    inputs, targets = dataset[0]
    print("Time series:", inputs.shape, targets.shape)

    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    batch_inputs, batch_targets = next(iter(loader))
    print("Time-series batch:", batch_inputs.shape, batch_targets.shape)
    return True


def test_factory() -> bool:
    loader = build_benchmark_dataloader(
        "adding_problem",
        batch_size=16,
        seq_len=50,
        train_samples=100,
    )
    batch_inputs, batch_targets = next(iter(loader))
    print("Factory loader:", batch_inputs.shape, batch_targets.shape)
    return True


def main() -> int:
    tests = [
        ("Adding Problem", test_adding_problem),
        ("Copying Memory", test_copying_memory),
        ("Sequential MNIST", test_sequential_mnist),
        ("Time Series", test_time_series),
        ("Factory", test_factory),
    ]

    results = {}
    for name, test_fn in tests:
        try:
            results[name] = test_fn()
        except Exception as exc:  # pragma: no cover - manual test harness
            print(f"[FAIL] {name}: {exc}")
            results[name] = False

    passed = sum(bool(value) for value in results.values())
    total = len(results)
    print(f"{passed}/{total} dataset tests passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
