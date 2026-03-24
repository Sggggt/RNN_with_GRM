"""Dataset definitions for synthetic and text benchmarks."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


@dataclass
class AddingProblemConfig:
    """Configuration for the synthetic addition benchmark."""

    seq_len: int = 100
    num_samples: int = 10000
    train: bool = True
    seed: Optional[int] = None


class AddingProblemDataset(Dataset):
    """Synthetic long-range addition benchmark."""

    def __init__(self, num_samples: int = 10000, seq_len: int = 100, seed: Optional[int] = None) -> None:
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.seed = seed
        self.data, self.targets = self._generate_data()

    def _build_generator(self) -> Optional[torch.Generator]:
        if self.seed is None:
            return None
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        return generator

    def _generate_data(self) -> Tuple[Tensor, Tensor]:
        generator = self._build_generator()
        data = torch.randn(self.num_samples, self.seq_len, 2, generator=generator)
        targets = torch.zeros(self.num_samples, 1)

        for index in range(self.num_samples):
            first_position = torch.randint(0, self.seq_len // 2, (1,), generator=generator).item()
            second_position = torch.randint(self.seq_len // 2, self.seq_len, (1,), generator=generator).item()
            first_value = torch.randn(1, generator=generator).item()
            second_value = torch.randn(1, generator=generator).item()

            data[index, first_position, 0] = first_value + 5.0
            data[index, second_position, 1] = second_value + 5.0
            targets[index, 0] = first_value + second_value

        return data, targets

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        return self.data[index], self.targets[index]

    @classmethod
    def build_dataloaders(
        cls,
        batch_size: int = 32,
        seq_len: int = 100,
        train_samples: int = 50000,
        val_samples: int = 10000,
    ) -> Tuple[DataLoader, DataLoader]:
        train_dataset = cls(num_samples=train_samples, seq_len=seq_len, seed=42)
        val_dataset = cls(num_samples=val_samples, seq_len=seq_len, seed=43)
        return (
            DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
            DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        )


class CopyingMemoryDataset(Dataset):
    """Discrete delayed-recall benchmark with blank and delimiter tokens."""

    def __init__(
        self,
        num_samples: int = 10000,
        seq_len: int = 100,
        num_copy: int = 10,
        num_symbols: int = 8,
        seed: Optional[int] = None,
    ) -> None:
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.num_copy = num_copy
        self.num_symbols = num_symbols
        self.blank_token_id = 0
        self.delimiter_token_id = num_symbols + 1
        self.vocab_size = num_symbols + 2
        self.delay_len = seq_len - (2 * num_copy) - 1
        self.seed = seed

        if self.delay_len < 1:
            raise ValueError(
                f"CopyingMemoryDataset requires seq_len >= 2 * num_copy + 2, "
                f"got seq_len={seq_len}, num_copy={num_copy}."
            )

        self.data, self.targets = self._generate_data()

    def _build_generator(self) -> Optional[torch.Generator]:
        if self.seed is None:
            return None
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        return generator

    def _generate_data(self) -> Tuple[Tensor, Tensor]:
        generator = self._build_generator()
        data = torch.zeros(self.num_samples, self.seq_len, self.vocab_size, dtype=torch.float32)
        data[:, :, self.blank_token_id] = 1.0
        targets = torch.full((self.num_samples, self.seq_len), self.blank_token_id, dtype=torch.long)

        memory_tokens = torch.randint(
            low=1,
            high=self.num_symbols + 1,
            size=(self.num_samples, self.num_copy),
            generator=generator,
        )

        sample_indices = torch.arange(self.num_samples).unsqueeze(1)
        copy_positions = torch.arange(self.num_copy).unsqueeze(0)
        delimiter_position = self.num_copy + self.delay_len
        output_start = delimiter_position + 1

        data[sample_indices, copy_positions, self.blank_token_id] = 0.0
        data[sample_indices, copy_positions, memory_tokens] = 1.0
        data[:, delimiter_position, self.blank_token_id] = 0.0
        data[:, delimiter_position, self.delimiter_token_id] = 1.0
        targets[:, output_start:] = memory_tokens
        return data, targets

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        return self.data[index], self.targets[index]

    @property
    def output_size(self) -> int:
        return self.vocab_size


class SequentialMNIST(Dataset):
    """Sequential MNIST benchmark in row-wise or pixel-wise form."""

    def __init__(
        self,
        root: str = "./data",
        train: bool = True,
        pixel_level: bool = False,
        download: bool = True,
        normalize: bool = True,
    ) -> None:
        from torchvision import datasets

        self.pixel_level = pixel_level
        self.normalize = normalize
        self.mnist = datasets.MNIST(root=root, train=train, download=download)
        self._process_data()

    def _process_data(self) -> None:
        data = self.mnist.data.float()
        if self.normalize:
            data = data / 255.0

        if self.pixel_level:
            self.sequences = data.view(-1, 784, 1)
            self.seq_len = 784
            self.input_size = 1
        else:
            self.sequences = data
            self.seq_len = 28
            self.input_size = 28
        self.labels = self.mnist.targets

    def __len__(self) -> int:
        return len(self.mnist)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        return self.sequences[index], self.labels[index]

    def get_output_shape(self) -> Tuple[int, int]:
        return self.seq_len, self.input_size

    @classmethod
    def build_dataloaders(
        cls,
        batch_size: int = 64,
        root: str = "./data",
        pixel_level: bool = False,
    ) -> Tuple[DataLoader, DataLoader]:
        train_dataset = cls(root=root, train=True, pixel_level=pixel_level)
        test_dataset = cls(root=root, train=False, pixel_level=pixel_level)
        return (
            DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
            DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
        )


class WikiTextDataset(Dataset):
    """WikiText language-modeling dataset with reusable vocabulary."""

    def __init__(
        self,
        data_path: str,
        seq_len: int = 2048,
        split: str = "train",
        stride: Optional[int] = None,
        vocab: Optional[Dict[str, int]] = None,
        vocab_path: Optional[str] = None,
        max_vocab_size: Optional[int] = None,
    ) -> None:
        self.seq_len = seq_len
        self.split = split
        self.max_vocab_size = max_vocab_size
        self.vocab_path = vocab_path
        self.stride = stride if stride is not None else (max(1, seq_len // 2) if split == "train" else max(1, seq_len))

        self.text = Path(data_path).read_text(encoding="utf-8")
        self.tokens = self.text.split()

        if vocab is not None:
            self.vocab = vocab
        elif vocab_path is not None and Path(vocab_path).exists():
            self.vocab = self._load_vocab(vocab_path)
        else:
            self.vocab = self._build_vocab()

        self.encoded = self._encode_text()
        self.num_sequences = max(0, (len(self.encoded) - (seq_len + 1)) // self.stride + 1)

    def _load_vocab(self, vocab_path: str) -> Dict[str, int]:
        vocab: Dict[str, int] = {}
        for raw_line in Path(vocab_path).read_text(encoding="utf-8").splitlines():
            if not raw_line:
                continue
            token, index = raw_line.rsplit(" ", 1)
            vocab[token] = int(index)
        return vocab

    def _build_vocab(self) -> Dict[str, int]:
        word_counts = Counter(self.tokens)
        special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]
        for token in special_tokens:
            word_counts.pop(token, None)

        vocab = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
        if self.max_vocab_size is not None and self.max_vocab_size > len(vocab):
            budget = max(0, self.max_vocab_size - len(vocab))
            ordered_words = [
                token
                for token, _ in sorted(word_counts.items(), key=lambda item: (-item[1], item[0]))[:budget]
            ]
        else:
            ordered_words = sorted(word_counts)

        for index, token in enumerate(ordered_words, start=len(vocab)):
            vocab[token] = index
        return vocab

    def _encode_text(self) -> List[int]:
        unk_id = self.vocab.get("<unk>", 1)
        encoded: List[int] = []
        for token in self.tokens:
            encoded.append(self.vocab.get(token, self.vocab.get(token.lower(), unk_id)))
        return encoded

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        actual_index = index * self.stride
        inputs = self.encoded[actual_index : actual_index + self.seq_len]
        targets = self.encoded[actual_index + 1 : actual_index + self.seq_len + 1]
        if len(inputs) != self.seq_len or len(targets) != self.seq_len:
            raise IndexError(
                f"WikiText window {index} is incomplete: "
                f"len(inputs)={len(inputs)}, len(targets)={len(targets)}, seq_len={self.seq_len}."
            )
        return torch.tensor(inputs), torch.tensor(targets)


class TimeSeriesDataset(Dataset):
    """Sliding-window dataset for multivariate forecasting."""

    def __init__(
        self,
        data: np.ndarray,
        seq_len: int = 720,
        pred_len: int = 96,
        target_col: int = 0,
        stride: int = 1,
    ) -> None:
        self.data = torch.from_numpy(data).float()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.target_col = target_col
        self.stride = stride
        self.num_samples = (len(data) - seq_len - pred_len) // stride + 1

    def __len__(self) -> int:
        return max(0, self.num_samples)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        start = index * self.stride
        end = start + self.seq_len
        inputs = self.data[start:end]
        targets = self.data[end : end + self.pred_len, self.target_col]
        return inputs, targets

    @classmethod
    def from_csv(
        cls,
        csv_path: str,
        seq_len: int = 720,
        pred_len: int = 96,
        target_col: int = 0,
        **kwargs,
    ) -> "TimeSeriesDataset":
        import pandas as pd

        frame = pd.read_csv(csv_path)
        numeric_columns = frame.select_dtypes(include=[np.number]).columns
        return cls(frame[numeric_columns].values, seq_len, pred_len, target_col, **kwargs)


def build_benchmark_dataloader(dataset_name: str, batch_size: int = 32, **kwargs) -> DataLoader:
    if dataset_name == "adding_problem":
        return AddingProblemDataset.build_dataloaders(batch_size, **kwargs)[0]
    if dataset_name == "copying_memory":
        dataset = CopyingMemoryDataset(**kwargs)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    if dataset_name == "sequential_mnist":
        return SequentialMNIST.build_dataloaders(batch_size, **kwargs)[0]
    if dataset_name == "wikitext":
        dataset = WikiTextDataset(**kwargs)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    if dataset_name == "timeseries":
        dataset = TimeSeriesDataset.from_csv(**kwargs)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    raise ValueError(f"Unknown dataset: {dataset_name}")


__all__ = [
    "AddingProblemConfig",
    "AddingProblemDataset",
    "CopyingMemoryDataset",
    "SequentialMNIST",
    "TimeSeriesDataset",
    "WikiTextDataset",
    "build_benchmark_dataloader",
]
