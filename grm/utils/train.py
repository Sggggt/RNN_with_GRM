"""Training entry point for the active segment-memory research stack."""

import os
import sys
import json
import time
import math
import random
import shutil
import argparse
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, Tuple, List, Union

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch import Tensor
    from torch.utils.data import DataLoader
    from torch.amp import GradScaler, autocast
except ModuleNotFoundError as exc:
    if exc.name == "torch":
        raise SystemExit(
            "PyTorch is not installed in the current WSL python3 interpreter. "
            "Activate the project virtual environment first with "
            "`source .venv/bin/activate`, then rerun `python3 grm/utils/train.py ...`."
        ) from exc
    raise

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is expected in the local env
    tqdm = None

import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from grm.core import (
    SegmentRecurrentMemoryModel,
    HierarchicalSegmentMemoryModel,
    get_grm_cuda_runtime_status,
)
from grm.data import (
    AddingProblemDataset,
    CopyingMemoryDataset,
    SequentialMNIST,
    TimeSeriesDataset,
    WikiTextDataset
)
from grm.evaluation.paper_tasks import run_proxy_benchmark_suite
from grm.utils.config import MemoryArchitectureConfig



def setup_logger(log_dir: Path, stage_label: str) -> logging.Logger:
    """Create a stage-specific logger with file and console handlers."""
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{stage_label}_{timestamp}.log"

    logger = logging.getLogger("SegmentMemoryTraining")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    """Set process-wide RNG state before building data loaders or models."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Align DataLoader workers with the parent process seed."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _is_wsl_runtime() -> bool:
    if os.name != "posix":
        return False
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        return True
    version_file = Path("/proc/version")
    if not version_file.exists():
        return False
    version_text = version_file.read_text(encoding="utf-8", errors="ignore").lower()
    return "microsoft" in version_text or "wsl" in version_text


def _prepare_wsl_compiler_env() -> Dict[str, Any]:
    status: Dict[str, Any] = {
        "is_wsl": _is_wsl_runtime(),
        "path_prepended": [],
        "cc_env": os.environ.get("CC", ""),
        "cxx_env": os.environ.get("CXX", ""),
        "compiler_path_env": os.environ.get("COMPILER_PATH", ""),
        "nvcc": shutil.which("nvcc"),
        "cc": shutil.which("cc") or shutil.which("gcc"),
        "as": shutil.which("as"),
    }
    if not status["is_wsl"]:
        return status

    status.update(_get_wsl_storage_status(project_root))

    current_path = os.environ.get("PATH", "")
    path_entries = [entry for entry in current_path.split(os.pathsep) if entry]
    prepend_entries: List[str] = []
    for candidate in ("/usr/local/cuda/bin", "/usr/bin", "/bin"):
        if Path(candidate).exists() and candidate not in path_entries:
            prepend_entries.append(candidate)
    if prepend_entries:
        os.environ["PATH"] = os.pathsep.join(prepend_entries + path_entries)
        status["path_prepended"] = prepend_entries

    if not os.environ.get("CC") and Path("/usr/bin/gcc").exists():
        os.environ["CC"] = "/usr/bin/gcc"
    if not os.environ.get("CXX") and Path("/usr/bin/g++").exists():
        os.environ["CXX"] = "/usr/bin/g++"
    if not os.environ.get("COMPILER_PATH") and Path("/usr/bin").exists():
        os.environ["COMPILER_PATH"] = "/usr/bin"

    status["cc_env"] = os.environ.get("CC", "")
    status["cxx_env"] = os.environ.get("CXX", "")
    status["compiler_path_env"] = os.environ.get("COMPILER_PATH", "")
    status["nvcc"] = shutil.which("nvcc")
    status["cc"] = shutil.which("cc") or shutil.which("gcc")
    status["as"] = shutil.which("as")
    return status


def _get_wsl_storage_status(path: Path) -> Dict[str, Any]:
    resolved = path.resolve()
    parts = resolved.parts
    is_windows_mount = (
        len(parts) >= 3
        and parts[0] == "/"
        and parts[1] == "mnt"
        and len(parts[2]) == 1
        and parts[2].isalpha()
    )

    suggested_root = f"~/workspace/{resolved.name}"
    return {
        "project_on_windows_mount": is_windows_mount,
        "mount_drive": parts[2].lower() if is_windows_mount else "",
        "project_location_label": "windows-mounted filesystem" if is_windows_mount else "linux filesystem",
        "suggested_linux_root_label": suggested_root,
    }



class SegmentMemoryLanguageModel(nn.Module):
    """Language-model wrapper around the segment-memory recurrent backbone."""
    def __init__(
        self,
        grm_rnn: nn.Module,
        vocab_size: int,
        embed_dim: int,
        token_to_id: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        self.grm_rnn = grm_rnn
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.output_layer = nn.Linear(grm_rnn.hidden_size, vocab_size)
        self.batch_first = grm_rnn.batch_first
        self.hidden_size = grm_rnn.hidden_size
        self.token_to_id = dict(token_to_id or {})
        self.pad_token_id = self.token_to_id.get("<pad>", 0)
        self.unk_token_id = self.token_to_id.get("<unk>", 1)
        self.bos_token_id = self.token_to_id.get("<bos>", self.unk_token_id)
        self.eos_token_id = self.token_to_id.get("<eos>", self.unk_token_id)

    def forward(self, x, return_hidden_states=False, return_all_outputs=True, **kwargs):
        x_embed = self.embedding(x)

        outputs, h_final, c_final, aux = self.grm_rnn(
            x_embed,
            return_hidden_states=return_hidden_states,
            return_all_outputs=return_all_outputs,
            **kwargs
        )

        logits = self.output_layer(outputs)

        return logits, h_final, c_final, aux

    @property
    def memory_bank(self):
        return self.grm_rnn.memory_bank



@dataclass
class TrainingExperimentConfig:
    """Experiment configuration for training, evaluation, and checkpointing."""

    dataset: str
    experiment_name: str
    batch_size: int = 32
    num_epochs: int = 10
    learning_rate: float = 4e-4
    weight_decay: float = 0.1
    warmup_steps: int = 1000
    min_lr: float = 1e-6

    input_size: int = 2
    hidden_size: int = 128
    output_size: int = 1
    num_layers: int = 1
    rnn_type: str = "linear"
    memory_key_dim: Optional[int] = None
    memory_initialization_mode: str = "checkpoint"
    retrieval_query_source: str = "input"
    segment_summary_source: str = "input_mean"
    segment_length: int = 64
    memory_capacity_segments: int = 64
    retrieval_top_k: int = 8
    synthetic_sequence_length: Optional[int] = None
    sequence_length: int = 2048

    memory_storage_dtype: torch.dtype = torch.float32
    recomputation_ratio: float = 0.02
    enable_activation_checkpointing: bool = False
    cuda_cpp_debug_fallback: bool = False
    auto_memory_guard: bool = True
    cuda_memory_fraction: float = 0.92
    gradient_accumulation_factor: int = 1
    gradient_clip_norm: float = 1.0
    retrieval_fusion_mode: str = "residual"
    dropout: float = 0.0
    batch_first: bool = True
    sequence_output: bool = False
    vocab_size: Optional[int] = None
    effective_batch_size: Optional[int] = None
    seed: int = 42
    deterministic: bool = False

    use_mixed_precision: bool = False
    fp16_storage: bool = False
    preset_recomputation_ratio: float = 0.02

    precision_stage: str = "stage1"
    enable_precision_curriculum: bool = True
    stage1_min_epochs: int = 3
    stage1_stability_window: int = 2
    stage1_relative_improvement_threshold: float = 0.01

    enable_logarithmic_segmentation: bool = False
    maximum_hierarchy_level: int = 10
    segments_per_hierarchy_level: int = 256
    base_segment_length: int = 1
    enable_adaptive_hierarchy_level: bool = True

    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    save_every: int = 1
    log_every: int = 10
    max_train_batches: Optional[int] = None
    max_val_batches: Optional[int] = None

    device: str = "auto"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the configuration to a JSON-friendly dictionary."""
        d = asdict(self)
        d["memory_storage_dtype"] = str(d["memory_storage_dtype"])
        return d



class ExperimentTrainer:
    """Trainer for benchmark-aligned segment-memory experiments."""

    def __init__(
        self,
        config: TrainingExperimentConfig,
        logger: Optional[logging.Logger] = None
    ):
        self.config = config
        set_global_seed(config.seed, deterministic=config.deterministic)
        self.runtime_env_status = _prepare_wsl_compiler_env()

        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        if self.config.effective_batch_size is None:
            self.config.effective_batch_size = self.config.batch_size

        if logger is None:
            self.logger = logging.getLogger("SegmentMemoryTrainer")
            self.logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(handler)
        else:
            self.logger = logger

        self.cuda_runtime_status = (
            get_grm_cuda_runtime_status()
            if self.device.type == "cuda"
            else {}
        )
        self._apply_wikitext_cuda_memory_guard()

        if self.runtime_env_status.get("is_wsl") and self.runtime_env_status.get("project_on_windows_mount"):
            dataset_hint = (
                "especially for wikitext"
                if config.dataset == "wikitext"
                else "especially for long-running datasets such as wikitext"
            )
            self.logger.warning(
                "The active WSL checkout is on a Windows-mounted filesystem. "
                f"This can become an I/O bottleneck {dataset_hint}. "
                "Prefer moving the active training checkout to a Linux filesystem path such as "
                f"{self.runtime_env_status.get('suggested_linux_root_label')}."
            )

        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        self.current_precision_stage = self.config.precision_stage
        self.precision_stage_transitioned = self.current_precision_stage == "stage2"
        self.validation_loss_history: List[float] = []

        self.scaler = GradScaler() if (config.use_mixed_precision and self.device.type == "cuda") else None

        self.metrics_history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': [],
            'recompute_rate': [],
            'grad_norm': [],
            'learning_rate': []
        }
        self.token_vocabulary: Optional[Dict[str, int]] = None
        self.copy_target_length: Optional[int] = None

    def _estimate_wikitext_safe_micro_batch(self) -> int:
        """Return a conservative micro-batch cap for CUDA WikiText language modeling."""
        sequence_length = max(1, int(self.config.sequence_length))
        if sequence_length >= 8192:
            base_cap = 1
        elif sequence_length >= 4096:
            base_cap = 2
        elif sequence_length >= 2048:
            base_cap = 4
        else:
            base_cap = 8

        hidden_penalty = max(1.0, self.config.hidden_size / 384.0)
        retrieval_penalty = max(1.0, self.config.retrieval_top_k / 4.0)
        layer_penalty = max(1.0, float(self.config.num_layers))
        penalty = hidden_penalty * retrieval_penalty * math.sqrt(layer_penalty)
        adjusted_cap = max(1, int(base_cap / penalty))
        return adjusted_cap

    def _apply_wikitext_cuda_memory_guard(self) -> None:
        """Downshift CUDA WikiText runs before DataLoader construction to avoid OOM."""
        if self.device.type != "cuda":
            return
        if self.config.dataset != "wikitext":
            return

        runtime_status = self.cuda_runtime_status or {}
        kernel_policy = runtime_status.get("kernel_policy", {})
        forced_fallback = (
            self.config.cuda_cpp_debug_fallback
            or kernel_policy.get("global") == "fallback"
            or kernel_policy.get("batched_memory_gather") == "fallback"
        )
        using_native_backend = runtime_status.get("cuda_backend_available", False) and not forced_fallback
        runtime_label = "native CUDA kernels" if using_native_backend else "PyTorch fallback kernels"
        load_error = runtime_status.get("load_error") or "native extension unavailable"
        safe_batch_cap = self._estimate_wikitext_safe_micro_batch()
        original_batch_size = max(1, int(self.config.batch_size))
        original_accumulation = max(1, int(self.config.gradient_accumulation_factor))

        if not self.config.enable_activation_checkpointing:
            self.config.enable_activation_checkpointing = True
            self.logger.warning(
                "Enabled activation checkpointing to stabilize CUDA memory usage "
                f"for this WikiText run ({runtime_label}; detail: {load_error})."
            )

        if original_batch_size <= safe_batch_cap:
            return

        preserved_samples_per_update = original_batch_size * original_accumulation
        new_batch_size = safe_batch_cap
        new_accumulation = max(
            original_accumulation,
            math.ceil(preserved_samples_per_update / new_batch_size),
        )

        self.config.batch_size = new_batch_size
        self.config.gradient_accumulation_factor = new_accumulation
        self.config.effective_batch_size = new_batch_size

        self.logger.warning(
            "Reduced the WikiText micro-batch to fit the active CUDA memory budget: "
            f"batch_size {original_batch_size} -> {new_batch_size}, "
            f"gradient_accumulation_factor {original_accumulation} -> {new_accumulation} "
            f"({runtime_label})."
        )

    def _create_progress_bar(self, total: int, desc: str):
        if tqdm is None or total <= 0:
            return None
        return tqdm(total=total, desc=desc, dynamic_ncols=True, leave=False)

    def _make_dataloader_generator(self, offset: int) -> torch.Generator:
        generator = torch.Generator()
        generator.manual_seed(self.config.seed + offset)
        return generator

    def _legacy_setup_data(self) -> Tuple[DataLoader, DataLoader]:
        """Legacy data-loader construction retained for compatibility experiments."""
        self.logger.info("=" * 60)
        self.logger.info("Setting up Data Loaders")
        self.logger.info("=" * 60)
        self.copy_target_length = None

        if self.config.dataset == "adding_problem":
            train_loader, val_loader = AddingProblemDataset.build_dataloaders(
                batch_size=self.config.batch_size,
                seq_len=self.config.segment_length * 4,
                train_samples=50000,
                val_samples=5000
            )
            self.logger.info(f"Dataset: Adding Problem")
            self.logger.info(f"  Train batches: {len(train_loader)}")
            self.logger.info(f"  Val batches: {len(val_loader)}")

        elif self.config.dataset == "copying_memory":
            copy_seq_len = self.config.segment_length * 4
            train_dataset = CopyingMemoryDataset(
                num_samples=50000,
                seq_len=copy_seq_len,
                num_copy=10,
                num_symbols=8,
            )
            val_dataset = CopyingMemoryDataset(
                num_samples=5000,
                seq_len=copy_seq_len,
                num_copy=10,
                num_symbols=8,
            )
            train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)

            self.config.input_size = train_dataset.output_size
            self.config.output_size = train_dataset.output_size
            self.config.sequence_output = True
            self.copy_target_length = train_dataset.num_copy

            self.logger.info(f"Dataset: Copying Memory")
            self.logger.info(f"  Input size: {self.config.input_size}")
            self.logger.info(f"  Output size: {self.config.output_size}")
            self.logger.info(f"  Sequence length: {copy_seq_len}")
            self.logger.info(f"  Symbols to copy: {train_dataset.num_copy}")
            self.logger.info(f"  Symbol vocabulary: {train_dataset.num_symbols} + blank + delimiter")

        elif self.config.dataset == "sequential_mnist":
            train_loader, val_loader = SequentialMNIST.build_dataloaders(
                batch_size=self.config.batch_size,
                pixel_level=True
            )
            sample_x, sample_y = next(iter(train_loader))
            self.logger.info(f"Dataset: Sequential MNIST")
            self.logger.info(f"  Input shape: {sample_x.shape}")
            self.logger.info(f"  Label shape: {sample_y.shape} (expected: [batch])")
            self.logger.info(f"  Sample labels: {sample_y[:5]}")

            self.config.input_size = sample_x.size(-1)
            self.config.output_size = 10

        elif self.config.dataset == "timeseries":
            possible_csv_paths = [
                Path(f"data/{self.config.experiment_name}/synthetic_timeseries.csv"),
                Path("data/timeseries/synthetic_timeseries.csv"),
                Path("data/adding_problem/synthetic_timeseries.csv"),
            ]

            csv_path = next((path for path in possible_csv_paths if path.exists()), None)
            if csv_path is None:
                raise FileNotFoundError(
                    f"Time series CSV not found. Tried paths: {[str(path) for path in possible_csv_paths]}"
                )

            dataset = TimeSeriesDataset.from_csv(
                csv_path=str(csv_path),
                seq_len=self.config.segment_length * 4,
                pred_len=20,
                target_col=0,
                stride=10,
            )
            num_windows = len(dataset)
            base_train_end = max(1, int(num_windows * 0.8))
            gap_windows = max(1, math.ceil((dataset.seq_len + dataset.pred_len) / max(1, dataset.stride)))
            val_start = min(num_windows - 1, base_train_end + gap_windows)
            if val_start <= 0 or val_start >= num_windows:
                val_start = max(1, int(num_windows * 0.8))
            train_indices = list(range(0, val_start - gap_windows))
            val_indices = list(range(val_start, num_windows))
            if not train_indices or not val_indices:
                raise ValueError(
                    f"Time series dataset is too small for a leakage-safe chronological split: {num_windows} windows."
                )

            train_dataset = torch.utils.data.Subset(dataset, train_indices)
            val_dataset = torch.utils.data.Subset(dataset, val_indices)
            train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)

            sample_x, sample_y = dataset[0]
            self.config.input_size = sample_x.size(-1)
            self.config.output_size = sample_y.size(-1)

            self.logger.info("Dataset: Time Series")
            self.logger.info(f"  CSV: {csv_path}")
            self.logger.info(f"  Input shape: {sample_x.shape}")
            self.logger.info(f"  Target shape: {sample_y.shape}")
            self.logger.info(
                f"  Chronological split: train_windows={len(train_indices)}, "
                f"val_windows={len(val_indices)}, gap_windows={gap_windows}"
            )
            self.logger.info(f"  Train batches: {len(train_loader)}")
            self.logger.info(f"  Val batches: {len(val_loader)}")

        elif self.config.dataset == "wikitext":
            possible_train_paths = [
                Path("data/wikitext/wiki103.train.tokens"),
                Path("data/wikitext/wikitext-103/wiki.train.tokens"),
                Path("data/wikitext/wiki2.train.tokens"),
                Path("data/wikitext/wikitext-2/wiki.train.tokens"),
                Path("data/wikitext/mini_wiki.train.tokens")
            ]
            
            train_path = None
            for p in possible_train_paths:
                if p.exists():
                    train_path = p
                    break
            
            if train_path is None:
                raise FileNotFoundError(
                    f"WikiText training data not found. Tried paths: {[str(p) for p in possible_train_paths]}"
                )
            
            val_path = train_path.parent / train_path.name.replace(".train.", ".valid.")
            if not val_path.exists():
                val_path = train_path.parent / "wiki.valid.tokens"
            vocab_path = train_path.parent / train_path.name.replace(".train.tokens", ".vocab")
            
            self.logger.info(f"Loading WikiText:")
            self.logger.info(f"  Train: {train_path}")
            if not val_path.exists():
                raise FileNotFoundError(f"WikiText validation data not found: {val_path}")
            self.logger.info(f"  Valid: {val_path}")
            if vocab_path.exists():
                self.logger.info(f"  Vocab: {vocab_path}")
            else:
                self.logger.warning("  [WARNING] WikiText vocab file not found, falling back to on-the-fly vocabulary build.")

            wiki_seq_len = getattr(self.config, 'sequence_length', self.config.segment_length * 32)

            wiki_max_vocab_size = 50000
            existing_vocab = self.token_vocabulary

            train_dataset = WikiTextDataset(
                str(train_path),
                seq_len=wiki_seq_len,
                vocab=existing_vocab,
                vocab_path=str(vocab_path) if vocab_path.exists() else None,
                max_vocab_size=wiki_max_vocab_size
            )

            val_dataset = WikiTextDataset(
                str(val_path),
                seq_len=wiki_seq_len,
                vocab_path=str(vocab_path) if vocab_path.exists() else None,
                vocab=train_dataset.vocab
            )

            effective_batch_size = self.config.batch_size
            use_pin_memory = self.device.type == "cuda"
            num_workers = 0
            if self.device.type == "cuda":
                cpu_count = os.cpu_count() or 1
                num_workers = max(1, min(4, cpu_count - 1))

            loader_kwargs: Dict[str, Any] = {
                "batch_size": effective_batch_size,
                "pin_memory": use_pin_memory,
                "num_workers": num_workers,
            }
            if num_workers > 0:
                loader_kwargs["persistent_workers"] = True
                loader_kwargs["prefetch_factor"] = 2

            train_loader = DataLoader(
                train_dataset,
                shuffle=True,
                **loader_kwargs,
            )
            val_loader = DataLoader(
                val_dataset,
                shuffle=False,
                **loader_kwargs,
            )

            self.logger.info(f"  Sequence length: {wiki_seq_len}")
            self.logger.info(f"  Segment size: {self.config.segment_length}")
            self.logger.info(f"  Max vocab size: {wiki_max_vocab_size}")
            self.logger.info(f"  Effective batch size: {effective_batch_size}")
            self.logger.info(f"  Pin memory: {use_pin_memory}")
            self.logger.info(f"  DataLoader workers: {num_workers}")
            self.config.effective_batch_size = effective_batch_size
            self.token_vocabulary = train_dataset.vocab
            
            vocab_size = len(train_dataset.vocab)
            self.config.input_size = self.config.hidden_size
            self.config.output_size = vocab_size
            self.config.vocab_size = vocab_size
            self.logger.info(f"Dataset: WikiText (Vocab size: {vocab_size})")

        else:
            raise ValueError(f"Unknown dataset: {self.config.dataset}")

        return train_loader, val_loader

    def setup_data(self) -> Tuple[DataLoader, DataLoader]:
        """Construct deterministic data loaders with task lengths decoupled from memory length."""
        self.logger.info("=" * 60)
        self.logger.info("Setting up Data Loaders")
        self.logger.info("=" * 60)
        self.copy_target_length = None
        task_seq_len = self.config.synthetic_sequence_length or max(128, self.config.segment_length * 4)

        if self.config.dataset == "adding_problem":
            train_dataset = AddingProblemDataset(
                num_samples=50000,
                seq_len=task_seq_len,
                seed=self.config.seed,
            )
            val_dataset = AddingProblemDataset(
                num_samples=5000,
                seq_len=task_seq_len,
                seed=self.config.seed + 1,
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                generator=self._make_dataloader_generator(100),
                worker_init_fn=seed_worker,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                generator=self._make_dataloader_generator(101),
                worker_init_fn=seed_worker,
            )
            self.logger.info("Dataset: Adding Problem")
            self.logger.info(f"  Sequence length: {task_seq_len}")
            self.logger.info(f"  Train batches: {len(train_loader)}")
            self.logger.info(f"  Val batches: {len(val_loader)}")

        elif self.config.dataset == "copying_memory":
            copy_seq_len = task_seq_len
            train_dataset = CopyingMemoryDataset(
                num_samples=50000,
                seq_len=copy_seq_len,
                num_copy=10,
                num_symbols=8,
                seed=self.config.seed,
            )
            val_dataset = CopyingMemoryDataset(
                num_samples=5000,
                seq_len=copy_seq_len,
                num_copy=10,
                num_symbols=8,
                seed=self.config.seed + 1,
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                generator=self._make_dataloader_generator(110),
                worker_init_fn=seed_worker,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                generator=self._make_dataloader_generator(111),
                worker_init_fn=seed_worker,
            )

            self.config.input_size = train_dataset.output_size
            self.config.output_size = train_dataset.output_size
            self.config.sequence_output = True
            self.copy_target_length = train_dataset.num_copy

            self.logger.info("Dataset: Copying Memory")
            self.logger.info(f"  Input size: {self.config.input_size}")
            self.logger.info(f"  Output size: {self.config.output_size}")
            self.logger.info(f"  Sequence length: {copy_seq_len}")
            self.logger.info(f"  Symbols to copy: {train_dataset.num_copy}")
            self.logger.info(f"  Symbol vocabulary: {train_dataset.num_symbols} + blank + delimiter")

        elif self.config.dataset == "sequential_mnist":
            train_dataset = SequentialMNIST(train=True, pixel_level=True)
            val_dataset = SequentialMNIST(train=False, pixel_level=True)
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                generator=self._make_dataloader_generator(120),
                worker_init_fn=seed_worker,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                generator=self._make_dataloader_generator(121),
                worker_init_fn=seed_worker,
            )

            sample_x, sample_y = next(iter(train_loader))
            self.logger.info("Dataset: Sequential MNIST")
            self.logger.info(f"  Input shape: {sample_x.shape}")
            self.logger.info(f"  Label shape: {sample_y.shape} (expected: [batch])")
            self.logger.info(f"  Sample labels: {sample_y[:5]}")

            self.config.input_size = sample_x.size(-1)
            self.config.output_size = 10

        elif self.config.dataset == "timeseries":
            possible_csv_paths = [
                Path(f"data/{self.config.experiment_name}/synthetic_timeseries.csv"),
                Path("data/timeseries/synthetic_timeseries.csv"),
                Path("data/adding_problem/synthetic_timeseries.csv"),
            ]

            csv_path = next((path for path in possible_csv_paths if path.exists()), None)
            if csv_path is None:
                raise FileNotFoundError(
                    f"Time series CSV not found. Tried paths: {[str(path) for path in possible_csv_paths]}"
                )

            dataset = TimeSeriesDataset.from_csv(
                csv_path=str(csv_path),
                seq_len=task_seq_len,
                pred_len=20,
                target_col=0,
                stride=10,
            )
            num_windows = len(dataset)
            base_train_end = max(1, int(num_windows * 0.8))
            gap_windows = max(1, math.ceil((dataset.seq_len + dataset.pred_len) / max(1, dataset.stride)))
            val_start = min(num_windows - 1, base_train_end + gap_windows)
            if val_start <= 0 or val_start >= num_windows:
                val_start = max(1, int(num_windows * 0.8))
            train_indices = list(range(0, val_start - gap_windows))
            val_indices = list(range(val_start, num_windows))
            if not train_indices or not val_indices:
                raise ValueError(
                    f"Time series dataset is too small for a leakage-safe chronological split: {num_windows} windows."
                )

            train_dataset = torch.utils.data.Subset(dataset, train_indices)
            val_dataset = torch.utils.data.Subset(dataset, val_indices)
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                generator=self._make_dataloader_generator(130),
                worker_init_fn=seed_worker,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                generator=self._make_dataloader_generator(131),
                worker_init_fn=seed_worker,
            )

            sample_x, sample_y = dataset[0]
            self.config.input_size = sample_x.size(-1)
            self.config.output_size = sample_y.size(-1)

            self.logger.info("Dataset: Time Series")
            self.logger.info(f"  CSV: {csv_path}")
            self.logger.info(f"  Input shape: {sample_x.shape}")
            self.logger.info(f"  Target shape: {sample_y.shape}")
            self.logger.info(f"  Sequence length: {task_seq_len}")
            self.logger.info(
                f"  Chronological split: train_windows={len(train_indices)}, "
                f"val_windows={len(val_indices)}, gap_windows={gap_windows}"
            )
            self.logger.info(f"  Train batches: {len(train_loader)}")
            self.logger.info(f"  Val batches: {len(val_loader)}")

        elif self.config.dataset == "wikitext":
            possible_train_paths = [
                Path("data/wikitext/wiki103.train.tokens"),
                Path("data/wikitext/wikitext-103/wiki.train.tokens"),
                Path("data/wikitext/wiki2.train.tokens"),
                Path("data/wikitext/wikitext-2/wiki.train.tokens"),
                Path("data/wikitext/mini_wiki.train.tokens"),
            ]

            train_path = next((path for path in possible_train_paths if path.exists()), None)
            if train_path is None:
                raise FileNotFoundError(
                    f"WikiText training data not found. Tried paths: {[str(path) for path in possible_train_paths]}"
                )

            val_path = train_path.parent / train_path.name.replace(".train.", ".valid.")
            if not val_path.exists():
                val_path = train_path.parent / "wiki.valid.tokens"
            vocab_path = train_path.parent / train_path.name.replace(".train.tokens", ".vocab")

            self.logger.info("Loading WikiText:")
            self.logger.info(f"  Train: {train_path}")
            if not val_path.exists():
                raise FileNotFoundError(f"WikiText validation data not found: {val_path}")
            self.logger.info(f"  Valid: {val_path}")
            if vocab_path.exists():
                self.logger.info(f"  Vocab: {vocab_path}")
            else:
                self.logger.warning("  [WARNING] WikiText vocab file not found, falling back to on-the-fly vocabulary build.")

            wiki_seq_len = getattr(self.config, "sequence_length", self.config.segment_length * 32)
            wiki_max_vocab_size = 50000
            existing_vocab = self.token_vocabulary

            train_dataset = WikiTextDataset(
                str(train_path),
                seq_len=wiki_seq_len,
                split="train",
                vocab=existing_vocab,
                vocab_path=str(vocab_path) if vocab_path.exists() else None,
                max_vocab_size=wiki_max_vocab_size,
            )
            val_dataset = WikiTextDataset(
                str(val_path),
                seq_len=wiki_seq_len,
                split="valid",
                vocab_path=str(vocab_path) if vocab_path.exists() else None,
                vocab=train_dataset.vocab,
                max_vocab_size=wiki_max_vocab_size,
            )

            effective_batch_size = self.config.batch_size
            use_pin_memory = self.device.type == "cuda"
            num_workers = 0
            if self.device.type == "cuda":
                cpu_count = os.cpu_count() or 1
                num_workers = max(1, min(4, cpu_count - 1))

            train_loader_kwargs: Dict[str, Any] = {
                "batch_size": effective_batch_size,
                "pin_memory": use_pin_memory,
                "num_workers": num_workers,
                "shuffle": True,
                "generator": self._make_dataloader_generator(140),
                "worker_init_fn": seed_worker,
            }
            val_loader_kwargs: Dict[str, Any] = {
                "batch_size": effective_batch_size,
                "pin_memory": use_pin_memory,
                "num_workers": num_workers,
                "shuffle": False,
                "generator": self._make_dataloader_generator(141),
                "worker_init_fn": seed_worker,
            }
            if num_workers > 0:
                train_loader_kwargs["persistent_workers"] = True
                train_loader_kwargs["prefetch_factor"] = 2
                val_loader_kwargs["persistent_workers"] = True
                val_loader_kwargs["prefetch_factor"] = 2

            train_loader = DataLoader(train_dataset, **train_loader_kwargs)
            val_loader = DataLoader(val_dataset, **val_loader_kwargs)

            self.logger.info(f"  Sequence length: {wiki_seq_len}")
            self.logger.info(f"  Validation stride: {val_dataset.stride}")
            self.logger.info(f"  Segment size: {self.config.segment_length}")
            self.logger.info(f"  Max vocab size: {wiki_max_vocab_size}")
            self.logger.info(f"  Effective batch size: {effective_batch_size}")
            self.logger.info(f"  Pin memory: {use_pin_memory}")
            self.logger.info(f"  DataLoader workers: {num_workers}")
            self.config.effective_batch_size = effective_batch_size
            self.token_vocabulary = train_dataset.vocab

            vocab_size = len(train_dataset.vocab)
            self.config.input_size = self.config.hidden_size
            self.config.output_size = vocab_size
            self.config.vocab_size = vocab_size
            self.logger.info(f"Dataset: WikiText (Vocab size: {vocab_size})")

        else:
            raise ValueError(f"Unknown dataset: {self.config.dataset}")

        return train_loader, val_loader

    def _legacy_setup_model(self) -> nn.Module:
        """Legacy single-layer model construction preserved for controlled comparisons."""
        self.logger.info("=" * 60)
        self.logger.info("Setting up Model")
        self.logger.info("=" * 60)

        if self.device.type == "cuda" and self.config.auto_memory_guard:
            if 0.0 < self.config.cuda_memory_fraction < 1.0:
                try:
                    torch.cuda.set_per_process_memory_fraction(self.config.cuda_memory_fraction)
                    self.logger.info(f"Applied CUDA memory fraction guard: {self.config.cuda_memory_fraction:.2f}")
                except Exception as exc:
                    self.logger.warning(f"Failed to apply CUDA memory fraction guard: {exc}")

            estimated_seq = self.config.sequence_length if self.config.dataset == "wikitext" else (self.config.segment_length * 4)
            effective_batch = self.config.effective_batch_size or self.config.batch_size
            estimated_tokens = effective_batch * estimated_seq * max(1, self.config.num_layers)
            if not self.config.enable_activation_checkpointing and estimated_tokens >= 32768:
                self.config.enable_activation_checkpointing = True
                self.logger.warning(
                    "Auto-enabled gradient checkpoint to reduce risk of CUDA memory spill "
                    f"(estimated tokens/step={estimated_tokens}, effective_batch={effective_batch}, "
                    f"num_layers={self.config.num_layers})."
                )

        if self.config.dataset == "wikitext":
            grm_output_size = self.config.hidden_size
        else:
            grm_output_size = self.config.output_size

        self.logger.info("Using the segment-memory recurrent backend")
        model = SegmentRecurrentMemoryModel(
            input_size=self.config.input_size,
            hidden_size=self.config.hidden_size,
            output_size=grm_output_size,
            rnn_type=self.config.rnn_type,
            memory_key_dim=self.config.memory_key_dim,
            memory_initialization_mode=self.config.memory_initialization_mode,
            retrieval_query_source=self.config.retrieval_query_source,
            segment_summary_source=self.config.segment_summary_source,
            segment_length=self.config.segment_length,
            memory_capacity_segments=self.config.memory_capacity_segments,
            retrieval_top_k=self.config.retrieval_top_k,
            memory_storage_dtype=self.config.memory_storage_dtype,
            recomputation_ratio=self.config.recomputation_ratio,
            segment_pooling_mode="mean",
            retrieval_fusion_mode=self.config.retrieval_fusion_mode,
            use_layer_norm=True,
            dropout=self.config.dropout,
            batch_first=self.config.batch_first,
            enable_activation_checkpointing=self.config.enable_activation_checkpointing,
            cuda_cpp_debug_fallback=self.config.cuda_cpp_debug_fallback,
        )

        if self.config.dataset == "wikitext":
            model = SegmentMemoryLanguageModel(
                model,
                vocab_size=getattr(self.config, 'vocab_size', 10000),
                embed_dim=self.config.hidden_size,
                token_to_id=self.token_vocabulary,
            )

        model = model.to(self.device)

        compile_target = model.grm_rnn if hasattr(model, 'grm_rnn') else model
        runtime_status = None
        if hasattr(compile_target, 'get_runtime_optimization_status'):
            runtime_status = compile_target.get_runtime_optimization_status(device_type=self.device.type)

        self.logger.info(f"Model created:")
        if hasattr(model, 'grm_rnn'):
            self.logger.info(f"  {model.grm_rnn.extra_repr()}")
            self.logger.info(f"  Vocab size: {model.output_layer.out_features}")
        else:
            self.logger.info(f"  {model.extra_repr()}")
        self.logger.info(f"  Device: {self.device}")
        self.logger.info(f"  Storage dtype: {self.config.memory_storage_dtype}")
        self.logger.info(f"  Grad clip: {self.config.gradient_clip_norm}")
        self.logger.info(f"  Gradient checkpoint: {self.config.enable_activation_checkpointing}")
        self.logger.info("  Runtime path: CUDA/C++ wrapper with automatic PyTorch fallback")
        self.logger.info(f"  CUDA/C++ debug fallback: {self.config.cuda_cpp_debug_fallback}")
        self.logger.info(f"  Auto memory guard: {self.config.auto_memory_guard}")
        if self.config.max_train_batches is not None or self.config.max_val_batches is not None:
            self.logger.info(
                f"  Batch limits: train={self.config.max_train_batches or 'all'}, "
                f"val={self.config.max_val_batches or 'all'}"
            )
        if self.runtime_env_status.get("is_wsl"):
            self.logger.info(
                f"  WSL compiler env: nvcc={self.runtime_env_status.get('nvcc') or 'missing'}, "
                f"cc={self.runtime_env_status.get('cc') or 'missing'}, "
                f"as={self.runtime_env_status.get('as') or 'missing'}"
            )
            self.logger.info(
                f"  WSL project location: {self.runtime_env_status.get('project_location_label')} "
                f"(windows_mount={self.runtime_env_status.get('project_on_windows_mount', False)})"
            )
        if runtime_status is not None:
            self.logger.info(f"  Runtime optimization status: {runtime_status}")
            self._log_runtime_optimization_status(model, prefix="  ")

        return model

    def setup_model(self) -> nn.Module:
        """Construct the active segment-memory model family with optional depth scaling."""
        self.logger.info("=" * 60)
        self.logger.info("Setting up Model")
        self.logger.info("=" * 60)

        if self.device.type == "cuda" and self.config.auto_memory_guard:
            if 0.0 < self.config.cuda_memory_fraction < 1.0:
                try:
                    torch.cuda.set_per_process_memory_fraction(self.config.cuda_memory_fraction)
                    self.logger.info(f"Applied CUDA memory fraction guard: {self.config.cuda_memory_fraction:.2f}")
                except Exception as exc:
                    self.logger.warning(f"Failed to apply CUDA memory fraction guard: {exc}")

            estimated_seq = (
                self.config.sequence_length
                if self.config.dataset == "wikitext"
                else (self.config.synthetic_sequence_length or max(128, self.config.segment_length * 4))
            )
            effective_batch = self.config.effective_batch_size or self.config.batch_size
            estimated_tokens = effective_batch * estimated_seq * max(1, self.config.num_layers)
            if not self.config.enable_activation_checkpointing and estimated_tokens >= 32768:
                self.config.enable_activation_checkpointing = True
                self.logger.warning(
                    "Auto-enabled gradient checkpoint to reduce risk of CUDA memory spill "
                    f"(estimated tokens/step={estimated_tokens}, effective_batch={effective_batch}, "
                    f"num_layers={self.config.num_layers})."
                )

        grm_output_size = self.config.hidden_size if self.config.dataset == "wikitext" else self.config.output_size
        model_cls = HierarchicalSegmentMemoryModel if self.config.num_layers > 1 else SegmentRecurrentMemoryModel
        common_kwargs = dict(
            input_size=self.config.input_size,
            hidden_size=self.config.hidden_size,
            output_size=grm_output_size,
            rnn_type=self.config.rnn_type,
            memory_key_dim=self.config.memory_key_dim,
            memory_initialization_mode=self.config.memory_initialization_mode,
            retrieval_query_source=self.config.retrieval_query_source,
            segment_summary_source=self.config.segment_summary_source,
            segment_length=self.config.segment_length,
            memory_capacity_segments=self.config.memory_capacity_segments,
            retrieval_top_k=self.config.retrieval_top_k,
            memory_storage_dtype=self.config.memory_storage_dtype,
            recomputation_ratio=self.config.recomputation_ratio,
            retrieval_fusion_mode=self.config.retrieval_fusion_mode,
            use_layer_norm=True,
            dropout=self.config.dropout,
            batch_first=self.config.batch_first,
            enable_activation_checkpointing=self.config.enable_activation_checkpointing,
            cuda_cpp_debug_fallback=self.config.cuda_cpp_debug_fallback,
        )
        if model_cls is HierarchicalSegmentMemoryModel:
            common_kwargs["num_layers"] = self.config.num_layers

        self.logger.info("Using the segment-memory recurrent backend")
        model = model_cls(**common_kwargs)

        if self.config.dataset == "wikitext":
            model = SegmentMemoryLanguageModel(
                model,
                vocab_size=getattr(self.config, "vocab_size", 10000),
                embed_dim=self.config.hidden_size,
                token_to_id=self.token_vocabulary,
            )

        model = model.to(self.device)

        compile_target = model.grm_rnn if hasattr(model, "grm_rnn") else model
        runtime_status = None
        if hasattr(compile_target, "get_runtime_optimization_status"):
            runtime_status = compile_target.get_runtime_optimization_status(device_type=self.device.type)

        self.logger.info("Model created:")
        if hasattr(model, "grm_rnn"):
            self.logger.info(f"  {model.grm_rnn.extra_repr()}")
            self.logger.info(f"  Vocab size: {model.output_layer.out_features}")
        else:
            self.logger.info(f"  {model.extra_repr()}")
        self.logger.info(f"  Device: {self.device}")
        self.logger.info(f"  Num layers: {self.config.num_layers}")
        self.logger.info(f"  Storage dtype: {self.config.memory_storage_dtype}")
        self.logger.info(f"  Grad clip: {self.config.gradient_clip_norm}")
        self.logger.info(f"  Gradient checkpoint: {self.config.enable_activation_checkpointing}")
        self.logger.info(f"  Seed: {self.config.seed}")
        self.logger.info(f"  Deterministic: {self.config.deterministic}")
        self.logger.info("  Runtime path: CUDA/C++ wrapper with automatic PyTorch fallback")
        self.logger.info(f"  CUDA/C++ debug fallback: {self.config.cuda_cpp_debug_fallback}")
        self.logger.info(f"  Auto memory guard: {self.config.auto_memory_guard}")
        if self.config.max_train_batches is not None or self.config.max_val_batches is not None:
            self.logger.info(
                f"  Batch limits: train={self.config.max_train_batches or 'all'}, "
                f"val={self.config.max_val_batches or 'all'}"
            )
        if self.runtime_env_status.get("is_wsl"):
            self.logger.info(
                f"  WSL compiler env: nvcc={self.runtime_env_status.get('nvcc') or 'missing'}, "
                f"cc={self.runtime_env_status.get('cc') or 'missing'}, "
                f"as={self.runtime_env_status.get('as') or 'missing'}"
            )
            self.logger.info(
                f"  WSL project location: {self.runtime_env_status.get('project_location_label')} "
                f"(windows_mount={self.runtime_env_status.get('project_on_windows_mount', False)})"
            )
        if runtime_status is not None:
            self.logger.info(f"  Runtime optimization status: {runtime_status}")
            self._log_runtime_optimization_status(model, prefix="  ")

        return model

    def _build_optimizer_param_groups(self, model: nn.Module) -> List[Dict[str, Any]]:
        decay_params: List[nn.Parameter] = []
        no_decay_params: List[nn.Parameter] = []

        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if (
                parameter.ndim <= 1
                or name.endswith(".bias")
                or "norm" in name.lower()
                or "embedding" in name.lower()
            ):
                no_decay_params.append(parameter)
            else:
                decay_params.append(parameter)

        param_groups: List[Dict[str, Any]] = []
        if decay_params:
            param_groups.append({"params": decay_params, "weight_decay": self.config.weight_decay})
        if no_decay_params:
            param_groups.append({"params": no_decay_params, "weight_decay": 0.0})
        return param_groups

    def _compute_wikitext_loss_stats(
        self,
        outputs: Tensor,
        targets: Tensor,
        ignore_index: int = 0,
    ) -> Tuple[Tensor, float, int]:
        loss, total_nll, total_tokens = self._compute_wikitext_chunked_loss(
            outputs,
            targets,
            ignore_index=ignore_index,
        )
        return loss, float(total_nll.item()), total_tokens

    def _get_wikitext_loss_chunk_length(self, outputs: Tensor) -> int:
        """Choose a conservative sequence chunk length for language-model loss evaluation."""
        if outputs.ndim != 3:
            return 1

        sequence_dim = 1 if self.config.batch_first else 0
        sequence_length = int(outputs.size(sequence_dim))
        if self.device.type != "cuda":
            return sequence_length

        batch_dim = 0 if self.config.batch_first else 1
        micro_batch = max(1, int(outputs.size(batch_dim)))
        target_tokens_per_chunk = 256
        chunk_length = max(32, target_tokens_per_chunk // micro_batch)
        return min(sequence_length, chunk_length)

    def _compute_wikitext_chunked_loss(
        self,
        outputs: Tensor,
        targets: Tensor,
        ignore_index: int = 0,
    ) -> Tuple[Tensor, Tensor, int]:
        """Compute token-normalized NLL in sequence chunks to reduce peak CE workspace."""
        if outputs.ndim != 3:
            raise ValueError(f"Expected rank-3 logits for WikiText, got {outputs.ndim}")

        if not self.config.batch_first:
            outputs = outputs.transpose(0, 1)

        sequence_length = int(outputs.size(1))
        vocab_size = int(outputs.size(-1))
        chunk_length = self._get_wikitext_loss_chunk_length(outputs)
        total_nll = outputs.new_zeros(())
        total_tokens = 0

        for start in range(0, sequence_length, chunk_length):
            end = min(sequence_length, start + chunk_length)
            logits_chunk = outputs[:, start:end, :]
            targets_chunk = targets[:, start:end]
            valid_tokens = int(targets_chunk.ne(ignore_index).sum().item())
            if valid_tokens == 0:
                continue

            chunk_nll = F.cross_entropy(
                logits_chunk.transpose(1, 2),
                targets_chunk,
                ignore_index=ignore_index,
                reduction="sum",
            )
            total_nll = total_nll + chunk_nll
            total_tokens += valid_tokens

        if total_tokens == 0:
            zero = outputs.new_zeros(())
            return zero, zero, 0

        return total_nll / total_tokens, total_nll, total_tokens

    def _log_runtime_optimization_status(self, model: nn.Module, prefix: str = "") -> None:
        """Emit a compact summary of the active CUDA/C++ execution path."""
        compile_target = model.grm_rnn if hasattr(model, 'grm_rnn') else model
        if not hasattr(compile_target, 'get_runtime_optimization_status'):
            return

        status = compile_target.get_runtime_optimization_status(device_type=self.device.type)
        cuda_cpp_status = status.get('cuda_cpp_runtime', {})
        if cuda_cpp_status:
            self.logger.info(
                f"{prefix}CUDA/C++ runtime: "
                f"loaded={cuda_cpp_status.get('extension_loaded', False)}, "
                f"cuda_sources={cuda_cpp_status.get('compiled_with_cuda_sources', False)}, "
                f"cuda_kernels={cuda_cpp_status.get('compiled_with_cuda_kernels', False)}, "
                f"cuda_dispatch={cuda_cpp_status.get('cuda_backend_available', False)}, "
                f"torch_cuda={cuda_cpp_status.get('torch_cuda_available', False)}, "
                f"debug_fallback={status.get('cuda_cpp_debug_fallback', False)}, "
                f"load_error={cuda_cpp_status.get('load_error') or 'n/a'}"
            )

    def _create_criterion(self) -> nn.Module:
        if self.config.dataset == "wikitext":
            return nn.CrossEntropyLoss(ignore_index=0)
        if self.config.dataset in ["copying_memory", "sequential_mnist"]:
            return nn.CrossEntropyLoss()
        return nn.MSELoss()

    def _resolve_scheduler_plan(self, train_loader_len: int) -> Tuple[int, int, int, int]:
        """Compute optimizer-step, warmup, and cosine schedules from the active run budget."""
        effective_train_loader_len = train_loader_len
        if self.config.max_train_batches is not None:
            effective_train_loader_len = min(train_loader_len, self.config.max_train_batches)

        accumulation = max(1, self.config.gradient_accumulation_factor)
        steps_per_epoch = max(1, (effective_train_loader_len + accumulation - 1) // accumulation)
        total_steps = max(1, self.config.num_epochs * steps_per_epoch)

        requested_warmup_steps = max(0, int(self.config.warmup_steps))
        if total_steps <= 1:
            effective_warmup_steps = 0
        else:
            warmup_cap = max(1, math.ceil(total_steps * 0.1))
            effective_warmup_steps = min(requested_warmup_steps, warmup_cap)

        cosine_steps = max(1, total_steps - effective_warmup_steps)
        return steps_per_epoch, total_steps, effective_warmup_steps, cosine_steps

    def _legacy_setup_optimizer_and_criterion(self, model, train_loader_len: int):
        """Legacy optimizer construction retained for comparison against earlier runs."""
        self.logger.info("=" * 60)
        self.logger.info("Setting up Optimizer and Criterion")
        self.logger.info("=" * 60)

        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        criterion = self._create_criterion()

        steps_per_epoch, total_steps, effective_warmup_steps, cosine_steps = self._resolve_scheduler_plan(train_loader_len)
        if effective_warmup_steps > 0:
            cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=cosine_steps,
                eta_min=self.config.min_lr,
            )
            warmup_scheduler = optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1e-6,
                end_factor=1.0,
                total_iters=effective_warmup_steps,
            )
            scheduler = optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[effective_warmup_steps],
            )
        else:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=cosine_steps,
                eta_min=self.config.min_lr,
            )

        self.logger.info(f"Optimizer: AdamW (lr={self.config.learning_rate}, wd={self.config.weight_decay})")
        self.logger.info(f"Criterion: {criterion.__class__.__name__}")
        if effective_warmup_steps != self.config.warmup_steps:
            self.logger.warning(
                f"Adjusted warmup_steps from {self.config.warmup_steps} to {effective_warmup_steps} "
                f"for this run budget (total optimizer steps={total_steps})."
            )
        self.logger.info(
            f"Scheduler: {'SequentialLR' if effective_warmup_steps > 0 else 'CosineAnnealingLR'}"
            f"(warmup={effective_warmup_steps}, cosine_steps={cosine_steps}, total_steps={total_steps})"
        )

        return optimizer, criterion, scheduler

    def setup_optimizer_and_criterion(self, model, train_loader_len: int):
        """Construct AdamW parameter groups with linear warmup and cosine decay."""
        self.logger.info("=" * 60)
        self.logger.info("Setting up Optimizer and Criterion")
        self.logger.info("=" * 60)

        optimizer = optim.AdamW(
            self._build_optimizer_param_groups(model),
            lr=self.config.learning_rate,
            weight_decay=0.0,
        )

        criterion = self._create_criterion()

        steps_per_epoch, total_steps, effective_warmup_steps, cosine_steps = self._resolve_scheduler_plan(train_loader_len)
        if effective_warmup_steps > 0:
            cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=cosine_steps,
                eta_min=self.config.min_lr,
            )
            warmup_scheduler = optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1e-6,
                end_factor=1.0,
                total_iters=effective_warmup_steps,
            )
            scheduler = optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[effective_warmup_steps],
            )
        else:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=cosine_steps,
                eta_min=self.config.min_lr,
            )

        self.logger.info(f"Optimizer: AdamW (lr={self.config.learning_rate}, wd={self.config.weight_decay})")
        self.logger.info(f"Criterion: {criterion.__class__.__name__}")
        if effective_warmup_steps != self.config.warmup_steps:
            self.logger.warning(
                f"Adjusted warmup_steps from {self.config.warmup_steps} to {effective_warmup_steps} "
                f"for this run budget (total optimizer steps={total_steps})."
            )
        self.logger.info(
            f"Scheduler: {'SequentialLR' if effective_warmup_steps > 0 else 'CosineAnnealingLR'}"
            f"(warmup={effective_warmup_steps}, cosine_steps={cosine_steps}, total_steps={total_steps})"
        )

        return optimizer, criterion, scheduler

    def compute_gradient_norm(self, model: nn.Module) -> float:
        """Compute the global L2 norm of all available parameter gradients."""
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm

    def _compute_loss(
        self,
        model: nn.Module,
        outputs: Tensor,
        targets: Tensor,
        criterion: nn.Module
    ) -> Tensor:
        """Select the task-appropriate training objective for the active benchmark."""
        is_wikitext = self.config.dataset == "wikitext"
        is_copying = self.config.dataset == "copying_memory"

        if is_wikitext:
            ignore_index = getattr(criterion, "ignore_index", 0)
            loss, _, _ = self._compute_wikitext_chunked_loss(
                outputs,
                targets,
                ignore_index=ignore_index,
            )
            return loss

        elif is_copying:
            return self._compute_copying_memory_loss(outputs, targets, criterion)

        elif self.config.dataset == "sequential_mnist":
            return criterion(outputs, targets)

        else:
            return criterion(outputs, targets)

    def _compute_copying_memory_loss(
        self,
        outputs: Tensor,
        targets: Tensor,
        criterion: nn.Module
    ) -> Tensor:
        """Cross-entropy over the full discrete copying sequence."""
        if self.config.batch_first:
            logits = outputs.reshape(-1, outputs.size(-1))
        else:
            logits = outputs.transpose(0, 1).reshape(-1, outputs.size(-1))
        return criterion(logits, targets.reshape(-1).long())

    def _compute_accuracy(
        self,
        outputs: Tensor,
        targets: Tensor
    ) -> Tuple[int, int]:
        """Compute token-level or label-level accuracy for the active benchmark."""
        is_wikitext = self.config.dataset == "wikitext"
        is_copying = self.config.dataset == "copying_memory"

        if is_wikitext:
            if self.config.batch_first:
                pred = outputs.argmax(dim=-1)
                valid_mask = targets.ne(0)
                correct = ((pred == targets) & valid_mask).sum().item()
                total = valid_mask.sum().item()
            else:
                pred = outputs.argmax(dim=-1).transpose(0, 1)  # [batch, seq]
                valid_mask = targets.ne(0)
                correct = ((pred == targets) & valid_mask).sum().item()
                total = valid_mask.sum().item()
            return correct, total

        elif is_copying:
            num_copy = self.copy_target_length or 0
            if num_copy <= 0:
                raise RuntimeError("copy_target_length is not initialized for copying-memory evaluation.")

            if self.config.batch_first:
                pred_outputs = outputs.argmax(dim=-1)[:, -num_copy:]
                target_outputs = targets[:, -num_copy:]
            else:
                pred_outputs = outputs.argmax(dim=-1).transpose(0, 1)[:, -num_copy:]
                target_outputs = targets[:, -num_copy:]

            correct = (pred_outputs == target_outputs).sum().item()
            return int(correct), int(target_outputs.numel())

        elif self.config.dataset == "sequential_mnist":
            pred = outputs.argmax(dim=1)
            correct = (pred == targets).sum().item()
            return correct, targets.size(0)

        else:
            return 0, 0

    def _compute_copying_sequence_accuracy(self, outputs: Tensor, targets: Tensor) -> Tuple[int, int]:
        num_copy = self.copy_target_length or 0
        if num_copy <= 0:
            return 0, 0

        if self.config.batch_first:
            pred_outputs = outputs.argmax(dim=-1)[:, -num_copy:]
        else:
            pred_outputs = outputs.argmax(dim=-1).transpose(0, 1)[:, -num_copy:]

        target_outputs = targets[:, -num_copy:]
        sequence_ok = (pred_outputs == target_outputs).all(dim=1)
        return int(sequence_ok.sum().item()), int(sequence_ok.numel())

    def train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        scheduler: optim.lr_scheduler._LRScheduler,
        epoch: int
    ) -> Dict[str, float]:
        """Run one optimization epoch and return aggregate training metrics."""
        model.train()

        total_loss = 0.0
        total_grad_norm = 0.0
        total_step_time = 0.0
        total_items = 0
        num_batches = 0
        accum_steps = max(1, self.config.gradient_accumulation_factor)
        optimizer.zero_grad()

        last_grad_norm = 0.0
        max_batches = self.config.max_train_batches
        display_total_batches = min(len(train_loader), max_batches) if max_batches is not None else len(train_loader)
        progress_bar = self._create_progress_bar(display_total_batches, f"Train Epoch {epoch}")

        for batch_idx, (x, y) in enumerate(train_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            non_blocking = self.device.type == "cuda"
            x = x.to(self.device, non_blocking=non_blocking)
            y = y.to(self.device, non_blocking=non_blocking)
            batch_items = int(x.shape[0] * x.shape[1]) if x.ndim >= 2 else int(x.shape[0])

            is_wikitext = self.config.dataset == "wikitext"
            is_copying = self.config.dataset == "copying_memory"
            return_all = is_wikitext or is_copying

            if self.device.type == "cuda":
                torch.cuda.synchronize()
            step_start = time.perf_counter()

            if self.config.use_mixed_precision and self.scaler is not None:
                with autocast(device_type=self.device.type, enabled=(self.device.type == 'cuda')):
                    outputs, h_final, c_final, aux = model(
                        x,
                        return_all_outputs=return_all,
                        return_hidden_states=False,
                        reset_memory=True
                    )

                    loss = self._compute_loss(model, outputs, y, criterion)

                self.scaler.scale(loss / accum_steps).backward()
            else:
                outputs, h_final, c_final, aux = model(
                    x,
                    return_all_outputs=return_all,
                    return_hidden_states=False,
                    reset_memory=True
                )

                loss = self._compute_loss(model, outputs, y, criterion)

                (loss / accum_steps).backward()

            should_step = (
                (batch_idx + 1) % accum_steps == 0 or
                (batch_idx + 1) == display_total_batches
            )

            if should_step:
                if self.config.use_mixed_precision and self.scaler is not None:
                    self.scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        self.config.gradient_clip_norm
                    )
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        self.config.gradient_clip_norm
                    )
                    optimizer.step()

                last_grad_norm = float(grad_norm)
                optimizer.zero_grad()
                scheduler.step()

            if self.device.type == "cuda":
                torch.cuda.synchronize()
            step_time = time.perf_counter() - step_start

            total_loss += loss.item()
            total_grad_norm += last_grad_norm
            total_step_time += step_time
            total_items += batch_items
            num_batches += 1

            current_lr = optimizer.param_groups[0]['lr']
            items_per_sec = batch_items / step_time if step_time > 0 else 0.0
            if progress_bar is not None:
                progress_bar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    grad=f"{last_grad_norm:.4f}",
                    lr=f"{current_lr:.2e}",
                    step=f"{step_time:.2f}s",
                    ips=f"{items_per_sec:.1f}",
                )
                progress_bar.update(1)
            elif batch_idx % self.config.log_every == 0:
                self.logger.info(
                    f"  Epoch {epoch} | Batch {batch_idx + 1}/{display_total_batches} | "
                    f"Loss: {loss.item():.6f} | Grad: {last_grad_norm:.4f} | "
                    f"LR: {current_lr:.2e} | Step: {step_time:.4f}s | Items/s: {items_per_sec:.1f}"
                )

            self.global_step += 1

        if progress_bar is not None:
            progress_bar.close()

        avg_loss = total_loss / max(1, num_batches)
        avg_grad_norm = total_grad_norm / max(1, num_batches)
        avg_step_time = total_step_time / max(1, num_batches)
        items_per_sec = total_items / total_step_time if total_step_time > 0 else 0.0

        return {
            'loss': avg_loss,
            'grad_norm': avg_grad_norm,
            'step_time': avg_step_time,
            'items_per_sec': items_per_sec
        }

    @torch.no_grad()
    def _legacy_validate(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        criterion: nn.Module,
        epoch: int
    ) -> Dict[str, float]:
        """Legacy validation loop retained for historical comparisons."""
        model.eval()

        total_loss = 0.0
        total_step_time = 0.0
        total_items = 0
        recompute_rates = []
        correct = 0
        total = 0
        copying_sequence_correct = 0
        copying_sequence_total = 0
        num_batches = 0
        max_batches = self.config.max_val_batches
        display_total_batches = min(len(val_loader), max_batches) if max_batches is not None else len(val_loader)
        progress_bar = self._create_progress_bar(display_total_batches, f"Val Epoch {epoch}")

        for batch_idx, (x, y) in enumerate(val_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            non_blocking = self.device.type == "cuda"
            x = x.to(self.device, non_blocking=non_blocking)
            y = y.to(self.device, non_blocking=non_blocking)
            batch_items = int(x.shape[0] * x.shape[1]) if x.ndim >= 2 else int(x.shape[0])

            is_wikitext = self.config.dataset == "wikitext"
            is_copying = self.config.dataset == "copying_memory"
            return_all = is_wikitext or is_copying

            if self.device.type == "cuda":
                torch.cuda.synchronize()
            step_start = time.perf_counter()

            outputs, h_final, c_final, aux = model(
                x,
                return_all_outputs=return_all,
                return_hidden_states=False,
                reset_memory=True
            )

            current_recompute_rate = float(aux.get('recompute_rate', 0.0)) if aux else 0.0
            if aux and 'recompute_rate' in aux:
                recompute_rates.append(aux['recompute_rate'])

            loss = self._compute_loss(model, outputs, y, criterion)

            correct_batch, total_batch = self._compute_accuracy(outputs, y)
            correct += correct_batch
            total += total_batch
            if is_copying:
                seq_correct, seq_total = self._compute_copying_sequence_accuracy(outputs, y)
                copying_sequence_correct += seq_correct
                copying_sequence_total += seq_total

            if self.device.type == "cuda":
                torch.cuda.synchronize()
            step_time = time.perf_counter() - step_start

            total_loss += loss.item()
            total_step_time += step_time
            total_items += batch_items
            num_batches += 1

            if progress_bar is not None:
                items_per_sec = batch_items / step_time if step_time > 0 else 0.0
                postfix = {
                    'loss': f"{loss.item():.4f}",
                    'step': f"{step_time:.2f}s",
                    'ips': f"{items_per_sec:.1f}",
                    'recomp': f"{current_recompute_rate:.2%}",
                }
                if total > 0:
                    postfix['acc'] = f"{100 * correct / total:.2f}%"
                progress_bar.set_postfix(**postfix)
                progress_bar.update(1)

        if progress_bar is not None:
            progress_bar.close()

        avg_loss = total_loss / max(1, num_batches)
        avg_step_time = total_step_time / max(1, num_batches)
        items_per_sec = total_items / total_step_time if total_step_time > 0 else 0.0

        avg_recompute_rate = sum(recompute_rates) / len(recompute_rates) if recompute_rates else 0.0

        self.logger.info(f"\n  Validation Loss: {avg_loss:.6f}")
        if self.config.dataset == "wikitext":
            self.logger.info(f"  Perplexity: {math.exp(min(avg_loss, 20.0)):.4f}")
        self.logger.info(f"  Recompute Rate: {avg_recompute_rate:.2%}")
        self.logger.info(f"  Validation Step: {avg_step_time:.4f}s | Items/s: {items_per_sec:.1f}")

        if avg_recompute_rate < self.config.preset_recomputation_ratio * 0.5:
            self.logger.warning(
                f"  [WARNING] Recompute rate ({avg_recompute_rate:.2%}) < 50% of preset threshold "
                f"({self.config.preset_recomputation_ratio:.2%}) - "
                f"Gradients may be insufficient!"
            )

        result = {
            'val_loss': avg_loss,
            'recompute_rate': avg_recompute_rate,
            'val_step_time': avg_step_time,
            'val_items_per_sec': items_per_sec
        }
        if self.config.dataset == "wikitext":
            result['perplexity'] = math.exp(min(avg_loss, 20.0))

        if total > 0:
            result['accuracy'] = correct / total
            self.logger.info(f"  Accuracy: {100 * correct / total:.2f}%")
        if copying_sequence_total > 0:
            result['sequence_accuracy'] = copying_sequence_correct / copying_sequence_total
            self.logger.info(f"  Sequence Accuracy: {100 * copying_sequence_correct / copying_sequence_total:.2f}%")

        return result

    @torch.no_grad()
    def validate(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        criterion: nn.Module,
        epoch: int
    ) -> Dict[str, float]:
        """Validation with token-weighted WikiText loss."""
        model.eval()

        total_loss = 0.0
        total_step_time = 0.0
        total_items = 0
        total_nll = 0.0
        total_tokens = 0
        recompute_rates = []
        correct = 0
        total = 0
        copying_sequence_correct = 0
        copying_sequence_total = 0
        num_batches = 0
        max_batches = self.config.max_val_batches
        display_total_batches = min(len(val_loader), max_batches) if max_batches is not None else len(val_loader)
        progress_bar = self._create_progress_bar(display_total_batches, f"Val Epoch {epoch}")

        for batch_idx, (x, y) in enumerate(val_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            non_blocking = self.device.type == "cuda"
            x = x.to(self.device, non_blocking=non_blocking)
            y = y.to(self.device, non_blocking=non_blocking)
            batch_items = int(x.shape[0] * x.shape[1]) if x.ndim >= 2 else int(x.shape[0])

            is_wikitext = self.config.dataset == "wikitext"
            is_copying = self.config.dataset == "copying_memory"
            return_all = is_wikitext or is_copying

            if self.device.type == "cuda":
                torch.cuda.synchronize()
            step_start = time.perf_counter()

            outputs, h_final, c_final, aux = model(
                x,
                return_all_outputs=return_all,
                return_hidden_states=False,
                reset_memory=True,
            )

            current_recompute_rate = float(aux.get("recompute_rate", 0.0)) if aux else 0.0
            if aux and "recompute_rate" in aux:
                recompute_rates.append(aux["recompute_rate"])

            if is_wikitext:
                loss, batch_nll, batch_tokens = self._compute_wikitext_loss_stats(outputs, y, ignore_index=0)
                total_nll += batch_nll
                total_tokens += batch_tokens
            else:
                loss = self._compute_loss(model, outputs, y, criterion)

            correct_batch, total_batch = self._compute_accuracy(outputs, y)
            correct += correct_batch
            total += total_batch
            if is_copying:
                seq_correct, seq_total = self._compute_copying_sequence_accuracy(outputs, y)
                copying_sequence_correct += seq_correct
                copying_sequence_total += seq_total

            if self.device.type == "cuda":
                torch.cuda.synchronize()
            step_time = time.perf_counter() - step_start

            total_loss += loss.item()
            total_step_time += step_time
            total_items += batch_items
            num_batches += 1

            if progress_bar is not None:
                items_per_sec = batch_items / step_time if step_time > 0 else 0.0
                postfix = {
                    "loss": f"{loss.item():.4f}",
                    "step": f"{step_time:.2f}s",
                    "ips": f"{items_per_sec:.1f}",
                    "recomp": f"{current_recompute_rate:.2%}",
                }
                if total > 0:
                    postfix["acc"] = f"{100 * correct / total:.2f}%"
                progress_bar.set_postfix(**postfix)
                progress_bar.update(1)

        if progress_bar is not None:
            progress_bar.close()

        if self.config.dataset == "wikitext" and total_tokens > 0:
            avg_loss = total_nll / total_tokens
        else:
            avg_loss = total_loss / max(1, num_batches)
        avg_step_time = total_step_time / max(1, num_batches)
        items_per_sec = total_items / total_step_time if total_step_time > 0 else 0.0
        avg_recompute_rate = sum(recompute_rates) / len(recompute_rates) if recompute_rates else 0.0

        self.logger.info(f"\n  Validation Loss: {avg_loss:.6f}")
        if self.config.dataset == "wikitext":
            self.logger.info(f"  Perplexity: {math.exp(min(avg_loss, 20.0)):.4f}")
            self.logger.info(f"  Evaluated tokens: {total_tokens}")
        self.logger.info(f"  Recompute Rate: {avg_recompute_rate:.2%}")
        self.logger.info(f"  Validation Step: {avg_step_time:.4f}s | Items/s: {items_per_sec:.1f}")

        if avg_recompute_rate < self.config.preset_recomputation_ratio * 0.5:
            self.logger.warning(
                f"  [WARNING] Recompute rate ({avg_recompute_rate:.2%}) < 50% of preset threshold "
                f"({self.config.preset_recomputation_ratio:.2%}) - "
                f"Gradients may be insufficient!"
            )

        result = {
            "val_loss": avg_loss,
            "recompute_rate": avg_recompute_rate,
            "val_step_time": avg_step_time,
            "val_items_per_sec": items_per_sec,
        }
        if self.config.dataset == "wikitext":
            result["perplexity"] = math.exp(min(avg_loss, 20.0))

        if total > 0:
            result["accuracy"] = correct / total
            self.logger.info(f"  Accuracy: {100 * correct / total:.2f}%")
        if copying_sequence_total > 0:
            result["sequence_accuracy"] = copying_sequence_correct / copying_sequence_total
            self.logger.info(f"  Sequence Accuracy: {100 * copying_sequence_correct / copying_sequence_total:.2f}%")

        return result

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler,
        epoch: int,
        metrics: Dict[str, float],
        filename: Optional[str] = None
    ):
        """Persist a training checkpoint and refresh the best-model snapshot when improved."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            filename = f"epoch_{epoch}.pt"

        checkpoint_path = checkpoint_dir / filename

        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config.to_dict(),
            'best_val_loss': self.best_val_loss,
            'precision_stage': self.current_precision_stage,
            'precision_stage_transitioned': self.precision_stage_transitioned,
        }
        if self.token_vocabulary is not None:
            checkpoint['token_vocabulary'] = self.token_vocabulary

        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"  Checkpoint saved: {checkpoint_path}")

        current_val_loss = metrics.get('val_loss', float('inf'))
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            best_filename = f"best_{self.config.experiment_name}_model.pt"
            best_path = checkpoint_dir / best_filename
            torch.save(checkpoint, best_path)
            self.logger.info(f"  [SUCCESS] Best model saved: {best_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load a checkpoint onto the active device and restore metadata."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        self.logger.info(f"  Epoch: {checkpoint['epoch']}")
        self.logger.info(f"  Val Loss: {checkpoint['metrics']['val_loss']:.6f}")

        precision_stage = checkpoint.get('precision_stage', checkpoint.get('phase'))
        if precision_stage is not None:
            if precision_stage == "phase1":
                precision_stage = "stage1"
            elif precision_stage == "phase2":
                precision_stage = "stage2"
            self.current_precision_stage = precision_stage
            self.precision_stage_transitioned = checkpoint.get(
                'precision_stage_transitioned',
                checkpoint.get('phase_transitioned', self.current_precision_stage == "stage2"),
            )
            self.config.precision_stage = self.current_precision_stage
            self.logger.info(f"  Precision stage: {self.current_precision_stage}")
        vocabulary = checkpoint.get('token_vocabulary', checkpoint.get('text_vocab'))
        if vocabulary is not None:
            self.token_vocabulary = vocabulary

        return checkpoint

    def _apply_checkpoint_config(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint_config = checkpoint.get('config')
        if not checkpoint_config:
            return

        dtype_map = {
            "torch.float16": torch.float16,
            "torch.float32": torch.float32,
            "torch.bfloat16": torch.bfloat16,
        }
        override_field_map = {
            'dataset': 'dataset',
            'experiment_name': 'experiment_name',
            'run_name': 'experiment_name',
            'input_size': 'input_size',
            'hidden_size': 'hidden_size',
            'output_size': 'output_size',
            'num_layers': 'num_layers',
            'rnn_type': 'rnn_type',
            'memory_key_dim': 'memory_key_dim',
            'memory_initialization_mode': 'memory_initialization_mode',
            'retrieval_query_source': 'retrieval_query_source',
            'segment_summary_source': 'segment_summary_source',
            'segment_length': 'segment_length',
            'memory_capacity_segments': 'memory_capacity_segments',
            'retrieval_top_k': 'retrieval_top_k',
            'sequence_length': 'sequence_length',
            'synthetic_sequence_length': 'synthetic_sequence_length',
            'memory_storage_dtype': 'memory_storage_dtype',
            'recomputation_ratio': 'recomputation_ratio',
            'preset_recomputation_ratio': 'preset_recomputation_ratio',
            'enable_activation_checkpointing': 'enable_activation_checkpointing',
            'retrieval_fusion_mode': 'retrieval_fusion_mode',
            'dropout': 'dropout',
            'batch_first': 'batch_first',
            'sequence_output': 'sequence_output',
            'vocab_size': 'vocab_size',
            'precision_stage': 'precision_stage',
            'auto_phase_transition': 'enable_precision_curriculum',
            'enable_precision_curriculum': 'enable_precision_curriculum',
            'phase1_min_epochs': 'stage1_min_epochs',
            'stage1_min_epochs': 'stage1_min_epochs',
            'phase1_stable_epochs': 'stage1_stability_window',
            'stage1_stability_window': 'stage1_stability_window',
            'phase1_improvement_threshold': 'stage1_relative_improvement_threshold',
            'stage1_relative_improvement_threshold': 'stage1_relative_improvement_threshold',
            'use_log_segment': 'enable_logarithmic_segmentation',
            'enable_logarithmic_segmentation': 'enable_logarithmic_segmentation',
            'max_level': 'maximum_hierarchy_level',
            'maximum_hierarchy_level': 'maximum_hierarchy_level',
            'segments_per_level': 'segments_per_hierarchy_level',
            'segments_per_hierarchy_level': 'segments_per_hierarchy_level',
            'base_segment_size': 'base_segment_length',
            'base_segment_length': 'base_segment_length',
            'enable_adaptive_level': 'enable_adaptive_hierarchy_level',
            'enable_adaptive_hierarchy_level': 'enable_adaptive_hierarchy_level',
        }

        self.logger.info("Applying checkpoint architecture/configuration for evaluation.")
        for checkpoint_field_name, config_field_name in override_field_map.items():
            if checkpoint_field_name not in checkpoint_config:
                continue
            value = checkpoint_config[checkpoint_field_name]
            if config_field_name == 'memory_storage_dtype' and isinstance(value, str):
                value = dtype_map.get(value, torch.float32)
            if config_field_name == "precision_stage" and value in {"phase1", "phase2"}:
                value = "stage1" if value == "phase1" else "stage2"
            setattr(self.config, config_field_name, value)

    def _save_results_json(self, results: Dict[str, Any], prefix: str) -> Path:
        output_dir = Path(self.config.log_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"{prefix}_{timestamp}.json"
        with open(output_path, 'w', encoding='utf-8') as fh:
            json.dump(results, fh, ensure_ascii=False, indent=2)
        self.logger.info(f"Evaluation results saved: {output_path}")
        return output_path

    def _run_proxy_benchmark_suite(self, model: nn.Module, benchmark_data_path: str) -> Dict[str, Any]:
        if self.config.dataset != "wikitext":
            reason = "The proxy benchmark suite only supports language-model checkpoints trained on WikiText."
            self.logger.warning(reason)
            return {"status": "skipped", "reason": reason}

        self.logger.info("\n" + "=" * 60)
        self.logger.info("Running Proxy Benchmark Suite")
        self.logger.info("=" * 60)

        results = run_proxy_benchmark_suite(
            model,
            self.device,
            data_path=benchmark_data_path,
            hidden_size=self.config.hidden_size,
        )

        for task_name, task_result in results.items():
            if 'accuracy' in task_result:
                self.logger.info(f"  {task_name}: accuracy={task_result['accuracy']:.4f}")
            else:
                self.logger.warning(f"  {task_name}: {task_result}")

        return results

    def run_evaluation_suite(
        self,
        model: nn.Module,
        val_loader: Optional[DataLoader] = None,
        criterion: Optional[nn.Module] = None,
        epoch: int = 0,
        run_proxy_benchmarks: bool = False,
        benchmark_data_path: str = "./data",
        result_prefix: str = "evaluation",
    ) -> Dict[str, Any]:
        if val_loader is None:
            _, val_loader = self.setup_data()
        if criterion is None:
            criterion = self._create_criterion()

        validation_metrics = self.validate(model, val_loader, criterion, epoch)
        results: Dict[str, Any] = {
            "epoch": epoch,
            "validation": validation_metrics,
        }

        if run_proxy_benchmarks:
            results["proxy_benchmark_suite"] = self._run_proxy_benchmark_suite(model, benchmark_data_path)

        self._save_results_json(results, result_prefix)
        return results

    def evaluate_only(
        self,
        checkpoint_path: str,
        run_proxy_benchmarks: bool = False,
        benchmark_data_path: str = "./data",
    ) -> Dict[str, Any]:
        self.logger.info("\n" + "=" * 60)
        self.logger.info("Segment-Memory Evaluation")
        self.logger.info("=" * 60)

        checkpoint = self.load_checkpoint(checkpoint_path)
        self._apply_checkpoint_config(checkpoint)

        self.logger.info("\nEvaluation Configuration:")
        for key, value in self.config.to_dict().items():
            self.logger.info(f"  {key}: {value}")

        _, val_loader = self.setup_data()
        model = self.setup_model()
        model.load_state_dict(checkpoint['model_state_dict'])

        return self.run_evaluation_suite(
            model,
            val_loader=val_loader,
            criterion=self._create_criterion(),
            epoch=checkpoint.get('epoch', 0),
            run_proxy_benchmarks=run_proxy_benchmarks,
            benchmark_data_path=benchmark_data_path,
            result_prefix="eval_only",
        )

    def train(self, resume_from: Optional[str] = None):
        """Execute the full training protocol, optionally resuming from a checkpoint."""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("Segment-Memory Training Protocol")
        self.logger.info("=" * 60)

        self.logger.info("\nTraining Configuration:")
        for key, value in self.config.to_dict().items():
            self.logger.info(f"  {key}: {value}")

        train_loader, val_loader = self.setup_data()
        model = self.setup_model()
        optimizer, criterion, scheduler = self.setup_optimizer_and_criterion(
            model,
            train_loader_len=len(train_loader)
        )

        start_epoch = 1
        if resume_from:
            checkpoint = self.load_checkpoint(resume_from)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            self.current_epoch = checkpoint['epoch']
            self.global_step = checkpoint['global_step']
            self.best_val_loss = checkpoint['best_val_loss']

        self.logger.info("\n" + "=" * 60)
        self.logger.info("Starting Training")
        self.logger.info("=" * 60)

        for epoch in range(start_epoch, self.config.num_epochs + 1):
            self.current_epoch = epoch

            self.logger.info(f"\nEpoch {epoch}/{self.config.num_epochs}")
            self.logger.info("-" * 40)

            train_metrics = self.train_epoch(
                model, train_loader, optimizer, criterion, scheduler, epoch
            )

            val_metrics = self.validate(
                model, val_loader, criterion, epoch
            )
            self._log_runtime_optimization_status(model, prefix="  ")

            self.metrics_history['train_loss'].append(train_metrics['loss'])
            self.metrics_history['val_loss'].append(val_metrics['val_loss'])
            self.metrics_history['recompute_rate'].append(val_metrics.get('recompute_rate', 0))
            self.metrics_history['grad_norm'].append(train_metrics['grad_norm'])
            self.metrics_history['learning_rate'].append(optimizer.param_groups[0]['lr'])

            if self.config.enable_precision_curriculum and not self.precision_stage_transitioned:
                should_transition = self._should_advance_precision_stage(epoch, val_metrics['val_loss'])
                if should_transition:
                    self._advance_to_stage2_mixed_precision(model, optimizer)

            if epoch % self.config.save_every == 0:
                self.save_checkpoint(
                    model, optimizer, scheduler, epoch,
                    {**val_metrics, 'train_loss': train_metrics['loss']}
                )

        self.logger.info("\n" + "=" * 60)
        self.logger.info("Training Complete!")
        self.logger.info("=" * 60)
        self.logger.info(f"Best validation loss: {self.best_val_loss:.6f}")
        self.logger.info(f"Final precision stage: {self.current_precision_stage}")

        return model, self.metrics_history

    def _should_advance_precision_stage(self, epoch: int, current_val_loss: float) -> bool:
        """Advance from stage 1 to stage 2 when validation loss has stabilized."""
        self.validation_loss_history.append(current_val_loss)

        if epoch < self.config.stage1_min_epochs:
            return False

        required_history = self.config.stage1_stability_window + 1
        if len(self.validation_loss_history) < required_history:
            return False

        recent_losses = self.validation_loss_history[-required_history:]
        first_loss = recent_losses[0]
        last_loss = recent_losses[-1]
        min_loss = min(recent_losses)
        max_loss = max(recent_losses)

        if first_loss > 1e-8:
            relative_change = abs(last_loss - first_loss) / first_loss
        else:
            relative_change = 0.0

        if len(recent_losses) >= 2:
            last_two = recent_losses[-2:]
            is_rising = last_two[1] > last_two[0] * 1.02  # 2%
            if is_rising:
                return False

        is_stable = relative_change < self.config.stage1_relative_improvement_threshold

        if is_stable:
            self.logger.info(
                f"\n  [Precision Curriculum] Validation loss has stabilized:\n"
                f"    First loss: {first_loss:.6f}\n"
                f"    Last loss: {last_loss:.6f}\n"
                f"    Relative change: {relative_change:.4f} < threshold: {self.config.stage1_relative_improvement_threshold}\n"
                f"    Min/Max in window: {min_loss:.6f} / {max_loss:.6f}"
            )
            return True

        return False

    def _advance_to_stage2_mixed_precision(self, model: nn.Module, optimizer: optim.Optimizer):
        """Switch to the mixed-precision stage while preserving the target recomputation ratio."""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("TRANSITIONING TO STAGE 2 (FP16 + Balanced Recompute)")
        self.logger.info("=" * 60)

        self.current_precision_stage = "stage2"
        self.precision_stage_transitioned = True
        self.config.precision_stage = "stage2"

        def update_memory_dtype(module):
            if hasattr(module, 'memory_bank'):
                old_threshold = module.memory_bank.recomputation_ratio
                module.memory_bank.memory_storage_dtype = torch.float16
                module.memory_bank.recomputation_ratio = self.config.preset_recomputation_ratio
                self.logger.info(f"  Updated {module.__class__.__name__}.memory_bank:")
                self.logger.info(f"    memory_storage_dtype: float32 -> float16")
                self.logger.info(f"    recomputation_ratio: {old_threshold} -> {self.config.preset_recomputation_ratio} (balanced preset)")

        model.apply(update_memory_dtype)

        self.config.use_mixed_precision = True
        self.scaler = GradScaler() if self.device.type == "cuda" else None

        def clear_recompute_cache(module):
            if hasattr(module, 'memory_bank') and hasattr(module.memory_bank, '_recompute_cache'):
                module.memory_bank._recompute_cache.clear()

        model.apply(clear_recompute_cache)

        self.logger.info("  Mixed precision: enabled")
        self.logger.info("  GradScaler: initialized")
        self.logger.info("=" * 60)



def main():
    parser = argparse.ArgumentParser(
        description="Train the active segment-memory architecture.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available presets:
  auto
  laptop_2k
  laptop_4k
  desktop_4k
  desktop_8k
  server_16k
  paper_2k / paper_4k / paper_8k

Examples:
  python grm/utils/train.py --dataset adding_problem --preset auto --enable_precision_curriculum
  python grm/utils/train.py --dataset timeseries --experiment_name timeseries --preset auto
""",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["adding_problem", "copying_memory", "sequential_mnist", "timeseries", "wikitext"],
        help="Benchmark dataset used for training.",
    )

    parser.add_argument("--preset", type=str, default="auto", help="Hardware or benchmark-aligned preset.")

    parser.add_argument(
        "--experiment_name",
        "--run_name",
        dest="experiment_name",
        type=str,
        default=None,
        help="Experiment identifier used for checkpoint and log directories.",
    )

    parser.add_argument(
        "--precision_stage",
        "--phase",
        dest="precision_stage",
        type=str,
        default="stage1",
        choices=["stage1", "stage2", "phase1", "phase2"],
        help="Optimization stage governing numeric precision and storage policy.",
    )

    parser.add_argument("--hidden_size", type=int, default=None)
    parser.add_argument("--memory_key_dim", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--segment_length", type=int, default=None)
    parser.add_argument("--memory_capacity_segments", type=int, default=None)
    parser.add_argument("--retrieval_top_k", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--sequence_length", type=int, default=None)
    parser.add_argument("--synthetic_sequence_length", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--gradient_clip_norm", type=float, default=1.0)
    parser.add_argument("--recomputation_ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true", default=False)
    parser.add_argument("--max_train_batches", type=int, default=None, help="Optional capped number of training batches.")
    parser.add_argument("--max_val_batches", type=int, default=None, help="Optional capped number of validation batches.")

    parser.add_argument(
        "--enable_activation_checkpointing",
        action="store_true",
        default=False,
        help="Enable activation checkpointing to trade compute for memory.",
    )
    parser.add_argument(
        "--cuda_cpp_debug_fallback",
        action="store_true",
        default=False,
        help="Keep the CUDA entry points but force the numerical path through the PyTorch reference implementation.",
    )
    parser.add_argument(
        "--disable_memory_guard",
        action="store_true",
        default=False,
        help="Disable the CUDA memory safety guard.",
    )
    parser.add_argument(
        "--cuda_memory_fraction",
        type=float,
        default=0.92,
        help="Maximum fraction of device memory reserved by the process.",
    )

    parser.add_argument(
        "--enable_precision_curriculum",
        "--auto_transition",
        dest="enable_precision_curriculum",
        action="store_true",
        help="Automatically advance from the full-precision stage to the mixed-precision stage.",
    )
    parser.add_argument("--stage1_min_epochs", "--phase1_min_epochs", dest="stage1_min_epochs", type=int, default=3)
    parser.add_argument("--stage1_stability_window", "--phase1_stable_epochs", dest="stage1_stability_window", type=int, default=2)
    parser.add_argument(
        "--stage1_relative_improvement_threshold",
        "--phase1_improvement_threshold",
        dest="stage1_relative_improvement_threshold",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--enable_logarithmic_segmentation",
        "--log_segment",
        dest="enable_logarithmic_segmentation",
        action="store_true",
        help="Reserved flag for the unsupported logarithmic segmentation variant.",
    )
    parser.add_argument(
        "--maximum_hierarchy_level",
        "--max_level",
        dest="maximum_hierarchy_level",
        type=int,
        default=10,
        help="Maximum hierarchy level used by the reserved logarithmic variant.",
    )
    parser.add_argument(
        "--segments_per_hierarchy_level",
        "--segments_per_level",
        dest="segments_per_hierarchy_level",
        type=int,
        default=256,
        help="Maximum number of segments allocated to each hierarchy level.",
    )
    parser.add_argument(
        "--base_segment_length",
        "--base_segment_size",
        dest="base_segment_length",
        type=int,
        default=1,
        help="Base segment length used by the reserved logarithmic variant.",
    )
    parser.add_argument(
        "--enable_adaptive_hierarchy_level",
        "--enable_adaptive_level",
        dest="enable_adaptive_hierarchy_level",
        action="store_true",
        default=True,
        help="Enable adaptive hierarchy-level selection in the reserved logarithmic variant.",
    )

    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--eval_only", action="store_true", help="Skip training and evaluate a checkpoint only.")
    parser.add_argument(
        "--run_proxy_benchmarks",
        "--run_paper_eval",
        dest="run_proxy_benchmarks",
        action="store_true",
        help="Run the proxy benchmark suite after validation.",
    )
    parser.add_argument(
        "--benchmark_data_root",
        "--paper_eval_data_root",
        dest="benchmark_data_root",
        type=str,
        default="./data",
        help="Root directory containing proxy benchmark data.",
    )
    parser.add_argument("--checkpoint_root", type=str, default="./checkpoints")
    parser.add_argument("--log_root", type=str, default="./logs")

    args = parser.parse_args()

    if args.enable_logarithmic_segmentation:
        parser.error("`--enable_logarithmic_segmentation` is not supported by the active architecture.")
    if args.eval_only and not args.resume:
        parser.error("`--eval_only` requires `--resume <checkpoint_path>`.")

    from grm.utils.config import load_preset_config
    preset_config = load_preset_config(args.preset)

    precision_stage = "stage1" if args.precision_stage == "phase1" else "stage2" if args.precision_stage == "phase2" else args.precision_stage
    experiment_name = args.experiment_name or args.dataset

    config = TrainingExperimentConfig(
        dataset=args.dataset,
        experiment_name=experiment_name,
        hidden_size=args.hidden_size or preset_config.hidden_size,
        memory_key_dim=args.memory_key_dim or getattr(preset_config, 'memory_key_dim', None),
        num_layers=args.num_layers or getattr(preset_config, 'num_layers', 1),
        rnn_type=getattr(preset_config, 'rnn_type', 'linear'),
        memory_initialization_mode=getattr(preset_config, 'memory_initialization_mode', 'checkpoint'),
        retrieval_query_source=getattr(preset_config, 'retrieval_query_source', 'input'),
        segment_summary_source=getattr(preset_config, 'segment_summary_source', 'input_mean'),
        segment_length=args.segment_length or preset_config.segment_length,
        memory_capacity_segments=args.memory_capacity_segments or preset_config.memory_capacity_segments,
        retrieval_top_k=args.retrieval_top_k or preset_config.retrieval_top_k,
        sequence_length=args.sequence_length or getattr(preset_config, 'sequence_length', 2048),
        synthetic_sequence_length=args.synthetic_sequence_length or getattr(preset_config, 'synthetic_sequence_length', None),
        batch_size=args.batch_size or getattr(preset_config, 'batch_size', 16),
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip_norm,
        recomputation_ratio=args.recomputation_ratio if args.recomputation_ratio is not None else preset_config.recomputation_ratio,
        preset_recomputation_ratio=args.recomputation_ratio if args.recomputation_ratio is not None else preset_config.recomputation_ratio,
        enable_activation_checkpointing=args.enable_activation_checkpointing or preset_config.enable_activation_checkpointing,
        cuda_cpp_debug_fallback=args.cuda_cpp_debug_fallback,
        auto_memory_guard=not args.disable_memory_guard,
        cuda_memory_fraction=args.cuda_memory_fraction,
        memory_storage_dtype=preset_config.memory_storage_dtype,
        gradient_accumulation_factor=getattr(preset_config, 'gradient_accumulation_factor', 1),
        checkpoint_dir=os.path.join(args.checkpoint_root, experiment_name),
        log_dir=os.path.join(args.log_root, experiment_name),
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        seed=args.seed,
        deterministic=args.deterministic,
        precision_stage=precision_stage,
        enable_precision_curriculum=args.enable_precision_curriculum,
        stage1_min_epochs=args.stage1_min_epochs,
        stage1_stability_window=args.stage1_stability_window,
        stage1_relative_improvement_threshold=args.stage1_relative_improvement_threshold,
        enable_logarithmic_segmentation=args.enable_logarithmic_segmentation,
        maximum_hierarchy_level=args.maximum_hierarchy_level,
        segments_per_hierarchy_level=args.segments_per_hierarchy_level,
        base_segment_length=args.base_segment_length,
        enable_adaptive_hierarchy_level=args.enable_adaptive_hierarchy_level,
    )

    if precision_stage == 'stage1':
        config.memory_storage_dtype = torch.float32
        config.use_mixed_precision = False
        stage_name = "Stage1"
    else:
        config.memory_storage_dtype = torch.float16
        config.use_mixed_precision = True
        stage_name = "Stage2"

    if args.dataset == "wikitext":
        config.input_size = config.hidden_size

    logger = setup_logger(Path(config.log_dir), stage_name)
    logger.info(f"Preset: {args.preset}")
    logger.info(
        f"Architecture: segment-memory recurrent model, "
        f"segment_length={config.segment_length}, memory_capacity_segments={config.memory_capacity_segments}"
    )

    logger.info(
        f"Hidden: {config.hidden_size}, Key: {config.memory_key_dim}, Layers: {config.num_layers}, "
        f"Batch: {config.batch_size}, Top-K: {config.retrieval_top_k}"
    )

    trainer = ExperimentTrainer(config, logger)
    if args.eval_only:
        results = trainer.evaluate_only(
            checkpoint_path=args.resume,
            run_proxy_benchmarks=args.run_proxy_benchmarks,
            benchmark_data_path=args.benchmark_data_root,
        )
        logger.info("\nEvaluation completed!")
        logger.info(f"  val_loss: {results['validation']['val_loss']:.6f}")
        if 'perplexity' in results['validation']:
            logger.info(f"  perplexity: {results['validation']['perplexity']:.6f}")
        if 'accuracy' in results['validation']:
            logger.info(f"  accuracy: {results['validation']['accuracy']:.6f}")
        return

    model, metrics = trainer.train(resume_from=args.resume)

    logger.info(f"\nTraining completed!")
    for key, values in metrics.items():
        if values:
            logger.info(f"  {key}: {values[-1]:.6f}")

    if args.run_proxy_benchmarks:
        trainer.run_evaluation_suite(
            model,
            epoch=trainer.current_epoch,
            run_proxy_benchmarks=True,
            benchmark_data_path=args.benchmark_data_root,
            result_prefix="post_train_eval",
        )


if __name__ == "__main__":
    main()
