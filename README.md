# GRM

GRM is a research codebase for segment-level recurrent memory models inspired by the paper summarized in [`Memory_Caching_RNNs_with_Growing_Memory.md`](./Memory_Caching_RNNs_with_Growing_Memory.md). The active implementation targets the benchmark-aligned linear-memory path, sparse top-k retrieval, segment-level memory archival, and Linux-first CUDA execution.

This repository is now organized around academic terminology:

- `SegmentRecurrentMemoryModel` for the single-layer architecture
- `HierarchicalSegmentMemoryModel` for stacked segment-memory blocks
- `TopKMemoryRetriever` for sparse retrieval over segment summaries
- `RetrievedStateFusion` for online-state and retrieved-state fusion
- `MemoryArchitectureConfig` and `TrainingExperimentConfig` for preset and experiment configuration

## Status

- Linux and WSL2 are the supported training environments.
- The CUDA extension is part of the main execution path for Linux.
- The benchmark helpers under `grm/evaluation/paper_tasks.py` are proxy evaluations, not official benchmark harnesses.
- The project keeps small model sizes for local experimentation even when the surrounding configuration follows the paper more closely.

## Requirements

- Python 3.12
- PyTorch with CUDA support for native-kernel training
- A Linux or WSL2 environment for training
- A C++ compiler and CUDA toolkit if you want to rebuild `grm_cuda_ext`

Recommended Linux packages:

- `build-essential`
- `binutils`
- `python3-dev`

## Installation

Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision tqdm numpy pandas
```

## Build The CUDA Extension

Rebuild the extension in place:

```bash
python3 setup.py build_ext --inplace --force
```

Check runtime status:

```bash
python3 -c "from grm.core.cuda_ops import get_grm_cuda_runtime_status; print(get_grm_cuda_runtime_status())"
```

The most important fields are:

- `extension_loaded`
- `compiled_with_cuda_kernels`
- `dispatch_kernels`
- `kernel_policy`

## Repository Layout

- [`grm/core`](./grm/core): model architecture, retrieval, fusion, and CUDA entry points
- [`grm/data`](./grm/data): synthetic and text benchmark datasets
- [`grm/utils`](./grm/utils): presets, training entry point, and optional parallel helpers
- [`grm/evaluation`](./grm/evaluation): proxy benchmark suite
- [`examples`](./examples): smoke tests and small evaluation scripts
- [`datasets`](./datasets): dataset sanity checks and usage examples

## Presets

Hardware presets are defined in [`hardware_presets.json`](./hardware_presets.json). The most relevant preset families are:

- `laptop_*` for low-memory local GPUs
- `desktop_*` for mainstream desktop GPUs
- `highend_*` for larger local accelerators
- `server_*` for workstation or datacenter GPUs
- `paper_2k`, `paper_4k`, `paper_8k` for benchmark-aligned small-model experiments
- `emergency_tiny`, `emergency_nano` for smoke tests and extreme memory pressure

## Training

The main entry point is [`grm/utils/train.py`](./grm/utils/train.py).

Fast smoke test:

```bash
python3 grm/utils/train.py \
  --dataset adding_problem \
  --preset emergency_tiny \
  --experiment_name adding_problem_smoke \
  --epochs 1 \
  --max_train_batches 1 \
  --max_val_batches 1
```

Benchmark-aligned small-model WikiText smoke test:

```bash
python3 grm/utils/train.py \
  --dataset wikitext \
  --preset paper_2k \
  --experiment_name wikitext_paper2k \
  --epochs 1 \
  --max_train_batches 1 \
  --max_val_batches 1
```

Longer training with automatic phase transition:

```bash
python3 grm/utils/train.py \
  --dataset wikitext \
  --preset paper_2k \
  --experiment_name wikitext_full \
  --enable_precision_curriculum \
  --epochs 10
```

Useful CLI arguments:

- `--segment_length`
- `--memory_capacity_segments`
- `--retrieval_top_k`
- `--memory_key_dim`
- `--sequence_length`
- `--synthetic_sequence_length`
- `--recomputation_ratio`
- `--enable_activation_checkpointing`
- `--experiment_name`
- `--resume`
- `--eval_only`

## Evaluation

Validation-only mode:

```bash
python3 grm/utils/train.py \
  --dataset wikitext \
  --preset paper_2k \
  --resume checkpoints/wikitext_full/best_wikitext_full_model.pt \
  --eval_only
```

Validation plus proxy benchmark suite:

```bash
python3 grm/utils/train.py \
  --dataset wikitext \
  --preset paper_2k \
  --resume checkpoints/wikitext_full/best_wikitext_full_model.pt \
  --eval_only \
  --run_proxy_benchmarks
```

Proxy benchmark data, when available, should live under:

- `data/piqa`
- `data/hellaswag`

## Dataset Notes

The main in-tree datasets are:

- `adding_problem`
- `copying_memory`
- `sequential_mnist`
- `timeseries`
- `wikitext`

See [`Dataset_Recommendations.md`](./Dataset_Recommendations.md) for task selection guidance.

## Tests And Smoke Checks

Core smoke test:

```bash
python3 examples/test_grm.py
```

Dataset checks:

```bash
python3 datasets/test_datasets.py
```

Checkpoint evaluation:

```bash
python3 examples/test_model.py --checkpoint <path> --dataset adding_problem
```

CUDA microbenchmark:

```bash
python3 examples/benchmark_cuda_chunk_update.py --device cuda --iters 200
```

## Current Limitations

- The active model path supports only `rnn_type="linear"`.
- Linux is the supported training platform.
- The proxy benchmark suite is not a drop-in replacement for official benchmark harnesses.
- The repository is optimized for small local experiments rather than full-scale paper reproduction.

## Troubleshooting

If the extension does not load:

1. Rebuild with `python3 setup.py build_ext --inplace --force`.
2. Confirm `torch.version.cuda` matches the available toolkit.
3. Check `nvcc`, `gcc`, `g++`, and `as` in the active shell.
4. Keep the active training checkout on the Linux filesystem when using WSL2.

If training is too slow or runs out of memory:

1. Move to a smaller preset such as `emergency_tiny` or `laptop_1k`.
2. Reduce `--batch_size`.
3. Lower `--memory_capacity_segments` or `--segment_length`.
4. Enable `--enable_activation_checkpointing`.
5. Use bounded smoke tests with `--max_train_batches` and `--max_val_batches`.
