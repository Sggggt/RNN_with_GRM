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

- All in Linux or WSL2 environment
- Python 3.12
- PyTorch with CUDA support for native-kernel training
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

## Complete CLI Reference

The training and evaluation entry point is [`grm/utils/train.py`](./grm/utils/train.py). The tables below document every user-facing CLI parameter exposed by the current parser.

### Run Selection

| Parameter | Default | Meaning |
| --- | --- | --- |
| `--dataset` | required | Dataset to train or evaluate. Choices: `adding_problem`, `copying_memory`, `sequential_mnist`, `timeseries`, `wikitext`. |
| `--preset` | `auto` | Hardware-aware or benchmark-aligned preset loaded from [`hardware_presets.json`](./hardware_presets.json). |
| `--experiment_name`, `--run_name` | dataset name | Identifier used to build checkpoint and log subdirectories. |
| `--precision_stage`, `--phase` | `stage1` | Starting optimization stage. Accepted values are `stage1`, `stage2`, plus legacy aliases `phase1`, `phase2`. |

### Architecture And Memory Overrides

| Parameter | Default | Meaning |
| --- | --- | --- |
| `--hidden_size` | preset value | Override the model hidden dimension. |
| `--memory_key_dim` | preset value or auto-derived | Override the key dimension used by the linear memory cell. |
| `--num_layers` | preset value | Override the number of stacked segment-memory layers. |
| `--segment_length` | preset value | Number of sequence steps processed in one segment before archival and retrieval. |
| `--memory_capacity_segments` | preset value | Maximum number of archived segments retained in the external memory bank. |
| `--retrieval_top_k` | preset value | Number of archived segments retrieved for each routing query. |
| `--batch_size` | preset value | Micro-batch size loaded by the DataLoader before gradient accumulation. |
| `--sequence_length` | preset value | Text context length for `wikitext`, measured in tokens. |
| `--synthetic_sequence_length` | preset value or derived from segment length | Sequence length used by synthetic benchmarks such as adding problem or copying memory. |

### Optimization, Reproducibility, And Bounded Runs

| Parameter | Default | Meaning |
| --- | --- | --- |
| `--epochs` | `10` | Number of training epochs. |
| `--lr` | `4e-4` | Peak learning rate passed to AdamW. |
| `--weight_decay` | `0.1` | Weight decay used by AdamW parameter groups. |
| `--gradient_clip_norm` | `1.0` | Global gradient clipping norm applied before optimizer steps. |
| `--recomputation_ratio` | preset value | Fraction of retrieved routes recomputed from checkpoints instead of directly reusing archived memories. |
| `--seed` | `42` | Global random seed used for Python, NumPy, PyTorch, and DataLoader worker seeding. |
| `--deterministic` | disabled | Enable deterministic PyTorch behavior where available. |
| `--max_train_batches` | unlimited | Cap the number of training batches per epoch for smoke tests or controlled probes. |
| `--max_val_batches` | unlimited | Cap the number of validation batches per epoch. |

### CUDA And Memory Controls

| Parameter | Default | Meaning |
| --- | --- | --- |
| `--enable_activation_checkpointing` | disabled unless enabled by preset or guard | Trade extra compute for lower activation memory usage during training. |
| `--cuda_cpp_debug_fallback` | `False` | Keep the CUDA entry points active but force the numerical path through the PyTorch reference implementation for debugging. |
| `--disable_memory_guard` | `False` | Disable the trainer's automatic CUDA memory safety logic. |
| `--cuda_memory_fraction` | `0.92` | Maximum fraction of total device memory the process is allowed to reserve. |

### Precision Curriculum

| Parameter | Default | Meaning |
| --- | --- | --- |
| `--enable_precision_curriculum`, `--auto_transition` | disabled | Automatically transition from `stage1` to `stage2` after validation loss stabilizes. |
| `--stage1_min_epochs`, `--phase1_min_epochs` | `3` | Minimum number of epochs to remain in stage 1 before transition is considered. |
| `--stage1_stability_window`, `--phase1_stable_epochs` | `2` | Number of recent validation intervals used to judge stage-1 stability. |
| `--stage1_relative_improvement_threshold`, `--phase1_improvement_threshold` | `0.01` | Maximum relative validation-loss change allowed before stage 1 is considered stable enough to transition. |

### Reserved Logarithmic-Hierarchy Flags

| Parameter | Default | Meaning |
| --- | --- | --- |
| `--enable_logarithmic_segmentation`, `--log_segment` | disabled | Reserved flag for an unsupported logarithmic segmentation variant. The current training entry point rejects this mode. |
| `--maximum_hierarchy_level`, `--max_level` | `10` | Reserved maximum hierarchy depth for the unsupported logarithmic variant. |
| `--segments_per_hierarchy_level`, `--segments_per_level` | `256` | Reserved per-level segment budget for the unsupported logarithmic variant. |
| `--base_segment_length`, `--base_segment_size` | `1` | Reserved base segment length for the unsupported logarithmic variant. |
| `--enable_adaptive_hierarchy_level`, `--enable_adaptive_level` | enabled | Reserved toggle for adaptive hierarchy-level selection in the unsupported logarithmic variant. |

### Evaluation And Artifact Paths

| Parameter | Default | Meaning |
| --- | --- | --- |
| `--resume` | none | Path to a checkpoint used for resume or evaluation-only mode. |
| `--eval_only` | disabled | Skip training and run validation or proxy evaluation from the checkpoint specified by `--resume`. |
| `--run_proxy_benchmarks`, `--run_paper_eval` | disabled | Run the repository's proxy benchmark suite after validation. |
| `--benchmark_data_root`, `--paper_eval_data_root` | `./data` | Root directory containing proxy benchmark data such as PIQA or HellaSwag files. |
| `--checkpoint_root` | `./checkpoints` | Root directory under which per-experiment checkpoints are written. |
| `--log_root` | `./logs` | Root directory under which per-experiment training logs are written. |

### Practical Notes

- `--eval_only` requires `--resume <checkpoint_path>`.
- `--enable_precision_curriculum` affects training only when the run starts in `stage1`.
- `effective_batch_size` reported in logs refers to the loader micro-batch; the true update batch is `batch_size * gradient_accumulation_factor`.
- The active architecture supports only the linear-memory backend, even though several presets retain paper-inspired naming.

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
