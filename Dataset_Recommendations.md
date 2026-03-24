# Dataset Recommendations

This document is scoped to the active paper-aligned segment-memory implementation and its Linux CUDA execution path.

## Best In-Tree Choices

### Adding Problem

Use this benchmark for:

- gradient-flow smoke tests
- quick regression checks after kernel or routing changes
- verifying that long-range supervision still propagates

Strength:

- the fastest full training loop in the repository

Weakness:

- not representative of realistic language modeling or retrieval-heavy workloads

### Copying Memory

Use this benchmark for:

- delayed-recall experiments
- checking whether archived segment memories preserve discrete symbols

Strength:

- directly measures memory retention under delayed supervision

Weakness:

- synthetic and narrow

### Sequential MNIST

Use this benchmark for:

- non-text sequence sanity checks
- verifying that the architecture is not tied to tokenized corpora

Strength:

- easy to compare against simple recurrent baselines

Weakness:

- not a demanding long-context benchmark

### Time Series

Use this benchmark for:

- segmented forecasting experiments
- checking behavior on continuous multivariate inputs

Strength:

- natural fit for fixed-step segmentation

Weakness:

- difficulty depends strongly on preprocessing and forecast horizon choices

### WikiText-2

Use this benchmark for:

- the main local language-model baseline
- realistic profiling of the retrieval path
- validation of `SegmentMemoryLanguageModel`
- end-to-end CUDA runtime checks

Strength:

- best supported real-text path in the repository

Weakness:

- still much smaller than serious long-context language-model corpora

## Recommended Progression

1. `adding_problem`
2. `copying_memory`
3. `wikitext`

This order keeps iteration fast while still exercising:

- forward correctness
- backward correctness
- segment archival
- sparse retrieval
- language-model wrapping

## Preset Mapping

| Dataset | First Preset | Reason |
|---|---|---|
| `adding_problem` | `emergency_tiny` | fastest smoke path |
| `copying_memory` | `laptop_1k` | enough width for delayed recall |
| `sequential_mnist` | `laptop_2k` | balanced local baseline |
| `timeseries` | `laptop_2k` | practical default for structured features |
| `wikitext` | `paper_2k` | closest benchmark-aligned small-model path |

## Practical Commands

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

Realistic text smoke test:

```bash
python3 grm/utils/train.py \
  --dataset wikitext \
  --preset paper_2k \
  --experiment_name wikitext_smoke \
  --epochs 1 \
  --max_train_batches 1 \
  --max_val_batches 1
```

CUDA kernel microbenchmark:

```bash
python3 examples/benchmark_cuda_chunk_update.py --device cuda --iters 200
```

## External Data Guidance

If you add external datasets, prefer corpora with:

- stable train, validation, and test splits
- explicit long-context or delayed-recall pressure
- segmentable structure
- token-level or step-level supervision compatible with continuation scoring or sequence prediction

Avoid datasets that are:

- too short to trigger multiple archived segments
- dominated by single-step labels with weak temporal dependence
- tied to unsupported legacy backends

## Evaluation Caveat

Proxy benchmark helpers in [`grm/evaluation/paper_tasks.py`](./grm/evaluation/paper_tasks.py) require a language-model wrapper. Bare `SegmentRecurrentMemoryModel` instances are not sufficient for PIQA, HellaSwag, or needle-in-haystack scoring.
