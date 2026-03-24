"""Benchmark the refactored GRM C++/CUDA op entrypoint against its fallback path."""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grm.core.cuda_ops import cuda_chunk_update, get_grm_cuda_runtime_status  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark GRM cuda_chunk_update entrypoint.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", type=str, default="float32", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--chunk-len", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--key-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--memory-decay", type=float, default=0.97)
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    return device_arg


def resolve_dtype(device: str, dtype_arg: str) -> torch.dtype:
    dtype = getattr(torch, dtype_arg)
    if device == "cpu" and dtype in {torch.float16, torch.bfloat16}:
        return torch.float32
    return dtype


def make_inputs(args, device: str, dtype: torch.dtype):
    keys = torch.randn(args.batch_size, args.chunk_len, args.memory_key_dim, device=device, dtype=dtype)
    queries = torch.randn(args.batch_size, args.chunk_len, args.memory_key_dim, device=device, dtype=dtype)
    values = torch.randn(args.batch_size, args.chunk_len, args.hidden_size, device=device, dtype=dtype)
    memory = torch.randn(args.batch_size, args.hidden_size, args.memory_key_dim, device=device, dtype=dtype)
    keys = torch.nn.functional.normalize(keys, p=2, dim=-1, eps=1e-6)
    queries = torch.nn.functional.normalize(queries, p=2, dim=-1, eps=1e-6)
    return keys, values, queries, memory


def synchronize(device: str):
    if device == "cuda":
        torch.cuda.synchronize()


def benchmark_case(name: str, debug_fallback: bool, inputs, args, device: str):
    for _ in range(args.warmup):
        cuda_chunk_update(*inputs, memory_decay=args.memory_decay, enabled=True, debug_fallback=debug_fallback)
        synchronize(device)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    synchronize(device)
    start = time.perf_counter()
    for _ in range(args.iters):
        cuda_chunk_update(*inputs, memory_decay=args.memory_decay, enabled=True, debug_fallback=debug_fallback)
    synchronize(device)
    elapsed = time.perf_counter() - start

    avg_ms = elapsed * 1000.0 / args.iters
    tokens_per_s = (args.batch_size * args.chunk_len * args.iters) / max(elapsed, 1e-9)
    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if device == "cuda" else 0.0
    return {"name": name, "avg_ms": avg_ms, "tokens_per_s": tokens_per_s, "peak_mb": peak_mb}


def main():
    args = parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype(device, args.dtype)
    inputs = make_inputs(args, device=device, dtype=dtype)
    runtime = get_grm_cuda_runtime_status()

    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print(f"Runtime: {runtime}")

    fallback = benchmark_case("python_fallback", True, inputs, args, device)
    live = benchmark_case("cuda_ops_entrypoint", False, inputs, args, device)

    for result in (fallback, live):
        print(
            f"{result['name']}: avg_ms={result['avg_ms']:.3f}, "
            f"tokens/s={result['tokens_per_s']:.1f}, peak_mb={result['peak_mb']:.1f}"
        )

    if live["avg_ms"] > 0:
        print(f"speedup={fallback['avg_ms'] / live['avg_ms']:.3f}x")


if __name__ == "__main__":
    raise SystemExit(main())
