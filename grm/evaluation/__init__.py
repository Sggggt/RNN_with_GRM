"""Public evaluation exports for the proxy benchmark suite."""

from .paper_tasks import (
    HellaSwagBenchmarkDataset,
    NeedleInHaystackBenchmarkDataset,
    PIQABenchmarkDataset,
    evaluate_hellaswag_benchmark,
    evaluate_needle_in_haystack,
    evaluate_piqa_benchmark,
    run_proxy_benchmark_suite,
)

__all__ = [
    "HellaSwagBenchmarkDataset",
    "NeedleInHaystackBenchmarkDataset",
    "PIQABenchmarkDataset",
    "evaluate_hellaswag_benchmark",
    "evaluate_needle_in_haystack",
    "evaluate_piqa_benchmark",
    "run_proxy_benchmark_suite",
]
