"""Smoke tests for the active segment-memory recurrent implementation."""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grm import MemoryArchitectureConfig, SegmentRecurrentMemoryModel, estimate_memory_footprint
from grm.core.cuda_ops import (
    cuda_apply_query,
    cuda_batched_memory_gather,
    cuda_chunk_update,
    get_grm_cuda_runtime_status,
)


def unpack_model_output(result):
    if isinstance(result, tuple) and len(result) == 4:
        outputs, h_final, c_final, aux = result
        return outputs, h_final, aux
    if isinstance(result, tuple) and len(result) == 3:
        outputs, h_final, aux = result
        return outputs, h_final, aux
    raise ValueError(f"Unexpected model output format: {type(result)}")


def make_chunk_case(chunk_len, batch_size=2, hidden_size=16, memory_key_dim=8):
    decay = 0.97
    k = torch.randn(batch_size, chunk_len, memory_key_dim)
    v = torch.randn(batch_size, chunk_len, hidden_size)
    q = torch.randn(batch_size, chunk_len, memory_key_dim)
    q = torch.nn.functional.normalize(q, p=2, dim=-1, eps=1e-6)
    k = torch.nn.functional.normalize(k, p=2, dim=-1, eps=1e-6)
    memory = torch.randn(batch_size, hidden_size, memory_key_dim)
    return k, v, q, memory, decay


def run_cuda_chunk_entrypoint_case(chunk_len, device="cpu", check_grad=False):
    k, v, q, memory, decay = make_chunk_case(chunk_len)
    device_obj = torch.device(device)
    k = k.to(device_obj)
    v = v.to(device_obj)
    q = q.to(device_obj)
    memory = memory.to(device_obj)

    hidden_ref, memory_ref = cuda_chunk_update(
        k,
        v,
        q,
        memory,
        memory_decay=decay,
        enabled=True,
        debug_fallback=True,
    )
    hidden_live, memory_live = cuda_chunk_update(
        k,
        v,
        q,
        memory,
        memory_decay=decay,
        enabled=True,
        debug_fallback=False,
    )

    atol = 5e-4 if device_obj.type == "cuda" else 1e-5
    rtol = 5e-4 if device_obj.type == "cuda" else 1e-5
    torch.testing.assert_close(hidden_live.cpu(), hidden_ref.cpu(), atol=atol, rtol=rtol)
    torch.testing.assert_close(memory_live.cpu(), memory_ref.cpu(), atol=atol, rtol=rtol)

    if not check_grad:
        return

    k_ref = k.clone().requires_grad_(True)
    v_ref = v.clone().requires_grad_(True)
    q_ref = q.clone().requires_grad_(True)
    memory_ref_grad = memory.clone().requires_grad_(True)
    hidden_ref, memory_ref_out = cuda_chunk_update(
        k_ref,
        v_ref,
        q_ref,
        memory_ref_grad,
        memory_decay=decay,
        enabled=True,
        debug_fallback=True,
    )
    loss_ref = hidden_ref.square().mean() + memory_ref_out.square().mean()
    loss_ref.backward()

    k_live = k.clone().requires_grad_(True)
    v_live = v.clone().requires_grad_(True)
    q_live = q.clone().requires_grad_(True)
    memory_live = memory.clone().requires_grad_(True)
    hidden_live, memory_live_out = cuda_chunk_update(
        k_live,
        v_live,
        q_live,
        memory_live,
        memory_decay=decay,
        enabled=True,
        debug_fallback=False,
    )
    loss_live = hidden_live.square().mean() + memory_live_out.square().mean()
    loss_live.backward()

    for grad_live, grad_ref in [
        (k_live.grad, k_ref.grad),
        (v_live.grad, v_ref.grad),
        (q_live.grad, q_ref.grad),
        (memory_live.grad, memory_ref_grad.grad),
    ]:
        torch.testing.assert_close(grad_live.cpu(), grad_ref.cpu(), atol=atol, rtol=rtol)


def test_cuda_cpp_runtime_status():
    print("=" * 60)
    print("Test 7: CUDA/C++ Runtime Status")
    print("=" * 60)

    status = get_grm_cuda_runtime_status()
    print(f"Runtime status: {status}")
    required_keys = {
        "extension_loaded",
        "registered_ops",
        "compiled_with_cuda_sources",
        "compiled_with_cuda_kernels",
        "dispatch_kernels",
        "kernel_policy",
    }
    if not required_keys.issubset(status):
        raise AssertionError(f"Unexpected runtime status payload: {status}")
    expected_policy_keys = {"global", "chunk_update", "apply_query", "batched_memory_gather"}
    if set(status["kernel_policy"]) != expected_policy_keys:
        raise AssertionError(f"Unexpected kernel policy payload: {status['kernel_policy']}")
    if any(policy not in {"auto", "native", "fallback"} for policy in status["kernel_policy"].values()):
        raise AssertionError(f"Unexpected kernel policy values: {status['kernel_policy']}")
    if status["compiled_with_cuda_kernels"]:
        for op_name, dispatch_status in status["dispatch_kernels"].items():
            if not dispatch_status.get("cuda", False):
                raise AssertionError(
                    f"Runtime reported CUDA kernels as available, but {op_name} lacks CUDA dispatch: {status}"
                )
    if os.name == "posix" and torch.cuda.is_available():
        if not status.get("extension_loaded", False):
            raise AssertionError(f"CUDA runtime extension is not loaded in a CUDA-capable Linux environment: {status}")
        if not status.get("compiled_with_cuda_kernels", False):
            raise AssertionError(f"CUDA kernels are unavailable in a CUDA-capable Linux environment: {status}")

    print("[PASS] Test 7 passed!\n")
    return True


def test_cuda_chunk_update_parity():
    print("=" * 60)
    print("Test 8: CUDA/C++ Chunk Update Entry Parity")
    print("=" * 60)

    run_cuda_chunk_entrypoint_case(chunk_len=8, device="cpu", check_grad=True)
    if torch.cuda.is_available():
        run_cuda_chunk_entrypoint_case(chunk_len=64, device="cuda", check_grad=True)

    print("[PASS] Test 8 passed!\n")
    return True


def test_cuda_apply_query_parity():
    print("=" * 60)
    print("Test 9: CUDA/C++ Apply Query Entry Parity")
    print("=" * 60)

    memories = torch.randn(2, 3, 4, 5, 6)
    queries = torch.randn(2, 3, 6)
    queries = torch.nn.functional.normalize(queries, p=2, dim=-1, eps=1e-6)

    hidden_ref = cuda_apply_query(memories, queries, enabled=True, debug_fallback=True)
    hidden_live = cuda_apply_query(memories, queries, enabled=True, debug_fallback=False)
    torch.testing.assert_close(hidden_live, hidden_ref, atol=1e-5, rtol=1e-5)

    memories_ref = memories.clone().requires_grad_(True)
    queries_ref = queries.clone().requires_grad_(True)
    hidden_ref = cuda_apply_query(memories_ref, queries_ref, enabled=True, debug_fallback=True)
    hidden_ref.square().mean().backward()

    memories_live = memories.clone().requires_grad_(True)
    queries_live = queries.clone().requires_grad_(True)
    hidden_live = cuda_apply_query(memories_live, queries_live, enabled=True, debug_fallback=False)
    hidden_live.square().mean().backward()

    torch.testing.assert_close(memories_live.grad, memories_ref.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(queries_live.grad, queries_ref.grad, atol=1e-5, rtol=1e-5)

    print("[PASS] Test 9 passed!\n")
    return True


def test_cuda_batched_gather_parity():
    print("=" * 60)
    print("Test 10: CUDA/C++ Batched Gather Entry Parity")
    print("=" * 60)

    memories = torch.randn(2, 7, 4, 5)
    topk_chunk = torch.tensor(
        [
            [[1, 3], [2, 5], [0, 6]],
            [[4, 2], [1, 0], [3, 5]],
        ],
        dtype=torch.long,
    )
    topk_flat = torch.tensor([[0, 2, 4], [1, 3, 5]], dtype=torch.long)

    selected_chunk_ref = cuda_batched_memory_gather(memories, topk_chunk, enabled=True, debug_fallback=True)
    selected_chunk_live = cuda_batched_memory_gather(memories, topk_chunk, enabled=True, debug_fallback=False)
    torch.testing.assert_close(selected_chunk_live, selected_chunk_ref, atol=0.0, rtol=0.0)

    selected_flat_ref = cuda_batched_memory_gather(memories, topk_flat, enabled=True, debug_fallback=True)
    selected_flat_live = cuda_batched_memory_gather(memories, topk_flat, enabled=True, debug_fallback=False)
    torch.testing.assert_close(selected_flat_live, selected_flat_ref, atol=0.0, rtol=0.0)

    print("[PASS] Test 10 passed!\n")
    return True


def test_basic_forward():
    print("=" * 60)
    print("Test 1: Basic Forward Pass")
    print("=" * 60)

    config = MemoryArchitectureConfig.load_preset_config("minimal")
    model = SegmentRecurrentMemoryModel(
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        output_size=config.output_size,
        rnn_type=config.rnn_type,
        segment_length=config.segment_length,
        memory_capacity_segments=config.memory_capacity_segments,
        retrieval_top_k=config.retrieval_top_k,
        memory_key_dim=config.memory_key_dim,
        batch_first=True,
    )

    x = torch.randn(4, 200, config.input_size)

    with torch.no_grad():
        outputs, h_final, aux = unpack_model_output(model(x))

    print(f"Output shape: {outputs.shape}")
    print(f"Final hidden shape: {h_final.shape}")
    print(f"Number of cached segments: {model.get_num_segments()}")
    print(f"Memory info: {model.get_memory_info()}")
    print("[PASS] Test 1 passed!\n")
    return True


def test_gradient_flow():
    print("=" * 60)
    print("Test 2: Gradient Flow")
    print("=" * 60)

    config = MemoryArchitectureConfig.load_preset_config("minimal")
    model = SegmentRecurrentMemoryModel(
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        output_size=config.output_size,
        rnn_type="linear",
        segment_length=32,
        memory_capacity_segments=64,
        retrieval_top_k=4,
        memory_key_dim=config.memory_key_dim,
        batch_first=True,
    )

    x = torch.randn(2, 150, config.input_size)
    target = torch.randn(2, 150, config.output_size)
    outputs, _, _ = unpack_model_output(model(x))
    loss = torch.nn.functional.mse_loss(outputs, target)
    print(f"Loss: {loss.item():.4f}")
    loss.backward()

    grad_count = 0
    grad_norm_sq = 0.0
    for _, param in model.named_parameters():
        if param.grad is not None:
            grad_count += 1
            grad_norm_sq += param.grad.norm().item() ** 2

    print(f"Parameters with gradients: {grad_count}")
    print(f"Gradient norm: {grad_norm_sq ** 0.5:.4f}")
    print(f"Recompute rate: {model.memory_bank.get_recompute_rate():.2%}")
    print("[PASS] Test 2 passed!\n")
    return True


def test_backend_selection():
    print("=" * 60)
    print("Test 3: Backend Selection")
    print("=" * 60)

    x = torch.randn(2, 100, 64)
    model = SegmentRecurrentMemoryModel(
        input_size=64,
        hidden_size=128,
        output_size=64,
        rnn_type="linear",
        segment_length=32,
        memory_capacity_segments=32,
        retrieval_top_k=4,
        memory_key_dim=16,
        batch_first=True,
    )

    with torch.no_grad():
        outputs = model(x)

    print(f"LINEAR output shape: {outputs[0].shape}")

    for retired_backend in ["rnn", "gru", "lstm", "deep_linear"]:
        try:
            SegmentRecurrentMemoryModel(
                input_size=64,
                hidden_size=128,
                output_size=64,
                rnn_type=retired_backend,
                segment_length=32,
                memory_capacity_segments=32,
                retrieval_top_k=4,
                batch_first=True,
            )
        except ValueError:
            print(f"  retired backend {retired_backend!r} correctly rejected")
        else:
            raise AssertionError(f"Expected backend {retired_backend!r} to be rejected")

    for removed_runtime_arg, value in [
        ("use_torch_compile", True),
        ("torch_compile_mode", "reduce-overhead"),
        ("use_triton_kernels", True),
        ("use_cuda_cpp_kernels", False),
    ]:
        try:
            SegmentRecurrentMemoryModel(
                input_size=64,
                hidden_size=128,
                output_size=64,
                rnn_type="linear",
                segment_length=32,
                memory_capacity_segments=32,
                retrieval_top_k=4,
                batch_first=True,
                **{removed_runtime_arg: value},
            )
        except TypeError:
            print(f"  removed runtime arg {removed_runtime_arg!r} correctly rejected")
        else:
            raise AssertionError(f"Expected removed runtime arg {removed_runtime_arg!r} to be rejected")

    print("[PASS] Test 3 passed!\n")
    return True


def test_memory_estimation():
    print("=" * 60)
    print("Test 4: Memory Estimation")
    print("=" * 60)

    for preset in ["minimal", "standard", "long_sequence"]:
        config = MemoryArchitectureConfig.load_preset_config(preset)
        memory_info = estimate_memory_footprint(config, batch_size=32)
        print(f"{preset.upper()} total: {memory_info['total_gb']:.2f} GB")

    print("[PASS] Test 4 passed!\n")
    return True


def test_segment_processing():
    print("=" * 60)
    print("Test 5: Segment Processing")
    print("=" * 60)

    config = MemoryArchitectureConfig.load_preset_config("minimal")
    model = SegmentRecurrentMemoryModel(
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        output_size=config.output_size,
        rnn_type="linear",
        segment_length=32,
        memory_capacity_segments=64,
        retrieval_top_k=4,
        memory_key_dim=config.memory_key_dim,
        batch_first=True,
    )

    x = torch.randn(1, 128, config.input_size)
    model(x)
    actual_segments = model.get_num_segments()
    expected_segments = 4

    print(f"Expected cached segments: {expected_segments}")
    print(f"Actual cached segments: {actual_segments}")
    if actual_segments != expected_segments:
        raise AssertionError(f"Expected {expected_segments} segments, got {actual_segments}")

    print("[PASS] Test 5 passed!\n")
    return True


def test_checkpoint_mechanism():
    print("=" * 60)
    print("Test 6: Checkpoint Mechanism")
    print("=" * 60)

    config = MemoryArchitectureConfig.load_preset_config("minimal")
    model = SegmentRecurrentMemoryModel(
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        output_size=config.output_size,
        rnn_type="linear",
        segment_length=32,
        memory_capacity_segments=32,
        retrieval_top_k=4,
        memory_key_dim=config.memory_key_dim,
        batch_first=True,
    )

    x = torch.randn(2, 96, config.input_size)
    _, _, _, aux = model(x, return_hidden_states=True)

    if aux is None or "memory_info" not in aux:
        raise AssertionError("Expected hidden-state auxiliary information with memory stats")

    print(f"Aux keys: {sorted(aux.keys())}")
    print("[PASS] Test 6 passed!\n")
    return True


def main():
    tests = [
        ("Basic Forward", test_basic_forward),
        ("Gradient Flow", test_gradient_flow),
        ("Backend Selection", test_backend_selection),
        ("Memory Estimation", test_memory_estimation),
        ("Segment Processing", test_segment_processing),
        ("Checkpoint Mechanism", test_checkpoint_mechanism),
        ("CUDA/C++ Runtime Status", test_cuda_cpp_runtime_status),
        ("CUDA/C++ Chunk Update Entry Parity", test_cuda_chunk_update_parity),
        ("CUDA/C++ Apply Query Entry Parity", test_cuda_apply_query_parity),
        ("CUDA/C++ Batched Gather Entry Parity", test_cuda_batched_gather_parity),
    ]

    passed = 0
    for _, fn in tests:
        try:
            if fn():
                passed += 1
        except Exception as exc:
            print(f"[FAIL] {fn.__name__}: {exc}\n")

    print("=" * 60)
    print(f"Passed {passed}/{len(tests)} tests")
    print("=" * 60)
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    raise SystemExit(main())
