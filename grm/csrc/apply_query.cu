#include "grm_cuda.h"

#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_runtime.h>

namespace {

constexpr int kApplyQueryThreads = 256;

template <typename scalar_t>
__global__ void apply_query_rank4_forward_kernel(
    const scalar_t* __restrict__ memories,
    const scalar_t* __restrict__ queries,
    scalar_t* __restrict__ hidden,
    int batch_size,
    int routes,
    int hidden_size,
    int key_size) {
    const int64_t linear_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = static_cast<int64_t>(batch_size) * routes * hidden_size;
    if (linear_idx >= total) {
        return;
    }

    const int hidden_idx = static_cast<int>(linear_idx % hidden_size);
    const int route_idx = static_cast<int>((linear_idx / hidden_size) % routes);
    const int batch_idx = static_cast<int>(linear_idx / (static_cast<int64_t>(routes) * hidden_size));

    const scalar_t* memory_ptr =
        memories + (((static_cast<int64_t>(batch_idx) * routes + route_idx) * hidden_size + hidden_idx) * key_size);
    const scalar_t* query_ptr = queries + static_cast<int64_t>(batch_idx) * key_size;

    float sum = 0.0f;
    for (int key_idx = 0; key_idx < key_size; ++key_idx) {
        sum += static_cast<float>(memory_ptr[key_idx]) * static_cast<float>(query_ptr[key_idx]);
    }
    hidden[linear_idx] = static_cast<scalar_t>(sum);
}

template <typename scalar_t>
__global__ void apply_query_rank5_forward_kernel(
    const scalar_t* __restrict__ memories,
    const scalar_t* __restrict__ queries,
    scalar_t* __restrict__ hidden,
    int batch_size,
    int query_count,
    int routes,
    int hidden_size,
    int key_size) {
    const int64_t linear_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = static_cast<int64_t>(batch_size) * query_count * routes * hidden_size;
    if (linear_idx >= total) {
        return;
    }

    const int hidden_idx = static_cast<int>(linear_idx % hidden_size);
    const int route_idx = static_cast<int>((linear_idx / hidden_size) % routes);
    const int query_idx = static_cast<int>((linear_idx / (static_cast<int64_t>(hidden_size) * routes)) % query_count);
    const int batch_idx = static_cast<int>(linear_idx / (static_cast<int64_t>(query_count) * routes * hidden_size));

    const scalar_t* memory_ptr =
        memories +
        ((((static_cast<int64_t>(batch_idx) * query_count + query_idx) * routes + route_idx) * hidden_size + hidden_idx) * key_size);
    const scalar_t* query_ptr = queries + (static_cast<int64_t>(batch_idx) * query_count + query_idx) * key_size;

    float sum = 0.0f;
    for (int key_idx = 0; key_idx < key_size; ++key_idx) {
        sum += static_cast<float>(memory_ptr[key_idx]) * static_cast<float>(query_ptr[key_idx]);
    }
    hidden[linear_idx] = static_cast<scalar_t>(sum);
}

template <typename scalar_t>
__global__ void apply_query_rank4_backward_kernel(
    const scalar_t* __restrict__ grad_hidden,
    const scalar_t* __restrict__ memories,
    const scalar_t* __restrict__ queries,
    scalar_t* __restrict__ grad_memories,
    scalar_t* __restrict__ grad_queries,
    int batch_size,
    int routes,
    int hidden_size,
    int key_size) {
    const int64_t memory_linear_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t memory_total = static_cast<int64_t>(batch_size) * routes * hidden_size * key_size;
    if (memory_linear_idx < memory_total) {
        const int key_idx = static_cast<int>(memory_linear_idx % key_size);
        const int hidden_idx = static_cast<int>((memory_linear_idx / key_size) % hidden_size);
        const int route_idx = static_cast<int>((memory_linear_idx / (static_cast<int64_t>(key_size) * hidden_size)) % routes);
        const int batch_idx = static_cast<int>(memory_linear_idx / (static_cast<int64_t>(routes) * hidden_size * key_size));
        const int64_t hidden_linear_idx =
            (static_cast<int64_t>(batch_idx) * routes + route_idx) * hidden_size + hidden_idx;
        const int64_t query_linear_idx = static_cast<int64_t>(batch_idx) * key_size + key_idx;
        grad_memories[memory_linear_idx] = static_cast<scalar_t>(
            static_cast<float>(grad_hidden[hidden_linear_idx]) * static_cast<float>(queries[query_linear_idx]));
    }

    const int64_t query_linear_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t query_total = static_cast<int64_t>(batch_size) * key_size;
    if (query_linear_idx < query_total) {
        const int key_idx = static_cast<int>(query_linear_idx % key_size);
        const int batch_idx = static_cast<int>(query_linear_idx / key_size);
        float sum = 0.0f;
        for (int route_idx = 0; route_idx < routes; ++route_idx) {
            for (int hidden_idx = 0; hidden_idx < hidden_size; ++hidden_idx) {
                const int64_t hidden_linear_idx =
                    (static_cast<int64_t>(batch_idx) * routes + route_idx) * hidden_size + hidden_idx;
                const int64_t memory_linear_idx =
                    (((static_cast<int64_t>(batch_idx) * routes + route_idx) * hidden_size + hidden_idx) * key_size) + key_idx;
                sum += static_cast<float>(grad_hidden[hidden_linear_idx]) * static_cast<float>(memories[memory_linear_idx]);
            }
        }
        grad_queries[query_linear_idx] = static_cast<scalar_t>(sum);
    }
}

template <typename scalar_t>
__global__ void apply_query_rank5_backward_kernel(
    const scalar_t* __restrict__ grad_hidden,
    const scalar_t* __restrict__ memories,
    const scalar_t* __restrict__ queries,
    scalar_t* __restrict__ grad_memories,
    scalar_t* __restrict__ grad_queries,
    int batch_size,
    int query_count,
    int routes,
    int hidden_size,
    int key_size) {
    const int64_t memory_linear_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t memory_total = static_cast<int64_t>(batch_size) * query_count * routes * hidden_size * key_size;
    if (memory_linear_idx < memory_total) {
        const int key_idx = static_cast<int>(memory_linear_idx % key_size);
        const int hidden_idx = static_cast<int>((memory_linear_idx / key_size) % hidden_size);
        const int route_idx = static_cast<int>((memory_linear_idx / (static_cast<int64_t>(key_size) * hidden_size)) % routes);
        const int query_idx = static_cast<int>(
            (memory_linear_idx / (static_cast<int64_t>(key_size) * hidden_size * routes)) % query_count);
        const int batch_idx = static_cast<int>(
            memory_linear_idx / (static_cast<int64_t>(query_count) * routes * hidden_size * key_size));
        const int64_t hidden_linear_idx =
            (((static_cast<int64_t>(batch_idx) * query_count + query_idx) * routes + route_idx) * hidden_size) + hidden_idx;
        const int64_t query_linear_idx =
            (static_cast<int64_t>(batch_idx) * query_count + query_idx) * key_size + key_idx;
        grad_memories[memory_linear_idx] = static_cast<scalar_t>(
            static_cast<float>(grad_hidden[hidden_linear_idx]) * static_cast<float>(queries[query_linear_idx]));
    }

    const int64_t query_linear_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t query_total = static_cast<int64_t>(batch_size) * query_count * key_size;
    if (query_linear_idx < query_total) {
        const int key_idx = static_cast<int>(query_linear_idx % key_size);
        const int query_idx = static_cast<int>((query_linear_idx / key_size) % query_count);
        const int batch_idx = static_cast<int>(query_linear_idx / (static_cast<int64_t>(query_count) * key_size));
        float sum = 0.0f;
        for (int route_idx = 0; route_idx < routes; ++route_idx) {
            for (int hidden_idx = 0; hidden_idx < hidden_size; ++hidden_idx) {
                const int64_t hidden_linear_idx =
                    (((static_cast<int64_t>(batch_idx) * query_count + query_idx) * routes + route_idx) * hidden_size) + hidden_idx;
                const int64_t memory_linear_idx =
                    ((((static_cast<int64_t>(batch_idx) * query_count + query_idx) * routes + route_idx) * hidden_size + hidden_idx) * key_size) + key_idx;
                sum += static_cast<float>(grad_hidden[hidden_linear_idx]) * static_cast<float>(memories[memory_linear_idx]);
            }
        }
        grad_queries[query_linear_idx] = static_cast<scalar_t>(sum);
    }
}

bool supports_apply_query_cuda_kernel(const at::Tensor& memories, const at::Tensor& queries) {
    if (!memories.is_cuda() || !queries.is_cuda()) {
        return false;
    }
    if (memories.scalar_type() != queries.scalar_type()) {
        return false;
    }
    const auto dtype = memories.scalar_type();
    return dtype == at::kFloat || dtype == at::kHalf || dtype == at::kBFloat16;
}

}  // namespace

namespace grm {

at::Tensor apply_query_forward_cuda(const at::Tensor& memories, const at::Tensor& queries) {
    if (!supports_apply_query_cuda_kernel(memories, queries)) {
        return apply_query_forward(memories, queries);
    }

    auto memories_contig = memories.contiguous();
    auto queries_contig = queries.contiguous();
    c10::cuda::CUDAGuard device_guard(memories_contig.device());

    if (memories_contig.dim() == 4) {
        auto hidden = at::empty(
            {memories_contig.size(0), memories_contig.size(1), memories_contig.size(2)},
            memories_contig.options());
        const int64_t total = hidden.numel();
        const dim3 grid(static_cast<unsigned int>((total + kApplyQueryThreads - 1) / kApplyQueryThreads));

        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            memories_contig.scalar_type(),
            "apply_query_rank4_forward_cuda",
            [&] {
                apply_query_rank4_forward_kernel<scalar_t><<<
                    grid,
                    kApplyQueryThreads,
                    0,
                    at::cuda::getCurrentCUDAStream()>>>(
                    memories_contig.data_ptr<scalar_t>(),
                    queries_contig.data_ptr<scalar_t>(),
                    hidden.data_ptr<scalar_t>(),
                    static_cast<int>(memories_contig.size(0)),
                    static_cast<int>(memories_contig.size(1)),
                    static_cast<int>(memories_contig.size(2)),
                    static_cast<int>(memories_contig.size(3)));
            });
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return hidden;
    }

    auto hidden = at::empty(
        {memories_contig.size(0), memories_contig.size(1), memories_contig.size(2), memories_contig.size(3)},
        memories_contig.options());
    const int64_t total = hidden.numel();
    const dim3 grid(static_cast<unsigned int>((total + kApplyQueryThreads - 1) / kApplyQueryThreads));

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        memories_contig.scalar_type(),
        "apply_query_rank5_forward_cuda",
        [&] {
            apply_query_rank5_forward_kernel<scalar_t><<<
                grid,
                kApplyQueryThreads,
                0,
                at::cuda::getCurrentCUDAStream()>>>(
                memories_contig.data_ptr<scalar_t>(),
                queries_contig.data_ptr<scalar_t>(),
                hidden.data_ptr<scalar_t>(),
                static_cast<int>(memories_contig.size(0)),
                static_cast<int>(memories_contig.size(1)),
                static_cast<int>(memories_contig.size(2)),
                static_cast<int>(memories_contig.size(3)),
                static_cast<int>(memories_contig.size(4)));
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return hidden;
}

std::tuple<at::Tensor, at::Tensor> apply_query_backward_cuda(
    const at::Tensor& grad_hidden,
    const at::Tensor& memories,
    const at::Tensor& queries) {
    if (!supports_apply_query_cuda_kernel(memories, queries) ||
        !grad_hidden.is_cuda() ||
        grad_hidden.scalar_type() != memories.scalar_type()) {
        return apply_query_backward(grad_hidden, memories, queries);
    }

    auto grad_hidden_contig = grad_hidden.contiguous();
    auto memories_contig = memories.contiguous();
    auto queries_contig = queries.contiguous();
    auto grad_memories = at::empty_like(memories_contig);
    auto grad_queries = at::empty_like(queries_contig);
    c10::cuda::CUDAGuard device_guard(memories_contig.device());

    if (memories_contig.dim() == 4) {
        const int64_t total = std::max<int64_t>(grad_memories.numel(), grad_queries.numel());
        const dim3 grid(static_cast<unsigned int>((total + kApplyQueryThreads - 1) / kApplyQueryThreads));

        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            memories_contig.scalar_type(),
            "apply_query_rank4_backward_cuda",
            [&] {
                apply_query_rank4_backward_kernel<scalar_t><<<
                    grid,
                    kApplyQueryThreads,
                    0,
                    at::cuda::getCurrentCUDAStream()>>>(
                    grad_hidden_contig.data_ptr<scalar_t>(),
                    memories_contig.data_ptr<scalar_t>(),
                    queries_contig.data_ptr<scalar_t>(),
                    grad_memories.data_ptr<scalar_t>(),
                    grad_queries.data_ptr<scalar_t>(),
                    static_cast<int>(memories_contig.size(0)),
                    static_cast<int>(memories_contig.size(1)),
                    static_cast<int>(memories_contig.size(2)),
                    static_cast<int>(memories_contig.size(3)));
            });
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return {grad_memories, grad_queries};
    }

    const int64_t total = std::max<int64_t>(grad_memories.numel(), grad_queries.numel());
    const dim3 grid(static_cast<unsigned int>((total + kApplyQueryThreads - 1) / kApplyQueryThreads));

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        memories_contig.scalar_type(),
        "apply_query_rank5_backward_cuda",
        [&] {
            apply_query_rank5_backward_kernel<scalar_t><<<
                grid,
                kApplyQueryThreads,
                0,
                at::cuda::getCurrentCUDAStream()>>>(
                grad_hidden_contig.data_ptr<scalar_t>(),
                memories_contig.data_ptr<scalar_t>(),
                queries_contig.data_ptr<scalar_t>(),
                grad_memories.data_ptr<scalar_t>(),
                grad_queries.data_ptr<scalar_t>(),
                static_cast<int>(memories_contig.size(0)),
                static_cast<int>(memories_contig.size(1)),
                static_cast<int>(memories_contig.size(2)),
                static_cast<int>(memories_contig.size(3)),
                static_cast<int>(memories_contig.size(4)));
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {grad_memories, grad_queries};
}

}  // namespace grm
