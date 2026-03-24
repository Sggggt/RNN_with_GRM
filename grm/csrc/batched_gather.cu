#include "grm_cuda.h"

#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_runtime.h>

namespace {

constexpr int kGatherThreads = 256;

template <typename scalar_t>
__global__ void batched_memory_gather_rank2_kernel(
    const scalar_t* __restrict__ memories,
    const int64_t* __restrict__ indices,
    scalar_t* __restrict__ output,
    int batch_size,
    int segments,
    int routes,
    int hidden_size,
    int key_size) {
    const int64_t linear_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = static_cast<int64_t>(batch_size) * routes * hidden_size * key_size;
    if (linear_idx >= total) {
        return;
    }

    const int64_t per_batch = static_cast<int64_t>(routes) * hidden_size * key_size;
    const int batch_idx = static_cast<int>(linear_idx / per_batch);
    const int64_t batch_offset = linear_idx % per_batch;
    const int route_idx = static_cast<int>(batch_offset / (static_cast<int64_t>(hidden_size) * key_size));
    const int inner = static_cast<int>(batch_offset % (static_cast<int64_t>(hidden_size) * key_size));
    const int hidden_idx = inner / key_size;
    const int key_idx = inner % key_size;

    const int64_t source_segment = indices[static_cast<int64_t>(batch_idx) * routes + route_idx];
    const int64_t source_offset =
        (((static_cast<int64_t>(batch_idx) * segments + source_segment) * hidden_size + hidden_idx) * key_size) + key_idx;
    output[linear_idx] = memories[source_offset];
}

template <typename scalar_t>
__global__ void batched_memory_gather_rank3_kernel(
    const scalar_t* __restrict__ memories,
    const int64_t* __restrict__ indices,
    scalar_t* __restrict__ output,
    int batch_size,
    int segments,
    int queries,
    int routes,
    int hidden_size,
    int key_size) {
    const int64_t linear_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = static_cast<int64_t>(batch_size) * queries * routes * hidden_size * key_size;
    if (linear_idx >= total) {
        return;
    }

    const int64_t per_batch = static_cast<int64_t>(queries) * routes * hidden_size * key_size;
    const int batch_idx = static_cast<int>(linear_idx / per_batch);
    const int64_t batch_offset = linear_idx % per_batch;
    const int query_idx = static_cast<int>(batch_offset / (static_cast<int64_t>(routes) * hidden_size * key_size));
    const int64_t query_offset = batch_offset % (static_cast<int64_t>(routes) * hidden_size * key_size);
    const int route_idx = static_cast<int>(query_offset / (static_cast<int64_t>(hidden_size) * key_size));
    const int inner = static_cast<int>(query_offset % (static_cast<int64_t>(hidden_size) * key_size));
    const int hidden_idx = inner / key_size;
    const int key_idx = inner % key_size;

    const int64_t index_offset =
        (static_cast<int64_t>(batch_idx) * queries + query_idx) * routes + route_idx;
    const int64_t source_segment = indices[index_offset];
    const int64_t source_offset =
        (((static_cast<int64_t>(batch_idx) * segments + source_segment) * hidden_size + hidden_idx) * key_size) + key_idx;
    output[linear_idx] = memories[source_offset];
}

bool supports_batched_gather_cuda_kernel(const at::Tensor& memories_per_batch, const at::Tensor& topk_indices) {
    if (!memories_per_batch.is_cuda() || !topk_indices.is_cuda()) {
        return false;
    }
    const auto dtype = memories_per_batch.scalar_type();
    return dtype == at::kFloat || dtype == at::kHalf || dtype == at::kBFloat16;
}

}  // namespace

namespace grm {

at::Tensor batched_memory_gather_cuda(const at::Tensor& memories_per_batch, const at::Tensor& topk_indices) {
    if (!supports_batched_gather_cuda_kernel(memories_per_batch, topk_indices)) {
        return batched_memory_gather(memories_per_batch, topk_indices);
    }

    TORCH_CHECK(memories_per_batch.dim() == 4, "memories_per_batch must have shape [B, S, H, K]");
    TORCH_CHECK(topk_indices.dim() == 2 || topk_indices.dim() == 3, "topk_indices must have shape [B, R] or [B, Q, R]");
    TORCH_CHECK(memories_per_batch.size(0) == topk_indices.size(0), "batch dim mismatch between memories_per_batch and topk_indices");

    auto memories_contig = memories_per_batch.contiguous();
    auto indices_contig = topk_indices.toType(at::kLong).contiguous();

    const int batch_size = static_cast<int>(memories_contig.size(0));
    const int segments = static_cast<int>(memories_contig.size(1));
    const int hidden_size = static_cast<int>(memories_contig.size(2));
    const int key_size = static_cast<int>(memories_contig.size(3));

    at::Tensor output;
    c10::cuda::CUDAGuard device_guard(memories_contig.device());

    if (indices_contig.dim() == 2) {
        const int routes = static_cast<int>(indices_contig.size(1));
        output = at::empty(
            {memories_contig.size(0), indices_contig.size(1), memories_contig.size(2), memories_contig.size(3)},
            memories_contig.options());
        const int64_t total = output.numel();
        const dim3 grid(static_cast<unsigned int>((total + kGatherThreads - 1) / kGatherThreads));

        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            memories_contig.scalar_type(),
            "batched_memory_gather_rank2_cuda",
            [&] {
                batched_memory_gather_rank2_kernel<scalar_t><<<
                    grid,
                    kGatherThreads,
                    0,
                    at::cuda::getCurrentCUDAStream()>>>(
                    memories_contig.data_ptr<scalar_t>(),
                    indices_contig.data_ptr<int64_t>(),
                    output.data_ptr<scalar_t>(),
                    batch_size,
                    segments,
                    routes,
                    hidden_size,
                    key_size);
            });
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return output;
    }

    const int queries = static_cast<int>(indices_contig.size(1));
    const int routes = static_cast<int>(indices_contig.size(2));
    output = at::empty(
        {
            memories_contig.size(0),
            indices_contig.size(1),
            indices_contig.size(2),
            memories_contig.size(2),
            memories_contig.size(3),
        },
        memories_contig.options());
    const int64_t total = output.numel();
    const dim3 grid(static_cast<unsigned int>((total + kGatherThreads - 1) / kGatherThreads));

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        memories_contig.scalar_type(),
        "batched_memory_gather_rank3_cuda",
        [&] {
            batched_memory_gather_rank3_kernel<scalar_t><<<
                grid,
                kGatherThreads,
                0,
                at::cuda::getCurrentCUDAStream()>>>(
                memories_contig.data_ptr<scalar_t>(),
                indices_contig.data_ptr<int64_t>(),
                output.data_ptr<scalar_t>(),
                batch_size,
                segments,
                queries,
                routes,
                hidden_size,
                key_size);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}

}  // namespace grm
