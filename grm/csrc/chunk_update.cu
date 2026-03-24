#include "grm_cuda.h"

#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>

namespace {

constexpr int kChunkWarpSize = 32;
constexpr int kChunkRowsPerBlock = 8;
constexpr int kChunkMaxKeyRegs = 4;

__inline__ __device__ float warp_reduce_sum(float value) {
    for (int offset = kChunkWarpSize / 2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__device__ __forceinline__ int padded_key_index(int key_idx) {
    return key_idx + key_idx / kChunkWarpSize;
}

template <typename scalar_t, int KeyRegs>
__global__ void chunk_update_forward_kernel(
    const scalar_t* __restrict__ keys,
    const scalar_t* __restrict__ values,
    const scalar_t* __restrict__ queries,
    const scalar_t* __restrict__ memory,
    scalar_t* __restrict__ hidden_out,
    scalar_t* __restrict__ memory_out,
    float* __restrict__ memory_steps,
    int chunk_len,
    int hidden_size,
    int key_size,
    float decay,
    float beta,
    float inv_scale) {
    const int batch_idx = blockIdx.x;
    const int tile_idx = blockIdx.y;
    const int lane = threadIdx.x;
    const int row_in_block = threadIdx.y;
    const int thread_linear = row_in_block * kChunkWarpSize + lane;
    const int hidden_idx = tile_idx * kChunkRowsPerBlock + row_in_block;
    const bool valid_row = hidden_idx < hidden_size;
    const int padded_key_size = key_size + key_size / kChunkWarpSize + 1;

    extern __shared__ float shared_storage[];
    float* shared_keys = shared_storage;
    float* shared_queries = shared_storage + padded_key_size;
    float* shared_values = shared_queries + padded_key_size;

    const int64_t batch_memory_offset = static_cast<int64_t>(batch_idx) * hidden_size * key_size;
    const scalar_t* batch_keys = keys + static_cast<int64_t>(batch_idx) * chunk_len * key_size;
    const scalar_t* batch_values = values + static_cast<int64_t>(batch_idx) * chunk_len * hidden_size;
    const scalar_t* batch_queries = queries + static_cast<int64_t>(batch_idx) * chunk_len * key_size;
    const scalar_t* batch_memory = memory + batch_memory_offset;
    scalar_t* batch_hidden_out = hidden_out + static_cast<int64_t>(batch_idx) * chunk_len * hidden_size;
    scalar_t* batch_memory_out = memory_out + batch_memory_offset;
    float* batch_memory_steps = memory_steps == nullptr
        ? nullptr
        : memory_steps + static_cast<int64_t>(batch_idx) * chunk_len * hidden_size * key_size;

    float memory_reg[KeyRegs] = {0.0f};
    if (valid_row) {
#pragma unroll
        for (int seg = 0; seg < KeyRegs; ++seg) {
            const int key_idx = seg * kChunkWarpSize + lane;
            if (key_idx < key_size) {
                memory_reg[seg] = static_cast<float>(
                    batch_memory[static_cast<int64_t>(hidden_idx) * key_size + key_idx]);
            }
        }
    }

    for (int t = 0; t < chunk_len; ++t) {
        const scalar_t* key_t = batch_keys + static_cast<int64_t>(t) * key_size;
        const scalar_t* value_t = batch_values + static_cast<int64_t>(t) * hidden_size;
        const scalar_t* query_t = batch_queries + static_cast<int64_t>(t) * key_size;
        scalar_t* hidden_t = batch_hidden_out + static_cast<int64_t>(t) * hidden_size;

        if (thread_linear < key_size) {
            const int padded_idx = padded_key_index(thread_linear);
            shared_keys[padded_idx] = static_cast<float>(key_t[thread_linear]);
            shared_queries[padded_idx] = static_cast<float>(query_t[thread_linear]);
        }
        if (lane == 0) {
            shared_values[row_in_block] = valid_row ? static_cast<float>(value_t[hidden_idx]) : 0.0f;
        }
        __syncthreads();

        const float value_scalar = shared_values[row_in_block];
        float hidden_partial = 0.0f;

        if (valid_row) {
#pragma unroll
            for (int seg = 0; seg < KeyRegs; ++seg) {
                const int key_idx = seg * kChunkWarpSize + lane;
                if (key_idx < key_size) {
                    const int padded_idx = padded_key_index(key_idx);
                    memory_reg[seg] =
                        memory_reg[seg] * decay + beta * inv_scale * value_scalar * shared_keys[padded_idx];
                    hidden_partial += memory_reg[seg] * shared_queries[padded_idx];
                    if (batch_memory_steps != nullptr) {
                        batch_memory_steps[
                            (static_cast<int64_t>(t) * hidden_size + hidden_idx) * key_size + key_idx] = memory_reg[seg];
                    }
                }
            }
        }

        hidden_partial = warp_reduce_sum(hidden_partial);
        if (lane == 0 && valid_row) {
            hidden_t[hidden_idx] = static_cast<scalar_t>(hidden_partial);
        }
        __syncthreads();
    }

    if (valid_row) {
#pragma unroll
        for (int seg = 0; seg < KeyRegs; ++seg) {
            const int key_idx = seg * kChunkWarpSize + lane;
            if (key_idx < key_size) {
                batch_memory_out[static_cast<int64_t>(hidden_idx) * key_size + key_idx] =
                    static_cast<scalar_t>(memory_reg[seg]);
            }
        }
    }
}

template <typename scalar_t, int KeyRegs>
__global__ void chunk_update_backward_kernel(
    const scalar_t* __restrict__ grad_hidden,
    const scalar_t* __restrict__ grad_memory_out,
    const scalar_t* __restrict__ keys,
    const scalar_t* __restrict__ values,
    const scalar_t* __restrict__ queries,
    const float* __restrict__ memory_steps,
    float* __restrict__ partial_grad_keys,
    scalar_t* __restrict__ grad_values,
    float* __restrict__ partial_grad_queries,
    scalar_t* __restrict__ grad_memory,
    int chunk_len,
    int hidden_size,
    int key_size,
    int hidden_tiles,
    float decay,
    float beta,
    float inv_scale) {
    const int batch_idx = blockIdx.x;
    const int tile_idx = blockIdx.y;
    const int lane = threadIdx.x;
    const int row_in_block = threadIdx.y;
    const int thread_linear = row_in_block * kChunkWarpSize + lane;
    const int hidden_idx = tile_idx * kChunkRowsPerBlock + row_in_block;
    const bool valid_row = hidden_idx < hidden_size;
    const int padded_key_size = key_size + key_size / kChunkWarpSize + 1;

    extern __shared__ float shared_storage[];
    float* shared_keys = shared_storage;
    float* shared_queries = shared_storage + padded_key_size;
    float* shared_grad_hidden = shared_queries + padded_key_size;
    float* shared_values = shared_grad_hidden + kChunkRowsPerBlock;
    float* shared_partial_grad_keys = shared_values + kChunkRowsPerBlock;
    float* shared_partial_grad_queries =
        shared_partial_grad_keys + kChunkRowsPerBlock * padded_key_size;

    const int64_t batch_memory_offset = static_cast<int64_t>(batch_idx) * hidden_size * key_size;
    const scalar_t* batch_grad_hidden = grad_hidden + static_cast<int64_t>(batch_idx) * chunk_len * hidden_size;
    const scalar_t* batch_grad_memory_out = grad_memory_out + batch_memory_offset;
    const scalar_t* batch_keys = keys + static_cast<int64_t>(batch_idx) * chunk_len * key_size;
    const scalar_t* batch_values = values + static_cast<int64_t>(batch_idx) * chunk_len * hidden_size;
    const scalar_t* batch_queries = queries + static_cast<int64_t>(batch_idx) * chunk_len * key_size;
    const float* batch_memory_steps =
        memory_steps + static_cast<int64_t>(batch_idx) * chunk_len * hidden_size * key_size;
    scalar_t* batch_grad_values = grad_values + static_cast<int64_t>(batch_idx) * chunk_len * hidden_size;
    scalar_t* batch_grad_memory = grad_memory + batch_memory_offset;

    float grad_memory_reg[KeyRegs] = {0.0f};
    if (valid_row) {
#pragma unroll
        for (int seg = 0; seg < KeyRegs; ++seg) {
            const int key_idx = seg * kChunkWarpSize + lane;
            if (key_idx < key_size) {
                grad_memory_reg[seg] = static_cast<float>(
                    batch_grad_memory_out[static_cast<int64_t>(hidden_idx) * key_size + key_idx]);
            }
        }
    }

    for (int t = chunk_len - 1; t >= 0; --t) {
        const scalar_t* grad_hidden_t = batch_grad_hidden + static_cast<int64_t>(t) * hidden_size;
        const scalar_t* key_t = batch_keys + static_cast<int64_t>(t) * key_size;
        const scalar_t* value_t = batch_values + static_cast<int64_t>(t) * hidden_size;
        const scalar_t* query_t = batch_queries + static_cast<int64_t>(t) * key_size;
        scalar_t* grad_value_t = batch_grad_values + static_cast<int64_t>(t) * hidden_size;

        if (thread_linear < key_size) {
            const int padded_idx = padded_key_index(thread_linear);
            shared_keys[padded_idx] = static_cast<float>(key_t[thread_linear]);
            shared_queries[padded_idx] = static_cast<float>(query_t[thread_linear]);
        }
        if (lane == 0) {
            shared_grad_hidden[row_in_block] = valid_row ? static_cast<float>(grad_hidden_t[hidden_idx]) : 0.0f;
            shared_values[row_in_block] = valid_row ? static_cast<float>(value_t[hidden_idx]) : 0.0f;
        }
        __syncthreads();

        const float grad_hidden_scalar = shared_grad_hidden[row_in_block];
        const float value_scalar = shared_values[row_in_block];
        float* row_partial_grad_keys = shared_partial_grad_keys + row_in_block * padded_key_size;
        float* row_partial_grad_queries = shared_partial_grad_queries + row_in_block * padded_key_size;
        float grad_value_partial = 0.0f;

#pragma unroll
        for (int seg = 0; seg < KeyRegs; ++seg) {
            const int key_idx = seg * kChunkWarpSize + lane;
            if (key_idx < key_size) {
                const int padded_idx = padded_key_index(key_idx);
                float grad_query_partial = 0.0f;
                float grad_key_partial = 0.0f;
                if (valid_row) {
                    const float memory_step_value =
                        batch_memory_steps[(static_cast<int64_t>(t) * hidden_size + hidden_idx) * key_size + key_idx];
                    grad_query_partial = grad_hidden_scalar * memory_step_value;
                    grad_memory_reg[seg] += grad_hidden_scalar * shared_queries[padded_idx];
                    grad_value_partial += grad_memory_reg[seg] * shared_keys[padded_idx];
                    grad_key_partial = beta * inv_scale * grad_memory_reg[seg] * value_scalar;
                    grad_memory_reg[seg] *= decay;
                }
                row_partial_grad_queries[padded_idx] = grad_query_partial;
                row_partial_grad_keys[padded_idx] = grad_key_partial;
            }
        }

        grad_value_partial = warp_reduce_sum(grad_value_partial);
        if (lane == 0 && valid_row) {
            grad_value_t[hidden_idx] = static_cast<scalar_t>(beta * inv_scale * grad_value_partial);
        }
        __syncthreads();

        float* partial_grad_key_t =
            partial_grad_keys +
            (((static_cast<int64_t>(batch_idx) * hidden_tiles + tile_idx) * chunk_len + t) * key_size);
        float* partial_grad_query_t =
            partial_grad_queries +
            (((static_cast<int64_t>(batch_idx) * hidden_tiles + tile_idx) * chunk_len + t) * key_size);
        if (thread_linear < key_size) {
            const int padded_idx = padded_key_index(thread_linear);
            float grad_key_sum = 0.0f;
            float grad_query_sum = 0.0f;
            for (int row = 0; row < kChunkRowsPerBlock; ++row) {
                grad_key_sum += shared_partial_grad_keys[row * padded_key_size + padded_idx];
                grad_query_sum += shared_partial_grad_queries[row * padded_key_size + padded_idx];
            }
            partial_grad_key_t[thread_linear] = grad_key_sum;
            partial_grad_query_t[thread_linear] = grad_query_sum;
        }
        __syncthreads();
    }

    if (valid_row) {
#pragma unroll
        for (int seg = 0; seg < KeyRegs; ++seg) {
            const int key_idx = seg * kChunkWarpSize + lane;
            if (key_idx < key_size) {
                batch_grad_memory[static_cast<int64_t>(hidden_idx) * key_size + key_idx] =
                    static_cast<scalar_t>(grad_memory_reg[seg]);
            }
        }
    }
}

template <typename scalar_t>
void launch_chunk_update_forward_kernel(
    const at::Tensor& keys,
    const at::Tensor& values,
    const at::Tensor& queries,
    const at::Tensor& memory,
    at::Tensor& hidden,
    at::Tensor& memory_out,
    at::Tensor* memory_steps,
    int batch_size,
    int hidden_tiles,
    int chunk_len,
    int hidden_size,
    int key_size,
    float decay,
    float beta,
    float inv_scale) {
    const int key_segments = (key_size + kChunkWarpSize - 1) / kChunkWarpSize;
    const int padded_key_size = key_size + key_size / kChunkWarpSize + 1;
    const size_t shared_bytes =
        static_cast<size_t>(2 * padded_key_size + kChunkRowsPerBlock) * sizeof(float);
    const dim3 block(kChunkWarpSize, kChunkRowsPerBlock);
    const dim3 grid(batch_size, hidden_tiles);
    float* memory_steps_ptr = memory_steps == nullptr ? nullptr : memory_steps->data_ptr<float>();

    switch (key_segments) {
        case 1:
            chunk_update_forward_kernel<scalar_t, 1><<<grid, block, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
                keys.data_ptr<scalar_t>(),
                values.data_ptr<scalar_t>(),
                queries.data_ptr<scalar_t>(),
                memory.data_ptr<scalar_t>(),
                hidden.data_ptr<scalar_t>(),
                memory_out.data_ptr<scalar_t>(),
                memory_steps_ptr,
                chunk_len,
                hidden_size,
                key_size,
                decay,
                beta,
                inv_scale);
            return;
        case 2:
            chunk_update_forward_kernel<scalar_t, 2><<<grid, block, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
                keys.data_ptr<scalar_t>(),
                values.data_ptr<scalar_t>(),
                queries.data_ptr<scalar_t>(),
                memory.data_ptr<scalar_t>(),
                hidden.data_ptr<scalar_t>(),
                memory_out.data_ptr<scalar_t>(),
                memory_steps_ptr,
                chunk_len,
                hidden_size,
                key_size,
                decay,
                beta,
                inv_scale);
            return;
        case 3:
            chunk_update_forward_kernel<scalar_t, 3><<<grid, block, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
                keys.data_ptr<scalar_t>(),
                values.data_ptr<scalar_t>(),
                queries.data_ptr<scalar_t>(),
                memory.data_ptr<scalar_t>(),
                hidden.data_ptr<scalar_t>(),
                memory_out.data_ptr<scalar_t>(),
                memory_steps_ptr,
                chunk_len,
                hidden_size,
                key_size,
                decay,
                beta,
                inv_scale);
            return;
        case 4:
            chunk_update_forward_kernel<scalar_t, 4><<<grid, block, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
                keys.data_ptr<scalar_t>(),
                values.data_ptr<scalar_t>(),
                queries.data_ptr<scalar_t>(),
                memory.data_ptr<scalar_t>(),
                hidden.data_ptr<scalar_t>(),
                memory_out.data_ptr<scalar_t>(),
                memory_steps_ptr,
                chunk_len,
                hidden_size,
                key_size,
                decay,
                beta,
                inv_scale);
            return;
        default:
            TORCH_CHECK(false, "Unsupported key_size for chunk_update_forward_cuda: ", key_size);
    }
}

template <typename scalar_t>
void launch_chunk_update_backward_kernel(
    const at::Tensor& grad_hidden,
    const at::Tensor& grad_memory_out,
    const at::Tensor& keys,
    const at::Tensor& values,
    const at::Tensor& queries,
    const at::Tensor& memory_steps,
    at::Tensor& partial_grad_keys,
    at::Tensor& grad_values,
    at::Tensor& partial_grad_queries,
    at::Tensor& grad_memory,
    int batch_size,
    int hidden_tiles,
    int chunk_len,
    int hidden_size,
    int key_size,
    float decay,
    float beta,
    float inv_scale) {
    const int key_segments = (key_size + kChunkWarpSize - 1) / kChunkWarpSize;
    const int padded_key_size = key_size + key_size / kChunkWarpSize + 1;
    const size_t shared_bytes = static_cast<size_t>(
        2 * padded_key_size +
        2 * kChunkRowsPerBlock +
        2 * kChunkRowsPerBlock * padded_key_size) * sizeof(float);
    const dim3 block(kChunkWarpSize, kChunkRowsPerBlock);
    const dim3 grid(batch_size, hidden_tiles);

    switch (key_segments) {
        case 1:
            chunk_update_backward_kernel<scalar_t, 1><<<grid, block, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
                grad_hidden.data_ptr<scalar_t>(),
                grad_memory_out.data_ptr<scalar_t>(),
                keys.data_ptr<scalar_t>(),
                values.data_ptr<scalar_t>(),
                queries.data_ptr<scalar_t>(),
                memory_steps.data_ptr<float>(),
                partial_grad_keys.data_ptr<float>(),
                grad_values.data_ptr<scalar_t>(),
                partial_grad_queries.data_ptr<float>(),
                grad_memory.data_ptr<scalar_t>(),
                chunk_len,
                hidden_size,
                key_size,
                hidden_tiles,
                decay,
                beta,
                inv_scale);
            return;
        case 2:
            chunk_update_backward_kernel<scalar_t, 2><<<grid, block, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
                grad_hidden.data_ptr<scalar_t>(),
                grad_memory_out.data_ptr<scalar_t>(),
                keys.data_ptr<scalar_t>(),
                values.data_ptr<scalar_t>(),
                queries.data_ptr<scalar_t>(),
                memory_steps.data_ptr<float>(),
                partial_grad_keys.data_ptr<float>(),
                grad_values.data_ptr<scalar_t>(),
                partial_grad_queries.data_ptr<float>(),
                grad_memory.data_ptr<scalar_t>(),
                chunk_len,
                hidden_size,
                key_size,
                hidden_tiles,
                decay,
                beta,
                inv_scale);
            return;
        case 3:
            chunk_update_backward_kernel<scalar_t, 3><<<grid, block, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
                grad_hidden.data_ptr<scalar_t>(),
                grad_memory_out.data_ptr<scalar_t>(),
                keys.data_ptr<scalar_t>(),
                values.data_ptr<scalar_t>(),
                queries.data_ptr<scalar_t>(),
                memory_steps.data_ptr<float>(),
                partial_grad_keys.data_ptr<float>(),
                grad_values.data_ptr<scalar_t>(),
                partial_grad_queries.data_ptr<float>(),
                grad_memory.data_ptr<scalar_t>(),
                chunk_len,
                hidden_size,
                key_size,
                hidden_tiles,
                decay,
                beta,
                inv_scale);
            return;
        case 4:
            chunk_update_backward_kernel<scalar_t, 4><<<grid, block, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
                grad_hidden.data_ptr<scalar_t>(),
                grad_memory_out.data_ptr<scalar_t>(),
                keys.data_ptr<scalar_t>(),
                values.data_ptr<scalar_t>(),
                queries.data_ptr<scalar_t>(),
                memory_steps.data_ptr<float>(),
                partial_grad_keys.data_ptr<float>(),
                grad_values.data_ptr<scalar_t>(),
                partial_grad_queries.data_ptr<float>(),
                grad_memory.data_ptr<scalar_t>(),
                chunk_len,
                hidden_size,
                key_size,
                hidden_tiles,
                decay,
                beta,
                inv_scale);
            return;
        default:
            TORCH_CHECK(false, "Unsupported key_size for chunk_update_backward_cuda: ", key_size);
    }
}

bool supports_chunk_cuda_kernel(
    const at::Tensor& keys,
    const at::Tensor& values,
    const at::Tensor& queries,
    const at::Tensor& memory) {
    if (!keys.is_cuda() || !values.is_cuda() || !queries.is_cuda() || !memory.is_cuda()) {
        return false;
    }
    if (keys.scalar_type() != values.scalar_type() ||
        keys.scalar_type() != queries.scalar_type() ||
        keys.scalar_type() != memory.scalar_type()) {
        return false;
    }
    const auto dtype = keys.scalar_type();
    if (dtype != at::kFloat && dtype != at::kHalf && dtype != at::kBFloat16) {
        return false;
    }
    const auto key_size = keys.size(2);
    return key_size > 0 && key_size <= static_cast<int64_t>(kChunkMaxKeyRegs * kChunkWarpSize);
}

bool supports_chunk_cuda_tensor(const at::Tensor& tensor) {
    if (!tensor.is_cuda()) {
        return false;
    }
    const auto dtype = tensor.scalar_type();
    return dtype == at::kFloat || dtype == at::kHalf || dtype == at::kBFloat16;
}

}  // namespace

namespace grm {

std::tuple<at::Tensor, at::Tensor, at::Tensor> chunk_update_forward_cuda(
    const at::Tensor& keys,
    const at::Tensor& values,
    const at::Tensor& queries,
    const at::Tensor& memory,
    double memory_decay) {
    if (!supports_chunk_cuda_kernel(keys, values, queries, memory)) {
        return chunk_update_forward(keys, values, queries, memory, memory_decay);
    }

    auto keys_contig = keys.contiguous();
    auto values_contig = values.contiguous();
    auto queries_contig = queries.contiguous();
    auto memory_contig = memory.contiguous();

    const int batch_size = static_cast<int>(keys_contig.size(0));
    const int chunk_len = static_cast<int>(keys_contig.size(1));
    const int key_size = static_cast<int>(keys_contig.size(2));
    const int hidden_size = static_cast<int>(values_contig.size(2));
    const int hidden_tiles = (hidden_size + kChunkRowsPerBlock - 1) / kChunkRowsPerBlock;

    if (chunk_len == 0) {
        return {
            at::zeros({keys_contig.size(0), 0, values_contig.size(2)}, values_contig.options()),
            memory_contig,
            at::empty({0}, memory_contig.options()),
        };
    }

    auto hidden = at::empty({keys_contig.size(0), keys_contig.size(1), values_contig.size(2)}, values_contig.options());
    auto memory_out = at::empty_like(memory_contig);

    const float decay = static_cast<float>(memory_decay);
    const float beta = 1.0f - decay;
    const float inv_scale = 1.0f / std::sqrt(static_cast<float>(key_size));

    c10::cuda::CUDAGuard device_guard(keys_contig.device());
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        keys_contig.scalar_type(),
        "chunk_update_forward_cuda",
        [&] {
            launch_chunk_update_forward_kernel<scalar_t>(
                keys_contig,
                values_contig,
                queries_contig,
                memory_contig,
                hidden,
                memory_out,
                nullptr,
                batch_size,
                hidden_tiles,
                chunk_len,
                hidden_size,
                key_size,
                decay,
                beta,
                inv_scale);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {hidden, memory_out, at::empty({0}, memory_contig.options())};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> chunk_update_backward_cuda(
    const at::Tensor& grad_hidden,
    const at::Tensor& grad_memory_out,
    const at::Tensor& keys,
    const at::Tensor& values,
    const at::Tensor& queries,
    const at::Tensor& memory,
    const at::Tensor& aux,
    double memory_decay) {
    (void)aux;

    if (!supports_chunk_cuda_kernel(keys, values, queries, memory) ||
        !supports_chunk_cuda_tensor(grad_hidden) ||
        !supports_chunk_cuda_tensor(grad_memory_out) ||
        grad_hidden.scalar_type() != values.scalar_type() ||
        grad_memory_out.scalar_type() != memory.scalar_type()) {
        return chunk_update_backward(
            grad_hidden,
            grad_memory_out,
            keys,
            values,
            queries,
            memory,
            aux,
            memory_decay);
    }

    auto grad_hidden_contig = grad_hidden.contiguous();
    auto grad_memory_out_contig = grad_memory_out.contiguous();
    auto keys_contig = keys.contiguous();
    auto values_contig = values.contiguous();
    auto queries_contig = queries.contiguous();
    auto memory_contig = memory.contiguous();

    const int batch_size = static_cast<int>(keys_contig.size(0));
    const int chunk_len = static_cast<int>(keys_contig.size(1));
    const int key_size = static_cast<int>(keys_contig.size(2));
    const int hidden_size = static_cast<int>(values_contig.size(2));
    const int hidden_tiles = (hidden_size + kChunkRowsPerBlock - 1) / kChunkRowsPerBlock;

    if (chunk_len == 0) {
        return {
            at::zeros_like(keys_contig),
            at::zeros_like(values_contig),
            at::zeros_like(queries_contig),
            grad_memory_out_contig.contiguous(),
        };
    }

    auto hidden_tmp = at::empty(
        {keys_contig.size(0), keys_contig.size(1), values_contig.size(2)},
        values_contig.options());
    auto memory_out_tmp = at::empty_like(memory_contig);
    auto memory_steps = at::empty(
        {keys_contig.size(0), keys_contig.size(1), values_contig.size(2), keys_contig.size(2)},
        memory_contig.options().dtype(at::kFloat));

    auto partial_grad_keys = at::zeros(
        {keys_contig.size(0), hidden_tiles, keys_contig.size(1), keys_contig.size(2)},
        memory_contig.options().dtype(at::kFloat));
    auto grad_values = at::empty_like(values_contig);
    auto partial_grad_queries = at::zeros(
        {keys_contig.size(0), hidden_tiles, keys_contig.size(1), keys_contig.size(2)},
        memory_contig.options().dtype(at::kFloat));
    auto grad_memory = at::empty_like(memory_contig);

    const float decay = static_cast<float>(memory_decay);
    const float beta = 1.0f - decay;
    const float inv_scale = 1.0f / std::sqrt(static_cast<float>(key_size));

    c10::cuda::CUDAGuard device_guard(keys_contig.device());
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        keys_contig.scalar_type(),
        "chunk_update_backward_cuda_forward_recompute",
        [&] {
            launch_chunk_update_forward_kernel<scalar_t>(
                keys_contig,
                values_contig,
                queries_contig,
                memory_contig,
                hidden_tmp,
                memory_out_tmp,
                &memory_steps,
                batch_size,
                hidden_tiles,
                chunk_len,
                hidden_size,
                key_size,
                decay,
                beta,
                inv_scale);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        keys_contig.scalar_type(),
        "chunk_update_backward_cuda",
        [&] {
            launch_chunk_update_backward_kernel<scalar_t>(
                grad_hidden_contig,
                grad_memory_out_contig,
                keys_contig,
                values_contig,
                queries_contig,
                memory_steps,
                partial_grad_keys,
                grad_values,
                partial_grad_queries,
                grad_memory,
                batch_size,
                hidden_tiles,
                chunk_len,
                hidden_size,
                key_size,
                decay,
                beta,
                inv_scale);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto grad_keys = partial_grad_keys.sum(1).to(keys_contig.scalar_type()).contiguous();
    auto grad_queries = partial_grad_queries.sum(1).to(queries_contig.scalar_type()).contiguous();

    return {grad_keys, grad_values, grad_queries, grad_memory};
}

}  // namespace grm
