#pragma once

#include <torch/extension.h>

#include <tuple>

namespace grm {

std::tuple<at::Tensor, at::Tensor, at::Tensor> chunk_update_forward(
    const at::Tensor& keys,
    const at::Tensor& values,
    const at::Tensor& queries,
    const at::Tensor& memory,
    double memory_decay);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> chunk_update_backward(
    const at::Tensor& grad_hidden,
    const at::Tensor& grad_memory_out,
    const at::Tensor& keys,
    const at::Tensor& values,
    const at::Tensor& queries,
    const at::Tensor& memory,
    const at::Tensor& aux,
    double memory_decay);

at::Tensor apply_query_forward(const at::Tensor& memories, const at::Tensor& queries);

std::tuple<at::Tensor, at::Tensor> apply_query_backward(
    const at::Tensor& grad_hidden,
    const at::Tensor& memories,
    const at::Tensor& queries);

at::Tensor batched_memory_gather(const at::Tensor& memories_per_batch, const at::Tensor& topk_indices);

std::tuple<at::Tensor, at::Tensor, at::Tensor> chunk_update_forward_cuda(
    const at::Tensor& keys,
    const at::Tensor& values,
    const at::Tensor& queries,
    const at::Tensor& memory,
    double memory_decay);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> chunk_update_backward_cuda(
    const at::Tensor& grad_hidden,
    const at::Tensor& grad_memory_out,
    const at::Tensor& keys,
    const at::Tensor& values,
    const at::Tensor& queries,
    const at::Tensor& memory,
    const at::Tensor& aux,
    double memory_decay);

at::Tensor apply_query_forward_cuda(const at::Tensor& memories, const at::Tensor& queries);

std::tuple<at::Tensor, at::Tensor> apply_query_backward_cuda(
    const at::Tensor& grad_hidden,
    const at::Tensor& memories,
    const at::Tensor& queries);

at::Tensor batched_memory_gather_cuda(const at::Tensor& memories_per_batch, const at::Tensor& topk_indices);

}  // namespace grm
