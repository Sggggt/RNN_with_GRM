#include "grm_cuda.h"

#include <torch/csrc/autograd/autograd.h>

#include <cmath>
#include <tuple>
#include <vector>

namespace grm {

namespace {

void check_chunk_inputs(
    const at::Tensor& keys,
    const at::Tensor& values,
    const at::Tensor& queries,
    const at::Tensor& memory) {
    TORCH_CHECK(keys.dim() == 3, "keys must have shape [B, T, K]");
    TORCH_CHECK(values.dim() == 3, "values must have shape [B, T, H]");
    TORCH_CHECK(queries.dim() == 3, "queries must have shape [B, T, K]");
    TORCH_CHECK(memory.dim() == 3, "memory must have shape [B, H, K]");
    TORCH_CHECK(keys.sizes().slice(0, 2) == values.sizes().slice(0, 2), "keys and values batch/time dims must match");
    TORCH_CHECK(keys.sizes() == queries.sizes(), "keys and queries must have the same shape");
    TORCH_CHECK(values.size(0) == memory.size(0), "memory batch dim must match values batch dim");
    TORCH_CHECK(values.size(2) == memory.size(1), "memory hidden dim must match values hidden dim");
    TORCH_CHECK(keys.size(2) == memory.size(2), "memory key dim must match keys key dim");
  }

void check_apply_query_inputs(const at::Tensor& memories, const at::Tensor& queries) {
    TORCH_CHECK(memories.dim() == 4 || memories.dim() == 5, "memories must have shape [B, R, H, K] or [B, Q, R, H, K]");
    if (memories.dim() == 4) {
        TORCH_CHECK(queries.dim() == 2, "queries must have shape [B, K] when memories rank is 4");
        TORCH_CHECK(memories.size(0) == queries.size(0), "batch dim mismatch between memories and queries");
        TORCH_CHECK(memories.size(3) == queries.size(1), "key dim mismatch between memories and queries");
        return;
    }

    TORCH_CHECK(queries.dim() == 3, "queries must have shape [B, Q, K] when memories rank is 5");
    TORCH_CHECK(memories.size(0) == queries.size(0), "batch dim mismatch between memories and queries");
    TORCH_CHECK(memories.size(1) == queries.size(1), "query dim mismatch between memories and queries");
    TORCH_CHECK(memories.size(4) == queries.size(2), "key dim mismatch between memories and queries");
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> chunk_update_forward(
    const at::Tensor& keys,
    const at::Tensor& values,
    const at::Tensor& queries,
    const at::Tensor& memory,
    double memory_decay) {
    check_chunk_inputs(keys, values, queries, memory);

    const auto batch_size = keys.size(0);
    const auto chunk_len = keys.size(1);
    const auto hidden_size = values.size(2);
    const auto key_size = keys.size(2);

    if (chunk_len == 0) {
        return {
            at::zeros({batch_size, 0, hidden_size}, values.options()),
            memory.contiguous(),
            at::empty({0}, memory.options()),
        };
    }

    auto memory_state = memory.contiguous();
    std::vector<at::Tensor> hidden_steps;
    hidden_steps.reserve(static_cast<size_t>(chunk_len));

    const double inv_scale = 1.0 / std::sqrt(static_cast<double>(key_size));
    const double beta = 1.0 - memory_decay;

    for (int64_t t = 0; t < chunk_len; ++t) {
        auto key_t = keys.select(1, t);
        auto value_t = values.select(1, t);
        auto query_t = queries.select(1, t);

        auto update = at::bmm(value_t.unsqueeze(2), key_t.unsqueeze(1)) * inv_scale;
        memory_state = memory_state * memory_decay + update * beta;
        auto hidden_t = at::bmm(memory_state, query_t.unsqueeze(2)).squeeze(2);
        hidden_steps.push_back(hidden_t);
    }

    return {
        at::stack(hidden_steps, 1).contiguous(),
        memory_state.contiguous(),
        at::empty({0}, memory.options()),
    };
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> chunk_update_backward(
    const at::Tensor& grad_hidden,
    const at::Tensor& grad_memory_out,
    const at::Tensor& keys,
    const at::Tensor& values,
    const at::Tensor& queries,
    const at::Tensor& memory,
    const at::Tensor& aux,
    double memory_decay) {
    (void)aux;

    auto keys_req = keys.detach().clone();
    auto values_req = values.detach().clone();
    auto queries_req = queries.detach().clone();
    auto memory_req = memory.detach().clone();
    keys_req.set_requires_grad(true);
    values_req.set_requires_grad(true);
    queries_req.set_requires_grad(true);
    memory_req.set_requires_grad(true);

    auto outputs = chunk_update_forward(keys_req, values_req, queries_req, memory_req, memory_decay);
    auto grads = torch::autograd::grad(
        {std::get<0>(outputs), std::get<1>(outputs)},
        {keys_req, values_req, queries_req, memory_req},
        {grad_hidden.contiguous(), grad_memory_out.contiguous()},
        false,
        false);

    return {
        grads[0].contiguous(),
        grads[1].contiguous(),
        grads[2].contiguous(),
        grads[3].contiguous(),
    };
}

at::Tensor apply_query_forward(const at::Tensor& memories, const at::Tensor& queries) {
    check_apply_query_inputs(memories, queries);
    if (memories.dim() == 5) {
        return at::einsum("bqrhk,bqk->bqrh", {memories, queries}).contiguous();
    }
    return at::einsum("brhk,bk->brh", {memories, queries}).contiguous();
}

std::tuple<at::Tensor, at::Tensor> apply_query_backward(
    const at::Tensor& grad_hidden,
    const at::Tensor& memories,
    const at::Tensor& queries) {
    auto memories_req = memories.detach().clone();
    auto queries_req = queries.detach().clone();
    memories_req.set_requires_grad(true);
    queries_req.set_requires_grad(true);

    auto hidden = apply_query_forward(memories_req, queries_req);
    auto grads = torch::autograd::grad(
        {hidden},
        {memories_req, queries_req},
        {grad_hidden.contiguous()},
        false,
        false);

    return {grads[0].contiguous(), grads[1].contiguous()};
}

at::Tensor batched_memory_gather(
    const at::Tensor& memories_per_batch,
    const at::Tensor& topk_indices) {
    TORCH_CHECK(memories_per_batch.dim() == 4, "memories_per_batch must have shape [B, S, H, K]");
    TORCH_CHECK(topk_indices.dim() == 2 || topk_indices.dim() == 3, "topk_indices must have shape [B, R] or [B, Q, R]");
    TORCH_CHECK(memories_per_batch.size(0) == topk_indices.size(0), "batch dim mismatch between memories_per_batch and topk_indices");

    const auto batch_size = memories_per_batch.size(0);
    std::vector<at::Tensor> selected_batches;
    selected_batches.reserve(static_cast<size_t>(batch_size));

    for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        auto memory_batch = memories_per_batch.select(0, batch_idx);
        auto indices = topk_indices.select(0, batch_idx).toType(at::kLong).contiguous();
        auto gathered = memory_batch.index_select(0, indices.reshape({-1}));

        std::vector<int64_t> out_shape(indices.sizes().begin(), indices.sizes().end());
        out_shape.push_back(memory_batch.size(1));
        out_shape.push_back(memory_batch.size(2));
        selected_batches.push_back(gathered.reshape(out_shape));
    }

    return at::stack(selected_batches, 0).contiguous();
}

}  // namespace grm
