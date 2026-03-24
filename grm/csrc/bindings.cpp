#include "grm_cuda.h"

#include <torch/extension.h>

TORCH_LIBRARY(grm_cuda, m) {
    m.def("chunk_update_forward(Tensor keys, Tensor values, Tensor queries, Tensor memory, float memory_decay) -> (Tensor hidden, Tensor memory_out, Tensor aux)");
    m.def("chunk_update_backward(Tensor grad_hidden, Tensor grad_memory_out, Tensor keys, Tensor values, Tensor queries, Tensor memory, Tensor aux, float memory_decay) -> (Tensor grad_keys, Tensor grad_values, Tensor grad_queries, Tensor grad_memory)");
    m.def("apply_query_forward(Tensor memories, Tensor queries) -> Tensor");
    m.def("apply_query_backward(Tensor grad_hidden, Tensor memories, Tensor queries) -> (Tensor grad_memories, Tensor grad_queries)");
    m.def("batched_memory_gather(Tensor memories_per_batch, Tensor topk_indices) -> Tensor");
}

TORCH_LIBRARY_IMPL(grm_cuda, CPU, m) {
    m.impl("chunk_update_forward", grm::chunk_update_forward);
    m.impl("chunk_update_backward", grm::chunk_update_backward);
    m.impl("apply_query_forward", grm::apply_query_forward);
    m.impl("apply_query_backward", grm::apply_query_backward);
    m.impl("batched_memory_gather", grm::batched_memory_gather);
}

TORCH_LIBRARY_IMPL(grm_cuda, CUDA, m) {
    m.impl("chunk_update_forward", grm::chunk_update_forward_cuda);
    m.impl("chunk_update_backward", grm::chunk_update_backward_cuda);
    m.impl("apply_query_forward", grm::apply_query_forward_cuda);
    m.impl("apply_query_backward", grm::apply_query_backward_cuda);
    m.impl("batched_memory_gather", grm::batched_memory_gather_cuda);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "GRM C++/CUDA extension scaffold";
#if GRM_CUDA_EXT_WITH_KERNELS
    m.attr("__grm_cuda_with_kernels__") = pybind11::bool_(true);
#else
    m.attr("__grm_cuda_with_kernels__") = pybind11::bool_(false);
#endif
}
