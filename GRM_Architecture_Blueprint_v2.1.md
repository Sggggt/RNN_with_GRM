# GRM Architecture Blueprint

> Version: v3.2  
> Status: current implementation blueprint  
> Scope: paper-aligned GRM only  
> Last updated: 2026-03-23

## 1. 蓝图目标

本文档描述当前仓库里仍然生效的架构边界、模块职责和数据流，只覆盖 paper-aligned GRM。

当前硬约束：

- 不改动 `Memory_Caching_RNNs_with_Growing_Memory.md` 的理论与公式
- 不恢复任何 `torch.compile` 接口
- 不恢复任何 Triton 训练接口
- 热点算子统一经由 CUDA C++ 封装层进入扩展

## 2. 论文公式与实现映射

当前实现仍然保持：

```math
k_t = \mathrm{normalize}(W_k x_t),\quad
v_t = W_v x_t,\quad
q_t = \mathrm{normalize}(W_q x_t)
```

```math
M_t = \lambda M_{t-1} + (1-\lambda)\frac{v_t k_t^\top}{\sqrt{d_k}}
```

```math
h_t^{online} = M_t q_t
```

历史片段检索仍然保持：

```math
a_t^{(i)} = q_t^\top k^{(i)},\quad
\gamma_t^{(i)} = \mathrm{softmax}_{Top-K}(a_t^{(i)})
```

融合表示仍然保持：

```math
\tilde{h}_t = \mathrm{Fuse}\big(h_t^{online}, \{c_t^{(i)}\}, \{\gamma_t^{(i)}\}\big)
```

蓝图只改变“在哪里算”，不改变“算什么”。

## 3. 当前活动架构

当前仓库只保留一条活动架构：

- `GRMEnhancedRNN`
- `MultiLayerGRM`
- `GRMLanguageModel`

已经彻底移除的公开运行时接口：

- 旧的编译器优化入口
- Triton 训练入口
- 关闭 CUDA C++ 主路径的对外开关

## 4. 分层结构

### 4.1 Training Layer

文件：

- `grm/utils/train.py`

职责：

- CLI
- 数据加载
- phase1 / phase2
- optimizer / scheduler
- checkpoint / logger

### 4.2 Model Layer

文件：

- `grm/core/paper_grm.py`

职责：

- `GRMEnhancedRNN`
- segment 级循环
- online memory 路径
- retrieval 路径
- `PaperMemoryBank`

### 4.3 Routing And Fusion Layer

文件：

- `grm/core/gating_unit.py`
- `grm/core/aggregator.py`

职责：

- 生成 Top-K 与 gates
- 执行 `residual` / `concat` / `gate` / `linear` / `memory_soup`

### 4.4 Runtime Ops Layer

文件：

- `grm/core/cuda_ops.py`

职责：

- lazy load `grm_cuda_ext`
- 暴露统一入口：
  - `cuda_chunk_update`
  - `cuda_apply_query`
  - `cuda_batched_memory_gather`
- 暴露 runtime status
- 提供 `auto/native/fallback` 算子级策略
- 提供 debug fallback

### 4.5 Extension Layer

文件：

- `setup.py`
- `grm/csrc/grm_cuda.h`
- `grm/csrc/grm_cuda.cpp`
- `grm/csrc/bindings.cpp`
- `grm/csrc/chunk_update.cu`
- `grm/csrc/apply_query.cu`
- `grm/csrc/batched_gather.cu`

职责：

- 注册 `torch.ops.grm_cuda.*`
- 维持稳定接口
- 承接真实 CUDA kernel

## 5. 单次前向数据流

### 5.1 输入

- 普通任务：`[B, T, D_in]`
- WikiText：token ids `[B, T]`

### 5.2 Segment 切分

序列按固定 `segment_size` 切分。每个 segment 内依次做：

1. online memory 更新
2. 对历史 summaries 做 sparse Top-K routing
3. gather 相关 memory
4. apply query
5. fuse 在线分支与检索分支
6. 把当前 segment 的 memory 与 summary 推入 bank

### 5.3 Online Path

当前 segment 内：

1. `LinearMemoryCell` 产生 `k / v / q`
2. 调用 `cuda_chunk_update`
3. 得到：
   - `h_online`
   - `memory_end`

截至 `2026-03-23`，`chunk_update` 的 CUDA 主路径已经是 Phase 3 重写后的 native kernel。

### 5.4 Retrieval Path

对当前 query：

1. 读取 `PaperMemoryBank` 中的历史 summaries
2. `SparseGRMGatingUnit` 给出 `gates` 与 `topk_indices`
3. `cuda_batched_memory_gather` 取回选中的 memory
4. 必要时做 selective recomputation
5. `cuda_apply_query` 应用 query
6. `GRMAggregator` 做融合

## 6. Memory Bank 设计

`PaperMemoryBank` 以 `batch_size` 分组缓存：

- `memory`: `[B, H, K]`
- `summary`: `[B, D_in]`
- `checkpoint.inputs`
- `checkpoint.memory_start`

关键特性：

- 支持 `state_dict()` 序列化
- 支持 `model.to(device)` 时迁移缓存
- 支持 deterministic recomputation

## 7. 关键张量形状

### 7.1 Online Path

- `batch_inputs`: `[B, C, D]`
- `keys`: `[B, C, K]`
- `values`: `[B, C, H]`
- `queries`: `[B, C, K]`
- `memory`: `[B, H, K]`
- `h_online`: `[B, C, H]`

### 7.2 Retrieval Path

- `segment_keys`: `[B, S, D]`
- `topk_indices`: `[B, C, R]` 或 `[B, R]`
- `memories_per_batch`: `[B, S, H, K]`
- `selected_memories`: `[B, C, R, H, K]` 或 `[B, R, H, K]`
- `retrieved`: `[B, C, R, H]` 或 `[B, R, H]`
- `gates`: `[B, C, R]` 或 `[B, R]`

## 8. 当前运行时约束

当前运行时只有固定主路径：

- 模型内部始终调用 `cuda_ops.py`
- `GRMEnhancedRNN` 会拒绝旧 runtime 参数
- 唯一保留的命令行调试开关是 `cuda_cpp_debug_fallback`

除此之外，运行时还支持环境变量级别的算子策略：

- `GRM_CUDA_KERNEL_POLICY`
- `GRM_CUDA_CHUNK_UPDATE_POLICY`
- `GRM_CUDA_APPLY_QUERY_POLICY`
- `GRM_CUDA_BATCHED_GATHER_POLICY`

也就是说：

- 不再存在“启用旧路径”的外部接口
- 不再存在“关闭 CUDA C++ 主路径”的外部接口
- 训练、测试、文档都以同一条调用图为准

## 9. 扩展层当前状态

当前已经落地：

- `torch.ops.grm_cuda.*` 显式 `CPU/CUDA` 注册
- `chunk_update` 原生 CUDA forward/backward
- `apply_query` 原生 CUDA forward/backward
- `batched_memory_gather` CUDA dispatch
- runtime status
- `auto/native/fallback` 策略
- parity 测试

当前最关键的状态判断是：

- `extension_loaded=True`
- `compiled_with_cuda_kernels=True`
- `dispatch_kernels.*.cuda=True`

截至 `2026-03-23`，这些条件在验证用 WSL 环境中均已满足。

## 10. 当前已验证结果

当前蓝图对应的关键验证包括：

- `python3 setup.py build_ext --inplace --force`
- `python3 examples/test_grm.py`
- `python3 examples/benchmark_cuda_chunk_update.py --device cuda --iters 200`
- WikiText 单 batch smoke training

其中 `chunk_update` 白盒 benchmark 在默认 `auto` 下实测约为：

- fallback `0.724 ms`
- `cuda_ops` 主入口 `0.219 ms`

这说明当前蓝图里 `chunk_update` 已经从“真实 kernel 接通”推进到“默认 native 获胜”的阶段。

## 11. WSL 构建策略

工作环境统一按 WSL 使用：

```bash
~/workspace/GRM
```

`setup.py` 的策略：

- 检测到与 `torch.version.cuda` 匹配的 `nvcc` 时构建 `CUDAExtension`
- 否则自动构建 `CppExtension`

当前验证环境已经对齐到 CUDA `12.8`。

## 12. 当前边界

当前蓝图仍有这些边界：

- 只支持 `rnn_type='linear'`
- 只支持 fixed segments
- `query_source='input'`
- `summary_source='input_mean'`
- `pool_method='mean'`
- 端到端训练速度尚未按白盒 benchmark 的比例同步下降

这意味着当前真正的剩余工作不再是“接线”，而是：

- retrieval 路径进一步 profiling
- `apply_query` rank-5 与多 query 场景继续调优
- Python / memory bank / routing 的端到端热点清理

## 13. 下一阶段

当前蓝图下的后续工作顺序应保持单链推进：

1. 保持 `cuda_ops.py` 接口稳定
2. 继续做端到端 profiling，而不是回退到旧运行时路线
3. 针对 WikiText 的真实训练路径定位非 `chunk_update` 热点
4. 视 profiling 结果继续优化 `apply_query`、routing、memory bank 和数据路径

## 14. 总结

当前仓库的正确心智模型是：

1. 论文公式不变
2. Python 负责结构与训练调度
3. 热点算子统一经由 CUDA C++ 封装层进入扩展
4. `chunk_update` 已经进入默认 native 运行状态
5. 后续优化重点已经转移到端到端训练链路，而不是接口连通性
