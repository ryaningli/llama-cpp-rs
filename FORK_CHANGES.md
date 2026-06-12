# Fork 本地改动说明

> 本文件记录 fork 仓库相对于上游 `utilityai/llama-cpp-rs` 的所有改动，
> 以便主线更新时评估合并冲突和回归风险。

## 改动概览

共修改 7 个源码文件（不含 Cargo.lock），新增 3 个 feature + 3 个 API，改进 backends 部署体验，修复 zigbuild 交叉编译兼容性。

### 1. 新增 feature: `dynamic-backends-no-variants`

基于 `dynamic-backends`，区别是不设置 CMake 的 `GGML_CPU_ALL_VARIANTS=ON`，
只构建一个通用的 `libggml-cpu.so`，而不是为每种 CPU 微架构（haswell/zen4/...）各编译一份。

`dynamic-backends-no-variants` 在 feature 层面自动激活 `dynamic-backends`，
因此代码中只需 `#[cfg(feature = "dynamic-backends")]` 即可覆盖两种模式，
不需要 `any(...)` 条件判断。唯一区分两者的是 `build.rs` 中
`cfg!(feature = "dynamic-backends-no-variants")` 控制 `GGML_CPU_ALL_VARIANTS` 开关。

**修改文件链：**

| 文件 | 改动 |
|------|------|
| `llama-cpp-sys-2/Cargo.toml` | +1 行 feature 定义 |
| `llama-cpp-2/Cargo.toml` | +1 行 feature 透传 |
| `examples/simple/Cargo.toml` | +1 行 feature 透传 |

**逻辑修改：**

| 文件 | 改动 |
|------|------|
| `llama-cpp-sys-2/build.rs` | `GGML_CPU_ALL_VARIANTS` 条件化，`no-variants` 时不设置 |
| `llama-cpp-2/src/llama_backend.rs` | 新增 `find_lib_dir()` + 重写 `load_backends()` 多策略搜索 |
| `examples/simple/src/main.rs` | 添加 `#[cfg(feature = "dynamic-backends")] load_backends()` 调用 |

### 2. 新增 feature: `prebuilt-dynamic-backends`

基于 `dynamic-backends-no-variants`，启用时跳过 CMake 编译 llama.cpp，
直接链接到用户提供的预构建共享库（`libllama.so`、`libggml.so`、`libggml-base.so`）。

用户通过 `RUSTFLAGS="-L /path/to/lib"` 或 `.cargo/config.toml` 提供库搜索路径，
build.rs 不硬编码任何路径。运行时 backends 通过 `dladdr` 自动发现。

**修改文件链：**

| 文件 | 改动 |
|------|------|
| `llama-cpp-sys-2/Cargo.toml` | +1 行 feature 定义 |
| `llama-cpp-2/Cargo.toml` | +1 行 feature 透传 |
| `examples/simple/Cargo.toml` | +1 行 feature 透传 |

**逻辑修改：**

| 文件 | 改动 |
|------|------|
| `llama-cpp-sys-2/build.rs` | 在 `common_wrapper_build.compile()` 前后各插入一个 `cfg!(feature = "prebuilt-dynamic-backends")` 块：生成 `build-info.cpp` + 声明链接库 + `return` 跳过 CMake |

### 3. Backends 与共享库统一目录部署

**原问题：** 上游将 backends 安装到 `out/backends/` 子目录，需要单独设置 `LD_LIBRARY_PATH` 指向 backends 目录，部署时需要两条路径配置。

**改动：** 将 `GGML_BACKEND_DIR` 从 `out/backends/` 改为 `out/lib/`，使 backends 的 `.so` 文件（如 `libggml-cpu.so`）和 `libllama.so` 安装在同一目录。部署时只需一个 `LD_LIBRARY_PATH`。

| 文件 | 改动 |
|------|------|
| `llama-cpp-sys-2/build.rs` | `GGML_BACKEND_DIR` 改为 `out_dir.join("lib")`，`cargo:backends_dir` 也改为 `out/lib/` |

### 4. `load_backends()` 多策略自动发现

`load_backends()` 现在按以下优先级查找 backends 目录：

1. `GGML_BACKEND_PATH` 环境变量
2. `libllama.so` 所在目录（通过 `dladdr` 自动检测，Unix 平台）
3. 编译时嵌入的 `BACKENDS_DIR`

这使得生产部署时无需额外配置 backends 路径——只要 `libggml-cpu.so` 等文件和 `libllama.so` 在同一目录（由上面的改动 3 保证），就会被自动发现。

| 文件 | 改动 |
|------|------|
| `llama-cpp-2/src/llama_backend.rs` | 新增 `find_lib_dir()` 函数（Unix 上使用 `dladdr`），重写 `load_backends()` |

### 5. Bug 修复: build.rs hard_link 竞态条件

`llama-cpp-sys-2/build.rs` 硬链接共享库部分：

原代码用 `if !dst.exists() { hard_link() }` 模式复制共享库到 `target/release/`，
在重复构建时 `exists()` 检查与 `hard_link()` 之间存在竞态，导致 `EEXIST` 错误。

修复方式：在 `hard_link` 前无条件 `remove_file`（忽略"文件不存在"错误）。

涉及 3 处相同模式（target 目录、examples、deps）。

### 6. 依赖修复: `hf-hub` TLS 后端切换为 rustls

**原问题：** 上游 `hf-hub` 默认使用 `native-tls`（依赖系统 OpenSSL），
在使用 `cargo zigbuild` 交叉编译时找不到 OpenSSL 头文件导致构建失败。

**改动：** 在 workspace `Cargo.toml` 中将 `hf-hub` 改为 `default-features = false`，
显式启用 `rustls-tls`、`tokio`、`ureq`。`rustls` 是纯 Rust 实现的 TLS，无系统依赖。

| 文件 | 改动 |
|------|------|
| `Cargo.toml` | `hf-hub` 依赖改为 `default-features = false, features = ["rustls-tls", "tokio", "ureq"]` |

### 7. 新增 API: `LlamaBatch::new_with_embd()` + `LlamaBatch::set_embd()` + `LlamaBatch::set_logits_at()`

为 embedding 输入场景（如 Qwen3 ASR 音频模型）添加批量设置嵌入数据的方法。

**`new_with_embd(n_tokens, embd_dim, n_seq_max)`：**
- 创建 batch 时指定 `embd_dim > 0`，使 `llama_batch_init` 分配 embd 缓冲区
- 原 `new()` 不变，代理到 `new_with_embd(..., 0, ...)`

**`set_embd(embd_data, dim, positions, stride, seq_ids)`：**
- 将完整 embedding 向量、位置编码、seq_id、logits 一次性写入 batch
- 支持 stride 位置编码（如 Qwen3 的 4x stride）
- `seq_ids` 参数支持两种模式：
  - `&[id]`（长度 1）：所有 token 共享同一 seq_id
  - `&[id0, id1, ...]`（长度 == n_tokens）：每个 token 独立指定 seq_id
- 直接操作 `llama_batch` 已分配的 `embd`、`pos`、`seq_id` 等缓冲区
- 更新 `initialized_logits` 以确保 `get_logits_ith()` 可正确读取最后一个 token 的 logits

**`set_logits_at(idx, logits)`：**
- 设置指定 token 位置的 logits 标志
- 用于多序列 batch 场景：`set_embd()` 只设置最后一个 token 的 logits，当多个序列合并到同一个 batch 时，需要通过此方法为每个序列的最后一个 token 追加设置 logits
- 同步更新 `initialized_logits`

| 文件 | 改动 |
|------|------|
| `llama-cpp-2/src/llama_batch.rs` | `new()` 代理到 `new_with_embd`，新增 `new_with_embd()` 构造函数、`set_embd()` 方法和 `set_logits_at()` 方法 |

## 代码改动约束

> 所有后续改动必须遵循以下原则，以保持与上游 `utilityai/llama-cpp-rs` 的可合入性。

### 原则 1：优先易于合入，而非代码优雅

- **不提取公共函数**：即使 prebuilt 分支和 CMake 分支有相似的平台链接逻辑，也不抽取为共享函数。提取函数会改动上游代码结构，增加合并冲突范围。
- **不重构上游代码**：不重命名变量、不调整代码顺序、不添加抽象层。上游代码保持原样，我们的改动作为"插件"插入。
- **接受适度重复**：代码重复（如平台链接 match 块）优于改动上游代码结构。

### 原则 2：纯新增优于修改

- **优先添加新代码块**：prebuilt feature 的实现方式是在现有代码之间插入 `if cfg!(feature = "prebuilt-dynamic-backends") { ... return; }` 块，通过 early return 跳过后面的 CMake 流程，而不是修改 CMake 流程本身。
- **新增 feature 不修改现有 feature 行为**：`dynamic-backends-no-variants` 和 `prebuilt-dynamic-backends` 只在各自的 `cfg!` 块内生效，不影响原有 `dynamic-backends` 的行为。
- **Cargo.toml 只做纯行追加**：feature 定义在文件末尾追加，不修改现有 feature 行。

### 原则 3：最小化改动行数

- **改动行数越少越好**：每多改一行上游代码，就多一行可能的合并冲突。
- **只改动必要的行**：backends 目录从 `out/backends/` 改为 `out/lib/` 是必要的语义变更（改变了运行时行为），无法避免。`GGML_CPU_ALL_VARIANTS` 条件化只加了一行 `if !cfg!(...)` 包裹。
- **不改无关代码**：不同时做格式化、注释清理、变量重命名等。

### 原则 4：改动集中在已知区域

当前所有改动集中在以下区域，上游合并时只需关注这些位置：

| 区域 | 文件:行范围 | 改动性质 |
|------|-------------|----------|
| prebuilt build-info | `build.rs:~522-539` | 纯新增 |
| prebuilt 链接 | `build.rs:~543-580` | 纯新增 |
| backends CMake 配置 | `build.rs:~879-900` | 最小修改（3 行改 5 行） |
| backends_dir 输出 | `build.rs:~900` | 1 行修改 |
| hard_link 修复 | `build.rs:~1165-1180` | 每处 2 行改 2 行 |
| load_backends 重写 | `llama_backend.rs:~200-279` | 函数替换（区域独立） |
| feature 定义 | 各 `Cargo.toml` | 纯行追加 |
| batch embd API | `llama_batch.rs:~148-340` | 纯新增（`new_with_embd` + `set_embd` + `set_logits_at`） |

## 合入注意事项

### 上游合并时可能冲突的位置

1. **`llama-cpp-sys-2/build.rs` 第 879-900 行**
   - `dynamic-backends` CMake 配置块
   - 我们改动：`GGML_BACKEND_DIR` 指向 `out/lib/` 而非 `out/backends/`，`GGML_CPU_ALL_VARIANTS` 条件化
   - 如果上游修改了此区域的 CMake 参数，需要保留我们的改动

2. **`llama-cpp-sys-2/build.rs` 第 1165-1180 行**
   - hard_link 复制逻辑
   - 如果上游重构了这段代码（比如改用 `std::fs::copy`），合并时优先采用上游方式，
     但需确保也修复了 `EEXIST` 竞态问题

3. **`llama-cpp-2/src/llama_backend.rs` 第 200-279 行**
   - `find_lib_dir()`、`load_backends()` 重写
   - 如果上游修改了此区域的 API，需要保留 dladdr 回退逻辑和多策略搜索

4. **`examples/simple/Cargo.toml` 和 `examples/simple/src/main.rs`**
   - 上游可能重构 example 结构或添加新 feature
   - 合并时确保 feature 透传和 `load_backends()` 调用不丢失

### 合入策略

```bash
# 1. 添加上游 remote（首次）
git remote add upstream https://github.com/utilityai/llama-cpp-rs.git

# 2. 拉取上游最新代码
git fetch upstream

# 3. 合并到当前分支
git merge upstream/main

# 4. 解决冲突时，参考上述"可能冲突的位置"
#    优先保留上游代码，重新应用我们的改动

# 5. 合并后重新构建验证
cargo clean -p llama-cpp-sys-2 --release
cargo clean -p llama-cpp-2 --release
cargo build --release -p simple --features dynamic-backends-no-variants
```

### 如果上游也修复了 hard_link 问题

删除我们对 `build.rs` hard_link 部分的改动，使用上游的修复即可。

### 如果上游也添加了类似的 no-variants 功能

评估是否可以切换到上游的实现，删除我们的 `dynamic-backends-no-variants` feature。

## 运行命令

```bash
# === CMake 构建（本地编译 llama.cpp）===

# 带 CPU 变体（原行为，构建多种微架构版本）
cargo build --release -p simple --features dynamic-backends

# 不带 CPU 变体（只构建通用版本）
cargo build --release -p simple --features dynamic-backends-no-variants

# 运行（只需一个 LD_LIBRARY_PATH）
LIB_DIR=$(ls -d target/release/build/llama-cpp-sys-2-*/out/lib | tail -1) \
LD_LIBRARY_PATH="$LIB_DIR" \
./target/release/simple -p "你好" --n-len 64 local /path/to/model.gguf

# === 预构建库（跳过 CMake，链接外部 .so）===

# 构建（通过 RUSTFLAGS 提供库搜索路径）
RUSTFLAGS="-L /path/to/llama-cpp/lib -L /path/to/openmp/lib" \
cargo build --release -p simple --features prebuilt-dynamic-backends

# 或在 .cargo/config.toml 中配置（一次设置永久生效）：
# [target.x86_64-unknown-linux-gnu]
# rustflags = ["-L", "/path/to/llama-cpp/lib", "-L", "/path/to/openmp/lib"]
#
# 然后直接：
cargo build --release -p simple --features prebuilt-dynamic-backends

# 运行
LD_LIBRARY_PATH=/path/to/llama-cpp/lib \
./target/release/simple -p "你好" --n-len 64 local /path/to/model.gguf

# === 交叉编译（zigbuild）===

RUSTFLAGS="-L /path/to/llama-cpp/lib -L /path/to/openmp/lib" \
cargo zigbuild --release -p simple --target x86_64-unknown-linux-gnu --features prebuilt-dynamic-backends
```
