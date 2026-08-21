# Fork Changes

> This file documents this fork's changes relative to upstream
> `utilityai/llama-cpp-rs` (based on tag `0.1.154`), so future upstream merges
> can assess conflicts and regression risk.

## Overview

Two self-contained, additive changes layered on top of upstream `0.1.154`:

1. **CANN (Huawei Ascend NPU) backend** — Cargo feature flags + build-script wiring
2. **`LlamaBatch` embedding API** — three new methods for embedding/audio input

No files inside the `llama.cpp` submodule are modified; every change lives on
the Rust side. Both changes are designed to be merge-friendly: new features and
new methods only, with no restructuring of upstream code.

---

## 1. CANN (Huawei Ascend NPU) backend features

Exposes llama.cpp's existing `GGML_CANN` CMake backend through Cargo features.
The base `cann` feature turns the backend on; four SOC-specific variants each
imply `cann` and pre-set `SOC_TYPE`, so cross-builds don't need `npu-smi`
running on the build host.

### Feature matrix

| Feature      | Implies `cann` | Sets `SOC_TYPE` | Notes                                              |
|--------------|----------------|-----------------|----------------------------------------------------|
| `cann`       | —              | no              | Base flag; relies on `SOC_TYPE` env or `npu-smi`.  |
| `cann-310p`  | yes            | `Ascend310P`    |                                                    |
| `cann-910`   | yes            | `Ascend910`     |                                                    |
| `cann-910b`  | yes            | `Ascend910B`    |                                                    |
| `cann-910c`  | yes            | `Ascend910C`    |                                                    |

### `SOC_TYPE` resolution precedence

First active SOC feature (in declaration order) > `SOC_TYPE` env var > `npu-smi` autodetect (inside CMake). When multiple SOC features are active, only the first is used — see [Multiple SOC features](#multiple-soc-features) below.

### Multiple SOC features

If more than one `cann-*` SOC feature is enabled (as `cargo --all-features`
does), the build does **not** abort. The first matching variant in declaration
order is selected silently and the rest are ignored:

declaration order: `cann-310p` → `cann-910` → `cann-910b` → `cann-910c`.

Production builds should still pass exactly one SOC (e.g. `--features cann-910b`)
to avoid building for the wrong card; the silent pick exists only so that
`--all-features` type-checks without a host NPU.

### Files

| File                                              | Change                                                                                         |
|---------------------------------------------------|------------------------------------------------------------------------------------------------|
| `llama-cpp-sys-2/Cargo.toml`                      | +5 feature defs (`cann`, `cann-310p`, `cann-910`, `cann-910b`, `cann-910c`); add `ggml-cann/**/*` to `include` |
| `llama-cpp-2/Cargo.toml`                          | +5 feature pass-throughs                                                                       |
| `examples/{simple,embeddings,mtmd,reranker}/Cargo.toml` | +5 feature pass-throughs each                                                            |
| `llama-cpp-sys-2/build.rs`                        | CANN CMake config block (`GGML_CANN` + SOC resolution) after the `rocm` block; CANN link block after `hipblas` |

### Linking

The CANN link block runs under `cfg!(feature = "cann") && !build_shared_libs`.
It links the minimal verified set (3 libraries): `ascendcl`, `nnopbase`,
`opapi`, resolved from the first applicable link dir:

1. `CANN_LIB_DIR` — explicit directory holding **target-arch** CANN libs. Its
   architecture is validated (ELF `e_machine` of the first CANN lib found);
   a mismatch fails the build immediately with a pointer to `CANN_LIB_DIR`.
2. `<toolkit>/lib64` (toolkit from `ASCEND_TOOLKIT_HOME`, falling back to
   `CANN_INSTALL_DIR`) — validated the same way. On mismatch (a host-arch
   toolkit under a cross build) the path is **skipped with a warning** while
   the `-l` flags stay: the wrong-arch `lib64/` can never satisfy the link,
   and emitting it would only poison the link search order (lld takes the
   first match and hard-errors on an incompatible ELF). Target-arch libs
   must then come from another `-L` path (e.g. prebuilt copies added by a
   dependent crate's build script) or from `CANN_LIB_DIR`.

```rust
println!("cargo:rustc-link-lib=dylib=ascendcl");
println!("cargo:rustc-link-lib=dylib=nnopbase");
println!("cargo:rustc-link-lib=dylib=opapi");
```

This split is deliberate for cross builds against a host-arch toolkit: the
CMake phase keeps using that toolkit (it needs the host-side ascendc
compiler and headers), while the link phase takes its libraries from a
target-arch source. Deploy the same target-arch set on the target machine.

### Build examples

```bash
# Native build on an Ascend host (NPU visible via npu-smi)
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cargo build --release --features cann-910b -p simple

# Cross-compile to Ascend 310P (aarch64) with a target-arch (cann-arm) toolkit:
# lib64/ is aarch64 and passes the arch check
export ASCEND_TOOLKIT_HOME=/path/to/cann-arm
export RUSTFLAGS="-L /path/to/openmp-arm/lib"
cargo zigbuild --release --features cann-310p --target aarch64-unknown-linux-gnu -p simple

# Cross-compile against a host-arch (x86_64) toolkit: CMake uses it for the
# ascendc compiler; CANN_LIB_DIR points the linker at target-arch libs
export ASCEND_TOOLKIT_HOME=/path/to/cann-x86
export CANN_LIB_DIR=/path/to/target-arch-cann-libs
export RUSTFLAGS="-L /path/to/openmp-arm/lib"
cargo zigbuild --release --features cann-310p --target aarch64-unknown-linux-gnu -p simple
```

---

## 2. `LlamaBatch` embedding API

Adds three methods for inference paths that feed continuous embedding vectors
instead of token ids (e.g. audio/ASR models such as Qwen3 ASR).

| Method                                                                                  | Purpose                                                                                                                              |
|-----------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| `LlamaBatch::new_with_embd(n_tokens, embd_dim, n_seq_max)`                              | Allocate a batch with an embedding buffer (`embd_dim > 0`). The existing `new()` now delegates here with `embd_dim = 0` (no behavior change). |
| `LlamaBatch::set_embd(embd_data, dim, positions, stride, seq_ids)`                      | Write embeddings, positions, per-token seq ids, and the last-token logits flag in one call. Supports a position stride and two seq-id modes (one shared id, or one id per token). |
| `LlamaBatch::set_logits_at(idx, logits)`                                                | Enable logits at an additional position for multi-sequence batches, since `set_embd` only flags the last token.                      |

### `seq_ids` modes

- `&[id]` (length 1): every token shares the same seq id.
- `&[id0, id1, ...]` (length == actual token count): each token gets its own seq id.

### Files

| File                          | Change                                                                                              |
|-------------------------------|-----------------------------------------------------------------------------------------------------|
| `llama-cpp-2/src/llama_batch.rs` | +`new_with_embd`, +`set_embd`, +`set_logits_at`; `new()` body now delegates to `new_with_embd`. |

---

## mtmd output embeddings (2026-08-21)

Expose the audio/image embedding output buffer so callers can run the mmproj
tower standalone (encode chunks themselves, then feed embeddings into their own
llama.cpp batch path) instead of going through `mtmd_helper_eval_chunks`.

| File | Change |
|---|---|
| `llama-cpp-2/src/mtmd.rs` | +`MtmdContext::get_output_embd(&self, len: usize) -> Option<&[f32]>` borrowing the C buffer of the last `encode_chunk` (caller-supplied length; additive, no upstream code touched). |

---

## mtmd log redirection in `send_logs_to_tracing` (2026-08-21)

mtmd/clip logging (`clip_model_loader`, tower load info) goes through a
separate global callback (`mtmd_log_set`), not `llama_log_set` — so callers
of `send_logs_to_tracing` still saw raw mtmd loader logs on stderr. Redirect
it with the same `logs_to_trace` callback, reusing the ggml state (mtmd LOG
macros emit GGML_LOG_LEVEL-graded complete lines).

| File | Change |
|---|---|
| `llama-cpp-2/src/lib.rs` | `send_logs_to_tracing` additionally calls `llama_cpp_sys_2::mtmd_log_set(Some(logs_to_trace), ggml_heap_state)` behind `#[cfg(feature = "mtmd")]` (one statement inside the existing `unsafe` block; no upstream code touched). |

---

## Upstream-merge notes

Both changes are additive and avoid restructuring upstream code, to keep merges
low-conflict:

- **CANN build.rs blocks** insert immediately after the existing `rocm` CMake
  block and the `hipblas` link block. If upstream reorders backend blocks,
  re-locate the inserts next to the equivalent `rocm`/`hip` code.
- **CANN feature lines** are appended within each `[features]` table; no
  existing feature line is modified.
- **`llama_batch.rs`** additions are appended within the existing `impl` block;
  only `new()`'s body changes (it now calls `new_with_embd`). Upstream has not
  touched the `new()` / `n_tokens()` region, so conflict risk is low.

### Merge log

| Date       | Upstream tag | Submodule  | Result                                                        |
|------------|--------------|------------|---------------------------------------------------------------|
| 2026-08-06 | `0.1.154`    | `b10200`   | Clean 3-way merge, **zero conflicts** (5 files auto-merged: `llama-cpp-sys-2/{Cargo.toml,build.rs}`, `llama-cpp-2/Cargo.toml`, `examples/{simple,embeddings}/Cargo.toml`). CANN blocks verified in-place after `rocm`/`hipblas`; upstream `GGML_*` env passthrough (0.1.153) is compatible with the CANN `SOC_TYPE` precedence. Fork commit `59dda3a` retained. |

The merge was validated with `git merge-tree --write-tree` (dry run) and then
performed as a real merge on branch `etsllm-dep-v2`; the merged tree kept the
CANN CMake/link blocks at their expected positions and upstream's b10200 port
intact. Smoke checks after merge: default-feature build and the `cann-*`
feature matrix type-check; a full CANN link build still requires an Ascend
toolkit host (see [Build examples](#build-examples)).
