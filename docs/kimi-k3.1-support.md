# Adding Moonshot Kimi-K3.1-Pro Support to llama.cpp

This document describes how support for `KimiK31ForConditionalGeneration`
(**Kimi-K3.1-Pro**, Moonshot AI, released 2026-11-02) was added to llama.cpp:
the text model, its quantization, and the vision encoder (which also covers
Kimi-K3). For each change it says what changed, why, and how it was verified.

All work is on branch `kimi-k31`. The text model is commit `736eaa1e7`, pushed to
remote `kecso`. The vision work and this document are not committed yet.

The primary reference for every semantic claim is vLLM. The K3.1 Hugging Face repo
ships configuration and preprocessing code but **no modeling code**, so vLLM is the
only executable reference available:

- text / DSA indexer: vLLM PR #65685, `vllm/models/kimi_k3/nvidia/dsa.py`
- vision tower: `vllm/model_executor/models/kimi_k25_vit.py` (K3 reuses the K2.5 code with different flags)
- image prompt format: `vllm/models/kimi_k3/common/mm_preprocess.py`

Claims are labelled **measured** (ran and observed), **from code** (read in the
reference implementation) or **inferred** (reasoned, not tested).

## What K3.1 is

K3.1-Pro is Kimi-K3 post-trained with RL. Its weights are new, so the K3 GGUF cannot be
reused. The architecture is K3's with **one** structural change:

| | Kimi-K3 | Kimi-K3.1-Pro |
|---|---|---|
| Layers | 93 (69 KDA linear-attention, 24 MLA full-attention) | same |
| MoE | 896 experts, latent MoE, MXFP4 experts | same |
| Full-attention layers | MLA, nope-only | **MLA + "Gated DSA" lightning indexer** |
| DSA layers (0-based) | – | 3, 7, 11, …, 91, 92 (exactly the 24 MLA layers) |
| Tokenizer | `tiktoken.model`, kimi-k2 pre-tokenizer | identical file |
| Vision tower | MoonViT3d, 27 layers | identical `vision_config` |

### The DSA indexer (from code)

On every full-attention layer, a small "lightning indexer" scores the past tokens and picks
the top 2048 for the MLA attention to look at:

```
q     = indexer.attn_q_b( q_a_norm( attn_q_a(x) ) )      32 heads x 128, reuses the MLA q latent
k     = LayerNorm( indexer.attn_k(x) )                   weight + bias, fp32, eps = rms_norm_eps
w     = indexer.proj(x) * 128^-0.5 * 32^-0.5
score = sum_h  w_h * relu(q_h . k)                      causal, keep top 2048
```

It differs from DeepSeek-V3.2's indexer, which llama.cpp already has in `deepseek32.cpp`:

- **no RoPE** (`index_rope_dim = 0`, consistent with K3's nope-only MLA)
- **no Hadamard rotation**
- `index_topk_freq = 1`: every DSA layer computes its own selection, and none is shared across layers

The indexer adds about 7.4M parameters per DSA layer, 120 tensors in total.

## Part 1: Text model (commit `736eaa1e7`)

### Design choice: a separate architecture, dense attention for now

K3.1 gets its own architecture string, `kimi-k31`, with its own model file
`src/models/kimi-k31.cpp`. The loader class inherits from K3's. Reasons:

- A K3.1 GGUF carries tensors and keys that K3 does not have. A separate arch makes a
  K3 binary refuse a K3.1 file instead of misreading it.
- The future sparse-attention graph has a clear home without adding `if (k31)`
  branches to K3's graph.

**The indexer is loaded but not evaluated yet.** The graph is K3's, so every MLA layer
attends to every past token:

- **Up to 2048 tokens per sequence this is exact.** The top-2048 selection then
  contains every token. vLLM itself skips the indexer for prefills this short (from code).
- **Beyond 2048 tokens it is an approximation.** The model sees more context than it was
  trained to attend to. I expect long-context quality to degrade, but I have **not measured** it.
  The loader logs a warning once when `n_ctx_seq > 2048`.

This was a deliberate shortcut. It unblocked conversion, quantization and testing,
because perplexity and imatrix both run at 512-token chunks. Sparse attention changes
only the inference graph: the GGUFs already contain the indexer, so **no re-conversion
or re-quantization** will be needed.

### Changes, file by file

| File | What | Why |
|---|---|---|
| `conversion/kimi_k3.py` | New `KimiK31Model(KimiK3Model)`, registered for `KimiK31ForConditionalGeneration`. Writes the indexer head count (32), key length (128) and top-k (2048). Raises `NotImplementedError` unless the DSA config matches what the code implements: `index_rope_dim == 0`, `index_k_norm_type == "layernorm"`, `index_topk_freq` 1 or unset, and DSA layers == full-attention layers. | Turns an unsupported future variant into a hard failure instead of a silently wrong GGUF. The indexer tensor names already map through the existing DSA entries in `tensor_mapping.py`. |
| `conversion/__init__.py` | `"KimiK31ForConditionalGeneration": "kimi_k3"` | Converter dispatch. |
| `conversion/base.py` | Added `KimiK31ForConditionalGeneration` to the top-level-arch exception list in `get_model_architecture`. | K3.1's `text_config` reports `KimiLinearForCausalLM`, an older and incompatible architecture. Without the exception the converter picked that class (**measured** in a dry run). |
| `gguf-py/gguf/constants.py` | `MODEL_ARCH.KIMI_K31 = "kimi-k31"`, tensor list = K3's plus `INDEXER_K_NORM`, `INDEXER_PROJ`, `INDEXER_ATTN_K`, `INDEXER_ATTN_Q_B`. | New arch. |
| `src/llama-arch.h/.cpp` | `LLM_ARCH_KIMI_K31` / `"kimi-k31"`. Added next to K3 in the hybrid, rs-rollback and `supports_sm_tensor` (false) switches. | Behaves like K3 in all of them. |
| `src/llama-model.cpp` | Constructs `llama_model_kimi_k31`. Added to the `LLAMA_ROPE_TYPE_NONE` list. | Nope-only MLA, as in K3. |
| `src/llama-context.cpp` | Added to K3's larger graph-node budget. | The 93-layer graph exceeds the default budget at batch 2048, where `graph_reserve` aborts on K3. |
| `src/models/models.h`, `src/models/kimi-k31.cpp` | `llama_model_kimi_k31 : llama_model_kimi_k3`. Reads the 3 indexer keys, sets `f_norm_eps = rms_norm_eps` for the indexer LayerNorm, loads 5 indexer tensors per MLA layer, warns above 2048 tokens, builds K3's graph. | See the design choice above. |
| `src/llama-quant.cpp` | For `kimi-k31`, never quantize `indexer.attn_q_b`, `indexer.attn_k`, `indexer.proj`. | See Part 2. |

No changes were needed for the chat template or the tool-call parser. llama.cpp detects
K3's template by its marker tokens, which K3.1 shares. K3.1's `encoding_k3.py` differs
from K3's only in tool-argument edge cases.

### Verification (measured)

- Dry-run and full conversion produce the 120 indexer tensors: `proj` and the norms in F32,
  `attn_k` and `attn_q_b` in BF16. The metadata is arch `kimi-k31`, 32/128/2048, 93 blocks,
  2693 tensors.
- The BF16 GGUF (1,489,180 MiB, 33 shards) and the Q2_K load and generate text.
- A BF16 perplexity run (v1 calibration corpus, 512-token chunks, the same setup as K3's
  baseline of 4.7587 ± 0.196) was **stopped at chunk 12 of 20** with a running value of
  **4.29**. A 12-chunk running value is not comparable to K3's 20-chunk final, so there is
  **no quality comparison yet**.

### Converting

The input is the **Hugging Face safetensors directory**, not a GGUF directory.
`tokenization_kimi.py` imports `encoding_k3` at top level, so the model directory must be
on `PYTHONPATH`. `PYTHONDONTWRITEBYTECODE=1` stops Python writing `__pycache__` into a
read-only model folder.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<hf>/Kimi-K3.1-Pro \
python3 convert_hf_to_gguf.py <hf>/Kimi-K3.1-Pro \
  --outfile /media/kecso/8t_nvme/moonshotai.Kimi-K3.1-Pro-GGUF/BF16/moonshotai.Kimi-K3.1-Pro-BF16.gguf \
  --outtype bf16
```

Use BF16, not F16. The source is BF16, and F16 has a smaller exponent range.

## Part 2: Quantization

### Keep the indexer unquantized

`llama-quant.cpp` already has an indexer exclusion list, but it matches MiniMax's tensor
names (`indexer.k_proj` / `indexer.q_proj`). The DSA names used here
(`indexer.attn_q_b` / `attn_k` / `proj`) fall through to the file's default type. That
causes **silent breakage** at low bit widths: the quantizer exits 0, and the model then
attends to the wrong tokens. This happened on GLM-5.3, which uses the same tensor names.

For `kimi-k31` the three indexer weights are now never quantized. The cost is small
(~15 MiB per DSA layer). There is also a second reason. While the graph is dense, the
indexer never runs, so `llama-imatrix` would collect **no** importance data for it.

### MXFP4 experts need `--allow-requantize`

Moonshot ships the routed experts in MXFP4, so they are already quantized and there is
no higher-precision source. `llama-quantize` refuses this without `--allow-requantize`.
The refusal comes 0.4 s into the run, and **`--dry-run` does not catch it** (measured).

### Builds produced (measured)

| Build | Size | BPW | Time | Experts | Rest |
|---|---|---|---|---|---|
| BF16 (source) | 1,489,180 MiB | 4.49 | – | MXFP4 (native) | BF16 / F32 |
| **MXFP4_MOE** | 1,438,990 MiB | 4.34 | 21 min | MXFP4, copied bit-exact | Q8_0 |
| **Q2_K** (no imatrix) | 962,551 MiB | 2.90 | 58 min | gate/up Q2_K, down Q3_K | mixed Q2_K/Q3_K, `attn_k_b` Q4_0 |

Both quantized builds keep the indexer at BF16/F32.

```bash
build-k31-cuda/bin/llama-quantize --allow-requantize --keep-split \
  .../BF16/moonshotai.Kimi-K3.1-Pro-BF16-00001-of-00033.gguf \
  .../Q2_K/moonshotai.Kimi-K3.1-Pro-Q2_K.gguf Q2_K          # or MXFP4_MOE
```

`attn_k_b` falls back to Q4_0 because its rows are 128 wide, which isn't divisible by Q2_K's 256-element block. That is harmless.

### Why MXFP4_MOE and not Q4_K_M as "the 4-bit build"

Dry runs (measured) give **Q4_K_M = 1,604,116 MiB**, larger than the BF16 GGUF it comes
from. The experts are already MXFP4 at ~4.25 bpw, so Q4_K_M re-quantizes them *up* to
Q4_K/Q6_K and loses precision in the conversion. MXFP4_MOE copies the experts unchanged
and quantizes only the small remainder to Q8_0. It is both smaller and lossless where it matters.

The other dry runs: Q3_K_S = 1,141,864 MiB, Q3_K_M = 1,263,944 MiB.

### Fitting in memory

The machine has 1,002 GB RAM and 96 + 32 GB VRAM. **Only Q2_K fits.** MXFP4_MOE runs,
but part of the expert weights is read from the NVMe on every token.

## Part 3: Running

```bash
# optional: pre-fill the page cache with a sequential read at full NVMe speed
cat /media/kecso/8t_nvme/moonshotai.Kimi-K3.1-Pro-GGUF/Q2_K/*.gguf > /dev/null

llama-server -m .../Q2_K/moonshotai.Kimi-K3.1-Pro-Q2_K-00001-of-00033.gguf \
  -ngl 99 --n-cpu-moe 85 -c 2048          # or: -ngl 99 -cmoe -c 2048
```

- **Cold start looks hung.** On the first request the server pulled ~900 GB from disk at
  ~211 MB/s through random page faults, with the process in `D` state and GPUs idle
  (measured). Pre-warming with `cat` avoids it.
- **`-ngl N` without `-cmoe` / `--n-cpu-moe` runs out of memory.** One Q2_K MoE layer's
  experts are ~10 GB (3,087 + 3,087 + 4,042 MiB), so `-ngl 40` would need ~400 GB of VRAM.
- **Keep `-c` at or below 2048** until sparse attention lands. See Part 1.
- `-ngl 99 --n-cpu-moe 85` (non-expert weights plus 8 layers of experts on the GPUs) is
  the configuration confirmed working.

## Part 4: Vision (Kimi-K3 and K3.1)

### What the tower is (from code)

K3.1's `vision_config` is identical to K3's. vLLM builds K3's tower with Kimi-K2.5's
`MoonViT3dPretrainedModel` / `KimiK25MultiModalProjector` classes, changing only
config flags. llama.cpp already supports K2.5 (`kimik25` projector), so the port is a new
projector type, **`kimik3`**, that reuses K2.5's position-embedding resize and 2D RoPE and
changes the rest:

| | Kimi-K2.5 (`kimik25`) | Kimi-K3 / K3.1 (`kimik3`) |
|---|---|---|
| Norms | LayerNorm | **RMSNorm** |
| Biases | yes | **none** (attention, MLP, patch embed, projector) |
| Head dim | `n_embd / n_head` | **`qkv_hidden_size / n_head` = 1536 / 12 = 128** (wider than 1024 / 12) |
| Pos-emb resize | bicubic | **bilinear** |
| Projector | pre-LayerNorm → linear → GELU → linear | **linear_1 → GELU (erf) → linear_2 → RMSNorm (eps 1e-5)** |
| ViT MLP activation | GELU | GELU (tanh approx, `gelu_pytorch_tanh`) |
| Image prompt | `<\|media_begin\|>` … `<\|media_end\|>` | **`<\|media_begin\|>image {W}x{H}<\|media_content\|>` … `<\|media_end\|>`** |
| Patch limit | `in_patch_limit` 16384 | **65536** |

### Changes, file by file

| File | What | Why |
|---|---|---|
| `conversion/kimivl.py` | New `KimiK3VisionModel(KimiK25Model)` registered for both `KimiK3…` and `KimiK31ForConditionalGeneration`. Writes projector type `kimik3`, `head_dim` = qkv/heads, vision norm eps 2^-7. Raises `NotImplementedError` unless 12 config flags match what the graph implements. Small refactor of `KimiK25Model`: `projector_type` class attribute and a `vit_norm_eps()` hook, with K2.5 output unchanged. | The base class writes the projector type and eps once. Overriding through hooks avoids duplicate GGUF keys. K3 names the head count `vt_num_attention_heads`, and K2.5's tensor permute would otherwise default to 16 heads, which is wrong. |
| `conversion/__init__.py` | Two `MMPROJ_MODEL_MAP` entries → `kimivl` | `--mmproj` dispatch. |
| `gguf-py/gguf/constants.py` | `VisionProjectorType.KIMIK3 = "kimik3"` | |
| `gguf-py/gguf/tensor_mapping.py` | `mm_projector.post_norm` → `V_MM_POST_NORM` (`mm.post_norm`) | The only new tensor. `proj.0`/`proj.2` reuse K2.5's `mm.1`/`mm.2` mapping. |
| `tools/mtmd/clip-impl.h` | `PROJECTOR_TYPE_KIMIK3`, name `kimik3` | |
| `tools/mtmd/models/kimik3.cpp`, `models.h`, `CMakeLists.txt` | `clip_graph_kimik3 : clip_graph_kimik25`, new `build()` | The graph, described below. |
| `tools/mtmd/clip.cpp` | Builder dispatch. Hparams shared with K2.5, plus a warmup cap. Tensor loading: `mm.1`, `mm.2`, `mm.post_norm`, no biases. Added to the dynamic-size, 2D-position and `n_mmproj_embd` switches. | |
| `tools/mtmd/mtmd.cpp` | `img_beg_has_size`: `img_beg` becomes a printf format filled with each image's **original** width and height. | vLLM and Moonshot's processor put the uploaded image size (not the resized one) into the prompt. mtmd's `img_beg` was a fixed string. |

The graph (`kimik3.cpp`):

1. Patch embed (no bias), plus the learned 64×64 position embedding resized **bilinearly**.
2. `build_vit` with `NORM_TYPE_RMS` and split 2D RoPE (Q/K permuted at conversion, as for K2.5).
3. `build_patch_merge_permute` for the 2×2 `sd2_tpool` merge. With a single frame the temporal mean is a no-op.
4. `build_ffn(mm_1, mm_2, FFN_GELU_ERF)`, then an RMSNorm with `mm.post_norm` at eps 1e-5.

### Decisions worth knowing

- **Vision RMSNorm eps = 2^-7 (inferred, not confirmed by Moonshot).** vLLM builds the
  encoder norms as `nn.RMSNorm(hidden_dim)` with no eps, and PyTorch then uses
  `finfo(input.dtype).eps` at run time. The tower runs in BF16, which gives
  2^-7 = 0.0078125. The HF repo has no modeling code to confirm this. The value is stored
  in the GGUF (`clip.vision.attention.layer_norm_epsilon`), so it can be changed without
  touching code. The projector's post-norm uses the explicit `projector_ln_eps = 1e-5`.
- **Warmup is capped at 1024 merged tokens (896×896).** K3 allows 65,536 patches per
  image. The K2.5 code path warms up at the maximum size, which for K3 would need a
  12 × 65,536² attention matrix (~206 GB in F32) without flash attention.
- **Known preprocessing difference.** llama.cpp stretches the image to a multiple of 28
  pixels. Moonshot's processor resizes, then zero-pads the bottom and right edges. This is
  the same approximation llama.cpp already makes for K2.5. Transparent-background
  chessboard filling (RGBA inputs) is also not reproduced.

### Generating the adapter

Only 2 of the 96 safetensors shards hold vision weights (`model-00095`: projector, 3
tensors; `model-00096`: tower, 165 tensors), about 0.9 GB in total. They and the small
config/processor files were downloaded from Hugging Face into
`/media/kecso/8t_nvme/hf-Kimi-K3.1-Pro-vision/`. The index in that folder was trimmed to
those two shards so the converter does not look for the other 94.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 convert_hf_to_gguf.py /media/kecso/8t_nvme/hf-Kimi-K3.1-Pro-vision \
  --mmproj --outtype bf16 \
  --outfile /media/kecso/8t_nvme/moonshotai.Kimi-K3.1-Pro-GGUF/mmproj-moonshotai.Kimi-K3.1-Pro-BF16.gguf
```

Result (measured): 904 MB, 168 tensors, `kimik3`, 27 blocks, 12 heads, head_dim 128,
projection_dim 7168 (matches the text model), image mean/std 0.5, max 12,845,056 pixels.

Usage: `llama-server -m <text GGUF> --mmproj mmproj-moonshotai.Kimi-K3.1-Pro-BF16.gguf …`

### Verification (measured)

An independent fp32 PyTorch reference was written from vLLM's code
(`/media/kecso/8t_nvme/hf-Kimi-K3.1-Pro-vision/verify/ref.py`). It reads the **original, unpermuted** HF tensors, so
it also tests the converter's Q/K permutation. A small C++ harness
(`verify/enc.cpp`, same folder) runs only the llama.cpp encoder through `clip.h`. Both
encode the same normalized random image, and the outputs are compared.

Real K3.1 weights:

| mmproj | Image | rel. L2 error | cosine |
|---|---|---|---|
| F32 | 336×224 (pos-emb interpolated) | 0.12% | 0.9999991 |
| F32 | 896×896 (native 64×64 grid) | 0.15% | 0.9999988 |
| BF16 | 336×224 | 1.25% | 0.99992 |
| BF16 | 896×896 | 0.84% | 0.99997 |

The BF16 error is BF16 rounding of activations in the matmuls. That is expected, and it
is the shipped format.

Sensitivity check on a tiny random-weight model built from the real config: deliberately
wrong choices in the reference increase the error over the baseline 7.5e-4 as follows.
Bicubic instead of bilinear: ×35. Eps 1e-6 instead of 2^-7: ×5. Tanh instead of erf GELU
in the projector: ×1.4, because the two functions are numerically very close. So the test
can tell these choices apart.

The chat-template path was checked by reading the code. llama-server turns each image
into a `media_marker` text part. K3's template renders any part that isn't
`image`/`image_url` as plain text, so the marker reaches mtmd unchanged and mtmd replaces
it with the size-aware image block.

## Open items

1. **Sparse DSA attention** in `kimi-k31.cpp`, so output is exact beyond 2048 tokens. The
   template is `glm5-next.cpp` (KDA + nope MLA + DSA, `llama-memory-hybrid-idx`) plus
   `deepseek32.cpp`'s per-token indexer graph, without RoPE and without Hadamard.
2. **Perplexity**: a complete 20-chunk BF16 run and a Q2_K run, to compare with K3's 4.7587.
3. **End-to-end image test**: Q2_K + mmproj with a real image (e.g. `tools/mtmd/test-1.jpeg`).
   Not run yet, because it needs the 940 GB model reloaded with `--mmproj`.
4. **Confirm the vision eps (2^-7)** against Moonshot's reference implementation if one is published.
5. Commit and push the vision changes and this document.

## Artifacts

| Path | What |
|---|---|
| `/media/kecso/8t_nvme/moonshotai.Kimi-K3.1-Pro-GGUF/BF16/` | 33 shards, 1,489,180 MiB |
| `/media/kecso/8t_nvme/moonshotai.Kimi-K3.1-Pro-GGUF/MXFP4_MOE/` | 33 shards, 1,438,990 MiB |
| `/media/kecso/8t_nvme/moonshotai.Kimi-K3.1-Pro-GGUF/Q2_K/` | 33 shards, 962,551 MiB |
| `/media/kecso/8t_nvme/moonshotai.Kimi-K3.1-Pro-GGUF/mmproj-moonshotai.Kimi-K3.1-Pro-BF16.gguf` | vision adapter, 904 MB |
| `/media/kecso/8t_nvme/hf-Kimi-K3.1-Pro-vision/` | downloaded vision shards and configs (trimmed index), plus `mmproj-f32-test.gguf` used for verification |
| `/media/kecso/8t_nvme/k31-quant-q2k.log`, `k31-quant-mxfp4moe.log`, `k31-ppl-bf16.log` | quantization logs and the partial perplexity log |
