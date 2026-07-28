# Adding Kimi K3 support to llama.cpp

Status: design + work in progress on branch `kimi-k3`.
Target: convert -> quantize to ~890 GB -> inference.

This document is written to be read start to finish by someone who wants to
understand *how* a new architecture gets ported, not just *what* to type. Every
claim below is backed by a command you can re-run against the local weights.

Model under analysis:
`~/NAS/tresor/huggingface/hf-safetensors/moonshotai/moonshotai.Kimi-K3`

---

## 1. The method: how to size up an unknown model

Before writing a line of C++, you answer five questions. This ordering matters,
because each answer can kill the project and you want the cheap kills first.

1. **What arch family is it really?** Vendors rename things. Read
   `config.json:architectures` AND `model_type`, then read the *nested* configs.
2. **What are the real tensor names and shapes?** Never trust the modeling code
   alone; ship-time weights routinely disagree with the reference implementation.
3. **What is genuinely new** vs. what llama.cpp already has an op/graph for?
4. **What dtype is on disk?** This decides whether conversion is lossless.
5. **Does the result physically fit anywhere?**

### 1.1 Identify the family

```bash
python3 -c "
import json; c=json.load(open('config.json'))
print(c['architectures'], c['model_type'])
print(c['text_config']['architectures'], c['text_config']['model_type'])
"
```

Result: outer `KimiK3ForConditionalGeneration` / `kimi_k3`, but the text tower is
`KimiLinearForCausalLM` / `kimi_linear`. That single fact is worth a week: it
means K3 is the *same family* as Kimi-Linear-48B-A3B, which llama.cpp already
supports as `LLM_ARCH_KIMI_LINEAR`. We are extending a known arch, not starting
from zero.

Note K3 is a different lineage from Kimi K2.7 (which is DeepSeek-style dense
MLA). Do not use K2.7 as the reference.

### 1.2 Get the ground truth on tensors

The index file maps every tensor to a shard. Collapsing the digits gives you the
architecture skeleton in one screen:

```bash
python3 -c "
import json,re,collections
d=json.load(open('model.safetensors.index.json'))['weight_map']
c=collections.Counter(re.sub(r'\.\d+\.','.N.',n) for n in d)
for k,v in sorted(c.items(), key=lambda x:-x[1]): print(f'{v:8d}  {k}')
"
```

Reading shapes requires parsing the safetensors header (first 8 bytes = header
length, then JSON):

```bash
python3 -c "
import json,struct
with open('model-00002-of-000096.safetensors','rb') as f:
    n=struct.unpack('<Q',f.read(8))[0]; hdr=json.loads(f.read(n))
for k,v in hdr.items():
    if k!='__metadata__': print(k, v['dtype'], v['shape'])
"
```

This is how the two KDA surprises in section 4.5 were found. The modeling code
says one thing, the shipped tensors say another, and **the tensors win**.

---

## 2. What Kimi K3 is

| Property | Value |
| --- | --- |
| Total params | 2.78 T (measured, see 6.1) |
| Active params | 104 B |
| Layers | 93 (1 dense + 92 MoE) |
| Attention | 69 KDA (linear) + 24 gated MLA |
| Hidden size | 7168 |
| Routed experts | 896, top-16, sigmoid gate, `noaux_tc` |
| Shared experts | 2 |
| MoE latent dim | 3584 (experts do NOT run at 7168) |
| Expert FFN dim | 3072 |
| Activation | SiTU-GLU |
| Vocab | 163840, TikToken |
| Context | 1048576 |
| Vision | MoonViT-V2, 27 layers, 401 M |
| Weights on disk | MXFP4 (QAT) for routed experts, BF16 for everything else |

Layer indices in `linear_attn_config` are **1-based**. `full_attn_layers`
contains 4,8,...,92,93 and `kda_layers` contains 1,2,3,5,.... The existing
converter already handles this with `if il + 1 in _full_attn_layers`.

---

## 3. What llama.cpp already gives us for free

This is the good news, and it is most of the model.

| Need | Already exists |
| --- | --- |
| KDA recurrence | `ggml_kda_scan`, `llm_build_delta_net_base` |
| KDA causal conv1d | `causal_conv1d()` in `src/models/kimi-linear.cpp` |
| MLA + absorbed KV cache | `src/models/kimi-linear.cpp` MLA branch |
| Hybrid recurrent+attention memory | `build_inp_mem_hybrid()` |
| Sigmoid MoE gate + `exp_probs_b` bias | `build_moe_ffn` |
| MXFP4 as a runnable type | `GGML_TYPE_MXFP4` (from gpt-oss) |
| MoE over MXFP4 (`mul_mat_id`) | gpt-oss path |
| **compressed-tensors MXFP4 repack** | `conversion/deepseek.py:600` |
| Vision encoder (MoonViT3d) | `PROJECTOR_TYPE_KIMIK25` in mtmd |
| TikToken pre-tokenizer | `kimi-k2` pre-type; K3's `pat_str` is identical |

Verify the last one yourself:

```bash
grep -n "pat_str" -A 4 tokenization_kimi.py    # in the model dir
```

The regex is byte-identical to Kimi-K2's Han-aware pattern, so only the vocab
hash needs registering in `convert_hf_to_gguf_update.py`.

---

## 4. The five new mechanisms

### 4.1 Attention Residuals (AttnRes) - the hard one

Config: `attn_res_block_size: 12`.

Normally a transformer layer does `x = x + f(norm(x))`. K3 replaces the *read*
of the residual stream with a learned, per-token softmax mixture over depth.

The model keeps a stack `block_residual` of hidden-state snapshots. Every 12
layers the current stream is pushed onto the stack (93 layers / 12 -> 8 pushes).
At each layer, twice (before attention and before the MLP), and once more at the
very end of the model, it computes:

```
v      = concat(block_residual, prefix_sum)       # (tokens, n_blocks+1, 7168)
k      = rms_norm(v)                              # over the 7168 axis
scores = sum(k * (norm.weight * proj.weight))     # -> (tokens, n_blocks+1)
probs  = softmax(scores)
out    = probs @ v                                # -> (tokens, 7168)
```

Reference: `modeling_kimi_linear.py:_apply_attn_res` (line 1075).

Three things to notice, because they make the implementation tractable:

1. **The two weight vectors fuse.** `norm.weight * proj.weight` is a single
   `[7168]` vector that can be precomputed at load time. `proj.weight` is stored
   as `[1, 7168]`, i.e. it really is just a vector.
2. **Every op already exists**: `ggml_rms_norm`, `ggml_mul`, `ggml_soft_max`,
   `ggml_mul_mat`. No new kernel, no new backend work.
3. **It never touches the KV cache.** The stack lives only for the current
   batch, so it costs activation memory (up to 9x hidden state for the batch),
   not per-token cache.

What *is* new is structural: the llama.cpp layer loop carries one `inpL` tensor.
K3 needs the loop to carry a growing list of tensors and a separate
`prefix_sum`. That is the single most invasive change in this port.

Note the subtlety in `_forward_attn_residual`: on a push layer
(`layer_idx % 12 == 0`) `prefix_sum` is set to `None` and the attention output
*becomes* the new prefix sum rather than being added to it. Getting this wrong
produces a model that loads and generates plausible-looking garbage, which is
why section 8 insists on numerical validation rather than vibe-checking output.

### 4.2 Latent MoE

Config: `routed_expert_hidden_size: 3584`, `latent_moe_use_norm: true`.

Routed experts do not operate on the 7168-dim stream. The block is:

```
h      = routed_expert_down_proj(x)        # 7168 -> 3584
y      = moe(h)                            # experts are 3584 -> 3072 -> 3584
y      = routed_expert_norm(y)             # RMSNorm at 3584
y      = routed_expert_up_proj(y)          # 3584 -> 7168
out    = y + shared_experts(x)             # shared experts stay at 7168
```

Confirmed by shapes: `experts.0.w1.weight_packed` is `[3072, 1792]` where 1792
bytes = 3584 values, and `routed_expert_down_proj.weight` is `[3584, 7168]`.

Note the router still scores on the **full 7168** input (`gate.weight` is
`[896, 7168]`), not on the latent. So routing happens before the down
projection.

Implementation: three new tensors per layer plus a second `n_embd` for expert
tensor creation. `build_moe_ffn` is reusable as-is; we just wrap it.

### 4.3 SiTU-GLU activation

Config: `hidden_act: "situ"`, `activation_situ_beta: 4.0`,
`activation_situ_linear_beta: 25.0`.

```
situ(gate, up) = [b * tanh(gate/b) * sigmoid(gate)] * [lb * tanh(up/lb)]
```

with b = 4.0, lb = 25.0. Both halves are soft-clipping: the gate branch is a
bounded SiLU-like curve, the up branch is a bounded linear. This is what lets
the model train stably at MXFP4 - activations cannot blow up.

Composable from `ggml_tanh`, `ggml_sigmoid`, `ggml_scale`, `ggml_mul`. A fused
op is a later optimization, not a blocker.

Applies to the dense layer 0 MLP, the shared experts, and every routed expert.

### 4.4 MLA output gate

Config: `mla_use_output_gate: true`. Tensor: `self_attn.g_proj.weight`
`[12288, 7168]` on all 93 layers.

```
attn_out = attn_out * sigmoid(g_proj(x))
```

applied after attention, before `o_proj`. Trivially cheap to add. Note
`mla_use_nope: true` means MLA applies **no RoPE at all** - the "rope" 64 dims
are passed through untouched, exactly as the existing kimi-linear code already
assumes.

### 4.5 KDA deltas from the 48B model

Three differences, all found by reading shipped shapes rather than the modeling
code:

**a. `use_full_rank_gate: true`.** The 48B factors the output gate as
`g_b_proj(g_a_proj(x))` through a 128-dim bottleneck. K3 uses one full-rank
`g_proj` `[12288, 7168]`. Existing code creates `ssm_g_a`/`ssm_g_b`; K3 needs a
single tensor.

**b. `A_log` is `[128]`, not `[96]`.** This one is a genuine trap. The modeling
code says `A_log = torch.log(torch.empty(self.num_heads))` with `num_heads=96`,
but the shipped tensor has 128 elements = `head_dim`. So the decay is
per-channel-within-head, shared across heads - the **transpose** of the 48B
convention. Existing code does:

```c
ggml_tensor * A = ggml_reshape_3d(ctx0, layer.ssm_a, 1, n_head, 1);
```

K3 needs `ggml_reshape_3d(ctx0, layer.ssm_a, head_dim, 1, 1)`. The shapes 96 vs
128 disambiguate the two cases, so the converter can decide automatically.

**c. `gate_lower_bound: -5.0`.** The KDA log-decay is clamped from below before
the scan (`safe_gate=True` in the reference kernel). Needs a `ggml_clamp` that
the current code does not have. Without it, long-context decay can underflow.

### 4.6 One-line change

`LLAMA_MAX_EXPERTS` is 512 in `src/llama-hparams.h:10`. K3 has 896.

---

## 5. Conversion: keeping MXFP4 lossless

This is the part that makes the whole project viable, so it is worth
understanding at the byte level.

### 5.1 Why the formats already match

K3's `quantization_config` says `mxfp4-pack-quantized`, `group_size: 32`,
`num_bits: 4`, `scale_dtype: torch.uint8`. That is:

- 32 weights share one scale
- each weight is 4 bits (E2M1: 0, +-0.5, +-1, +-1.5, +-2, +-3, +-4, +-6)
- the scale is one uint8 holding an E8M0 exponent

A ggml `block_mxfp4` is *exactly* the same thing: `QK_MXFP4 = 32` values, one
`uint8` e8m0 scale, 16 bytes of packed nibbles.

Verify the layout on disk:

```
experts.0.w1.weight_packed   U8 [3072, 1792]     # 1792 bytes = 3584 values
experts.0.w1.weight_scale    U8 [3072,  112]     # 3584 / 32 = 112 scales
```

So conversion is a **byte repack, not a requantization**. Zero loss. The only
work is nibble ordering: compressed-tensors packs consecutive pairs, while ggml
stores values 0..15 in low nibbles and 16..31 in high nibbles.

### 5.2 The code already exists

`conversion/deepseek.py:600` has `_pack_mxfp4_blocks(weight, scale)` doing
precisely this, written for DeepSeek V4. We subclass and reuse it. Do not write
a new one.

```bash
sed -n '595,650p' conversion/deepseek.py
```

### 5.3 What the converter must do

1. Strip the `language_model.model.` prefix (K3 nests the text tower).
2. Stack 896 experts per layer per projection into one 3D tensor. **This is the
   memory hazard**: one layer's experts are 896 x 3 x ~5.5 MB = ~16 GB held at
   once. Budget for it.
3. Repack MXFP4 experts losslessly; leave everything else BF16.
4. Emit the new hparams: `attn_res_block_size`, `routed_expert_hidden_size`,
   the situ betas, `gate_lower_bound`.
5. Handle `A_log` shape 96 vs 128 (section 4.5b), and `g_proj` vs `g_a/g_b`.

There are 497,220 tensors in the index. 494,592 of them are expert shards.

---

## 6. Getting to ~890 GB

### 6.1 The real numbers

Measured, not estimated:

```
routed experts    2722.74 G params    (98.0% of the model)
non-expert        57.19 G params      (2.0%)
total             2779.93 G params
```

The `ignore` list in `quantization_config` tells you what is *not* MXFP4:
attention, shared experts, the dense layer-0 MLP, `lm_head`, and the vision
tower. All BF16.

At source precision: 1446 GB experts + 114 GB rest = **1560 GB**.

### 6.2 The correction to my first assessment

My initial read was that requantizing QAT-MXFP4 weights below 4 bits would fall
off a cliff. On reflection that is wrong, and the distinction matters for your
decision:

Because the model was **quantization-aware trained** at MXFP4, the MXFP4 values
*are* the true weights. Dequantizing them to F32 is exact and lossless. So
going MXFP4 -> IQ2_S is a **single** quantization step from ground truth, not a
double quantization. It is exactly as lossy as taking any BF16 model to 2 bits -
no more.

The real caveat is different and milder: QAT at 4 bits proves the model is
robust *at* 4 bits, not below it. 2.6 bpw is still a ~40% precision cut and will
behave like any aggressive 2-bit MoE quant. Use an imatrix.

### 6.3 Size math

Experts (2722.74 G params):

| Type | bpw | Size |
| --- | --- | --- |
| MXFP4 (source) | 4.25 | 1446 GB |
| IQ3_XXS | 3.0625 | 1042 GB |
| Q2_K | 2.5625 | 872 GB |
| IQ2_S | 2.5 | 851 GB |
| IQ2_XS | 2.3125 | 787 GB |
| IQ2_XXS | 2.0625 | 702 GB |

Non-expert (57.19 G params):

| Type | bpw | Size |
| --- | --- | --- |
| BF16 | 16 | 114 GB |
| Q8_0 | 8.5 | 61 GB |
| Q6_K | 6.5625 | 47 GB |

Combinations:

| Recipe | Total |
| --- | --- |
| IQ2_XS experts + Q8_0 rest | 848 GB |
| **IQ2_S experts + Q6_K rest** | **898 GB** |
| Q2_K experts + Q6_K rest | 919 GB |

So your ~890 GB target is **IQ2_S routed experts + Q6_K everything else**.

### 6.4 Where to spend the budget

The 57 G non-expert params break down roughly as:

```
KDA attention (69 layers)     30.4 G
shared experts (92 layers)    12.2 G
MLA attention (24 layers)      5.6 G
latent down/up (92 layers)     4.7 G
embed + lm_head                2.3 G
dense layer 0 MLP              0.7 G
```

Shared experts run on **every token** and cost only 12 G params. Attention is
36 G and is where 2-bit quants usually break. Keeping all of that at Q6_K/Q8_0
costs ~50 GB out of ~900 and is the highest-value 5% of the budget you will
ever spend. Push the aggression onto the routed experts, which are 98% of the
bytes and only 16/896 active per token.

Use per-tensor overrides rather than a blanket ftype:

```bash
llama-quantize --imatrix k3.imatrix \
  --tensor-type ffn_gate_exps=iq2_s \
  --tensor-type ffn_down_exps=iq2_s \
  --tensor-type ffn_up_exps=iq2_s \
  --tensor-type attn=q6_k \
  --tensor-type shexp=q8_0 \
  --output-tensor-type q8_0 \
  --token-embedding-type q6_k \
  k3-mxfp4.gguf k3-iq2s.gguf IQ2_S
```

`llama-quantize --dry-run` prints the resulting sizes **without doing the work**.
Use it to converge on the recipe before spending days of I/O.

### 6.5 The imatrix problem

An imatrix is what makes 2-bit usable, and generating one requires *running* the
1560 GB model. Chicken and egg.

Options, in order of preference:

1. **mmap from NVMe.** llama.cpp mmaps the GGUF; the page cache pulls in only
   what is touched. Only 104 B params are active per token, but over a
   calibration corpus every expert eventually gets hit. Expect roughly
   0.1-1 tok/s. A 100k-token calibration run is then 1-10 days. Slow, but it is
   unattended and it is a one-time cost.
2. **Short calibration.** Fewer chunks trades imatrix quality for wall time.
   For MoE specifically you want *coverage of experts* more than token count, so
   a diverse corpus beats a long one.
3. **No imatrix.** Q2_K without imatrix on a 2-bit MoE is usually poor. Not
   recommended, but it is a valid first smoke test.

### 6.6 Disk budget

```
source safetensors      1447 GB   (already downloaded)
intermediate GGUF       1560 GB
final IQ2_S             ~898 GB
```

You need the intermediate and the output on disk simultaneously: **~2.5 TB
free** on top of the source. The intermediate cannot be avoided, because
`convert_hf_to_gguf.py` only emits F32/F16/BF16/Q8_0/MXFP4 - it cannot write
IQ2 directly. Keeping the experts as MXFP4 in the intermediate is what keeps it
at 1560 GB instead of 5.5 TB.

---

## 7. Vision

K3 ships MoonViT-V2 (401 M). llama.cpp's `PROJECTOR_TYPE_KIMIK25` is close.
Deltas:

- Encoder blocks use **RMSNorm**, not LayerNorm (`norm_type: "rmsnorm"`).
- **No biases anywhere** (`attn_bias: false`, `linear_bias: false`).
- Projector is `PatchMergerMLPV2`: drop the `pre_norm` LayerNorm, add a
  `post_norm` RMSNorm *after* the projection.

One useful simplification: despite `init_pos_emb_time: 4` in the config, there
is **no `pos_emb.time_weight` tensor** on disk, and the model card lists
modality as text+image only. So no video path is needed.

Vision is deferred to phase 5. The text model is independently useful.

---

## 8. Implementation plan

Phases are ordered so that each one is independently testable and the risky part
is validated before the expensive part.

### Phase 1 - foundations (low risk)
- `LLAMA_MAX_EXPERTS` 512 -> 1024
- Register `LLM_ARCH_KIMI_K3` + new tensor enums
- New hparams: `attn_res_block_size`, `routed_expert_hidden_size`, situ betas,
  `gate_lower_bound`
- SiTU-GLU helper, MLA output gate, KDA full-rank gate, `A_log` axis fix, clamp

### Phase 2 - Latent MoE
Tensor creation + graph wrapping around the existing `build_moe_ffn`.

### Phase 3 - AttnRes
The structural change. Build it standalone first (section 8.1).

### Phase 4 - converter
Subclass `KimiLinearModel`, borrow `_pack_mxfp4_blocks` from `deepseek.py`.

### Phase 5 - vision (optional)
mtmd subclass of the kimik25 graph.

### 8.1 How to validate without 1.5 TB of RAM

You cannot smoke-test a 2.8 T model on a workstation, so do not try. Instead:

1. **Build a tiny K3.** Write a script that emits a randomly-initialised model
   with the same `config.json` structure but `hidden_size=256`,
   `num_hidden_layers=13` (so AttnRes pushes at layers 0 and 12),
   `num_experts=8`, `routed_expert_hidden_size=128`. A few hundred MB.
2. **Dump reference activations** by running the HF modeling code on it.
3. **Compare per-layer** against llama.cpp using the `cb()` callback that
   `kimi-linear.cpp` already sprinkles through the graph
   (`cb(cur, "attn_norm", il)`).

This catches the AttnRes push-layer semantics (section 4.1), the `A_log`
transpose, and the situ betas - three bugs that all produce a model that runs
and generates fluent nonsense.

Only after the tiny model matches do you spend two days converting 1.4 TB.

---

## 9. Hardware reality

At ~898 GB the model still does not fit in any single machine's RAM that is
likely on hand. Realistic options:

- A server with 1 TB+ RAM: works, CPU-bound.
- mmap from fast NVMe with less RAM: works, slowly, and MoE sparsity helps more
  than usual here (16/896 experts per token).
- GPU offload of attention + shared experts only (~50 GB at Q6_K) with routed
  experts on CPU: this is the shape that actually makes K3 usable, and it maps
  onto the existing `--n-cpu-moe` style flags.

That last configuration is worth designing toward from the start.

---

## 10. Upstreaming

Per `AGENTS.md`, this is a large change introducing a new pattern (the AttnRes
stack in the layer loop). If it is ever proposed upstream it needs a discussion
first, and the contributor must be able to defend every line without AI help -
which is the reason this document exists in this much detail.

Relevant prior art:
- https://github.com/ggml-org/llama.cpp/discussions/26041 (K3 pre-release analysis)
- https://github.com/ggml-org/llama.cpp/pull/18381 (Kimi-Linear)
- https://github.com/ggml-org/llama.cpp/pull/18755 (Kimi-Linear, backend agnostic)
