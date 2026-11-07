#include "models.h"

// Kimi-K3 / K3.1 vision: the Kimi-K2.5 MoonViT3d encoder with K3's config flags
// (reference: vLLM vllm/model_executor/models/kimi_k25_vit.py)
//   - RMSNorm everywhere, no biases
//   - head_dim = qkv_hidden_size / n_head (1536 / 12 = 128), wider than n_embd / n_head
//   - bilinear position embedding interpolation (K2.5 is bicubic)
//   - PatchMergerMLPV2 projector: no pre-norm, linear_1 -> gelu (erf) -> linear_2 -> RMSNorm
ggml_cgraph * clip_graph_kimik3::build() {
    ggml_tensor * pos_h = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_patches);
    ggml_set_name(pos_h, "pos_h");
    ggml_set_input(pos_h);

    ggml_tensor * pos_w = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_patches);
    ggml_set_name(pos_w, "pos_w");
    ggml_set_input(pos_w);

    ggml_tensor * learned_pos_embd = resize_position_embeddings_3d(GGML_SCALE_MODE_BILINEAR);

    // Q / K are permuted from interleaved to split 2D RoPE at conversion time, as for K2.5
    auto add_pos = [&](ggml_tensor * cur, const clip_layer &) {
        return build_rope_2d(ctx0, cur, pos_w, pos_h, hparams.rope_theta, false);
    };

    ggml_tensor * inp = build_inp();
    inp = ggml_add(ctx0, inp, learned_pos_embd);

    ggml_tensor * cur = build_vit(inp, n_patches, NORM_TYPE_RMS, hparams.ffn_op, nullptr, add_pos);
    cb(cur, "vit_out", -1);

    // sd2_tpool patch merger (a single frame, so the temporal mean is a no-op)
    cur = build_patch_merge_permute(cur, hparams.n_merge);

    cur = build_ffn(cur,
        model.mm_1_w, nullptr,
        nullptr, nullptr,
        model.mm_2_w, nullptr,
        FFN_GELU_ERF,
        -1);

    // post_norm eps is projector_ln_eps, which the converter checks is 1e-5
    cur = build_norm(cur, model.mm_post_norm_w, nullptr, NORM_TYPE_RMS, 1e-5f, -1);
    cb(cur, "proj_out", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}
