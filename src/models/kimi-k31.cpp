#include "models.h"

#include <atomic>

//
// Kimi-K3.1: Kimi-K3 with Gated DSA on the full-attention layers.
//
// The MLA weights are unchanged, each full-attention layer gains a DeepSeek-style
// lightning indexer that picks the top-k (2048) past tokens to attend to:
//   q     = indexer.attn_q_b(q_a_norm(attn_q_a(x)))       [n_head_idx, head_dim_idx]
//   k     = LayerNorm(indexer.attn_k(x)) (weight + bias)   [head_dim_idx]
//   w     = indexer.proj(x) / sqrt(head_dim_idx * n_head_idx)
//   score = sum_h w_h * relu(q_h . k), causal top-k
// unlike DeepSeek-V3.2 there is no RoPE and no Hadamard rotation in the indexer
// (reference: vLLM vllm/models/kimi_k3/nvidia/dsa.py)
//
// TODO: sparse attention. for now the indexer is loaded but not evaluated, and the
// MLA layers attend densely through the K3 graph. this is exact while the context
// holds no more than indexer_top_k tokens (the indexer then selects every token),
// which covers imatrix and perplexity runs at the usual chunk sizes
//

void llama_model_kimi_k31::load_arch_hparams(llama_model_loader & ml) {
    llama_model_kimi_k3::load_arch_hparams(ml);

    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k);

    // the indexer key LayerNorm uses rms_norm_eps
    hparams.f_norm_eps = hparams.f_norm_rms_eps;
}

void llama_model_kimi_k31::load_arch_tensors(llama_model_loader & ml) {
    llama_model_kimi_k3::load_arch_tensors(ml);

    LLAMA_LOAD_LOCALS;

    const int64_t q_lora_rank    = hparams.n_lora_q;
    const int64_t n_idx_head     = hparams.indexer_n_head;
    const int64_t n_embd_idx_hd  = hparams.indexer_head_size;

    GGML_ASSERT(q_lora_rank > 0 && "Kimi-K3.1 DSA indexer reads the q_lora latent");

    for (int i = 0; i < n_layer; ++i) {
        if (hparams.is_recr(i)) {
            continue;
        }
        auto & layer = layers[i];

        layer.indexer_k_norm   = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM,   "weight", i), {n_embd_idx_hd}, 0);
        layer.indexer_k_norm_b = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM,   "bias",   i), {n_embd_idx_hd}, 0);
        layer.indexer_proj     = create_tensor(tn(LLM_TENSOR_INDEXER_PROJ,     "weight", i), {n_embd, n_idx_head}, 0);
        layer.indexer_attn_k   = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_K,   "weight", i), {n_embd, n_embd_idx_hd}, 0);
        layer.indexer_attn_q_b = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_Q_B, "weight", i), {q_lora_rank, n_idx_head * n_embd_idx_hd}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_kimi_k31::build_arch_graph(const llm_graph_params & params) const {
    // dense fallback, see the TODO at the top
    static std::atomic<bool> warned{false};
    if (params.cparams.n_ctx_seq > hparams.indexer_top_k && !warned.exchange(true)) {
        LLAMA_LOG_WARN("%s: Kimi-K3.1 sparse attention is not implemented yet, the MLA layers attend densely; "
                       "output is exact up to %u tokens per sequence and approximate beyond (n_ctx_seq = %u)\n",
                       __func__, hparams.indexer_top_k, params.cparams.n_ctx_seq);
    }

    return std::make_unique<llama_model_kimi_k3::graph>(params);
}
