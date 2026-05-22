from __future__ import annotations

import re

from typing import Callable, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf, logger


@ModelBase.register("CohereForCausalLM")
class CommandR2Model(TextModel):
    model_arch = gguf.MODEL_ARCH.COMMAND_R

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # max_position_embeddings = 8192 in config.json but model was actually
        # trained on 128k context length
        # aya-23 models don't have model_max_length specified
        self.hparams["max_position_embeddings"] = self.find_hparam(["model_max_length", "max_position_embeddings"])

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_logit_scale(self.hparams["logit_scale"])
        self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.NONE)


@ModelBase.register("Cohere2ForCausalLM")
class Cohere2Model(TextModel):
    model_arch = gguf.MODEL_ARCH.COHERE2

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        self.gguf_writer.add_logit_scale(self.hparams["logit_scale"])
        self.gguf_writer.add_sliding_window(self.hparams["sliding_window"])
        self.gguf_writer.add_vocab_size(self.hparams["vocab_size"])

        rotary_pct = self.hparams["rotary_pct"]
        hidden_size = self.hparams["hidden_size"]
        num_attention_heads = self.hparams["num_attention_heads"]
        self.gguf_writer.add_rope_dimension_count(int(rotary_pct * (hidden_size // num_attention_heads)))
        self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.NONE)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # Cohere2 runtime in llama.cpp expects no bias tensors;
        # the actual weight only contains 0-value tensors as bias, we can skip them
        if name.endswith(".bias"):
            if torch.any(data_torch != 0):
                raise ValueError(f"Bias tensor {name!r} is not zero.")
            logger.debug(f"Skipping bias tensor {name!r} for Cohere2 conversion.")
            return

        yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("Cohere2MoeForCausalLM")
@ModelBase.register("Cohere2VisionForConditionalGeneration")
class Cohere2MoeModel(TextModel):
    model_arch = gguf.MODEL_ARCH.COHERE2_MOE

    # accumulated per-expert weights before merging into a 3D tensor
    _experts: list[dict[str, Tensor]] | None = None

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        # base class merges text_config into hparams, so all keys are at the top level
        p = self.hparams
        self.gguf_writer.add_logit_scale(p.get("logit_scale", 1.0))
        self.gguf_writer.add_sliding_window(p["sliding_window"])
        self.gguf_writer.add_vocab_size(p["vocab_size"])
        # expert_count and expert_used_count are already set by the base class
        self.gguf_writer.add_expert_shared_count(p["num_shared_experts"])
        self.gguf_writer.add_expert_feed_forward_length(p["intermediate_size"])
        self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)
        self.gguf_writer.add_expert_weights_norm(p.get("norm_topk_prob", False))

        # head_dim is explicit in the config (128); hidden_size/n_heads = 4096/128 = 32 would be wrong
        rotary_pct = p.get("rotary_pct", 1.0)
        head_dim = p.get("head_dim", p["hidden_size"] // p["num_attention_heads"])
        self.gguf_writer.add_rope_dimension_count(int(rotary_pct * head_dim))
        self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.NONE)

        swa_pattern = p.get("layer_switch", p.get("_sliding_window_pattern", 4))
        self.gguf_writer.add_sliding_window_pattern(swa_pattern)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        # apply base class filtering first (strips "language_model." prefix)
        result = super().filter_tensors(item)
        if result is None:
            return None
        name, gen = result
        # skip vision encoder and multimodal projector tensors for text-only GGUF
        if name.startswith(("model.vision_tower.", "model.multi_modal_projector.")):
            return None
        return name, gen

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # skip zero bias tensors (all biases in Cohere2 are zero-valued)
        if name.endswith(".bias"):
            if torch.any(data_torch != 0):
                raise ValueError(f"Bias tensor {name!r} is not zero.")
            return

        # experts are stored as 128 individual 2D tensors; merge them into one 3D tensor
        if re.search(r'\.mlp\.experts\.\d+\.', name):
            n_experts = self.hparams["num_experts"]
            assert bid is not None

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch

            if len(self._experts[bid]) >= n_experts * 3:
                for w_name in ["down_proj", "gate_proj", "up_proj"]:
                    datas: list[Tensor] = []
                    for xid in range(n_experts):
                        ename = f"model.layers.{bid}.mlp.experts.{xid}.{w_name}.weight"
                        datas.append(self._experts[bid].pop(ename))
                    merged = torch.stack(datas, dim=0)
                    merged_name = f"model.layers.{bid}.mlp.experts.{w_name}.weight"
                    yield from super().modify_tensors(merged, merged_name, bid)
            return

        yield from super().modify_tensors(data_torch, name, bid)
