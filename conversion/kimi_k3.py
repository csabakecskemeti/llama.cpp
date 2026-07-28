from __future__ import annotations

from typing import Callable, Iterable, TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, gguf, logger
from .kimi_linear import KimiLinearModel


@ModelBase.register("KimiK3ForConditionalGeneration")
class KimiK3Model(KimiLinearModel):
    """Kimi K3: 2.8T MoE, 69 KDA + 24 gated MLA layers, latent MoE, attention residuals.

    The text tower reports itself as KimiLinearForCausalLM, so routing to this class
    is done by the top-level architecture name (see get_model_architecture).

    Routed experts ship as MXFP4 (compressed-tensors, group_size 32, e8m0 scales),
    which is byte-compatible with ggml MXFP4. They are repacked, never requantized.
    """

    model_arch = gguf.MODEL_ARCH.KIMI_K3

    # attention residual norm/proj pairs waiting to be fused, keyed by output name
    _res_pending: dict[str, Tensor]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._res_pending = {}

        # note: TextModel.__init__ already merged text_config into hparams
        self.moe_latent_size = self.hparams.get("routed_expert_hidden_size")
        self.kda_head_dim = self.hparams["linear_attn_config"]["head_dim"]
        self.n_experts = self.hparams["num_experts"]

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        # vision tower and projector belong to the mmproj file
        if name.startswith(("vision_tower", "mm_projector")):
            return None

        # strip the multimodal wrapper so the shared tensor maps apply
        if name.startswith("language_model."):
            name = name[len("language_model."):]

        return super().filter_tensors((name, gen))

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        if self.moe_latent_size is not None:
            self.gguf_writer.add_moe_latent_size(self.moe_latent_size)

        if (block_size := self.hparams.get("attn_res_block_size")) is not None:
            self.gguf_writer.add_attn_res_block_size(block_size)

        if (beta := self.hparams.get("activation_situ_beta")) is not None:
            self.gguf_writer.add_situ_beta(beta)
        if (linear_beta := self.hparams.get("activation_situ_linear_beta")) is not None:
            self.gguf_writer.add_situ_linear_beta(linear_beta)

        if (lower_bound := self.hparams["linear_attn_config"].get("gate_lower_bound")) is not None:
            self.gguf_writer.add_kda_gate_lower_bound(lower_bound)

    def prepare_tensors(self):
        super().prepare_tensors()
        # routed experts are written as raw MXFP4 blocks, label the file accordingly
        self._is_mxfp4 = True
        self.ftype = gguf.LlamaFileType.MOSTLY_MXFP4_MOE

    def _is_kda_layer(self, bid: int) -> bool:
        # linear_attn_config layer indices are 1-based
        return (bid + 1) not in self.hparams["linear_attn_config"]["full_attn_layers"]

    @staticmethod
    def _pack_mxfp4_blocks(weight: Tensor, scale: Tensor) -> np.ndarray:
        """Byte repack from compressed-tensors MXFP4 to ggml MXFP4. Lossless.

        Both formats are 32 values per block with one e8m0 uint8 scale. Only the
        nibble order differs: safetensors packs adjacent values as low/high nibbles,
        ggml puts values 0..15 in low nibbles and 16..31 in high nibbles.
        """
        # prepare_tensors() upcasts anything that is not f16/f32 to float32, so the
        # packed bytes arrive as floats. Values are 0..255 and survive that exactly,
        # so cast them back rather than reinterpreting the wider storage.
        packed = weight.contiguous()
        packed = packed if packed.dtype == torch.uint8 else packed.to(torch.uint8)

        scale_u8 = scale.contiguous()
        scale_u8 = scale_u8 if scale_u8.dtype == torch.uint8 else scale_u8.to(torch.uint8)

        out_features, packed_cols = packed.shape
        logical_cols = packed_cols * 2
        if logical_cols % 32 != 0:
            raise ValueError(f"MXFP4 source row has {logical_cols} values, expected a multiple of 32")

        n_blocks = logical_cols // 32
        if tuple(scale_u8.shape) != (out_features, n_blocks):
            raise ValueError(f"MXFP4 scale shape {tuple(scale_u8.shape)} does not match {(out_features, n_blocks)}")

        src = packed.reshape(out_features, n_blocks, 16)
        low = src & 0x0F
        high = (src >> 4) & 0x0F

        vals = torch.stack((low, high), dim=-1).reshape(out_features, n_blocks, 32)
        qs = vals[:, :, :16] | (vals[:, :, 16:] << 4)
        raw = torch.cat((scale_u8.unsqueeze(-1), qs.to(torch.uint8)), dim=-1)
        return raw.reshape(out_features, n_blocks * 17).cpu().numpy()

    def _fuse_res_weight(self, name: str, data_torch: Tensor, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        """Fold the attention-residual RMSNorm weight into its projection.

        _apply_attn_res only ever uses norm.weight and proj.weight as a product
        (score_weight = norm.weight * proj.weight), so one vector per site is enough.
        """
        base = name.replace("_res_norm", "_res").replace("_res_proj", "_res")

        other = self._res_pending.pop(base, None)
        if other is None:
            self._res_pending[base] = data_torch
            return

        fused = (data_torch.float().flatten() * other.float().flatten()).to(torch.float32)

        proj_name = base + "_proj"
        logger.info(f"{proj_name}: fused res norm and proj into a single {tuple(fused.shape)} vector")
        yield from super().modify_tensors(fused, proj_name + ".weight", bid)

    def _yield_mxfp4_experts(self, bid: int) -> Iterable[tuple[str, Tensor]]:
        """Stack and repack all routed experts of one layer, three projections at a time."""
        prefix = f"model.layers.{bid}.block_sparse_moe.experts"

        for wid, tensor_key in (("w1", gguf.MODEL_TENSOR.FFN_GATE_EXP),
                                ("w2", gguf.MODEL_TENSOR.FFN_DOWN_EXP),
                                ("w3", gguf.MODEL_TENSOR.FFN_UP_EXP)):
            data: np.ndarray | None = None

            for eid in range(self.n_experts):
                weight = self._experts[bid].pop(f"{prefix}.{eid}.{wid}.weight_packed")
                scale = self._experts[bid].pop(f"{prefix}.{eid}.{wid}.weight_scale")

                packed = self._pack_mxfp4_blocks(weight, scale)
                if data is None:
                    data = np.empty((self.n_experts, *packed.shape), dtype=packed.dtype)
                data[eid] = packed

            assert data is not None
            new_name = self.format_tensor_name(tensor_key, bid)
            shape = gguf.quant_shape_from_byte_shape(data.shape, gguf.GGMLQuantizationType.MXFP4)
            logger.info(f"{new_name}: repacked {self.n_experts} routed experts to MXFP4, shape = {{{', '.join(str(n) for n in reversed(shape))}}}")
            self.gguf_writer.add_tensor(new_name, data, raw_dtype=gguf.GGMLQuantizationType.MXFP4)

        return
        yield  # make this a generator

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.startswith("language_model."):
            name = name[len("language_model."):]

        # routed experts: buffer per layer, then stack and repack in one go
        if ".block_sparse_moe.experts." in name:
            assert bid is not None

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch

            # 896 experts x {w1,w2,w3} x {weight_packed,weight_scale}
            if len(self._experts[bid]) >= self.n_experts * 6:
                yield from self._yield_mxfp4_experts(bid)
            return

        # attention residuals: fold the norm weight into the projection
        if "_res_norm" in name or "_res_proj" in name:
            yield from self._fuse_res_weight(name.removesuffix(".weight"), data_torch, bid)
            return

        # A_log is [head_dim] in K3 and [n_head] in Kimi-Linear. Both are consumed by
        # the same graph code, which picks the broadcast axis from the shape.
        if name.endswith(".A_log"):
            data_torch = -torch.exp(data_torch)
            yield from super(KimiLinearModel, self).modify_tensors(data_torch, name, bid)
            return

        # self_attn.g_proj means the KDA output gate on KDA layers and the MLA output
        # gate on full-attention layers. Disambiguate before the shared tensor map.
        if name.endswith(".self_attn.g_proj.weight"):
            assert bid is not None
            if self._is_kda_layer(bid):
                name = name.replace(".g_proj.", ".g_kda_proj.")

        yield from super().modify_tensors(data_torch, name, bid)
