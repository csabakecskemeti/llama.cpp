# Gemma 4 12B-IT Conversion Patch

Two fixes are needed to successfully convert `google/gemma-4-12B-it` (and
other Gemma 4 variants that set `use_double_wide_mlp=true`) to GGUF with
llama.cpp's `convert_hf_to_gguf.py`.

## Problem 1 - duplicate `feed_forward_length` key in GGUF output

### Root cause

`Gemma4Model.set_gguf_parameters()` in `conversion/gemma.py` delegates to
`super().set_gguf_parameters()`, which unconditionally writes the scalar value
of `intermediate_size` as the `feed_forward_length` metadata key.  When
`use_double_wide_mlp=True`, the code later tries to overwrite that key with a
per-layer array, which produces a duplicate-key warning and results in the
wrong (scalar) value being stored.

### Fix (`conversion/gemma.py`)

Before calling `super()`, temporarily pop `intermediate_size` out of
`self.hparams` when `use_double_wide_mlp` is set.  This prevents the base
class from writing the scalar key at all.  After `super()` returns, restore
the value so the per-layer array logic below can read it and write the correct
array-valued `feed_forward_length`:

```python
use_double_wide_mlp = self.hparams.get("use_double_wide_mlp", False)

_saved_ff = self.hparams.pop("intermediate_size", None) if use_double_wide_mlp else None

super().set_gguf_parameters()

if _saved_ff is not None:
    self.hparams["intermediate_size"] = _saved_ff
```

The `first_kv_shared_layer_idx` local variable was also moved inside the
`if use_double_wide_mlp` block where it is actually used.

## Problem 2 - crash loading the tokenizer

### Root cause

The Gemma 4 `tokenizer_config.json` stores `extra_special_tokens` as a plain
list of token strings (e.g. `["<|start_of_turn|>", ...]`) rather than the
`{name: token}` dict that `transformers.SpecialTokensMixin._set_model_specific_special_tokens`
expects.  This causes a `TypeError` (or silent misbehaviour) when
`AutoTokenizer.from_pretrained` is called inside `LlamaHfVocab`.

### Fix (`gguf-py/gguf/vocab.py`)

Monkey-patch `SpecialTokensMixin._set_model_specific_special_tokens` before
calling `AutoTokenizer.from_pretrained`.  The patch converts a list value to
the expected dict by using each token string (stripped of angle-bracket
decoration) as the key:

```python
try:
    from transformers.tokenization_utils_base import SpecialTokensMixin as _STM
    _orig_ssm = _STM._set_model_specific_special_tokens
    def _patched_ssm(self, special_tokens):
        if isinstance(special_tokens, list):
            special_tokens = {tok.strip("<|> "): tok for tok in special_tokens}
        return _orig_ssm(self, special_tokens=special_tokens)
    _STM._set_model_specific_special_tokens = _patched_ssm
except (ImportError, AttributeError):
    pass
```

The `try/except` ensures nothing breaks if the transformers API changes in a
future release.

## Files changed

| File | Change |
|------|--------|
| `conversion/gemma.py` | Avoid duplicate scalar `feed_forward_length` when `use_double_wide_mlp=True` |
| `gguf-py/gguf/vocab.py` | Patch tokenizer to accept list-format `extra_special_tokens` |

## Usage

Run conversion as usual after applying the patch:

```sh
python convert_hf_to_gguf.py \
    /path/to/google/gemma-4-12B-it \
    --outtype bf16 \
    --outfile gemma-4-12B-it-BF16.gguf
```
