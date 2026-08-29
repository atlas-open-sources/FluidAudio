# Nemotron Multilingual Streaming — CoreML export (parameterised lookahead)

Exports `nvidia/nemotron-3.5-asr-streaming-0.6b` (the prompted multilingual
checkpoint, `EncDecRNNTBPEModelWithPrompt`) to the 4-component CoreML bundle
layout that `StreamingNemotronMultilingualAsrManager` consumes:

```
<out>/preprocessor.mlpackage   audio -> mel
<out>/encoder.mlpackage        mel + cache + prompt_id -> encoded + new cache
<out>/decoder.mlpackage        token + LSTM state -> decoder_out + new state
<out>/joint.mlpackage          encoder + decoder -> logits
<out>/metadata.json            config incl. prompt_dictionary / lang_tag_token_ids
<out>/tokenizer.json           id -> piece map (includes "<blank>" at blank_idx)
```

The fused `decoder_joint*` fast-path models are NOT exported here; the Swift
manager falls back to separate decoder+joint when they are absent.

## Why this exists

The published FluidInference bundles ship only two latency tiers (1120 ms,
560 ms), but the checkpoint itself supports four lookahead settings
(`supported_num_lookahead_tokens: [13, 6, 3, 0]`, 80 ms per token at
subsampling factor 8):

| `--lookahead` | chunk | note |
|---|---|---|
| 13 | 1120 ms | shipped tier |
| 6  | 560 ms  | shipped tier |
| 3  | 320 ms  | NVIDIA's own `default_num_lookahead_tokens` — never exported |
| 0  | 80 ms   | minimum latency |

This exporter derives every shape from `encoder.streaming_cfg` after
`setup_streaming_params(att_context_size=[left, lookahead])` instead of
hardcoding the 1120 ms numbers, so any supported lookahead exports correctly.

## Differences vs the shipped bundles (deliberate / known)

- **Prompt fusion**: the language prompt is applied to the encoder OUTPUT in
  NeMo (`PromptStreamingMixin._apply_prompt_to_encoded`: one-hot concat +
  `prompt_kernel` linear). Like the shipped `encoder.mlmodelc`, we fuse this
  into the exported encoder, which therefore takes a `prompt_id` int32 [1]
  input.
- **Checkpoint lineage**: today's public checkpoint has left context 56
  (`att_context_size [56, N]`, channel cache `[1, 24, 56, 1024]`); the shipped
  bundles record `[42, 13]` / cache 42 — they were built from an older
  checkpoint that is no longer published. Bundles from this script are NOT
  cache-compatible with the shipped ones; each bundle's `metadata.json` is the
  source of truth and the Swift side reads shapes from it.
- **Full vocab (13,087)**: no vocab pruning. The shipped `latin/` family is
  pruned to 2,828 tokens; pruning is a separate post-processing step this
  script does not perform.
- **FP32, unquantized**: use `--precision FLOAT16` or a later int8 pass for
  size/speed parity with shipped bundles (shipped encoders are int8).

## Environment (pins that matter)

```bash
uv venv --python 3.11 .venv
uv pip install "nemo_toolkit[asr]==3.0.0" "coremltools==9.0" \
    "torch==2.7.0" "numpy==2.2.6" typer soundfile
```

- **`numpy==2.2.6` is THE load-bearing pin.** numpy >= 2.3 enforces the old
  deprecation that size-1 ndarrays with ndim > 0 no longer convert to Python
  scalars; coremltools 9.0's torch frontend still does `int(np.array([...]))`
  in `_cast` (ops.py), so the encoder conversion dies ~272/3997 ops in with
  `TypeError: only 0-dimensional arrays can be converted to Python scalars`.
  This looks like NeMo/coremltools graph incompatibility but is purely numpy.
- torch 2.7.x: coremltools 9.0's max tested torch line.

## Usage

```bash
.venv/bin/python convert_nemotron_streaming.py --output-dir out/320ms --lookahead 3
.venv/bin/python convert_nemotron_streaming.py --output-dir out/80ms  --lookahead 0
```

Test against the real Swift pipeline without touching the model cache:

```bash
swift run -c release fluidaudiocli nemotron-multilingual-transcribe \
    --model-dir out/320ms --input audio-16k.wav --language en-US
```

## Known warts

- At the end of the encoder export an `E5RT ... slice_by_index: zero shape
  error` line appears: the ANE compiler cannot shape-propagate the graph when
  coremltools instantiates the model with `CPU_AND_NE`. The artifact saves
  fine; at runtime the encoder falls back to GPU/CPU. Not yet root-caused.
- Punctuation sparsity at low-latency tiers on long sessions (upstream issue
  #687) presumably worsens below 560 ms — measure before shipping a default.
