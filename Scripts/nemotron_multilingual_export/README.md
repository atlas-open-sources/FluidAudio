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
  bundles record `[42, 13]` / cache 42 — their exact checkpoint lineage has not been proven by these metadata alone. Bundles from this script are NOT
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


## Local qualification mode

This branch requires `--checkpoint /absolute/path/model.nemo`,
`--checkpoint-sha256 <verified digest>` and `--validation-audio <real mono16k WAV>`.
All four base components restore that local checkpoint without resolving a remote
model name. The fused decoder requires the same hash and a passed destination
receipt. Existing output directories are rejected; failed runs remain inspectable
and retries need a fresh directory. Do not reuse partial bundles.

The base exporter seeds Python/NumPy/Torch, records exact installed package versions,
source hashes, arguments, input/output byte hashes, and compares each CoreML component
with its PyTorch wrapper on real-audio-derived inputs. Integer outputs must match;
float normalized RMSE must be at most0.05. This provisional single-input numerical
gate is **not** an end-to-end WER, multi-chunk streaming or language-switching gate.
Captured validation NPZ arrays are derived from the recorded real WAV.

Int8 and LUT derivative scripts require an unchanged, passed FP16 parent receipt,
verify every recorded parent file hash, and retain parent/checkpoint lineage. They
compare the compressed encoder against the same captured real-input reference.
No claim about quality, power or ANE placement follows merely from successful
conversion or weight size. Benchmark matched variants on held-out human recordings.

### Warmed-state verification and explicit weight-storage experiments

`verify_streaming_rollout.py` checks complete real mono16k recordings against the
same local checkpoint without rewriting model packages or old receipts. Supply
`--model-dir`, `--checkpoint`, one or more `--audio` paths, and a fresh `--output`
directory. `--prompt auto` is the default; unknown prompt names fail. Recordings
must span at least two chunks. Use the isolated locked environment and set
`HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=disabled`.

The verifier uses CPU_ONLY for every component and records hardware, macOS,
source hashes, tracked Git diff hash, untracked paths, model/input hashes and
per-step numerical checks. It follows the Swift fixed-shape chunk convention:
zero pre-encode history on the first chunk, actual trailing mel history thereafter,
and zero audio padding at the tail. This is explicitly not the distinct NeMo
variable-first-chunk streaming API. Encoder checks separate teacher-forced source
caches from independently fed-back CoreML caches and preprocessing. Source greedy
tokens exercise nonblank decoder inputs and warmed recurrent state; joint checks
cover all decoded frames. Independent CoreML greedy tokens must match. If a fused
model is present, its receipt/hashes are verified and warmed fused outputs are
also checked. Symbol-limit hits or absent nonblank/nonzero-state coverage fail.

A rollout pass remains a bounded numerical/token-conformance observation, not
multilingual WER acceptance. The 5% normalized-RMSE threshold is provisional and
unchanged; a numerical failure must not be described as proof of WER degradation.
Keep separate held-out human-transcript evaluations and compute-unit comparisons.

Both weight compression commands accept `--parent-precision FLOAT32` explicitly
(the default remains FLOAT16), along with `--compute-units CPU_ONLY`. The parent's
receipt must have passed and declare exactly that precision. This allows a matched
FP32-parent Int8/6bit storage experiment without overriding a failed FP16 gate.
Core ML weight compression preserves the parent's floating computation graph;
compressed weight storage and floating compute precision are separate receipt
fields. Verify resulting MIL/compiled metadata as well as numerical outputs.
For 6bit grouped palettes, specify `--nbits 6 --granularity per_grouped_channel
--group-size 16`. These are experiment parameters, not quality recommendations.

Preprocessor, decoder, joint, tokenizer, and optional fused decoder companions
must remain byte-identical to their matched parent. The fused receipt must bind
the exact parent receipt. Derivative outputs use fresh directories and retain
failures. An artifact may be compiled for descriptive speech testing after a
numerical failure only if its compilation receipt clearly says `compiled_only`
and preserves the source failure; this does not change qualification status.
