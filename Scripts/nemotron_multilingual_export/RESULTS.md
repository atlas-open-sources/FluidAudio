# Results: 320 ms and 80 ms exports (2026-08-29 runs, published 2026-10-07)

Published (private for now):
https://huggingface.co/mochiexists528/nemotron-3.5-asr-streaming-0.6b-coreml-lookahead

| folder | source variant | encoder | encoder size | avg WER | en WER | zh CER | RTFx |
|---|---|---|---|---|---|---|---|
| `multilingual/320ms` | `out/320ms-6bit` | 6-bit k-means LUT, per-grouped-channel g16 | 431 MB | 13.17% | 8.58% | 17.76% | 36.4x |
| `multilingual/80ms` | `out/80ms-ane` | int8 linear, per-channel, CPU_AND_NE | 566 MB | 14.79% | 11.01% | 18.58% | 8.8x |

FLEURS, 100 utterances each of `en_us` + `cmn_hans_cn` (zh scored as CER),
`fluidaudiocli nemotron-multilingual-benchmark`, MacBook Pro M4 Pro. Raw JSON and
logs for every variant are in `benchres/` (the `modelDir` paths inside point at
the original `~/src/nemotron-export/out/` working directory).

## Why these two

- 320 ms: the 6-bit encoder has the best WER, the highest speed and the smallest
  size of every 320 ms variant (fp32 13.49%, fp16 13.70%, int8 13.49%, int8/ANE
  13.37%). 4-bit LUT breaks the encoder (55% WER grouped, 100% with per-channel
  scale).
- 80 ms: int8/ANE is the fastest (8.8x vs fp16 6.7x, fp32/int8-CPU ~2.5x) with WER
  within noise of the best (int8-CPU 14.64%). No 6-bit 80 ms export was made.
- Latin (pruned vocab) 320 ms: 9.14% en WER vs 8.96% multilingual at the same
  speed, so not published.
- 80 ms numbers predate the fused `decoder_joint`; on 320/560 ms ANE the fused
  path gave identical WER.

## Files here

- `helpers/`: the working-directory scripts that produced the matrix
  (`run_*.sh` drivers, `quantize_encoder.py`, `benchmark_wer.py`, NeMo
  reference/probe/test scripts). The drivers assume `~/src/nemotron-export` with
  a `.venv` and a debug `fluidaudiocli` build at `~/src/FluidAudio`.
- `benchres/`: benchmark JSON + logs, one per variant.
