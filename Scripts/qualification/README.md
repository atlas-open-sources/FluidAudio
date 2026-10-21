# FluidAudio qualification

This suite separates source parity, model export comparisons and causal streaming observations. A successful command is not a universal model-quality pass.

## Frozen human corpus

`python3 Scripts/qualification/corpus.py --root <new-corpus-dir> --count 100`

Acquires eight languages from pinned FluidInference repackaging of the official FLEURS test split, deterministic SHA-ranked selection. Audio LFS hashes and byte sizes must match. Manifest records original transcript, license, source URLs, revision and per-file hashes; attribution is retained. This repackaging lacks speaker IDs. Do not claim speaker-disjointness beyond the source split or speaker-bootstrap confidence intervals. This held-out set is for evaluation; use a separate dev split for tuning. Do not tune language thresholds on it.

## Whole-utterance paired evaluation

Build release binaries for old fork d116b79, pristine upstream c79c410 and candidate16f6b21, applying exactly the same benchmark instrumentation. Record production revision and instrumentation separately. Keep original branches untouched. No defaults are changed by the harness.

```sh
python3 Scripts/qualification/run.py --binary <release-cli> --corpus <corpus-dir> \
  --model <model-dir> --output <new-run-dir> --revision <production-sha> \
  --instrumentation <instrumentation-sha> --prompt known
python3 Scripts/qualification/compare.py <baseline-run> <candidate-run> --output <comparison.json>
```

Run `--prompt auto` separately; a language-prompted run is not auto-detect proof. No forced-prefix switch is enabled. Use same compute units across paired runs. `compare.py --allow-model-change` explicitly labels a model-variant comparison; differences in checkpoint lineage, context size and decoder assets remain confounders, not quantization effects.

The wrapper hashes models/binary/corpus, checks exact rows, reference equality, finite timing, positive denominators and zero skipped samples. Input hashes are checked before and after. New output directories only. Timeouts terminate the process group. The CLI reports corpus-weighted per-language errors; legacy macro fields remain labeled for backward compatibility. WER and CJK CER are never pooled as a single quality score. Bootstrap intervals resample paired utterances, not speakers. They quantify sampling uncertainty, not timing repeatability or device variance. Repeat timing runs sequentially with fixed thermal/power conditions before speed claims.

## Causal streaming and model exports

The replay CLI feeds real audio in ordered short chunks and records partial transcript, process duration, first detected language and (only on forks exposing it) current detected language. Offline causal replay is not real-time device latency; pacing is explicit. Word/switch timestamps require independently reviewed reference annotation. Never treat a first-language getter as a latest-language getter.

Qualify export axes separately: checkpoint/revision, lookahead, encoder precision, decoder fusion, vocabulary, compute units and platform. Require NeMo versus CoreML same-checkpoint output tests before attributing a difference solely to quantization. Available cached320ms has full vocabulary and duplicated Latin/multilingual folders; folder labels are not model provenance. Historical80ms/6bit artifacts must be acquired and hashed before a fresh comparison. A missing variant is blocked, never skipped into a green matrix.

## Acceptance matrix

| Claim | Evidence needed |
|---|---|
| Int8 offsets/resource availability | Same regression fails baseline, passes fix; eligible modern SDK; older Swift build separately |
| Pointer lifetime | Compiler diagnostic, real-audio FFT parity, AddressSanitizer path coverage |
| Accelerate declarations | Actual compiler flag propagation and numerical controls, not mere manifest presence |
| Download bytes | Concurrent transfers, unknown totals, retry/cancel/resume and lifecycle tests |
| Model precision | Same checkpoint/decoder/lookahead; paired per-language WER/CER and repeat runtime/memory measures |
| New latency tier | Causal partial/final timings, quality tradeoff, cold/warm loads, device acceptance |
| Language heuristic | Auto-mode monolingual and natural-switch corpora; false flips/min, misses, stable delay, wrong-language time; per tier/path |
| Cat adoption | API compatibility, application builds, microphone/background/interruptions and physical-device behavior |

Default-change thresholds must be agreed and frozen before evaluation. No automatic default switch or global 'parity' verdict is generated. Full upstream CI matrix and reviewer acceptance remain separate gates.

## Additional corpora

- FLEURS CC-BY4: https://huggingface.co/datasets/google/fleurs — multilingual read speech, no natural switch labels.
- VoxPopuli data CC0, code/models NC: https://github.com/facebookresearch/voxpopuli#license — longer-form/accents, use independent loader.
- DRAL CC0: https://cs.utep.edu/nigel/dral/ — human bilingual conversations and re-enactments; matched translation pairs are not natural switch annotations.
- MLS CC-BY4: https://www.openslr.org/94/ — eight-language audiobooks; acquire evaluation subsets, not entire training archive.
- Common Voice CC0 has dataset-specific MDC acquisition/re-sharing terms: https://github.com/common-voice/cv-dataset and https://mozilladatacollective.com/terms/consumers — user-acquired cache only until terms handled.
- Bangor Miami: https://bangortalk.org.uk/ and https://bangortalk.org.uk/docs/miami_doc.pdf — natural EN/ES, original GPLv3+ and site conditions; group connected speakers and exclude untranscribed spans; human switch-boundary annotation needed.
- Excluded from unrestricted defaults: GigaSpeech audio noncommercial access terms; CS-FLEURS and CS-Dialogue noncommercial licenses. Metadata/code licenses do not clear audio rights.

Publish manifests and acquisition instructions by default, not third-party audio archives or private user recordings. Upstream posting requires separate explicit authorization.

## Harness tests

`python3 -m unittest discover -s Scripts/qualification -p 'test_*.py' -v`

Tests use receipt metadata, not fabricated audio or dummy models. Swift benchmark accumulator tests additionally cover invalid denominators, weighted aggregation and insertion rates above100%.
