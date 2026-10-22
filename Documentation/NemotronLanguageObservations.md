# Nemotron language observations

The multilingual streaming manager exposes two independent signals:

- `detectedLanguage()` / `currentDetectedLanguage()`: the first / latest accepted language-tag token **emitted by the live decoder**. These getters honor `setExpectedLanguages`. They exclude rescue replay, user-forced prefixes, and soft logit inference.
- `inferredLanguage()`: the latest provisional estimate committed by the existing soft-language evidence rule. This can disagree with decoded tags, can be wrong, and is not a calibrated probability.

`languageObservation()` reads both atomically. Its `source` is `soft_logits` when an inferred value exists and `none` otherwise. `capability` and `path` describe the latest live observer invocation, not all paths exercised during the session:

| Capability | Meaning |
| --- | --- |
| `not_observed` | No observer path has run since construction/reset. Not a claim of model support. |
| `available` | A supported logits tensor was read. The evidence gate may still reject its values. |
| `unsupported_tensor` | Rank, stride, dtype, frame bounds or finite-value requirements failed. |
| `unavailable_no_language_tags` | No configured language tokens could be mapped. |
| `unavailable_argmax_only` | The current fused path emits token IDs without logits. |

An old inferred value is retained until reset or a new committed estimate. An unavailable capability does not erase a previously observed estimate. Consumers must not treat a retained value as a fresh measurement. Use replay chunk availability times; do not infer acoustic switch boundaries from them.

## Compatibility

The previous fork populated decoded getters from soft estimates, seeded them from forced prompts, and could seed first-language state when reading a transcript. Those behaviors contradicted their decoder-tag documentation. Applications relying on early language UI must deliberately read `inferredLanguage()` or the snapshot and label that signal provisional. Reading a partial transcript is now side-effect-free for language state.

Allowlist matching uses case-insensitive primary subtags (`en-US`, `en_US`, `en`) and explicitly supports ISO 639-2 aliases `eng`, `nld/dut`, `spa`, `fra/fre`, `deu/ger`, `zho/chi`, `jpn`, `kor`. Other primary subtags are preserved, not guessed. The allowlist survives reset; observations do not. Restricting an allowlist can hide real unexpected speech and is a separately evaluated experiment.

## Replay

Build with `swift build --jobs 4 -Xswiftc -DQUALIFICATION_FORK`, then:

```sh
.build/debug/fluidaudiocli nemotron-qualification-replay \
  --model-dir "$MODEL_DIR" --input "$RECORDING_WAV" --output "$FRESH_JSONL" \
  --chunk-ms 20 --compute-units cpu --expected-languages en,nl
```

Omit `--expected-languages` for open language detection. `--language` conditions the decoder; it is not inferred evidence. Schema 2 preserves existing chunk/complete timing and text fields and adds `language_observation` with `decoded_first`, `decoded_latest`, `inferred`, `source`, `capability`, `path`. Start events record the prompt, allowlist and timing semantics. Offline causal replay is not live microphone latency.

## Verification and remaining limits

Run model-free API/tensor tests:

```sh
swift test --jobs 4 -Xswiftc -DQUALIFICATION_FORK \
  --filter 'NemotronLanguageObservationTests|NemotronMultilingualTests|NemotronQualificationReplayTests'
```

The optional `testCapturedRealLogitsAcrossLayouts` requires `FLUIDAUDIO_LANGUAGE_LOGITS_FIXTURE` pointing to an actual captured logit JSON (`values`: vocabulary floats, `languages`: token ID to language map). It skips explicitly when no capture is provided. Such a capture must retain model/audio/runtime hashes and the capture method. Repacking one real vector into batched and float16 tensors proves reader-layout equivalence, not a speculative model's execution or fp16 model accuracy.

The ordinary observer samples each decode step. The existing speculative observer samples the first nonblank frame, omits blank-only windows, and does not sample soft logits during drain. This change does not alter those policies or the evidence thresholds. Actual speculative export timing, fused export behavior, human-annotated natural language-switch quality, and performance regression remain separate qualification gates. Neither decoded tags nor inferred estimates identify speakers.
