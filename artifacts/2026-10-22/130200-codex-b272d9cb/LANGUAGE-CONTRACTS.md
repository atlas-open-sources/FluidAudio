status: green — contract fixes and real-logit reader proof complete; language-quality/speculative-runtime acceptance remains open

# Language API and observation-path packet

Branch `codex/language-contracts-20261023`, worktree `worktrees/language-contracts`, base `5770ec3`.

Commits:
- `8dd7c85`: separate decoded/inferred language, explicit allowlist aliases, validate logits layouts and capabilities.
- `a692cc3`: schema-2 replay adapter and explicit allowlist option.

Both patches are retained under `patches/`. These depend on the existing fork feature stack; they are not clean additions directly atop pristine upstream. No push/submission/app integration/default change occurred.

## Fixed

1. First/latest decoded getters no longer consume soft estimates. `inferredLanguage()` and atomic `languageObservation()` expose that evidence independently.
2. Forced prefix is conditioning, not decoder evidence. Reading partial text no longer mutates detected-language state; finish does not reconstruct language outside live recorder guards.
3. Documented `eng` now matches `en-US`; explicit ISO aliases include Dutch and benchmark languages. Reset preserves allowlist and clears all observations.
4. Reader validates rank, singleton axes, positive strides, frame bounds, fp16/fp32 dtype, finite values, token bounds, and offset overflow. Negative or out-of-range IDs cannot address memory. Both ordinary and batched layouts share the same reader.
5. Argmax-only live decode reports unavailable logits explicitly, including the speculative drain's fused branch. Decoder tags still remain usable; the soft estimate is not manufactured.
6. Replay schema2 adds independent language observation with provenance and records allowlist and timing semantics.

## Evidence

`final-contract-tests.log`: **56 tests pass, no skips**. Includes six new contract/layout/captured-real-logit tests,44 existing multilingual tests,six replay option tests. Tests labelled model-free do not claim speech accuracy. `git diff --check` passes. Lint reports only inherited uppercase K at manager680 and Decode57; new files/changed additions are formatted.

Actual human-recorded English audio `en_us_0305.wav` (4.98seconds), historical1120ms model, CPU replay completed. Final text:

> He referred to the rumors as political chatter and silliness

Snapshot: decoded first/latest=null; inferred=en-US; source=soft_logits; capability=available; path=ordinary_logits. This directly demonstrates why the old API mislabeled provisional evidence. It does not prove correctness for every English recording.

`real-en-logits.json` is an actual13088-value joint output from that real decode, with39 configured language tokens. Temporary capture-only instrumentation is retained in `capture-instrumentation.txt`; source was restored before final build. `real-logit-receipt.json` records audio, model files, logit capture and instrumented binary SHA256. The final production binary SHA is recorded separately in `receipt.json`. No synthetic audio or mock models were used.

The real vector was read in ordinary and repacked batched layouts with both fp32 and fp16 representations: candidates and vocabulary maximum match the corresponding expected float representation exactly. Repacking a real output validates reader equivalence; it does **not** exercise a speculative model package or establish fp16 model quality. Noncontiguous float16/float32 stride and malformed-layout tests use explicitly labelled tensor fixtures.

## Replay API for Cat benchmarks

Build:

```sh
swift build --jobs 4 -Xswiftc -DQUALIFICATION_FORK
```

Run:

```sh
.build/debug/fluidaudiocli nemotron-qualification-replay \
  --model-dir "$MODEL_DIR" --input "$RECORDING_WAV" --output "$FRESH_JSONL" \
  --chunk-ms 20 --compute-units cpu --expected-languages eng,nl
```

Omit allowlist for open mode. `--language` remains model conditioning. No inference on/off toggle was added: decoded and inferred channels can be evaluated separately from the same run. Existing stabilization thresholds and observation sampling policy are unchanged.

Chunk and completion events retain `first_detected_language`, `latest_detected_language`, source audio times, elapsed time and partial/final text. New object:

```json
{"language_observation":{"decoded_first":null,"decoded_latest":null,"inferred":"en-US","source":"soft_logits","capability":"available","path":"ordinary_logits"}}
```

These are chunk-availability snapshots. They may omit transitions inside a chunk and must not be sold as precise acoustic switch timing, diarization, or speaker attribution. Retained inferred values can be stale: capability describes the latest observer, not freshness of the inferred value. Reset clears observations.

## Deliberately unresolved

- Ordinary observes every decode step; speculative observes first nonblank frame only, with no soft update in drain or blank-only windows. Sampling differs even if the reader is numerically equivalent. Actual speculative model execution and quality require a dedicated matrix before upstreaming the early speculative feature.
- Fused capability hook has model-free state coverage and source-path coverage. This packet has not run an actual fused-argmax model. Benchmark suite may add that evidence separately.
- No claims about natural switch latency, speaker diversity, allowlist safety, calibration, or speedup. Parent language benchmark suite owns actual multilingual/engineering matrix and output adapters.
- No whole-device/iOS/hardware matrix. Swift package build and focused tests only.
- Known raw source API changed intentionally: callers that previously used decoded getters for inferred UI must migrate deliberately. This should be reviewed as an API proposal together with the originating fork language feature.
- First real-logit fixture is external evidence rather than committed repository test resource. Optional fixture test skips explicitly if its environment path is absent; retain provenance if distributing it.

## Handoff

`Documentation/NemotronLanguageObservations.md` provides reusable API/migration, CLI and verification instructions. Parent/language_bench_suite received final CLI binary path/hash before starting its matrix. No jobs owned by this worker remain running at report time. Worktree is clean; no external submission.
