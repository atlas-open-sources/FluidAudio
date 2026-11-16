# UPSTREAM-DELTA: rebase/v0.18.3-cat

Branch `rebase/v0.18.3-cat` = upstream `FluidInference/FluidAudio` tag `v0.18.3` (commit `c8b41f98ea`, 246 commits and
13 releases after v0.15.5) plus the Cat's fork commits. It replaces the Cat's pin
`d116b79fb5a9386a9eac67ecbc76230d2e524fc1` (v0.15.5 + 8 commits, branch `fix/open-mode-language-prior`).
Written 2026-11-17 by w-fluidaudio (Claude Opus, M4) from brief
`scratch/2026-11-16/transcription-translation-codex-briefs/fluidaudio-0183-rebase.md`.

## How it was built

Not a fresh replay of the 8 commits: the October qualified rebase `codex/rebase-qualified-20261021` (v0.17.7 + 38,
then our series with its conflicts already resolved and reviewed) was cherry-picked onto `v0.18.3` with `-x`. All 14
commits applied without a conflict; each commit message names its source sha.

## Our commits

| d116b79 series (v0.15.5) | on this branch | |
|---|---|---|
| 4f9eee09 currentDetectedLanguage() | 658ee5c9 | kept |
| ca6eb46e test: currentDetectedLanguage | 68ea9315 | kept |
| 92a228f9 early language via lang-tag logit peek | 67d08cfb | kept |
| 53557ffe expected-languages allowlist | 4916c255 | kept |
| db5cf62c early soft language on smart-speculative path | 109afb2e | kept |
| ac59e8be completedBytes/totalBytes on DownloadProgress | 07b038bf | kept (upstream DownloadProgress still has only fractionCompleted + phase) |
| d116b79f open auto-detect: Spanish not Slovenian | 336fcb5b | kept |
| 1911536b resolve compiler warnings | — | **dropped**, see below |

Added from the October qualification (not in d116b79, none upstream in v0.18.3; `git cherry v0.18.3` lists each `+`):

- bda0592b fix(memory): one-byte offsets for Int8 arrays
- f1b3b9f5 fix(cohere): scope FFT input pointer to array lifetime
- 2ab5a423 / a37a449c test(packaging): bundle and parse the medical ARPA fixture (+ format)
- 4d803fa3 build: select modern LP64 Accelerate declarations
- bdf558af fix(tts): handle Int8 enum cases explicitly
- 6bda79f0 fix: preserve live language and concurrent byte progress (follow-up to our language + bytes commits)

**Dropped 1911536b** (warnings-only, no behaviour). The October rebase replaced it with the focused fixes above.
Re-applied to v0.18.3 it conflicts in 6 files that upstream rewrote (Package.swift, OfflineEmbeddingExtractor batch
embedding, AudioConverter, StyleTTS2/Supertonic3 MultiArray, DiarizationBenchmarkUtils). The fork still builds with
38 warnings under `swift build`; the Cat consumes FluidAudio as a remote package, where SwiftPM suppresses dependency
warnings, and the Cat's LocalSpeechKit build reports 0 warnings. A fresh warnings commit can follow if wanted.

## What v0.15.5..v0.18.3 changes for the Cat

The Cat's call sites (LocalSpeechKit, speech-bench) compile against this branch with **no source change**: no public
API the Cat uses was removed or renamed. Behaviour changes on paths the Cat calls:

- **Offline diarizer keeps overlapping speech by default** (d28bde8d, #1081): `OfflineDiarizerConfig.default`
  now has `exclusiveSegments = false`. The Cat attributes each word to one speaker, so its pin branch sets
  `exclusiveSegments = true` explicitly (behaviour unchanged, with a test).
- Offline diarizer: pyannote-parity clustering (df1417ce), speaker caps (3ebb2859), silent frames excluded from the
  FBank mean (3e5a323b), short spans tiled (a5f70903), cancellation propagated (ce9f85e5), batched embeddings.
- Sortformer: streaming mel no longer emits phantom frames that drifted timestamps late (baa11f65); complete audio
  gets a prediction for every frame (41c53ce4).
- Nemotron multilingual streaming (the Cat's live engine): all-blank pause spans rescued (2ef3a0bc), partial
  callback publishes only rescue-settled text (038b50f9) and skips unchanged partials (e90e84f8), cancelled
  recognition waits to exit (6924debc), decode-time vocabulary biasing available (0dc46a3c), raw encoder-output
  callback (9a404d63).
- Parakeet TDT sliding window: seam repairs (b20d2098, f59050ed, 30379135), dropped opening speech recovered
  (8ff96513), monotonic stable prefix for token revisions (4d1b57bc).
- Downloads: resume instead of purge (96b25669), transient network errors keep the cache (19a1c9d6), stall
  watchdog (64760ba2), pinned immutable revisions for diarization artifacts (a6f1ca89) with legacy cache adoption
  (04e363c2).

New and available, not adopted by this move (later work per the speech research): SenseVoice fixes and confidences,
Parakeet Ultra / Nemotron Unified, Cohere Transcribe v3 decoder, Canary speech translation, Nemotron 3.5
Transcribe Lite, LS-EEND / Nemotron-3 diarization, Kokoro ANE / Pocket TTS / Phonon / Paradee voices, FSMN-VAD.

## Packaging changes to know

- New platforms in the manifest: visionOS 2, tvOS 17 (no effect on the Cat).
- **New prebuilt binary target `NemoTextProcessing`** (Rust text normalization, xcframework from
  `FluidInference/text-processing-rs` v0.3.1, checksum-pinned) and a `LuxTtsG2pResources` target. Both are linked by
  default. `Package@swift-6.2.swift` makes them opt-out traits (`NemoTextProcessing`, `TTS`, `Diarizer`). The Cat
  uses ASR + VAD + diarization only, so `traits: ["Diarizer"]` would keep the ~18 MB engine and the TTS code out of
  the app. Not done in the pin branch: every Cat package is `swift-tools-version: 6.0` (traits on a dependency need
  6.1+), and `scripts/ci/package-graph.py` re-emits remote dependencies without traits. Follow-up decision.

## Tests (M4, Xcode 27.2.1 toolchain, through scripts/support/lac-build)

- `swift build`: complete, 0 errors (38 dependency warnings, see above).
- `swift test --skip SpeakerEnrollmentTests` (upstream CI's split): XCTest 3249 executed, 119 skipped, 0 failures;
  Swift Testing 66 tests in 5 suites passed.
- `swift test --filter SpeakerEnrollmentTests`: 21 executed, 0 failures.
- Note: upstream's suite downloads models on first run (parakeet-ja, campplus, a Sortformer high-latency bundle,
  LS-EEND ami, online pyannote/wespeaker). They were removed from the shared cache after the run except the small
  online diarization files inside `speaker-diarization`.
