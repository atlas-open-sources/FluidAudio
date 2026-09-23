import Foundation
import XCTest

@testable import FluidAudio

final class NemotronMultilingualTests: XCTestCase {

    // MARK: - Config

    func testDefaultConfigShape() {
        let config = NemotronMultilingualStreamingConfig()
        XCTAssertEqual(config.sampleRate, 16000)
        XCTAssertEqual(config.melFeatures, 128)
        XCTAssertEqual(config.chunkMelFrames, 112)
        XCTAssertEqual(config.chunkMs, 1120)
        XCTAssertEqual(config.preEncodeCache, 9)
        XCTAssertEqual(config.totalMelFrames, 121)
        XCTAssertEqual(config.vocabSize, 13087)
        XCTAssertEqual(config.blankIdx, 13087)
        XCTAssertEqual(config.cacheChannelShape, [1, 24, 56, 1024])
        XCTAssertEqual(config.cacheTimeShape, [1, 24, 1024, 8])
        XCTAssertEqual(config.defaultPromptId, 101)
        XCTAssertEqual(config.chunkSamples, 112 * 160)
    }

    // MARK: - Detected language (first vs current)

    /// `detectedLanguage()` keeps the FIRST tag (unchanged behavior); the new
    /// `currentDetectedLanguage()` follows the LATEST. Model-free — exercises the
    /// tag-recording logic directly, no weights needed.
    func testCurrentDetectedLanguageFollowsLatestWhileFirstStaysStable() async {
        let manager = StreamingNemotronMultilingualAsrManager()

        let firstBefore = await manager.detectedLanguage()
        let currentBefore = await manager.currentDetectedLanguage()
        XCTAssertNil(firstBefore)
        XCTAssertNil(currentBefore)

        // First tag anchors both.
        await manager.recordDetectedLanguage("en-US")
        // A later, different tag (a speaker switches language mid-stream).
        await manager.recordDetectedLanguage("es-419")

        let first = await manager.detectedLanguage()
        let current = await manager.currentDetectedLanguage()
        XCTAssertEqual(first, "en-US", "detectedLanguage() must remain the FIRST tag (no behavior change)")
        XCTAssertEqual(current, "es-419", "currentDetectedLanguage() must follow the LATEST tag")

        // Re-observing a tag keeps both stable (idempotent).
        await manager.recordDetectedLanguage("es-419")
        let firstAgain = await manager.detectedLanguage()
        let currentAgain = await manager.currentDetectedLanguage()
        XCTAssertEqual(firstAgain, "en-US")
        XCTAssertEqual(currentAgain, "es-419")
    }

    // MARK: - Soft (early) language pick

    /// English mass split across en-US/en/en-GB must AGGREGATE and beat a single
    /// higher-scoring es-ES token — the whole point of primary-subtag pooling.
    func testSoftLanguagePickAggregatesByPrimarySubtag() {
        let pick = StreamingNemotronMultilingualAsrManager.softLanguagePick(from: [
            ("en-US", 3.0), ("en", 2.5), ("en-GB", 2.0), ("es-ES", 3.2), ("fr-FR", -1.0),
        ])
        XCTAssertEqual(pick?.primary, "en", "pooled en mass should win over a single higher es token")
        XCTAssertEqual(pick?.piece, "en-US", "best-scoring piece within the winning primary")
        XCTAssertGreaterThan(pick?.confidence ?? 0, 0.5)
    }

    /// A clearly dominant single language yields near-1 confidence.
    func testSoftLanguagePickConfidentSingleLanguage() {
        let pick = StreamingNemotronMultilingualAsrManager.softLanguagePick(from: [
            ("es-419", 8.0), ("en-US", -2.0), ("fr-FR", -3.0),
        ])
        XCTAssertEqual(pick?.piece, "es-419")
        XCTAssertEqual(pick?.primary, "es")
        XCTAssertGreaterThan(pick?.confidence ?? 0, 0.9)
    }

    /// A near-tie between two primaries keeps confidence low (so the caller's
    /// threshold rejects it) — guards against flipping on noisy early frames.
    func testSoftLanguagePickAmbiguousIsLowConfidence() {
        let pick = StreamingNemotronMultilingualAsrManager.softLanguagePick(from: [
            ("en-US", 1.0), ("es-ES", 1.0),
        ])
        XCTAssertLessThan(pick?.confidence ?? 1, 0.6)
    }

    func testSoftLanguagePickEmptyReturnsNil() {
        XCTAssertNil(StreamingNemotronMultilingualAsrManager.softLanguagePick(from: []))
    }

    /// An expected-languages allowlist drops anomalies BEFORE the softmax: a
    /// spuriously high "tr-TR" is ignored when only en/es are expected, and the
    /// real (lower-scoring) "es-ES" wins with full confidence — not merely
    /// smoothed by debounce.
    func testSoftLanguagePickAllowlistDropsAnomalies() {
        let candidates: [(piece: String, logit: Float)] = [
            ("tr-TR", 9.0), ("es-ES", 2.0), ("en-US", -1.0),
        ]
        let unfiltered = StreamingNemotronMultilingualAsrManager.softLanguagePick(from: candidates)
        XCTAssertEqual(unfiltered?.primary, "tr", "without an allowlist the anomaly wins")

        let filtered = StreamingNemotronMultilingualAsrManager.softLanguagePick(
            from: candidates, allowed: ["en", "es"])
        XCTAssertEqual(filtered?.primary, "es", "allowlist drops tr-TR; es wins")
        XCTAssertGreaterThan(filtered?.confidence ?? 0, 0.9, "no anomaly left to dilute confidence")
    }

    /// An empty allowlist behaves like no allowlist (accept all).
    func testSoftLanguagePickEmptyAllowlistAcceptsAll() {
        let pick = StreamingNemotronMultilingualAsrManager.softLanguagePick(
            from: [("ja-JP", 5.0), ("en-US", 0.0)], allowed: [])
        XCTAssertEqual(pick?.primary, "ja")
    }

    // MARK: - Out-of-range frame gate (open auto-detect tagged Spanish as "sl")

    /// The shape of a real degenerate decode step from the multilingual 1120 ms
    /// export: blank dominates at about -950, every lang tag sits ~90 nats below
    /// it, and among the tags `sl-SL` leads by a wide margin — so the tag-only
    /// softmax says "sl" with confidence 1.0 regardless of the audio.
    private static let degenerateTags: [(piece: String, logit: Float)] = [
        ("sl-SL", -1039.0), ("nn-NO", -1052.0), ("th-TH", -1055.0),
        ("es-ES", -1061.0), ("en-US", -1063.0),
    ]

    /// NEGATIVE CONTROL: the tag-only pick alone is what produced the bug — it
    /// reports "sl" at near-1 confidence on the degenerate frame. If this ever
    /// stops holding, the gate tests below no longer prove anything.
    func testDegenerateFrameFoolsTheTagOnlyPick() {
        let pick = StreamingNemotronMultilingualAsrManager.softLanguagePick(from: Self.degenerateTags)
        XCTAssertEqual(pick?.primary, "sl")
        XCTAssertGreaterThan(pick?.confidence ?? 0, 0.99)
    }

    private typealias Manager = StreamingNemotronMultilingualAsrManager

    /// The frame reader rejects the degenerate frame outright (open mode).
    func testFrameRejectsOutOfRangeFrame() {
        XCTAssertNil(Manager.softLanguageFrame(from: Self.degenerateTags, frameMaxLogit: -950.0))
    }

    /// The SAME tag logits read in range (shifted so the gap is 20, like a real
    /// speech frame) are accepted — the gate keys on the tag-to-frame gap, not
    /// on the absolute logit level, so a constant shift cannot fool it.
    func testFrameAcceptsSameTagsInRange() {
        let shift: Float = 1039.0 - 32.0  // best tag at -32, frame max at -12 → gap 20
        let shifted = Self.degenerateTags.map { ($0.piece, $0.logit + shift) }
        let frame = Manager.softLanguageFrame(from: shifted, frameMaxLogit: -12.0)
        XCTAssertGreaterThan(frame?.distribution["sl"] ?? 0, 0.99, "in range the model's belief is taken as-is")
        XCTAssertEqual(frame?.bestPiece["sl"], "sl-SL")
    }

    /// The gap boundary is sharp: just inside the limit is read, just outside is
    /// not — the gate can go either way, it is not a check that always passes.
    func testFrameGapBoundaryIsSharp() {
        let limit = Manager.softLanguageMaxTagGap
        let tags: [(piece: String, logit: Float)] = [("es-ES", -40.0), ("sl-SL", -46.0)]
        XCTAssertNotNil(Manager.softLanguageFrame(from: tags, frameMaxLogit: -40.0 + limit - 0.5))
        XCTAssertNil(Manager.softLanguageFrame(from: tags, frameMaxLogit: -40.0 + limit + 0.5))
    }

    /// The allowlist still applies: a spuriously high out-of-list tag is
    /// dropped before the softmax, and the allowed language takes the mass.
    func testFrameKeepsAllowlistBehaviour() {
        let frame = Manager.softLanguageFrame(
            from: [("sl-SL", -20.0), ("es-ES", -25.0), ("en-US", -30.0)],
            frameMaxLogit: -12.0, allowed: ["en", "es"])
        XCTAssertNil(frame?.distribution["sl"])
        XCTAssertGreaterThan(frame?.distribution["es"] ?? 0, 0.99)
    }

    /// Non-finite logits are never a reading.
    func testFrameRejectsNonFiniteFrame() {
        XCTAssertNil(Manager.softLanguageFrame(from: [("es-ES", -20.0)], frameMaxLogit: .nan))
        XCTAssertNil(Manager.softLanguageFrame(from: [("es-ES", -.infinity)], frameMaxLogit: -12.0))
    }

    // MARK: - Evidence rule (when the live language commits or flips)

    private func feed(_ evidence: inout SoftLanguageEvidence, _ frames: [[String: Float]]) -> [String?] {
        frames.map { evidence.observe($0) }
    }

    /// Synthetic NEAR-TIE: es and sl at 0.48/0.47 frame after frame never
    /// commit — neither can reach the commit share, so the rule abstains
    /// instead of picking whichever happens to be a hair ahead.
    func testEvidenceAbstainsOnSustainedNearTie() {
        var evidence = SoftLanguageEvidence()
        let results = feed(&evidence, Array(repeating: ["es": 0.48, "sl": 0.47, "pt": 0.05], count: 60))
        XCTAssertTrue(results.allSatisfy { $0 == nil }, "a near-tie must never commit; got \(results.compactMap { $0 })")
    }

    /// Break the tie by a clear margin and the SAME rule commits — so the
    /// near-tie abstention above is the rule working, not a rule that never fires.
    func testEvidenceCommitsWhenTieBreaks() {
        var evidence = SoftLanguageEvidence()
        let results = feed(&evidence, Array(repeating: ["es": 0.75, "sl": 0.20, "pt": 0.05], count: 20))
        XCTAssertEqual(results.last, "es")
        let firstCommit = results.firstIndex { $0 != nil } ?? Int.max
        XCTAssertLessThanOrEqual(firstCommit, 10, "a clear language should commit within ~10 frames")
    }

    /// The observed failure: a four-frame burst of a confident wrong language
    /// inside steady Spanish. The old rule (two agreeing frames at >= 0.45)
    /// flipped on it; the evidence rule stays on Spanish.
    func testEvidenceRidesOutAShortConfidentBurst() {
        var evidence = SoftLanguageEvidence()
        let spanish: [String: Float] = ["es": 0.9, "en": 0.1]
        let burst: [String: Float] = ["nn": 0.95, "es": 0.05]
        _ = feed(&evidence, Array(repeating: spanish, count: 15))
        let during = feed(&evidence, Array(repeating: burst, count: 4))
        XCTAssertFalse(during.contains("nn"), "a 4-frame burst must not flip the language; got \(during)")
        let after = feed(&evidence, Array(repeating: spanish, count: 3))
        XCTAssertEqual(after.last, "es")
    }

    /// A SUSTAINED switch still flips — the rule delays, it does not freeze.
    func testEvidenceFollowsASustainedSwitch() {
        var evidence = SoftLanguageEvidence()
        _ = feed(&evidence, Array(repeating: ["en": 0.95, "es": 0.05], count: 20))
        let switched = feed(&evidence, Array(repeating: ["es": 0.95, "en": 0.05], count: 12))
        XCTAssertEqual(switched.last, "es")
    }

    /// Reset starts a turn from no evidence.
    func testEvidenceResetClearsShares() {
        var evidence = SoftLanguageEvidence()
        _ = feed(&evidence, Array(repeating: ["en": 1.0], count: 20))
        evidence.reset()
        XCTAssertNil(evidence.observe(["es": 1.0]))
        XCTAssertTrue(evidence.share.keys.allSatisfy { $0 == "es" })
    }

    func testConfigLoadFromMetadata() throws {
        // Stand-in metadata.json matching the multilingual build format.
        let json: [String: Any] = [
            "sample_rate": 16000,
            "mel_features": 128,
            "chunk_mel_frames": 112,
            "chunk_ms": 1120,
            "pre_encode_cache": 9,
            "total_mel_frames": 121,
            "vocab_size": 13087,
            "blank_idx": 13087,
            "encoder_dim": 1024,
            "decoder_hidden": 640,
            "decoder_layers": 2,
            "cache_channel_shape": [1, 24, 56, 1024],
            "cache_time_shape": [1, 24, 1024, 8],
            "num_prompts": 128,
            "default_prompt_id": 101,
            "prompt_dictionary": [
                "en-US": 0,
                "zh-CN": 4,
                "ja-JP": 10,
                "fr-FR": 12,
                "auto": 101,
            ],
            "lang_tag_token_ids": [1, 256, 397],
        ]
        let data = try JSONSerialization.data(withJSONObject: json)
        let tmpURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("multilingual_metadata_test_\(UUID().uuidString).json")
        try data.write(to: tmpURL)
        defer { try? FileManager.default.removeItem(at: tmpURL) }

        let config = try NemotronMultilingualStreamingConfig(from: tmpURL)
        XCTAssertEqual(config.numPrompts, 128)
        XCTAssertEqual(config.defaultPromptId, 101)
        XCTAssertEqual(config.promptDictionary["en-US"], 0)
        XCTAssertEqual(config.promptDictionary["zh-CN"], 4)
        XCTAssertEqual(config.promptDictionary["auto"], 101)
        XCTAssertEqual(config.langTagTokenIds, Set([1, 256, 397]))
    }

    // MARK: - promptId(forLanguage:)

    func testPromptIdDirectLookup() throws {
        let config = try makeConfig(
            promptDictionary: ["en-US": 0, "zh-CN": 4, "ja-JP": 10, "auto": 101]
        )
        XCTAssertEqual(config.promptId(forLanguage: "en-US"), 0)
        XCTAssertEqual(config.promptId(forLanguage: "zh-CN"), 4)
        XCTAssertEqual(config.promptId(forLanguage: "ja-JP"), 10)
    }

    func testPromptIdNilFallsBackToDefault() throws {
        let config = try makeConfig(promptDictionary: ["en-US": 0, "auto": 101])
        XCTAssertEqual(config.promptId(forLanguage: nil), 101)
        XCTAssertEqual(config.promptId(forLanguage: ""), 101)
    }

    func testPromptIdUnderscoreNormalization() throws {
        let config = try makeConfig(promptDictionary: ["en-US": 0, "auto": 101])
        // "en_us" should normalize to "en-US"
        XCTAssertEqual(config.promptId(forLanguage: "en_us"), 0)
        XCTAssertEqual(config.promptId(forLanguage: "EN-us"), 0)
    }

    func testPromptIdBareLanguageFallback() throws {
        let config = try makeConfig(promptDictionary: ["en": 7, "auto": 101])
        // "en-XX" should fall back to bare "en"
        XCTAssertEqual(config.promptId(forLanguage: "en-XX"), 7)
    }

    func testPromptIdUnknownLanguageReturnsDefault() throws {
        let config = try makeConfig(promptDictionary: ["en-US": 0, "auto": 101])
        XCTAssertEqual(config.promptId(forLanguage: "xx-YY"), 101)
    }

    // MARK: - Tokenizer

    func testTokenizerStripAngleBrackets() {
        XCTAssertEqual(NemotronMultilingualTokenizer.stripAngleBrackets("<en-US>"), "en-US")
        XCTAssertEqual(NemotronMultilingualTokenizer.stripAngleBrackets("<zh-CN>"), "zh-CN")
        XCTAssertEqual(NemotronMultilingualTokenizer.stripAngleBrackets("no-brackets"), "no-brackets")
        XCTAssertEqual(NemotronMultilingualTokenizer.stripAngleBrackets("<>"), "")
        XCTAssertEqual(NemotronMultilingualTokenizer.stripAngleBrackets(""), "")
    }

    func testTokenizerFiltersLangTagsAndSurfacesDetectedLanguage() throws {
        // Synthesize a minimal vocab JSON: {"id": "piece"}
        // Token 1 is `<en-US>` (lang tag), 2 is `▁hello`, 3 is `▁world`.
        let vocab: [String: String] = [
            "0": "<unk>",
            "1": "<en-US>",
            "2": "\u{2581}hello",
            "3": "\u{2581}world",
        ]
        let vocabData = try JSONSerialization.data(withJSONObject: vocab)
        let tmpURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("multilingual_vocab_test_\(UUID().uuidString).json")
        try vocabData.write(to: tmpURL)
        defer { try? FileManager.default.removeItem(at: tmpURL) }

        let tokenizer = try NemotronMultilingualTokenizer(
            vocabPath: tmpURL,
            langTagTokenIds: Set([1])
        )
        let decoded = tokenizer.decode(ids: [1, 2, 3])
        XCTAssertEqual(decoded.text, "hello world")
        XCTAssertEqual(decoded.detectedLanguage, "en-US")
    }

    func testTokenizerWithNoLangTag() throws {
        let vocab: [String: String] = [
            "0": "<unk>",
            "1": "<en-US>",
            "2": "\u{2581}hi",
        ]
        let vocabData = try JSONSerialization.data(withJSONObject: vocab)
        let tmpURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("multilingual_vocab_test_\(UUID().uuidString).json")
        try vocabData.write(to: tmpURL)
        defer { try? FileManager.default.removeItem(at: tmpURL) }

        let tokenizer = try NemotronMultilingualTokenizer(
            vocabPath: tmpURL,
            langTagTokenIds: Set([1])
        )
        let decoded = tokenizer.decode(ids: [2])
        XCTAssertEqual(decoded.text, "hi")
        XCTAssertNil(decoded.detectedLanguage)
    }

    func testRawTokenPreservesWordBoundaryMarker() throws {
        // rawToken must return the UNMODIFIED SentencePiece vocab piece, with the
        // `▁` word-boundary marker intact, so callers can group per-token timings
        // into words. decode()/the visible transcript strip `▁`; rawToken must not,
        // otherwise word starts can't be located and word-level timing breaks.
        let vocab: [String: String] = [
            "0": "<unk>",
            "1": "\u{2581}hello",  // word-start piece (has ▁)
            "2": "ing",  // mid-word continuation (no ▁)
        ]
        let vocabData = try JSONSerialization.data(withJSONObject: vocab)
        let tmpURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("multilingual_vocab_test_\(UUID().uuidString).json")
        try vocabData.write(to: tmpURL)
        defer { try? FileManager.default.removeItem(at: tmpURL) }

        let tokenizer = try NemotronMultilingualTokenizer(
            vocabPath: tmpURL,
            langTagTokenIds: Set<Int>()
        )

        // Word-start piece keeps the `▁` marker...
        XCTAssertEqual(tokenizer.rawToken(for: 1), "\u{2581}hello")
        // ...continuation piece has no marker...
        XCTAssertEqual(tokenizer.rawToken(for: 2), "ing")
        // ...the visible transcript strips the marker (why callers need rawToken)...
        XCTAssertFalse(tokenizer.decode(ids: [1]).text.contains("\u{2581}"))
        // ...and an out-of-vocab id returns nil so the caller skips its timing.
        XCTAssertNil(tokenizer.rawToken(for: 999))
    }

    // MARK: - ModelNames

    func testNemotronMultilingualModelNames() {
        XCTAssertTrue(ModelNames.NemotronMultilingualStreaming.preprocessorFile.hasSuffix(".mlmodelc"))
        XCTAssertTrue(ModelNames.NemotronMultilingualStreaming.encoderFile.hasSuffix(".mlmodelc"))
        XCTAssertTrue(ModelNames.NemotronMultilingualStreaming.decoderFile.hasSuffix(".mlmodelc"))
        XCTAssertTrue(ModelNames.NemotronMultilingualStreaming.jointFile.hasSuffix(".mlmodelc"))
        XCTAssertTrue(ModelNames.NemotronMultilingualStreaming.preprocessorPackage.hasSuffix(".mlpackage"))
        XCTAssertEqual(ModelNames.NemotronMultilingualStreaming.tokenizer, "tokenizer.json")
        XCTAssertEqual(ModelNames.NemotronMultilingualStreaming.metadata, "metadata.json")
    }

    // MARK: - Corrupt tokenizer detection (issue #687)

    /// Write a metadata.json + tokenizer.json pair into a fresh temp
    /// directory and return their URLs. Caller removes the directory.
    private func makeVariantDir(
        blankIdx: Int,
        vocab: [String: String]
    ) throws -> (dir: URL, tokenizer: URL, metadata: URL) {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("multilingual_variant_\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        let metadata: [String: Any] = [
            "vocab_size": blankIdx,
            "blank_idx": blankIdx,
            "prompt_dictionary": ["auto": 101],
            "lang_tag_token_ids": [Int](),
        ]
        let metadataURL = dir.appendingPathComponent("metadata.json")
        try JSONSerialization.data(withJSONObject: metadata).write(to: metadataURL)

        let tokenizerURL = dir.appendingPathComponent("tokenizer.json")
        try JSONSerialization.data(withJSONObject: vocab).write(to: tokenizerURL)
        return (dir, tokenizerURL, metadataURL)
    }

    func testCorruptBlankEntryDetected() throws {
        // Pre-2026-05-31 latin tokenizer: "<blank>" at id 2224 (should be
        // "▁there") in addition to the legitimate blank at blank_idx 2828.
        let (dir, tokenizer, metadata) = try makeVariantDir(
            blankIdx: 2828,
            vocab: ["0": "<unk>", "2224": "<blank>", "2828": "<blank>"]
        )
        defer { try? FileManager.default.removeItem(at: dir) }

        XCTAssertTrue(
            StreamingNemotronMultilingualAsrManager.tokenizerHasCorruptBlankEntry(
                tokenizerPath: tokenizer, metadataPath: metadata))
    }

    func testHealthyTokenizerWithBlankAtBlankIdxPasses() throws {
        // Fixed latin tokenizer: "▁there" restored at 2224, "<blank>" only
        // at the blank index.
        let (dir, tokenizer, metadata) = try makeVariantDir(
            blankIdx: 2828,
            vocab: ["0": "<unk>", "2224": "▁there", "2828": "<blank>"]
        )
        defer { try? FileManager.default.removeItem(at: dir) }

        XCTAssertFalse(
            StreamingNemotronMultilingualAsrManager.tokenizerHasCorruptBlankEntry(
                tokenizerPath: tokenizer, metadataPath: metadata))
    }

    func testHealthyTokenizerWithoutBlankPiecePasses() throws {
        // Full multilingual tokenizer ships no "<blank>" entry at all.
        let (dir, tokenizer, metadata) = try makeVariantDir(
            blankIdx: 13087,
            vocab: ["0": "<unk>", "2224": "▁lahko"]
        )
        defer { try? FileManager.default.removeItem(at: dir) }

        XCTAssertFalse(
            StreamingNemotronMultilingualAsrManager.tokenizerHasCorruptBlankEntry(
                tokenizerPath: tokenizer, metadataPath: metadata))
    }

    func testMissingTokenizerFileIsNotCorrupt() throws {
        // Missing files are the normal-download path's job, not the
        // repair pass's.
        let (dir, tokenizer, metadata) = try makeVariantDir(
            blankIdx: 2828,
            vocab: ["0": "<unk>"]
        )
        defer { try? FileManager.default.removeItem(at: dir) }
        try FileManager.default.removeItem(at: tokenizer)

        XCTAssertFalse(
            StreamingNemotronMultilingualAsrManager.tokenizerHasCorruptBlankEntry(
                tokenizerPath: tokenizer, metadataPath: metadata))
    }

    // MARK: - Helpers

    private func makeConfig(
        promptDictionary: [String: Int],
        defaultPromptId: Int = 101
    ) throws -> NemotronMultilingualStreamingConfig {
        let json: [String: Any] = [
            "prompt_dictionary": promptDictionary,
            "default_prompt_id": defaultPromptId,
            "num_prompts": 128,
            "lang_tag_token_ids": [Int](),
        ]
        let data = try JSONSerialization.data(withJSONObject: json)
        let tmpURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("multilingual_cfg_\(UUID().uuidString).json")
        try data.write(to: tmpURL)
        defer { try? FileManager.default.removeItem(at: tmpURL) }
        return try NemotronMultilingualStreamingConfig(from: tmpURL)
    }
}
