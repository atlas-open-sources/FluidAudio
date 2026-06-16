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
