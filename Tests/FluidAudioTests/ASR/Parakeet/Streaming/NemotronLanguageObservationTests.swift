import CoreML
import XCTest

@testable import FluidAudio

/// Model-free API and layout contracts, not speech-accuracy evidence.
final class NemotronLanguageObservationTests: XCTestCase {
    func testSoftEvidenceDoesNotBecomeDecodedTag() async {
        let manager = StreamingNemotronMultilingualAsrManager()
        for _ in 0..<20 {
            await manager.applySoftLanguage(candidates: [("en-US", 0), ("es-419", -20)], frameMaxLogit: 0)
        }
        var snapshot = await manager.languageObservation()
        XCTAssertNil(snapshot.decodedFirst)
        XCTAssertNil(snapshot.decodedLatest)
        XCTAssertEqual(snapshot.inferred, "en-US")
        await manager.recordDetectedLanguage("es-419")
        await manager.recordDetectedLanguage("nl-NL")
        snapshot = await manager.languageObservation()
        XCTAssertEqual(snapshot.decodedFirst, "es-419")
        XCTAssertEqual(snapshot.decodedLatest, "nl-NL")
        XCTAssertEqual(snapshot.inferred, "en-US")
        await manager.reset()
        snapshot = await manager.languageObservation()
        XCTAssertNil(snapshot.decodedFirst)
        XCTAssertNil(snapshot.decodedLatest)
        XCTAssertNil(snapshot.inferred)
        XCTAssertEqual(snapshot.capability, "not_observed")
    }

    func testDocumentedAliasesAndAllowlistReset() async {
        for (alias, tag) in [("eng", "en-US"), ("DUT", "nl-NL"), ("zho", "zh-CN"), ("fr_FR", "fr-CA")] {
            let manager = StreamingNemotronMultilingualAsrManager()
            await manager.setExpectedLanguages([alias])
            await manager.recordDetectedLanguage("es-419")
            await manager.recordDetectedLanguage(tag)
            let first = await manager.detectedLanguage()
            XCTAssertEqual(first, tag)
            for _ in 0..<20 {
                await manager.applySoftLanguage(candidates: [(tag, 0), ("es-419", 10)], frameMaxLogit: 10)
            }
            let inferred = await manager.inferredLanguage()
            XCTAssertEqual(inferred, tag, "The same alias contract filters soft observations")
            await manager.reset()
            await manager.recordDetectedLanguage("es-419")
            let reset = await manager.detectedLanguage()
            XCTAssertNil(reset)
            await manager.setExpectedLanguages(nil)
            await manager.recordDetectedLanguage("es-419")
            let cleared = await manager.detectedLanguage()
            XCTAssertEqual(cleared, "es-419")
        }
    }

    func testFusedCapabilityDoesNotInventEstimate() async {
        let manager = StreamingNemotronMultilingualAsrManager()
        await manager.recordArgmaxOnlyLanguageCapability()
        let snapshot = await manager.languageObservation()
        XCTAssertEqual(snapshot.capability, "unavailable_argmax_only")
        XCTAssertEqual(snapshot.path, "fused_argmax")
        XCTAssertNil(snapshot.inferred)
    }

    func testReadersAgreeAcrossFloatTypesAndStrides() throws {
        // Exactly representable tensor fixtures, not mocked model outputs.
        for dtype in [MLMultiArrayDataType.float16, .float32] {
            let ordinary = try MLMultiArray(shape: [1, 1, 1, 4], dataType: dtype)
            let batched = try MLMultiArray(shape: [1, 2, 1, 4], dataType: dtype)
            for index in 0..<4 {
                ordinary[index] = NSNumber(value: index - 3)
                batched[[0, 1, 0, NSNumber(value: index)]] = NSNumber(value: index - 3)
            }
            let first = try XCTUnwrap(NemotronLanguageLogits.read(ordinary, frame: nil, languages: [0: "en", 2: "nl"]))
            let second = try XCTUnwrap(NemotronLanguageLogits.read(batched, frame: 1, languages: [0: "en", 2: "nl"]))
            XCTAssertEqual(first.maximum, second.maximum)
            XCTAssertEqual(first.candidates.map(\.logit), second.candidates.map(\.logit))
        }
        for dtype in [MLMultiArrayDataType.float16, .float32] {
            let storage = UnsafeMutableRawPointer.allocate(byteCount: 128, alignment: 16)
            storage.initializeMemory(as: UInt8.self, repeating: 0, count: 128)
            let strided = try MLMultiArray(
                dataPointer: storage, shape: [1, 2, 1, 4], dataType: dtype,
                strides: [32, 16, 8, 2], deallocator: { $0.deallocate() })
            for index in 0..<4 { strided[[0, 1, 0, NSNumber(value: index)]] = NSNumber(value: index) }
            let result = try XCTUnwrap(
                NemotronLanguageLogits.read(
                    strided, frame: 1, languages: [-1: "bad", 0: "en", 3: "nl", 4: "bad"]))
            XCTAssertEqual(result.maximum, 3)
            XCTAssertEqual(result.candidates.map(\.logit), [0, 3])
        }
    }

    /// Optional real model output captured from documented human-recorded audio.
    /// Repacking tests reader equivalence, not actual speculative model execution.
    func testCapturedRealLogitsAcrossLayouts() throws {
        guard let path = ProcessInfo.processInfo.environment["FLUIDAUDIO_LANGUAGE_LOGITS_FIXTURE"] else {
            throw XCTSkip("Set FLUIDAUDIO_LANGUAGE_LOGITS_FIXTURE to the documented real-logit capture")
        }
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        let row = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let values = try XCTUnwrap(row["values"] as? [Double]).map(Float.init)
        let rawLanguages = try XCTUnwrap(row["languages"] as? [String: String])
        let languages = Dictionary(
            uniqueKeysWithValues: try rawLanguages.map { key, value in
                (try XCTUnwrap(Int(key)), value)
            })
        XCTAssertGreaterThan(values.count, 1000)
        XCTAssertFalse(languages.isEmpty)
        for dtype in [MLMultiArrayDataType.float16, .float32] {
            let ordinary = try MLMultiArray(shape: [NSNumber(value: values.count)], dataType: dtype)
            let batched = try MLMultiArray(shape: [1, 2, 1, NSNumber(value: values.count)], dataType: dtype)
            for (index, value) in values.enumerated() {
                ordinary[index] = NSNumber(value: value)
                batched[[0, 1, 0, NSNumber(value: index)]] = NSNumber(value: value)
            }
            let first = try XCTUnwrap(NemotronLanguageLogits.read(ordinary, frame: nil, languages: languages))
            let second = try XCTUnwrap(NemotronLanguageLogits.read(batched, frame: 1, languages: languages))
            XCTAssertEqual(first.maximum, second.maximum)
            XCTAssertEqual(first.candidates.map(\.piece), second.candidates.map(\.piece))
            XCTAssertEqual(first.candidates.map(\.logit), second.candidates.map(\.logit))
            let expected = values.map { dtype == .float16 ? Float(Float16($0)) : $0 }
            XCTAssertEqual(first.maximum, expected.max())
            for candidate in first.candidates {
                let id = try XCTUnwrap(languages.first { $0.value == candidate.piece }?.key)
                XCTAssertEqual(candidate.logit, expected[id])
            }
        }
    }

    func testRejectsInvalidRankFrameTypeAndNonFiniteValues() throws {
        let matrix = try MLMultiArray(shape: [2, 4], dataType: .float32)
        XCTAssertNil(NemotronLanguageLogits.read(matrix, frame: 0, languages: [:]))
        XCTAssertNil(NemotronLanguageLogits.read(matrix, frame: nil, languages: [:]))
        let batched = try MLMultiArray(shape: [1, 2, 1, 4], dataType: .float32)
        XCTAssertNil(NemotronLanguageLogits.read(batched, frame: -1, languages: [:]))
        XCTAssertNil(NemotronLanguageLogits.read(batched, frame: 2, languages: [:]))
        let integer = try MLMultiArray(shape: [4], dataType: .int32)
        XCTAssertNil(NemotronLanguageLogits.read(integer, frame: nil, languages: [:]))
        let vector = try MLMultiArray(shape: [4], dataType: .float32)
        for index in 0..<4 { vector[index] = 0 }
        vector[3] = NSNumber(value: Float.nan)
        XCTAssertNil(NemotronLanguageLogits.read(vector, frame: nil, languages: [:]))
    }
}
