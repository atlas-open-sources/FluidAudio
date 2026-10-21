#if os(macOS)
import XCTest

@testable import FluidAudioCLI

final class NemotronQualificationReplayTests: XCTestCase {
    private let required = ["--model-dir", "models", "--input", "recording.wav", "--output", "replay.jsonl"]

    func testDefaultsToOfflineCausalTwentyMilliseconds() throws {
        let options = try NemotronQualificationReplay.Options(arguments: required)
        XCTAssertEqual(options.chunkMilliseconds, 20)
        XCTAssertFalse(options.realtime)
        XCTAssertNil(options.language)
        XCTAssertEqual(options.computeUnits, .cpuAndNeuralEngine)
    }

    func testExplicitRealtimeAndPromptSettings() throws {
        let options = try NemotronQualificationReplay.Options(
            arguments: required + ["--realtime", "--chunk-ms", "10", "--language", "es-ES", "--compute-units", "cpu"])
        XCTAssertTrue(options.realtime)
        XCTAssertEqual(options.chunkMilliseconds, 10)
        XCTAssertEqual(options.language, "es-ES")
        XCTAssertEqual(options.computeUnits, .cpuOnly)
    }

    func testRejectsMissingUnknownAndDuplicateArguments() {
        XCTAssertThrowsError(try NemotronQualificationReplay.Options(arguments: []))
        XCTAssertThrowsError(try NemotronQualificationReplay.Options(arguments: required + ["--unknown", "yes"]))
        XCTAssertThrowsError(try NemotronQualificationReplay.Options(arguments: required + ["--chunk-ms"]))
        XCTAssertThrowsError(try NemotronQualificationReplay.Options(arguments: required + ["--input", "other.wav"]))
        XCTAssertThrowsError(
            try NemotronQualificationReplay.Options(arguments: required + ["--realtime", "--realtime"]))
    }

    func testRejectsInvalidChunkAndComputeSettings() {
        for value in ["0", "-1", "1001", "1.5", "nan"] {
            XCTAssertThrowsError(
                try NemotronQualificationReplay.Options(arguments: required + ["--chunk-ms", value]))
        }
        XCTAssertThrowsError(
            try NemotronQualificationReplay.Options(arguments: required + ["--compute-units", "unknown"]))
    }

    func testRequiresWavInput() {
        XCTAssertThrowsError(
            try NemotronQualificationReplay.Options(
                arguments: ["--model-dir", "models", "--input", "recording.mp3", "--output", "replay.jsonl"]))
    }
}
#endif
