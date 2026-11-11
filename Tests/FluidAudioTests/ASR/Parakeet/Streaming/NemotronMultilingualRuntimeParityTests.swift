@preconcurrency import CoreML
import CryptoKit
import Foundation
import XCTest

@testable import FluidAudio

/// Real-model regression for opening speech lost with a nonempty initial cache.
/// Audio: Google FLEURS, CC BY 4.0, pinned FluidInference/fleurs-full repackaging
/// revision 1cca811bb8ea4d370345f108f00518167040282c. No network acquisition in tests.
@MainActor
final class NemotronMultilingualRuntimeParityTests: XCTestCase {
    func testOpeningSpeechSurvivesInitialAndResetSessions() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let modelPath = environment["FLUIDAUDIO_PARITY_MODEL_DIR"],
            let corpusPath = environment["FLUIDAUDIO_PARITY_CORPUS_DIR"]
        else {
            throw XCTSkip(
                "Set FLUIDAUDIO_PARITY_MODEL_DIR to the qualified FP32 model and FLUIDAUDIO_PARITY_CORPUS_DIR to the pinned FLEURS corpus"
            )
        }
        let configuration = MLModelConfiguration()
        switch environment["FLUIDAUDIO_PARITY_COMPUTE_UNITS", default: "cpuOnly"] {
        case "cpuOnly": configuration.computeUnits = .cpuOnly
        case "cpuAndNeuralEngine": configuration.computeUnits = .cpuAndNeuralEngine
        default:
            XCTFail("Unsupported parity compute-unit setting")
            return
        }
        let manager = StreamingNemotronMultilingualAsrManager(configuration: configuration)
        try await manager.loadModels(from: URL(fileURLWithPath: modelPath))
        await manager.setLanguage("en-US")

        let fixtures: [(id: String, sha256: String, opening: String)] = [
            (
                "en_us_0429", "7d0df713a15b29417bc1954c314e151178e4c5e8a26f8ac15396f21c3bc738a4",
                "if you're not used to driving on country roads"
            ),
            (
                "en_us_0263", "5aa5d1b313d211a349f8c85158d2a816feebd05bcc546a7d3497158f0a4e3bcb",
                "these are sometimes crowded family beaches"
            ),
        ]
        // Repeat the first real utterance after another session to catch stale
        // history as well as incorrect initialization on the first session.
        let sessions = fixtures + [fixtures[0]]
        for fixture in sessions {
            let audioURL = URL(fileURLWithPath: corpusPath)
                .appendingPathComponent("en_us").appendingPathComponent("\(fixture.id).wav")
            let bytes = try Data(contentsOf: audioURL)
            let hash = SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
            XCTAssertEqual(hash, fixture.sha256, "Pinned real fixture changed")
            guard hash == fixture.sha256 else { return }
            let audio = try AudioConverter().resampleAudioFile(path: audioURL.path)
            await manager.reset()
            _ = try await manager.process(samples: audio)
            let transcript = try await manager.finish()
            XCTAssertTrue(
                normalized(transcript).hasPrefix(normalized(fixture.opening)),
                "Opening speech lost for \(fixture.id): \(transcript)")
        }
    }

    private func normalized(_ text: String) -> String {
        text.lowercased().components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { !$0.isEmpty }.joined(separator: " ")
    }
}
