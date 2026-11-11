@preconcurrency import CoreML
import CryptoKit
import Foundation
import XCTest

@testable import FluidAudio

/// Pinned real Mandarin speech that previously saturated the RNNT symbol limit
/// with Thai-token repetitions. FLEURS audio: CC BY 4.0, repack revision
/// 1cca811bb8ea4d370345f108f00518167040282c. These are regression examples,
/// not an independent quality evaluation. No network acquisition in tests.
@MainActor
final class NemotronMultilingualHistoryRegressionTests: XCTestCase {
    func testMandarinHistoryDoesNotInsertThaiRepetitions() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let modelPath = environment["FLUIDAUDIO_PARITY_MODEL_DIR"],
            let corpusPath = environment["FLUIDAUDIO_PARITY_CORPUS_DIR"]
        else {
            throw XCTSkip(
                "Set FLUIDAUDIO_PARITY_MODEL_DIR and FLUIDAUDIO_PARITY_CORPUS_DIR to the pinned FP32 model and real corpus"
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
        await manager.setLanguage("zh-CN")
        let fixtures: [(id: String, sha256: String)] = [
            ("cmn_hans_cn_0286", "b666c59e0232ba9758466da491bf39d8f568fb08229ecc8f285383b8a2afcd18"),
            ("cmn_hans_cn_0278", "1818b4545ebbf670b13d08cab9f42491531154387e9666e3ea9caa743af97d1c"),
            ("cmn_hans_cn_0925", "136a75958d6c4a25f98a77141e5232b5fb6276f9fb6b8a3a4e478d31a758cce4"),
        ]
        for fixture in fixtures {
            let audioURL = URL(fileURLWithPath: corpusPath)
                .appendingPathComponent("cmn_hans_cn").appendingPathComponent("\(fixture.id).wav")
            let bytes = try Data(contentsOf: audioURL)
            let hash = SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
            XCTAssertEqual(hash, fixture.sha256, "Pinned real fixture changed")
            guard hash == fixture.sha256 else { return }
            let audio = try AudioConverter().resampleAudioFile(path: audioURL.path)
            await manager.reset()
            _ = try await manager.process(samples: audio)
            let transcript = try await manager.finish()
            XCTAssertFalse(transcript.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            XCTAssertFalse(
                transcript.unicodeScalars.contains { (0x0E00...0x0E7F).contains(Int($0.value)) },
                "Unexpected Thai output for pinned Mandarin speech \(fixture.id): \(transcript)"
            )
        }
    }
}
