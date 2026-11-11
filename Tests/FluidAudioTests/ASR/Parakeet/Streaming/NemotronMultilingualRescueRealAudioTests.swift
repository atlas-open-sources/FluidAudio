@preconcurrency import CoreML
import CryptoKit
import Foundation
import XCTest

@testable import FluidAudio

@MainActor
final class NemotronMultilingualRescueRealAudioTests: XCTestCase {
    func testExplicitRealAudioRescuePreservesOpeningSpeech() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let modelPath = environment["FLUIDAUDIO_PARITY_MODEL_DIR"],
            let corpusPath = environment["FLUIDAUDIO_PARITY_CORPUS_DIR"]
        else {
            throw XCTSkip("Set FLUIDAUDIO_PARITY_MODEL_DIR and FLUIDAUDIO_PARITY_CORPUS_DIR for real-model rescue")
        }
        XCTAssertTrue(StreamingNemotronMultilingualAsrManager.blankRescueEnabled)
        XCTAssertGreaterThan(StreamingNemotronMultilingualAsrManager.rescueRmsThreshold, 0)
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

        // Google FLEURS CC BY 4.0, FluidInference/fleurs-full pinned revision
        // 1cca811bb8ea4d370345f108f00518167040282c. No downloads or synthesized audio.
        let audioURL = URL(fileURLWithPath: corpusPath)
            .appendingPathComponent("en_us/en_us_0429.wav")
        let hash = SHA256.hash(data: try Data(contentsOf: audioURL))
            .map { String(format: "%02x", $0) }.joined()
        let expectedHash = "7d0df713a15b29417bc1954c314e151178e4c5e8a26f8ac15396f21c3bc738a4"
        XCTAssertEqual(hash, expectedHash)
        guard hash == expectedHash else { return }
        let samples = try AudioConverter().resampleAudioFile(path: audioURL.path)
        XCTAssertFalse(samples.isEmpty)
        await manager.reset()
        await manager.prepareExplicitRealAudioRescue(samples)
        try await manager.finalizeRescueSpanIfNeeded()

        let detected = await manager.detectedBlankSpanCount
        let rescued = await manager.blankRescueCount
        XCTAssertEqual(detected, 1, "The real rescue decoder must execute")
        XCTAssertEqual(rescued, 1, "The rescue must recover lexical speech")
        let transcript = await manager.getPartialTranscript()
        let normalized = transcript.lowercased().components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { !$0.isEmpty }.joined(separator: " ")
        XCTAssertTrue(
            normalized.hasPrefix("if you re not used to driving on country roads"),
            "Rescue lost opening speech: \(transcript)")

        // Closing the same span twice must not invoke or duplicate the rescue.
        try await manager.finalizeRescueSpanIfNeeded()
        let repeated = await manager.getPartialTranscript()
        let repeatedCount = await manager.blankRescueCount
        XCTAssertEqual(repeated, transcript)
        XCTAssertEqual(repeatedCount, rescued)
    }
}

extension StreamingNemotronMultilingualAsrManager {
    fileprivate func prepareExplicitRealAudioRescue(_ samples: [Float]) {
        // Deliberately arrange an all-blank span containing a real recording.
        // This tests fresh-state rescue decoding, NOT natural RMS/span detection
        // or human-aligned speech boundaries. Do not alter the actual samples.
        let encoderFrameSamples = 1280
        let frameCount = (samples.count + encoderFrameSamples - 1) / encoderFrameSamples
        rescueSpanOpen = true
        rescueSpanAudio = samples
        rescueSpanStartFrame = 0
        rescueSpanLastSpeechFrame = frameCount - 1
        rescueSpanSpeechWindows = frameCount
        rescueSpanPreRollFrames = 0
        rescueSpanOverflowed = false
    }
}
