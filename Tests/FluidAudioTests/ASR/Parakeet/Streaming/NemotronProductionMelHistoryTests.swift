@preconcurrency import CoreML
import Foundation
import XCTest

@testable import FluidAudio

/// Exercises production cache updates using an existing upstream human recording.
/// Requires a local real model; does not acquire models or audio over the network.
@MainActor
final class NemotronProductionMelHistoryTests: XCTestCase {
    func testNormalAndPrefetchedPathsRetainConsumedRealFrames() async throws {
        guard let modelPath = ProcessInfo.processInfo.environment["FLUIDAUDIO_PARITY_MODEL_DIR"] else {
            throw XCTSkip("Set FLUIDAUDIO_PARITY_MODEL_DIR to the pinned FP32 model")
        }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuOnly
        let manager = StreamingNemotronMultilingualAsrManager(configuration: configuration)
        try await manager.loadModels(from: URL(fileURLWithPath: modelPath))
        let name = "01-validation-request-21.4s.wav"
        let url = try XCTUnwrap(
            Bundle.module.url(forResource: "Fixtures/\(name)", withExtension: nil)
                ?? Bundle.module.url(forResource: name, withExtension: nil))
        let samples = try AudioConverter().resampleAudioFile(path: url.path)
        for prefetch in [false, true] {
            await manager.reset()
            let result = try await manager.captureProductionHistory(samples, prefetch: prefetch)
            XCTAssertEqual(result.actual, result.expected, "Production history mismatch; prefetch=\(prefetch)")
            XCTAssertEqual(result.usedPrefetch, prefetch, "Requested production path did not execute")
        }
    }
}

extension StreamingNemotronMultilingualAsrManager {
    fileprivate func captureProductionHistory(
        _ samples: [Float], prefetch: Bool
    ) async throws
        -> (actual: [Float], expected: [Float], usedPrefetch: Bool)
    {
        let count = config.chunkSamples
        guard samples.count >= count * 2 else { throw ASRError.notInitialized }
        let first = Array(samples[0..<count])
        let second = Array(samples[count..<(count * 2)])
        try await processChunk(first, nextChunkSamples: prefetch ? second : nil)
        let history = try XCTUnwrap(melCache)
        let usedPrefetch = prefetchedEncoded != nil
        // Independent frame sequence: append only consumed frames from real audio,
        // retaining prior zero context where a chunk is shorter than the history.
        let extractor = NemotronMelExtractor()
        var expected = Array(repeating: [Float](repeating: 0, count: config.preEncodeCache), count: config.melFeatures)
        for chunk in prefetch ? [first, second] : [first] {
            let mel = try extractor.melSpectrogram(samples: chunk)
            let valid = chunk.count / 160
            for feature in 0..<config.melFeatures {
                for frame in 0..<valid {
                    expected[feature].append(mel[[0, NSNumber(value: feature), NSNumber(value: frame)]].floatValue)
                }
                expected[feature].removeFirst(expected[feature].count - config.preEncodeCache)
            }
        }
        return ((0..<history.count).map { history[$0].floatValue }, expected.flatMap { $0 }, usedPrefetch)
    }
}
