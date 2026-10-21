@preconcurrency import CoreML
import Foundation
import XCTest

@testable import FluidAudio

@MainActor
final class NemotronMelHistoryTests: XCTestCase {
    private let melFeatures = 128
    private let historyFrames = 9
    private let samplesPerFrame = 160

    func testRollingHistoryUsesOnlyValidRealAudioFramesAcrossAllChunkSizes() throws {
        let samples = try realRecording()
        for validFrames in [8, 32, 56, 112, 224] {
            let chunkSamples = validFrames * samplesPerFrame
            var cache: MLMultiArray?
            var expectedHistory = Array(
                repeating: [Float](repeating: 0, count: historyFrames), count: melFeatures)
            let extractor = NemotronMelExtractor()
            for chunkIndex in 0..<4 {
                let start = chunkIndex * chunkSamples
                let chunk = Array(samples[start..<(start + chunkSamples)])
                let mel = try extractor.melSpectrogram(samples: chunk)
                XCTAssertGreaterThan(mel.shape[2].intValue, validFrames)
                let encoderInput = try StreamingNemotronMultilingualAsrManager.prependMelCachePure(
                    melCache: cache, chunkMel: mel, totalMelFrames: historyFrames + validFrames,
                    melFeatures: melFeatures, preEncodeCache: historyFrames)
                for feature in 0..<melFeatures {
                    // The next encoder receives the history accumulated independently below.
                    for frame in 0..<historyFrames {
                        XCTAssertEqual(value(encoderInput, feature, frame), expectedHistory[feature][frame])
                    }
                    // Append real valid frames to an independent rolling sequence. In the
                    // 80ms case this retains one older frame, including an initial zero.
                    for frame in 0..<validFrames {
                        expectedHistory[feature].append(value(mel, feature, frame))
                    }
                    expectedHistory[feature].removeFirst(expectedHistory[feature].count - historyFrames)
                }
                cache = try StreamingNemotronMultilingualAsrManager.extractMelCachePure(
                    chunkMel: encoderInput, melFeatures: melFeatures, preEncodeCache: historyFrames)
                let actual = try XCTUnwrap(cache)
                XCTAssertEqual(actual.shape.map(\.intValue), [1, melFeatures, historyFrames])
                for feature in 0..<melFeatures {
                    for frame in 0..<historyFrames {
                        XCTAssertEqual(
                            value(actual, feature, frame), expectedHistory[feature][frame],
                            "validFrames=\(validFrames), chunk=\(chunkIndex), feature=\(feature), frame=\(frame)")
                    }
                }
            }
        }
    }

    func testInstanceHistoryMatchesRealValidFramesAcrossSuccessiveChunks() async throws {
        let samples = try realRecording()
        let validFrames = 112
        let chunkSamples = validFrames * samplesPerFrame
        let chunks = (0..<4).map { index in
            Array(samples[(index * chunkSamples)..<((index + 1) * chunkSamples)])
        }
        let manager = StreamingNemotronMultilingualAsrManager()
        let actual = try await manager.captureRealAudioInstanceHistories(chunks)
        var expectedHistory = Array(
            repeating: [Float](repeating: 0, count: historyFrames), count: melFeatures)
        let extractor = NemotronMelExtractor()
        for (chunkIndex, chunk) in chunks.enumerated() {
            let mel = try extractor.melSpectrogram(samples: chunk)
            for feature in 0..<melFeatures {
                for frame in 0..<validFrames {
                    expectedHistory[feature].append(value(mel, feature, frame))
                }
                expectedHistory[feature].removeFirst(expectedHistory[feature].count - historyFrames)
            }
            XCTAssertEqual(actual[chunkIndex], expectedHistory.flatMap { $0 })
        }
    }

    private func realRecording() throws -> [Float] {
        // Existing upstream human recording cleared by its speaker for public release.
        let name = "01-validation-request-21.4s.wav"
        let url = try XCTUnwrap(
            Bundle.module.url(forResource: "Fixtures/\(name)", withExtension: nil)
                ?? Bundle.module.url(forResource: name, withExtension: nil))
        let samples = try AudioConverter().resampleAudioFile(path: url.path)
        return try XCTUnwrap(
            samples.count >= 4 * 224 * samplesPerFrame ? samples : nil,
            "Real recording is too short for the requested chunk contexts")
    }

    private func value(_ array: MLMultiArray, _ feature: Int, _ frame: Int) -> Float {
        array[[0, NSNumber(value: feature), NSNumber(value: frame)]].floatValue
    }
}

extension StreamingNemotronMultilingualAsrManager {
    fileprivate func captureRealAudioInstanceHistories(_ chunks: [[Float]]) throws -> [[Float]] {
        let extractor = NemotronMelExtractor()
        var histories: [[Float]] = []
        for chunk in chunks {
            let mel = try extractor.melSpectrogram(samples: chunk)
            let encoderInput = try prependMelCache(to: mel)
            let nextCache = try extractMelCache(from: encoderInput)
            melCache = nextCache
            histories.append((0..<nextCache.count).map { nextCache[$0].floatValue })
        }
        return histories
    }
}
