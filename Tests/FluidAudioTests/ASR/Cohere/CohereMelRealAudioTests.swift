import CryptoKit
import XCTest

@testable import FluidAudio

@available(macOS 14.0, iOS 17.0, *)
final class CohereMelRealAudioTests: XCTestCase {
    func testRealRecordingProducesFiniteRepeatableSpectrogram() throws {
        // Existing upstream human recording, cleared by its speaker for public release.
        let name = "01-validation-request-21.4s.wav"
        let url = try XCTUnwrap(
            Bundle.module.url(forResource: "Fixtures/\(name)", withExtension: nil)
                ?? Bundle.module.url(forResource: name, withExtension: nil))
        let samples = try AudioConverter().resampleAudioFile(path: url.path)
        XCTAssertFalse(samples.isEmpty)
        let spectrogram = CohereMelSpectrogram()
        let first = spectrogram.compute(audio: samples)
        let second = spectrogram.compute(audio: samples)
        XCTAssertEqual(first.validFrames, samples.count / 160)
        XCTAssertEqual(first.mel.count, 128)
        XCTAssertEqual(first.mel, second.mel)
        let flattened = first.mel.flatMap { $0 }
        XCTAssertFalse(flattened.isEmpty)
        XCTAssertTrue(flattened.allSatisfy { $0.isFinite })
        // Same-host before/after evidence; not a cross-platform golden-float promise.
        let checksum = flattened.withUnsafeBytes { SHA256.hash(data: Data($0)) }
        print("COHERE_REAL_AUDIO_SHA256 \(checksum) frames=\(first.validFrames) values=\(flattened.count)")
    }
}
