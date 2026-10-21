import CoreML
import CryptoKit
import Foundation
import XCTest

@testable import FluidAudio

final class NemotronMelExtractorTests: XCTestCase {
    private struct Reference: Decodable {
        let sourceAudio: String
        let sourceAudioSha256: String
        let cases: [Case]

        struct Case: Decodable {
            let samples: Int
            let sampleBytesSha256: String
            let shape: [Int]
            let validFrames: Int
            let referenceFile: String
            let sha256: String
        }
    }

    func testRealAudioMatchesIndependentCoreMLReferenceForAlignedAndNonalignedPrefixes() throws {
        // Existing human recording, cleared by its speaker for public release.
        // Goldens came from the checkpoint's FP32 CoreML preprocessor, not this implementation.
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let manifest = try decoder.decode(
            Reference.self, from: Data(contentsOf: resource("nemotron-mel-reference.json")))
        let audioURL = try resource(manifest.sourceAudio)
        XCTAssertEqual(digest(try Data(contentsOf: audioURL)), manifest.sourceAudioSha256)
        let samples = try AudioConverter().resampleAudioFile(path: audioURL.path)
        XCTAssertEqual(manifest.cases.count, 2)

        for reference in manifest.cases {
            XCTAssertGreaterThanOrEqual(samples.count, reference.samples)
            let prefix = Array(samples.prefix(reference.samples))
            XCTAssertEqual(prefix.withUnsafeBytes { digest(Data($0)) }, reference.sampleBytesSha256)
            let expectedData = try Data(contentsOf: resource(reference.referenceFile))
            XCTAssertEqual(digest(expectedData), reference.sha256)
            let expected = expectedData.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
            let actual = try NemotronMelExtractor().melSpectrogram(samples: prefix)
            XCTAssertEqual(actual.shape.map(\.intValue), reference.shape)
            XCTAssertEqual(actual.count, expected.count)
            let frames = reference.shape[2]
            var maximumAbsoluteError: Float = 0
            var squaredError = 0.0
            actual.withUnsafeBufferPointer(ofType: Float.self) { values in
                for index in 0..<min(values.count, expected.count) {
                    XCTAssertTrue(values[index].isFinite)
                    let difference = abs(values[index] - expected[index])
                    maximumAbsoluteError = max(maximumAbsoluteError, difference)
                    squaredError += Double(difference) * Double(difference)
                    if index % frames >= reference.validFrames {
                        // This includes the extra frame copied into subsequent chunk history.
                        XCTAssertEqual(values[index], expected[index], "Masked frame must match reference exactly")
                    }
                }
            }
            XCTAssertLessThan(maximumAbsoluteError, 0.001, "prefix samples=\(reference.samples)")
            XCTAssertLessThan(sqrt(squaredError / Double(expected.count)), 0.00005)
        }
    }

    private func resource(_ name: String) throws -> URL {
        try XCTUnwrap(
            Bundle.module.url(forResource: "Fixtures/\(name)", withExtension: nil)
                ?? Bundle.module.url(forResource: name, withExtension: nil))
    }

    private func digest(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}
