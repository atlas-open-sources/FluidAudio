import CoreML
import XCTest

@testable import FluidAudio

final class DiarizerEmbeddingOutputTests: XCTestCase {
    func testFloat32ContiguousFirstRowPreservesValues() throws {
        let output = try MLMultiArray(shape: [2, 256], dataType: .float32)
        for index in 0..<256 {
            output[[0, NSNumber(value: index)]] = NSNumber(value: Float(index) / 256)
            output[[1, NSNumber(value: index)]] = -99
        }
        XCTAssertEqual(try DiarizerEmbeddingOutput.extract(from: output), expectedValues)
    }

    func testFloat16AndDoubleRowsConvertWithoutReinterpretingBytes() throws {
        for type in [MLMultiArrayDataType.float16, .double] {
            let output = try MLMultiArray(shape: [256], dataType: type)
            for index in 0..<256 { output[index] = NSNumber(value: Float(index) / 256) }
            XCTAssertEqual(try DiarizerEmbeddingOutput.extract(from: output), expectedValues)
        }
    }

    func testPaddedRowsIgnorePaddingAndLaterRows() throws {
        let storage = try MLMultiArray(shape: [600], dataType: .float32)
        for index in 0..<storage.count { storage[index] = NSNumber(value: Float.nan) }
        let output = try ANEMemoryUtils.createZeroCopyView(
            from: storage, offset: 0, shape: [2, 256], strides: [300, 1])
        for index in 0..<256 { output[[0, NSNumber(value: index)]] = NSNumber(value: Float(index) / 256) }
        XCTAssertEqual(try DiarizerEmbeddingOutput.extract(from: output), expectedValues)
    }

    func testTransposedRowUsesLogicalIndices() throws {
        let storage = try MLMultiArray(shape: [512], dataType: .float32)
        for index in 0..<storage.count { storage[index] = NSNumber(value: Float.nan) }
        let output = try ANEMemoryUtils.createZeroCopyView(
            from: storage, offset: 0, shape: [2, 256], strides: [1, 2])
        for index in 0..<256 { output[[0, NSNumber(value: index)]] = NSNumber(value: Float(index) / 256) }
        XCTAssertEqual(try DiarizerEmbeddingOutput.extract(from: output), expectedValues)
    }

    func testShortShapesAreRejectedBeforeCopying() throws {
        for shape: [NSNumber] in [[1], [255], [2, 127]] {
            let output = try MLMultiArray(shape: shape, dataType: .float32)
            assertExtractionFails(output)
        }
    }

    func testHistoricalPrefixDoesNotRequireA256ValueFinalAxis() throws {
        for shape: [NSNumber] in [[256, 1], [16, 16], [2, 128], [257]] {
            let output = try MLMultiArray(shape: shape, dataType: .float32)
            for index in 0..<output.count { output[index] = NSNumber(value: Float(index) / 256) }
            XCTAssertEqual(try DiarizerEmbeddingOutput.extract(from: output), expectedValues)
        }
    }

    func testPaddedAndTransposedMatrixPrefixesUseLogicalOrder() throws {
        for strides: [NSNumber] in [[32, 1], [1, 16]] {
            let storage = try MLMultiArray(shape: [512], dataType: .float32)
            for index in 0..<storage.count { storage[index] = NSNumber(value: Float.nan) }
            let output = try ANEMemoryUtils.createZeroCopyView(
                from: storage, offset: 0, shape: [16, 16], strides: strides)
            for row in 0..<16 {
                for column in 0..<16 {
                    output[[NSNumber(value: row), NSNumber(value: column)]] =
                        NSNumber(value: Float(row * 16 + column) / 256)
                }
            }
            XCTAssertEqual(try DiarizerEmbeddingOutput.extract(from: output), expectedValues)
        }
    }

    func testIntegerOutputsAreRejected() throws {
        let output = try MLMultiArray(shape: [256], dataType: .int32)
        assertExtractionFails(output)
    }

    func testNonFiniteValuesAndNarrowingOverflowAreRejected() throws {
        for value in [Double.nan, Double.infinity, -Double.infinity, Double.greatestFiniteMagnitude] {
            let output = try MLMultiArray(shape: [256], dataType: .double)
            for index in 0..<256 { output[index] = 0 }
            output[17] = NSNumber(value: value)
            assertExtractionFails(output)
        }
        let output = try MLMultiArray(shape: [256], dataType: .float32)
        for index in 0..<256 { output[index] = 0 }
        output[255] = NSNumber(value: Float.nan)
        assertExtractionFails(output)
    }

    private var expectedValues: [Float] { (0..<256).map { Float($0) / 256 } }

    private func assertExtractionFails(_ output: MLMultiArray, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertThrowsError(try DiarizerEmbeddingOutput.extract(from: output), file: file, line: line) { error in
            guard case DiarizerError.embeddingExtractionFailed = error else {
                return XCTFail("Unexpected extraction error: \(error)", file: file, line: line)
            }
        }
    }
}
