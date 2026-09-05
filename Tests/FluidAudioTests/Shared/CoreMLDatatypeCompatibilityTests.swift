import CoreML
import XCTest

@testable import FluidAudio

final class CoreMLDatatypeCompatibilityTests: XCTestCase {

    #if compiler(>=6.2)
    @available(macOS 26, iOS 26, *)
    func testInt8ElementSizeAndPaddedCopy() throws {
        XCTAssertEqual(ANEMemoryUtils.getElementSize(for: .int8), MemoryLayout<Int8>.stride)
        let source = try MLMultiArray(shape: [2, 3], dataType: .int8)
        let values: [Int8] = [-128, -1, 0, 1, 42, 127]
        for (index, value) in values.enumerated() {
            source[index] = NSNumber(value: value)
        }
        let padded = try ANEMemoryUtils.createAlignedArray(shape: [2, 3], dataType: .int8)
        XCTAssertEqual(padded.strides.map(\.intValue), [16, 1])
        ANEMemoryUtils.strideAwareCopy(from: source, to: padded)
        for (index, value) in values.enumerated() {
            XCTAssertEqual(padded[index].int8Value, value)
        }
        let restored = try MLMultiArray(shape: [2, 3], dataType: .int8)
        ANEMemoryUtils.strideAwareCopy(from: padded, to: restored)
        for (index, value) in values.enumerated() {
            XCTAssertEqual(restored[index].int8Value, value)
        }
    }

    @available(macOS 26, iOS 26, *)
    func testInt8ExtractionPreservesSignedRange() throws {
        let array = try MLMultiArray(shape: [4], dataType: .int8)
        let values: [Int8] = [-128, -1, 0, 127]
        for (index, value) in values.enumerated() {
            array[index] = NSNumber(value: value)
        }
        let expected = values.map(Float.init)
        XCTAssertEqual(StyleTTS2MultiArray.extractFloats(array), expected)
        XCTAssertEqual(Supertonic3MultiArray.extractFloats(array), expected)
    }
    #endif
}
