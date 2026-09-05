import CoreML
import XCTest

@testable import FluidAudio

final class ANEZeroCopyViewTests: XCTestCase {
    func testBothFactoriesRejectInvalidOffsetsAndShapes() throws {
        let source = try ANEMemoryUtils.createAlignedArray(shape: [32], dataType: .float32)
        let invalid: [(offset: Int, shape: [NSNumber])] = [
            (-1, [1]), (32, [1]), (31, [2]), (Int.max, [1]),
            (0, []), (0, [0]), (0, [-1]), (0, [NSNumber(value: Int.max), 2]),
            (0, [2, NSNumber(value: Int.max)]),
            (0, [1.5]), (0, [NSNumber(value: Double.nan)]), (0, [NSNumber(value: Double.infinity)]),
            (0, [NSNumber(value: UInt64.max)]),
        ]
        for request in invalid {
            XCTAssertThrowsError(
                try ANEMemoryUtils.createZeroCopyView(from: source, offset: request.offset, shape: request.shape)
            ) { error in
                guard case ANEMemoryUtils.ANEMemoryError.invalidShape = error else {
                    return XCTFail("Unexpected error: \(error)")
                }
            }
            XCTAssertThrowsError(
                try ANEMemoryOptimizer().createZeroCopyView(from: source, shape: request.shape, offset: request.offset)
            ) { error in
                guard case DiarizerError.invalidArrayBounds = error else {
                    return XCTFail("Unexpected error: \(error)")
                }
            }
        }
    }

    func testInvalidExplicitStridesAreRejected() throws {
        let source = try ANEMemoryUtils.createAlignedArray(shape: [32], dataType: .float32)
        let invalid: [[NSNumber]] = [
            [], [1], [0, 1], [-1, 1], [NSNumber(value: Int.max), 1], [32, 1], [1.5, 1],
            [NSNumber(value: Double.nan), 1], [NSNumber(value: Double.infinity), 1], [NSNumber(value: UInt64.max), 1],
        ]
        for strides in invalid {
            XCTAssertThrowsError(
                try ANEMemoryUtils.createZeroCopyView(from: source, offset: 0, shape: [2, 2], strides: strides)
            )
        }
    }

    func testPaddedViewCannotOverrunContiguousStorage() throws {
        let source = try MLMultiArray(shape: [2, 3], dataType: .float32)
        XCTAssertEqual(source.strides, [3, 1])
        // Six logical values with ANE strides [16, 1] require nineteen addressable elements.
        XCTAssertThrowsError(try ANEMemoryOptimizer().createZeroCopyView(from: source, shape: [2, 3]))
        XCTAssertThrowsError(try ANEMemoryUtils.createZeroCopyView(from: source, offset: 0, shape: [2, 3]))

        let contiguous = try ANEMemoryUtils.createZeroCopyView(
            from: source, offset: 0, shape: [2, 3], strides: source.strides)
        contiguous[[1, 2] as [NSNumber]] = 73
        XCTAssertEqual(source[[1, 2] as [NSNumber]].floatValue, 73)
    }

    func testPaddedAndTransposedLayoutsUseEveryDimension() throws {
        let padded = try ANEMemoryUtils.createAlignedArray(shape: [2, 3], dataType: .float32)
        let paddedView = try ANEMemoryOptimizer().createZeroCopyView(from: padded, shape: [2, 3])
        paddedView[[1, 2] as [NSNumber]] = 73
        XCTAssertEqual(padded[[1, 2] as [NSNumber]].floatValue, 73)

        let source = try MLMultiArray(shape: [2, 3], dataType: .float32)
        source[[1, 2] as [NSNumber]] = 41
        let transposed = try ANEMemoryUtils.createZeroCopyView(
            from: source, offset: 0, shape: [3, 2], strides: [1, 3])
        let lastElement = try ANEMemoryUtils.createZeroCopyView(from: transposed, offset: 5, shape: [1])
        XCTAssertEqual(lastElement[0].floatValue, 41)
        XCTAssertThrowsError(try ANEMemoryUtils.createZeroCopyView(from: transposed, offset: 6, shape: [1]))
    }

    func testUtilityViewRetainsSourceUntilReleased() throws {
        try verifySourceLifetime(useOptimizer: false)
    }

    func testOptimizerViewRetainsSourceUntilReleased() throws {
        try verifySourceLifetime(useOptimizer: true)
    }

    func testNestedViewRetainsEntireStorageChain() throws {
        weak var sourceReference: MLMultiArray?
        weak var intermediateReference: MLMultiArray?
        var nested: MLMultiArray?
        try autoreleasepool {
            let source = try ANEMemoryUtils.createAlignedArray(shape: [32], dataType: .float32)
            source[8] = 73
            sourceReference = source
            let intermediate = try ANEMemoryUtils.createZeroCopyView(from: source, offset: 4, shape: [16])
            intermediateReference = intermediate
            nested = try ANEMemoryOptimizer().createZeroCopyView(from: intermediate, shape: [8], offset: 4)
        }

        XCTAssertNotNil(sourceReference)
        XCTAssertNotNil(intermediateReference)
        // A failed ownership assertion must never be followed by a freed-pointer read.
        if sourceReference != nil && intermediateReference != nil {
            XCTAssertEqual(nested?[0].floatValue, 73)
        }
        autoreleasepool { nested = nil }
        XCTAssertNil(intermediateReference)
        XCTAssertNil(sourceReference)
    }

    func testProviderRetainsViewAfterScratchIsCleared() throws {
        weak var sourceReference: MLMultiArray?
        var provider: ZeroCopyDiarizerFeatureProvider?
        try autoreleasepool {
            let scratch = ANEMemoryOptimizer().makeScratchBuffers()
            let source = try scratch.buffer(key: "input", shape: [32], dataType: .float32)
            source[4] = 41
            sourceReference = source
            let view = try ANEMemoryUtils.createZeroCopyView(from: source, offset: 4, shape: [8])
            provider = ZeroCopyDiarizerFeatureProvider(features: ["input": MLFeatureValue(multiArray: view)])
            scratch.removeAll()
            let replacement = try scratch.buffer(key: "input", shape: [32], dataType: .float32)
            XCTAssertFalse(replacement === source)
            replacement[4] = 99
        }

        XCTAssertNotNil(sourceReference)
        if sourceReference != nil {
            XCTAssertEqual(provider?.featureValue(for: "input")?.multiArrayValue?[0].floatValue, 41)
        }
        autoreleasepool { provider = nil }
        XCTAssertNil(sourceReference)
    }

    private func verifySourceLifetime(useOptimizer: Bool) throws {
        weak var sourceReference: MLMultiArray?
        var view: MLMultiArray?
        try autoreleasepool {
            let source = try ANEMemoryUtils.createAlignedArray(shape: [32], dataType: .float32)
            sourceReference = source
            source[4] = 41
            if useOptimizer {
                view = try ANEMemoryOptimizer().createZeroCopyView(from: source, shape: [8], offset: 4)
            } else {
                view = try ANEMemoryUtils.createZeroCopyView(from: source, offset: 4, shape: [8])
            }
            XCTAssertEqual(view?.dataPointer, source.dataPointer.advanced(by: 4 * MemoryLayout<Float>.stride))
            view?[0] = 73
            XCTAssertEqual(source[4].floatValue, 73)
        }

        XCTAssertNotNil(sourceReference)
        if sourceReference != nil {
            XCTAssertEqual(view?[0].floatValue, 73)
        }
        autoreleasepool { view = nil }
        XCTAssertNil(sourceReference)
    }
}
