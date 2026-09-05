@preconcurrency import CoreML
import XCTest

@testable import FluidAudio

final class ANEMemoryOptimizerTests: XCTestCase {

    var optimizer: ANEMemoryOptimizer!
    var scratchBuffers: ANEScratchBuffers!

    override func setUp() {
        super.setUp()
        optimizer = ANEMemoryOptimizer()
        scratchBuffers = optimizer.makeScratchBuffers()
    }

    override func tearDown() {
        scratchBuffers = nil
        optimizer = nil
        super.tearDown()
    }

    // MARK: - Buffer Creation Tests

    func testCreateAlignedArray() throws {
        let shape = [3, 240000] as [NSNumber]
        let array = try optimizer.createAlignedArray(
            shape: shape,
            dataType: .float32
        )

        XCTAssertEqual(array.shape, shape)
        XCTAssertEqual(array.dataType, .float32)

        // Verify alignment
        let dataPointer = array.dataPointer
        let address = Int(bitPattern: dataPointer)
        XCTAssertEqual(
            address % ANEMemoryOptimizer.aneAlignment, 0,
            "Array should be 64-byte aligned")
    }

    func testCreateSmallAlignedArray() throws {
        // Test with very small array
        let shape = [1, 10] as [NSNumber]
        let array = try optimizer.createAlignedArray(
            shape: shape,
            dataType: .float32
        )

        XCTAssertEqual(array.shape, shape)
        XCTAssertEqual(array.count, 10)
    }

    func testCreateLargeAlignedArray() throws {
        // Test with large array (10MB)
        let shape = [1, 2_500_000] as [NSNumber]
        let array = try optimizer.createAlignedArray(
            shape: shape,
            dataType: .float32
        )

        XCTAssertEqual(array.shape, shape)
        XCTAssertEqual(array.count, 2_500_000)
    }

    // MARK: - Buffer Pooling Tests

    func testBufferPoolReuse() throws {
        let shape = [3, 1000] as [NSNumber]
        let key = "test_buffer"

        // Get buffer from pool
        let buffer1 = try scratchBuffers.buffer(
            key: key,
            shape: shape,
            dataType: .float32
        )

        // Get same buffer again
        let buffer2 = try scratchBuffers.buffer(
            key: key,
            shape: shape,
            dataType: .float32
        )

        // Should be the same instance
        XCTAssertTrue(buffer1 === buffer2)
    }

    func testBufferPoolDifferentKeys() throws {
        let shape = [3, 1000] as [NSNumber]

        let buffer1 = try scratchBuffers.buffer(
            key: "buffer1",
            shape: shape,
            dataType: .float32
        )

        let buffer2 = try scratchBuffers.buffer(
            key: "buffer2",
            shape: shape,
            dataType: .float32
        )

        // Should be different instances
        XCTAssertFalse(buffer1 === buffer2)
    }

    func testBufferPoolShapeChange() throws {
        let key = "changing_buffer"

        // Create with one shape
        let buffer1 = try scratchBuffers.buffer(
            key: key,
            shape: [3, 1000] as [NSNumber],
            dataType: .float32
        )
        XCTAssertEqual(buffer1.count, 3000)

        // Request with different shape
        let buffer2 = try scratchBuffers.buffer(
            key: key,
            shape: [5, 2000] as [NSNumber],
            dataType: .float32
        )
        XCTAssertEqual(buffer2.count, 10000)

        // Should be different buffers
        XCTAssertFalse(buffer1 === buffer2)
    }

    func testClearPool() throws {
        // Create multiple buffers
        _ = try scratchBuffers.buffer(
            key: "buffer1",
            shape: [100] as [NSNumber],
            dataType: .float32
        )
        _ = try scratchBuffers.buffer(
            key: "buffer2",
            shape: [200] as [NSNumber],
            dataType: .float32
        )

        // Clear pool
        scratchBuffers.removeAll()

        // Get buffer again - should be new instance
        let newBuffer = try scratchBuffers.buffer(
            key: "buffer1",
            shape: [100] as [NSNumber],
            dataType: .float32
        )

        // Verify it's properly initialized
        XCTAssertEqual(newBuffer.count, 100)
    }

    // MARK: - Zero-Copy Feature Provider Tests

    func testZeroCopyFeatureProvider() throws {
        let waveform = try optimizer.createAlignedArray(
            shape: [3, 240000] as [NSNumber],
            dataType: .float32
        )
        let mask = try optimizer.createAlignedArray(
            shape: [3, 1000] as [NSNumber],
            dataType: .float32
        )

        // Fill with test data
        for i in 0..<waveform.count {
            waveform[i] = NSNumber(value: Float(i) / Float(waveform.count))
        }

        let features: [String: MLFeatureValue] = [
            "logmel": MLFeatureValue(multiArray: waveform),
            "mask": MLFeatureValue(multiArray: mask),
        ]

        let provider = ZeroCopyDiarizerFeatureProvider(features: features)

        XCTAssertEqual(provider.featureNames, ["logmel", "mask"])

        // Verify features
        let waveformFeature = provider.featureValue(for: "logmel")
        XCTAssertNotNil(waveformFeature)
        XCTAssertEqual(waveformFeature?.multiArrayValue?.count, waveform.count)

        let maskFeature = provider.featureValue(for: "mask")
        XCTAssertNotNil(maskFeature)
        XCTAssertEqual(maskFeature?.multiArrayValue?.count, mask.count)
    }

    // MARK: - Memory Pressure Tests

    func testMemoryPressureHandling() throws {
        // Create many buffers to simulate memory pressure
        var buffers: [MLMultiArray] = []

        for i in 0..<100 {
            let buffer = try scratchBuffers.buffer(
                key: "pressure_test_\(i)",
                shape: [1000, 1000] as [NSNumber],  // 4MB each
                dataType: .float32
            )
            buffers.append(buffer)
        }

        // Clear cached references; retained caller buffers remain alive.
        scratchBuffers.removeAll()

        // Verify we can still create new buffers
        let newBuffer = try optimizer.createAlignedArray(
            shape: [100] as [NSNumber],
            dataType: .float32
        )
        XCTAssertEqual(newBuffer.count, 100)
    }

    // MARK: - Thread Safety Tests

    func testConcurrentBufferAccess() async throws {
        let allocator = try XCTUnwrap(optimizer)
        let workers = 10
        let barrier = ScratchBufferTestBarrier(participants: workers)

        let results = try await withThrowingTaskGroup(of: ScratchBufferObservation.self) { group in
            for index in 0..<workers {
                group.addTask {
                    do {
                        let scratch = allocator.makeScratchBuffers()
                        let buffer = try scratch.buffer(key: "shared_key", shape: [100], dataType: .float32)
                        let sentinel = Float(index + 1)
                        for element in 0..<buffer.count {
                            buffer[element] = NSNumber(value: sentinel)
                        }
                        let address = Int(bitPattern: buffer.dataPointer)

                        // All same-key buffers are live before any worker verifies its contents.
                        try await barrier.arriveAndWait()
                        let contents = (0..<buffer.count).map { buffer[$0].floatValue }
                        let reused = try scratch.buffer(key: "shared_key", shape: [100], dataType: .float32)
                        return ScratchBufferObservation(
                            address: address,
                            contents: contents,
                            expectedValue: sentinel,
                            reusedWithinOperation: reused === buffer
                        )
                    } catch {
                        await barrier.abort()
                        throw error
                    }
                }
            }
            return try await group.reduce(into: []) { $0.append($1) }
        }

        XCTAssertEqual(Set(results.map(\.address)).count, workers)
        for result in results {
            XCTAssertEqual(result.contents, Array(repeating: result.expectedValue, count: 100))
            XCTAssertTrue(result.reusedWithinOperation)
        }
    }

    // MARK: - Performance Tests

    func testBufferCreationPerformance() {
        measure {
            do {
                for _ in 0..<100 {
                    _ = try optimizer.createAlignedArray(
                        shape: [3, 16000] as [NSNumber],
                        dataType: .float32
                    )
                }
            } catch {
                XCTFail("Performance test failed: \(error)")
            }
        }
    }

    func testBufferPoolingPerformance() {
        measure {
            do {
                for i in 0..<1000 {
                    _ = try scratchBuffers.buffer(
                        key: "perf_test_\(i % 10)",  // Reuse 10 buffers
                        shape: [3, 16000] as [NSNumber],
                        dataType: .float32
                    )
                }
            } catch {
                XCTFail("Performance test failed: \(error)")
            }
        }
    }
}
