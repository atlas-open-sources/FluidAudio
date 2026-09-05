import CoreML
import XCTest

@testable import FluidAudio

final class ANEScratchBufferLifetimeTests: XCTestCase {
    func testTypeReplacementPreservesPreviousStorage() throws {
        let scratch = ANEMemoryOptimizer().makeScratchBuffers()
        let original = try scratch.buffer(key: "input", shape: [32], dataType: .float32)
        original[0] = 41

        let replacement = try scratch.buffer(key: "input", shape: [32], dataType: .int32)

        XCTAssertFalse(original === replacement)
        XCTAssertEqual(replacement.dataType, .int32)
        XCTAssertEqual(replacement[0].intValue, 0)
        XCTAssertEqual(original[0].floatValue, 41)
        let reused = try scratch.buffer(key: "input", shape: [32], dataType: .int32)
        XCTAssertTrue(replacement === reused)
    }

    func testExplicitPreparationReplacesStorageAndPreservesProvider() throws {
        let scratch = ANEMemoryOptimizer().makeScratchBuffers()
        let original = try scratch.buffer(key: "input", shape: [32], dataType: .float32)
        original[0] = 41
        let provider = ZeroCopyDiarizerFeatureProvider(features: ["input": MLFeatureValue(multiArray: original)])

        try scratch.prepareBuffer(key: "input", shape: [32], dataType: .float32)
        let replacement = try scratch.buffer(key: "input", shape: [32], dataType: .float32)

        XCTAssertFalse(original === replacement)
        XCTAssertEqual(replacement[0].floatValue, 0)
        XCTAssertEqual(provider.featureValue(for: "input")?.multiArrayValue?[0].floatValue, 41)
    }

    func testClearPreservesRetainedProviderAndOtherOperation() throws {
        let allocator = ANEMemoryOptimizer()
        let first = allocator.makeScratchBuffers()
        let second = allocator.makeScratchBuffers()
        let original = try first.buffer(key: "input", shape: [32], dataType: .float32)
        original[0] = 41
        let provider = ZeroCopyDiarizerFeatureProvider(features: ["input": MLFeatureValue(multiArray: original)])
        let independent = try second.buffer(key: "input", shape: [32], dataType: .float32)
        independent[0] = 73

        first.removeAll()
        let replacement = try first.buffer(key: "input", shape: [32], dataType: .float32)

        XCTAssertFalse(original === replacement)
        XCTAssertEqual(replacement[0].floatValue, 0)
        XCTAssertEqual(provider.featureValue(for: "input")?.multiArrayValue?[0].floatValue, 41)
        XCTAssertEqual(independent[0].floatValue, 73)
        let reused = try second.buffer(key: "input", shape: [32], dataType: .float32)
        XCTAssertTrue(reused === independent)
    }

    func testInputSurvivesAnotherOperationThrowingAcrossAwait() async throws {
        try await verifyInputSurvivesInterruptedOperation(cancel: false)
    }

    func testInputSurvivesAnotherOperationBeingCancelledAcrossAwait() async throws {
        try await verifyInputSurvivesInterruptedOperation(cancel: true)
    }

    private func verifyInputSurvivesInterruptedOperation(cancel: Bool) async throws {
        let allocator = ANEMemoryOptimizer()
        let started = ScratchBufferTestBarrier(participants: 2)
        let interrupted = ScratchBufferTestBarrier(participants: 2)
        let task = Task<Void, Error> {
            do {
                let scratch = allocator.makeScratchBuffers()
                let buffer = try scratch.buffer(key: "input", shape: [32], dataType: .float32)
                buffer[0] = 41
                let provider = ZeroCopyDiarizerFeatureProvider(features: ["input": MLFeatureValue(multiArray: buffer)])
                try await started.arriveAndWait()
                // No bytes are released/recycled while an input provider is suspended.
                try await interrupted.arriveAndWait()
                try Task.checkCancellation()
                guard provider.featureValue(for: "input")?.multiArrayValue?[0].floatValue == 41 else {
                    throw ScratchBufferTestError.contentsChanged
                }
                throw ScratchBufferTestError.operationFailed
            } catch {
                await started.abort()
                throw error
            }
        }

        let scratch = allocator.makeScratchBuffers()
        let buffer = try scratch.buffer(key: "input", shape: [32], dataType: .float32)
        buffer[0] = 73
        let provider = ZeroCopyDiarizerFeatureProvider(features: ["input": MLFeatureValue(multiArray: buffer)])

        do {
            try await started.arriveAndWait()
            if cancel {
                task.cancel()
            } else {
                try await interrupted.arriveAndWait()
            }

            do {
                try await task.value
                XCTFail("The interrupted operation must fail")
            } catch is CancellationError {
                XCTAssertTrue(cancel)
            } catch let error as ScratchBufferTestError {
                switch error {
                case .operationFailed:
                    XCTAssertFalse(cancel)
                case .aborted:
                    XCTAssertTrue(cancel)
                case .contentsChanged, .timedOut:
                    XCTFail("Unexpected operation failure: \(error)")
                }
            }
        } catch {
            task.cancel()
            await interrupted.abort()
            _ = await task.result
            throw error
        }

        XCTAssertEqual(provider.featureValue(for: "input")?.multiArrayValue?[0].floatValue, 73)
        let reused = try scratch.buffer(key: "input", shape: [32], dataType: .float32)
        XCTAssertTrue(reused === buffer)
        let next = allocator.makeScratchBuffers()
        let independent = try next.buffer(key: "input", shape: [32], dataType: .float32)
        XCTAssertFalse(independent === buffer)
        XCTAssertEqual(independent[0].floatValue, 0)
    }
}
