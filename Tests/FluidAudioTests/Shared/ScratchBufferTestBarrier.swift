import Foundation

enum ScratchBufferTestError: Error {
    case aborted
    case timedOut
    case contentsChanged
    case operationFailed
}

struct ScratchBufferObservation: Sendable {
    let address: Int
    let contents: [Float]
    let expectedValue: Float
    let reusedWithinOperation: Bool
}

/// Reusable rendezvous for real overlapping tensor operations, without blocking
/// cooperative executor threads or requiring a CoreML model.
actor ScratchBufferTestBarrier {
    private let participants: Int
    private var waiters: [CheckedContinuation<Void, Error>] = []
    private var failure: ScratchBufferTestError?
    private var timeout: Task<Void, Never>?

    init(participants: Int) {
        precondition(participants > 0)
        self.participants = participants
    }

    func arriveAndWait() async throws {
        try Task.checkCancellation()
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
                if let failure {
                    continuation.resume(throwing: failure)
                    return
                }

                waiters.append(continuation)
                if waiters.count == participants {
                    timeout?.cancel()
                    timeout = nil
                    let ready = waiters
                    waiters.removeAll(keepingCapacity: true)
                    for waiter in ready {
                        waiter.resume()
                    }
                } else if waiters.count == 1 {
                    timeout = Task {
                        do {
                            try await Task.sleep(for: .seconds(10))
                            abort(reason: .timedOut)
                        } catch {
                            // A completed rendezvous cancels its own timeout.
                        }
                    }
                }
            }
        } onCancel: {
            Task { await self.abort() }
        }
    }

    func abort(reason: ScratchBufferTestError = .aborted) {
        guard failure == nil else { return }
        failure = reason
        timeout?.cancel()
        timeout = nil
        let pending = waiters
        waiters.removeAll()
        for waiter in pending {
            waiter.resume(throwing: reason)
        }
    }
}
