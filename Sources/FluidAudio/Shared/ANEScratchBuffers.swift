import CoreML
import Foundation

/// Reusable mutable buffers confined to one sequential operation.
///
/// Create a session with `ANEMemoryOptimizer.makeScratchBuffers()` inside the
/// operation, before its loop. The session owns cached arrays across suspension;
/// the next iteration may reuse them only after inference and output consumption
/// finish. Different operations must have different sessions, even when their
/// keys and shapes match. This type deliberately does not conform to Sendable.
///
/// Returned arrays alias this session's storage. Do not use the session from
/// overlapping tasks or actor reentrant operations, and do not use an earlier
/// array concurrently with another use of the same key. Sharing the stateless
/// allocator is safe; sharing this mutable storage is not.
public final class ANEScratchBuffers {
    private let allocator: ANEMemoryOptimizer
    private var buffers: [String: MLMultiArray] = [:]

    init(allocator: ANEMemoryOptimizer) {
        self.allocator = allocator
    }

    /// Get storage for a sequential step of this operation.
    ///
    /// A matching key, shape and type reuses the same array without clearing it.
    /// A changed shape or type replaces the cached array; any retained previous
    /// array stays alive. Populate every element needed by the next prediction.
    public func buffer(
        key: String,
        shape: [NSNumber],
        dataType: MLMultiArrayDataType
    ) throws -> MLMultiArray {
        if let existing = buffers[key], existing.shape == shape, existing.dataType == dataType {
            return existing
        }

        let array = try allocator.createAlignedArray(shape: shape, dataType: dataType)
        buffers[key] = array
        return array
    }

    /// Replace a key with newly allocated, zero-initialized storage.
    ///
    /// An allocation failure preserves the existing cached array. Retained old
    /// arrays or feature providers stay valid after successful replacement.
    public func prepareBuffer(
        key: String,
        shape: [NSNumber],
        dataType: MLMultiArrayDataType
    ) throws {
        let array = try allocator.createAlignedArray(shape: shape, dataType: dataType)
        buffers[key] = array
    }

    /// Drop this session's cached references.
    ///
    /// Arrays retained by a caller or feature provider remain alive. Other
    /// operations' sessions are unaffected. Do not clear while another task
    /// accesses this same session.
    public func removeAll() {
        buffers.removeAll()
    }
}
