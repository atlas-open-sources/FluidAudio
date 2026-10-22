import CoreML
import Foundation

/// Independent decoder and inferred language state. Values are observations,
/// not ground truth, speaker identities, or acoustic switch timestamps.
public struct NemotronLanguageObservation: Codable, Sendable, Equatable {
    public let decodedFirst: String?
    public let decodedLatest: String?
    public let inferred: String?
    /// available, not_observed, unsupported_tensor, unavailable_no_language_tags,
    /// or unavailable_argmax_only. Describes the most recent live observer path.
    public let capability: String
    public let path: String?
    public var source: String { inferred == nil ? "none" : "soft_logits" }
}

/// Layout-only reader. Tensor fixtures test memory contracts, not model quality.
enum NemotronLanguageLogits {
    struct Reading {
        let candidates: [(piece: String, logit: Float)]
        let maximum: Float
    }

    static func read(_ logits: MLMultiArray, frame: Int?, languages: [Int: String]) -> Reading? {
        let shape = logits.shape.map(\.intValue)
        let strides = logits.strides.map(\.intValue)
        guard !shape.isEmpty, shape.count == strides.count,
            shape.allSatisfy({ $0 > 0 }), strides.allSatisfy({ $0 > 0 }),
            logits.dataType == .float16 || logits.dataType == .float32
        else { return nil }
        let base: Int
        if let frame {
            guard shape.count == 4, shape[0] == 1, shape[2] == 1,
                frame >= 0, frame < shape[1]
            else { return nil }
            let offset = frame.multipliedReportingOverflow(by: strides[1])
            guard !offset.overflow else { return nil }
            base = offset.partialValue
        } else {
            guard shape.dropLast().allSatisfy({ $0 == 1 }) else { return nil }
            base = 0
        }
        guard let vocab = shape.last, let stride = strides.last else { return nil }
        let last = (vocab - 1).multipliedReportingOverflow(by: stride)
        guard !last.overflow, !base.addingReportingOverflow(last.partialValue).overflow else { return nil }
        let value: (Int) -> Float
        if logits.dataType == .float16 {
            let pointer = logits.dataPointer.assumingMemoryBound(to: UInt16.self)
            value = { nemotronHalfBitsToFloat(pointer[base + $0 * stride]) }
        } else {
            let pointer = logits.dataPointer.assumingMemoryBound(to: Float.self)
            value = { pointer[base + $0 * stride] }
        }
        var maximum = -Float.infinity
        for id in 0..<vocab {
            let logit = value(id)
            guard logit.isFinite else { return nil }
            maximum = max(maximum, logit)
        }
        let candidates = languages.sorted(by: { $0.key < $1.key }).compactMap { id, piece in
            (0..<vocab).contains(id) ? (piece: piece, logit: value(id)) : nil
        }
        return Reading(candidates: candidates, maximum: maximum)
    }
}
