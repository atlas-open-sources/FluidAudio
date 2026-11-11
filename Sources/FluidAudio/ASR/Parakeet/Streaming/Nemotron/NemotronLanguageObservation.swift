import Accelerate
import CoreML
import Foundation

/// Independent decoder and inferred language state. Values are observations,
/// not ground truth, speaker identities, or acoustic switch timestamps.
public struct NemotronLanguageObservation: Codable, Sendable, Equatable {
    public let decodedFirst: String?
    public let decodedLatest: String?
    public let inferred: String?
    /// available, not_observed, unsupported_tensor, unavailable_no_language_tags,
    /// invalid_numeric, or unavailable_argmax_only. Describes the most recent live observer path.
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

    enum Failure: Error, Equatable {
        case unsupportedLayout
        case invalidNumeric
    }

    static func read(_ logits: MLMultiArray, frame: Int?, languages: [Int: String]) -> Reading? {
        try? readResult(logits, frame: frame, languages: languages).get()
    }

    static func readResult(
        _ logits: MLMultiArray, frame: Int?, languages: [Int: String]
    ) -> Result<Reading, Failure> {
        let shape = logits.shape.map(\.intValue)
        let strides = logits.strides.map(\.intValue)
        guard !shape.isEmpty, shape.count == strides.count,
            shape.allSatisfy({ $0 > 0 }), strides.allSatisfy({ $0 > 0 }),
            logits.dataType == .float16 || logits.dataType == .float32
        else { return .failure(.unsupportedLayout) }
        let base: Int
        if let frame {
            guard shape.count == 4, shape[0] == 1, shape[2] == 1,
                frame >= 0, frame < shape[1]
            else { return .failure(.unsupportedLayout) }
            let offset = frame.multipliedReportingOverflow(by: strides[1])
            guard !offset.overflow else { return .failure(.unsupportedLayout) }
            base = offset.partialValue
        } else {
            guard shape.dropLast().allSatisfy({ $0 == 1 }) else { return .failure(.unsupportedLayout) }
            base = 0
        }
        guard let vocab = shape.last, let stride = strides.last else { return .failure(.unsupportedLayout) }
        let last = (vocab - 1).multipliedReportingOverflow(by: stride)
        guard !last.overflow, !base.addingReportingOverflow(last.partialValue).overflow else { return .failure(.unsupportedLayout) }
        let value: (Int) -> Float
        if logits.dataType == .float16 {
            let pointer = logits.dataPointer.assumingMemoryBound(to: UInt16.self)
            value = { nemotronHalfBitsToFloat(pointer[base + $0 * stride]) }
        } else {
            let pointer = logits.dataPointer.assumingMemoryBound(to: Float.self)
            value = { pointer[base + $0 * stride] }
        }
        var maximum = -Float.infinity
        if logits.dataType == .float32, stride == 1 {
            let pointer = logits.dataPointer.assumingMemoryBound(to: Float.self).advanced(by: base)
            // Preserve the vectorized maximum for the common contiguous fp32
            // path. Validate NaN/+inf separately; -inf is valid masked mass.
            for id in 0..<vocab {
                guard !pointer[id].isNaN, pointer[id] != .infinity else {
                    return .failure(.invalidNumeric)
                }
            }
            vDSP_maxv(pointer, 1, &maximum, vDSP_Length(vocab))
        } else {
            for id in 0..<vocab {
                let logit = value(id)
                guard !logit.isNaN, logit != .infinity else { return .failure(.invalidNumeric) }
                maximum = max(maximum, logit)
            }
        }
        let candidates = languages.sorted(by: { $0.key < $1.key }).compactMap { id, piece in
            (0..<vocab).contains(id) ? (piece: piece, logit: value(id)) : nil
        }
        return .success(Reading(candidates: candidates, maximum: maximum))
    }
}
