import CoreML
import Foundation

/// Preserves the online extractor's first-256-values contract in logical row-major
/// order, without assuming a model axis or reinterpreting typed/padded storage.
enum DiarizerEmbeddingOutput {
    static let dimension = 256

    static func extract(from array: MLMultiArray) throws -> [Float] {
        switch array.dataType {
        case .float32, .float16, .double:
            break
        default:
            throw DiarizerError.embeddingExtractionFailed
        }
        do {
            _ = try ANEMemoryViewLayout.span(shape: array.shape, strides: array.strides)
        } catch {
            throw DiarizerError.embeddingExtractionFailed
        }
        guard array.count >= dimension else {
            throw DiarizerError.embeddingExtractionFailed
        }

        let shape = array.shape.map(\.intValue)
        let strides = array.strides.map(\.intValue)
        let values: [Float]
        if array.dataType == .float32, hasContiguousPrefix(shape: shape, strides: strides) {
            // Only the selected prefix must be contiguous; later rows may be padded.
            let pointer = array.dataPointer.assumingMemoryBound(to: Float.self)
            values = Array(UnsafeBufferPointer(start: pointer, count: dimension))
        } else {
            var indices = [NSNumber](repeating: 0, count: array.shape.count)
            values = (0..<dimension).map { index in
                var remaining = index
                for axis in shape.indices.reversed() {
                    indices[axis] = NSNumber(value: remaining % shape[axis])
                    remaining /= shape[axis]
                }
                return array[indices].floatValue
            }
        }
        guard values.allSatisfy(\.isFinite) else {
            throw DiarizerError.embeddingExtractionFailed
        }
        return values
    }

    private static func hasContiguousPrefix(shape: [Int], strides: [Int]) -> Bool {
        var contiguousElements = 1
        for axis in shape.indices.reversed() {
            // A singleton axis never advances the address, whatever its stride.
            guard shape[axis] > 1 else { continue }
            guard strides[axis] == contiguousElements else { return false }
            let needed = (dimension + contiguousElements - 1) / contiguousElements
            if shape[axis] >= needed { return true }
            // The product is below dimension because this axis did not cover it.
            contiguousElements *= shape[axis]
        }
        return false
    }
}
