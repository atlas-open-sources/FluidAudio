import Accelerate
import CoreML
import Foundation

/// Stateless ANE-aligned allocation and copy operations.
///
/// The allocator can be shared between concurrent operations. Mutable arrays cannot:
/// create a separate scratch session for each operation that needs sequential reuse.
public final class ANEMemoryOptimizer: Sendable {
    // Use shared ANE constants
    public static let aneAlignment = ANEMemoryUtils.aneAlignment
    public static let aneTileSize = ANEMemoryUtils.aneTileSize

    public init() {}

    /// Create ANE-aligned MLMultiArray with optimized memory layout
    public func createAlignedArray(
        shape: [NSNumber],
        dataType: MLMultiArrayDataType
    ) throws -> MLMultiArray {
        do {
            return try ANEMemoryUtils.createAlignedArray(
                shape: shape,
                dataType: dataType,
                zeroClear: true
            )
        } catch ANEMemoryUtils.ANEMemoryError.allocationFailed {
            throw DiarizerError.memoryAllocationFailed
        } catch {
            throw DiarizerError.memoryAllocationFailed
        }
    }

    /// Create mutable scratch storage owned by one sequential operation.
    ///
    /// Create the session inside the operation, before its processing loop. Keep it
    /// through any awaited inference and output consumption. Do not store it on a
    /// shared worker or use it from overlapping operations, including reentrant
    /// calls on an actor. The session deliberately does not conform to Sendable.
    public func makeScratchBuffers() -> ANEScratchBuffers {
        ANEScratchBuffers(allocator: self)
    }

    /// Create a zero-copy memory view between models, retaining the source storage.
    /// The view shares mutable bytes with its source and requires serialized access.
    public func createZeroCopyView(
        from sourceArray: MLMultiArray,
        shape: [NSNumber],
        offset: Int = 0
    ) throws -> MLMultiArray {
        do {
            return try ANEMemoryUtils.createZeroCopyView(from: sourceArray, offset: offset, shape: shape)
        } catch ANEMemoryUtils.ANEMemoryError.invalidShape {
            throw DiarizerError.invalidArrayBounds
        }
    }

    /// Copy data using ANE-optimized memory operations
    public func optimizedCopy<C>(
        from source: C,
        to destination: MLMultiArray,
        offset: Int = 0,
        pad: Bool = false
    ) where C: Collection, C.Element == Float {
        guard destination.dataType == .float32 else { return }

        let destPtr = destination.dataPointer.assumingMemoryBound(to: Float.self)
        let count = min(source.count, destination.count - offset)

        // If source is contiguous array, use optimized copy
        if let array = source as? [Float] {
            array.withUnsafeBufferPointer { srcBuffer in
                vDSP_mmov(
                    srcBuffer.baseAddress!,
                    destPtr.advanced(by: offset),
                    vDSP_Length(count),
                    vDSP_Length(1),
                    vDSP_Length(1),
                    vDSP_Length(count)
                )
            }
        } else if let slice = source as? ArraySlice<Float> {
            // ArraySlice may share contiguous memory
            slice.withUnsafeBufferPointer { srcBuffer in
                vDSP_mmov(
                    srcBuffer.baseAddress!,
                    destPtr.advanced(by: offset),
                    vDSP_Length(count),
                    vDSP_Length(1),
                    vDSP_Length(1),
                    vDSP_Length(count)
                )
            }
        } else if let contiguous = source as? ContiguousArray<Float> {
            // ContiguousArray is also contiguous
            contiguous.withUnsafeBufferPointer { srcBuffer in
                vDSP_mmov(
                    srcBuffer.baseAddress!,
                    destPtr.advanced(by: offset),
                    vDSP_Length(count),
                    vDSP_Length(1),
                    vDSP_Length(1),
                    vDSP_Length(count)
                )
            }
        } else {
            // Fallback for other collections - copy element by element
            var destIndex = offset
            for element in source.prefix(count) {
                destPtr[destIndex] = element
                destIndex += 1
            }
        }

        if pad {
            let padStart = offset + count
            let remaining = destination.count - padStart
            if remaining > 0 {
                vDSP_vclr(destPtr.advanced(by: padStart), 1, vDSP_Length(remaining))
            }
        }
    }

}

/// Extension for MLMultiArray to enable zero-copy operations
extension MLMultiArray {
    /// Prefetch data to Neural Engine (iOS 17+/macOS 14+)
    public func prefetchToNeuralEngine() {
        // Trigger ANE prefetch by accessing first and last elements
        // This causes the ANE to initiate DMA transfer
        if count > 0 {
            _ = self[0]
            _ = self[count - 1]
        }
    }
}

/// Zero-copy feature provider for chaining models
public final class ZeroCopyDiarizerFeatureProvider: NSObject, MLFeatureProvider {
    private let features: [String: MLFeatureValue]

    public init(features: [String: MLFeatureValue]) {
        self.features = features
        super.init()
    }

    public var featureNames: Set<String> {
        Set(features.keys)
    }

    public func featureValue(for featureName: String) -> MLFeatureValue? {
        features[featureName]
    }

    /// Create a provider that chains output from one model to input of another
    public static func chain(
        from outputProvider: MLFeatureProvider,
        outputName: String,
        to inputName: String,
        additionalFeatures: [String: MLFeatureValue] = [:]
    ) -> ZeroCopyDiarizerFeatureProvider? {
        guard let outputValue = outputProvider.featureValue(for: outputName) else {
            return nil
        }

        var features = additionalFeatures
        features[inputName] = outputValue

        return ZeroCopyDiarizerFeatureProvider(features: features)
    }

    /// Create a provider for batch processing multiple inputs
    public static func batchProvider(
        waveforms: [MLMultiArray],
        masks: [MLMultiArray]
    ) -> [ZeroCopyDiarizerFeatureProvider] {
        return zip(waveforms, masks).map { waveform, mask in
            ZeroCopyDiarizerFeatureProvider(features: [
                "waveform": MLFeatureValue(multiArray: waveform),
                "mask": MLFeatureValue(multiArray: mask),
            ])
        }
    }
}
