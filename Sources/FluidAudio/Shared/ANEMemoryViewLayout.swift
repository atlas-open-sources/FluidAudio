import Foundation

/// Checked addressable spans for borrowed multi-array storage, measured in elements.
enum ANEMemoryViewLayout {
    static func span(shape: [NSNumber], strides: [NSNumber]) throws -> Int {
        guard !shape.isEmpty, shape.count == strides.count else {
            throw ANEMemoryUtils.ANEMemoryError.invalidShape
        }
        var span = 1
        for (dimension, stride) in zip(shape, strides) {
            let dimensionValue = try positiveInteger(dimension)
            let strideValue = try positiveInteger(stride)
            let (distance, productOverflow) = (dimensionValue - 1)
                .multipliedReportingOverflow(by: strideValue)
            let (nextSpan, sumOverflow) = span.addingReportingOverflow(distance)
            guard !productOverflow, !sumOverflow else {
                throw ANEMemoryUtils.ANEMemoryError.invalidShape
            }
            span = nextSpan
        }
        return span
    }

    static func alignedStrides(shape: [NSNumber]) throws -> [NSNumber] {
        guard !shape.isEmpty else { throw ANEMemoryUtils.ANEMemoryError.invalidShape }
        var strides = [NSNumber](repeating: 1, count: shape.count)
        var stride = 1
        for index in shape.indices.reversed() {
            let dimension = try positiveInteger(shape[index])
            strides[index] = NSNumber(value: stride)
            // The outermost dimension never contributes another stride.
            guard index > 0 else { continue }
            var extent = dimension
            let tileSize = ANEMemoryUtils.aneTileSize
            if index == shape.count - 1, dimension % tileSize != 0 {
                let (padded, overflow) = dimension.addingReportingOverflow(tileSize - dimension % tileSize)
                guard !overflow else { throw ANEMemoryUtils.ANEMemoryError.invalidShape }
                extent = padded
            }
            let (nextStride, overflow) = stride.multipliedReportingOverflow(by: extent)
            guard !overflow else { throw ANEMemoryUtils.ANEMemoryError.invalidShape }
            stride = nextStride
        }
        return strides
    }

    private static func positiveInteger(_ number: NSNumber) throws -> Int {
        let value = number.intValue
        guard value > 0, number.doubleValue.isFinite,
            number.compare(NSNumber(value: value)) == .orderedSame
        else {
            throw ANEMemoryUtils.ANEMemoryError.invalidShape
        }
        return value
    }
}
