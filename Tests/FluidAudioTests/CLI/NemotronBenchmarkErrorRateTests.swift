#if os(macOS)
import XCTest

@testable import FluidAudioCLI

final class NemotronBenchmarkErrorRateTests: XCTestCase {
    func testWeightsByReferenceLengthRatherThanUtteranceCount() throws {
        var aggregate = NemotronBenchmarkErrorRate()
        try aggregate.add(errors: 1, referenceUnits: 1)
        try aggregate.add(errors: 0, referenceUnits: 9)
        XCTAssertEqual(try aggregate.rate(), 0.1, accuracy: 0.000001)
        XCTAssertEqual(aggregate.errors, 1)
        XCTAssertEqual(aggregate.referenceUnits, 10)
    }

    func testRejectsInvalidCountsWithoutMutatingAggregate() throws {
        var aggregate = NemotronBenchmarkErrorRate()
        try aggregate.add(errors: 2, referenceUnits: 5)
        XCTAssertThrowsError(try aggregate.add(errors: 0, referenceUnits: 0))
        XCTAssertThrowsError(try aggregate.add(errors: 1, referenceUnits: -1))
        XCTAssertThrowsError(try aggregate.add(errors: -1, referenceUnits: 5))
        XCTAssertEqual(try aggregate.rate(), 0.4, accuracy: 0.000001)
    }

    func testEmptyAggregateCannotReportPerfectAccuracy() {
        XCTAssertThrowsError(try NemotronBenchmarkErrorRate().rate())
    }

    func testAllowsInsertionsToExceedReferenceLength() throws {
        var aggregate = NemotronBenchmarkErrorRate()
        try aggregate.add(errors: 4, referenceUnits: 2)
        XCTAssertEqual(try aggregate.rate(), 2)
    }
}
#endif
