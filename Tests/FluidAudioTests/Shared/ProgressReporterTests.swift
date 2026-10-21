import XCTest

@testable import FluidAudio

final class ProgressReporterTests: XCTestCase {

    func testSubdirectoryProgressUsesByteWeightingDuringLargeFile() {
        let fraction = ProgressReporter.downloadFraction(
            completedBytes: 399,
            totalBytes: 665,
            completedFiles: 9,
            totalFiles: 22
        )

        XCTAssertEqual(fraction, 399.0 / 665.0, accuracy: 0.0001)
        XCTAssertGreaterThan(fraction, Double(9) / Double(22))
    }

    func testSubdirectoryProgressIsMonotonicForRealisticSequence() {
        let progress = [
            ProgressReporter.downloadFraction(
                completedBytes: 0,
                totalBytes: 665,
                completedFiles: 0,
                totalFiles: 22
            ),
            ProgressReporter.downloadFraction(
                completedBytes: 40,
                totalBytes: 665,
                completedFiles: 4,
                totalFiles: 22
            ),
            ProgressReporter.downloadFraction(
                completedBytes: 120,
                totalBytes: 665,
                completedFiles: 9,
                totalFiles: 22
            ),
            ProgressReporter.downloadFraction(
                completedBytes: 399,
                totalBytes: 665,
                completedFiles: 9,
                totalFiles: 22
            ),
            ProgressReporter.downloadFraction(
                completedBytes: 565,
                totalBytes: 665,
                completedFiles: 10,
                totalFiles: 22
            ),
            ProgressReporter.downloadFraction(
                completedBytes: 665,
                totalBytes: 665,
                completedFiles: 22,
                totalFiles: 22
            ),
        ]

        XCTAssertEqual(progress, progress.sorted())
    }

    func testSubdirectoryProgressClampsToOne() {
        let fraction = ProgressReporter.downloadFraction(
            completedBytes: 700,
            totalBytes: 665,
            completedFiles: 22,
            totalFiles: 22
        )

        XCTAssertEqual(fraction, 1.0)
    }

    func testSubdirectoryProgressFallsBackToFileCountWhenTotalBytesUnknown() {
        let fraction = ProgressReporter.downloadFraction(
            completedBytes: 0,
            totalBytes: 0,
            completedFiles: 9,
            totalFiles: 22
        )

        XCTAssertEqual(fraction, Double(9) / Double(22), accuracy: 0.0001)
    }

    func testSubdirectoryProgressIsCompleteWhenThereAreNoFiles() {
        let fraction = ProgressReporter.downloadFraction(
            completedBytes: 0,
            totalBytes: 0,
            completedFiles: 0,
            totalFiles: 0
        )

        XCTAssertEqual(fraction, 1.0)
    }
}

extension ProgressReporterTests {
    func testConcurrentDownloadsPreserveAggregateByteCounts() {
        let recorder = ProgressStreamRecorder()
        let reporter = ProgressReporter(handler: { recorder.append($0) }, downloadPhaseWeight: 0.5)
        let progress = ConcurrentProgress(reporter: reporter, totalBytes: 100, totalFiles: 2)
        progress.liveBytesCallback(fileIndex: 0)?(20, 40)
        progress.liveBytesCallback(fileIndex: 1)?(30, 60)
        _ = progress.fileCompleted(fileIndex: 0, size: 40)
        _ = progress.fileCompleted(fileIndex: 1, size: 60)
        let events = recorder.snapshot()
        XCTAssertEqual(events.map(\.completedBytes), [20, 50, 70, 100])
        XCTAssertEqual(events.map(\.totalBytes), [100, 100, 100, 100])
        XCTAssertEqual(events.last?.fractionCompleted, 0.5)
    }

    func testUnknownTotalsAndNonDownloadPhasesDoNotInventBytes() {
        let recorder = ProgressStreamRecorder()
        let reporter = ProgressReporter(handler: { recorder.append($0) }, downloadPhaseWeight: 0.5)
        reporter.listing()
        reporter.fileBoundary(completedBytes: 25, totalBytes: 0, completedFiles: 1, totalFiles: 2)
        let concurrent = ConcurrentProgress(reporter: reporter, totalBytes: 0, totalFiles: 2)
        _ = concurrent.fileCompleted(fileIndex: 0, size: 25)
        reporter.cachedModelsAvailable()
        reporter.compiling(name: "model", index: 1, count: 1)
        reporter.finished()
        for event in recorder.snapshot() {
            XCTAssertNil(event.completedBytes)
            XCTAssertNil(event.totalBytes)
        }
    }
}
