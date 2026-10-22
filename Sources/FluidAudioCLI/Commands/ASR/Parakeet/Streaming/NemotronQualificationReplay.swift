#if os(macOS)
import CoreML
import FluidAudio
import Foundation

/// Replays real audio causally; offline replay measures compute, not live end-to-end latency.
enum NemotronQualificationReplay {
    struct Options {
        let modelDirectory: String
        let input: String
        let output: String
        let chunkMilliseconds: Int
        let realtime: Bool
        let language: String?
        let computeUnits: MLComputeUnits
        let expectedLanguages: Set<String>?

        init(arguments: [String]) throws {
            var values: [String: String] = [:]
            var realtime = false
            var index = 0
            let valuedFlags: Set<String> = [
                "--model-dir", "--input", "--output", "--chunk-ms", "--language", "--compute-units",
                "--expected-languages",
            ]
            while index < arguments.count {
                let flag = arguments[index]
                if flag == "--realtime" {
                    guard !realtime else { throw ReplayError.invalidArguments("Duplicate --realtime") }
                    realtime = true
                    index += 1
                    continue
                }
                guard valuedFlags.contains(flag), index + 1 < arguments.count, values[flag] == nil else {
                    throw ReplayError.invalidArguments("Unknown, duplicate, or incomplete argument: \(flag)")
                }
                values[flag] = arguments[index + 1]
                index += 2
            }
            guard let model = values["--model-dir"], let input = values["--input"], let output = values["--output"]
            else {
                throw ReplayError.invalidArguments("Required: --model-dir PATH --input WAV --output JSONL")
            }
            guard URL(fileURLWithPath: input).pathExtension.lowercased() == "wav" else {
                throw ReplayError.invalidArguments("--input must be a real WAV recording")
            }
            guard let milliseconds = Int(values["--chunk-ms"] ?? "20"), (1...1000).contains(milliseconds) else {
                throw ReplayError.invalidArguments("--chunk-ms must be an integer from 1 through 1000")
            }
            switch values["--compute-units"] ?? "ane" {
            case "ane": computeUnits = .cpuAndNeuralEngine
            case "cpu": computeUnits = .cpuOnly
            case "gpu": computeUnits = .cpuAndGPU
            case "all": computeUnits = .all
            default: throw ReplayError.invalidArguments("--compute-units must be ane, cpu, gpu, or all")
            }
            modelDirectory = model
            self.input = input
            self.output = output
            chunkMilliseconds = milliseconds
            self.realtime = realtime
            language = values["--language"]
            if let allowed = values["--expected-languages"] {
                let codes = allowed.split(separator: ",", omittingEmptySubsequences: false).map {
                    $0.trimmingCharacters(in: .whitespacesAndNewlines)
                }
                guard codes.allSatisfy({ !$0.isEmpty }) else {
                    throw ReplayError.invalidArguments("--expected-languages needs comma-separated nonempty codes")
                }
                expectedLanguages = Set(codes)
            } else {
                expectedLanguages = nil
            }
        }
    }

    enum ReplayError: Error, LocalizedError {
        case invalidArguments(String)
        case invalidAudio
        case outputExists

        var errorDescription: String? {
            switch self {
            case .invalidArguments(let message): return message
            case .invalidAudio: return "Audio must contain finite, nonempty decoded samples"
            case .outputExists: return "Output already exists; choose a fresh JSONL path"
            }
        }
    }

    static func runCLI(arguments: [String]) async {
        do {
            try await run(options: Options(arguments: arguments))
        } catch {
            let message = "Qualification replay failed: \(error.localizedDescription)\n"
            try? FileHandle.standardError.write(contentsOf: Data(message.utf8))
            exit(1)
        }
    }

    static func run(options: Options) async throws {
        let outputURL = URL(fileURLWithPath: options.output)
        guard !FileManager.default.fileExists(atPath: outputURL.path) else { throw ReplayError.outputExists }
        let conversionStart = ProcessInfo.processInfo.systemUptime
        let samples = try AudioConverter().resampleAudioFile(path: options.input)
        guard !samples.isEmpty, samples.allSatisfy({ $0.isFinite }) else { throw ReplayError.invalidAudio }
        let conversionSeconds = ProcessInfo.processInfo.systemUptime - conversionStart
        let configuration = MLModelConfiguration()
        configuration.computeUnits = options.computeUnits
        let manager = StreamingNemotronMultilingualAsrManager(configuration: configuration)
        let modelStart = ProcessInfo.processInfo.systemUptime
        try await manager.loadModels(from: URL(fileURLWithPath: options.modelDirectory))
        let modelLoadSeconds = ProcessInfo.processInfo.systemUptime - modelStart
        await manager.setLanguage(options.language)
        #if QUALIFICATION_FORK
        await manager.setExpectedLanguages(options.expectedLanguages)
        #else
        guard options.expectedLanguages == nil else {
            throw ReplayError.invalidArguments("This build does not support --expected-languages")
        }
        #endif
        guard FileManager.default.createFile(atPath: outputURL.path, contents: nil) else {
            throw ReplayError.invalidArguments("Could not create output file")
        }
        let handle = try FileHandle(forWritingTo: outputURL)
        defer { try? handle.close() }
        #if QUALIFICATION_FORK
        let latestLanguageSupported = true
        #else
        let latestLanguageSupported = false
        #endif
        try write(
            [
                "event": "start", "schema_version": 2,
                "mode": options.realtime ? "realtime_causal" : "offline_causal",
                "input": options.input, "model_directory": options.modelDirectory,
                "sample_rate": 16000, "sample_count": samples.count,
                "chunk_ms": options.chunkMilliseconds, "language_prompt": options.language as Any? ?? NSNull(),
                "compute_units": String(describing: options.computeUnits),
                "model_load_seconds": modelLoadSeconds, "audio_conversion_seconds": conversionSeconds,
                "latest_language_supported": latestLanguageSupported,
                "expected_languages": options.expectedLanguages.map { $0.sorted() } as Any? ?? NSNull(),
                "language_timing_semantics": "snapshot available after chunk; not acoustic boundary",
                "first_language_semantics":
                    "first accepted decoder-emitted language tag; excludes soft estimates and forced prompts",
            ], to: handle)
        let chunkSamples = options.chunkMilliseconds * 16
        let replayStart = ProcessInfo.processInfo.systemUptime
        var chunkIndex = 0
        for start in stride(from: 0, to: samples.count, by: chunkSamples) {
            let end = min(start + chunkSamples, samples.count)
            let audioEnd = Double(end) / 16000
            if options.realtime {
                let wait = replayStart + audioEnd - ProcessInfo.processInfo.systemUptime
                if wait > 0 { try await Task.sleep(for: .seconds(wait)) }
            }
            let processStart = ProcessInfo.processInfo.systemUptime
            _ = try await manager.process(samples: Array(samples[start..<end]))
            let processSeconds = ProcessInfo.processInfo.systemUptime - processStart
            let text = await manager.getPartialTranscript()
            var row = await languageFields(manager: manager)
            row.merge([
                "event": "chunk", "chunk_index": chunkIndex,
                "audio_start_seconds": Double(start) / 16000, "audio_end_seconds": audioEnd,
                "process_seconds": processSeconds,
                "elapsed_seconds": ProcessInfo.processInfo.systemUptime - replayStart,
                "partial_text": text,
            ]) { _, new in new }
            try write(row, to: handle)
            chunkIndex += 1
        }
        let finishStart = ProcessInfo.processInfo.systemUptime
        let finalText = try await manager.finish()
        let finishSeconds = ProcessInfo.processInfo.systemUptime - finishStart
        var final = await languageFields(manager: manager)
        final.merge([
            "event": "complete", "status": "ok", "chunk_count": chunkIndex,
            "audio_end_seconds": Double(samples.count) / 16000,
            "finish_seconds": finishSeconds,
            "elapsed_seconds": ProcessInfo.processInfo.systemUptime - replayStart,
            "final_text": finalText,
        ]) { _, new in new }
        try write(final, to: handle)
        try handle.synchronize()
    }

    private static func languageFields(manager: StreamingNemotronMultilingualAsrManager) async -> [String: Any] {
        let first = await manager.detectedLanguage()
        #if QUALIFICATION_FORK
        let latest = await manager.currentDetectedLanguage()
        let observation = await manager.languageObservation()
        let languageObservation: [String: Any] = [
            "decoded_first": observation.decodedFirst as Any? ?? NSNull(),
            "decoded_latest": observation.decodedLatest as Any? ?? NSNull(),
            "inferred": observation.inferred as Any? ?? NSNull(),
            "source": observation.source, "capability": observation.capability,
            "path": observation.path as Any? ?? NSNull(),
        ]
        #else
        let latest: String? = nil
        let languageObservation: [String: Any] = ["capability": "unavailable_build", "source": "none"]
        #endif
        return [
            "language_observation": languageObservation,
            "first_detected_language": first as Any? ?? NSNull(),
            "latest_detected_language": latest as Any? ?? NSNull(),
        ]
    }

    private static func write(_ row: [String: Any], to handle: FileHandle) throws {
        let bytes = try JSONSerialization.data(withJSONObject: row, options: [.sortedKeys])
        try handle.write(contentsOf: bytes)
        try handle.write(contentsOf: Data([0x0A]))
    }
}
#endif
