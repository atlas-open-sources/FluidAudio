# Operation-owned ANE scratch buffers

`ANEMemoryOptimizer` is now a stateless, `Sendable` allocator and copy utility. Each overlapping inference operation owns a separate `ANEScratchBuffers` session. A session reuses keyed arrays through sequential iterations and remains alive across awaited inference; it is intentionally **not Sendable**.

This is a source-breaking migration of the previous shared keyed pool. `getPooledBuffer`, `initializePooledBuffer`, and `clearBufferPool` have been removed from `ANEMemoryOptimizer`. They returned the same mutable array while previous callers still held it. Locking the dictionary did not protect those callers' writes or in-flight model inputs.

| Previous allocator API | Operation-owned replacement |
|---|---|
| `optimizer.getPooledBuffer(key:shape:dataType:)` | `scratch.buffer(key:shape:dataType:)` |
| `optimizer.initializePooledBuffer(key:shape:dataType:)` | `scratch.prepareBuffer(key:shape:dataType:)` |
| `optimizer.clearBufferPool()` | `scratch.removeAll()` |

Create `scratch` using `optimizer.makeScratchBuffers()` **inside each operation**, before the loop. Share the allocator, never the scratch session, between concurrent operations. Do not place a scratch session in a copyable model/worker container used by overlapping calls. An actor property is appropriate only when every use is synchronous and cannot reenter while an input is in flight.

```swift
let allocator = ANEMemoryOptimizer() // May be shared.

func processOperation() async throws {
    let scratch = allocator.makeScratchBuffers()
    for step in steps {
        let input = try scratch.buffer(key: "input", shape: inputShape, dataType: .float32)
        // Populate all input elements required by this step.
        let provider = ZeroCopyDiarizerFeatureProvider(
            features: ["input": MLFeatureValue(multiArray: input)]
        )
        let result = try await model.prediction(from: provider)
        // Consume/copy the result before the next step mutates this key's input.
    }
}
```

The example shows ownership placement; `steps`, input population, the real model, and result handling come from the calling pipeline. No new model adapter or alternate renderer is required.

## Lifetime contract

- Same session + key + shape + data type returns the same array without clearing it. Sequential callers must finish the previous read/prediction/output consumption before writing it again.
- Different sessions return independent arrays even if keys and shapes match. An operation may remain suspended while another operation fills its own same-key buffer.
- A changed shape/type replaces only that session's cached reference. `prepareBuffer` explicitly replaces it with zero-initialized storage. Allocation failure leaves the previous cached reference intact.
- `removeAll` drops only that session's references. Arrays retained by a caller, feature provider or other owner remain valid. It does not force release of externally retained bytes or clear another operation's storage.
- Errors and cancellation end the operation's ownership naturally. There is no shared return queue, early recycling, or attempt to reuse another operation's still-live input.
- The non-Sendable session is a single-operation API, not an exclusive-lease broker. Do not pass it between concurrent tasks or reuse it through reentrant calls on the same actor. A buffer or raw pointer must not escape to a task that outlives its intended use.

## Shipped callers

- **VAD:** one actor-private session; buffer fill, synchronous prediction and output copying occur without suspension in `processUnifiedModel`.
- **Offline segmentation:** a new session for each `process(audioSource:...)` call; warmup/batch inputs survive awaited predictions and are reused only afterward.
- **Offline embeddings:** a new session for each extraction stream. Row-major/transposed scratch is retained through awaited PLDA flushes and reused across later chunks within that operation.
- **PLDA:** each transform's one warmup uses a fresh local input. Actual transform batch inputs already had per-call ownership.

Numerical operations, shapes, model selection, feature names and output types are unchanged. The VAD benchmark's unrelated empty allocator/clear was removed; it never cleared the manager's actual buffers.

## Verification and remaining boundaries

CPU regressions use real aligned `MLMultiArray` buffers and real `ZeroCopyDiarizerFeatureProvider` retention across asynchronous rendezvous. Concurrent tests exercise overlapping operations sharing the allocator, distinct same-key storage, sequential reuse, clear, throw and cancellation. They do not require a model, generated audio, a GPU or network access. These tests prove the new operation ownership contract; they do not certify the removed shared-pool API.

Other memory owners retain their existing contracts and need separate review before being advertised as concurrent:

- `MLArrayCache` is an existing actor used by ASR. It removes a buffer on acquisition, but accepts a normally aliased `MLMultiArray` on return and resets it. It is not a consuming exclusive-lease API and was not adopted as a substitute here.
- `SortformerModels` and `OfflineSortformerModels` store mutable arrays directly; copying those structs shares scratch independently of this removed pool.
- Zero-copy views now retain their source through CoreML's deallocator closure, including nested views and retained providers. Both public factories use checked stride-aware bounds and reject negative offsets, invalid shapes/strides, arithmetic overflow and layouts extending beyond the source's last addressable element. Allocator tail padding is not assumed to be available. Views still share mutable bytes with their source: storage retention does not permit concurrent mutation or make them Sendable.

The minimum platforms remain macOS 14 and iOS 17. This migration introduces no new dependency, warning suppression, unchecked Sendable conformance, global actor, or inference serialization.
