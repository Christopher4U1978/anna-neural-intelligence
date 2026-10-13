# Benchmark protocol

Run after training and source/build/test tasks finish. Measurements use the available Xeon CPU in a shared container. CUDA and custom accelerator results are absent.

Three variants: FP32 contiguous, FP32 channels-last, and x86 static INT8. For each variant use batches 1, 8, 32 and 128, with intra-op threads 1, 2 and 4 and one inter-op thread. A fresh Python process per condition limits cross-variant state and allows independent process high-water RSS. Variants interleave within each batch/thread condition. This order is fixed, not randomized, and does not eliminate host drift.

Input is preallocated, seeded, synthetic FP32 with shape N×1×8×8. Identical seeded values drive timing, while full real held-out data drive accuracy separately. Inference uses eval and inference_mode. Data loading, normalization, disk access and host/device transfer are excluded from forward latency. Input quantization, all model operators, layout copies and output dequantization remain included.

Record the first model/input forward, then 30 warmup calls and five repetitions of 100 timed calls. CPU operations are synchronous. Use perf_counter_ns around the complete forward. Keep all 500 raw timing samples per condition. Report median, mean, p95, sample standard deviation, repeat median range, repeat-median coefficient of variation and throughput = batch × 1000 / mean_ms. Aggregate p95 describes a mixture of calls across repetitions, not a confidence bound.

First-forward latency is a cold operator/model-input observation inside an already imported process. Model load time excludes Python/PyTorch import. Neither number is a boot-to-response measurement. New-process end-to-end startup is measured separately by scripts/cold_start.py. CUDA would require synchronization/events and a transfer-aware scope, but is not implemented here because no compatible accelerator is available.

Peak memory is Linux ru_maxrss × 1024 in each fresh worker, including Python, PyTorch, checkpoint loading, model, input and warmup allocations. Do not interpret it as isolated activation memory. Weights-only serialized FP32 versus quantized state size is reported separately; optimizer checkpoints are a distinct larger artifact. No system memory-bandwidth counters, energy measurements, exclusive CPU pinning, clock locks or hardware performance counters were available in this experiment.

Torch CPU profiler records 20 forwards at batch 32 / two threads, with shapes and memory tracking after 10 warmups. Its overhead makes profile timings unsuitable as the benchmark. Operator self-time and nested total-time cannot both be summed. Chrome traces are included for inspection.

Strict deterministic FP32 training stays enabled. PyTorch 2.5.1's quantized_resize_cpu_ lacks strict deterministic registration. INT8 accuracy/reload uses a scoped exception that restores the setting, including after failure. Benchmarks run in fresh processes with strict mode disabled, the library default. Exact INT8 logits reproduced locally after reload, but no cross-backend determinism guarantee is made. This release pins the legacy torch.ao eager path for reproducibility; migration to newer torchao/PT2E is future work.

Do not generalize observed latency ratios to other workloads or silicon. More threads can lose on small kernels because scheduling/dispatch and conversion costs outweigh parallel work. A reduced checkpoint does not prove lower process RSS or faster inference. Classification accuracy is checked before interpreting optimization results.
