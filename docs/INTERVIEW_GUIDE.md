# Technical interview preparation

Speak only to the implementation and receipts you have personally inspected. Practice each command and explain its output. AI-assisted implementation is recorded in PROVENANCE.md. These are suggested answers, not invented employment history.

**Why this architecture?** Two small convolutions learn local digit structure. Early pooling cuts spatial compute. A 64-unit head keeps spatial information but dominates parameters. The model has 38,282 parameters and 116,352 MACs per image. The workload fits CPU retraining and permits controlled experiments; it is intentionally narrower than production perception.

**How does training work?** Cross-entropy compares logits with labels. Backpropagation differentiates the loss through the network. AdamW updates weights with a 0.002 learning rate and 0.0001 weight decay. We use batches of 64 and 40 epochs. The lowest validation-loss checkpoint wins; test accuracy never chooses the checkpoint.

**How was leakage prevented?** Validation comes only from the official training file. Mean/std fit only on the remaining training rows. INT8 observers see 512 training samples. Official test files and bytes are fixed and hashed. Split indices persist. We did not adjust hyperparameters after reading test accuracy.

**How did you measure latency?** CPU perf_counter_ns around a synchronous forward on preallocated input, after warmup, with five repetitions and raw samples. Fresh workers isolate conditions. Batch, threads, layout and precision are recorded. Include preprocessing or transfer separately when diagnosing a real service. GPU measurements need explicit synchronization or events.

**What does batch size change?** Larger batches amortize dispatch and reuse weights, potentially increasing images/second. They increase per-batch latency and memory and may violate online response deadlines. Throughput uses mean latency, while the report also gives median and p95. Find the Pareto frontier for the target workload instead of selecting the largest batch by default.

**What does quantization do?** Approximate activations and weights with scale/zero-point mappings, execute supported kernels with integer arithmetic and dequantize logits. Observer calibration estimates ranges from training-only data. Fusion can reduce intermediate traffic. Quantization introduces numerical error and conversion cost. Accuracy, size and latency must each be measured.

**Why might INT8 be slower?** Tiny kernels can be dominated by quantize/dequantize, packing, dispatch, layout transforms or backend setup. Vectorization and kernel availability matter more than nominal bit width alone. Compare actual operators and complete inference under the same conditions. Our measurements may differ by batch and thread count.

**How does memory bandwidth affect inference?** Time is bounded by compute and traffic: approximately max(FLOPs/attainable_FLOP_rate, bytes/attainable_bandwidth) plus overhead, using measured attainable values and actual traffic. Arithmetic intensity is FLOPs per byte moved. Our weights fit in cache; counting weights once is not a DRAM traffic measurement. We have no measured bandwidth counter, so bottleneck classification stays a hypothesis informed by operator and batch/thread behavior.

**How would you diagnose doubled latency?** Reproduce with identical inputs, batch, precision, model/runtime/compiler versions and concurrency. Separate preprocessing, transfer, compute and postprocessing. Compare cold and warm behavior. Check clocks, thermal/power state, contention, thread oversubscription, layout changes, lost fusion, fallback operators, allocations and synchronization. Bisect the runtime/model delta. On an accelerator inspect supported utilization and memory counters; do not assume they exist or invent Tesla telemetry access.

**How does hardware influence model design?** Align shapes and channel counts with supported vector/tile dimensions, reduce costly layout conversions and unsupported operators, consider cache/working-set capacity and memory traffic, and compare accuracy/cost tradeoffs. Batch size and precision are deployment decisions, not just training parameters. The dense head's storage and Conv 2's arithmetic lead to different optimization opportunities.

**How would you validate a compiler optimization?** Pin exact source/configuration/runtime, preserve a baseline, compare outputs within justified tolerances on broad real and edge inputs, measure held-out accuracy, check fallback paths, benchmark cold/compile and warm execution separately, and test the intended input-shape range. Do not count compilation time as steady-state kernel speed or hide it from startup requirements.

**How would results transfer to new hardware?** The test method and data/checkpoint provenance transfer. Latency ratios do not. Recheck operators, precision support, quantization backend, layout, threading, transfers, memory capacity and compiler lowering. Repeat accuracy and performance tests on that platform. No custom Tesla silicon was accessed here.

**What did you personally do?** State only what you can demonstrate: reviewed model and loss code, ran commands yourself if actually done, explained receipt files and identified limitations. A suitable truthful disclosure is: “I used AI assistance to implement the project. The delivered package contains executed tests and measured training and CPU benchmarks. I can walk through the source and reproduce the commands.” Do not claim independent code authorship or personal operation until true.
