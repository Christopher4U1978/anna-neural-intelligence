# Eight-minute engineering demonstration

Before presenting, extract the package, complete README setup and run the demo once. Source-only Git users train the checkpoint first; the portable review package includes small reviewed checkpoints separately from tracked source. A reviewer should not accept checkpoint provenance without independently reproducing training or checking the included receipts.

0:00–0:45 — Objective. “This standalone PyTorch project trains an original CNN and examines how CPU batch size, tensor layout and INT8 precision change accuracy, storage and inference behavior. I proposed the workload myself. It is distinct from ANNA orchestration.”

0:45–1:45 — Open src/anna_neural/model.py and docs/ARCHITECTURE.md. Explain 1x8x8 input, both convolutions, pooling, flatten and the ten logits. Point to the parameter/MAC table. Describe the dense head's parameter cost and Conv 2's arithmetic cost.

1:45–2:45 — Open src/anna_neural/data.py and training.py. Explain official split preservation, train-only normalization, AdamW, loss.backward(), optimizer.step() and validation-loss checkpoint selection. Show results/seed369/learning_curves.json and the plot. The final trained epoch can have higher accuracy but worse validation loss than the selected epoch.

2:45–3:30 — Run a single local prediction command from live_demo.sh. Explain truth/prediction and that softmax is not calibrated certainty. Show held-out confusion matrix, per-class metrics and the fixed reference seed. Clarify digit recognition's narrow scope.

3:30–4:45 — Open docs/BENCHMARK_METHOD.md and results/benchmarks/summary.json. Explain warmup, 500 samples per condition, one inter-op thread, wall-clock CPU timing and synthetic preallocated timing input. Show batch-one latency, p95 and repeated-run spread. Distinguish model-forward latency, first call and complete process startup.

4:45–5:45 — Explain INT8 observer calibration on training data, operator fusion and dequantized output. Show held-out accuracy and serialized state size before and after. Present any slower conditions clearly. Do not assume fourfold size or speed improvement.

5:45–6:45 — Show CPU profile and latency-versus-batch chart. Explain possible dispatch, conversion, layout-copy or parallelism overhead; do not claim a measured memory bandwidth bottleneck without counters. Refer to the complete matrix rather than only the fastest case.

6:45–7:30 — Run tests or show their full log if time is short. Show repeated-seed exact weight reproduction and checkpoint reload. Identify strict INT8 deterministic-registration limitation and scoped inference exception.

7:30–8:00 — Limitations and ownership. “AI assistance helped implement the project. The delivered source has executed tests and measurements. I can explain and reproduce these commands. No Tesla hardware, employer endorsement or production integration is claimed.” Customize this statement only to actions you have personally completed. Invite questions using docs/INTERVIEW_GUIDE.md.

A live full training run takes roughly the recorded seconds on this container, but other machines may differ. Use a new output directory to retain prior evidence. Training can be demonstrated if time allows. CI definition exists, but remote CI has not run until publication and workflow execution are observed.

The included build_deck-work-runtime.mjs records the exact Artifact Tool authoring source and Work runtime paths used for this deck. It requires that runtime and is not a general dependency of the Python project. All raw numeric evidence remains in JSON; PPTX chart snapshots round to twelve significant digits.
