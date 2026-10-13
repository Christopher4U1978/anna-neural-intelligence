# ANNA Neural Intelligence

A standalone, hardware-aware neural-network engineering demonstration by Anna Nexus Labs LLC. ForgeCNN learns ten-class UCI Optical Digits classification from random initialization using Python and PyTorch. The project records real held-out accuracy, checkpoint provenance, CPU timings, static INT8 comparisons and limitations.

**Status:** locally trained and tested. The owner approved GitHub publication and the development-site update on October 13, 2026. Public evidence: https://behindtheveil369.work/anna-neural. Source repository: https://github.com/Christopher4U1978/anna-neural-intelligence. No Tesla endorsement, custom silicon, GPU measurement or production integration is claimed. Exact employer assignment requirements have not been supplied. ANNA 1.3 is a separate orchestration system; its 101-test baseline remains intact and its proposed live experiment remains unapproved with $0 authorized spending.

See [engineering report](docs/ENGINEERING_REPORT.md), [architecture](docs/ARCHITECTURE.md), [dataset/license](docs/DATA_PROVENANCE.md), [benchmark method](docs/BENCHMARK_METHOD.md), [provenance](docs/PROVENANCE.md) and [verification](docs/VERIFICATION.md). The portable review package includes small locally produced checkpoints; tracked Git source intentionally excludes data/checkpoints. Reproduce training rather than treating supplied weights alone as evidence.

## Setup

Verified with Python 3.12 on Linux x86_64 CPU. Python 3.11 is permitted but not tested in this release. Use a new environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
python -m unittest discover -s tests -v
```

`requirements-cpu.lock.txt` pins observed transitive CPU dependency versions for Python 3.12. For that path, install the CPU torch wheel above, then `python -m pip install -r requirements-cpu.lock.txt` and `python -m pip install -e . --no-deps`. The lock records versions, not platform wheel hashes; it is not an air-gapped dependency bundle. Dataset retrieval requires HTTPS unless verified files are already in `data/`.

## Training and evaluation

Run each command from this repository root. Output directories for training and complete benchmark matrices must be new/empty to preserve past evidence.

```bash
python -m anna_neural.cli train --config configs/cpu.json --out checkpoints/my-run
python -m anna_neural.cli evaluate --checkpoint checkpoints/my-run/best.pt --out results/my-evaluation.json
python -m anna_neural.cli predict --checkpoint checkpoints/my-run/best.pt --index 0 --out results/my-prediction.json
```

The downloader checks the UCI archive and both CSV hashes. Training reserves validation only from the original training file, fits normalization on the training subset, logs every epoch, saves the best validation-loss checkpoint and evaluates the official test file after training. Architecture/configuration is fixed before test inspection. `best.pt` includes the selected model, optimizer, data manifest, config and epoch. `fp32-weights.pt` stores only model state. The source manifest is checked at training start/end. Optional fixed-seed repetitions use `--seed 370` and `--seed 371` with distinct output directories.

## Quantization and profiling

```bash
python -m anna_neural.cli quantize --checkpoint checkpoints/my-run/best.pt --out checkpoints/my-run/int8.pt
python -m anna_neural.cli evaluate --checkpoint checkpoints/my-run/best.pt --int8 checkpoints/my-run/int8.pt --out results/my-int8-evaluation.json
python -m anna_neural.cli profile --checkpoint checkpoints/my-run/best.pt --out results/my-profile.txt
```

Static INT8 uses the x86 CPU backend, fused operators and training-only calibration. Checkpoint reload must reproduce identical logits locally. Quantized inference temporarily disables strict deterministic enforcement because of a PyTorch 2.5.1 registration limitation; the scoped exception restores the prior setting. FP32 training stays strict. This is documented and tested.

## Benchmarks

Stop training and other CPU-heavy jobs before measuring:

```bash
python scripts/benchmark_matrix.py --checkpoint checkpoints/my-run/best.pt --int8 checkpoints/my-run/int8.pt --out results/my-benchmarks
python scripts/cold_start.py --checkpoint checkpoints/my-run/best.pt --int8 checkpoints/my-run/int8.pt --out results/my-cold-start.json
python -m anna_neural.cli benchmark-worker --checkpoint checkpoints/my-run/best.pt --int8 checkpoints/my-run/int8.pt --variant fp32 --batch 1 --threads 1 --out results/my-single-condition.json
```

The full matrix has 36 conditions, five repeats of 100 timed forwards each and 30 warmups. Raw samples, mean/median/p95, repeat spread, throughput, first-forward/model-load timing and process high-water RSS remain in JSON. Input generation/preprocessing is outside timing. Peak process RSS is not isolated tensor memory. The scripts target CPU only; a GPU comparison requires a separately implemented and synchronized benchmark.

## Checkpoint loading

```python
from anna_neural.training import load_checkpoint
from anna_neural.optimization import load_quantized, inference_context
model, receipt = load_checkpoint("checkpoints/my-run/best.pt")
int8 = load_quantized("checkpoints/my-run/int8.pt", model)
# x must be normalized with receipt['data']['normalization'], shape N x 1 x 8 x 8.
with inference_context(int8):
    logits = int8(x)
```

`torch.load(..., weights_only=True)` is used for state restoration. Load checkpoints only from a trusted source with verified hashes. No checkpoint/model code is automatically executed inside ANNA. Full interrupted-epoch training resumption is not implemented; optimizer state restoration is tested.

## Demonstration and publication

Run `bash demonstrations/live_demo.sh`. Read [eight-minute walkthrough](demonstrations/WALKTHROUGH.md) and [technical interview guide](docs/INTERVIEW_GUIDE.md). The package includes the engineering deck and a reviewed-source website patch. The Git bundle retains main, development branch and source-baseline tag. Restore local history with:

```bash
git clone repository.bundle anna-neural-intelligence-history
cd anna-neural-intelligence-history
git switch operation-neural-forge
```

Christopher4U1978 created the dedicated public repository after approving publication. Source is uploaded through the existing authenticated GitHub connection, using operation-neural-forge before a complete main release. Website version 31 is already published. Original local branch history and review checkpoints remain in the portable review package; GitHub clones can reproduce weights with the training commands. The CI workflow is prepared but has not run on GitHub. Phase 5 ANNA integration remains gated by standalone owner review.

## Troubleshooting

- Unsupported INT8 engine: this release uses x86. Check `torch.backends.quantized.supported_engines`. ARM/other backends need new validation.
- Dataset hash mismatch: stop and investigate upstream changes or a damaged cache. Do not silently bypass digests.
- Output directory exists: choose a new run directory. Prior evidence is retained.
- Source changed during training: retain the failed run, freeze source and start a new run. Do not report its metrics as final evidence.
- Observer or TypedStorage warnings: expected from the pinned torch.ao path; logs retain them. They are distinct from the strict deterministic error corrected in this release.
- Large timing variance: inspect every repeat and host contention; repeat under controlled conditions. CPU threads can slow small batches.
- Lower INT8 accuracy: calibration/precision is a tradeoff. No accuracy threshold or production requirement was assumed. Future tuning must protect an untouched evaluation set.
- Offline use: training can use cached verified data and installed packages. Dependencies and data are not bundled.

## License and scope

Our source uses MIT. UCI data uses CC BY 4.0 with creator attribution. Third-party dependencies retain their own licenses. Tiny digit classification is an engineering demonstration, not a vehicle perception or safety deployment. Hardware performance transfers only after measurements on the target platform. All reported implementation and measurement claims point to local receipts; owner operation and external independent reproduction remain to be completed.

## Regenerate report assets

`python scripts/build_report.py` reads the bundled result JSON and regenerates the full report and plots. The two-page PDF builder additionally uses `python -m pip install reportlab==4.4.9` then `python scripts/build_brief_pdf.py`. These commands reformat retained evidence and do not retrain or invent measurements. ReportLab is optional and not needed for neural training/tests.
