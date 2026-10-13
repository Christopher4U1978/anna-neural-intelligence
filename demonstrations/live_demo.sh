#!/usr/bin/env bash
set -euo pipefail
# Run from the standalone repository root after environment setup.
python -m unittest discover -s tests -v
python -m anna_neural.cli predict --checkpoint checkpoints/seed369/best.pt --index 0 --out results/live-prediction.json
python -m anna_neural.cli evaluate --checkpoint checkpoints/seed369/best.pt --out results/live-fp32-evaluation.json
python -m anna_neural.cli evaluate --checkpoint checkpoints/seed369/best.pt --int8 checkpoints/seed369/int8.pt --out results/live-int8-evaluation.json
# Fresh unique directory preserves earlier measurements.
python -m anna_neural.cli benchmark-worker --checkpoint checkpoints/seed369/best.pt --int8 checkpoints/seed369/int8.pt --variant fp32 --batch 1 --threads 1 --out results/live-fp32-b1.json
python -m anna_neural.cli benchmark-worker --checkpoint checkpoints/seed369/best.pt --int8 checkpoints/seed369/int8.pt --variant int8 --batch 1 --threads 1 --out results/live-int8-b1.json
