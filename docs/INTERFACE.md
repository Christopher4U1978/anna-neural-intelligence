# Standalone inference interface

Implemented command: `python -m anna_neural.cli predict --checkpoint PATH --index 0 --out PATH.json`. It reads a verified held-out dataset image and returns JSON with status, dataset, row index, true class, predicted class and ten softmax probabilities. CLI evaluation returns held-out accuracy/loss, confusion matrix and per-class metrics. These are local demonstration commands, not network services.

Python callable: `ForgeCNN.forward(x)` accepts normalized CPU float32 tensor `[N,1,8,8]` and returns ten logits per image. Normalization constants are carried in the model checkpoint's data manifest. FP32 and INT8 restoration routines return eval-mode models. INT8 requires the scoped inference context when caller strict determinism is enabled. Shapes/normalization are caller responsibilities; the CLI's dataset pipeline enforces its specific file schema.

The CLI has no provider requests, ANNA imports, external credentials, spend authorization or generated-source executor. This interface documents current standalone behavior. A future ANNA-facing service would require separate request validation, provenance, authority, isolation, resource limits and audit design. It is not implemented here.
