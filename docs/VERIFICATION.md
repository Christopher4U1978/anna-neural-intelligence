# Final verification and limitations

## Executed checks

| Check | Actual outcome | Evidence |
|---|---|---|
| ANNA 1.3 isolated original suite | 101/101 pass | results/anna-1.3-baseline.log |
| ANNA strict TypeScript | pass | same log, npm run verify |
| Neural automated tests | 29/29 pass | results/unit-tests-final.log |
| Original neural test attempt | 26 tests, 2 errors; corrected | results/unit-tests-initial-failure.log |
| Reference train/test pipeline | real training, held-out accuracy 97.6071% | results/seed369/training.json |
| Additional seeds | real training, 97.3289% and 97.3845% | results/seed370 and seed371 |
| Same-seed full retraining | identical selected weights, metrics, data and epoch | results/reproducibility.json |
| Checkpoint + optimizer restore | exact logits/state restore tested | unit tests |
| Static INT8 | genuine qint8 Conv/Linear, held-out 96.7168%, exact reload logits | results/int8-evaluation.json and tests |
| Channels-last FP32 | same held-out predictions, numerically compared | results/channels-last-evaluation.json |
| Final CPU benchmark matrix | 36 conditions, 18,000 timed forwards plus warmups | results/benchmarks/*.json |
| Fresh process cold start | three processes per FP32/INT8 variant | results/cold-start.json |
| CPU profiles | FP32 and INT8, batch 32 / two threads | profile text and Chrome traces |
| Live demo script | ran to completion; tests, prediction, re-evaluation and benchmark commands execute | results/live-demo.log |
| Website build | npm test exits successfully; Concordia and vinext build complete | results/website/website-verification.log |
| Full website automated suite | 29/29 pass, including existing game/render checks and four new route checks | results/website/website-all-tests.log |
| Website targeted new-route TypeScript | pass | tsc -p tsconfig.neural-qa.json; empty successful log |
| Website root TypeScript | FAIL: 184 diagnostics, exactly identical to unchanged baseline | both website-typescript logs |
| Presentation | eight slides, package/layout/font/chart gates pass, all final slides rendered and visually inspected | results/presentation-validation.json |
| PDF brief | two pages, rendered and visually inspected | output/pdf/ANNA-Neural-Intelligence-Engineering-Brief.pdf |

## Boundaries and outstanding work

- Website root TypeScript diagnostics arise in existing Concordia aliases, implicit types and Cloudflare bindings. The exact original source revision reproduces the same 184 diagnostics; no new diagnostics were introduced. We preserve this pre-existing limitation and do not change the game/build architecture to hide it. New ANNA routes pass targeted strict checking with a CSS-module declaration.
- Real desktop/mobile browser QA is unverified because the supported managed-browser capability was unavailable. Responsive CSS, semantic headings, table captions/headers, image descriptions and keyboard focus styles are implemented; HTTP render and type checks cannot prove visual behavior on devices.
- No remote CI run, GitHub repository publication, website source push/deployment, production change, Tesla hardware access or GPU measurement occurred. GitHub destination is pending between the two connected accounts.
- Phase 5 integration design/implementation is deferred until standalone owner review. No completed ANNA integration proposal is claimed. The existing standalone inference contract is documented separately.
- Three seeds and one dataset are a small sample. Exact reproducibility is demonstrated only on the current CPU software environment. No cross-platform guarantee, safety certification or production accuracy threshold is assumed.
- INT8 strict deterministic enforcement is scoped off for a PyTorch 2.5.1 kernel registration limitation; FP32 training remains strict. Observer/TypedStorage deprecation warnings are retained.
- Maximum repeat-median CV in the final matrix is about 0.432. Shared-host contention, fixed variant order and uncontrolled clocks limit causal inference. Process RSS is not isolated tensor memory and no sustained bandwidth counters were measured.
- Live-demo single-condition timing commands were checked functionally during other artifact verification. Their numbers are excluded from the controlled comparison. Final report comparisons use only the completed post-training benchmark matrix.
- Initial tests, an early training run without frozen start/end source checks, and a contended/interrupted benchmark are preserved as historical evidence and excluded from final reference comparisons.
- PowerPoint application opening is unverified. Artifact Tool imported/rendered the final PPTX and validated two editable charts with embedded numeric workbooks. Chart numeric values are rounded to twelve significant digits for Excel-compatible snapshots, while raw experiment JSON retains full numeric precision.
- Package installation/data downloads were no-cost local operations. API spend and authorized ANNA live-experiment spend remain $0.

A reviewer should execute README reproduction commands on a separate host and Christopher should operate/explain them personally before interview delivery. Local tests and receipts establish the measured engineering artifact, not personal experience or external independent certification.
