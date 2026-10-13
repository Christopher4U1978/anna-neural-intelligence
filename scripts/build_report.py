"""Generate plots/tables directly from retained measurement receipts."""
from pathlib import Path
import json,csv,statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
root=Path(__file__).resolve().parents[1];results=root/'results';docs=root/'docs';figs=results/'figures';figs.mkdir(exist_ok=True)
read=lambda p:json.loads((root/p).read_text())
tr=read('results/seed369/training.json');q=read('results/int8-evaluation.json');cl=read('results/channels-last-evaluation.json');summary=read('results/benchmarks/summary.json');rows=summary['rows']
plt.rcParams.update({'font.size':11,'figure.dpi':130,'axes.spines.top':False,'axes.spines.right':False})
def save(name):
 plt.tight_layout();plt.savefig(figs/(name+'.png'));plt.savefig(figs/(name+'.svg'));plt.close()
curves=read('results/seed369/learning_curves.json');epochs=[r['epoch'] for r in curves]
fig,ax=plt.subplots(1,2,figsize=(11,4.2));ax[0].plot(epochs,[r['train_loss'] for r in curves],label='Training');ax[0].plot(epochs,[r['val_loss'] for r in curves],label='Validation');ax[0].set(xlabel='Epoch',ylabel='Cross-entropy loss',title='Training and validation loss');ax[0].axvline(tr['selected_epoch'],ls='--',color='gray',label='Selected epoch');ax[0].legend()
ax[1].plot(epochs,[r['train_accuracy']*100 for r in curves],label='Training');ax[1].plot(epochs,[r['val_accuracy']*100 for r in curves],label='Validation');ax[1].set(xlabel='Epoch',ylabel='Accuracy (%)',title='Learning curves');ax[1].legend();save('learning-curves')
fig,ax=plt.subplots(figsize=(6,5.2));cm=np.array(tr['test']['confusion_matrix']);im=ax.imshow(cm,cmap='Blues');fig.colorbar(im,ax=ax,label='Images');ax.set(xticks=range(10),yticks=range(10),xlabel='Predicted digit',ylabel='True digit',title='FP32 held-out test confusion matrix')
for i in range(10):
 for j in range(10):
  if cm[i,j]:ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=8,color='white' if cm[i,j]>100 else '#1b2730')
save('confusion-matrix')
colors={'fp32':'#305a91','fp32_channels_last':'#268675','int8':'#b85440'};names={'fp32':'FP32 contiguous','fp32_channels_last':'FP32 channels-last','int8':'INT8 x86'}
fig,axes=plt.subplots(1,2,figsize=(11,4.5))
for variant in colors:
 v=sorted([x for x in rows if x['threads']==1 and x['variant']==variant],key=lambda x:x['batch_size']);batch=[x['batch_size'] for x in v]
 axes[0].plot(batch,[x['aggregate']['median_ms'] for x in v],marker='o',label=names[variant],color=colors[variant]);axes[1].plot(batch,[x['aggregate']['throughput_images_s'] for x in v],marker='o',label=names[variant],color=colors[variant])
for ax in axes:ax.set_xscale('log',base=2);ax.set_xticks([1,8,32,128],labels=['1','8','32','128']);ax.set_xlabel('Batch size');ax.grid(alpha=.2);ax.legend(fontsize=9)
axes[0].set(ylabel='Median forward latency (ms)',title='Latency / one CPU thread');axes[1].set(ylabel='Images / second from mean latency',title='Throughput / one CPU thread');save('batch-performance')
fig,ax=plt.subplots(figsize=(10,4.6))
for variant in colors:
 v=[x for x in rows if x['threads']==1 and x['variant']==variant and x['batch_size']==1][0];samples=[r['summary']['median_ms'] for r in read(f'results/benchmarks/{variant}-b1-t1.json')['runs']];ax.plot(range(1,6),samples,'o-',color=colors[variant],label=names[variant])
ax.set(xlabel='Measurement repeat',ylabel='Repeat median latency (ms)',title='Batch-one repeated timings / one CPU thread',xticks=range(1,6));ax.legend();save('repeat-variability')
fig,ax=plt.subplots(figsize=(8,4.2));vals=[tr['weights_bytes']/1024,q['serialized_bytes']/1024];ax.bar(['FP32 weights state','INT8 quantized state'],vals,color=['#305a91','#b85440']);ax.set(ylabel='Serialized bytes / 1024 (KiB)',title='Model state storage, excluding optimizer')
for i,v in enumerate(vals):ax.text(i,v+1,f'{v:.2f} KiB',ha='center');
save('model-size')
# Exact SVG architecture for the website; this is a labeled technical diagram.
labels=[('Input','N × 1 × 8 × 8'),('Conv 1 + ReLU','N × 16 × 8 × 8'),('MaxPool 2 × 2','N × 16 × 4 × 4'),('Conv 2 + ReLU','N × 32 × 4 × 4'),('Flatten','N × 512'),('Linear + ReLU','N × 64'),('Linear logits','N × 10')]
svg=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 680 730" role="img" aria-labelledby="title desc"><title id="title">ForgeCNN architecture</title><desc id="desc">Two convolution layers, one pooling layer, flatten and a two-layer classifier. Output ten logits.</desc><rect width="680" height="730" fill="#f5f7fa"/>']
for i,(name,shape) in enumerate(labels):
 y=20+i*100;svg.append(f'<rect x="100" y="{y}" width="480" height="76" rx="4" fill="#e1ebf5" stroke="#305a91"/><text x="340" y="{y+29}" text-anchor="middle" font-family="DejaVu Sans,sans-serif" font-size="21" fill="#18344d">{name}</text><text x="340" y="{y+58}" text-anchor="middle" font-family="DejaVu Sans,sans-serif" font-size="18" fill="#305a91">{shape}</text>')
 if i<6:svg.append(f'<path d="M340 {y+77}v21m-5-6 5 6 5-6" fill="none" stroke="#305a91" stroke-width="2"/>')
svg.append('</svg>');(figs/'architecture.svg').write_text(''.join(svg))
with (results/'benchmark-table.csv').open('w') as f:
 writer=csv.writer(f);writer.writerow(['variant','batch','threads','mean_ms','median_ms','p95_ms','std_ms','images_s','repeat_median_cv','peak_process_rss_bytes'])
 for x in rows:writer.writerow([x['variant'],x['batch_size'],x['threads'],*[x['aggregate'][k] for k in ['mean_ms','median_ms','p95_ms','std_ms','throughput_images_s']],x['repeat_median_cv'],x['peak_process_rss_bytes']])
seeds=[read(f'results/seed{s}/training.json') for s in [369,370,371]]
b1=lambda variant:next(x for x in rows if x['threads']==1 and x['batch_size']==1 and x['variant']==variant)
size_ratio=tr['weights_bytes']/q['serialized_bytes'];delta=(q['test']['accuracy']-tr['test']['accuracy'])*100
fp=b1('fp32');iq=b1('int8');lc=b1('fp32_channels_last');cvmax=max(x['repeat_median_cv'] for x in rows)
lines=['# ANNA Neural Intelligence — engineering report','', 'Anna Nexus Labs LLC • Christopher Payne • Operation Neural Forge • 13 October 2026','',
'## Verified outcome','',f'An original 38,282-parameter PyTorch CNN trained from random initialization on UCI Optical Digits. The predefined seed-369 reference correctly classifies {tr["test"]["correct"]}/{tr["test"]["samples"]} held-out images (**{tr["test"]["accuracy"]*100:.2f}% accuracy**). Its best validation-loss checkpoint is epoch {tr["selected_epoch"]} of 40. Automated neural tests pass 29/29. Repeating the reference seed produces bit-identical selected weights and identical test metrics in the same environment. ANNA 1.3 separately reproduces 101/101 tests plus strict TypeScript in an isolated copy.','',
'This is an independent portfolio demonstration for technical discussion, not an official Tesla assignment or endorsed product. No custom Tesla silicon, GPU inference, live model API requests or production changes occurred. Repository and website publication await owner review.','',
'## Data and training','',f'Original official split: 3,823 training-file examples and 1,797 test-file examples. Training-file split: 3,249 training / 574 stratified validation. Mean/std fit only training rows; INT8 calibration uses the first 512 training rows. Data and split manifests are retained. Train-only mean={tr["data"]["normalization"]["mean"]:.8f}, std={tr["data"]["normalization"]["std"]:.8f} after dividing pixels by 16. Exact-image overlap across official files: {tr["data"]["exact_image_overlap_official_train_test"]}.','',
'Fixed configuration: 40 epochs, batch 64, AdamW learning rate 0.002, weight decay 0.0001, two CPU intra-op threads, cross-entropy, no augmentation or pretrained network. Choose checkpoints only by validation loss. Training data ordering and initialization use fixed seeds. The test file was never used for training, normalization, observer calibration, early stopping or reference-model selection.','',
'| Training seed | Selected epoch | Validation accuracy | Test accuracy | Test loss | Training wall time |','|---|---:|---:|---:|---:|---:|']
for r in seeds:lines.append(f'| {r["config"]["seed"]} | {r["selected_epoch"]} | {r["validation"]["accuracy"]*100:.2f}% | {r["test"]["accuracy"]*100:.2f}% | {r["test"]["loss"]:.5f} | {r["training_seconds"]:.2f} s |')
lines+=['',f'Three-seed test accuracy: mean {statistics.mean(r["test"]["accuracy"]*100 for r in seeds):.2f}%, sample standard deviation {statistics.stdev(r["test"]["accuracy"]*100 for r in seeds):.3f} percentage points. These share one test set and are not independent estimates of population generalization. Seed 369 was predefined, not selected from this table. Training wall time includes validation and checkpoint saves, excludes setup/download and final held-out evaluation.','',f'Reference test binomial Wilson 95% interval: {tr["test"]["accuracy_wilson_95"][0]*100:.2f}%–{tr["test"]["accuracy_wilson_95"][1]*100:.2f}%. This interval does not address domain shift, correlated samples or model-selection uncertainty.','',
'![Learning curves](../results/figures/learning-curves.png)','',
'## Hardware and software','',f'{tr["environment"]["cpu"]}, Linux x86_64 shared container, nine visible logical CPUs, cgroup CPU quota eight cores, memory limit 8 GiB. Python 3.12, PyTorch {tr["environment"]["torch"]}, NumPy {tr["environment"]["numpy"]}, scikit-learn {tr["environment"]["sklearn"]}. No CUDA device. Full build configuration and package versions live in results/seed369/environment.json and requirements-cpu.lock.txt. Benchmark workers use one inter-op thread; intra-op count is recorded per row.','',
'## Precision and storage tradeoffs','',
'| Variant | Held-out accuracy | Held-out loss | Serialized model state |','|---|---:|---:|---:|',f'| FP32 | {tr["test"]["accuracy"]*100:.2f}% | {tr["test"]["loss"]:.5f} | {tr["weights_bytes"]:,} bytes |',f'| FP32 channels-last | {cl["test"]["accuracy"]*100:.2f}% | {cl["test"]["loss"]:.5f} | Same logical weights as FP32 |',f'| INT8 x86 | {q["test"]["accuracy"]*100:.2f}% | {q["test"]["loss"]:.5f} | {q["serialized_bytes"]:,} bytes |','',
f'INT8 accuracy changes by {delta:.3f} percentage points ({tr["test"]["correct"]-q["test"]["correct"]} fewer correct classifications). Serialized state is {size_ratio:.2f}× smaller, a {(1-1/size_ratio)*100:.1f}% reduction. This comparison excludes optimizer state, Python/runtime and activation memory. The training checkpoint is {tr["checkpoint_bytes"]:,} bytes. Quantized Conv/Linear weights are genuine qint8, validated in tests. Calibration, conversion, serialization/reload and full held-out inference executed successfully.','',
'![Model size](../results/figures/model-size.png)','',
'## Complete CPU benchmark matrix','',
'Preallocated identical seeded inputs. Forward-only wall-clock CPU scope, 30 warmups, five repeats × 100 timed calls per condition. Mean, median, p95 and repeated variability are distinct. Throughput uses mean latency. Variant order interleaves within every condition. Individual JSON files contain every sample and environment. No concurrent model training or website build ran during the final matrix. The shared host may still contend.','',
'| Variant | Batch | Threads | Median ms | Mean ms | p95 ms | Images/s | Repeat median CV | Peak process RSS MiB |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for x in rows:lines.append(f'| {names[x["variant"]]} | {x["batch_size"]} | {x["threads"]} | {x["aggregate"]["median_ms"]:.4f} | {x["aggregate"]["mean_ms"]:.4f} | {x["aggregate"]["p95_ms"]:.4f} | {x["aggregate"]["throughput_images_s"]:,.0f} | {x["repeat_median_cv"]:.3f} | {x["peak_process_rss_bytes"]/2**20:.1f} |')
lines+=['',f'At batch one / one thread: FP32 median {fp["aggregate"]["median_ms"]:.4f} ms, INT8 {iq["aggregate"]["median_ms"]:.4f} ms ({iq["aggregate"]["median_ms"]/fp["aggregate"]["median_ms"]:.2f}× baseline latency), channels-last FP32 {lc["aggregate"]["median_ms"]:.4f} ms. These are observations of this model/environment, not platform-independent speedup guarantees.','',
'INT8 is slower at batch one but faster than contiguous FP32 at batches 32 and 128 in the one-thread observations. Channels-last FP32 also improves several conditions without changing held-out predictions. Additional threads do not consistently help this tiny network. Conversion, dispatch, layout and parallelism overhead are plausible explanations; only the operator profile and timing behavior are directly measured.','',
f'Maximum repeat-median coefficient of variation across conditions is {cvmax:.3f}. Mean/p95 tails can be much larger than the median, indicating variability that limits strong causal performance claims. Reproduction should compare full distributions and repeat under the target host conditions. No exclusive CPU pinning, fixed clocks or energy measurement is asserted.','',
'![Batch performance](../results/figures/batch-performance.png)','',
'![Repeated batch-one timings](../results/figures/repeat-variability.png)','',
'## Cold start, memory and bottlenecks','',
'Worker first-forward and model-load observations appear in each condition receipt. They exclude interpreter/library import and boot. results/cold-start.json separately records three new processes per FP32/INT8 variant with import, load/configure and first-forward segments and parent launch-to-exit time. OS file pages may remain cached across launches. Do not mix those process-start numbers with warmed forward latency.','',
'Peak process RSS includes interpreter, imported libraries, model reconstruction and warmup allocations. It is not a measurement of isolated inference activations or DRAM bytes transferred. Operator profiles and Chrome traces capture 20 batch-32 forwards after warmup at two threads. Profiling adds overhead and is separate from benchmarking.','',
'The network estimates 116,352 MACs per image, or 232,704 convolution/linear arithmetic FLOPs excluding other operations. Conv 2 owns most MACs; the dense hidden layer owns most parameters. At batch one, 153,128 FP32 weight bytes fit within a modern cache, so simplistic DRAM roofline conclusions are unjustified. We did not measure sustainable bandwidth, hardware counters or achieved vector throughput. Compute/memory/dispatch classification remains a hypothesis to test with counter access.','',
'## Class-level evaluation','',
'![Confusion matrix](../results/figures/confusion-matrix.png)','',
'| Digit | FP32 precision | FP32 recall | FP32 F1 | Test support | INT8 recall |','|---|---:|---:|---:|---:|---:|']
for a,b in zip(tr['test']['per_class'],q['test']['per_class']):lines.append(f'| {a["class"]} | {a["precision"]:.4f} | {a["recall"]:.4f} | {a["f1"]:.4f} | {a["support"]} | {b["recall"]:.4f} |')
lines+=['','## Verification and limitations','',
'29 tests pass across shapes, logits, finite loss, all parameter gradients, optimizer updates, seed control, deterministic one-step training, disjoint splits, train-only normalization, batching, checkpoint and optimizer restore, known metrics, empty-data rejection, real INT8 operators/reload, unsupported backend rejection, channels-last numerical consistency, benchmark arithmetic/samples, source manifests, real-schema pipeline handling and cache tampering. Same-seed full retraining matches selected weights and metrics exactly. Raw receipts preserve the initial deterministic INT8 failure and the interrupted contaminated benchmark.','',
'CI source is prepared but not executed remotely. Source/Git bundle and website patch are reviewable locally. The website build/render checks are documented in VERIFICATION.md; real desktop/mobile browser QA remains unverified because the supported managed-browser capability was unavailable. No public deployment occurred.','',
'The model is small, the dataset is narrow and one local hardware environment cannot establish accelerator or production behavior. Static quantization uses a pinned legacy API with retained deprecation warnings. Model compilation, pruning, ONNX, BF16, GPU comparisons, calibration tuning and CIFAR-10 training are future experiments, not completed capabilities. Phase 5 ANNA integration awaits standalone owner review.','',
'## Reproduction and review','',
'Use README.md commands, fixed configs/cpu.json and exact source/data manifests. results/benchmarks/summary.json and benchmark-table.csv support reanalysis. demonstrations/live_demo.sh, WALKTHROUGH.md and INTERVIEW_GUIDE.md support an honest technical demonstration. Review both accounts before selecting the GitHub owner and approve website deployment separately.','',
'External references: [UCI dataset and license](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits), [PyTorch quantization recipe](https://docs.pytorch.org/tutorials/recipes/quantization.html), [PyTorch x86 INT8 backend](https://docs.pytorch.org/blog/int8-quantization/). Results above come from local executable receipts, not these references.','']
(docs/'ENGINEERING_REPORT.md').write_text('\n'.join(lines))
print('built report and figures from',len(rows),'conditions')
