import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation,PresentationFile,FileBlob} from '@oai/artifact-tool';
const workspaceDir='/workspace/scratch/5e869b31a612';
const root=workspaceDir+'/anna-neural-intelligence';
const data=async p=>JSON.parse(await fs.readFile(root+'/'+p,'utf8'));
const train=await data('results/seed369/training.json');
const q=await data('results/int8-evaluation.json');
const bench=await data('results/benchmarks/summary.json');
const curves=await data('results/seed369/learning_curves.json');
const SKILL_DIR='/root/.codex/skills/builtins/presentations';
const utils=await import(pathToFileURL(SKILL_DIR+'/container_tools/artifact_tool_utils.mjs').href);
const p=Presentation.create({slideSize:{width:1280,height:720}});
const font='DejaVu Sans', ink='#163249', muted='#465c6e', blue='#305a91', teal='#268675', red='#b85440';
function text(s,name,value,x,y,w,h,size=28,color=ink,bold=false){
 const a=s.shapes.add({name,geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{style:'solid',fill:'none',width:0}});a.text=value;a.text.style={typeface:font,fontSize:size,color,bold};return a;
}
function slide(title,notes){const s=p.slides.add();s.background.fill='#f5f7fa';text(s,'title',title,70,50,1130,86,46,ink,true);s.speakerNotes.text=notes;return s;}
function table(s,values,y=180,widths=[420,220,230,260],height=400){
 const t=s.tables.add({rows:values.length,columns:values[0].length,left:70,top:y,width:1130,height,columnWidths:widths,values});
 for(let r=0;r<values.length;r++)for(let c=0;c<values[0].length;c++){const cell=t.getCell(r,c);cell.fill=r===0?'#dce8f1':'#f5f7fa';cell.text.style={typeface:font,fontSize:23,color:r===0?ink:muted,bold:r===0};}
 t.borders.assign({style:'solid',fill:'#bdcad5',width:1});return t;
}
const pct=a=>(a*100).toFixed(2)+'%';
const get=(v,b)=>bench.rows.find(r=>r.variant===v&&r.batch_size===b&&r.threads===1);
let s=slide('ANNA Neural Intelligence','Independent portfolio demonstration. Owner: Christopher Payne, Anna Nexus Labs LLC. No official Tesla assignment was supplied. No endorsement, Tesla hardware or production integration is claimed. AI-assisted implementation; local test and measurement receipts accompany the package.');
text(s,'subtitle','Hardware-aware CNN engineering',70,195,1070,100,46,blue,true);
text(s,'owner','Christopher Payne\nAnna Nexus Labs LLC',70,370,1000,120,32,ink);
text(s,'status','Locally trained and measured on CPU\nRepository and website publication await review',70,560,1050,90,26,muted);
s=slide('Engineering objective','Source: README.md, PHASE_0.md and PROVENANCE.md. Existing ANNA orchestrates tasks. This project trains a neural model from random weights. They remain standalone. CIFAR-10 extension and accelerator experiments are future work. The smaller UCI workload supports reproducible CPU retraining with clear CC BY 4.0 licensing.');
text(s,'body','Build an original network and preserve evidence for every result.\n\nTrain on real images with an untouched test file.\nCompare precision, layout and batching on the available CPU.\nRetain source, checkpoints, raw timings and failed attempts.',70,180,1110,370,30,ink);
text(s,'boundary','Standalone from ANNA 1.3. No provider API calls or paid compute.',70,600,1100,55,26,muted);
s=slide('ForgeCNN architecture','Source: src/anna_neural/model.py and docs/ARCHITECTURE.md. Random Kaiming initialization, zero biases, logits for cross-entropy. Conv 1 includes ReLU then 2x2 maxpool. Conv 2 also includes ReLU. FC 1 includes ReLU. The dense layer owns most parameters and Conv 2 owns most MACs. Diagram in the report/website is equivalent.');
table(s,[['Layer','Output / batch N','Parameters','MACs / image'],['Input','N × 1 × 8 × 8','0','0'],['Conv 1 + ReLU','N × 16 × 8 × 8','160','9,216'],['MaxPool 2 × 2','N × 16 × 4 × 4','0','0'],['Conv 2 + ReLU','N × 32 × 4 × 4','4,640','73,728'],['Flatten + Linear + ReLU','N × 64','32,832','32,768'],['Linear logits','N × 10','650','640']],155,[390,300,220,220],420);
text(s,'total','38,282 parameters     116,352 MACs per image',70,615,1120,55,28,blue,true);
s=slide('Training and evaluation','Sources: results/seed369/learning_curves.json and training.json; UCI DOI https://doi.org/10.24432/C50P49 and dataset https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits. Original official train/test files. Validation only from training file. Train-only normalization. Fixed 40 epochs; select minimum validation loss, not best test accuracy. Test observed after training. Reference chosen in advance.');
const c=s.charts.add('line',{position:{left:65,top:170,width:700,height:390},categories:curves.map(r=>String(r.epoch)),series:[{name:'Training loss',values:curves.map(r=>Number(r.train_loss.toPrecision(12))),line:{style:'solid',fill:blue,width:2}},{name:'Validation loss',values:curves.map(r=>Number(r.val_loss.toPrecision(12))),line:{style:'solid',fill:red,width:2}}],hasLegend:true,legend:{position:'bottom'},xAxis:{title:'Epoch',textStyle:{typeface:font,fontSize:18,fill:muted}},yAxis:{title:'Cross-entropy',numberFormatCode:'0.0',textStyle:{typeface:font,fontSize:18,fill:muted},majorGridlines:{style:'solid',fill:'#dce3e9',width:1}},chartFill:'#f5f7fa'});utils.applyPresentationChartFont(c,{fontFamily:font});
text(s,'training-facts','3,249 training images\n574 validation images\n1,797 test images\n\nAdamW, LR 0.002\nBatch 64, 40 epochs\nSelected epoch '+train.selected_epoch,820,185,390,350,28,ink);
text(s,'seeds','Three training seeds. Repeating seed 369 reproduced identical selected weights.',70,610,1120,70,25,muted);
s=slide('Quantization tradeoffs','Sources: results/seed369/training.json, int8-evaluation.json and channels-last-evaluation.json. Static x86 INT8 weights and supported kernels, training-only 512-sample calibration. Fused conv/relu and linear/relu, float input/output. Precision/size/latency each measured. Serialized state sizes exclude optimizer and runtime. Quantized restore produced exactly identical logits locally. PyTorch recipe: https://docs.pytorch.org/tutorials/recipes/quantization.html.');
table(s,[['Variant','Test accuracy','Test loss','Model state KiB'],['FP32 contiguous',pct(train.test.accuracy),train.test.loss.toFixed(4),(train.weights_bytes/1024).toFixed(2)],['FP32 channels-last',pct(train.test.accuracy),train.test.loss.toFixed(4),'Same weights'],['INT8 x86',pct(q.test.accuracy),q.test.loss.toFixed(4),(q.serialized_bytes/1024).toFixed(2)]],190,[390,235,235,270],260);
text(s,'tradeoff','INT8 reduces storage by '+((1-q.serialized_bytes/train.weights_bytes)*100).toFixed(1)+'%.\nHeld-out accuracy drops by '+((train.test.accuracy-q.test.accuracy)*100).toFixed(2)+' percentage points.',70,520,1110,100,30,red,true);
s=slide('CPU latency by batch size','Source: results/benchmarks/summary.json and all raw condition JSON files. One intra-op thread, one inter-op thread. Median complete model forward, CPU synchronous timing, preallocated synthetic input, eval/inference_mode. 30 warmups, five repeats of 100 samples. Host contention and nonrandomized order limit causal inference. Mean/p95 and repeat CV are included in the report. No GPU or Tesla silicon benchmark.');
const bc=s.charts.add('bar',{position:{left:65,top:170,width:1140,height:385},categories:['Batch 1','Batch 32','Batch 128'],series:[{name:'FP32',values:[1,32,128].map(b=>Number(get('fp32',b).aggregate.median_ms.toPrecision(12))),fill:blue},{name:'Channels-last FP32',values:[1,32,128].map(b=>Number(get('fp32_channels_last',b).aggregate.median_ms.toPrecision(12))),fill:teal},{name:'INT8',values:[1,32,128].map(b=>Number(get('int8',b).aggregate.median_ms.toPrecision(12))),fill:red}],hasLegend:true,legend:{position:'bottom'},barOptions:{direction:'column',grouping:'clustered'},xAxis:{textStyle:{typeface:font,fontSize:22,fill:muted}},yAxis:{title:'Median forward latency (ms)',numberFormatCode:'0.0',textStyle:{typeface:font,fontSize:20,fill:muted},majorGridlines:{style:'solid',fill:'#dce3e9',width:1}},chartFill:'#f5f7fa'});utils.applyPresentationChartFont(bc,{fontFamily:font});
text(s,'latency-takeaway','INT8 is slower at batch one and faster at larger batches in these observations.',70,600,1130,80,28,ink);
s=slide('Measurement scope and hardware limits','Source: docs/BENCHMARK_METHOD.md, environment receipts, cold-start.json and profile text/Chrome traces. CPU Xeon Platinum 8573C, cgroup quota eight cores, memory 8 GiB, no CUDA. Benchmarks exclude data loading and normalization but include all forward operators/conversions. Peak process RSS includes library/runtime and is not isolated activation memory. Roofline inference requires actual traffic and attainable bandwidth, neither measured here.');
text(s,'hardware','Xeon Platinum 8573C CPU, shared Linux container\n36 conditions across precision, layout, batch and threads',70,180,1110,105,30,blue,true);
text(s,'method','500 timed forwards per condition after warmup\nMean, median, p95 and repeated variability retained\nFresh-process startup measured separately\nOperator profiles identify work without proving a bandwidth bottleneck',70,340,1100,225,28,ink);
text(s,'memory','Process peak RSS includes Python and PyTorch. No GPU or custom silicon results.',70,610,1120,60,25,muted);
s=slide('Verification and technical discussion','Sources: results/unit-tests.log, reproducibility.json, anna-1.3-baseline.log, docs/VERIFICATION.md and INTERVIEW_GUIDE.md. Initial two strict deterministic INT8 errors preserved and corrected with scoped inference exception. CI authored but not run remotely. Website build/render result is separately recorded. Real mobile/desktop browser QA unavailable. Owner must review and operate before claiming personal command execution. Phase 5 integration remains gated.');
text(s,'verification','29 neural tests pass. Checkpoint restore and full-seed reproduction pass.\nANNA 1.3 separately retains 101 passing tests and strict TypeScript.',70,180,1120,120,30,blue,true);
text(s,'discussion','Discussion: latency regression diagnosis, batch tradeoffs, quantization error,\nmemory traffic, compiler validation and transfer to new hardware.\n\nLimits: small digit workload, shared CPU variability and one tested platform.\nAI-assisted implementation. Independent reproduction remains open.',70,355,1120,235,28,ink);
text(s,'next','Next decision: review the source, measurements and proposed publication.',70,630,1120,55,26,muted);
await fs.mkdir(workspaceDir+'/presentation-build/rendered',{recursive:true});
for(let i=0;i<p.slides.items.length;i++){const blob=await p.export({slide:p.slides.items[i],format:'png',scale:1});await fs.writeFile(workspaceDir+'/presentation-build/rendered/slide-'+(i+1)+'.png',new Uint8Array(await blob.arrayBuffer()));}
const stagingDir=workspaceDir+'/.codex-finalizer';await fs.mkdir(stagingDir,{recursive:true});
const candidatePath=stagingDir+'/candidate.pptx';await(await PresentationFile.exportPptx(p)).save(candidatePath);
const finalPath=root+'/demonstrations/ANNA-Neural-Intelligence-Engineering-Demo.pptx';
const result=await utils.finalizePresentation({workspaceDir,candidatePath,finalPath,pythonExecutable:process.env.CODEX_PRIMARY_RUNTIME_PYTHON,integrityValidatorPath:SKILL_DIR+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:SKILL_DIR+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit','--require-native-table-slide','3','--require-native-table-slide','5'],requiredNativeTableOwnerSlides:[3,5],requiredNativeChartOwnerSlides:[4,6],materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'design',families:[font]},verifyArtifactToolImport:true,receiptPath:stagingDir+'/demo.validation.json'});
console.log(JSON.stringify(result));
const restored=await PresentationFile.importPptx(await FileBlob.load(finalPath));
for(let i=0;i<restored.slides.items.length;i++){const blob=await restored.export({slide:restored.slides.items[i],format:'png',scale:1});await fs.writeFile(workspaceDir+'/presentation-build/rendered/final-slide-'+(i+1)+'.png',new Uint8Array(await blob.arrayBuffer()));}
