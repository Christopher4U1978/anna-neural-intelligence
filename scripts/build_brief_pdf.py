from pathlib import Path
import json
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
pdfmetrics.registerFont(TTFont("ForgeSans","/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("ForgeBold","/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"))
root=Path(__file__).resolve().parents[1]
tr=json.loads((root/'results/seed369/training.json').read_text());q=json.loads((root/'results/int8-evaluation.json').read_text());rows=json.loads((root/'results/benchmarks/summary.json').read_text())['rows']
styles=getSampleStyleSheet();styles.add(ParagraphStyle(name='TitleForge',fontName='ForgeBold',fontSize=24,leading=29,textColor=colors.HexColor('#163249'),spaceAfter=14));styles.add(ParagraphStyle(name='SubForge',fontName='ForgeBold',fontSize=13,leading=17,textColor=colors.HexColor('#305a91'),spaceBefore=12,spaceAfter=7));styles.add(ParagraphStyle(name='BodyForge',fontName='ForgeSans',fontSize=9.5,leading=13,spaceAfter=6,textColor=colors.HexColor('#243d50')))
styles.add(ParagraphStyle(name='CellForge',fontName='ForgeSans',fontSize=9,leading=12,spaceAfter=0,textColor=colors.HexColor('#243d50')))
story=[]
def para(text,sty='BodyForge'):story.append(Paragraph(text,styles[sty]))
def table(data,widths):
 t=Table([[Paragraph(str(c),styles['CellForge']) for c in row] for row in data],colWidths=widths,hAlign='LEFT');t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e1ebf5')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,-1),.5,colors.HexColor('#c4d1db')),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),3)]));story.extend([t,Spacer(1,8)])
para('ANNA Neural Intelligence','TitleForge');para('Engineering brief | Christopher Payne | Anna Nexus Labs LLC | 13 October 2026')
para('An original PyTorch CNN, trained from random weights and verified locally on CPU. This independently proposed portfolio demonstration has no Tesla endorsement or custom-hardware result. Repository and website publication await owner review.')
para('Measured model results','SubForge')
table([['Variant','Test accuracy','Model state'],['FP32','97.61% (1,754 / 1,797)',f'{tr["weights_bytes"]:,} bytes'],['FP32 channels-last','97.61%; same predictions','Same logical weights'],['INT8 x86','96.72% (1,738 / 1,797)',f'{q["serialized_bytes"]:,} bytes']],[190,180,154])
para('INT8 reduces serialized state by 69.4% and loses 0.89 percentage points of held-out accuracy. Size excludes the optimizer, runtime and activations. No acceptable production accuracy threshold was assumed.')
para('Network and training','SubForge');para('ForgeCNN: 1x8x8 input, Conv 1 (16 channels), ReLU and 2x2 MaxPool, Conv 2 (32 channels) and ReLU, flatten, Linear 512-to-64 with ReLU, then ten logits. 38,282 parameters and 116,352 estimated convolution/linear MACs per image.')
para('UCI Optical Digits original files: 3,249 training, 574 validation and 1,797 test images. Normalization and INT8 calibration use training only. AdamW, learning rate 0.002, weight decay 0.0001, batch 64, fixed 40 epochs. Select minimum validation loss: reference epoch 14. Test data never selects weights or settings.')
para('Observed CPU latency','SubForge')
values=[['Variant','Batch','Median ms','p95 ms']]
for x in rows:
 if x['threads']==1 and x['batch_size'] in (1,32):values.append([{'fp32':'FP32','int8':'INT8','fp32_channels_last':'Channels-last FP32'}[x['variant']],x['batch_size'],f'{x["aggregate"]["median_ms"]:.4f}',f'{x["aggregate"]["p95_ms"]:.4f}'])
table(values,[244,60,110,110])
para('One CPU thread. Preallocated inputs. 30 warmups and five repeats of 100 forwards. Complete 36-condition matrix, raw samples, means and repeated variability are in the package.')
story.append(PageBreak());para('Hardware tradeoffs and verification','TitleForge')
story.append(Image(str(root/'results/figures/batch-performance.png'),width=524,height=205));story.append(Spacer(1,10))
para('INT8 is slower at batch one and faster than contiguous FP32 at larger one-thread batches in these observations. Channels-last preserves held-out predictions while improving several conditions. Additional threads do not consistently help.')
para('Measurement scope','SubForge')
para('Xeon Platinum 8573C, shared Linux container, eight-core CPU quota, 8 GiB memory limit. Python 3.12 and PyTorch 2.5.1 CPU. No GPU or custom Tesla hardware. Forward timing excludes preprocessing, but includes model operators and conversions. Process peak RSS includes Python/PyTorch. New-process startup is measured separately. Bandwidth and energy were not measured.')
para('Executed verification','SubForge')
para('29 neural tests pass, covering gradients, data handling, checkpoints, metrics, INT8 kernels/reload and benchmark calculations. Three fixed seeds achieve 97.61%, 97.33% and 97.38%. Repeating seed 369 reproduces identical selected weights and test metrics locally.')
para('A scoped INT8 inference exception handles a PyTorch deterministic-registration limit. FP32 training remains strict. Initial failures and interrupted contended timings are preserved and excluded from final comparisons. Shared-host variability limits causal performance claims.')
para('ANNA 1.3 retains 101 passing tests and strict TypeScript; its live experiment remains unapproved at $0. Website build and 29 site tests pass. Root TypeScript has 184 unchanged pre-existing diagnostics; new routes pass targeted checking. Real desktop/mobile browser QA is unverified.')
para('Review package','SubForge')
para('Source, checkpoints, hashes, raw results, charts, Git history, website patch, slides and interview guidance are included. AI assistance contributed to implementation. Christopher should personally inspect and operate the project before presenting it. Publication, deployment and ANNA integration await review. No employer endorsement is claimed.')
para('Data credit: Alpaydin and Kaynak (1998), UCI Optical Digits, DOI 10.24432/C50P49, CC BY 4.0. Our code: MIT. Full report and limitations: docs/ENGINEERING_REPORT.md and docs/VERIFICATION.md.')
def footer(canvas,doc):
 canvas.setFont('ForgeSans',7);canvas.setFillColor(colors.HexColor('#52697b'));canvas.drawString(44,24,'ANNA Nexus Labs LLC - Local engineering evidence - publication pending review');canvas.drawRightString(568,24,str(doc.page))
out=root/'output/pdf/ANNA-Neural-Intelligence-Engineering-Brief.pdf';out.parent.mkdir(parents=True,exist_ok=True)
SimpleDocTemplate(str(out),pagesize=(612,792),leftMargin=44,rightMargin=44,topMargin=38,bottomMargin=42,title='ANNA Neural Intelligence - Engineering Brief',author='Christopher Payne - Anna Nexus Labs LLC').build(story,onFirstPage=footer,onLaterPages=footer)
print(out)
