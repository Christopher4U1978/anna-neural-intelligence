# Phase 0 — engineering decision, 2026-10-13

Owner: Christopher Payne, Anna Nexus Labs LLC. Independently proposed portfolio demonstration for Tesla job 265477; no official assignment specification has been supplied. No employer approval or endorsement is claimed.

## Discovered environment
Linux x86_64 container, Intel Xeon Platinum 8573C, 9 visible logical CPUs, cgroup quota 8 cores, memory limit 8 GiB, about 29 GiB free disk at discovery. Python 3.12, NumPy, scikit-learn and matplotlib installed. PyTorch initially absent; CPU PyTorch 2.5.1 installed without paid services. No visible CUDA device or nvidia-smi. Container scheduling and host contention constrain timing.

## Sources
ANNA 1.3 archive: ANNA-1.3-First-Intelligence-Experiment(1).zip, version 1.3.0. Source, tests, authorization protocol and recovery releases inspected in a separate extracted copy. Its local spending authorization is $0. No provider requests or approvals will be made. The source archive is not modified.

Accessible GitHub accounts: Christopher4U1978 and christopher4u1978-cmd. Neither lists a dedicated Neural Intelligence repository. Existing names include A.N.N.A., anna-AI, AETERNA-3.0, Concordia-, Nexus and Samara. No listed repository is assumed to match the newer recovered ANNA 1.3 archive. Destination selection is pending owner review; no remote write is necessary for implementation.

Development website resolved through Sites to the existing Behind the Veil 369 development project, live at https://behindtheveil369.work. Source inspection and review patch are planned; deployment is explicitly withheld.

## Workload decision
Use UCI Optical Recognition of Handwritten Digits, original 3,823 training and 1,797 test images, 8x8 grayscale, 10 classes. Reserve 15% of the original training file for stratified validation. Normalize using only the remaining training subset. Freeze 40 epochs and architecture before test evaluation. Build an original compact CNN; initialize from random weights, no pretrained network.

CIFAR-10 is reachable but is a larger 32x32 color workload and would require longer CPU training and separate licensing review. For an urgent first verified engineering demonstration, Optical Digits gives short local retraining, clear CC BY 4.0 provenance and complete controlled CPU experiments. This is a small-workload demonstration, not a production perception model or evidence of large-model/accelerator skill. Future CIFAR adaptation must be separately measured.

## Incremental plan
1. Implement dataset validation, split manifest, train-only normalization, CNN, AdamW training and validation-selected checkpoints.
2. Test tensor shapes, gradients, data splitting, checkpoint restoration, metrics, deterministic controls and static quantization.
3. Run fixed training seeds 369/370/371 with the same split. Use seed 369 as the predefined reference; do not select based on test results.
4. Calibrate static INT8 only on training images. Check reload and held-out accuracy. Compare FP32, channels-last FP32 and INT8, batches 1/8/32/128, threads 1/2/4, five repeated timing runs in fresh worker processes.
5. Report all measured results, raw timings, CPU operator profile, process memory scope, limitations and failed attempts. Prepare plots, report, walkthrough, technical Q&A and local website patch. Commit a local baseline and development branch. Withhold GitHub publication and website deployment for review.

No ANNA integration implementation is authorized in this phase. A design may follow standalone review. No API credentials are needed.
