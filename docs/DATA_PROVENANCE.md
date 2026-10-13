# Dataset provenance and licensing

Dataset: Optical Recognition of Handwritten Digits, E. Alpaydin and C. Kaynak (1998), UCI Machine Learning Repository. DOI: https://doi.org/10.24432/C50P49. Official dataset and CC BY 4.0 notice: https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits. License: https://creativecommons.org/licenses/by/4.0/.

Data archive: https://archive.ics.uci.edu/static/public/80/optical+recognition+of+handwritten+digits.zip. Exact archive and file SHA-256 digests live in src/anna_neural/data.py and each result manifest. No raw dataset is committed or included in the source release. The downloader extracts only the two named, verified CSV files.

Use original optdigits.tra (3,823 examples) and optdigits.tes (1,797 examples). Do not re-split sklearn's bundled digits test set or call it the original full UCI training/test benchmark. Validation is a stratified 15% split of only optdigits.tra: 3,249 training and 574 validation images. The test file stays fixed across seeds. Dataset bytes are verified on every use. The current manifest reports zero exact-image overlap between the official files; this alone does not establish absence of all possible related samples.

Processing: reshape the first 64 integer attributes to 1x8x8 images, divide by the documented pixel maximum 16, subtract the training-subset mean and divide by its population standard deviation. Record the normalization constants and exact validation/training row indices. No learned normalization, augmentation, oversampling or calibration uses validation/test data. The original UCI preprocessing already reduced source digit images to 8x8 block features; this project does not process raw scanner images.

Training seeds 369, 370 and 371 vary initialization and shuffle order while preserving split seed 369. Seed 369 is the predefined reference, independent of test accuracy. Select checkpoints by minimum validation loss across fixed 40 epochs. Evaluate test only after each training run finishes. Re-evaluation for checkpoint/quantization verification does not change hyperparameters or select a different reference. Continued tuning after observing test results would require a new untouched holdout.

This release modifies the data representation through normalization and a new validation split. Credit UCI and the creators when sharing results. Repository MIT licensing applies to our code, not to third-party data or dependencies. Small data and digit recognition do not establish production vision generalization.
