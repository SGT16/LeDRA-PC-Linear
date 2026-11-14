# BRSET to DRTiD

Models trained on BRSET were evaluated on the ophthalmologist-reviewed DRTiD dataset. The same 1,550 eyes (663 positive and 887 negative) were scored for each saved checkpoint. The positive class is moderate or more severe diabetic retinopathy (ICDR grade 2–4). Each eye has a macula-centered photograph and an optic-disc-centered photograph. Its score is the larger of the two field probabilities.

| Training phase | Model identifier | Model | DRTiD conditions |
|---|---|---|---:|
| Matched | LINEAR_REF | LeDRA-PC-Linear (proposed) | 15 |
| Matched | BASE_MATCHED | LeDRA-Base, adapter and classifier only | 15 |
| Matched | LINEAR_POOL_ONLY | LeDRA-PC-Linear, lesion-weighted pooling without complementary reference subtraction | 15 |
| Benchmark | LP | Linear probing | 15 |
| Benchmark | HOULSBY | Houlsby-type adapter | 15 |
| Benchmark | ADAPTFORMER | AdaptFormer | 15 |
| Benchmark | ADAPTERPLUS | Adapter+ | 15 |
| Benchmark | ADAPTERDROP | AdapterDrop | 15 |

The original three-model comparison was specified before DRTiD access. The five benchmark models were added after the three-model results had been seen. All models use saved BRSET checkpoints. Each evaluation uses the complete target cohort. No DRTiD data were used for training, threshold selection or recalibration.

## Results

| File | Unit |
|---|---|
| [conditions.csv](conditions.csv) | A model, source training seed and source fold on all target eyes |
| [seeds.csv](seeds.csv) | Three source folds averaged within a model and seed |
| [models.csv](models.csv) | Mean and sample standard deviation across five seed means for each metric and model |
| [paired_differences.csv](paired_differences.csv) | Within-seed differences between PC-Linear and a comparator, summarized over five seeds |
| [benchmark_source_locks.csv](benchmark_source_locks.csv) | Checkpoint checksum and three fixed BRSET decision thresholds for each of the 75 additional benchmark conditions |

F1 and accuracy use the threshold selected to maximize F1 on the corresponding BRSET validation split. A score equal to the threshold is classified as positive. Sensitivity at specificity of at least 95% and specificity at sensitivity of at least 95% are retrospective DRTiD ROC operating points. Tied scores remain together and no interpolation is applied. Average precision uses the ranking of the labeled eyes. Brier score and expected calibration error (ECE, 10 equal-width bins) are better when lower.

Fold metrics are averaged within each of five source-training seeds. Standard deviations describe variation among those five means. Precision (positive predictive value) is undefined when a source threshold yields zero predicted-positive eyes. Its condition value and any seed, model or paired summary requiring that value are left blank. The same target eyes were evaluated repeatedly. Confirmed participant IDs linking the two eyes of an individual were unavailable, so the release gives descriptive differences and seed variability. It contains no patient-level confidence intervals or hypothesis tests.

To verify the released summaries from the condition-level CSV, run `python reproduce/verify_drtid.py` from the repository root. For inference with the authorized images and trained weights, see [the DRTiD commands](../../code/RUNNING.md#brset-to-drtid).
