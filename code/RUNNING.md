# LeDRA experiment code

This directory contains training, model, loss, preprocessing and evaluation code. Datasets, original membership metadata, trained checkpoints, frozen teacher parameters, encoder weights and token caches must be supplied locally under their applicable access conditions.

## Installation

Use a separate environment. Install the CUDA-compatible PyTorch build for the intended recorded environment before the remaining dependencies. The two recorded requirement files describe different execution phases; do not combine them into one environment.

```bash
python -m pip install -e ./code
python -m ledra_experiments --help
python -m unittest discover -s code/tests -v
```

Installation resolves dependencies but does not download any dataset or model weight and does not start training. The teacher feature route additionally requires Triton on a compatible CUDA/Linux environment. CPU tests exercise synthetic mathematical and training-step fixtures without a scientific data fit or GPU initialization.

## Reproduction levels

The numeric result replay provided elsewhere in this repository reproduces aggregate summaries from the released recorded numeric results. The commands below provide new training and frozen external inference using the original scientific kernels and recorded settings. They are not a claim that all historical fits were rerun from this cleaned release.

Exact historical retraining requires the same legally obtained image bytes, original retained/cohort-harmonized row order, patient/examination/dependency links, fixed FIT/INNER/OUTER memberships, frozen eight-probe teacher/scalers, general encoder weights and original feature cache payloads, together with the corresponding numerical runtime. File hashes must match the recorded artifacts. Regenerating partitions or caches under another runtime does not establish that equality.

The preparation command accepts already harmonized local canonical metadata. It does not reconstruct every original exclusion, disputed identity resolution, label adjudication, access condition or original split from a raw provider CSV. These controlled prerequisites are not silently fabricated or downloaded.

## Local input contract

For each dataset, provide a folder under `local_data/<dataset>/`:

| File | Required content |
|---|---|
| `rows.json` | Original canonical ordered image records, with original local identifiers and evaluation links |
| `splits.json` | Three dictionaries with `fold`, `fit`, `validation` and `holdout` zero-based image indexes |
| `pc.npy` | Original FP16 block-10 tokens `[N,789,384]` |
| `prefix0.npy` | Original FP16 pre-block-0 tokens `[N,789,384]` |
| `prefix2.npy`, `prefix6.npy`, `prefix10.npy`, `prefix11.npy` | BF16 frozen-prefix computation from `prefix0.npy`, stored as FP32 without FP16 requantization |
| `features.npy` | FP32 cached final encoder features `[N,384]`, used only for benchmark linear-probing training/internal scoring |
| `fov.npy` | Boolean RGB-derived field-of-view map `[N,784]` |
| `teacher.npy` | FP32 frozen minimum-B06/B12 lesion maps `[N,784,4]`, needed only by auxiliary-supervised fits |

All rows require `dataset`, `source_image_key`, `image_id`, `dependency_group_id`, and own-image `image_label`. Image cohorts additionally retain original `label`, `metric_eligible`, `scoring_unit`, `role` and `split`. Patient/examination cohorts retain their original `patient_id`, `patient_label` and `eligible_bilateral` fields; the historical Messidor examination grouping is preserved rather than inventing person identities. Unknown own-image labels remain unknown and are excluded from classification loss.

Image extraction additionally needs `image_path` and, when available, the original image `sha256`. These files stay local and are not part of the public results.

To validate/import exact supplied memberships:

```bash
ledra prepare --dataset brset --rows authorized/brset_rows.json --splits authorized/brset_splits.json --data-root local_data
```

Omitting `--splits` explicitly regenerates a three-fold group-stratified partition using split seed 823 and inner seed `823+1009*fold`. This provides a deterministic new partition, whose identity with the historical split still requires original canonical membership/hash verification.

## Feature and teacher preparation

The frozen general RETFound-Green weight file must have SHA-256 `431de5dbc1bebbb32f60e2c0bcf8daa4f8bcbf06f7cb1e1dc97ec589713942e1`.

```bash
ledra cache --dataset brset --data-root local_data --image-root authorized_images --weights authorized/retfoundgreen_statedict.pth --device cuda
```

This command refuses to overwrite existing caches. It regenerates label-blind arrays under the current runtime; it does not certify equality to historical cache payloads. For exact historical replay, restore and hash-check the original caches instead. The historical PC cache uses a different stored precision/preparation history from the later comparator prefix caches; the two are not substituted for each other.

Teacher probe fitting uses the retained pure float64 image-equal scaler/logistic objective, L2 coefficient .001, zero initialization and fixed L-BFGS-B options. Inputs are the original 44-image PROBE_TRAIN B06/B12 feature arrays and `[44,784,4]` known/unknown patch targets; unknown annotations are `-1`. Expected known-image counts by lesion are MA 44, HE 43, EX 44 and SE 21. The original 10 tuning and 27 test images are not probe-training inputs. Original annotation geometry and the original 44/10/27 memberships must be supplied; the command does not replace missing masks with negative labels.

```bash
ledra fit-teacher --features-b06 authorized/idrid_train_B06.npy --features-b12 authorized/idrid_train_B12.npy --labels authorized/idrid_train_patch_labels.npy --output local_teacher
ledra teacher-cache --dataset brset --data-root local_data --image-root authorized_images --weights authorized/retfoundgreen_statedict.pth --probes local_teacher --device cuda
```

For exact maps, use the original eight frozen combined probe/scaler NPZ files, named `B06_MA.npz` through `B12_SE.npz`, rather than newly refitted approximations. Each contains float64 `coefficients`, scalar `bias`, `mean`, `scale`, and boolean `zero_variance_mask`. The teacher encoder uses the retained fixed IEEE FP32 Triton affine route with per-image float64 probe evaluation; it is distinct from the BF16 classification suffix. The fitted teacher assets and original feature/annotation membership files are not redistributed.

## Complete experiment grids

```bash
ledra plan --phase benchmark --output rerun/benchmark_plan.csv
ledra plan --phase matched --output rerun/matched_plan.csv
```

The benchmark grid contains 15 configurations × 7 datasets × 5 seeds × 3 folds = 1,575 fits and 9,450 directed external conditions. The matched mechanism grid contains six configurations × 7 × 5 × 3 = 630 fits and 3,780 external conditions. The five recorded seeds are `823, 289, 592, 605, 329`; source folds are `1,2,3`. A source fold is not a target subdivision.

Benchmark model identifiers are `B0`, `B0_MATCHED`, `B0_EARLY`, `B0_MIDDLE`, `B0_ALLTOKENS`, `GENERIC_R46`, `GLOBAL_RECT`, `GLOBAL_LINEAR`, `BIDIR_PAIR`, `ALIGN_SHUFFLE`, `LP`, `HOULSBY`, `ADAPTFORMER`, `ADAPTERPLUS`, and `ADAPTERDROP`.

Matched identifiers are `LINEAR_REF`, `LINEAR_SHUFFLED`, `LINEAR_UNIFORM_REFERENCE`, `LINEAR_NOAUX`, `LINEAR_POOL_ONLY`, and `BASE_MATCHED`. `BASE_MATCHED` is the fresh 25,377-parameter independent Base control; `B0_MATCHED` belongs to the earlier benchmark recipe. They are not merged. `GLOBAL_LINEAR` and `LINEAR_REF` are separate trained phase references.

## Training and fixed source thresholds

```bash
ledra train --phase benchmark --model GLOBAL_LINEAR --dataset brset --seed 823 --fold 1 --data-root local_data --weights authorized/retfoundgreen_statedict.pth --output rerun/benchmark/brset_GLOBAL_LINEAR_seed823_fold1 --device cuda
ledra train --phase matched --model LINEAR_REF --dataset brset --seed 823 --fold 1 --data-root local_data --weights authorized/retfoundgreen_statedict.pth --output rerun/matched/brset_LINEAR_REF_seed823_fold1 --device cuda
ledra train --phase matched --model BASE_MATCHED --dataset brset --seed 823 --fold 1 --data-root local_data --weights authorized/retfoundgreen_statedict.pth --output rerun/matched/brset_BASE_MATCHED_seed823_fold1 --device cuda
```

All scientific kernels retain 100 maximum epochs, AP-based patience 20 with log-loss checkpoint tie-break, AdamW learning rate .001 and weight decay .0001. Original complete batching units are preserved. Historical `B0` uses microbatch 32 and its original full-batch loss graph. Other original LeDRA variants use microbatch 8, and published/close-adapter comparator kernels use microbatch 4. Matched six-arm runs use microbatch 8, complete 16-unit batches and the recorded FP16 block-10 inputs/BF16 suffix/FP32 branch arithmetic.

The CLI verifies the recorded training-phase runtime. `--allow-runtime-drift` permits a deliberately labeled new rerun, not an exact historical reproduction. The public Base initializer checks equality against a fresh PC-Linear initializer; proving equality to an actual original per-condition initialization additionally requires its original saved context, which is not public.

Output includes resumable optimizer/RNG state, selected task state, actual epoch history, input/weight hashes, source-INNER thresholds and OUTER metrics. No target labels enter training or source threshold selection. No automatic fleet jobs or operating-system priority changes are installed.

## External inference

```bash
ledra external --fit rerun/matched/brset_LINEAR_REF_seed823_fold1 --target deepdrid --data-root local_data --weights authorized/retfoundgreen_statedict.pth --output rerun/external/brset_to_deepdrid_LINEAR_REF_seed823_fold1 --device cuda
```

Apply each source fit unchanged to the other six datasets. Benchmark external inference uses the native recorded BF16 forward with batch 4; LP runs all encoder blocks from `prefix0.npy` rather than reusing its historical pooled training features. Matched inference retains batch 8. F1 and accuracy use the frozen source-INNER F1 threshold. Se@Sp95 and Sp@Se95 are empirical retrospective target ROC operating-point summaries, with indivisible tied-score groups and no interpolation; they are not target-selected deployment thresholds.

Aggregate the three source-fold metrics within each seed, then report the mean and sample SD over five seed means. Repeated seed predictions are not independent patients, and target probabilities are not pooled into an ensemble unless explicitly labeled as a different experiment.

## BRSET to DRTiD

The DRTiD module evaluates the five benchmark comparators on the ophthalmologist-reviewed two-field dataset. Supply your authorized photographs, the two provider grading tables and the original BRSET checkpoints. The grade files use columns `Macula`, `Optic disc` and `Grade`. Supply the original provider training table first and testing table second. The ordered field-pair/grade hash and the hash of the complete set of original photographs must match the study. All 1,550 eyes are external to the BRSET-trained models. No DRTiD split or teacher is required.

Run the feature step once with the original prefix-extraction environment (PyTorch 2.9.1+cu128, torchvision 0.24.1+cu128, timm 1.0.27, Pillow 12.0.0). It reads each field at 392 × 392, caches the pre-block tokens in FP16 and derives the block-11 tokens for AdapterDrop in FP32 under BF16 autocast. Private caches require approximately 5.3 GiB.

```bash
python -m ledra_experiments.drtid encode --grades "Ground Truths/DR_grade/a. DR_grade_Training.csv" "Ground Truths/DR_grade/b. DR_grade_Testing.csv" --image-root "Original Images" --weights retfoundgreen_statedict.pth --cache local_data/drtid_prefix
```

Run the model scoring step in the recorded native BF16 evaluation environment (PyTorch 2.12.1+cu132, timm 1.0.27). Each model uses its saved BRSET task state. The source-lock table identifies the correct checkpoint by hash and supplies the original `source_INNER_F1`, `source_INNER_Se95` and `source_INNER_Sp95` cutoffs for the selected model, seed and source fold.

```bash
python -m ledra_experiments.drtid score --model LP --seed 823 --fold 1 --checkpoint selected_backbone.pt --source-locks results/drtid/benchmark_source_locks.csv --grades "Ground Truths/DR_grade/a. DR_grade_Training.csv" "Ground Truths/DR_grade/b. DR_grade_Testing.csv" --weights retfoundgreen_statedict.pth --cache local_data/drtid_prefix --output rerun/drtid/LP_seed823_fold1.json
```

The public identifiers of the five models are `LP`, `HOULSBY`, `ADAPTFORMER`, `ADAPTERPLUS` and `ADAPTERDROP`. The full comparison requires 75 evaluations using the five recorded seeds and three source folds. The result JSON contains eye-level summary metrics; field or eye records are not written. The previously evaluated three matched models are included in the downloadable numerical results and aggregate replay.

## Excluded prerequisites

This release supplies scientific kernels and runnable commands, not an unrestricted redistribution of the seven providers' data. It does not include provider downloads, original retained image identities, original split/teacher assets, or source checkpoints. A user with only this public release can replay the released aggregate numeric results and run the CPU fixtures, but cannot regenerate the historical 2,205 learned task checkpoints without obtaining/restoring those prerequisites. No bit-for-bit guarantee across different hardware, numerical kernels or software versions is made.
