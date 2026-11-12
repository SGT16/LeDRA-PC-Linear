# LeDRA PC Linear

Experiment code and recorded results for lesion-conditioned retinal adaptation with **LeDRA-PC-Linear**. The completed study covers seven cohorts, five training seeds, three fixed source folds and every directed cross-cohort transfer.

[GitHub](https://github.com/SGT16/LeDRA-PC-Linear) · [OSF](https://osf.io/4q5te/)

## Method

LeDRA-PC-Linear combines a frozen RETFound-Green encoder, a patch-only bottleneck adapter and a lesion-conditioned decision readout. Four teacher-supervised readers weight same-image lesion queries and complementary non-self references. A linear readout of the signed feature differences is added to the Base prediction. The teacher is used during training and is absent at inference.

The proposed model contains 36,229 trainable parameters. The independently trained exact-recipe Base contains 25,377 and removes the lesion branch and auxiliary objective together. The code also includes the matched component controls and benchmark comparison configurations.

## Completed experiments

| Phase | Configurations | Training conditions | Source evaluation records | External conditions |
|---|---:|---:|---:|---:|
| Benchmark | 15 | 1,575 | 3,150 | 9,450 |
| Matched mechanism analysis | 6 | 630 | 1,260 | 3,780 |
| Total | Two separate phases | **2,205** | **4,410** | **13,230** |

Each source cohort is evaluated on the other six cohorts, giving **42 source–target directions**. Both phases preserve all seven cohorts and all directions.

| Cohort | Retained images | Scoring unit |
|---|---:|---|
| BRSET | 7,282 | Patient |
| DeepDRiD | 1,120 | Patient |
| mBRSET | 4,536 | Patient |
| Messidor-2 | 1,740 | Examination |
| SUSTech-SYSU | 1,219 | Image |
| Paraguay v0.3 | 1,437 | Image |
| APTOS | 3,662 | Image |

Training seeds are **823, 289, 592, 605 and 329**. They share the original fixed source folds. Training uses up to 100 epochs with patience 20, source-INNER checkpoint selection and a source-INNER F1 threshold that remains fixed during target evaluation.

The six matched configurations are `LINEAR_REF`, `LINEAR_SHUFFLED`, `LINEAR_UNIFORM_REFERENCE`, `LINEAR_NOAUX`, `LINEAR_POOL_ONLY` and `BASE_MATCHED`. Benchmark `GLOBAL_LINEAR`, historical `B0`, benchmark `B0_MATCHED` and matched-phase models retain their own experiment identities and results.

## Reproduce the recorded numerical results

The aggregate replay requires **Python 3.10 or later** and the standard library. It runs offline without a GPU or dataset credentials.

From the repository root:

```console
python reproduce/verify_files.py
python reproduce/verify_aggregates.py
```

To write newly calculated full-precision summaries to a separate directory:

```console
python reproduce/verify_aggregates.py --write-recomputed replayed_results
```

The verifier checks file checksums, condition identities, seed/fold coverage, all source–target directions, means, sample standard deviations and exact-Base contrasts. The recorded release check passed **18,032 checks** at an absolute numerical tolerance of `1e-12`.

## Training and external evaluation

The experiment implementation and phase-specific execution instructions are supplied in [code/RUNNING.md](code/RUNNING.md). Install the recorded environment for the selected phase, then install the experiment package:

```console
python -m pip install --no-deps -e ./code
python -m ledra_experiments --help
python -m unittest discover -s code/tests -v
```

Generate the complete experiment grids without starting training:

```console
ledra plan --phase benchmark --output rerun/benchmark_plan.csv
ledra plan --phase matched --output rerun/matched_plan.csv
```

The release passed 26 CPU tests and eight command-line help checks without initializing CUDA or starting a dataset experiment. Obtain the source datasets and encoder weights under their provider terms and follow `code/RUNNING.md` for local data preparation, training and external inference.

Keep the original group memberships, seeds, source folds, numeric settings, checkpoint selection and source threshold policy when matching the recorded results. Training also requires the lesion-teacher inputs; inference uses the trained task model and encoder. See [DATA_ACCESS.md](DATA_ACCESS.md) for prerequisites.

The public replay recalculates aggregate arithmetic. Rerunning model training or individual-image inference additionally requires authorized data, original controlled membership mappings and the recorded model dependencies. Exact trained weights are not guaranteed across different hardware or software versions.

## Recorded results

The table below compares PC-Linear with the independently trained exact-recipe Base. Entries give **PC-Linear higher / tied / Base higher** point estimates.

| Metric | Internal, 7 cohorts | External, 42 directions |
|---|---:|---:|
| F1 | 2 / 0 / 5 | 22 / 0 / 20 |
| Accuracy | 1 / 0 / 6 | 20 / 0 / 22 |
| Se@Sp95 | 3 / 0 / 4 | 20 / 0 / 22 |
| Sp@Se95 | 2 / 1 / 4 | 19 / 0 / 23 |

The results show condition-dependent gains. PC-Linear also leads benchmark F1 in nine of 42 directions. Complete means, seed variability, paired differences and conditional intervals are retained, including unfavorable results.

F1 and accuracy use each model's fixed source-INNER threshold. Se@Sp95 and Sp@Se95 are retrospective empirical ROC summaries calculated from target labels. Three source-fold metrics are averaged within each seed, then the five seed means are averaged. Repeated predictions are not pooled as independent patients.

The interval analysis uses 2,000 shared dependency-group bootstrap draws and conditions on fixed trained models, seeds and folds. The earlier cohort outcomes were available during method development. These estimates describe the recorded exploratory comparison; they do not establish universal superiority or independent confirmation on an untouched cohort. The Base contrast jointly changes capacity and lesion supervision.

## Files

| Path | Contents |
|---|---|
| `code/` | Model implementations, configurations and training/evaluation entry points |
| `reproduce/` | Offline verifier and aggregation instructions |
| `results/benchmark15/` | Complete benchmark condition records and summaries |
| `results/matched6/` | Matched reference and control results |
| `results/exact_base_comparison/` | All seven internal and 42 external PC-Linear–Base contrasts |
| `results/conditional_intervals/` | Conditional intervals and aggregate analysis records |
| `results/computational_cost/` | Recorded training times, hardware and inference profiles |
| `results/tables/` | Result tables and source map |

The comparison workbook is in `results/LeDRA_PCLinear_vs_ExactBase_AllResults_20261112.xlsx`. Results are provided as numerical CSV, JSON and Excel files. Images, graphics, patient-level records, credentials and model weights are not distributed.

## Use and contact

No additional reuse license is granted for author-created code and results in this release. Existing third-party terms remain in force.

Suleyman Gokhan Taskin · gokhan@bandirma.edu.tr
