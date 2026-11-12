# Numerical reproduction

The public results contain aggregate metrics for each trained condition, seed means, final summaries, conditional interval endpoints and the 19 main-paper tables. They contain no plots, diagrams, curve-rendering inputs, manuscript files, patient-level prediction rows or fundus photographs. The standard-library replay requires Python 3.10 or later and runs without network access, external packages, a GPU or dataset credentials.

From the repository root, run:

```console
python reproduce/verify_aggregates.py
```

To save newly calculated full-precision summary CSVs and seed means in a separate local directory, run:

```console
python reproduce/verify_aggregates.py --write-recomputed replayed_results
```

The command exits with status zero only when result-file checksums, condition identities, all expected seed/fold memberships, all 42 source-target directions, seed means, sample standard deviations, the PC-Linear-minus-Base differences, and their interval joins pass. Absolute numeric tolerance is `1e-12`. Files under `replayed_results` are generated arithmetic outputs; they are not new experimental observations.

## Preserved experimental phases

| Phase | Configurations | Training conditions | Source evaluation roles | External conditions |
|---|---:|---:|---:|---:|
| Benchmark | 15 | 1,575 | 3,150 | 9,450 |
| Matched mechanism analysis | 6 | 630 | 1,260 | 3,780 |
| Whole study | 21 phase-specific configurations | 2,205 | 4,410 | 13,230 |

The 21 configurations are phase-specific conditions and are not 21 independently proposed architectures. INNER and DEVELOPMENT_OUTER are two records from the same fitted condition. INNER selected checkpoints and source thresholds; it is not an independent test set. Each external direction uses three source-fold models on the same complete target cohort, rather than three disjoint target folds.

The benchmark `GLOBAL_LINEAR` and matched `LINEAR_REF` are separate trained references. Benchmark `B0`, benchmark `B0_MATCHED` and the new matched `BASE_MATCHED` are separate experimental records. They must not be pooled. The matched six-arm phase consists of `LINEAR_REF`, `LINEAR_SHUFFLED`, `LINEAR_UNIFORM_REFERENCE`, `LINEAR_NOAUX`, `LINEAR_POOL_ONLY` and `BASE_MATCHED`.

## Arithmetic and units

Each fit is evaluated at its own saved source-INNER threshold for F1 and accuracy. Metrics are calculated for the original patient, examination or image scoring unit before this release-level arithmetic. Three fixed source-fold metric values are averaged within each training seed. The five seed means are then equally averaged; `seed_SD` is their sample standard deviation (`ddof=1`). Probabilities are not ensembled, and repeated seed predictions are not pooled as independent patients. The training seeds are 823, 289, 592, 605 and 329; source folds are 1, 2 and 3.

Unqualified metric values are proportions in `[0,1]`. `_pp` denotes percentage-point differences. F1, accuracy, Se@Sp95 and Sp@Se95 are higher-is-better. Se@Sp95 and Sp@Se95 are retrospective attainable ROC summaries calculated with target labels, without interpolation; they do not establish transport of a fixed clinical operating point. `legacy_S5` or `S5` fields retained in historical tables preserve the historical eligibility policy and must not be substituted for standard Sp@Se95.

## What is and is not replayed

The public command recalculates model means and seed variability from all 4,410 source-role and 13,230 external condition-level records. It checks all 49 new Base comparison domains and four requested metrics. It checks stored conditional interval endpoints against the reported Base table, and verifies numerical artifact integrity. The 19 table CSVs preserve the final manuscript's displayed cell text; full precision is provided separately.

The command does not retrain models, perform image inference, reconstruct patient membership, regenerate cluster bootstrap draws or establish that a new environment yields bit-identical trained weights. Recomputing the conditional intervals from scratch requires the original individual predictions, dependency-group membership and the fixed resampling inputs under the source datasets' access rules. The experiment code and controlled-input requirements are documented separately in `code/`.

The intervals condition on fixed trained models, seeds and folds, and use 2,000 shared original dependency-group bootstrap draws per cohort. They do not include the full method-search or training-population uncertainty. They are exploratory pointwise percentile intervals without familywise multiplicity adjustment. No p-values, unseen-cohort independent confirmation or prospective clinical performance are implied.

## Numerical result scope

`results/tables/TABLE_SOURCE_MAP.csv` maps each main-paper table to released numerical sources. All graphics and manuscript files are outside this numerical artifact.

The reported cost profiles distinguish recorded training/INNER duration from resident-synthetic full-forward inference. The linear-time profile is an implementation comparison using the same trained checkpoint; it is not a new accuracy experiment. Dense arithmetic counts omit documented unsupported operations and are not exact all-operator FLOPs totals.
