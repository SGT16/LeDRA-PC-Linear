# Recorded experimental results

This directory contains the recorded numerical evidence for LeDRA-PC-Linear. The completed benchmark phase and matched mechanism phase remain separate. All seven cohorts and all 42 directional transfers are retained, including unfavorable results. No new training or selective rerun was used to prepare these exports.

| Directory | Contents |
|---|---|
| `benchmark15/` | 15 registered arms, 3,150 source-role records, 9,450 external records, seed means and complete summaries |
| `matched6/` | Six matched configurations including the newly trained exact-recipe Base; 1,260 source-role records and 3,780 external records |
| `exact_base_comparison/` | PC-Linear versus the exact-recipe Base in all seven internal domains and 42 external directions, with four primary metrics |
| `conditional_intervals/` | Six-model conditional interval summaries, five paired controls, aggregate calibration and confusion counts, analysis protocols |
| `tables/` | Nineteen result tables in CSV format and a numerical-source map |
| `computational_cost/` | Recorded training timing, hardware/protocol and resident-synthetic inference measurements |
| [`drtid/`](drtid) | Additional BRSET-to-DRTiD evaluation on 1,550 eyes, with per-condition, per-seed and per-model metrics for eight configurations and paired PC-Linear differences |

`LeDRA_PCLinear_vs_ExactBase_AllResults_20261112.xlsx` contains the complete PC-Linear versus exact-recipe Base comparison, including cohort summaries, seed variability, differences and conditional intervals.

## Interpretation

PC-Linear has higher mean F1 than exact-recipe Base in 2/7 internal domains and 22/42 external directions. Its external point gains for accuracy, Se@Sp95 and Sp@Se95 occur in 20/42, 20/42 and 19/42 directions. This is a condition-dependent result, not a claim of overall superiority.

The matched Base has 25,377 trainable parameters versus 36,229 for PC-Linear. It removes the whole lesion decision branch and auxiliary objective together; the comparison does not isolate lesion semantics at equal capacity. Prior development exposed the seven cohorts' outcomes. These external tests evaluate transfer to another dataset for a fitted model, but they are not independent confirmation on a cohort unseen throughout method development.

See `../reproduce/REPRODUCTION.md` for aggregation, metric, uncertainty and replay details. `source_provenance.json` binds exported artifacts to SHA256 hashes of their original numerical sources using relative source labels. `checksums.sha256` verifies the released result bytes. The release contains numerical results only: no plots, architecture diagrams, curve-rendering inputs, case-image galleries or manuscript files. Raw images, original patient/group identifiers, individual prediction rows, encoder/teacher weights and machine-specific operations are not distributed here.
