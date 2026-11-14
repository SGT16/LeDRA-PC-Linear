# Data Access

The repository provides experiment code and aggregate results. Obtain source data and model weights directly from their providers under the applicable terms.

| Resource | Source | Preparation requirement |
|---|---|---|
| BRSET 1.0.2 | [PhysioNet](https://physionet.org/content/brset/1.0.2/) | Provider-authorized access; preserve the retained patient groups and source roles |
| DRTiD two-field fundus images | [Provider project and application instructions](https://github.com/FDU-VTS/DRTiD) | Obtain the ophthalmologist-reviewed release under its academic-use agreement. Preserve the provider's macula–disc pairing and eye grade. |
| mBRSET 1.0 | [PhysioNet](https://physionet.org/content/mbrset/1.0/) | Provider-authorized access; preserve exclusions and unknown-label handling |
| DeepDRiD v1.1 regular fundus | [Provider record](https://doi.org/10.1016/j.patter.2022.100512) | Use the recorded regular-fundus version and patient mapping |
| Messidor-2 images | [ADCIS](https://www.adcis.net/en/third-party/messidor2/) | Preserve the official examination pairing |
| Messidor-2 DR labels | [Google Brain grade release](https://www.kaggle.com/datasets/google-brain/messidor2-dr-grades) | Use `adjudicated_dr_grade`; retain the separate annotation provenance |
| SUSTech-SYSU | [Figshare v1](https://doi.org/10.6084/m9.figshare.12570770.v1) | Apply the official treatment-category reclassification table |
| Paraguay v0.3 | [Zenodo](https://doi.org/10.5281/zenodo.4891308) | Preserve the seven original categories and the recorded binary mapping |
| APTOS | [Provider](https://www.kaggle.com/competitions/aptos2019-blindness-detection) and [processed-source instructions](https://github.com/rmaphoh/RETFound/blob/main/BENCHMARK.md) | Match the AutoMorph-processed input source used by the recorded experiments |
| IDRiD A Segmentation | [IEEE DataPort](https://doi.org/10.21227/H25W98) | Required for the fixed lesion-teacher preparation; preserve the recorded 44/10/27 image split |

BRSET and mBRSET acquisition records identify the PhysioNet Credentialed Health Data License v1.5.0. The remaining resources have their own provider terms. Public accessibility does not transfer redistribution rights through this repository.

RETFound-Green pretrained weights are a separate dependency and are not bundled. The recorded model provenance specifies a noncommercial research license. Obtain the correct encoder and check its current terms. Training requires the recorded lesion-teacher dependencies; the teacher is absent at inference. Provider-data conditions and model-weight conditions apply separately from the code.

## Local inputs

Full training and inference require authorized images, original group and fold mappings, known/unknown labels and the phase-compatible feature/cache bindings described in `code/`. The common image input is 392×392. Preserve patient, examination or image scoring units and the original dependency groups. Do not derive new patient identities from filenames or copy patient labels to individually unlabeled images.

The aggregate verifier operates entirely on the released condition-level metrics. Recomputing patient-group bootstrap intervals from scratch requires the original individual predictions and dependency mappings, which are not included in the public archive.

## Distributed material

Included: experiment implementation, recorded settings, condition-level performance metrics, seed summaries, aggregate uncertainty results and numerical timing/hardware records. Result files use CSV, JSON and Excel formats.

Excluded: images, graphics, image/person identifiers, individual prediction rows, controlled membership mappings, credentials, pretrained encoder/teacher weights and trained checkpoints.

No additional reuse license is granted for author-created code and results in this release. Existing third-party licenses and notices remain in force. Contact gokhan@bandirma.edu.tr about author-controlled material.
