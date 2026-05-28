# Survey Report — FedVLM-DG (`equimel`)

*A field record of a federated survey campaign. The party occupied seven dermatology registers,
mapped melanoma risk at each station without removing raw images, and reduced the observations to
one equitable atlas. This report documents what was carried, where it was occupied, how it was run,
and what was read back.*

| field | entry |
| --- | --- |
| Report no. | EQ / FedVLM-DG / v0.1.0 |
| Commission | Equitable melanoma screening at scale (npj Digital Medicine, under review) |
| Method name | FedVLM-DG |
| Datum | ViT-B/16 vision tower, BERT-base text tower, embedding dim 512, six Fitzpatrick strata |
| Field method | federated training across simulated clinical sites; only parameters and low-frequency amplitude leave a site |
| Status | research code; the campaign reproduces the manuscript tables |

## 1. Object of the survey

The campaign trains a melanoma-versus-benign classifier across clinical sites that never exchange raw
images. Three instruments are carried into the field together:

- **FedVLA** — a ViT-B/16 image tower and a BERT-base text tower are aligned by an InfoNCE
  contrastive reading over image and ABCDE-descriptor pairs, anchoring the visual representation to
  clinical language so it holds across skin tone.
- **FairDG** — dense stochastic weight averaging, group distributionally robust optimization, and
  frequency-space amplitude sharing, run together at each station.
- **EAP** — the equitable aggregation protocol weights each site by its worst-group AUROC instead of
  its sample count, and doubles a site's weight when that worst-group AUROC drops by at least five
  points between rounds.

Readings are taken across Fitzpatrick skin-type groups, not on pooled accuracy alone.

## 2. Instruments and outfitting

Outfit one station in any of three ways.

- pip:

  ```
  python -m pip install -e .
  ```

- conda:

  ```
  conda env create -f environment.yml
  conda activate equimel
  python -m pip install -e . --no-deps
  ```

- docker:

  ```
  docker build -t equimel:latest .
  docker run --gpus all equimel:latest --config configs/experiment/main.gin
  ```

ONNX emission needs the optional extra: `python -m pip install -e ".[export]"`.

## 3. Stations occupied

Seven public dermatology registers were occupied. Each shard is described by a manifest with the
columns `image_path,label,fitzpatrick,age,sex,site,diagnosis,features`. The three registers without
expert Fitzpatrick labels (ISIC 2019, HAM10000, SLICE-3D) are assigned an ITA-derived proxy stratum.

| station | images | modality | Fitzpatrick | license | source |
| --- | --- | --- | --- | --- | --- |
| ISIC 2019 | 25,331 | dermoscopic | ITA proxy | CC BY-NC 4.0 | https://challenge.isic-archive.com/data/#2019 |
| HAM10000 | 10,015 | dermatoscopic | ITA proxy | CC BY-NC 4.0 | https://doi.org/10.7910/DVN/DBW86T |
| Fitzpatrick17k | 16,577 | clinical | expert I–VI | research-use | https://github.com/mattgroh/fitzpatrick17k |
| DDI | 656 | clinical | balanced I–VI | Stanford AIMI | https://ddi-dataset.github.io/ |
| PAD-UFES-20 | 2,298 | smartphone | expert + metadata | CC BY 4.0 | https://data.mendeley.com/datasets/zr7vgbcyr2/1 |
| Derm7pt | 1,011 | dermoscopic | 7-point checklist | research-use | https://derm.cs.sfu.ca/ |
| SLICE-3D | 42,848 | 3D total-body crops | binary only | CC BY-NC 4.0 | https://www.kaggle.com/competitions/isic-2024-challenge |

A station-preparation note is printed by `scripts/prepare_data.sh`.

## 4. Field procedure

Seven entry points run the campaign. Each accepts repeatable `--config` (a gin file) and `--bind`
(an inline `key=value` override). Every run parameter is a field of `build_runspec`
(`src/equimel/survey/commission.py`); gin `include` statements resolve from the repository root, so
commands are issued from there.

| entry point | role | reads back |
| --- | --- | --- |
| `equimel-train` | run the federated round loop, write a checkpoint | round log + global weights |
| `equimel-evaluate` | score a checkpoint per station | AUROC, equity gap, EqOdds, DemPar |
| `equimel-ablate` | run one factorial ablation cell | the fairness panels |
| `equimel-crosssite` | run the leave-two-out cross-site protocol | in-domain minus OOD drop |
| `equimel-scaling` | occupy one client-count point of the scaling study | scaling curve point |
| `equimel-infer` | emit per-case melanoma probabilities | per-image scores |
| `equimel-export` | export the model to ONNX | one `.onnx` artefact |

Representative occupations:

```
equimel-train     --config configs/experiment/main.gin
equimel-ablate    --config configs/experiment/ablation_vla_fairness.gin
equimel-crosssite --config configs/experiment/supplementary_crosssite.gin
equimel-scaling   --config configs/experiment/supplementary_client_scaling.gin --bind 'build_runspec.num_clients=10'
equimel-train     --config configs/experiment/_smoke.gin
```

The `_smoke.gin` occupation exercises the pipeline on synthetic shards; it is not a scientific run.

## 5. Observations and reductions

The figures below are the paper-reported mean ± standard deviation over 15 seeds. Reading them back
requires the real registers of section 3 and the resource profile of section 6; the smoke occupation
reproduces none of them.

| paper item | command | expected reading |
| --- | --- | --- |
| Table 1 · DDI AUROC | `equimel-evaluate --config configs/experiment/main.gin` | 0.782 ± 0.021 |
| Table 1 · ISIC 2019 AUROC | same | 0.924 ± 0.008 |
| Table 1 · HAM10000 AUROC | same | 0.936 ± 0.017 |
| Table 1 · SLICE-3D AUROC | same | 0.918 ± 0.008 |
| Table 2 · DDI equity gap (full system) | `equimel-ablate --config configs/experiment/ablation_full.gin` | 0.028 ± 0.009 |
| Table 3 · DDI worst-group (FST V–VI) AUROC | `equimel-evaluate --config configs/experiment/main.gin` | 0.771 ± 0.025 |
| Table 3 · DDI equity gap | same | 0.027 |
| Table 4 · average cross-site AUROC drop | `equimel-crosssite --config configs/experiment/supplementary_crosssite.gin` | 3.2 pp |

## 6. Resource expenditure

| resource | value |
| --- | --- |
| accelerators | 5 × NVIDIA A100 40 GB, one per simulated site, in parallel |
| schedule | 100 federated rounds × 5 local epochs |
| wall-clock | ≈ 12 min / round, ≈ 20 h / full run |
| communication | 784 MB / round (parameters) + 2.1 MB / round (low-frequency amplitude) |
| parameters | 196 M (86 M vision + 110 M text) |
| inference | ≈ 52 ms / image |

These figures are reported as measured and are not softened.

## 7. Custody of records

Trained weights are available upon reasonable request; released artefacts carry a SHA-256 manifest.

## 8. Sources and survey ethics

The object is equal screening access across Fitzpatrick skin types. EAP applies a Rawlsian maximin
over demographic groups, and evaluation reports the AUROC equity gap, equalized-odds difference, and
demographic-parity difference next to accuracy. Raw images do not leave a site; only parameters and
low-frequency amplitude spectra are exchanged. The Fitzpatrick scale measures ultraviolet
sensitivity rather than melanin, and the stratum label is a proxy where expert annotation is absent,
so equity claims are bounded by the populations represented in the seven registers.

## 9. Availability of records

> Code for reproducing all experiments, including the FedVLM-DG framework, FL training pipeline, and
> evaluation scripts, will be made available upon publication at [repository URL].

## 10. Survey register

```
configs/        gin fragments (model, data, train) and experiment cells
src/equimel/
    survey/         run commission, gin charting, the gin shim (sextant)
    ground/         dataset sites, the federation consortium, Dirichlet parcels,
                    Fitzpatrick strata, descriptor legend, image rectification
    drafting/       relief (vision tower), toponym (text tower), datum (projection head),
                    grading (classifier head), montage (full-model assembly)
    reconciliation/ alignment (InfoNCE), omnibus (composite objective),
                    equalization (group DRO), spectral (frequency divergence)
    party/          fieldwork (local trainer), averaging (SWAD), fourier (FDA mixer)
    bureau/         consolidation (equitable aggregation, FedBN, worst-group AUROC)
    season/         expedition (federated round loop), archive (checkpoints, seeds)
    atlas/          accuracy, disparity, portability, gather (predictions), significance
    stores/         hardware (device), dossier (io), logbook (logging),
                    entropy (seeding), notation (shared types)
    fieldbook/      the equimel-* commands: chart, review, locate, vary, traverse, sweep, emit
docs/           project-context, implementation-map, deviations
scripts/        launch_train, launch_eval, prepare_data
tests/          shape, gradient, overfit, metrics, invariant, regression, smoke
```

## 11. Citation

```bibtex
@article{fedvlmdg2025,
  title   = {Equitable Melanoma Screening at Scale: A Federated Vision-Language Model with
             Domain Generalization for Skin Cancer Detection Across Underserved Populations},
  author  = {FedVLM-DG Authors},
  journal = {npj Digital Medicine},
  year    = {2025},
  note    = {Under review}
}
```

## 12. Certification

Released under the MIT License; see `LICENSE`. The survey register, file naming, and this report are
particular to this release of `equimel`.
