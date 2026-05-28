# Implementation Map

This is the single source of paper provenance for the release. Source files carry no comments or
docstrings; every manuscript artefact (equation, table, metric, protocol, ablation) is traced to its
implementing module here.

## Equations

| paper loc | artefact | file | symbol |
| --- | --- | --- | --- |
| §4.1 Eq. (1) | maximin objective `min_θ max_g L(θ; D_g)` | `src/equimel/season/expedition.py` | `FederatedLoop.run` |
| §4.2 Eq. (2) | VL contrastive InfoNCE, learnable τ=0.07, ℓ2-norm cosine | `src/equimel/reconciliation/alignment.py` | `VisionLanguageContrast` |
| §4.2 Eq. (3) | local objective `L_local = L_cls + λ_vl·L_vl`, λ_vl=0.5 | `src/equimel/reconciliation/omnibus.py` | `local_objective` |
| §4.3 Eq. (4) | group DRO `max_g L(θ; D_{k,g})`, online mult. weights | `src/equimel/reconciliation/equalization.py` | `GroupDRO` |
| §4.3 Eq. (5) | frequency KL `KL(A_k ∥ Ā)` over LF amplitude | `src/equimel/reconciliation/spectral.py` | `frequency_divergence` |
| §4.3 Eq. (6) | DG objective `L_dg = L_dro + β·L_freq`, β=0.3 | `src/equimel/reconciliation/omnibus.py` | `domain_generalization` |
| §4.4 Eq. (7) | EAP weights `softmax(−AUROC_worst/τ_agg)·n_k`, τ_agg=0.1, ≥5pp→2x | `src/equimel/bureau/consolidation.py` | `EquitableAggregation` |
| §4.2 | total loss `L_total = L_cls + λ_vl·L_vl + L_dg` | `src/equimel/reconciliation/omnibus.py` | `total_objective` |

## Architecture & protocols

| paper loc | artefact | file | symbol |
| --- | --- | --- | --- |
| §4.2 | ViT-B/16 vision encoder, BiomedCLIP init, 86M | `src/equimel/drafting/relief.py` | `VisionTower` |
| §4.2 | BERT-base text encoder, PubMedBERT init, 110M | `src/equimel/drafting/toponym.py` | `LanguageTower` |
| §4.2 | structured clinical text template (ABCDE / FST) | `src/equimel/ground/legend.py` | `render_descriptor` |
| §4.6 | concatenated [CLS]_vision;[CLS]_text → linear head | `src/equimel/drafting/grading.py` | `FusedClassifier` |
| §4.2 | FedBN — BatchNorm buffers stay local | `src/equimel/bureau/consolidation.py` | `is_personal_parameter` |
| §4.3 | SWAD dense checkpoint averaging, overfit-aware window | `src/equimel/party/averaging.py` | `DenseWeightAverager` |
| §4.3 | frequency-space (FDA) augmentation + LF amplitude | `src/equimel/party/fourier.py` | `LowFrequencyMixer` |
| §4.4 | worst-group AUROC per client for EAP | `src/equimel/bureau/consolidation.py` | `worst_group_auroc` |
| §4.5 | Dirichlet non-IID partition, α=0.5 | `src/equimel/ground/parcels.py` | `dirichlet_partition` |
| §4.5 | ITA-based Fitzpatrick proxy when labels absent | `src/equimel/ground/strata.py` | `ita_to_fitzpatrick` |
| §4.6 | round lifecycle, K=5, T=100, E=5 | `src/equimel/season/expedition.py` | `FederatedLoop` |
| §4.6 | atomic checkpoint write + seed persistence | `src/equimel/season/archive.py` | `save_checkpoint` |

## Datasets (§4.5)

| dataset | file | symbol |
| --- | --- | --- |
| ISIC 2019, HAM10000, Fitzpatrick17k, DDI, PAD-UFES-20, Derm7pt, SLICE-3D | `src/equimel/ground/sites.py` | `DermatologyShard`, `dataset_registry` |
| image transforms (resize, normalize, flip/rotate/jitter) | `src/equimel/ground/rectify.py` | `build_transform` |

## Metrics & statistics (§4.8, §4.9)

| paper loc | artefact | file | symbol |
| --- | --- | --- | --- |
| §4.8 | AUROC, sensitivity, specificity, balanced accuracy, F1 | `src/equimel/atlas/accuracy.py` | `discrimination_report` |
| §4.8 | AUROC equity gap `Δ_eq = max_g − min_g AUROC_g` | `src/equimel/atlas/disparity.py` | `auroc_equity_gap` |
| §4.8 | equalized-odds difference | `src/equimel/atlas/disparity.py` | `equalized_odds_difference` |
| §4.8 | worst-group AUROC | `src/equimel/atlas/disparity.py` | `worst_group_auroc` |
| §4.8 | demographic-parity difference | `src/equimel/atlas/disparity.py` | `demographic_parity_difference` |
| §4.8 | cross-site AUROC drop (in-domain − OOD) | `src/equimel/atlas/portability.py` | `cross_site_drop` |
| §4.9 | bootstrap 95% CI, 1000 resamples, stratified | `src/equimel/atlas/significance.py` | `stratified_bootstrap_ci` |
| §4.9 | McNemar paired test + Bonferroni (α/17) | `src/equimel/atlas/significance.py` | `mcnemar_bonferroni` |
| §4.9 | Cohen's d, ceiling d<2.0 | `src/equimel/atlas/significance.py` | `cohens_d` |

## Reported tables / experiments

| table | content | config / command |
| --- | --- | --- |
| Table 1 | AUROC across 7 datasets vs 17 baselines | `configs/experiment/main.gin` → `equimel-evaluate` |
| Table 2 | 2^3 factorial ablation (DDI/Fitz17k/PAD AUROC, equity gap, params) | `configs/experiment/ablation_*.gin` → `equimel-ablate` |
| Table 3 | per-Fitzpatrick equity on DDI (gap, EqOdds, DemPar) | `equimel-evaluate experiment=main split=ddi by_fitzpatrick=true` |
| Table 4 | leave-two-out cross-site generalization | `configs/experiment/supplementary_crosssite.gin` → `equimel-crosssite` |
| SI (scaling) | client scaling K=2..10 | `configs/experiment/supplementary_client_scaling.gin` → `equimel-scaling` |
| SI (efficiency) | data efficiency 25% / 100% | `configs/experiment/supplementary_data_efficiency.gin` → `equimel-train` |

Supplementary studies described only in prose (robustness under Gaussian/adversarial perturbation,
text-removed-at-inference, pairwise component synergy) are reconstructed per the in-text descriptions;
see `docs/deviations.md` D2.

## Headline numbers (for README expected-value table)

- Table 1 mean AUROC (FedVLM-DG): ISIC 2019 0.924±0.008, HAM10000 0.936±0.017, Fitzpatrick17k
  0.893±0.010, DDI 0.782±0.021, PAD-UFES-20 0.901±0.015, Derm7pt 0.894±0.014, SLICE-3D 0.918±0.008.
- Table 2 full system: DDI AUROC 0.782±0.021, equity gap 0.028±0.009 (vs FedAvg 0.142).
- Table 3 DDI worst-group (FST V–VI) AUROC 0.771±0.025; equity gap 0.027; EqOdds 0.031; DemPar 0.024.
- Table 4 cross-site average AUROC drop 3.2pp (best baseline 5.9pp).
