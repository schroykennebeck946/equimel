# Project Context — FedVLM-DG code release

This document records the metadata that anchors the implementation to the source manuscript
*"Equitable Melanoma Screening at Scale: A Federated Vision-Language Model with Domain
Generalization for Skin Cancer Detection Across Underserved Populations"* (method name
**FedVLM-DG**). Every field lists the manuscript location it was read from and a confidence label.

    project_name       : equimel                              [HIGH]
    domain             : computational dermatology — federated vision-language equity   [HIGH]
    framework          : PyTorch 2.x (torch.nn + torch.fft), HF transformers backbones  [HIGH]
    venue              : npj Digital Medicine                 [MEDIUM]
    primary_datasets   : 7 datasets (see §6)                  [HIGH]
    compute_target     : 5 x NVIDIA A100 40GB, ~20 h / run    [HIGH]
    hparams_reference  : Methods §4.6 + Tables 1, 2           [HIGH]
    supp_path          : none (SI tables cited in-text, no separate SI file supplied)
    extra_signals      : 7 numbered equations; 2^3 factorial ablation; verbatim code-availability quote; strong equity/ethics framing

    NEEDS_USER_DECISION: 0
    Status: resolved.

---

## 1. project_name → `equimel`
- Derived value: `equimel` (contraction of "equitable melanoma").
- Location: Title (p.1); Abstract L1–L9; method name "FedVLM-DG" used throughout Results §2.
- Confidence: HIGH. The acronym `fedvlm` is reserved by an unrelated prior release on a different
  organ system, so a method-faithful but collision-free name was selected.

## 2. supp_path → none
- Derived value: `none`. No separate supplementary PDF accompanies the manuscript.
- Location: Supplementary Tables S1–S30+ are cross-referenced in Results §2.1–§2.5 and Discussion §3
  (e.g. Supplementary Tables 6, 7, 8, 9) but no SI file is bundled.
- Confidence: HIGH. Sibling-location globbing found no `*supp*`, `*_si.*`, or `appendix*` file for
  this manuscript.

## 3. domain → computational dermatology, federated vision-language equity
- Derived value: skin-cancer (melanoma) screening across Fitzpatrick skin types, under federated
  learning with vision-language alignment and domain generalization.
- Location: Abstract; Introduction §1; Keywords (federated learning, vision-language model, domain
  generalization, skin cancer, health equity, Fitzpatrick skin type, melanoma screening).
- Confidence: HIGH.

## 4. framework → PyTorch 2.x
- Derived value: PyTorch 2.x with `torch.nn`, `torch.fft` (frequency-space augmentation), and Hugging
  Face `transformers` / `torchvision` for the dual encoders. Federated training is a single-process
  simulation.
- Location: Methods §4.2 (ViT-B/16 vision transformer, BERT-base text encoder), §4.6 ("One computer
  cluster simulates federated training"), §4.6 (Adam optimizer, cosine decay), §4.3 (frequency-space
  amplitude via FFT, Eq. 5).
- Confidence: HIGH for PyTorch from the operator/class-name fingerprints (ViT-B/16, BERT-base,
  BiomedCLIP / PubMedBERT initialization, FedBN, SWAD); the simulation paradigm is stated explicitly.

## 5. venue → npj Digital Medicine
- Derived value: npj Digital Medicine.
- Location: Reference list is dominated by npj Digital Medicine entries (refs [2,3,4,6,19,20,47,48,
  57,58,59,60]); structured abstract + keyword block + numbered Methods-last layout match the npj
  template; Data/Code-availability and Competing-interests headings follow Nature-portfolio style.
- Confidence: MEDIUM. The template is consistent with npj Digital Medicine but the manuscript carries
  no explicit "submitted to" line; placeholders ("[To be added]", "[repository URL]") indicate a
  pre-submission draft.

## 6. primary_datasets → 7 public dermatology datasets
Each entry lists {name, size, modality, demographic labels, license, access URL}. Licenses not stated
in the manuscript were read from each dataset's canonical distribution page.

| name | size | modality | FST labels | license | access |
| --- | --- | --- | --- | --- | --- |
| ISIC 2019 | 25,331 | dermoscopic | none (ITA proxy) | CC BY-NC 4.0 | https://challenge.isic-archive.com/data/#2019 |
| HAM10000 | 10,015 | dermatoscopic | none (ITA proxy) | CC BY-NC 4.0 | https://doi.org/10.7910/DVN/DBW86T |
| Fitzpatrick17k | 16,577 | clinical | expert FST I–VI | research-use (DermaAmin / Atlas Dermatologico) | https://github.com/mattgroh/fitzpatrick17k |
| DDI | 656 | clinical | balanced FST (208 I–II / 241 III–IV / 207 V–VI) | Stanford AIMI research-use | https://ddi-dataset.github.io/ |
| PAD-UFES-20 | 2,298 | smartphone | FST + age/sex/site metadata | CC BY 4.0 | https://data.mendeley.com/datasets/zr7vgbcyr2/1 |
| Derm7pt | 1,011 | dermoscopic + clinical | 7-point checklist | research-use (registration) | https://derm.cs.sfu.ca/ |
| SLICE-3D | 42,848 | 3D total-body crops | none (binary only) | CC BY-NC 4.0 | https://www.kaggle.com/competitions/isic-2024-challenge |

- Location: Methods §4.5 (Datasets and preprocessing); Table 1 column headers; Results §2.1.
- Confidence: HIGH for names/sizes/modalities (stated); MEDIUM for licenses (read from canonical
  source pages, not the manuscript — the Data-availability statement only says "can be found here").
- All datasets are de-identified public research releases; none are private/restricted.

## 7. compute_target → 5 x A100 40GB
- Derived value: NVIDIA A100 (40 GB) x 5 clients trained in parallel; ~12 min per federated round;
  ~20 h for a full 100-round run; communication 784 MB/round (model parameters) + 2.1 MB/round
  (low-frequency amplitude spectrum); inference ~52 ms/image; total model 196 M parameters
  (86 M vision + 110 M text).
- Location: Methods §4.6 (training details), §4.6/Discussion §2.5 ("100 federated rounds on A100 GPUs
  is 20 hours"; "784 MB"; "52 milliseconds"); Table 2 (Params column, 86 M / 196 M).
- Confidence: HIGH (explicitly reported, not estimated).

## 8. hparams_reference → Methods §4.6 + Tables 1–2
- Derived values: K=5 simulated sites; Dirichlet partition concentration α=0.5; T=100 rounds; E=5
  local epochs/round; Adam optimizer, initial lr η=1e-4, cosine learning-rate decay, weight decay
  1e-4; batch size B=32; vision ViT-B/16 initialized from BiomedCLIP (86 M params); text BERT-base
  initialized from PubMedBERT (110 M params); contrastive temperature τ=0.07 (learnable);
  VL weight λ_vl=0.5; frequency weight β=0.3; aggregation temperature τ_agg=0.1; 15 random seeds.
- Location: Methods §4.6 (federated learning simulation, full hyperparameter prose); §4.2 (Eq. 3,
  λ_vl=0.5; τ=0.07); §4.3 (Eq. 6, β=0.3); §4.4 (Eq. 7, τ_agg=0.1, ≥5pp drop → 2x weight); §4.9
  (15 seeds, bootstrap, McNemar, Cohen's d).
- Confidence: HIGH.

## 9. extra_signals
- Numbered equations: 7 (Eq. 1 maximin objective; Eq. 2 VL contrastive; Eq. 3 local objective; Eq. 4
  group DRO; Eq. 5 frequency KL; Eq. 6 DG objective; Eq. 7 EAP weighting).
- Algorithm boxes: none numbered; the three protocols (FedVLA, FairDG, EAP) are described in prose.
- Released checkpoints: not mentioned → README states "available upon reasonable request".
- Proprietary tokenizer: none (standard BERT / PubMedBERT vocabulary).
- Ablations: full 2^3 factorial over {VLA, DG, Fairness} = 8 configurations (Table 2); per-Fitzpatrick
  equity table (Table 3); leave-two-out cross-site generalization (Table 4); SI-only studies (client
  scaling K=2..10, data efficiency 25%/100%, text-removed-at-inference, adversarial robustness,
  calibration, pairwise component synergy).
- Ethics / fairness: strong — Rawlsian maximin framing (EAP), Equalized Odds / Demographic Parity
  reporting, FQHC deployment motivation, Fitzpatrick-vs-ITA/Monk skin-tone discussion, gradient-
  inversion privacy noted as future work.
- Code-availability statement (verbatim): "Code for reproducing all experiments, including the
  FedVLM-DG framework, FL training pipeline, and evaluation scripts, will be made available upon
  publication at [repository URL]."
- Statistical-analysis stack (verbatim): Python 3.10, scikit-learn v1.3, scipy v1.11.
