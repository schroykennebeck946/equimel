# Deviations

Intentional differences between this release and the manuscript. Each entry links the manuscript
section and gives the engineering reason.

## D1 — compact backbone fallback for offline tests
- Manuscript: §4.2 / §4.6 — ViT-B/16 (BiomedCLIP, 86M) and BERT-base (PubMedBERT, 110M).
- Release: `drafting/relief.py` and `drafting/toponym.py` load the full encoders when
  `transformers` weights are reachable, and otherwise fall back to dimension-matched compact
  encoders so unit tests and the smoke run execute on CPU without network access. The reporting
  configuration (`configs/experiment/main.gin`) uses the full encoders; the compact path is gated by
  `configs/experiment/_smoke.gin`, which is labelled for unit-test use only.

## D2 — supplementary tables without a supplied SI file
- Manuscript: Results §2.1–§2.5 cross-reference Supplementary Tables S1–S30+.
- Release: no SI file accompanies the manuscript, so supplementary experiments are reconstructed from
  the in-text descriptions (client scaling, data efficiency, text-removed inference, robustness,
  pairwise synergy). Configs are provided under `configs/experiment/supplementary_*.gin`; numeric
  targets quoted in prose are recorded in `docs/implementation-map.md`.

## D3 — proxy Fitzpatrick group for label-free datasets
- Manuscript: §4.5 — datasets lacking Fitzpatrick annotations (ISIC 2019, HAM10000, SLICE-3D) receive
  an ITA-derived proxy group label.
- Release: `ground/strata.py` implements the ITA → Fitzpatrick bucketing exactly as described; the
  78% agreement figure quoted for datasets that carry both labels is a validation statistic, not a
  tunable, and is not re-fit here.
