#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-data}"
mkdir -p "${DATA_ROOT}"

echo "Place the following public datasets under ${DATA_ROOT} and a manifest CSV per shard:"
echo "  isic2019        https://challenge.isic-archive.com/data/#2019"
echo "  ham10000        https://doi.org/10.7910/DVN/DBW86T"
echo "  fitzpatrick17k  https://github.com/mattgroh/fitzpatrick17k"
echo "  ddi             https://ddi-dataset.github.io/"
echo "  pad_ufes_20     https://data.mendeley.com/datasets/zr7vgbcyr2/1"
echo "  derm7pt         https://derm.cs.sfu.ca/"
echo "  slice3d         https://www.kaggle.com/competitions/isic-2024-challenge"
echo
echo "Each manifest must provide columns: image_path,label,fitzpatrick,age,sex,site,diagnosis,features"

python -m equimel.fieldbook.chart --help >/dev/null 2>&1 || true
