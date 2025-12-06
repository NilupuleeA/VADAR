#!/bin/bash
cd data
hf download dmarsili/Omni3D-Bench omni3d-bench.zip --repo-type dataset --local-dir .
unzip omni3d-bench.zip -d omni3d-bench
pip install gdown
gdown https://drive.google.com/uc?id=1gWxr19YjQQAvgIUQSsw2Kr4w-q-F0pgx

mkdir -p clevr_subset_tmp
unzip -n clevr_subset.zip -d clevr_subset_tmp
mv clevr_subset_tmp/*/* clevr_subset/
rm -r clevr_subset_tmp
