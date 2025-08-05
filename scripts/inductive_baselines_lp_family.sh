#!/bin/bash

DATA=$1
METH=$2
SHOT=$3

datasets=("imagenet" "sun397" "fgvc" "eurosat" "stanford_cars" "food101" "oxford_pets" "oxford_flowers" "caltech101" "dtd" "ucf101")

for dataset in "${datasets[@]}"; do
  if [ "$dataset" = "imagenet" ]; then
    RUNNER="baseline_vl.lp_family_runner_imagenet"
  else
    RUNNER="baseline_vl.lp_family_runner"
  fi

  echo "Running dataset: $dataset with runner: $RUNNER"

  python3 -m "$RUNNER" \
            --base_config baseline_vl/configs/base.yaml \
            --dataset_config baseline_vl/configs/${dataset}.yaml \
            --opt root_path "${DATA}" \
                  method "${METH}" \
                  shots "${SHOT}"
done