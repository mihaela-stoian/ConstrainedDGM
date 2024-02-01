#!/bin/bash

export PYTHONPATH=$PYTHONPATH:gather_results

use_case="url"
model="tablegan"
version="unconstrained"
orderings="corr kde random"

for order in $orderings ;
do
  CUDA_VISIBLE_DEVICES=-1 python gather_results/reeval_final.py ${use_case} ${model} ${version} --postprocessing --postprocessing_label_ordering=$order
done
