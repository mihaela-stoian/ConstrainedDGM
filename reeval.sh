#!/bin/bash

export PYTHONPATH=$PYTHONPATH:gather_results

use_case="url"
model="octgan"
versions="unconstrained random corr kde"

for version in $versions ;
do
  CUDA_VISIBLE_DEVICES=-1 python gather_results/reeval_final.py ${use_case} ${model} ${version}
done
