#!/bin/bash

# conda activate prune

# Set common variables
set -x
# model="meta-llama/Llama-2-7b-hf"
# model="facebook/opt-6.7b" 
# model="meta-llama/Meta-Llama-3-8B"
# sparsity_ratio=0.5
# models=("Enoch/llama-7b-hf" "meta-llama/Llama-2-7b-hf" "meta-llama/Llama-2-13b-hf" "Enoch/llama-13b-hf" "meta-llama/Meta-Llama-3-8B")


Lamdas=(0.02 0.05 0.08 0.1 0.12 0.15 0.18 0.2)


# (0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

sparsity_ratios=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8)


cuda_device=3



# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=$cuda_device

run_svd () {
  python   main_singular_value_threshold.py \
    --rank_reduction_ratio $1 
}


for sparsity_ratio in "${sparsity_ratios[@]}"
do
  echo "Running sparsity_ratio: $sparsity_ratio" 
  run_svd  ${sparsity_ratio} 

done
 

set +x

