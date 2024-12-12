#!/bin/bash

# Set common variables
set -x
# model="meta-llama/Llama-2-7b-hf"
# model="facebook/opt-6.7b" 
# model="meta-llama/Meta-Llama-3-8B"
# sparsity_ratio=0.5
# models=("Enoch/llama-7b-hf" "meta-llama/Llama-2-7b-hf" "meta-llama/Llama-2-13b-hf" "Enoch/llama-13b-hf" "meta-llama/Meta-Llama-3-8B")



# (0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

sparsity_ratios=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)


cuda_device=3

# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=$cuda_device



run_uniform_svd () {
  python   main_singular_value_threshold.py \
    --prune_type uniform \
    --rank_reduction_ratio $1 
}


  

echo "Running sparsity_ratio: $sparsity_ratio"
for sparsity_ratio in "${sparsity_ratios[@]}"
do
  echo "Running sparsity_ratio: $sparsity_ratio"
  run_uniform_svd  ${sparsity_ratio}
done
 

set +x


