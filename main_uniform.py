import argparse
import os 
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version
from lib.rank import rank_analysis_weight, rank_reduction_weight, rank_reduction_dynamic_pruning
from lib.eval import eval_ppl
from utils import *
from timeit import default_timer as timer
from datetime import timedelta
from categories import subcategories, categories
import pandas as pd

def get_llm(model_name, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        cache_dir=cache_dir, 
        low_cpu_mem_usage=True, 
        device_map="auto",
        use_auth_token="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"
    )

    model.seqlen = model.config.max_position_embeddings 
    return model

"""
-> rank_thresold
Overall: 0.0535 - Singular value
"""
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="meta-llama/Llama-2-7b-chat-hf", type=str, help='LLaMA model')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument("--cache_dir", default="/data/ajay_data/MCI/llm_weights", type=str )
    parser.add_argument('--save', type=str, default=None, help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    parser.add_argument('--estimate_rank', type=bool, default=True, help='Check if the layerwise singular values need to be calculated.')
    parser.add_argument('--rank_thresold', type=float, default=0.0535, help='Rank thresold to prune the model.')
    parser.add_argument("--data_dir", "-d", type=str, default="data")
    parser.add_argument("--ntrain", "-k", type=int, default=5)

    args = parser.parse_args()
    

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    model_name = args.model.split("/")[-1]
    print(f"loading llm model {args.model}")
    model = get_llm(args.model, args.cache_dir)

    file_name = open(f"logs/{model_name}_singular_value_uniform.txt", "w")
    print(f"{args}", file=file_name, flush=True)

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False, use_auth_token="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")

    print("Model and tokenizer loaded", file=file_name, flush=True)

    device = torch.device("cuda:0")
    if "30b" in args.model or "65b" in args.model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        device = model.hf_device_map["lm_head"]
    print("use device ", device)

    layers_singular_value = None
    if args.estimate_rank:
        if os.path.exists("data/singular_values_llama-2-7b.pt"):
            layers_singular_value = torch.load("data/singular_values_llama-2-7b.pt")
        else:
            layers_singular_value = rank_analysis_weight(args, model, tokenizer, device)
            torch.save(layers_singular_value, "data/singular_values_llama-2-7b.pt")

    

    ppl = eval_ppl(model, tokenizer, device)
    print(f"{args}\nBefore Rank Reduction PPL on C4: {ppl}\n", file=file_name, flush=True)
    


    rank_pruning = uniform_rank_pruning(args, 0.21151, layers_singular_value, file_name)
    rank_reduction_weight(args, model, tokenizer, rank_pruning, device)

    ppl = eval_ppl(model, tokenizer, device)
    print(f"{args}\nAfter Rank Reduction PPL on C4: {ppl}\n", file=file_name, flush=True)
    file_name.flush()


if __name__ == '__main__':
    main()