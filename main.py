import argparse
import os 
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version
from lib.rank import rank_analysis_weight, rank_reduction_weight, rank_reduction_dynamic_pruning
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="Enoch/llama-7b-hf", type=str, help='LLaMA model')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument("--cache_dir", default="./llm_weights", type=str )
    parser.add_argument('--save', type=str, default=None, help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    parser.add_argument('--estimate_rank', type=bool, default=True, help='True for debug.')
    parser.add_argument('--rank_thresold', type=float, default=0.0535, help='Rank thresold to prune the model.')
    parser.add_argument("--data_dir", "-d", type=str, default="data")
    parser.add_argument('--prune_type', type=str, default="error_weight_dynamic_thresold", help='Path to save the pruned model.')
    parser.add_argument("--ntrain", "-k", type=int, default=5)

    args = parser.parse_args()
    

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    model_name = args.model.split("/")[-1]
    print(f"loading llm model {args.model}")
    model = get_llm(args.model, args.cache_dir)

    # file_name = open(f"logs/{model_name}_rank_thresold_{args.rank_thresold}_2.txt", "w")
    # file_name = open(f"logs/{model_name}_full_rank.txt", "w")
    file_name = open(f"logs/{model_name}_error_weight_dynamic_thresold_att_ffn.txt", "w")
    print(f"{args}", file=file_name, flush=True)

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False, use_auth_token="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")

    print("Model and tokenizer loaded", file=file_name, flush=True)

    device = torch.device("cuda:0")
    if "30b" in args.model or "65b" in args.model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        device = model.hf_device_map["lm_head"]
    print("use device ", device)


    print ("model",model)
    layers_singular_value = None
    if args.estimate_rank:
        if os.path.exists("data/singular_values_{}.pt".format(model_name)):
            layers_singular_value = torch.load("data/singular_values_{}.pt".format(model_name))
        else:
            layers_singular_value = rank_analysis_weight(args, model, tokenizer, device)
            torch.save(layers_singular_value, "data/singular_values_{}.pt".format(model_name))

    if layers_singular_value is not None:

        print ("args.prune_type ",args.prune_type)
        if args.prune_type == "uniform":
            print(f"\n Pruning type: {args.prune_type}\n", file=file_name, flush=True)
            rank_pruning = uniform_rank_pruning(args, 0.21151, layers_singular_value, file_name)
            rank_reduction_weight(args, model, tokenizer, rank_pruning, device)
        elif args.prune_type == "weight_thresold":
            print(f"\n Pruning type: {args.prune_type}\n", file=file_name, flush=True)
            rank_pruning = weight_thresold_rank_pruning(args, layers_singular_value, file_name)
            rank_reduction_weight(args, model, tokenizer, rank_pruning, device)
        elif args.prune_type == "error_weight_dynamic_thresold":
            print(f"\n Pruning type: {args.prune_type}\n", file=file_name, flush=True)
            rank_pruning = rank_reduction_dynamic_pruning(args, model, device, file_name)
            rank_reduction_weight(args, model, tokenizer, rank_pruning, device)
    print("Rank Reduction Done", file=file_name, flush=True)
    
    ####################### Details of the MMLU Benchmark ##########################

    
    start_time = timer()
    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(args.data_dir, "test"))
            if "_test.csv" in f
        ]
    )

    all_cors = []
    subcat_cors = {
        subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in categories}

    for subject in subjects:
        dev_df = pd.read_csv(
            os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None
        )[: args.ntrain]
        test_df = pd.read_csv(
            os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None
        )

        cors, acc, probs = eval(args, subject, model, tokenizer, dev_df, test_df, file_name)
        subcats = subcategories[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

        test_df["{}_correct".format(args.model)] = cors
        

    for subcat in subcat_cors:
        subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
        print("Average accuracy {:.3f} - {}".format(subcat_acc, subcat), file=file_name, flush=True)
    print("----------------------------------------", file=file_name)
    for cat in cat_cors:
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        print("Average accuracy {:.3f} - {}".format(cat_acc, cat), file=file_name, flush = True)
    weighted_acc = np.mean(np.concatenate(all_cors))
    print("----------------------------------------", file=file_name)
    print("Average accuracy: {:.3f}".format(weighted_acc), file=file_name)
    print(f"\n\n\n -------- Total time taken:   {timedelta(seconds=timer() - start_time)}-----------", file=file_name)
    file_name.flush()

    file_name.close()



if __name__ == '__main__':
    main()
