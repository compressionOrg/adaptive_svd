import argparse
import os 
import json
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
from pdb import set_trace as st 


####
from act_lib.model_wrapper_low import make_low_rank


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
    parser.add_argument('--model', default="Enoch/llama-7b-hf", type=str, help='LLaMA model')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument("--cache_dir", default="./llm_weights", type=str )
    parser.add_argument('--save', type=str, default=None, help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    parser.add_argument('--estimate_rank', type=bool, default=True, help='Check if the layerwise singular values need to be calculated.')
    parser.add_argument("--data_dir", "-d", type=str, default="data")

    parser.add_argument('--prune_type', type=str, default="weight_thresold_scale_dlp2", help='Path to save the pruned model.')
    parser.add_argument("--ntrain", "-k", type=int, default=5)



    ##### for zero shot 

    parser.add_argument(
        "--zero_shot_bench", action="store_true", help="performe zero_shot_bench."
    )


    ##### for svd
    parser.add_argument(
        "--prune_data",
        type=str,
        choices=[
            "wikitext",
            "alpaca",
            "alpaca_cleaned",
            "alpaca_cleaned_no_safety",
            "align",
            "align_short",
            "misalign",
            "align_misalign",
            "misalign_align",
            "align_short_misalign",
            "c4",
            "none",
        ],
        default="c4",
    )
    parser.add_argument(
        "--entangle_prompt_feat",
        dest="disentangle",
        action="store_false",
        help="entangle the prompt and response when computing the wanda score",
    )

    parser.add_argument('--rank_reduction_ratio', type=float, default=0.1, help='rank reduction ratio')
    parser.add_argument('--rank_thresold', type=float, default=0.13625, help='Rank thresold to prune the model.')
    parser.add_argument('--Lamda', type=float, default=0.2, help='Rank thresold to prune the model.')
    
    parser.add_argument(
        "--top_remove", action="store_true", help="Remove the top ranks."
    )



    parser.add_argument(
        "--dump_U", action="store_true", help="dump the U matrix for analysis"
    )
    args = parser.parse_args()
    

    

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    model_name = args.model.split("/")[-1]
    print(f"loading llm model {args.model}")
    
    # Offline load moodel
    args.model = args.cache_dir + "/models--" + args.model.replace("/", "--") + "/model"

    
    model = get_llm(args.model, args.cache_dir)

    file_name = open(f"logs/{model_name}_{args.prune_type}_rank_reduction_ratio_{args.rank_reduction_ratio}_rank_thresold_{args.rank_thresold}_Lamda_{args.Lamda}.txt", "w")
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
        if os.path.exists("data/singular_values_{}.pt".format(model_name)):
            layers_singular_value = torch.load("data/singular_values_{}.pt".format(model_name))
        else:
            layers_singular_value = rank_analysis_weight(args, model, tokenizer, device)
            torch.save(layers_singular_value, "data/singular_values_{}.pt".format(model_name))




    eval_before=0
    if eval_before:
        ppl = eval_ppl(model, tokenizer, device, "wikitext2")
        print(f"{args}\nBefore Rank Reduction PPL on wikitext2: {ppl}\n", file=file_name, flush=True)
        

    ######### rank reduction

    if layers_singular_value is not None:


        ####### weights 

        print ("args.prune_type ",args.prune_type)
        if args.prune_type == "uniform":
            print(f"\n Pruning type: {args.prune_type}\n", file=file_name, flush=True)
            rank_pruning = uniform_rank_pruning(args, args.rank_reduction_ratio, layers_singular_value, file_name)
            rank_reduction_weight(args, model, tokenizer, rank_pruning, device)

        elif args.prune_type == "weight_thresold":
            print(f"\n Pruning type: {args.prune_type}\n", file=file_name, flush=True)
            rank_pruning = weight_thresold_rank_pruning(args, layers_singular_value, file_name)
            rank_reduction_weight(args, model, tokenizer, rank_pruning, device)


        elif args.prune_type == "weight_thresold_scale":
            print(f"\n Pruning type: {args.prune_type}\n", file=file_name, flush=True)
            rank_pruning = weight_thresold_scale(args, layers_singular_value, file_name)
            rank_reduction_weight(args, model, tokenizer, rank_pruning, device)

        elif args.prune_type == "weight_thresold_scale_bymean":
            print(f"\n Pruning type: {args.prune_type}\n", file=file_name, flush=True)
            rank_pruning = weight_thresold_scale_bymean(args, layers_singular_value, file_name)
            rank_reduction_weight(args, model, tokenizer, rank_pruning, device)

        elif args.prune_type == "weight_thresold_scale_dlp":
            print(f"\n Pruning type: {args.prune_type}\n", file=file_name, flush=True)
            rank_pruning = weight_thresold_scale_dlp(args, layers_singular_value, file_name)
            rank_reduction_weight(args, model, tokenizer, rank_pruning, device)
            
        elif args.prune_type == "weight_thresold_scale_dlp2":
            print(f"\n Pruning type: {args.prune_type}\n", file=file_name, flush=True)
            ratio_file = "ratios" + "/" + model_name + "_sparsity_" + str(args.rank_reduction_ratio)  + ".json"
            with open(ratio_file,  'r', encoding='utf-8') as json_file:
                imp_ratio = json.load(json_file)
            rank_pruning = weight_thresold_scale_dlp2(args, imp_ratio, layers_singular_value, file_name)
            rank_reduction_weight(args, model, tokenizer, rank_pruning, device)        

        ######## act
        elif args.prune_type == "weight_thresold_act":
            print(f"\n Pruning type: {args.prune_type}\n", file=file_name, flush=True)
            rank_pruning = weight_thresold_rank_pruning(args, layers_singular_value, file_name)
            make_low_rank(args,rank_pruning, model, tokenizer, device, prune_data=args.prune_data)

        elif args.prune_type == "uniform_act":
            print(f"\n Pruning type: {args.prune_type}\n", file=file_name, flush=True)
            rank_pruning = uniform_rank_pruning(args, args.rank_reduction_ratio, layers_singular_value, file_name)
            make_low_rank(args,rank_pruning, model, tokenizer, device, prune_data=args.prune_data)




    print("Rank Reduction Done", file=file_name, flush=True)


    # rank_pruning = weight_thresold_rank_pruning(args, layers_singular_value, file_name)
    # rank_reduction_weight(args, model, tokenizer, rank_pruning, device)

    ppl = eval_ppl(model, tokenizer, device, "wikitext2")
    print(f"{args}\nAfter Rank Reduction PPL on wikitext2: {ppl}\n", file=file_name, flush=True)
    print(f"{args}\nAfter Rank Reduction PPL on wikitext2: {ppl}\n", flush=True)
    file_name.flush()

    ########## zero-shot


    if args.zero_shot_bench:

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



if __name__ == '__main__':
    main()