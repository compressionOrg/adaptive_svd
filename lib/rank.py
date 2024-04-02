import time 
import heapq 
import torch 
import torch.nn as nn 
from .data_utils import get_c4, get_wikitext2
from tqdm import tqdm
import numpy as np
import pdb
def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if "c4" in name:
        return get_c4(nsamples, seed, seqlen, tokenizer)
    
def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def prepare_calibration_input(model, dataloader, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # dev = model.hf_device_map["model.embed_tokens"]
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype

    print ("devicedevicedevicedevicedevicedevicedevicedevicedevicedevicedevicedevice",device)

    # pdb.set_trace()

    inps = torch.zeros((128, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)

    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass 
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids 

def rank_analysis_weight(args, model, tokenizer, device):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

    layers = model.model.layers
    
    layers_singular_value = {}
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        layers_singular_value[i] = {}
        # Perform Singular Value Decomposition (SVD)
        for name in subset:
            W = subset[name].weight.data 
            _, singular_values, _ = torch.svd(W.to(torch.float32))
            layers_singular_value[i][name] = singular_values

    return layers_singular_value

def do_low_rank(weight, desired_rank, debug=False, niter=2):
    assert weight.ndim == 2
    loss = torch.nn.L1Loss()
    max_rank = min(weight.shape[0], weight.shape[1])

    if debug:
        print(f"Shape is {weight.shape} and shape is {weight.dtype} => desired rank {desired_rank}")



    ####### previows
    # results = torch.svd_lowrank(weight,
    #                             q=desired_rank,
    #                             niter=niter)
    # weight_approx = results[0] @ torch.diag(results[1]) @ results[2].T

    ### accurate
    results= torch.svd(weight)
    U_r = results[0][:, :desired_rank]  
    S_r = results[1][:desired_rank]    
    V_r = results[2][:, :desired_rank]  
    weight_approx = torch.mm(U_r, torch.diag(S_r)).mm(V_r.t())


    if debug:
        print(f"New matrix has shape {weight_approx.shape}")

    assert weight_approx.shape[0] == weight.shape[0] and weight_approx.shape[1] == weight.shape[1]
    weight_approx = torch.nn.Parameter(weight_approx)

    with torch.no_grad():
        error = loss(weight, weight_approx)
    return weight_approx, error

def rank_reduction_weight(args, model, tokenizer, rank_pruning, device):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    layers = model.model.layers
    layers_singular_value = {}

    for i in tqdm(range(len(layers))):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data
            k = min(W.shape[0], W.shape[1]) - rank_pruning[i][name]
            approx_w, error = do_low_rank(W.to(torch.float32), k, True)
            print(f"layer.{i}.{name} ({k}): {error}")

            subset[name].weight.data = approx_w.data.to(torch.float16)

    print("Pruning completed")

def rank_reduction_dynamic_pruning(args, model, device, file_name):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    layers = model.model.layers

    rank_pruning = {}
    total_rank, error_thresold_att, error_thresold_ffn = 0, 5e-4, 5e-4
    pruning_bucket = [0.95, 0.9, 0.85, 0.8, 0.7, 0.75, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.2, 0.1]

    for i in tqdm(range(len(layers))):
        layer = layers[i]
        subset = find_layers(layer)
        rank_pruning[i] = {}
        for name in subset:
            W = subset[name].weight.clone().data
            if "mlp" in name: error_thresold = error_thresold_ffn
            else: error_thresold = error_thresold_att
            rank_pruning[i][name] = 0
            for prune_ratio in pruning_bucket:
                desired_rank = int(min(W.shape[0], W.shape[1]) * prune_ratio)
                approx_w, error = do_low_rank(W.to(torch.float32), desired_rank, False)
                if error > error_thresold:
                    break
                else:
                    rank_pruning[i][name] = min(W.shape[0], W.shape[1]) - desired_rank
            total_rank += int(min(W.shape[0], W.shape[1]))
            print(f"layer.{i}.{name} ({rank_pruning[i][name]}): {error}")
    
    pruned_rank = 0
    for i in tqdm(range(len(layers))):
        layer = layers[i]
        subset = find_layers(layer)
        for name in subset:
            pruned_rank += rank_pruning[i][name]
    print("Pruning completed")
    torch.save(rank_pruning, "./data/adative_rank_attention_ffn.pt")
    print(f"Rank Reduction: {(pruned_rank/total_rank)* 100:.3f} %", file=file_name, flush=True)
    return rank_pruning