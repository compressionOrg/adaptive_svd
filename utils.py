import pickle
import torch
import numpy as np
from pdb import set_trace as st 


choices = ["A", "B", "C", "D"]


def save_dict(item, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(item, handle, protocol=pickle.HIGHEST_PROTOCOL)

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


@torch.no_grad()
def eval(args, subject, model, tokenizer, dev_df, test_df, f):
    cors = []
    all_probs = []
    answers = choices[: test_df.shape[1] - 2]

    for i in range(test_df.shape[0]):
        # get prompt and make sure it fits
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end
        # print(prompt)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

        while input_ids.shape[-1] > 2048:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

        label = test_df.iloc[i, test_df.shape[1] - 1]

        generate_ids = model.generate(input_ids, max_length=len(input_ids[0]) + 1)
        output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        pred = output[-1:]
        print(label, pred)

        cor = pred == label
        cors.append(cor)

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject), file=f)
    f.flush()

    return cors, acc, all_probs


def uniform_rank_pruning(args, pruning_ratio, layers_singular_value, file_name):
    total_rank, pruned_rank = 0, 0
    rank_pruning = {}
    for index in range(0, len(layers_singular_value)):
        layer = layers_singular_value[index]
        subset = list(layer.keys())
        rank_pruning[index] = {}
        for name in subset:
            _data = layer[name].clone().cpu().numpy()
            rank_pruning[index][name] = int(pruning_ratio * len(_data))
            total_rank += len(_data)
            pruned_rank += rank_pruning[index][name]
            print(f"layer{index}.{name} rank reduction: \t\t{(rank_pruning[index][name]/len(_data))* 100:.3f} %", file=file_name, flush=True)
    print(f"Rank Reduction: {(pruned_rank/total_rank)* 100:.3f} %", file=file_name, flush=True)
    return rank_pruning

def weight_thresold_rank_pruning(args, layers_singular_value, file_name):
    """
    Given a rank thresold, normalize the singular values and prune each layer under the rank_thresold
    """

    print ("weight_thresold_rank_pruning,weight_thresold_rank_pruning,weight_thresold_rank_pruning")
    print (layers_singular_value)
    total_rank, pruned_rank = 0, 0
    rank_pruning = {}
    for index in range(0, len(layers_singular_value)):
        layer = layers_singular_value[index]
        subset = list(layer.keys())
        rank_pruning[index] = {}


        for name in subset:

            data = layer[name].clone().cpu().numpy()


            _data = (data-min(data))/(max(data)-min(data))


            rank_pruning[index][name] = len(_data) - sum(_data > args.rank_thresold) # Rank which will be pruned
            total_rank += len(_data)
            pruned_rank += rank_pruning[index][name]
            
            print(f"layer{index}.{name} rank reduction: \t\t{(rank_pruning[index][name]/len(_data))* 100:.3f} %", flush=True)
            print(f"layer{index}.{name} rank reduction: \t\t{(rank_pruning[index][name]/len(_data))* 100:.3f} %", file=file_name, flush=True)
    
    print(f"\n\n Total Rank Reduction: {(pruned_rank/total_rank)* 100:.3f} %", flush=True)
    print(f"\n\n Total Rank Reduction: {(pruned_rank/total_rank)* 100:.3f} %", file=file_name, flush=True)
    return rank_pruning


def weight_thresold_scale(args, layers_singular_value, file_name):
    """
    Given a rank thresold, normalize the singular values and prune each layer under the rank_thresold
    """

    print ("weight_thresold_rank_pruning,weight_thresold_rank_pruning,weight_thresold_rank_pruning")
    print (layers_singular_value)
    total_rank, pruned_rank = 0, 0
    rank_pruning = {}

    all_layer_ratio=[]
    for index in range(0, len(layers_singular_value)):
        layer = layers_singular_value[index]
        subset = list(layer.keys())
        rank_pruning[index] = {}


        for name in subset:

            data = layer[name].clone().cpu().numpy()


            _data = (data-min(data))/(max(data)-min(data))

            _data = sum(_data > args.rank_thresold) 

            all_layer_ratio.append( _data)

    print ("before maping",all_layer_ratio)

    all_layer_ratio=np.array(all_layer_ratio)
    
    all_layer_ratio = ((all_layer_ratio - all_layer_ratio.min()) * (1/(all_layer_ratio.max() - all_layer_ratio.min()) * args.Lamda*2))
    
    all_layer_ratio=all_layer_ratio-np.mean(all_layer_ratio)+(1-args.rank_reduction_ratio)
    all_layer_ratio=1-all_layer_ratio


    print ("after maping",all_layer_ratio)
    


    # print (pruned_rank)
    i=0
    for index in range(0, len(layers_singular_value)):
            layer = layers_singular_value[index]
            subset = list(layer.keys())
            rank_pruning[index] = {}


            for name in subset:

                data = layer[name].clone().cpu().numpy()

                rank_pruning[index][name] = int (len(data)* all_layer_ratio[i]) # Rank which will be pruned
                
                i+=1

                total_rank += len(data)
                pruned_rank += rank_pruning[index][name]
                
                print(f"layer{index}.{name} rank reduction: \t\t{(rank_pruning[index][name]/len(data))* 100:.3f} %", flush=True)
                print(f"layer{index}.{name} rank reduction: \t\t{(rank_pruning[index][name]/len(data))* 100:.3f} %", file=file_name, flush=True)
        



    print (all_layer_ratio,np.mean(all_layer_ratio),np.max(all_layer_ratio),np.min(all_layer_ratio))




    print(f"\n\n Total Rank Reduction: {(pruned_rank/total_rank)* 100:.3f} %", flush=True)
    print(f"\n\n Total Rank Reduction: {(pruned_rank/total_rank)* 100:.3f} %", file=file_name, flush=True)
    return rank_pruning





def check_outlier_mean(mask,threshold):


    W = mask
    count = 0 
    total_params = 0
    
    max_shred=torch.mean(W)*threshold
    count += (W>max_shred).sum().item()
    total_params += W.numel()



    outlier_ratio=float(count)/total_params*100
    
    return outlier_ratio


def weight_thresold_scale_bymean(args, layers_singular_value, file_name):
    """
    Given a rank thresold, normalize the singular values and prune each layer under the rank_thresold
    """

    print ("weight_thresold_rank_pruning,weight_thresold_rank_pruning,weight_thresold_rank_pruning")
    print (layers_singular_value)
    total_rank, pruned_rank = 0, 0
    rank_pruning = {}

    all_layer_ratio=[]
    for index in range(0, len(layers_singular_value)):
        layer = layers_singular_value[index]
        subset = list(layer.keys())
        rank_pruning[index] = {}


        for name in subset:

            # data = layer[name].clone().cpu().numpy()
            data = layer[name].clone().cpu()


            _data = check_outlier_mean ( data,args.rank_thresold)

            all_layer_ratio.append( _data)


    print ("before maping",all_layer_ratio)
    all_layer_ratio=np.array(all_layer_ratio)
    all_layer_ratio = ((all_layer_ratio - all_layer_ratio.min()) * (1/(all_layer_ratio.max() - all_layer_ratio.min()) * args.Lamda*2))
    all_layer_ratio=all_layer_ratio-np.mean(all_layer_ratio)+(1-args.rank_reduction_ratio)
    all_layer_ratio=1-all_layer_ratio


    print ("after maping",all_layer_ratio)
    


    # print (pruned_rank)
    i=0
    for index in range(0, len(layers_singular_value)):
            layer = layers_singular_value[index]
            subset = list(layer.keys())
            rank_pruning[index] = {}


            for name in subset:

                data = layer[name].clone().cpu().numpy()

                rank_pruning[index][name] = int (len(data)* all_layer_ratio[i]) # Rank which will be pruned
                
                i+=1

                total_rank += len(data)
                pruned_rank += rank_pruning[index][name]
                
                print(f"layer{index}.{name} rank reduction: \t\t{(rank_pruning[index][name]/len(data))* 100:.3f} %", flush=True)
                print(f"layer{index}.{name} rank reduction: \t\t{(rank_pruning[index][name]/len(data))* 100:.3f} %", file=file_name, flush=True)
        



    print (all_layer_ratio,np.mean(all_layer_ratio),np.max(all_layer_ratio),np.min(all_layer_ratio))




    print(f"\n\n Total Rank Reduction: {(pruned_rank/total_rank)* 100:.3f} %", flush=True)
    print(f"\n\n Total Rank Reduction: {(pruned_rank/total_rank)* 100:.3f} %", file=file_name, flush=True)
    return rank_pruning



def weight_thresold_scale_dlp(args, layers_singular_value, file_name):
    """
    Given a rank thresold, normalize the singular values and prune each layer under the rank_thresold
    """

    print ("weight_thresold_rank_pruning,weight_thresold_rank_pruning,weight_thresold_rank_pruning")
    print (layers_singular_value)
    total_rank, pruned_rank = 0, 0
    rank_pruning = {}


    layer_score = {}
    layer_wmetric=[]
    
    for index in range(0, len(layers_singular_value)):
        layer = layers_singular_value[index]
        subset = list(layer.keys())
        rank_pruning[index] = {}

        
        for name in subset:

            # data = layer[name].clone().cpu().numpy()
            data = layer[name].clone().cpu()
            layer_wmetric.append(data)   

    medians = [torch.median(tensor).item() for tensor in layer_wmetric]
    
    ratio_conn = [1 - layer / sum(medians) for layer in medians]

    all_layer_ratio=np.array(ratio_conn)
    all_layer_ratio = ((all_layer_ratio - all_layer_ratio.min()) * (1/(all_layer_ratio.max() - all_layer_ratio.min()) * args.Lamda*2))
    all_layer_ratio=all_layer_ratio-np.mean(all_layer_ratio)+(1-args.rank_reduction_ratio)
    all_layer_ratio=1-all_layer_ratio


    print ("after maping",all_layer_ratio)
    


    # print (pruned_rank)
    i=0
    for index in range(0, len(layers_singular_value)):
            layer = layers_singular_value[index]
            subset = list(layer.keys())
            rank_pruning[index] = {}


            for name in subset:

                data = layer[name].clone().cpu().numpy()
                # st()
                
                rank_pruning[index][name] = int (len(data)* all_layer_ratio[i]) # Rank which will be pruned
                
                i+=1

                total_rank += len(data)
                pruned_rank += rank_pruning[index][name]
                
                print(f"layer{index}.{name} rank reduction: \t\t{(rank_pruning[index][name]/len(data))* 100:.3f} %", flush=True)
                print(f"layer{index}.{name} rank reduction: \t\t{(rank_pruning[index][name]/len(data))* 100:.3f} %", file=file_name, flush=True)
        



    print (all_layer_ratio,np.mean(all_layer_ratio),np.max(all_layer_ratio),np.min(all_layer_ratio))




    print(f"\n\n Total Rank Reduction: {(pruned_rank/total_rank)* 100:.3f} %", flush=True)
    print(f"\n\n Total Rank Reduction: {(pruned_rank/total_rank)* 100:.3f} %", file=file_name, flush=True)
    return rank_pruning

