import torch

from typing import Dict,List
from tqdm import tqdm
def swap_token(tokens, p: float, vocab_size: int, distribution=None):
    if distribution is None:
        distribution = lambda x : torch.ones(size=(len(tokens),vocab_size)) / vocab_size

    idx = torch.randperm(len(tokens))[:int(p*len(tokens))]
    if idx.numel() == 0:
        print("No elements selected for swapping. Check the value of p.")
        return tokens
    
    new_probs = distribution(tokens).to(tokens.device)
    samples = torch.multinomial(new_probs,1).flatten()
    samples = samples.to(tokens.device)
    tokens[idx] = samples[idx]

    return tokens

def delete_token(tokens, p: float):
    idx = torch.randperm(len(tokens))[: int(p*len(tokens))]

    keep = torch.ones(len(tokens),dtype=torch.bool, device = "cuda")
    tokens = tokens.to(keep.device)
    keep[idx] = False
    tokens = tokens[keep]

    return tokens

def insert_token(tokens, p: float, vocab_size: int, distribution = None):

    if distribution is None:
        distribution = lambda x : torch.ones(size=(len(tokens),vocab_size)) / vocab_size

    idx = torch.randperm(len(tokens))[: int(p*len(tokens))]

    new_probs = distribution(tokens).to(tokens.device)

    samples = torch.multinomial(new_probs,1)
    samples = samples.to(tokens.device)
    for i in idx.sort(descending=True).values:
        tokens = torch.cat([tokens[:i], samples[i],tokens[i:]])

        tokens[i] = samples[i]

    return tokens

        

    
def require_attack() -> List[Dict]:
    attack_list = []
    dict_data = {
        "attack_name": "TokenAttack_0.01_0.01_0.01",
        "delete_p": float(0.01),
        "insert_p": float(0.01),
        "swap_p": float(0.01),
    }
    attack_list.append(dict_data)
    
    dict_data = {
        "attack_name": "TokenAttack_0.01_0.0_0.0",
        "delete_p": float(0.01),
        "insert_p": float(0.0),
        "swap_p": float(0.0),
    }
    attack_list.append(dict_data)

    dict_data = {
        "attack_name": "TokenAttack_0.0_0.01_0.0",
        "delete_p": float(0.0),
        "insert_p": float(0.01),
        "swap_p": float(0.0),
    }
    attack_list.append(dict_data)

    dict_data = {
        "attack_name": "TokenAttack_0.0_0.0_0.01",
        "delete_p": float(0.0),
        "insert_p": float(0.0),
        "swap_p": float(0.01),
    }
    attack_list.append(dict_data)

    return attack_list