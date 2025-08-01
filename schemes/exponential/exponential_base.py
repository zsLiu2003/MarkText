import torch


import sys
import numpy as np
import pyximport
pyximport.install(reload_support=True, language_level=sys.version_info[0],
                  setup_args={"include_dirs":np.get_include()})
from exponential.exponential_levenshtein import exponential_levenshtein

import torch

def exponential_score(tokens,xi):
    xi_samp = torch.gather(xi,-1,tokens.unsqueeze(-1)).squeeze()
    return -torch.sum(torch.log(1/(1-xi_samp)))

def exponential_edit_score(tokens,xi,gamma):
    tokens_cpu = tokens.cpu()
    tokens_cpu_long = tokens_cpu.long()
    xi_cpu = xi.cpu()
    return exponential_levenshtein(tokens_cpu_long.numpy(),xi_cpu.numpy(),gamma)


def exponential_key_func(generator,n,vocab_size,eff_vocab_size=None):
    if eff_vocab_size is None:
        eff_vocab_size = vocab_size
        
    pi = torch.arange(eff_vocab_size)
    xi = torch.rand((n,eff_vocab_size), generator=generator)

    return xi,pi

def exponential_sampling(probs,pi,xi):
    return torch.argmax(xi ** (1/torch.gather(probs, 1, pi)),axis=1).unsqueeze(-1)