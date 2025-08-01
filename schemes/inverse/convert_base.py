import torch

import sys
import numpy as np
import pyximport
pyximport.install(reload_support=True, language_level=sys.version_info[0],
                  setup_args={"include_dirs":np.get_include()})

from inverse.convert_levenshtein import convert_levenshtein

def convert_score(tokens,xi):
    return torch.pow(torch.linalg.norm(tokens-xi.squeeze(),ord=1),1)

def convert_edit_score(tokens,xi,gamma=1):
    tokens_cpu = tokens.cpu()
    xi_cpu = xi.squeeze().cpu()
    return convert_levenshtein(tokens_cpu.numpy(),xi_cpu.numpy(),gamma)

def convert_key_func(generator,n,vocab_size,eff_vocab_size=None):
    pi = torch.randperm(vocab_size, generator=generator)
    xi = torch.rand((n,1), generator=generator)

    return xi,pi


def convert_sampling(probs,pi,xi):
    cdf = torch.cumsum(torch.gather(probs, 1, pi), 1)
    return torch.gather(pi, 1, torch.searchsorted(cdf, xi))