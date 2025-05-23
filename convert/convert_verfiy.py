from convert.convert_base import Convert
from dataclasses import dataclass,field, replace
import math
from typing import Dict, List
import ast
import dacite
import torch
import scipy

@dataclass(frozen=False)
class VerifierOutput:
    """Output from a watermark's verifier"""

    sequence_token_count: int = 0
    pvalues: Dict[int, float] = field(default_factory=dict, hash=False)

    def update(self, token_index, pvalue):
        if token_index > self.sequence_token_count:
            self.sequence_token_count = token_index
        self.pvalues[token_index] = pvalue

    def get_pvalue(self):
        return (
            0.5
            if not len(self.pvalues)
            else self.pvalues[self.sequence_token_count]
        )

    def get_size(self, pvalue):
        """
        Returns the size of a watermark detection algorithm given a list of watermark scores and a p-value threshold.

        Returns:
            float: The efficiency of the watermark detection algorithm, defined as the detection time of the last detected watermark.
        """
        if not len(self.pvalues):
            return math.inf

        if self.pvalues[self.sequence_token_count] > pvalue:
            return math.inf

        idx = self.sequence_token_count - 1
        while idx > 0 and idx in self.pvalues and self.pvalues[idx] <= pvalue:
            idx -= 1

        return idx


class ConvertVerify:
    

    def __init__(self, pvalue, tokenizer, convert: Convert, skip_prob):
        self.pvalue = pvalue
        self.tokenizer = tokenizer
        self.convert = convert
        self.skip_prob = skip_prob
        self.device = "cuda"

    def _verify(self, tokens, index = 0):
        return_value = VerifierOutput()
        convert_tokens = self.convert.to_bit(tokens=tokens).squeeze()
        mask = convert_tokens >= 0
        max_bitlen = mask.sum(axis = 1).max()
        convert_tokens = convert_tokens[:,:max_bitlen]
        mask = mask[:, : max_bitlen]
        ctn = mask.sum(axis=1)
        xi = []
        for i in range(tokens.shape[-1]):
            prev_values = tokens[:i]
            bitlen = ctn[i].item()
            seed = self.rng.get_seed(prev_values, [index])
            xi.append(
                [self.rng.rand_index(seed, i).item() for i in range(bitlen)]
                + [-1 for _ in range(max_bitlen - bitlen)]
            )

        xi = torch.Tensor(xi).to(self.rng.device)

        v = (
            -(xi * convert_tokens + (1 - xi) * (1 - convert_tokens)).abs().log()
            * mask
        )
        cumul = v.sum(axis=-1).cumsum(0).tolist()
        ctn = mask.sum(axis=1).cumsum(0).tolist()

        # Compute average
        for i, v in enumerate(cumul):
            c = ctn[i]
            likelihood = scipy.stats.gamma.sf(v, c)
            return_value.update(i, likelihood)

        return return_value
    
    def verify(self, tokens, skip_edit = False, **kwargs):
        rtn = {}
        tokens = tokens.to("cuda")
        tokens = tokens.reshape(-1)
        return self._verify(tokens=tokens, )

    def verify_text(self, text: str, skip_edit = False, **kwargs):
        
        tokens = self.tokenizer.encode(
            text,
            add_special_tokens = False,
            return_tensors = "pt",
            truncation = True,
            max_length= 1536,
        ).to(self.device)
        
        return self.verify(tokens=tokens,ski)
        
