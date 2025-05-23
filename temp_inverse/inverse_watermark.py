# Modified from https://github.com/jthickstun/watermark/blob

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import json
from tqdm import tqdm
from typing import Dict,List

from kgw import kgw_watermark2
from util.classes import Generation,ConfigSpec,WatermarkConfig
import time
import numpy as np
import pyximport

from temp_inverse.mersenne import mersenne_rng
pyximport.install(reload_support=True, language_level=sys.version_info[0],
                  setup_args={"include_dirs":np.get_include()})
from inverse.inverse_levenshtein import levenshtein
from generate_text import load_file
from token_attack.token_attack import swap_token, insert_token, delete_token, require_attack


class Inverse():
    def __init__(self, config:ConfigSpec):
        self.model_name = config.model_name
        self.output_path = config.output_path
        self.watermark_name = "Inverse"
    def sampling(self,probs,u):
        return torch.argmax(u ** (1/probs),axis=1).unsqueeze(-1)

    def attack(self, tokens, delete_p: float, insert_p: float, swap_p: float, vocab_size: int):
        tokens = delete_token(tokens=tokens, p=delete_p)
        tokens = insert_token(tokens=tokens, p=insert_p, vocab_size=vocab_size)
        tokens = swap_token(tokens=tokens, p=swap_p, vocab_size=vocab_size)

        return tokens
    
    def watermark_token(self, model, prompt, vocab_size: int, max_length: int, watermark_length: int, key: int):
        rng = mersenne_rng(key)
        xi = torch.tensor([rng.rand() for _ in range(max_length*vocab_size)]).view(max_length,vocab_size)
        shift = torch.randint(max_length,(1,))
        inputs = prompt.to(model.device)
        attn = torch.ones_like(inputs)
        past = None
        for i in range(watermark_length):
            with torch.no_grad():
                if past:
                    output = model(inputs[:,-1:], past_key_values=past, attention_mask=attn)
                else:
                    output = model(inputs)
            
            probs = torch.nn.functional.softmax(output.logits[:,-1, : vocab_size], dim=-1).cpu()
            token = self.sampling(probs=probs,u=xi[(shift + i)%max_length, :]).to(model.device)
            inputs = torch.cat([inputs, token], dim=-1)
            
            past = output.past_key_values
            attn = torch.cat([attn,attn.new_ones((attn.shape[0], 1))], dim=-1)
        return inputs.detach().cpu()
    
    def injection(self,prompts:List[str], model_name:str, key: int = 56, max_length: int = 1536, seed: int = 0, watermark_length: int = 80):
        torch.manual_seed(seed=seed)
        device = "cuda"
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map = 'auto')
        tokenizer.pad_token = tokenizer.eos_token
        output_generations = []
        encode_watermark_tokens = []
        vocab_size =len(tokenizer.get_vocab().values())
        for idx, prompt in tqdm(enumerate(prompts), desc="Processing inverse watermark injection---"):
            tokens = tokenizer.encode(
                prompt,
                return_tensors='pt',
                max_length=512,
                truncation=True
            )
            watermark_tokens = self.watermark_token(
                model=model, 
                prompt = tokens, 
                vocab_size =vocab_size,
                max_length=max_length,
                watermark_length=watermark_length,
                key = key
            )
            print(watermark_tokens)
            watermark_text = tokenizer.decode(watermark_tokens, skip_special_tokens=True)
            generation = Generation(id=idx, prompt=prompt, response=watermark_text,watermark_name="Inverse")
            output_generations.append(generation)
        output_path = f"{self.output_path}/watermark/Inverse/generations_Inverse.json"        
        Generation.tofile(output_path,output_generations)
        watermark_result = []
        attack_list = require_attack()
        for attack in tqdm(attack_list, desc="Processing token attack-------"):
            delete_p = attack["delete_p"]
            insert_p = attack["insert_p"]
            swap_p = attack["swap_p"]
            attack_name = attack["attack_name"]
            is_watermark = []
            for idx, watermark_tokens in tqdm(enumerate(encode_watermark_tokens), desc=f"Processing {attack_name}-----"):
                watermark_tokens = self.attack(
                    tokens=watermark_tokens,
                    delete_p=delete_p,
                    insert_p=insert_p,
                    swap_p=swap_p,
                    vocab_size=vocab_size
                )
                _is_watermark = self.permutation_test(watermark_tokens, max_length=max_length, watermarked_token_length=len(watermark_tokens),vocab_size=vocab_size)
                is_watermark.append(_is_watermark)
            dict_data = {
                "attack_name": attack_name,
                "is_watermark": is_watermark
            }
            watermark_result.append(dict_data)
        output_path = f"{self.output_path}/watermark/{self.watermark_name}/quality/is_watermark_list.json"
        with open(output_path, 'w') as outputfile:
            json.dumps(watermark_result,outputfile, indent=4)

    def detect(self,input_ids, max_length: int, watermarked_token_length: int, xi, gamma: float = 0.0):
        m = len(input_ids)
        n = len(xi)
        A = np.empty((m-(watermarked_token_length-1), n))

        for i in range(m - (watermarked_token_length-1)):
            for j in range(n):
                A[i][j] = levenshtein(input_ids[i: i+watermarked_token_length], xi[(j + np.arange(watermarked_token_length))%n], gamma=gamma)
        
        return np.min(A)


    def permutation_test(self,input_ids, key: int, max_length: int, watermarked_token_length: int, vocab_size: int, n_runs: int = 100):
        rng = mersenne_rng(key)
        xi = np.array([rng.rand() for _ in range(max_length*vocab_size)], dtype=np.float32).reshape(max_length, vocab_size)
        test_result = self.detect(input_ids=input_ids,max_length=max_length, watermarked_token_length=watermarked_token_length,xi=xi)

        p_val = 0
        for run in range(n_runs):
            xi_alternative = np.random.rand(max_length, vocab_size).astype(np.float32)
            null_result = self.detect(input_ids=input_ids, max_length=max_length, watermarked_token_length=watermarked_token_length,xi=xi_alternative)

            p_val += null_result <= test_result
        
        return (p_val + 1.0) / (n_runs + 1.0)


    def detection(self, text: str, key: int, max_length: int, threshold: float = 0.01) -> bool:
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenizer.pad_token = tokenizer.eos_token
        input_ids = tokenizer.encode(
            text=text,
            return_tensors="pt",
            truncation=True,
            max_length=1536
        ).numpy()[0]
        start_time = time.time()
        pval = self.permutation_test(input_ids=input_ids, key=key, max_length=max_length, watermarked_token_length=len(input_ids), vocab_size=len(tokenizer.get_vocab().values()))
        if pval <= threshold:
            return True
        else:
            return False
        
