import torch
import torch.nn.functional as F

from transformers import AutoTokenizer,AutoModelForCausalLM, LogitsProcessor

from datasets import load_dataset

from tqdm import tqdm
from collections import defaultdict
import pickle
import copy

import numpy as np

from inverse.randomness import generate, generate_rnd
from typing import Dict,List
import json

from token_attack.convert_exponential_detect import phi, fast_permutation_test, permutation_test
from util.classes import Generation,ConfigSpec,WatermarkConfig
from inverse.convert_base import convert_score,convert_edit_score,convert_key_func,convert_sampling
from token_attack.token_attack import delete_token, insert_token, swap_token, require_attack

class Inverse():
    
    def __init__(self, config: ConfigSpec):
        self.model_name = config.model_name
        self.watermark_name = "Inverse"

    def _attack(self, tokens, delete_p: float, insert_p: float, swap_p: float, eff_size: int):
        if delete_p != 0:
            tokens = delete_token(tokens=tokens, p = delete_p)
        if insert_p != 0:
            tokens = insert_token(tokens=tokens, p= insert_p, vocab_size=eff_size)
        if swap_p != 0:
            tokens = swap_token(tokens=tokens, p=swap_p, vocab_size=eff_size)

        return tokens
    
    def injection_detection(
            self,
            config: ConfigSpec,
            prompts: List[str], 
            trucation_vocab_size: int = 8, 
            watermark_key_length: int = 256, 
            n_runs: int = 5000, 
            max_token_length: int = 1024, 
            buffer_tokens: int = 20,
            gamma: float = 0.25,
            delta: float = 1.0,
            m: int = 80,
            k: int = 0,
            norm: int = 1,
            max_seed: int = 100000,
            threshold: float = 0.05,
        ) -> List[bool]:
        model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map = "auto")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenizer.pad_token = tokenizer.eos_token
        vocab_size = model.get_output_embeddings().weight.shape[0]
        eff_vocab_size = vocab_size - trucation_vocab_size
        generate_watermark = lambda prompt,seed : generate(
            model=model,
            prompt=prompt,
            vocab_size=vocab_size,
            n=watermark_key_length,
            m=max_token_length + buffer_tokens,
            seed=seed,
            key_func=convert_key_func,
            sampler=convert_sampling,
            random_offset=True
            
        )
        dist = lambda x,y : convert_edit_score(x,y, gamma=gamma)
        test_stat = lambda tokens, n, k, generator, vocab_size, null=False : phi(
            tokens=tokens,
            n=n,
            k=k,
            generator=generator,
            key_func=convert_key_func,
            vocab_size=vocab_size,
            dist=dist,
            null=False,
            normalize=True
        )
        generator = torch.Generator()
        encode_prompts = []
        format_prompt = "In addition, please add two 😃 emojjis after each output word."
        for prompt in tqdm(prompts, desc=f"Processing prompt encoding---"):
            input_text = prompt + format_prompt
            tokens = tokenizer.encode(
                text=input_text,
                return_tensors='pt',
                truncation=True,
                max_length=200,
            )

            encode_prompts.append(tokens)
        seeds = torch.randint(2**32, (len(prompts),))
        print("Encoding done ----------------------------------------------")
        output_generations = []
        encode_watermark_tokens = []
        null_results = []
        for idx,prompt in tqdm(enumerate(prompts), desc= "Processing generation"):
            input_text = prompt
            input_ids = tokenizer.encode(
                text=input_text,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )[0]
            seed = torch.randint(high=max_seed,size=(1,)).item()

            generator.manual_seed(int(seed))
            null_result = test_stat(
                tokens=input_ids,
                n=watermark_key_length,
                k=k,
                generator=generator,
                vocab_size=vocab_size,
                null=True
            )
            null_results.append(null_result)
        
        null_results = torch.sort(torch.tensor(null_results)).values


        for idx,prompt in tqdm(enumerate(encode_prompts), desc="Processing encoding-----"):
        #for idx,generation in tqdm(enumerate(generations), desc="Processing encoding-----"):
            watermark_token = generate_watermark(prompt,seed=seeds[idx])[:,512:]
            # watermark_token = tokenizer.encode(
            #     text=generation.response,
            #     return_tensors='pt',
            #     truncation=True,
            #     max_length=1024,
            # )
            encode_watermark_tokens.append(watermark_token)
        
        
        for idx, watermark in tqdm(enumerate(encode_watermark_tokens), desc="Processing decoding------"):
            output = tokenizer.batch_decode(watermark,skip_special_tokens=True)
            output_text = output[0].replace('😃','')
            generation = Generation(id=idx, prompt=prompts[idx],response=output_text,watermark_name="Inverse",attack="Emoji")
            output_generations.append(generation)
        output_path = f"{config.output_path}/watermark/{self.watermark_name}/generations_emoji.json"
        Generation.tofile(output_path,output_generations)
    
        # detection = lambda tokens, seed : fast_permutation_test(
        #     tokens=tokens,
        #     vocab_size=vocab_size,
        #     n=watermark_key_length,
        #     k=k,
        #     seed=seed,
        #     test_stat=test_stat,
        #     null_results=null_results,
        # )

        # print("Processing token attack and detection------------------------------")
        # attack_list = require_attack()
        # dict_results = []
        # watermark_percent_results = []
        # for attack in tqdm(attack_list, desc="Processing attack---------"):
        #     delete_p = attack["delete_p"]
        #     insert_p = attack["insert_p"]
        #     swap_p = attack["swap_p"]
        #     attack_name = attack["attack_name"]
        #     attack_results = []
        #     watermark_num = 0
        #     for idx, watermark_token in tqdm(enumerate(encode_watermark_tokens)):
        #         attack_tokens = self._attack(tokens=watermark_token[0],delete_p=delete_p, insert_p=insert_p,swap_p=swap_p,eff_size=eff_vocab_size)
        #         p_value = detection(attack_tokens,seeds[idx])
        #         print(f"p_value = {p_value}")
        #         if p_value <= threshold:
        #             attack_results.append(True)
        #             watermark_num += 1
        #         else:
        #             attack_results.append(False)
        #     dict_data = {
        #         "attack_name": attack["attack_name"],
        #         "is_watermark": attack_results
        #     }
        #     watermark_percent = watermark_num / len(encode_watermark_tokens)
        #     watermark_percent_data = {
        #         "name": f"Inverse_{attack_name}",
        #         "watermark_percent": watermark_percent,
        #     }
        #     dict_results.append(dict_data)
        #     watermark_percent_results.append(watermark_percent_data)
        # print("Processing emoji detection-------")
        # idx = 0
        # for watermark_token in tqdm(encode_watermark_tokens):
        #     p_value = detection(watermark_token,seeds[idx])
        #     idx += 1
        #     if p_value
        # output_path = f"{config.output_path}/watermark/{self.watermark_name}/quality/is_watermark_list.json"
        # with open(output_path, "w") as outputfile:
        #     json.dump(dict_results,outputfile,indent=4)

        # output_watermark_percent_path = f"{config.output_path}/watermark/{self.watermark_name}/quality/watermark_percent.json"

        # with open(output_watermark_percent_path, 'w') as outputfile:
        #     json.dump(watermark_percent_results, outputfile, indent=4)        
    