import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from token_attack.token_attack import delete_token,swap_token,insert_token,require_attack
import sys
import json
from generate_text import load_file
from util.classes import Generation,ConfigSpec
from tqdm import tqdm
from typing import Dict,List


def token_attack(tokens, delete_p: float, insert_p: float, swap_p: float, max_length: int = 1024):
    if delete_p > 0:
        tokens = delete_token(tokens=tokens, p= delete_p)
    if insert_p > 0:
        tokens = insert_token(tokens=tokens,p= insert_p, vocab_size=max_length)
    if swap_p > 0:
        tokens = swap_token(tokens=tokens, p=swap_p,vocab_size=max_length)
    
    return tokens

def attack(config_path: str, watermark_list: List[str]):
    config = load_file(config_file=config_path)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    for watermark_name in watermark_list:
        input_path = f"{config.output_path}/watermark/{watermark_name}/generations_{watermark_name}.json"
        generations = Generation.fromfile(input_path)
        attack_list = require_attack()
        for attack in tqdm(attack_list, desc="attack-----"):
            delete_p = attack["delete_p"]
            insert_p = attack["insert_p"]
            swap_p = attack["swap_p"]
            attack_name = attack["attack_name"]
            output_generations = []
            print(attack_name)
            for generation in tqdm(generations, desc = "---------"):
                input_text = generation.response
                input = tokenizer(
                    text=input_text,
                    return_tensors='pt',
                    truncation=True,
                    max_length=1024
                )
                input_ids= input["input_ids"]
                input_ids = input_ids[0].to("cuda")
                tokens = token_attack(tokens=input_ids, delete_p=delete_p, insert_p=insert_p, swap_p=swap_p, max_length=1024)
                tokens = tokens.unsqueeze(0)
                output_text = tokenizer.batch_decode(tokens, skip_special_tokens=True)
                generation.response = output_text[0]
                output_generations.append(generation)
            output_path = f"{config.output_path}/watermark/{watermark_name}/generation/{attack_name}.json"
            Generation.tofile(output_path, output_generations)




if __name__ == "__main__":
    
    watermark_name = ["Unigram"]
    config_path = str(sys.argv[1])

    attack(config_path=config_path, watermark_list=watermark_name)
    
