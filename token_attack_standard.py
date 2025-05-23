
model_name = "/data2/huggingface-mirror/dataroot/models/meta-llama/Llama-2-7b-chat-hf"
from transformers import AutoTokenizer
from util.classes import Generation
from typing import Dict
from tqdm import tqdm

from token_attack.token_attack import delete_token, insert_token, swap_token



def token_attack(tokens, delete_p: float, insert_p: float, swap_p: float, max_length: int = 1024):
    if delete_p > 0:
        tokens = delete_token(tokens=tokens, p= delete_p)
    if insert_p > 0:
        tokens = insert_token(tokens=tokens,p= insert_p, vocab_size=max_length)
    if swap_p > 0:
        tokens = swap_token(tokens=tokens, p=swap_p,vocab_size=max_length)
    
    return tokens

def attack(attack_dict: Dict, output_path: str, generations,):
    delete_p = attack_dict["delete_p"]
    attack_name = attack_dict["attack_name"]
    insert_p = attack_dict["insert_p"]
    swap_p = attack_dict["swap_p"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    output_generations = []
    for generation in tqdm(generations, desc=f"Processing {attack_name}----"):
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
    
    Generation.tofile(output_path,output_generations)