from kgw.kgw_watermark2 import KGW
from inverse.inverse_watermark import Inverse
from exponential.exponential_watermark import Exponential
from  
import sys
import os
import json
from generate_text import load_file
from util.classes import Generation, ConfigSpec,WatermarkConfig
from typing import Dict,List
from tqdm import tqdm
from transformers import AutoTokenizer


def detect(config: ConfigSpec, watermark_list: List[str]):
    attack_list = ["Para_Llama-2", "Para_Llama-3", "TokenAttack_0.01_0.01_0.01", "TokenAttack_0.01_0.0_0.0", "TokenAttack_0.0_0.01_0.0.","TokenAttack_0.0_0.0_0.01"]  
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    for watermark_name in watermark_list:
        for attack_name in attack_list:
            input_path = f"{config.output_path}/watermark/{watermark_name}/{attack_name}.json"
            generations = Generation.fromfile(input_path)
            for generation in tqdm(generations, desc=f"Processing {watermark_name}_{attack_name} detect"):
                input_text = ''
                if len(generation.response) > 1000:
                    input_text = generation.response[:1000]
                else:
                    input_text = generation.response
                
            
    

if __name__ == "__main__":
    watermark_list = ["KGW", "Inverse", "Exponential", "Convert"]
    config_path = str(sys.argv[1])
    config = load_file(config_file=config_path)
    detect(config=config, watermark_list=watermark_list)
    
    