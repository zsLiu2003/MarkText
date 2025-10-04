from transformers import AutoTokenizer, AutoModelForCausalLM
from attack import require_attack
from generate_text import load_file
import json
from tqdm import tqdm
import sys
import os
from util.classes import Generation,ConfigSpec,WatermarkConfig
from typing import Dict, List

def swap_attack(config: ConfigSpec, watermark_list: List[str]):
    attack_list = require_attack(config=config)
    for watermark in watermark_list:
        for attack in attack_list:
            attack_name = attack["attack_name"]
            if "Modify" in attack_name:
                input_path = f"{config.output_path}/watermark/{watermark}/generations_watermark.json"
                generations = Generation.fromfile(input_path)
                instance = attack["attack"]
                output_generations = []
                for generation in tqdm(generations, desc=f"Processing {attack_name}----------"):
                    input_text = generation.response
                    output_text = instance.warp(input_text)
                    generation.response = output_text
                    output_generations.append(generation)
                output_path = f"{config.output_path}/watermark/{watermark}/{attack_name}.json"
                Generation.tofile(output_path,output_generations)
                

if __name__ == "__main__":
    config_path = str(sys.argv[1])
    watermark_list = ["Format", "UniSpach", "Unigram", "Lexical"]
    config = load_file(config_path)
    swap_attack(config=config, watermark_list=watermark_list)
    