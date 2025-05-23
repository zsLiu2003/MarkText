import torch
from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# Device

device = "cuda" if torch.cuda.is_available() else "cpu"
from generate_text import read_prompt
from util.classes import Generation,WatermarkConfig,ConfigSpec
from tqdm import tqdm
from attack import require_attack
from typing import Dict, List
import sys
import json
from generate_text import load_file
def generate_model_base(config:ConfigSpec, watermatk_list:List[str]):

    # Transformers config
    transformers_config = TransformersConfig(model=AutoModelForCausalLM.from_pretrained(config.model_name).to(device),
                                            tokenizer=AutoTokenizer.from_pretrained(config.model_name),
                                            vocab_size=32000,
                                            device=device,
                                            max_new_tokens=512,
                                            min_length=230,
                                            do_sample=True,
                                            no_repeat_ngram_size=4)
        
    # Load watermark algorithm
    # Prompt
    prompt_path = config.prompt_path
    prompts = read_prompt(prompt_path)
    #attack_list = ["Emoji_attack"]
    attack_list = require_attack(config=config)
    for watermark_name in watermatk_list:
        dict_list = []
        myWatermark = AutoWatermark.load(algorithm_name=watermark_name, 
                                    algorithm_config=f"/home/lsz/MarkText/config/{watermark_name}.json",
                                    transformers_config=transformers_config)
        output_generations = []
        if watermark_name == "EXP":
            watermark_name = "Exponential"
        elif watermark_name == "EXPEdit":
            watermark_name = "Inverse"
        #elif watermark_name == "Unigram":
        #    watermark_name = "Unigram2"
        for attack in attack_list:
            attack_name = attack["attack_name"]
            if attack_name is not None:
                # input_path = f"{config.output_path}/watermark/{watermark_name}/generation/{attack_name}.json"
                # generations = Generation.fromfile(input_path)
                watermark_num = 0
                idx = 0
                # for generation in tqdm(generations, desc=f"Processing {watermark_name}_{attack_name} detection--"):
                for prompt in tqdm(prompts, desc=f"processing {watermark_name}"): 
                    watermarked_text = myWatermark.generate_watermarked_text(prompt)
                    generation = Generation(id=idx, watermark_name=watermark_name,prompt=prompt, response=watermarked_text)
                    idx += 1
                    watermarked_text = generation.response
                    is_watermark,score_result = myWatermark.detect_watermark(watermarked_text)
                    output_generations.append(generation)
                    if is_watermark:
                       watermark_num += 1
                    results = myWatermark.detect_watermark(generation.response)
                    is_watermark = results["is_watermarked"]
                    if is_watermark:
                        watermark_num += 1
                percent = watermark_num / len(prompts)
                print(f"{watermark_name} percent = {percent}---")
                dict_data = {
                    "name": f"{watermark_name}_{attack_name}",
                    "percent": percent
                }
                dict_list.append(dict_data)
                output_path_generation = f"{config.output_path}/watermark/Unigram/generation/generations_new.json"
                Generation.tofile(output_path_generation,output_generations)
    output_path = f"{config.output_path}/watermark/{watermark_name}/generation/detect_unigram.json"
    with open(output_path, "w") as outputfile:
        json.dump(dict_list, outputfile, indent=4)

    
if __name__ == "__main__":
    watermark_list = ["KGW"]
    config_path = str(sys.argv[1])
    config = load_file(config_file=config_path)
    generate_model_base(config=config,watermatk_list=watermark_list)
    