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
    transformers_config = TransformersConfig(model=AutoModelForCausalLM.from_pretrained(config.model_name,device_map="auto"),
                                            tokenizer=AutoTokenizer.from_pretrained(config.model_name),
                                            vocab_size=32000,
                                            device=device,
                                            max_new_tokens=200,
                                            min_length=230,
                                            do_sample=True,
                                            no_repeat_ngram_size=4)
        
    # Load watermark algorithm
    format_prompt = "In addition, please add two 😃 emojjis after each output word."
    attack_name = "EmojjiAttack"
    # Prompt
    prompt_path = config.prompt_path
    prompts = read_prompt(prompt_path)
    #attack_list = ["LowercaseAttack","ContractionAttack","ExpansionAttack","SynonymAttack_0.25","SynonymAttack_0.5","SynonymAttack_0.75","SynonymAttack_1.0","ModifyAttack_0.05_0.05_0_0","ModifyAttack_0_0_0.05_0","ModifyAttack_0_0_0.1_0","TypoAttack_0.1","TypoAttack_0.05","MisspellingAttack_0.25","MisspellingAttack_0.5"]
    attack_list = require_attack(config=config)
    dict_list = []
    for watermark_name in watermatk_list:
        
        myWatermark = AutoWatermark.load(algorithm_name=watermark_name, 
                                    algorithm_config=f"/home/lsz/MarkText/config/{watermark_name}.json",
                                    transformers_config=transformers_config)
        output_generations = []
        if watermark_name == "EXP":
            watermark_name = "Exponential"
        elif watermark_name == "EXPEdit":
            watermark_name = "Inverse"
        idx = 0
        watermark_num = 0
        for prompt in tqdm(prompts):
            prompt_emo = prompt + format_prompt
            watermark_text = myWatermark.generate_watermarked_text(prompt_emo)
            generation = Generation(id=idx, prompt=prompt, response=watermark_text, watermark_name=watermark_name, attack=attack_name)
            output_generations.append(generation)
            results = myWatermark.detect_watermark(watermark_text)
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
        output_path = f"{config.output_path}/watermark/{watermark_name}/{attack_name}.json"
        Generation.tofile(output_path, output_generations)
    output_path = f"{config.output_path}/detect_Exponential.json"
    with open(output_path, "w") as outputfile:
        json.dump(dict_list, outputfile, indent=4)

    
if __name__ == "__main__":
    watermark_list = ["Unigram","EXP","EXPEdit"]
    config_path = str(sys.argv[1])
    config = load_file(config_file=config_path)
    generate_model_base(config=config,watermatk_list=watermark_list)
    