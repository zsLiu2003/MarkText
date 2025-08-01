import sys
import json
from util.classes import Generation,ConfigSpec,WatermarkConfig
from generate_text import load_file
from tqdm import tqdm
from watermark_scheme import LexicalWatermark,FormatWatermark,UniSpachWatermark
from unigram_watermark import unigram_detect
from typing import Dict,List
from attack import require_attack
def detect(config: ConfigSpec, input_path: str) -> Dict:
    genertaions = Generation.fromfile(input_path)
    watermark_name = str(genertaions[0].watermark_name)
    attack_name = str(genertaions[0].attack)
    watermark_percent = 0
    if watermark_name == "Lexical":
        watermark_instance = LexicalWatermark(config=config,tau_word=0.75,tau_sent=0.8,lamda=0.5)
    elif watermark_name == "Format":
        watermark_instance = FormatWatermark(0.6)
    elif watermark_name == "UniSpach":
        watermark_instance = UniSpachWatermark(0.6)
    elif watermark_name == "Unigram":
        dict_data = unigram_detect(model_name=config.model_name,
                       fraction=0.5,
                       intput_dir=input_path,
                       attack_name=attack_name,
                       watermark_name=watermark_name)
        watermark_percent = dict_data["watermark_percent"]

    if watermark_name != "Unigram":
        num = len(genertaions)
        watermark_num = 0
        for generation in tqdm(genertaions):
            input_text = generation.response
            is_watermark = watermark_instance.detection(text=input_text, p_value=0.01)
            if is_watermark:
                watermark_num += 1
        watermark_percent = watermark_num / num
        dict_data = {
            "name": f"{watermark_name}_{attack_name}",
            "watermark_percent": watermark_percent,
        }
    print(f"{watermark_name}_{attack_name} watermark percent = {watermark_percent}")      

    return dict_data   

def main(config:ConfigSpec, watermark_list: List[str]):
    dict_list = []
    attack_list = require_attack(config=config)
    attack_list = attack_list[20:]
    watermark_instance = LexicalWatermark(config=config,tau_word=0.75,tau_sent=0.8,lamda=0.5)
    for watermark_name in watermark_list:
        for attack in attack_list:
            attack_name = attack["attack_name"]
            input_generation_path = f"{config.output_path}/watermark/{watermark_name}/{attack_name}.json"
            generations = Generation.fromfile(input_generation_path)
            num = len(generations)
            for generation in tqdm(generations):
                input_text = generation.response
                is_watermark = watermark_instance.detection(text=input_text, p_value=0.01)
                if is_watermark:
                    watermark_num += 1
            watermark_percent = watermark_num / num
            dict_data = {
                "name": f"{watermark_name}_{attack_name}",
                "watermark_percent": watermark_percent,
            }
            dict_list.append(dict_data)
            print(f"{watermark_name}_{attack_name} watermark percent = {watermark_percent}")      
        
    output_path = f"{config.output_path}/watermark/detect_Lexical6.json"
    with open(output_path, 'w') as output_file:
        json.dump(dict_list,output_file,indent=4)


if __name__ == "__main__":
    config_path = str(sys.argv[1])
    watermark_list = ["Lexical"]
    config = load_file(config_file=config_path)
    
    main(config=config, watermark_list=watermark_list)
    
    
    
                
    