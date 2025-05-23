import sys
import json
from util.classes import Generation,ConfigSpec,WatermarkConfig
from generate_text import load_file
from tqdm import tqdm
from watermark_scheme import LexicalWatermark,FormatWatermark,UniSpachWatermark
from unigram_watermark import unigram_detect
from typing import Dict,List
import csv
from attack import require_attack,require_attack_multi
def detect(config: ConfigSpec) -> Dict:
    watermark_name = "UniSpach"
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

    attack_list = require_attack_multi(config)
    if watermark_name != "Unigram":
        dict_list = []
        attack_list_temp = []
        attack_list_temp.append("")
        for attack in attack_list:
            attack_name = attack["attack_name"]
            attack_list_temp.append(attack_name)
        dict_list.append(attack_list_temp)
        for attack1 in attack_list:
            attack1_name = attack1["attack_name"]
            percent_list = []
            percent_list.append(attack1_name)
            for attack2 in attack_list:
                attack2_name = attack2["attack_name"]
                input_path = f"{config.output_path}/watermark/{watermark_name}/multi_attack/{attack1_name}_{attack2_name}.json"
                genertaions = Generation.fromfile(input_path)
                watermark_num = 0
                for generation in tqdm(genertaions):
                    input_text = generation.response
                    is_watermark = watermark_instance.detection(text=input_text, p_value=0.05)
                    if is_watermark:
                        watermark_num += 1
                watermark_percent = watermark_num / len(genertaions)
                dict_data = {
                    "name": f"{watermark_name}_{attack_name}",
                    "watermark_percent": watermark_percent,
                }
                percent_list.append(watermark_percent)
            dict_list.append(percent_list)
        output_path = f"{config.output_path}/watermark/{watermark_name}/multi_detect.csv"
        with open(output_path, "w") as outputfile:
            writer = csv.writer(outputfile)
            writer.writerows(dict_list)  

def main(config:ConfigSpec, watermark_list: List[str]):
    dict_list = []
    attack_list = require_attack(config=config)
    for watermark_name in watermark_list:
        for attack in attack_list:
            attack_name = attack["attack_name"]
            if "Modify" in attack_name:
                input_generation_path = f"{config.output_path}/watermark/{watermark_name}/{attack_name}.json"
                dict_data = detect(config=config, input_path= input_generation_path,attack_name=attack_name)
                dict_list.append(dict_data)
    output_path = f"{config.output_path}/watermark/detect_llama.json"
    with open(output_path, 'w') as output_file:
        json.dump(dict_list,output_file,indent=4)


if __name__ == "__main__":
    config_path = str(sys.argv[1])
    watermark_list = ["UniSpach"]
    config = load_file(config_file=config_path)
    detect(config=config)
    
    
    
    
                
    