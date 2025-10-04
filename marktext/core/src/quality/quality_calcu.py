import json
import sys
from util.classes import Generation,ConfigSpec
from generate_text import load_file
from attack import require_attack,require_attack_temp
from typing import List, Dict
import csv

def calcu_quality(config: ConfigSpec, watermark_list: List[str]):
    attack_list = require_attack_temp()
    #attack_list = ["Emoji_attack"]
    dict_data = []
    name = ["watermark_attack","quality"]
    dict_data.append(name)
    dict_data.append()
    output_path = f"{config.output_path}/watermark/multi_attack.json"
    for watermark_name in watermark_list:    
        for attack in attack_list:
            attack_name = attack["attack_name"]
            if attack_name is not None:
                input_path = f"{config.output_path}/watermark/{watermark_name}/{attack_name}_quality.json"
                quality = 0
                generations = Generation.fromfile(input_path)
                for generation in generations:
                    quality += generation.quality
                quality = quality / len(generations)
                print(f"quality = {quality}")
                # data = {
                #     "watermark_attack": f"{watermark_name}_{attack_name}",
                #     "quality": quality
                # }
                data = [f"{watermark_name}_{attack_name}"]
                dict_data.append(data)
    with open(output_path, 'w') as outputfile:
        json.dump(dict_data, outputfile, indent=4)


if __name__ == "__main__":
    watermark_list = ["KGW","UniSpach"]
    config_path = str(sys.argv[1])
    config = load_file(config_file=config_path)
    calcu_quality(config=config, watermark_list=watermark_list)
    
