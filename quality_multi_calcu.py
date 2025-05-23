import json
import sys
from util.classes import Generation,ConfigSpec
from generate_text import load_file
from attack import require_attack,require_attack_multi
from typing import List, Dict
import csv
def calcu_quality(config: ConfigSpec, watermark_list: List[str]):
    attack_list = require_attack_multi(config=config)
    attack_list_temp = []
    attack_list_temp.append("Name")
    for attack in attack_list:
        attack_name = attack["attack_name"]
        attack_list_temp.append(attack_name)
    #attack_list = ["Emoji_attack"]
    output_path = f"{config.output_path}/watermark/quality_multi_attack.csv"
    for watermark_name in watermark_list:    
        dict_data = []
        dict_data.append(attack_list_temp)
        output_path = f"{config.output_path}/watermark/{watermark_name}/quality_multi_attack.csv"
        for attack1 in attack_list:
            attack1_name = attack1["attack_name"]
            attack1_list = []
            attack1_list.append(attack1_name)
            for attack2 in attack_list:
                attack2_name = attack2["attack_name"]
                input_path = f"{config.output_path}/watermark/{watermark_name}/multi_quality/{attack1_name}_{attack2_name}_quality.json"
                quality = 0
                generations = Generation.fromfile(input_path)
                for generation in generations[:148]:
                    quality += generation.quality
                quality = quality / len(generations)
                print(f"quality = {quality}")
                attack1_list.append(quality)
            dict_data.append(attack1_list)
        with open(output_path, 'w', newline='') as outputfile:
            writer = csv.writer(outputfile)
            writer.writerows(dict_data)

if __name__ == "__main__":
    watermark_list = ["KGW","UniSpach"]
    config_path = str(sys.argv[1])
    config = load_file(config_file=config_path)
    calcu_quality(config=config, watermark_list=watermark_list)
    
