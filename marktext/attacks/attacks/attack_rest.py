from helm_attack import MisspellingAttack,TypoAttack,LowercaseAttack,ContractionAttack,ExpansionAttack
from modify_attack import ModifyAttack
from synonym_attack import SynonymAttack
from phara_attack import Llama2Attack, Llama3Attack, TranslationAttack,DipperAttack
from typing import Dict,List
from util.classes import ConfigSpec, Generation, WatermarkConfig
from helm_attack import setup
from tqdm import tqdm
import sys
import json
from generate_text import load_file
from token_attack import token_attack

def require_attack(config: ConfigSpec) -> List[Dict]:
    dict_list = []
    dict_data = {
        "attack_name": "MisspellingAttack_0.25",
        "attack": MisspellingAttack(0.25)            
    }
    dict_list.append(dict_data)
    dict_data = {
        "attack_name": "MisspellingAttack_0.5",
        "attack": MisspellingAttack(0.5)
    }
    dict_list.append(dict_data)
    dict_data = {
        "attack_name": "TypoAttack_0.05",
        "attack": TypoAttack(0.05),
    }
    dict_list.append(dict_data)
    dict_data = {
        "attack_name": "TypoAttack_0.1",
        "attack": TypoAttack(0.1),
    }
    dict_list.append(dict_data)
    dict_data = {
        "attack_name": "LowercaseAttack",
        "attack": LowercaseAttack()
    }
    dict_list.append(dict_data)
    dict_data = {
        "attack_name": "ContractionAttack",
        "attack": ContractionAttack()
    }
    dict_list.append(dict_data)
    dict_data = {
        "attack_name": "ExpansionAttack",
        "attack": ExpansionAttack(),
    }
    dict_list.append(dict_data)
    dict_data = {
        "attack_name": "ModifyAttack_0.05_0.05_0_0",
        "attack": ModifyAttack(0.05,0.05,0,0),
    }
    dict_list.append(dict_data)
    dict_data = {
        "attack_name": "ModifyAttack_0_0_0.05_0",
        "attack": ModifyAttack(0,0,0.05,0),
    }
    dict_list.append(dict_data)
    dict_data = {
        "attack_name": "ModifyAttack_0_0_0.1_0",
        "attack": ModifyAttack(0,0,0.1,0),
    }
    dict_list.append(dict_data)
    dict_data = {
        "attack_name": "ModifyAttack_0_0_0_0.05",
        "attack": ModifyAttack(0,0,0,0.05),
    }
    dict_list.append(dict_data)
    syno_instance = SynonymAttack(p=1.0,confidence=0.01)
    dict_data = {
        "attack_name": "SynonymAttack_1.0",
        "attack": syno_instance,
    }
    dict_list.append(dict_data)
    syno_instance = SynonymAttack(p = 0.75, confidence=0.01)
    dict_data = {
        "attack_name": "SynonymAttack_0.75",
        "attack": syno_instance,
    }
    dict_list.append(dict_data)
    syno_instance = SynonymAttack(p = 0.5, confidence=0.01)
    dict_data = {
        "attack_name": "SynonymAttack_0.5",
        "attack": syno_instance
    }
    dict_list.append(dict_data)
    syno_instance = SynonymAttack(p = 0.25, confidence=0.01)
    dict_data = {
        "attack_name": "SynonymAttack_0.25",
        "attack": syno_instance
    }
    dict_list.append(dict_data)
    llama2_instance = Llama2Attack()
    llama3_instance = Llama3Attack()
    trans_instance = TranslationAttack()
    dipper_instance = DipperAttack(config)
    dict_data = {
        "attack_name": "MisspellingAttack_0.75",
        "attack": MisspellingAttack(0.75)
    }
    dict_list.append(dict_data)
    dict_data = {
        "attack_name": "TypoAttack_0.20",
        "attack": TypoAttack(0.20),
    }
    dict_list.append(dict_data)
    dict_data = {
        "attack_name": "Translate_german",
        "attack": trans_instance,
    }
    dict_list.append(dict_data)
    dict_data = {
        "attack_name": "Para_Llama-2",
        "attack": llama2_instance,
    }

    dict_list.append(dict_data)
    dict_data = {
        "attack_name": "Para_Llama-3",
        "attack": llama3_instance,
    }
    dict_list.append(dict_data)
    dict_data = {
        "attack_name": "Translate_france",
        "attack": trans_instance,
    }
    dict_list.append(dict_data)
    dict_data = {
        "attack_name": "Translate_russion",
        "attack": trans_instance,
    }
    dict_list.append(dict_data)
    dict_data = {
        "attack_name": "Para_dipper",
        "attack": dipper_instance,
    }
    dict_list.append(dict_data)
    attack_list = token_attack.require_attack()
    for attack in attack_list:
        dict_data = {
            "attack_name": attack["attack_name"],
            "attack": None,
        }
        dict_list.append(dict_data)

    return dict_list


def require_attack_temp():
    
    dict_list = []
    llama2_instance = Llama2Attack()
    llama3_instance = Llama3Attack()
    dict_data = {
        "attack_name": "Para_Llama-2",
        "attack": llama2_instance,
    }

    dict_list.append(dict_data)
    dict_data = {
        "attack_name": "Para_Llama-3",
        "attack": llama3_instance,
    }
    return dict_list


def attack(config: ConfigSpec, input_path: str):
    attack_list = require_attack(config)
    generations = Generation.fromfile(input_path)
    watermark_name = "KGW"
    for attack in attack_list:
        attack_instance = attack["attack"]
        attack_name = attack["attack_name"]
        if attack_name is not None:
            output_path = f"{config.output_path}/watermark/{watermark_name}/{attack_name}.json"
            if "Para_dipper" in attack_name:
                attack_instance.attack(input_path, config, watermark_name,output_path)
            elif "Translate_france" in attack_name or "Translate_russion" in attack_name:
                attack_instance.attack(input_path, config, watermark_name,attack_name,output_path)
            elif "Token" in attack_name:
                continue
            else:
                if "Translate" in attack_name or "Llama" in attack_name:
                    continue
                setup(config)
                output_generations = []
                for generation in generations:
                    input_text = generation.response
                    output_text = attack_instance.warp(text = input_text)
                    generation.attack = attack_name
                    generation.response = output_text
                    output_generations.append(generation)
                Generation.tofile(output_path,output_generations)

def attack_temp(config: ConfigSpec, watermark_list: List[str]):
    attack_list = require_attack_temp()
    for watermark_name in watermark_list:
        input_path = f"{config.output_path}/watermark/{watermark_name}/generations_{watermark_name}.json"
        generations = Generation.fromfile(input_path)
        for attack in attack_list:
            attack_name = attack["attack_name"]
            instance = attack["attack"]
            if "german" in attack_name:
                instance.attack(input_path,config,watermark_name, attack_name)
            else:
                setup(config)
                output_generations = []
                output_path = f"{config.output_path}/watermark/{watermark_name}/{attack_name}.json"
                for generation in generations:
                    input_text = generation.response
                    output_text = instance.warp(text = input_text)
                    generation.attack = attack_name
                    generation.response = output_text
                    output_generations.append(generation)
                Generation.tofile(output_path,output_generations)

if __name__ == "__main__":
    config_path = str(sys.argv[1])
    input_path = str(sys.argv[2])
    watermark_list = ["KGW"]
    config_class = load_file(config_file=config_path)
    attack(config=config_class, input_path= input_path)
    