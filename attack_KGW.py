import torch
from phara_attack import Llama2Attack, Llama3Attack
import sys
import csv
from util.classes import Generation
from generate_text import load_file
from util.classes import ConfigSpec
import json
def process_data(input_path: str, watermark_name: str) -> str:
    
    dict_list = []
    output_path = "/data1/lzs/MarkText/watermark/" + watermark_name + "/generation.json"
    key = 0
    if watermark_name == "Exponential":
        key = 7
    elif watermark_name == "KGW":
        key = 8
    elif watermark_name == "Inverse":
        key = 640
    elif watermark_name == "Convert":
        key = 832
    with open(input_path, "r") as infile:
        reader = csv.DictReader(infile, delimiter="\t")
        for raw in reader:
            if raw["watermark"] != '-':
                watermark = eval(raw["watermark"])
                if int(watermark["key_len"]) == key:
                    dict_list.append(raw)
    output_generations = []
    for idx,cur in enumerate(dict_list):
        generation = Generation(prompt=cur["prompt"], id=idx, response=cur["response"], watermark_name=watermark_name, )
        output_generations.append(generation)
    Generation.tofile(output_path, output_generations)
    return output_path

if __name__ == "__main__":
    watermark_name = str(sys.argv[1])
    input_path = str(sys.argv[2])
    config_path = str(sys.argv[3])
    config = load_file(config_file=config_path)
    output_path = process_data(input_path=input_path, watermark_name=watermark_name)
    instance1 = Llama2Attack()
    instance2 = Llama3Attack()
    if watermark_name == "Convert":
        instance1.attack(input_path=output_path,config=config, watermark_name=watermark_name)
        instance2.attack(input_path=output_path,config=config,watermark_name=watermark_name)
    else:
        instance1.attack(input_path=output_path,config=config, watermark_name=watermark_name)