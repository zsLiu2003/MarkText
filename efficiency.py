from attack import require_attack,require_attack_multi
from generate_text import load_file
from util.classes import Generation
from tqdm import tqdm
import sys
from phara_attack import Llama2Attack, Llama3Attack, TranslationAttack,DipperAttack
from helm_attack import setup

config_path = sys.argv[1]
config = load_file(config_file=config_path)
attack_list = require_attack(config=config)
temp_attack_list = require_attack_multi(config=config)

llama2_instance = Llama2Attack()
llama3_instance = Llama3Attack()
llama3_instance.attack(input_path="/data1/lzs/MarkText/watermark/KGW/generations_KGW.json", config=config, watermark_name="KGW")
le = len(temp_attack_list)

print(f"le = {le}")

# attack = temp_attack_list[9]
# attack_name = attack["attack_name"]
# attack_instance = attack["attack"]
# input_path = "/data1/lzs/MarkText/eff.json"


# print(f"attack name = {attack_name}")

# watermark_name = "KGW"
# input_path = "/data1/lzs/MarkText/watermark/KGW/generations_KGW.json"
# generations = Generation.fromfile(input_path)
# if "Para" in attack_name:
#     attack_instance.attack(input_path, config, watermark_name)
# elif "Translate" in attack_name:
#     attack_instance.attack(input_path, config, watermark_name,attack_name)
# elif "Token" in attack_name:
#     exit()
# else:
#     setup(config)
#     output_generations = []
#     output_path = f"{config.output_path}/watermark/{watermark_name}/generation/{attack_name}.json"
#     for generation in tqdm(generations):
#         input_text = generation.response
#         output_text = attack_instance.warp(text = input_text)
#         generation.attack = attack_name
#         generation.response = output_text
#         output_generations.append(generation)
#     Generation.tofile(output_path,output_generations)
