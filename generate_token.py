
from kgw.kgw_watermark2 import KGW
from inverse.inverse_watermark import Inverse
from exponential.exponential_watermark import Exponential
import sys
import os
import json
from generate_text import load_file,read_prompt
from util.classes import Generation, ConfigSpec, WatermarkConfig
from unigram_watermark import unigram_generate
from generate_text import read_prompt
import random
def generate_token(config: ConfigSpec, prompt_path):
    prompts = read_prompt(file_path=prompt_path)
    inverse_instance = Inverse(config=config)
    exponential_instance = Exponential(config=config)
    kgw_instance = KGW(model_name=config.model_name)
    inverse_instance.injection_detection(
       config=config,
        prompts=prompts,
       max_token_length=512,
        )
    
    exponential_instance.injection_detection(
        config=config,
        prompts=prompts,
        max_token_length=512,
    )
    kgw_instance.injection(
        prompts=prompts,
        config=config
    )
    prompts = read_prompt(config.prompt_path)
    unigram_generate(model_name=config.model_name,prompts=prompts, output_dir=config.output_path)
if __name__ == "__main__":
    config_path = str(sys.argv[1])
    config = load_file(config_file=config_path)
    prompts_path = config.prompt_path
    generate_token(config=config,prompt_path=prompts_path)
    