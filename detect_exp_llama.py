from exponential.exponential_watermark import Exponential
import json
import sys
from util.classes import Generation,ConfigSpec,WatermarkConfig
from generate_text import read_prompt, load_file
def detect(config:ConfigSpec):
    attack_list = ["Para_Llama-2", "Para_Llama-3"]
    prompts = read_prompt(config.prompt_path)
    watermark_name = "Exponential"
    instance = Exponential(config=config)
    for attack_name in attack_list:
        input_path = f"{config.output_path}/watermark/{watermark_name}/{attack_name}.json"
        instance.injection_detection(
            config=config,
            prompts=prompts,
            input_path=input_path
        )
        


if __name__ == "__main__":
    config_path = str(sys.argv[1]) 
    config = load_file(config_path)
    detect(config=config)
    
    