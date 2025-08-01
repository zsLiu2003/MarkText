from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import sys
from util.classes import Generation,ConfigSpec
from generate_text import load_file
from generate_text import read_prompt
from tqdm import tqdm
import torch

def generate(config: ConfigSpec, input_path: str):
    with open(input_path, "r") as infile:
        for line in infile:
            line = line.strip()
            model_name = line
            if "extraction" in model_name:
                attack_name = "Extraction"
            elif "LoRA" in model_name:
                attack_name = "LoRA"
            elif "full" in model_name:
                attack_name = "full"
            elif "dare_ties" in model_name:
                attack_name = "dare_ties"
            elif "dare_task" in model_name:
                attack_name = "dare_task"
            elif "ties" in model_name:
                attack_name = "ties"
            elif "task" in model_name:
                attack_name = "task"
            if "Llama" in model_name:
                name = "Llama-2-chat"
            elif "WizardLM" in model_name:
                name = "WizardLM"
            else:
                name = "Merge"
            tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map = "auto", load_in_8bit=True)
            output_generations = []
            
            prompts = read_prompt(config.prompt_path)
            for idx, prompt in tqdm(enumerate(prompts), desc=model_name):
                input_ids = tokenizer.encode(
                    text=prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=1024,        
                )
                
                input_ids = input_ids.to(model.device)
                le = len(input_ids)
                attention_mask = torch.ones_like(input_ids)
                output = model.generate(
                    input_ids = input_ids,
                    max_length = 1536,
                    attention_mask=attention_mask
                )
                le2 = len(prompt)
                output_text = tokenizer.batch_decode(output,skip_special_tokens=True)[0]
                generation = Generation(id=idx, prompt=prompt, response=output_text[le2:],watermark_name="Quantize",attack=attack_name)
                output_generations.append(generation)
                output_path = f"{config.output_path}/watermark/quantize/{name}_{attack_name}.json"
                Generation.tofile(output_path, output_generations)

if __name__ == "__main__":
    config_path = str(sys.argv[1])
    input_path = str(sys.argv[2])
    config = load_file(config_file=config_path)
    
    generate(config=config, input_path=input_path)
