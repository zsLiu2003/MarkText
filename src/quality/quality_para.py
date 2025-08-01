from util.classes import Generation
from generate_text import load_file
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import re
from dataclasses import replace
import sys
import json
from datasets import load_dataset
rate_prompt = "[INST] <<SYS>> You are given a prompt and a response, and you provide a grade out of 100 measuring the quality of the response, in terms of accuracy, level of details, and typographical, grammatical and lexical correctness. Remove points as soon as one of the criteria is missed. <</SYS>> Prompt: {}\nResponse: {}[/INST] Grade: "

def quality(config_file, watermark_name: str, attack_name: str, input_path: str):
    output_generations = []
    output_generations = Generation.fromfile(input_path)
    print(input_path)
    config = load_file(config_file)
    #generations = Generation.fromfile(config.output_path)
    model_name = config.model_name
    model_name = "/data2/huggingface-mirror/dataroot/models/meta-llama/Llama-2-13b-chat-hf"
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map = 'auto')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    tasks = []
    for generation in output_generations:
        tasks.append(
            rate_prompt.format(
                generation.prompt.replace("[/INST]", "")
                .replace("[/INST]", "")
                .replace("<<SYS>>","")
                .replace("<</SYS>>", "")
                .replace(
                    "You are a helpful assistant. Always answer in the most accurate way.",
                    "",
                )
                .strip(),
                generation.response,
            )
        )

    max_token_length = 1536
    print(f"len(tasks):{len(tasks)}")
    for i in tqdm(range(len(tasks)), total=len(tasks), desc="Encoding and then Rating-----"):
        task = tasks[i]
        if len(task) > max_token_length:
            encoded_task = tokenizer(task)["input_ids"]
            if (len(encoded_task) > max_token_length):
                print(
                "Warning: Task too long ({} tokens), clipping to {} tokens".format(
                    len(encoded_task), max_token_length
                )
            )
            task = tokenizer.decode(encoded_task[:max_token_length])
            tasks[i] = task

    print("Encoding done, ready for quality!")

    num_regex = re.compile("([0-9]+\.*[0-9]*)")  
    id = 0
    for task in tqdm(tasks, desc='rating------'):
        input_ids = tokenizer.encode(
            task,
            return_tensors='pt',
            truncation=True,
        )            
        device = 'cuda'
        le = len(input_ids)
        generated_texts = model.generate(
            input_ids = input_ids.to(device),
            max_length = 2500,
        )
        generated_texts = tokenizer.batch_decode(generated_texts, skip_special_tokens=True)
        generated_texts = str(generated_texts[0])
        quality = 0.0
        matches = re.search(r"Grade: (\d+)/100", generated_texts)
        if matches:
            grade_number = int(matches.group(1))
            quality = grade_number / 100.0
        else:
            quality = 0.0
        output_generations[id] = replace(output_generations[id], quality = quality)
        id += 1
        
    output_path = f"{config.output_path}/watermark/{watermark_name}/{attack_name}_quality.json"
    Generation.tofile(output_path, output_generations)
        

if __name__ == "__main__":
    config_path = str(sys.argv[1])
    watermark_name = str(sys.argv[2])
    attack_name = str(sys.argv[3])
    input_path = str(sys.argv[4])
    quality(config_file=config_path, watermark_name=watermark_name,attack_name=attack_name, input_path=input_path)