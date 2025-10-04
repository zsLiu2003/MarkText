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
from attack import require_attack

rate_prompt = "[INST] <<SYS>> You are given a prompt and a response, and you provide a grade out of 100 measuring the quality of the response, in terms of accuracy, level of details, and typographical, grammatical and lexical correctness. Remove points as soon as one of the criteria is missed. <</SYS>> Prompt: {}\nResponse: {}[/INST] Grade: "

def quality(config_file, watermark_name_list: List[str]):
    config = load_file(config_file)
    model_name = "/data2/huggingface-mirror/dataroot/models/meta-llama/Llama-2-13b-chat-hf"
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map = 'auto')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    attack_list = require_attack(config=config)
    #attack_list = ["TokenAttack_0.01_0.01_0.01","TokenAttack_0.01_0.0_0.0","TokenAttack_0.0_0.01_0.0","TokenAttack_0.0_0.0_0.01"]
    for watermark_name in tqdm(watermark_name_list, desc=f"Quality---"):
        for attack_dict in tqdm(attack_list):
            attack_name = attack_dict["attack_name"]
            if attack_name != None:
                input_path = f"{config.output_path}/watermark/{watermark_name}/{attack_name}.json"
                output_generations = Generation.fromfile(input_path)
                """
                dataset = load_dataset('json', data_files=str(input_path), split="train")
                prompts = dataset["prompt"]
                response = dataset["full_response"]
                le = len(prompts)

                print(f"response = {response[0]}")
                print(f"prompt = {prompts[0]}")
                print(f"len = {le}")
                
                for id in range(le):
                    generation = Generation(id=id, prompt=prompts[id][0], response=response[id], watermark_name=watermark_name,attack=attack_name)
                    output_generations.append(generation)
                """
                
                #generations = Generation.fromfile(config.output_path)
                #model_name = config.model_name

                
                tasks = []
                for generation in output_generations:
                    if len(generation.response) > 1000:
                        generation.response = generation.response[:1000]
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

                max_token_length = 1500
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
                        max_length = 1536,
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
                    
                output_path = f"{config.output_path}/watermark/{watermark_name}/generation/{attack_name}_quality.json"
                Generation.tofile(output_path, output_generations)
            

if __name__ == "__main__":
    config_path = str(sys.argv[1])
    watermark_name_list = ["Unigram"]
    quality(config_file=config_path, watermark_name_list=watermark_name_list)
