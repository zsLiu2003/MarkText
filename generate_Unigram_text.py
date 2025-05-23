import llama_standard
import yaml
import dacite
from dacite import from_dict
from util.classes import ConfigSpec
from util.classes import WatermarkConfig
from util.classes import Generation
from watermark_scheme import FormatWatermark, LexicalWatermark, UniSpachWatermark
import random
from random import randint
import multiprocessing
from dataclasses import replace
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import json
import torch
from torch.utils.data import DataLoader,Dataset
from generate_prompt import raw_prompts
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from unigram_watermark import unigram_generate



class MyDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer.encode(
            self.texts[idx], 
            max_length=self.max_length, 
            padding='max_length', 
            truncation=True, 
            return_tensors="pt"
            )
        return encoded


def load_file(config_file):
    with open(config_file, "r") as infile:
        config_dict = yaml.safe_load(infile)
        data_instance = from_dict(data_class=ConfigSpec, data = config_dict)
        return data_instance

def call_llama2(config_file,prompts, max_input_tokens = 1024,max_output_tokens = 2048):
    config = load_file(config_file)
    model_name = config.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map = 'auto')
    model.eval()
    output_path = str(str(config.output_path) + "/generations.json")
    output_generations = []
    id = 0
    for prompt in tqdm(prompts):
        input_ids = tokenizer.encode(
            prompt,
            return_tensors = "pt",
            truncation = True,
            max_length = max_input_tokens
        )
        device = 'cuda'
        generate_tokens_with_prompt = model.generate(
            input_ids = input_ids.to(device),
            max_length = max_output_tokens
        )
        generate_text_with_prompt = tokenizer.batch_decode(generate_tokens_with_prompt,skip_special_tokens=True)
        index = generate_text_with_prompt[0].find("[/INST]")
        generate_text = generate_text_with_prompt[0][index + len("[/INST]"):]
        generation = Generation(prompt = str(prompt), response=str(generate_text),id = id)
        output_generations.append(generation)
        id += 1
    Generation.tofile(str(output_path),output_generations)
    return output_path

def read_prompt(file_path):
    prompts = []
    with open(file_path, 'r') as file:
        for item in file:
            prompts.append(item.rsplit('\n'))
    return prompts

def generate_watermarktext(config_file, watermarks=None, generate_text_path = str):
    config_class= load_file(config_file)
    prompts = read_prompt(config_class.prompt_path)
    #output_path = call_llama2(config_file,prompts)
    output_path = str(generate_text_path)
    if not watermarks:
        with open(config_class.watermark_path, encoding='utf-8') as infile:
            watermarks = [
                replace(
                    WatermarkConfig.from_str(line.strip()), tokenizer = config_class.model_name
                ) for line in infile.read().split("\n")
                if len(line)
            ]
    generations = Generation.fromfile(output_path)
    efficiency = 0
    for watermark in tqdm(watermarks,desc = "Processing Watermarks"):
        output_generations = []
        id = 0
        watermark_path = ''
        watermark_name = watermark.name
        if watermark_name == "Lexical":
            instance_Lexical = LexicalWatermark(config=config_class, tau_word=0.75, tau_sent=0.8, lamda=0.5)
        if watermark_name == "Unigram":
            unigram_generate(model_name="/data2/huggingface-mirror/dataroot/models/meta-llama/Llama-2-7b-chat-hf", 
                             prompts= prompts,
                             output_dir= config_class.output_path)
        for generation in tqdm(generations,desc = f"Processing {watermark.name} watermarks:"):
            watermark_name = str(watermark.name)
            if watermark_name == "Format":
                watermark_path = config_class.output_path + "/watermark/Format.json"
                input = generation.response
                instance = FormatWatermark(0.6)
                output = instance.injection(str(input))
                generation.response = output
                generation.watermark_name = "Format"
                generation.id = id
                id += 1
                output_generations.append(generation)
            elif watermark_name == "UniSpach":
                watermark_path= config_class.output_path + "/watermark/UniSpach.json"
                input = str(generation.response)
                instance = UniSpachWatermark(0.6)
                output = instance.injection(input)
                generation.response = output
                generation.watermark_name = "UniSpach"
                generation.id = id
                id += 1
                output_generations.append(generation)
                
            elif watermark_name == "Lexical":
                watermark_path = config_class.output_path + "/watermark/Lexical.json"
                input = generation.response
                output = instance_Lexical.injection(str(input))
                generation.response = str(output)
                generation.watermark_name = "Lexical"
                generation.id = id
                id += 1
                output_generations.append(generation)
            elif watermark_name == "Generative":
                watermark_path = config_class.output_path + "/watermark/Generative.json"
                pass
        Generation.tofile(watermark_path,output_generations)


