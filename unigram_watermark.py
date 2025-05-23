import torch
import argparse
import json
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer,LlamaTokenizer,LogitsProcessorList
from unigram_base import UnigramWatermarkDetector,UnigramWatermarkGenerator
from tqdm import tqdm
from util.classes import Generation
from typing import List

def unigram_generate(model_name:str, fraction:float = 0.5, strength: float = 2.0, watermark_key: int = 0, 
                 prompts = List[str],
                 output_dir:str = '',
                 max_new_tokens: int = 512,
                 num_test: int = 500,
                 beam_size: int = None,
                 top_k: int = None,
                 top_p: float = 0.9) -> List[Generation]:
    
    tokenizer = LlamaTokenizer.from_pretrained(model_name, torch_dtype = torch.float16)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map = "auto")
    model = model.to("cuda")
    output_dir = output_dir + "/watermark/Unigram2/generations_emoji.json"
    model.eval()

    watermark_processor = LogitsProcessorList([UnigramWatermarkGenerator(
        fraction = fraction,
        strength = strength,
        vocab_size = model.config.vocab_size,
        watermark_key = watermark_key
    )])

    
    num_cur_outputs = 0
    output_generations = []
    format_prompt = "In addition, please add two 😃 emojjis after each output word."
    for idx, cur_data in tqdm(enumerate(prompts),total = min(len(prompts), num_test), desc="Processing Unigram watermark:"):
        if idx < num_cur_outputs or len(output_generations) >= num_test:
            continue
        tokenizer.pad_token = tokenizer.eos_token
        cur_data = cur_data + format_prompt
        batch_data = tokenizer(cur_data, truncation = True, return_tensors = 'pt', max_length = 512, padding = True)
        num_tokens = len(batch_data["input_ids"][0])
        batch_data = {k:v.to('cuda') for k,v in batch_data.items()}
        with torch.inference_mode():
            generate_args = {
                **batch_data,
                'logits_processor': watermark_processor,
                'output_scores': True,
                'return_dict_in_generate': True,
                'max_new_tokens': max_new_tokens,
            }
            generate_args['do_sample'] = True
            generate_args['top_k'] = top_k
            generate_args['top_p'] = top_p
            
            generation = model.generate(**generate_args)
            le = len(cur_data)
            gen_text = tokenizer.batch_decode(generation['sequences'][:, num_tokens:], skip_special_tokens = True)
            output_text = gen_text[0].replace('😃','')
            temp_generation = Generation(prompt=str(cur_data), response= output_text, id = int(idx), watermark_name='Unigram',attack="Emoji")
            output_generations.append(temp_generation)
    
    Generation.tofile(output_dir,output_generations)
    
    
def unigram_detect(model_name:str, fraction:float = 0.5, strength: float = 2.0, watermark_key: int = 0, 
                 intput_dir:str = '',
                 max_new_tokens: int = 512,
                 test_min_tokens = 512,
                 num_test: int = 500,
                 beam_size: int = None,
                 top_k: int = None,
                 top_p: float = 0.9,
                 threshold: float = 8.0,
                 attack_name: str = '',
                 watermark_name: str = ''):
    
    generations = Generation.fromfile(intput_dir)

    tokenizer = LlamaTokenizer.from_pretrained(model_name, torch_dtype = torch.float16)
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size
    
    detector = UnigramWatermarkDetector(fraction = fraction,
                                        strength = strength,
                                        vocab_size = vocab_size,
                                        watermark_key = watermark_key)
    
    z_socre_list = []
    for idx,cur_data in tqdm(enumerate(generations), total=len(generations)):
        batch_data = cur_data.response
        gen_tokens = tokenizer(batch_data, add_special_tokens = False)["input_ids"]
        
        if len(gen_tokens) >= test_min_tokens:
            z_socre_list.append(detector.detect(gen_tokens))
        else:
            print(f"Waring: sequence{idx} is too short")

        num = 0
        for z in z_socre_list:
            if z >= threshold:
                num += 1
    watermark_percent = num / len(generations) 
    save_dict = {
        'z_score': z_socre_list,
        'watermark_percent': [1 if z > threshold else 0 for z in z_socre_list]
    }

    output_dir = f"/data1/lzs/MarkText/watermark/Unigram/detect/{attack_name}_detect.json"
    with open(output_dir, 'w') as f:
        json.dump(save_dict, f, indent = 4)

    dict_data = {
            "name": f"{watermark_name}_{attack_name}",
            "watermark_percent": watermark_percent
        }
    return dict_data