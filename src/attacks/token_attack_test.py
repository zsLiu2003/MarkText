# Modified from https://github.com/jwkirchenbauer/lm-watermarking?tab=readme-ov-file


from kgw.KGW_base import WatermarkDetector, WatermarkLogitsProcessor
from transformers import AutoTokenizer,AutoModelForCausalLM,LogitsProcessorList
from typing import List,Dict
from tqdm import tqdm
import json

from util.classes import Generation,WatermarkConfig,ConfigSpec
from token_attack.token_attack import require_attack,delete_token,insert_token,swap_token


class KGW():

    def __init__(self, model_name:str):
        self.model_name = model_name
        self.watermark_name = "KGW"
    def attack(self,tokens, delete_p: float, insert_p: float, swap_p: float, vocab_size: int):
        if delete_p != 0:
            tokens = delete_token(tokens=tokens, p=delete_p)
        if insert_p != 0:
            tokens = insert_token(tokens=tokens,p=insert_p, vocab_size=vocab_size)
        if swap_p != 0:
            tokens = swap_token(tokens=tokens,p=swap_p, vocab_size= vocab_size)

        return tokens
    
    def injection(self,prompts: List[str],config: ConfigSpec):
        
        model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map = "auto")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenizer.pad_token = tokenizer.eos_token
        watermark_tokens = []
        watermark_processor = WatermarkLogitsProcessor(vocab = list(tokenizer.get_vocab().values()),
                                                       gamma = 0.25,
                                                       delta = 2.0,
                                                       seeding_scheme = "selfhash")
        output_genertaions = []
        for idx,input_text in tqdm(enumerate(prompts),desc="Processing KGW watermark---"):
            inputs = tokenizer(
                input_text,
                max_length=512,
                return_tensors='pt'
            )
            output_tokens = model.generate(**inputs,
                                           logits_processor = LogitsProcessorList([watermark_processor]))
            output_tokens = output_tokens[:,inputs["input_ids"].shape[-1]:]
            watermark_tokens.append(output_tokens)
            output_text = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]
            generation = Generation(id=idx,watermark_name="KGW",prompt=input_text,response=output_text)
            output_genertaions.append(generation)
        output_path = f"{config.output_path}/watermark/KGW/genertaions.json"
        Generation.tofile(output_path,output_genertaions)
        

        attack_list = require_attack()
        watermark_result = []
        for attack in tqdm(attack_list, desc="Processing token attack to KGW----------"):
            attack_result = []
            attack_name = attack["attack_name"]
            delete_p = attack["delete_p"]
            insert_p = attack["insert_p"]
            swap_p = attack["swap_p"]
            for idx, watermark_token in tqdm(enumerate(watermark_tokens), desc=f"Processing {attack_name}--------"):
                attack_token = self.attack(
                    tokens=watermark_token[0],
                    delete_p=delete_p,
                    insert_p=insert_p,
                    swap_p=swap_p,
                    vocab_size=len(tokenizer.get_vocab().values())
                )
                attack_token = attack_token.unsqueeze(0)
                attack_output = tokenizer.batch_decode(attack_token, skip_special_tokens=True)
                is_watermark = self.token_detection(input_text=attack_output[0],tokenizer=tokenizer,model=model)
                attack_result.append(is_watermark)
            dict_data = {
                "attack_name": attack_name,
                "is_watermark": attack_result
            }
            watermark_result.append(dict_data)
        
        output_path = f"{config.output_path}/watermark/{self.watermark_name}/quality/is_watermark_list.json"
    def token_detection(self, input_text: str, tokenizer, model) -> bool:
        watermark_detector = WatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
                                               gamma = 0.25,
                                               seeding_scheme = "selfhash",
                                               device=model.device,
                                               tokenizer=tokenizer,
                                               z_threshold=4.0,
                                               normalizers=[],
                                               ignore_repeated_ngrams=True)
        is_watermark = watermark_detector.detect(input_text)

        if is_watermark:
            return True
        else:
            return False


    def detection(self,input_path:str) -> float:
        generations = Generation.fromfile(input_path)
        model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenizer.pad_token = tokenizer.eos_token
        watermark_detector = WatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
                                               gamma = 0.25,
                                               seeding_scheme = "selfhash",
                                               device=model.device,
                                               tokenizer=tokenizer,
                                               z_threshold=4.0,
                                               normalizers=[],
                                               ignore_repeated_ngrams=True)
        
        watermark_num = 0
        le = len(generations)
        for generation in tqdm(generations, desc="Detect KGW watermark---"):
            input_text = generation.response
            is_watermark = watermark_detector.detect(input_text)
            if is_watermark:
                watermark_num += 1
        
        watermark_percent = watermark_num / le
        return watermark_percent
    
        

            

        
        


import sys
from generate_text import load_file,read_prompt
if __name__ == "__main__":
    config_path = str(sys.argv[1])
    config = load_file(config_path)
    prompts = read_prompt(config.prompt_path)
    instance = KGW(config.model_name)
    prompts = prompts[:1]
    instance.injection(
        config=config,
        prompts=prompts
    )
