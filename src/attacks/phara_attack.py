from transformers import T5ForConditionalGeneration,T5Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import argostranslate.package
import argostranslate.translate
import numpy as np
from nltk.tokenize import sent_tokenize
import torch
import math
from typing import List,Dict
from util.classes import Generation
from tqdm import tqdm
from util.classes import ConfigSpec
class DipperAttack():    

    def __init__(self,config) -> None:
        self.tokenizer_modelname = "/data2/huggingface-mirror/dataroot/models/google/t5-v1_1-xxl"
        self.model_name = "/data2/huggingface-mirror/dataroot/models/kalpeshk2011/dipper-paraphraser-xxl"
        self.device = "cuda"
    def get_param_list() -> str:
        name = "Para_dipper"
        return name
    
    def call_dipper(self, input_text,  model, tokenizer, lex_diversity=40, order_diversity=60, prefix="", sent_interval=3, **kwargs):
        """Paraphrase a text using the DIPPER model.

        Args:
            input_text (str): The text to paraphrase. Make sure to mark the sentence to be paraphrased between <sent> and </sent> blocks, keeping space on either side.
            lex_diversity (int): The lexical diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
            order_diversity (int): The order diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
            **kwargs: Additional keyword arguments like top_p, top_k, max_length.
        """
        assert lex_diversity in [0, 20, 40, 60, 80, 100], "Lexical diversity must be one of 0, 20, 40, 60, 80, 100."
        assert order_diversity in [0, 20, 40, 60, 80, 100], "Order diversity must be one of 0, 20, 40, 60, 80, 100."

        lex_code = int(100 - lex_diversity)
        order_code = int(100 - order_diversity)

        input_text = " ".join(input_text.split())
        sentences = sent_tokenize(input_text)
        prefix = " ".join(prefix.replace("\n", " ").split())
        output_text = ""

        for sent_idx in range(0, len(sentences), sent_interval):
            curr_sent_window = " ".join(sentences[sent_idx:sent_idx + sent_interval])
            final_input_text = f"lexical = {lex_code}, order = {order_code}"
            if prefix:
                final_input_text += f" {prefix}"
            final_input_text += f" <sent> {curr_sent_window} </sent>"

            final_input = tokenizer([final_input_text], return_tensors="pt")
            final_input = {k: v.cuda() for k, v in final_input.items()}

            with torch.inference_mode():
                outputs = model.generate(**final_input, **kwargs)
            outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            prefix += " " + outputs[0]
            output_text += " " + outputs[0]

        return output_text
    
    def attack(self,input_path,config : ConfigSpec, watermark_name: str, output_path: str = ''):
        tokenizer = T5Tokenizer.from_pretrained(self.tokenizer_modelname)
        model = T5ForConditionalGeneration.from_pretrained(self.model_name, device_map = "auto")
        model.eval()
        generations = Generation.fromfile(input_path)
        para_generations = []
        for generation in tqdm(generations[:148], desc="Processing Dipper attack---"):
            origin_text = str(generation.response)
            para_text = self.call_dipper(input_text=origin_text,model = model, tokenizer=tokenizer)
            generation.response = para_text
            generation.attack = "Para_attack"
            para_generations.append(generation)

        Generation.tofile(output_path,para_generations)
        
    
class Llama3Attack():
    def __init__(self):
        self.device = "cuda"
        self.model_name = "/data2/huggingface-mirror/dataroot/models/meta-llama/Meta-Llama-3-8B-Instruct"

    def get_param_list() -> str:
        name = "Para_Llama-3"
        return name

    def call_llama3(self,text,temperature, model,tokenizer):
        content = text + "\nPlease prarphase the above content without other contents:"
        prompt = [
            {"role":"user", "content":content}
        ]
        input_ids = tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            return_tensors="pt",
            max_length=1024,
        )
        
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
        outputs = model.generate(
            input_ids = input_ids.to(model.device),
            max_new_tokens=1024,
            eos_token_id=terminators,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
        )
        response = outputs[0][input_ids.shape[-1]:]
        outputs = tokenizer.decode(response, skip_special_tokens=True)
        return str(outputs)

    def attack(self,input_path, config : ConfigSpec, watermark_name, output_path: str = ''):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype = torch.bfloat16,
            device_map = "auto"
            )
        model.eval()
        generations = Generation.fromfile(str(input_path))
        para_generations = []
        for generation in tqdm(generations[:148], desc="Processing Para_Llama-3 Attack:"):
            origin_text = generation.response
            para_text = self.call_llama3(origin_text,generation.temperature, model=model, tokenizer=tokenizer)
            generation.attack = "Para_llama-3"
            generation.response = para_text
            para_generations.append(generation)
        Generation.tofile(output_path,para_generations)

class Llama2Attack():
    def __init__(self) -> None:
        self.device = "cuda"
        self.model_name = "/data2/huggingface-mirror/dataroot/models/meta-llama/Llama-2-7b-chat-hf"
        """
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype = torch.bfloat16,
            device_map = "auto"
            ).to(self.device)
        self.model.eval()
        """
    def get_param_list() -> str:
        name = "Para_Llama-2"
        return name
    
    def call_llama2(self, input_text, model, tokenizer) -> str:
        prompt = input_text + "<s>[INST] Please paraphase the above contents you answered without modify its meaning[/INST]"
        inputs = tokenizer.encode(
            prompt,
            max_length=512,
            return_tensors='pt',
            add_special_tokens=False,
            truncation=True,
        )
        device = "cuda"
        outputs = model.generate(
            input_ids = inputs.to(device),
            max_length = 1536,
        )
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=False)
        index = outputs[0].find("[/INST]")
        output = outputs[0][index + len('[/INST]'):]
        return str(output)
    
    def attack(self,input_path, config : ConfigSpec, watermark_name, output_path: str = ''):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map = "auto")
        generations = Generation.fromfile(input_path)
        output_generations = []
        for generation in tqdm(generations[:148], desc="Processing Para_Llama-2 Attack:"):
            input_text = str(generation.response)
            output_text = self.call_llama2(input_text,model=model, tokenizer=tokenizer)
            generation.attack = "Para_llama-2"
            generation.response = str(output_text)
            output_generations.append(generation)
        Generation.tofile(output_path,output_generations)


class TranslationAttack():
    def __init__(self):
        self.from_code = "en"
        self.to_code1 = "ru"
        self.to_code2 = "fr"
        self.to_code3 = 'de'
    def attack_russion(self,config,watermark_name:str, attack_name : str, generations, output_path: str):
        argostranslate.package.update_package_index()
        available_packages = argostranslate.package.get_available_packages()
        package_to_install = next(filter(lambda x: (x.from_code == self.from_code and x.to_code == self.to_code1), available_packages,))
        argostranslate.package.install_from_path(package_to_install.download())
        text_list = []
        outupt_generations = []
        for generation in tqdm(generations, desc="Translate English to France---"):
            text = generation.response
            translated_text = argostranslate.translate.translate(str(text), self.from_code, self.to_code1)
            text_list.append(translated_text)
        package_to_install2 = next(filter(lambda y: (y.from_code == self.to_code1 and y.to_code == self.from_code), available_packages))
        argostranslate.package.install_from_path(package_to_install2.download())
        for idx,generation in tqdm(enumerate(generations[:148]), desc="Translate France back to English----"):
            translated_text = text_list[idx]
            transback_text = argostranslate.translate.translate(str(translated_text), self.to_code1, self.from_code)
            generation.response = transback_text
            outupt_generations.append(generation)
        Generation.tofile(output_path,outupt_generations)
        
    def attack_france(self,config: ConfigSpec,watermark_name: str, attack_name: str, generations, output_path: str):
        argostranslate.package.update_package_index()
        available_packages = argostranslate.package.get_available_packages()
        package_to_install = next(filter(lambda x: (x.from_code == self.from_code and x.to_code == self.to_code2), available_packages,))
        argostranslate.package.install_from_path(package_to_install.download())
        text_list = []
        outupt_generations = []
        for generation in tqdm(generations[:148], desc="Translate English to France---"):
            text = generation.response
            translated_text = argostranslate.translate.translate(str(text), self.from_code, self.to_code2)
            text_list.append(translated_text)
        package_to_install2 = next(filter(lambda y: (y.from_code == self.to_code2 and y.to_code == self.from_code), available_packages))
        argostranslate.package.install_from_path(package_to_install2.download())
        for idx,generation in tqdm(enumerate(generations[:148]), desc="Translate France back to English----"):
            translated_text = text_list[idx]
            transback_text = argostranslate.translate.translate(str(translated_text), self.to_code2, self.from_code)
            generation.response = transback_text
            outupt_generations.append(generation)
        Generation.tofile(output_path,outupt_generations)
    
    
    def attack_german(self,config: ConfigSpec,watermark_name: str, attack_name: str, generations, output_path: str):
        argostranslate.package.update_package_index()
        available_packages = argostranslate.package.get_available_packages()
        package_to_install = next(filter(lambda x: (x.from_code == self.from_code and x.to_code == self.to_code3), available_packages,))
        argostranslate.package.install_from_path(package_to_install.download())
        text_list = []
        outupt_generations = []
        for generation in tqdm(generations, desc="Translate English to France---"):
            text = generation.response
            translated_text = argostranslate.translate.translate(str(text), self.from_code, self.to_code3)
            text_list.append(translated_text)
        package_to_install2 = next(filter(lambda y: (y.from_code == self.to_code3 and y.to_code == self.from_code), available_packages))
        argostranslate.package.install_from_path(package_to_install2.download())
        for idx,generation in tqdm(enumerate(generations[:148]), desc="Translate France back to English----"):
            translated_text = text_list[idx]
            transback_text = argostranslate.translate.translate(str(translated_text), self.to_code3, self.from_code)
            generation.response = transback_text
            outupt_generations.append(generation)
        Generation.tofile(output_path,outupt_generations)
    
    def attack(self,input_path, config: ConfigSpec, watermark_name:str, attack_name: str, output_path: str):
        generations = Generation.fromfile(input_path)
        if "france" in attack_name:
            self.attack_france(config=config,generations=generations, watermark_name=watermark_name, attack_name=attack_name, output_path = output_path)
        elif "russion" in attack_name:
            self.attack_russion(config=config,generations=generations, watermark_name= watermark_name, attack_name=attack_name, output_path=output_path)
        elif "german" in attack_name:
            self.attack_german(config=config,generations=generations, watermark_name= watermark_name, attack_name=attack_name, output_path=output_path)
