import torch
import torch.utils
import torch.utils.data
from transformers import LlamaTokenizer,LlamaForCausalLM,AutoTokenizer,AutoModelForCausalLM, BitsAndBytesConfig
from transformers import TrainingArguments, get_scheduler,AdamW
from trl import SFTTrainer
from datasets import load_dataset
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader,Dataset
from tqdm import tqdm
import json

import transformers
from datetime import datetime
from util.classes import Generation
from accelerate import accelerator
from trl import SFTTrainer

class myDataset(Dataset):

    def __init__(self, input_dir,model_name):
        self.input_dir = input_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        with open(self.input_dir, 'r') as input_file:
            self.dataset = json.load(input_file)
        self.prompt_list = self.dataset["prompt_list"]
        self.response_list = self.dataset["response_list"]
    def __len__(self):
        return len(self.prompt_list)
    
    def __getitem__(self, index):

        prompt = (f"{{instruction}}\n---\nAnswer:")
        temp_instruction = prompt.format(instruction = self.prompt_list[index])
        output = self.response_list[index]
        instruction = self.tokenizer.encode(self.tokenizer.bos_token + temp_instruction, add_special_tokens=False, padding=True, truncation=True)
        answer = self.tokenizer.encode(output + self.tokenizer.eos_token, add_special_tokens=False, padding=True, truncation=True)
        return {
            "input_ids":instruction + answer,
            "attention_mask": [1] * (len(instruction) + len(answer)),
            "labels": [-100] * len(instruction) + answer,
        }
    


class Extraction():

    def __init__(self):
        self.attack_name = "Extraction"

    
    def data_extraction(model_name: str,dataset_name: str):
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map = 'auto')
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        dataset = load_dataset(dataset_name,split='train[:5000]')
        dataset_dict = []
        response_dict = []
        for cur_data in tqdm(dataset):
            temp = cur_data["text"]
            temp = (temp.replace("\000", "")
                .replace("\r", "__LINE__")
                .replace("\t", "__TAB__")
                .replace("\n", "__LINE__")
                )
            input_data = f"{temp}[/INST]"
            dataset_dict.append(input_data)
            input = tokenizer.encode(
                input_data,
                max_length=512,
                truncation=True,
                padding=True,
                return_tensors='pt',
                add_special_tokens=True,
            )
            outputs = model.generate(
                input_ids = input.to('cuda'),
                max_length = 512,
            )
            output_text = tokenizer.batch_decode(outputs, skip_special_tokens= True)
            index = output_text[0].find("[/INST]")
            generate_text = output_text[0][index + len("[/INST]"):]
            response_dict.append(generate_text)
            
        output_dict = {
            "prompt_list": dataset_dict,
            "response_list": response_dict,
        }
        
        output_path = "/data1/lzs/MarkText/Attack/extraction_attack/Llama-2-chat_dictdata.json"
        with open(output_path, 'w') as outputfile:
            json.dump(output_dict, outputfile, indent=4)
            
        return output_path


    def train_new_model(
            new_model_name: str = "/data2/huggingface-mirror/dataroot/models/meta-llama/Llama-2-7b-hf",
            input_dictpath: str = ''
    ):
            
        dataset = load_dataset('json', data_files=input_dictpath, split='train')
        model_name = new_model_name
        num_epochs = 3
        def format_func(example):
            result = {}
            text = f"### Question: {example['instruction']}\n ### Answer: {example['output']}"
            result["text"] = text
            return result
        train_dataset = dataset.map(format_func)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side = "right", add_eos_token = True, add_bos_token = True)
        tokenizer.pad_token = tokenizer.eos_token
        def generate_and_tokenize_prompt(prompt):
            result = tokenizer(
                format_func(prompt),
                truncation=True,
                max_length=512,
                padding="max_length",
            )
            result["labels"] = result["input_ids"].copy()
            return result
        
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map = "auto")
        #tokenized_train_dataset = dataset.map(generate_and_tokenize_prompt)

        lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "lm_head",
            ],
            bias="none",
            lora_dropout=0.05,  # Conventional
            task_type="CAUSAL_LM",
        )
        model.is_parallelizable = True
        model.model_parallel = True
        base_model_name = "Llama-2-chat-7B"
        project = "Train_fine-tuning"
        run_name = base_model_name + "-" + project
        trainer = SFTTrainer(
            model=model,
            max_seq_length=512,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            dataset_text_field="text",
            args=transformers.TrainingArguments(
                output_dir="/data1/lzs/MarkText/Attack/extraction_attack/checkpoint/Llama-2-chat",
                num_train_epochs=4,
                per_device_train_batch_size=4,
                gradient_accumulation_steps=1,
                learning_rate=2e-5, # Want a small lr for finetuning
                fp16=False,
                bf16=False,
                optim="paged_adamw_32bit",
                max_grad_norm=0.3,
                max_steps=-1,
                warmup_ratio=0.03,
                group_by_length=True,
                lr_scheduler_type="constant",
                report_to="tensorboard",
                logging_dir="./logs",        
                save_strategy="no",                                       
                run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"     
            ),
        )
        model.config.use_cache = False
        trainer.train()
        tokenizer.save_pretrained("/data1/lzs/MarkText/extraction/Llama-2-chat")
        trainer.model.save_pretrained("/data1/lzs/MarkText/extraction/Llama-2-chat")



if __name__ == "__main__":
    #output_path = Extraction.data_extraction(str("/data2/model/Quantize/Llama-2-chat_normal"),str("togethercomputer/llama-instruct"))
    Extraction.train_new_model(str("/data2/huggingface-mirror/dataroot/models/meta-llama/Llama-2-7b-hf"), str("/data1/lzs/MarkText/Attack/extraction_attack/Llama-2-chat_dictdata.jsonl"))
    
    