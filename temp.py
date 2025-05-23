import torch
import sys
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)

from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

def train_LoRA(model_name,dataset_name,sft_args,lora_args):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map = "auto")
    model.config.use_cache = False
    training_dataset = load_dataset(dataset_name,split='train')
    lora_model = get_peft_model(model,lora_args)
    fine_tuning_with_lora = SFTTrainer(
        model=lora_model,
        train_dataset=training_dataset,
        peft_config=lora_args,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=sft_args
    )
    fine_tuning_with_lora.train()
    tokenizer.save_pretrained("/data1/lzs/MarkText/fine-tuning/WizardLM-LoRA2")
    fine_tuning_with_lora.save_model("/data1/lzs/MarkText/fine-tuning/WizardLM-LoRA2")


def train_full(model_name, dataset_name, sft_args):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map = "auto")
    model.config.use_cache = False
    training_dataset = load_dataset(dataset_name,split='train')
    fine_tuning_full = SFTTrainer(
        model=model,
        train_dataset=training_dataset,
        dataset_text_field='text',
        tokenizer=tokenizer,
        args=sft_args
    )
    fine_tuning_full.train()
    tokenizer.save_pretrained("/data1/lzs/MarkText/fine-tuning/Llama-2-chat-full")
    fine_tuning_full.model.save_pretrained("/data1/lzs/MarkText/fine-tuning/Llama-2-chat-full")

if __name__ == "__main__":
    model = "/data2/model/Quantize/llama2-chat_normal"
    dataset_name = "mlabonne/guanaco-llama2-1k"
    sft_train_params = TrainingArguments(
        output_dir="/data1/lzs/MarkText/fine-tuning/checkpoints/llama2-chat_full",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=50,
        logging_steps=50,
        learning_rate=2e-5,
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="tensorboard"
    )

    lora_train_params = LoraConfig(
        lora_alpha=8,
        lora_dropout=0.1,
        r=8,
        bias="none",
        task_type="CAUSAL_LM"
    )
    #train_LoRA(model,dataset_name,sft_train_params,lora_train_params)
    train_full(model,dataset_name,sft_train_params)

