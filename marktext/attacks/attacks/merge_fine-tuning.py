from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)
from peft import PeftModel

base_model1_name = '/data2/model/Quantize/WizardLM_normal'
base_model2_name = '/data2/model/Quantize/llama2-chat_normal'
lora_model1_name = '/data1/lzs/MarkText/fine-tuning/WizardLM-LoRA'
lora_model2_name = '/data1/lzs/MarkText/fine-tuning/Llama-2-LoRA'

tokenizer = AutoTokenizer.from_pretrained(base_model1_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
tokenizer.save_pretrained('/data1/lzs/MarkText/fine-tuning/WizardLM-LoRA2')
tokenizer2 = AutoTokenizer.from_pretrained(base_model2_name)
tokenizer2.pad_token = tokenizer2.eos_token
tokenizer2.padding_side = "right"
tokenizer2.save_pretrained('/data1/lzs/MarkText/fine-tuning/Llama-2-LoRA2')