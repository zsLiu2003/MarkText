import datasets
from transformers import AutoTokenizer,AutoModelForCausalLM
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import logging
from transformers.trainer import TrainingArguments
from transformers.trainer import Trainer
def tokenizer_function(dataset,tokenizer):
    if "question" in dataset and "answer" in dataset:
        text = dataset["question"][0] + dataset["answer"][0]
    elif "input" in dataset and "output" in dataset:
        text = dataset["input"][0] + dataset["output"][0]
    else:
        text = dataset["text"][0]
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer_inputs = tokenizer(
        text,
        return_tensors = "np",
        padding=True
    )

    max_length = min(tokenizer_inputs["input_ids"].shape[1], 2048)
    tokenizer.truncate_side = "left"
    tokenizer_inputs = tokenizer(
        text,
        return_tensors='np',
        truncation=True,
        max_length=max_length
    )
    return tokenizer_inputs


def dataset_split(config,tokenizer):

    finetuning_dataset = datasets.load_dataset('lamini/lamini_docs')
    tokenize_dataset = finetuning_dataset.map(
        tokenizer_function,
        batched=True,
        batch_size=4,
        drop_last_batch=True
    )        
    tokenize_dataset = tokenize_dataset.add_colume("lables", tokenize_dataset["input_ids"])
    split_dataset = tokenize_dataset.train_test_split(test_size = 0.2, shuffle = True,seed = 123)
    return split_dataset['train'], split_dataset['test']
#the first dataset

def inference(text, model, tokenizer, max_input_tokens = 1024, max_output_tokens = 100):
    input_ids = tokenizer.encode(
        text,
        return_tensors = "pt",
        truncation = True,
        max_length = max_input_tokens
    )
    device = model.device
    generate_tokens_with_prompt = model.generate(
        input_ids = input_ids.to(device),
        max_length = max_output_tokens
    )
    generate_text_with_prompt = tokenizer.batch_decode(generate_tokens_with_prompt,skip_special_tokens=True)
    generate_text = generate_text_with_prompt[0][len(text):]

    return generate_text

use_hf = True
dataset_path = "lamini/lamini_docs"
model_name = "EleutherAI/pythia-70m"
train_config = {
    "model":{
        "pretrained_name":model_name,
        "max_length": 2048
    },
    "datasets":{
        "use_hf": use_hf,
        "path": dataset_path
    },
    "verbose": True
}
base_model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
logger = logging.getLogger('mylogger')
device_count = torch.cuda.device_count()
if device_count > 0:
    logger.debug("Select GPU DEVICE")
    device = torch.device("cuda")
else:
    logger.debug("Selcet CPU DEVICE")
    device = torch.device("cpu")
split_dataset = datasets.load_dataset("lamini/lamini_docs")
split_dataset_dict = split_dataset['train']
train_dataset = split_dataset['train']
test_dataset = split_dataset['test']
split_dataset = split_dataset_dict.train_test_split(test_size=0.2)

test_text = test_dataset[0]['question']

max_steps = 3
train_model_name = f"lamini_docs_{max_steps}_steps"
output_dir = '/data1/lzs/DeeplWork'

train_args = TrainingArguments(
    #learning_rate
    learning_rate=1e-5,
    num_train_epochs=1,
    max_steps=max_steps,

    per_device_train_batch_size=1,

    output_dir=output_dir,

    overwrite_output_dir=False,
    disable_tqdm=False,
    eval_steps=120,
    save_steps=120,
    warmup_steps=1,
    per_device_eval_batch_size=1,
    evaluation_strategy='steps',
    logging_strategy='steps',
    logging_steps=1,
    optim='adafactor',
    gradient_accumulation_steps=4,
    save_total_limit=1,
    metric_for_best_model='eval_loss',
    greater_is_better=False
)

model_flops = (
    base_model.floating_point_ops(
    {
        "input_ids": torch.zeros(
            1,train_config["model"]["max_length"]
        )
    }
    )   
    * train_args.gradient_accumulation_steps
)

print(base_model)
print("Memory footprint", base_model.get_memory_footprint() / 1e9, "GB")
print("Flops", model_flops / 1e9, "GFLOPS")

trainer = Trainer(
    model=base_model,
    #model_flops=model_flops,
    #max_steps=max_steps,
    args=train_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

train_output = trainer.train()
trainer.save_model(output_dir)
device_count = torch.cuda.device_count()
fine_tunning_model = AutoModelForCausalLM.from_pretrained("lamini/lamini_docs_finetuned")
fine_tunning_tokenizer = AutoTokenizer.from_pretrained("lamini/lamini_docs_finetuned")
fine_tunning_model.to(device)
predict_text = inference(test_text,fine_tunning_model,fine_tunning_tokenizer)
base_text = inference(test_text,base_model,tokenizer)
answer = test_dataset[0]['answer']
print('Question:')
print(test_text)
print("--------------------------------------")
print("Answer")
print(answer)
print("--------------------------------------")
print("Without finetune: ")
print(base_text)
print("--------------------------------------")
print("With finetune: ")
print(predict_text)
