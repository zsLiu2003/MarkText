file_path = "/data1/lzs/MarkText/Attack/extraction_attack/Llama-2-chat_dictdata2.json"
import json


with open(file_path, 'r') as input_file:
    dataset = json.load(input_file)
    prompt_list = dataset["prompt_list"]
    response_list = dataset["response_list"]

dic_list = []
for idx, item in enumerate(prompt_list):
    
    dic = {
        "instruction": item,
        "output": response_list[idx],
    }
    dic_list.append(dic)

with open("/data1/lzs/MarkText/Attack/extraction_attack/Llama-2-chat_dictdata.jsonl", 'w') as outputfile:
    for dic in dic_list:
        json_str = json.dumps(dic)
        outputfile.write(json_str + '\n')

from datasets import load_dataset

dataset = load_dataset('json',data_files="/data1/lzs/MarkText/Attack/extraction_attack/Llama-2-chat_dictdata.jsonl", split='train')