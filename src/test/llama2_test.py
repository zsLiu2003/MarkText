from transformers import AutoTokenizer
import transformers
import torch

model = "/data2/huggingface-mirror/dataroot/models/meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)
content = 'Please only paraphase the following content without other contents and do not modify its meaning: They have never been known to mingle with humans. Today, it is believed these unicorns live in an unspoilt environment which is surrounded by mountains. Its edge is protected by a thick wattle of wattle trees, giving it a majestic appearance. Along with their so-called miracle of multicolored coat, their golden coloured feather makes them look like mirages. Some of them are rumored to be capable of speaking a large amount of different languages. They feed on elk and goats as they were selected from those animals that possess a fierceness to them, and can \"eat\" them with their long horns.' + "<s>[INST] Please paraphase the above contents you answered without modify its meaning[/INST]"

sequences = pipeline(
    content,    
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=512,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")
    
