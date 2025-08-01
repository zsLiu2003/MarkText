CUDA_VISIBLE_DEVICES=5 HF_ENDPOINT=https://hf-mirror.com \
python /home/lsz/MarkText/quality.py /home/lsz/MarkText/config.yml quantize full /data1/lzs/MarkText/watermark/quantize/llama2-chat_normal_test_full.jsonl llama-2-chat

CUDA_VISIBLE_DEVICES=5 HF_ENDPOINT=https://hf-mirror.com \
python /home/lsz/MarkText/quality.py /home/lsz/MarkText/config.yml quantize LoRA /data1/lzs/MarkText/watermark/quantize/llama2-chat_normal_test_LoRA.jsonl llama-2-chat

CUDA_VISIBLE_DEVICES=5 HF_ENDPOINT=https://hf-mirror.com \
python /home/lsz/MarkText/quality.py /home/lsz/MarkText/config.yml quantize Extraction /data1/lzs/MarkText/watermark/quantize/llama2-chat_normal_test.jsonl llama-2-chat

CUDA_VISIBLE_DEVICES=5 HF_ENDPOINT=https://hf-mirror.com \
python /home/lsz/MarkText/quality.py /home/lsz/MarkText/config.yml quantize full /data1/lzs/MarkText/watermark/quantize/WizardLM_normal_test_full.jsonl WizardLM

CUDA_VISIBLE_DEVICES=5 HF_ENDPOINT=https://hf-mirror.com \
python /home/lsz/MarkText/quality.py /home/lsz/MarkText/config.yml quantize LoRA /data1/lzs/MarkText/watermark/quantize/WizardLM_normal_test_LoRA.jsonl WizardLM

CUDA_VISIBLE_DEVICES=5 HF_ENDPOINT=https://hf-mirror.com \
python /home/lsz/MarkText/quality.py /home/lsz/MarkText/config.yml quantize Extraction /data1/lzs/MarkText/watermark/quantize/WizardLM_normal_test.jsonl WizardLM
