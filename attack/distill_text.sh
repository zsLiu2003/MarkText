CUDA_VISIBLE_DEVICES=4 HF_ENDPOINT=https://hf-mirror.com \
python /home/lsz/MarkText/generate_text_with_distill.py /home/lsz/MarkText/config.yml /data1/lzs/MarkText KGW /data1/lzs/MarkText/distill/kgw1/llama-2-7b-logit-watermark-distill-kgw-k0-gamma0.25-delta1

CUDA_VISIBLE_DEVICES=4 HF_ENDPOINT=https://hf-mirror.com \
python /home/lsz/MarkText/generate_text_with_distill.py /home/lsz/MarkText/config.yml /data1/lzs/MarkText Inverse /data1/lzs/MarkText/distill/kth-shift4/llama-2-7b-logit-watermark-distill-kth-shift4

CUDA_VISIBLE_DEVICES=4 HF_ENDPOINT=https://hf-mirror.com \
python /home/lsz/MarkText/generate_text_with_distill.py /home/lsz/MarkText/config.yml /data1/lzs/MarkText Exponential /data1/lzs/MarkText/distill/aar_k2/llama-2-7b-logit-watermark-distill-aar-k2
