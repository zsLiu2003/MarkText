CUDA_VISIBLE_DEVICES=3 HF_ENDPOINT=https://hf-mirror.com \
python /home/lsz/MarkText/attack_KGW.py KGW /data1/lzs/MarkText/watermark/KGW/generations.tsv /home/lsz/MarkText/config.yml

CUDA_VISIBLE_DEVICES=3 HF_ENDPOINT=https://hf-mirror.com \
python /home/lsz/MarkText/attack_KGW.py Convert /data1/lzs/MarkText/watermark/Convert/generations.tsv /home/lsz/MarkText/config.yml

CUDA_VISIBLE_DEVICES=3 HF_ENDPOINT=https://hf-mirror.com \
python /home/lsz/MarkText/attack_KGW.py Inverse /data1/lzs/MarkText/watermark/Inverse/generations.tsv /home/lsz/MarkText/config.yml

CUDA_VISIBLE_DEVICES=3 HF_ENDPOINT=https://hf-mirror.com \
python /home/lsz/MarkText/attack_KGW.py Exponential /data1/lzs/MarkText/watermark/Exponential/generations.tsv /home/lsz/MarkText/config.yml

