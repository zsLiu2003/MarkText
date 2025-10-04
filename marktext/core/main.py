from generate_text import generate_watermarktext,load_file
from util.classes import ConfigSpec,WatermarkConfig,Generation
import torch
import os
import sys

if __name__ == "__main__":
    config_path = str(sys.argv[1])
    watermark_path = generate_watermarktext(config_path,generate_text_path='/data1/lzs/MarkText/generations.json')
    