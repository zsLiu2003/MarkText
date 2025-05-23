from transformers import AutoModelForCausalLM,AutoTokenizer
from transformers.generation.logits_process import (
    LogitsProcessor,
    LogitsProcessorList
)
from typing import Optional, List, Dict,Any
from MarkText.util.classes import WatermarkConfig,ConfigSpec,Generation
from MarkText.origin_server import Server
import torch

class HFServer(Server,LogitsProcessor):
    """
    run the HuggingFace API model based on origin_server
    """
    def __init__(self, config:Dict[str,Any], **kwargs) -> None:
        """
        initize the Huggingface Server
        """
        model_name = config.model
        self.server = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token        

        self.devices = [[i for i in range(torch.cuda.device_count())][0]]
        self.server = self.server.to(self.devices[0])
        self.watermark_generator = None
        self.batch_size = config.batch_size
        self.current_batch = 0
        self.current_offset = 0
        
    def install(self, watermark_generator) -> None:
        """
        Install a watermark generate methods
        """
        self.watermark_generator = watermark_generator
        
    
