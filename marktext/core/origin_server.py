from abc import ABC, abstractmethod
from typing import Optional,List,Dict,Any
from MarkText.util.classes import WatermarkConfig,Generation

class Server(ABC):
    """
    This is the abstract class of model server
    """ 
    @abstractmethod
    def install(self, watermark_engine) -> None:
        self.device = "cpu"
    
    @abstractmethod
    def run(self, \
            inputs: List[str],\
            config: Dict[str,Any],\
            temperature: float,\
            keys: Optional[int] = None,
            watermarks: Optional[WatermarkConfig] = None) -> List[Generation]:
        return []
    
    @abstractmethod
    def tokenizer(self):
        pass
    
    @abstractmethod
    def devices(self):
        return self.devices

    
    