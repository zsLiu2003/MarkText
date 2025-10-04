import ast
import math
import re
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Set, Tuple
import json
import dacite

def str_or_none(val):
    if val is not None:
        return str(val)
    else:
        return "-"

def parse_string_val(val):
    if val == "-":
        return None
    return val

def str_to_dict(val):
    return ast.literal_eval(val)


@dataclass(frozen=True)
class VerifierSpec:
    """
    A class representing the verifier specification.

    Attributes:
        verifier (str): The verifier to use. Defaults to 'Theoretical'.
        empirical_method (str): The empirical method to use. Defaults to 'regular'.
        log (Optional[bool]): Whether to use a log score
        gamma (Optional[float]): The gamma value to use for edit distance. Defaults to 0.
    """

    verifier: str = "Theoretical"
    empirical_method: str = "regular"
    log: Optional[bool] = None
    gamma: Optional[float] = 0

    def __str__(self) -> str:
        return str(self.__dict__)

    def __repr__(self) -> str:
        return self.__str__()

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'VerifierSpec':
        return dacite.from_dict(VerifierSpec, d)

    @staticmethod
    def from_str(s: str) -> 'VerifierSpec':
        return VerifierSpec.from_dict(str_to_dict(s))

@dataclass(frozen=True)
class ConfigSpec:

    model_name: str = ''
    watermark_name: str = ''
    watermark_path: str = ''
    output_path: str = ''
    prompt_path: str = ''
    #trainin setting
    batch_size: int = 8
    temperature: float = 1.0
    devices: Optional[List[int]] = None
    misspellings: str = "MarkMyWords/run/static_data/misspellings.json"
    def get_devices(self):
        import torch
        if self.devices is not None:
            return self.devices
        elif not torch.cuda.is_available():
            return["cpu"]
        else:
            return list(range(torch.cuda.device_count()))
    def from_dict(d:Dict[str, Any]) -> "ConfigSpec":
        return dacite.from_dict(ConfigSpec,d)
    

class VerifyConfig:
    log: Optional[bool] = None
    gamma: Optional[float] = 0
    def __str__(self) -> str:
        return str(self.__dict__)

    def __repr__(self) -> str:
        return self.__str__()

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'VerifyConfig':
        return dacite.from_dict(VerifyConfig, d)

    @staticmethod
    def from_str(s: str) -> 'VerifyConfig':
        return VerifyConfig.from_dict(str_to_dict(s))
    
@dataclass(frozen=False)
class WatermarkConfig:
    name: str = ''
    tokenizer: str = ''
    temperature: float = 1.0
    @staticmethod
    def from_dict(d:Dict[str,Any]) -> 'WatermarkConfig':
        return dacite.from_dict(WatermarkConfig,d)    
    
    @staticmethod
    def from_str(s: str) -> "WatermarkConfig":
        return WatermarkConfig.from_dict(str_to_dict(s))

@dataclass(frozen=False)
class Generation:
    watermark: Optional[WatermarkConfig] = None
    watermark_name: str = ''
    key: Optional[int] = None
    attack: Optional[str] = None
    id: int = 0
    prompt: str = ''
    response: str = ''
    quality: Optional[float] = None
    token_count: int = 0
    temperature: float = 1.0
    def to_dict(prompt,response, id):
        generation = Generation()
        generation.prompt = str(
            prompt
            .replace("\000", "")
            .replace("\r", "__LINE__")
            .replace("\t", "__TAB__")
            .replace("\n", "__LINE__")
        )
        generation.response = str(
            response
            .replace("\000", "")
            .replace("\r", "__LINE__")
            .replace("\t", "__TAB__")
            .replace("\n", "__LINE__")
        )
        generation.id = id


    def keys()-> List[str]:
        return[
            "watermark",
            "watermark_name",
            "key",
            "attack",
            "id",
            "prompt",
            "response",
            "quality",
            "token_count",
            "temperature",
        ]
    def to_str(s:'Generation') -> 'Generation':
        s.prompt = (
            s.prompt
            .replace("\000", "")
            .replace("\r", "__LINE__")
            .replace("\t", "__TAB__")
            .replace("\n", "__LINE__")
        )
        s.response = (
            s.response
            .replace("\000", "")
            .replace("\r", "__LINE__")
            .replace("\t", "__TAB__")
            .replace("\n", "__LINE__")
        )
        return s
    
    def to_dict(g : 'Generation') -> Dict: 
        keys = Generation.keys()
        dicg = {}
        for key in keys:
            keydata = getattr(g, key)
            dicg[key] = keydata
        return dicg

    def tofile(filename, generations: Optional[List["Generation"]] = None) -> None:
        with open(str(filename), "w") as outputfile:
            if generations is not None and len(generations):
                dic_list = []
                for g in generations:
                    g = Generation.to_str(g)
                    dic = Generation.to_dict(g)
                    dic_list.append(dic)
                json.dump(dic_list,outputfile, indent=4)

    def to_generation(dict: Dict) -> 'Generation':
        g = Generation()
        for key, value in dict.items():
            if key == 'response' or key == 'prompt':
                value = (re.sub(r"(___LINE___)+$", "__LINE__", value)
                    .replace("__LINE__", "\n")
                    .replace("__TAB__", "\t")
                )
            setattr(g, key, value)
        return g
    def fromfile(filename: str) -> List['Generation']:
        with open(filename, "r") as infile:
            dict_list = json.load(infile)
            output_generations = []
            for dict in dict_list:
                output_generations.append(Generation.to_generation(dict))
            return output_generations
                
    