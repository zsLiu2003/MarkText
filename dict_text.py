import json
from util.classes import Generation
output_path = "/data1/lzs/MarkText/generations.json"
import ast
import re
from typing import Dict, List


def to_generation(dict: Dict) -> 'Generation':
        g = Generation()
        for key, value in dict.items():
            setattr(g, key, value)
        return g

def fromfile(filename: str) -> List['Generation']:
    with open(filename, "r") as infile:
        dict_list = json.load(infile)
        output_generations = []
        for dict in dict_list:
            output_generations.append(Generation.to_generation(dict))
        return output_generations




output = fromfile(output_path)
print(output[0].prompt)


