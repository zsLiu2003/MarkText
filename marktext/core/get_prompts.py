from dacite import from_dict
import yaml
from util.classes import ConfigSpec
from generate_prompt import raw_prompts
import sys
def load_file(config_file):
    with open(config_file, "r") as infile:
        config_dict = yaml.safe_load(infile)
        data_instance = from_dict(data_class=ConfigSpec, data = config_dict)
        return data_instance
   
def generate_prompt(config_file, watermarks = None):
    "This function is for Text Generation Process"
    
    from llama_standard import standard
    
    config_class= load_file(config_file)
    model_name = config_class.model_name
    prompts = [standard(model_name,s,p) for p,s in raw_prompts]
    file_path = config_class.output_path + "/prompts"
    with open(file_path,'w') as file:
        for item in prompts:
            file.write(f'{item}\n')
        file.close()
    return prompts

if __name__ == "__main__":
    config = str(sys.argv[1])
    generate_prompt(config)