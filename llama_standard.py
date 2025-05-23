import yaml


def standard(model, sys, user):
    '''Return a standard version if the prompt of the given model(llama-2)'''

    if "llama" in model:
        if sys:
            return f"[INST] <<SYS>> {sys} <</SYS>> {user} [/INST]"
        else:
            return f"[INST] {user} [/INST]"
        
    else:
        breakpoint()
        raise NotImplementedError(f"No known standardization for model {model}.\
                                  Please add it manually to llama-standard.py")
    

def load_config(config_file):
    with open(config_file, "r") as infile:
        config = yaml.safe_load(infile)