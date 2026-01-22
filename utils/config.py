import yaml
import os
from types import SimpleNamespace

def load_config(config_path="config.yaml"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Cannot find config file in: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    def dict_to_obj(d):
        if isinstance(d, dict):
            for k, v in d.items():
                d[k] = dict_to_obj(v)
            return SimpleNamespace(**d)
        else:
            return d

    print(f"Loaded configuration from {config_path}")
    return dict_to_obj(config_dict)