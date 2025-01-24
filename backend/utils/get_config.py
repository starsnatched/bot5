import yaml
import os
from typing import Any

_config_data = None

def load_config() -> None:
    global _config_data
    if _config_data is None:
        config_path = os.path.join("./configs/config.yaml")
        with open(config_path, "r") as f:
            _config_data = yaml.safe_load(f)

def get_config(key: str) -> Any:
    load_config()
    keys = key.split('.')
    data = _config_data
    for k in keys:
        data = data.get(k, {})
    return data