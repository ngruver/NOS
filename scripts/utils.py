import os
import pandas as pd
import torch
from collections.abc import MutableMapping

def flatten_config(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_config(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def convert_to_dict(obj):
    try:
        return {k: convert_to_dict(v) for k, v in dict(obj).items()}
    except Exception:
        return obj
