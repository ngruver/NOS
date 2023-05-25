import os
import glob
import logging
import pickle
import functools
import multiprocessing
from pathlib import Path
from typing import *
import tqdm

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset, DataLoader

import transformers

AB_SEQUENCE_LENGTH = 300
SPECIAL_TOKENS = ["[AbHC]", "[AbLC]", "[Ag]"]
SPECIAL_TOKEN_INDICES = [0, 150, 299]
DEFAULT_INFILL_SEED = " ".join(
    ["[AbHC]"] + 149 * ["[MASK]"] + ["[AbLC]"] + 148 * ["[MASK]"] + ["[Ag]"] 
)

def pad_to_length(arr, max_length):
    pad_length = max_length - len(arr)
    if pad_length >= 0:
        return torch.cat([arr, torch.zeros(pad_length).to(arr)])
    else:
        return arr[:max_length]

def random_sequences(num_samples, lengths, vocab_file):
    t = transformers.BertTokenizerFast(
        vocab_file=vocab_file, 
        do_lower_case=False,
    )
    aa_letters = set(t.vocab.keys()) - set(t.special_tokens_map.keys())
    aa_letters = [x for x in aa_letters if '[' not in x]
    
    def random_seq(length):
        return "".join(np.random.choice(aa_letters, size=length, replace=True))

    return [random_seq(l) for l in lengths]

def reformat(seq):
    if len(seq.split(" ")) <= 1:
        h, l = seq.split("[AbLC]")
        h = h.replace("[AbHC]","")
        l = l.replace("[Ag]","")
        return "[AbHC] " + " ".join(h) + " [AbLC] " + " ".join(l) + " [Ag]"
    else:
        return seq

class LabeledDataset(Dataset):
    def __init__(
        self,
        config,
        split,
    ):
        super().__init__()

        self.dataset_name = os.path.basename(config.data_dir)
        data_fn = os.path.join(config.data_dir, split)
        df = pd.read_csv(data_fn)
        
        split_wo_ext = split.split(".")[0]
        self.cache_fname = os.path.join(config.data_dir, f"{split_wo_ext}_cached.pkl")
        if os.path.exists(self.cache_fname):
            with open(self.cache_fname, "rb") as source:
                self.inputs = pickle.load(source)
        else:
            tokenizer = transformers.BertTokenizerFast(
                vocab_file=config.vocab_file, 
                do_lower_case=False,
            )

            self.inputs = []
            for seq in tqdm.tqdm(df["full_seq"].values, total=len(df)):
                seq = reformat(seq)

                if config.use_alignment_tokens:
                    seq = tokenizer.convert_tokens_to_ids(seq.split(" "))      
                    seq = torch.Tensor(seq).int() 

                    if len(seq) != AB_SEQUENCE_LENGTH:
                        continue

                    corrupt_mask = torch.ones_like(seq)
                    corrupt_mask[SPECIAL_TOKEN_INDICES] = 0
                    attn_mask = torch.ones_like(seq)

                else:
                    seq = seq.replace("- ","")
                    seq = tokenizer.convert_tokens_to_ids(seq.split(" "))      
                    seq = torch.Tensor(seq).int() 

                    corrupt_mask = torch.ones_like(seq)
                    special_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
                    for special_id in special_ids:
                        corrupt_mask[seq == special_id] = 0
                    attn_mask = torch.ones_like(seq)

                    pad = torch.zeros(AB_SEQUENCE_LENGTH - len(seq), dtype=seq.dtype)
                    
                    seq = torch.cat([seq, pad])
                    corrupt_mask = torch.cat([corrupt_mask, pad])
                    attn_mask = torch.cat([attn_mask, pad])

                self.inputs.append({
                    "seq": seq,
                    "corrupt_mask": corrupt_mask,
                    "attn_mask": attn_mask,
                })

            with open(self.cache_fname, "wb") as source:
                pickle.dump(self.inputs, source)

        if config.target_cols is not None:
            label_values = StandardScaler().fit_transform(df[config.target_cols])
            
            self.inputs = [
                {
                    "seq": x["seq"],
                    "attn_mask": x["attn_mask"],
                    "corrupt_mask": x["corrupt_mask"],
                    "labels": torch.tensor(l, dtype=torch.float32),
                }
                for x, l in zip(self.inputs, label_values)
            ]

        self.inputs = self.inputs[:config.max_samples]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError("Index out of range")

        vals = self.inputs[index]

        retval = {
            "attn_mask": vals["attn_mask"],
            "corrupt_mask": vals["corrupt_mask"],
            "seq": vals["seq"].long(),
        }
        if "labels" in vals:
            retval["labels"] = vals["labels"]

        return retval

def get_loaders(config):
    dsets = [LabeledDataset(config, split) for split in [config.train_fn, config.val_fn]]

    effective_batch_size = config.batch_size
    if torch.cuda.is_available():
        effective_batch_size = int(config.batch_size / torch.cuda.device_count())

    loaders = [
        DataLoader(
            dataset=ds,
            batch_size=effective_batch_size,
            shuffle=(i == 0),
            num_workers=config.loader_workers,
            pin_memory=True,
        )
        for i, ds in enumerate(dsets)
    ]

    return loaders

class BasicDataset(Dataset):
    def __init__(self, X, y):
        """Reads source and target sequences from txt files."""
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        data = {}
        data["X"] = self.X[index]
        data["y"] = self.y[index]
        return data
    
def make_discriminative_loader(config, model, loader):
    hiddens, labels = [], []
    for batch in loader:
        h = model.control_variables(
            batch["seq"], batch["attn_mask"]
        )
        hiddens.append(h)
        labels.append(batch["labels"])

    hiddens = np.array(hiddens)
    labels = np.array(labels)

    dset = BasicDataset(hiddens, labels)

    loader = DataLoader(
        dataset=dset,
        batch_size=config.batch_size,
        shuffle=True, #maybe change to not shuffle val loader
        num_workers=1,
        pin_memory=True,
    )

    return loader
