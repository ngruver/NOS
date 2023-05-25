import os
import re
import math
import glob
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt

from iglm import IgLM
from seq_models.metrics import (
    BioPythonSeqLabeler, 
    BioPythonStructLabeler,
    IgFoldWrapper,
)

def get_ranges(mask):
    end = 0
    ranges = []
    while True:
        start = mask.find("1", end)
        if start == -1:
            break

        end = mask.find("0", start)
        if end == -1:
            end = len(mask)

        ranges.append((start, end))
    
    return ranges

def iglm_perplexity(
    iglm_model,
    chain,
    ranges,
    chain_token,
    species_token,
):
    perplexity = 0
    for r in ranges:
        perplexity += -1 * iglm_model.log_likelihood(
            chain,
            chain_token,
            species_token,
            infill_range=r,
        )

    return perplexity

def infill_metrics_sample(
    iglm_model,
    vh_seed="",
    vl_seed="",
    vh_sample="",
    vl_sample="",
    vh_mask="",
    vl_mask="",
    species_token="[HUMAN]", 
    **kwargs,
):
    iglm_labels = {}

    vl_ranges = get_ranges(vl_mask)  
    if len(vl_ranges) > 0:
        # print(vl_seed)
        # print(vl_sample + "\n")
        print(vl_sample)
        label = "vl_infill_perplexity"
        iglm_labels[label] = (
            iglm_perplexity(
                iglm_model,
                vl_sample,
                vl_ranges,
                "[LIGHT]",
                species_token,
            )
        )

    vh_ranges = get_ranges(vh_mask)
    if len(vh_ranges) > 0:
        # print(vh_seed)
        # print(vh_sample + "\n")
        print(vh_sample)
        label = "vh_infill_perplexity"
        iglm_labels[label] = (
            iglm_perplexity(
                iglm_model,
                vh_sample,
                vh_ranges,
                "[HEAVY]",
                species_token,
            )
        )

    seq = vh_sample + vl_sample

    if vl_mask == "0":
        vl_mask = "".join(len(vl_seed) * ["0"])
    if vh_mask == "0":
        vh_mask = "".join(len(vh_seed) * ["0"])

    seed = vh_seed + vl_seed
    mask = np.array([int(x) for x in vh_mask + vl_mask], dtype=bool)

    if np.sum(mask) == 0:
        print(vh_mask, vl_mask)
        print(kwargs["sample_tag"])
        print(kwargs["model_tag"])
        print("")
        # print(1/0)

    if len(seq) == len(seed) == len(mask):        
        seq_recovery = (
            np.sum(
                np.array(
                    [x == y for x, y in zip(seed, seq)],
                    dtype=bool,
                ) & mask
            ) / np.sum(mask)
        )

        iglm_labels["seq_recovery"] = seq_recovery
    else:
        print(len(seq), len(seed), len(mask))
        print(vh_mask, vl_mask)
        print(kwargs["sample_tag"])
        print(kwargs["model_tag"])
        print("")

    return iglm_labels

def label_sample(
    seq_labeler,
    struct_labeler,
    vh_sample="",
    vl_sample="",
    **kwargs,  
):
    all_labels = {}

    seq = vh_sample + vl_sample
    all_labels.update(
        seq_labeler.label_seq(seq)
    )

    if struct_labeler is not None:
        struct_labeler, igfold = struct_labeler
        chains = {"H": vh_sample, "L": vl_sample}
        struct_labels = igfold.fold_and_label(chains, struct_labeler)
        all_labels.update(struct_labels)

    return all_labels

def label_csv(
    input_fn,
    structure_labels=False,
    species_token="[HUMAN]",
):
    iglm_model = IgLM()
    seq_labeler = BioPythonSeqLabeler()
    struct_labeler = None
    if structure_labels:
        struct_labeler = BioPythonStructLabeler()
        struct_labeler = (struct_labeler, IgFoldWrapper())

    df = pd.read_csv(
        input_fn, 
        dtype={
            "vh_mask": str,
            "vl_mask": str,
        },
    )

    df = df[:3 * len(df) // 5]
    df = df[df['sample_num'] < 3]

    new_df = []
    for sample_dict in df.to_dict('records'):
        infill_dict = infill_metrics_sample(
            iglm_model,
            **sample_dict,
            species_token=species_token,
        )

        label_dict = label_sample(
            seq_labeler,
            struct_labeler,
            **sample_dict,
        )

        # import pprint
        # pprint.pprint(label_dict)

        new_df.append({
            **sample_dict, 
            **infill_dict,
            **label_dict,
        })

    df = pd.DataFrame(new_df)

    # print(df)
    if 'vl_infill_perplexity' in df.columns:
        print(df['vl_infill_perplexity'].mean())
    elif 'vh_infill_perplexity' in df.columns:
        print(df['vh_infill_perplexity'].mean())
    if 'seq_recovery' in df.columns:
        print(df['seq_recovery'].mean())

    basename = os.path.splitext(os.path.basename(input_fn))[0]
    labeled_fn = os.path.join(
        os.path.dirname(input_fn), 
        f"{basename}_labeled.csv",
    )
    df.to_csv(labeled_fn, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add input file path")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input file path")
    parser.add_argument("-s", "--structure_labels", type=int, default=0, help="Whether to compute structure labels")

    args = parser.parse_args()

    label_csv(args.input, args.structure_labels > 0)
