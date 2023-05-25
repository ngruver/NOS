import os
import torch
import argparse
import itertools
import numpy as np
from tqdm import tqdm
import pandas as pd

from seq_models.model.autoregressive import (
    GuidedIgLM,
    load_guidance_models,
)
from seq_models.util.numbering import (
    mask_regions,
    get_species,
)
from seq_models.sample import (
    make_sampling_csv,
)

SPECIES_TO_TOKEN = {
    "camel": "[CAMEL]",
    "alpaca": "[CAMEL]",
    "human": "[HUMAN]",
    "mouse": "[MOUSE]",
    "rabbit": "[RABBIT]",
    "rat": "[RAT]",
    "rhesus": "[RHESUS]",
}

CHAIN_TOKENS = {
    "vh": "[HEAVY]",
    "vl": "[LIGHT]",
}

def infill_ranges(
    iglm_model, 
    seed, 
    ranges, 
    chain_token, 
    fixed_length,
):
    mask_arr = np.zeros(len(seed))
    for r in ranges:
        mask_arr[r[0]:r[1]] = 1

    mask_str = ''.join([str(int(x)) for x in mask_arr])

    if len(ranges) == 0:
        return seed, mask_str

    species = get_species(seed)
    species_token = SPECIES_TO_TOKEN[species]

    new_seed = seed
    for i in range(len(ranges)):
        r = ranges[i]

        min_new_tokens, max_new_tokens = None, None
        if fixed_length:
            min_new_tokens = int(r[1] - r[0]) + 1
            max_new_tokens = int(r[1] - r[0]) + 1

        sample = iglm_model.infill(
            new_seed,
            chain_token,
            species_token,
            infill_range=r,
            num_to_generate=1,
            min_new_tokens=min_new_tokens,
            max_new_tokens=max_new_tokens,
        )[0]

        delta = len(sample) - len(new_seed)
        for j in range(i+1, len(ranges)):
            ranges[j] = (
                ranges[j][0] + delta,
                ranges[j][1] + delta,
            )

        new_seed = sample

    return sample, mask_str

def guided_sample(
    iglm_model, 
    guidance_models,
    seeds_fn,
    results_dir,
    model_tag,
    num_samples=10,
    sampling_kwargs={},
    species_token="[HUMAN]", 
):
    fixed_length = sampling_kwargs.pop("fixed_length", False)

    seeds = pd.read_csv(seeds_fn).values

    df = []
    for vh, vl, mask_spec in tqdm(seeds):
        mask_info = mask_regions(
            {'vh': vh, 'vl': vl}, 
            mask_spec, 
            fixed_length=fixed_length,
        )

        samples = []
        for _ in range(200):
            if len(samples) == num_samples:
                break

            out = {}
            for k, v in CHAIN_TOKENS.items():
                iglm_model.model.set_guidance_model(
                    guidance_models[k].cuda(), 
                    **sampling_kwargs["guidance_kwargs"],
                )

                with torch.no_grad():
                    sample, mask_str = infill_ranges(
                        iglm_model,
                        mask_info[k]["seed"],
                        mask_info[k]["mask_ranges"],
                        v,
                        fixed_length,
                    )

                out.update({
                    f"{k}_seed": mask_info[k]["seed"],
                    f"{k}_sample": sample,
                    f"{k}_mask": mask_str,
                })

            len_change = any(
                [len(out[f"{k}_seed"]) != len(out[f"{k}_sample"]) for k in CHAIN_TOKENS]
            )
            if fixed_length and len_change:
                continue

            out.update({
                "sample_num": len(samples),
                "sample_tag": mask_info["tag"],
                "model_tag": model_tag,
                "fixed_length": fixed_length,
            })

            samples.append(out)

        df += samples

    df = pd.DataFrame(df)
    results_fn = os.path.join(
        results_dir, 
        f"{model_tag}_samples.csv"
    )
    df.to_csv(results_fn, index=False)



    # from seq_models.compare_samples import label_csv

    # label_csv(results_fn)
    # basename = os.path.splitext(os.path.basename(results_fn))[0]
    # labeled_fn = os.path.join(
    #     os.path.dirname(results_fn), 
    #     f"{basename}_labeled.csv",
    # )
    # labeled_df = pd.read_csv(labeled_fn)

    # labels=["ss_perc_sheet"]
    # train_data_path="/scratch/nvg7279/datasets/igfold_labeled_cleaned/train.csv"
    # train_df = pd.read_csv(train_data_path)
    # print(train_df[labels].mean())
    # print(labeled_df[labels].mean())
    # print("")

    return df

def main(args):
    iglm_model = GuidedIgLM()
    guidance_models = load_guidance_models(args)
    
    model_tag = 'iglm'
    numbering_schemes = ["aho"]
    cdr_combos = [["hcdr1", "hcdr2", "hcdr3"]]

    sample_spec_csv = os.path.join(
        args.results_dir, 
        f"{model_tag}_sample_spec.csv"
    )
    make_sampling_csv(
        args.seeds_fn,
        sample_spec_csv,
        numbering_schemes,
        cdr_combos
    )

    guidance_options = {
        "step_size": list(np.linspace(0.5, 2.0, 6)),
        "stability_coef": [10.0, 1.0, 0.1, 0.01, 0.001, 0.],
        "num_steps": [5, 10],
    }
    combos = pd.DataFrame(
        list(itertools.product(*guidance_options.values())), 
        columns=guidance_options.keys()
    )

    base_sampling_kwargs = {
        "fixed_length": True,
    }
    sampling_kwargs_list = []
    for guidance_options in combos.to_dict('records'):
        kwargs = base_sampling_kwargs.copy()
        kwargs.update({"guidance_kwargs": guidance_options})
        sampling_kwargs_list.append(kwargs)

    import random
    random.shuffle(sampling_kwargs_list)

    dfs = []
    for sampling_kwargs in sampling_kwargs_list:
        kwargs_tag = "_".join(
            f"{k}={v}" for k, v in sampling_kwargs.items() if k != "guidance_kwargs"
        )
        if "guidance_kwargs" in sampling_kwargs:
            guidance_kwargs = sampling_kwargs["guidance_kwargs"]
            kwargs_tag += "_" + "_".join(
                f"{k}={v}" for k, v in guidance_kwargs.items()
            )
        full_tag = f"{model_tag}_{kwargs_tag}"
        
        result_fn = os.path.join(
            args.results_dir, 
            f"{full_tag}_samples.csv"
        )

        if os.path.exists(result_fn):
            print(f"Skipping {full_tag}")
            continue

        try:
            dfs.append(
                guided_sample(
                    iglm_model, 
                    guidance_models,
                    sample_spec_csv,
                    args.results_dir,
                    full_tag,
                    sampling_kwargs=sampling_kwargs,
                )
            )
        except Exception as e:
            print(e)
            print(f"Failed to sample for {full_tag}")
    

    df = pd.concat(dfs)
    df.to_csv(
        os.path.join(
            args.results_dir,
            f"combined_samples.csv"
        ),
        index=False
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Guidance model path")
    parser.add_argument("--train_data_path", type=str, required=True, help="Training data path")
    parser.add_argument("--val_data_path", type=str, required=True, help="Validation data path")
    parser.add_argument("--labels", type=str, required=True, help="Training data labels")
    parser.add_argument("--guidance_model_dir", type=str, required=True, help="Guidance model directory")
    parser.add_argument("--seeds_fn", type=str, required=True, help="Filename of seeds")
    parser.add_argument("--results_dir", type=str, required=True, help="Results directory")

    args = parser.parse_args()

    main(args)
