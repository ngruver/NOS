import os
import copy
import torch
import pandas as pd
from pprint import pprint
import transformers

from seq_models.util.numbering import mask_regions
import seq_models.metrics as metrics
from seq_models.data import (
    DEFAULT_INFILL_SEED,
    SPECIAL_TOKENS,
    SPECIAL_TOKEN_INDICES,
    AB_SEQUENCE_LENGTH,
    pad_to_length
)

def formatted_to_chains(x, aligned_masks):
    x = x.split(" ")
    chains = {
        "vh": "".join(x[1:150]),
        "vl": "".join(x[151:-1])
    }

    new_masks = {}
    for k in chains:
        mask = aligned_masks[k]
        chain = chains[k]
        
        new_mask = "".join([x for x, y in zip(mask, chain) if y != "-"])
        chain = chain.replace("-", "")

        chains[k] = chain
        new_masks[k] = new_mask

    return chains, new_masks

def chains_to_formatted(vh, vl):
    return " ".join(
        ["[AgHC]"] + vh + ["[AgLC]"] + vl + ["[Ag]"]
    )

def make_tags(
    numbering_schemes, 
    cdr_combos
):
    tags = []
    for num_scheme in numbering_schemes:
        tags += [
            num_scheme + ":" + "/".join(cdr_ids) for cdr_ids in cdr_combos
        ]
    return tags

def make_sampling_csv(
    seeds_fn, 
    out_fn,
    numbering_schemes, 
    cdr_combos
):
    tags = make_tags(numbering_schemes, cdr_combos)
    seeds = pd.read_csv(seeds_fn).values

    df_vals = []
    for tag in tags:
        for vh, vl in seeds:
            df_vals.append({
                "vh": vh,
                "vl": vl,
                "ids": tag,
            })
    df = pd.DataFrame(df_vals)

    df.to_csv(out_fn, index=False)

def sample_model(
    model,
    num_samples,
    infill_seed_file,
    vocab_file,
    gt_data_file=None,
    guidance_kwargs=None,
    use_alignment_tokens=True,
    bad_word_ids=None,
    autoregressive_sample=False,
):
    log, wandb_log = {}, {}

    if infill_seed_file is None:
        infill_seeds = [None]
    else:
        with open(infill_seed_file, "r") as fd:
            infill_seeds = [x.strip() for x in fd.readlines()]

    device = next(model.parameters()).device
    tokenizer = transformers.BertTokenizerFast(
        vocab_file=vocab_file, 
        do_lower_case=False,
    )

    for i, infill_seed in enumerate(infill_seeds):
        
        tag = "" if guidance_kwargs is None else "guided_"
        if infill_seed is None:
            tag += "unconditional" 
        else:
            tag += f"infill_{i}" 

        if infill_seed is None:
            infill_seed = DEFAULT_INFILL_SEED

        if not use_alignment_tokens:
            infill_seed = infill_seed.replace("- ","")
        print(infill_seed)

        infill_seed = torch.Tensor(
            tokenizer.convert_tokens_to_ids(infill_seed.split(" "))
        ).long().to(device)

        if not use_alignment_tokens:
            infill_seed = pad_to_length(infill_seed, AB_SEQUENCE_LENGTH)

        infill_mask = infill_seed == tokenizer.mask_token_id
        corrupt_mask = torch.ones_like(infill_mask)
        
        if use_alignment_tokens:
            corrupt_mask[SPECIAL_TOKEN_INDICES] = 0
        else:
            special_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
            for special_id in special_ids:
                corrupt_mask[infill_seed == special_id] = 0

        sample_fn = model.sample
        if autoregressive_sample:
            sample_fn = model.sample_autoregressive
    
        if guidance_kwargs is not None:
            model.network.regression_head.stop_grad = False

        with torch.no_grad():
            samples = sample_fn(
                infill_seed=infill_seed,
                infill_mask=infill_mask,
                corrupt_mask=corrupt_mask,
                num_samples=num_samples,
                guidance_kwargs=copy.deepcopy(guidance_kwargs),
                bad_word_ids=bad_word_ids,
                # batch_size=256
            )
        samples = [tokenizer.decode(s) for s in samples]

        seed_log, seed_wandb_log = metrics.evaluate_samples(
            samples,
            infill_seed=None,
            vocab_file=vocab_file,
            gt_file=gt_data_file,
            log_prefix=tag
        )
                
        log.update(seed_log)
        wandb_log.update(seed_wandb_log)

    return log, wandb_log

def sample_inner_loop(
    seeds_fn,
    results_dir,
    model,
    model_tag,
    vocab_file,
    num_samples=10,
    sampling_kwargs={},
):
    pprint(sampling_kwargs)

    fixed_length = sampling_kwargs.pop("fixed_length", False)

    tokenizer = transformers.BertTokenizerFast(
        vocab_file=vocab_file, 
        do_lower_case=False,
    )
    mask_token = tokenizer.mask_token

    seeds = pd.read_csv(seeds_fn).values

    vals = []
    for vh, vl, mask_spec in seeds:
        mask_info = mask_regions(
            {'vh': vh, 'vl': vl}, 
            mask_spec, 
            mask_token=mask_token, 
            fixed_length=fixed_length,
        )

        masked_seed = chains_to_formatted(
            mask_info["vh"]["masked_seed"],
            mask_info["vl"]["masked_seed"],
        )

        vals.append({
            "info": mask_info,
            "masked": masked_seed,
        })

    infill_fn = os.path.join(
        results_dir, 
        f"{model_tag}_infill_seeds.txt"
    )
    with open(infill_fn, "w") as f:
        f.writelines([v["masked"] + "\n" for v in vals])

    bad_word_ids = tokenizer.convert_tokens_to_ids(
        SPECIAL_TOKENS + ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
    )
    if fixed_length:
        bad_word_ids += tokenizer.convert_tokens_to_ids(['-'])

    sample_log, _ = sample_model(
        model,
        num_samples,
        infill_fn,
        vocab_file,
        bad_word_ids=bad_word_ids,
        **sampling_kwargs,
    )  

    out = []
    keys = [
        [k for k in sample_log.keys() if str(i) in k][0]
            for i in range(len(seeds))
    ]
    for k in keys:
        seed_idx = int(k.split("/")[0].split("_")[-1])
        
        info = vals[seed_idx]["info"]
        vh_seed = info["vh"]["seed"]
        vl_seed = info["vl"]["seed"]
        vh_mask = info["vh"]["mask_str"]
        vl_mask = info["vl"]["mask_str"]
        vl_aligned_mask = info["vl"]["aligned_mask_str"]
        vh_aligned_mask = info["vh"]["aligned_mask_str"]
        aligned_masks = {
            "vl": vl_aligned_mask,
            "vh": vh_aligned_mask,
        }

        print(k)
        print(sample_log[k]['biopython']['samp']['ss_perc_sheet'].mean())

        samples = sample_log[k]["seq_samples"]
        for i, sample in enumerate(samples):
            sample_chains, sample_masks = formatted_to_chains(
                sample, aligned_masks
            )
            
            vh_sample = sample_chains['vh']
            vl_sample = sample_chains['vl']
            vh_mask = sample_masks["vh"]
            vl_mask = sample_masks["vl"]

            if len(vh_seed) != len(vh_sample) != len(vh_mask):
                print(len(vh_seed), len(vh_sample), len(vh_mask))
                print(vals[seed_idx]["info"]['vh'])
                print(sample)
                print("")
            elif len(vl_seed) != len(vl_sample) != len(vl_mask):
                print(len(vl_seed), len(vl_sample), len(vl_mask))
                print(vals[seed_idx]["info"]['vl'])
                print(sample)
                print("")

            if '[' in vh_sample:
                print(vh_sample)
                raise ValueError("Found a bracket in the sample")
            if '[' in vl_sample:
                print(vl_sample)
                raise ValueError("Found a bracket in the sample")


            # print(sample)
            # print(vh_seed)
            # print(vh_sample)
            # print("")

            sample_dict = {
                "vh_seed": vh_seed,
                "vl_seed": vl_seed, 
                "sample_num": i,
                "vh_sample": vh_sample,
                "vl_sample": vl_sample,
                "vh_mask": ''.join([str(int(x)) for x in vh_mask]),
                "vl_mask": ''.join([str(int(x)) for x in vl_mask]),
                "sample_tag": info["tag"],
                "model_tag": model_tag,
                "fixed_length": fixed_length,
            }

            if "guidance_kwargs" in sampling_kwargs:
                guidance_kwargs = sampling_kwargs["guidance_kwargs"]
                sample_dict.update({
                    f"guidance_{k}":v for k,v in guidance_kwargs.items()
                })
                sample_dict.update({
                    k:v for k,v in sampling_kwargs.items() if k != "guidance_kwargs"
                })
            else:
                sample_dict.update(sampling_kwargs)

            out.append(sample_dict)

    df = pd.DataFrame(out)
    df.to_csv(
        os.path.join(
            results_dir, 
            f"{model_tag}_samples.csv"
        ), 
        index=False
    )

    return df

def sample_outer_loop(
    model,
    model_tag,
    results_dir,
    seeds_fn,
    vocab_file,
    numbering_schemes,
    cdr_combos,
    sampling_kwargs_list,
    do_label=False,
):

    csv_fn = os.path.join(
        results_dir, 
        f"{model_tag}_sample_spec.csv"
    )
    make_sampling_csv(
        seeds_fn,
        csv_fn,
        numbering_schemes,
        cdr_combos
    )

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
            results_dir, 
            f"{full_tag}_samples.csv"
        )

        if os.path.exists(result_fn):
            print(f"Skipping {full_tag}")
            continue

        try:
            dfs.append(
                sample_inner_loop(
                    csv_fn,
                    results_dir,
                    model,
                    full_tag,
                    vocab_file,
                    num_samples=10,
                    sampling_kwargs=sampling_kwargs,
                )
            )
        except Exception as e:
            print(e)
            print(f"Failed to sample for {full_tag}")

    df = pd.concat(dfs)
    df.to_csv(
        os.path.join(
            results_dir,
            f"combined_samples.csv"
        ),
        index=False
    )
