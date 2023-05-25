import os
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np

from iglm import IgLM
from iglm.model.tokens import (
    BAD_WORD_IDS,
)
from iglm.model.utils import (
    validate_generated_seq,
    iglm_to_infilled,
    mask_span,
)

from seq_models.util.numbering import (
    mask_regions,
    get_species,
)

from scripts.utils import (
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

class LengthConstrainedIgLM(IgLM):

    def _generate(
        self, 
        starting_tokens, 
        num_to_generate,
        max_length, 
        min_new_tokens,
        max_new_tokens,
        top_p, 
        temperature
    ):
        decoded_seqs = set()  # Set to remove duplicates
        while len(decoded_seqs) < num_to_generate:

            if max_new_tokens is not None:
                seq = self.model.generate(
                    starting_tokens,
                    min_new_tokens=min_new_tokens,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.cls_token_id,
                    forced_eos_token_id=self.tokenizer.cls_token_id,
                    bad_words_ids=BAD_WORD_IDS + [[self.tokenizer.sep_token_id]],
                    do_sample=True,
                    top_p=top_p,
                    temperature=temperature).detach().cpu().numpy()
            else:
                seq = self.model.generate(
                    starting_tokens,
                    max_length=max_length,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.cls_token_id,
                    forced_eos_token_id=self.tokenizer.cls_token_id,
                    bad_words_ids=BAD_WORD_IDS + [[self.tokenizer.sep_token_id]],
                    do_sample=True,
                    top_p=top_p,
                    temperature=temperature).detach().cpu().numpy()

            seq = seq[0]  # Squeeze out batch dimension
            if validate_generated_seq(seq, self.tokenizer):
                decoded_tokens = self.tokenizer.decode(
                    iglm_to_infilled(seq, self.tokenizer))
                decoded_seq = ''.join(decoded_tokens).replace(' ', '')
                if decoded_seq not in decoded_seqs:
                    decoded_seqs.add(decoded_seq)
            else:
                decoded_tokens = self.tokenizer.decode(seq)
                decoded_seq = ''.join(decoded_tokens).replace(' ', '')
                print(''.join(self.tokenizer.decode(starting_tokens[0])).replace(' ', ''))
                print(decoded_seq)
                print("")

        return list(decoded_seqs)
    
    def infill(
        self,
        sequence,
        chain_token,
        species_token,
        infill_range,
        num_to_generate=1000,
        max_length=150,
        min_new_tokens=None,
        max_new_tokens=None,
        top_p=1,
        temperature=1,
    ):
        sequence = list(sequence)
        masked_seq = mask_span(
            sequence,
            infill_range[0],
            infill_range[1],
        )  # mask using provided indices
        start_tokens = [chain_token, species_token] + masked_seq
        start_tokens = torch.Tensor([
            self.tokenizer.convert_tokens_to_ids(start_tokens)
        ]).int().to(self.device)

        assert (start_tokens != self.tokenizer.unk_token_id
                ).all(), "Unrecognized token supplied in starting tokens"

        generated_seqs = self._generate(
            start_tokens,
            num_to_generate=num_to_generate,
            max_length=max_length,
            min_new_tokens=min_new_tokens,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            temperature=temperature,
        )

        return generated_seqs

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

    # print(seed)
    # print(ranges)

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

def infill(
    iglm_model,
    model_tag,
    info,
    num_samples=10,
    fixed_length=False,
):
    out = []
    for _ in range(200):
        if len(out) == num_samples:
            break

        vh_seed = info["vh"]["seed"]
        vl_seed = info["vl"]["seed"] 

        vh_sample, vh_mask_str = infill_ranges(
            iglm_model,
            info["vh"]["seed"],
            info["vh"]["mask_ranges"],
            "[HEAVY]",
            fixed_length,
        )

        vl_sample, vl_mask_str = infill_ranges(
            iglm_model,
            info["vl"]["seed"],
            info["vl"]["mask_ranges"],
            "[LIGHT]",
            fixed_length,
        )

        len_change = len(vh_sample) != len(vh_seed) or \
                     len(vl_sample) != len(vl_seed)
        if fixed_length and len_change:
            continue

        out.append({
            "vh_seed": info["vh"]["seed"],
            "vl_seed": info["vl"]["seed"], 
            "sample_num": len(out),
            "vh_sample": vh_sample,
            "vl_sample": vl_sample,
            "vh_mask": vh_mask_str,
            "vl_mask": vl_mask_str,
            "sample_tag": info["tag"],
            "model_tag": model_tag,
            "fixed_length": fixed_length,
        })

    out = pd.DataFrame(out)

    return out

def infill_seeds(
    seeds_fn,
    results_dir,
    iglm_model,
    model_tag,
    fixed_length=False,
):
    seeds = pd.read_csv(seeds_fn).values
    
    out = []
    for vh, vl, mask_spec in tqdm(seeds):

        mask_info = mask_regions(
            {"vh": vh, "vl": vl},
            mask_spec,
            fixed_length=fixed_length,
        )

        out.append(
            infill(
                iglm_model,
                model_tag,
                mask_info,
                fixed_length=fixed_length
            )
        )
    
    df = pd.concat(out)
    df.to_csv(
        os.path.join(
            results_dir, 
            f"{model_tag}_samples.csv"
        ), 
        index=False
    )

    return out

def main():
    iglm_model = LengthConstrainedIgLM()
    model_tag = 'iglm'

    seeds_fn = "/home/nvg7279/src/seq-struct/poas_seeds.csv"
    results_dir = "/home/nvg7279/src/seq-struct/infill_new"

    numbering_schemes = ["chothia", "aho"]
    cdr_combos = [
        ["hcdr1"],
        ["hcdr2"],
        ["hcdr3"],
        ["hcdr1", "hcdr2", "hcdr3"],
        ["lcdr1"],
        ["lcdr2"],
        ["lcdr3"],
    ]

    sample_spec_csv = os.path.join(
        results_dir, 
        f"{model_tag}_sample_spec.csv"
    )
    make_sampling_csv(
        seeds_fn,
        sample_spec_csv,
        numbering_schemes,
        cdr_combos
    )

    for fixed_length in [True, False]:
        full_tag = f"{model_tag}_{'fixed' if fixed_length else 'variable'}"

        infill_seeds(
            sample_spec_csv,
            results_dir,
            iglm_model,
            full_tag,
            fixed_length=fixed_length,
        )

if __name__ == "__main__":
    main()