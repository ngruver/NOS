import copy
import numpy as np
import anarci as an
import abnumber

CHOTIA_RANGES = [
    "fr1", "cdr1", "fr2", "cdr2", "fr3", "cdr3", "fr4"
]

AHO_CDR_RANGES = {
    "FR1": (0, 24),
    "CDR1": (24, 40),
    "FR2": (40, 57),
    "CDR2": (57, 77),
    "FR3": (77, 108),
    "CDR3": (108, 137),
    "FR4": (137, 148),
}

ALLOWED_SPECIES = ["rabbit", "rat", "mouse", "human", "rhesus", "camel", "alpaca"]

def align_with_anarci(chain_sequence: str):
    renumbered_aho = list(
        an.run_anarci(
            seq=chain_sequence,
            scheme="aho",
            assign_germline=True,
            allowed_species=ALLOWED_SPECIES,
            ncpu=1,
        )
    )
    # Anarci returns a list even though we processed only 1 sequence, so remove the extra index
    for i in range(len(renumbered_aho)):
        renumbered_aho[i] = renumbered_aho[i][0]

    return renumbered_aho

def align_chain(seq, chain_id):
    alignment = align_with_anarci(seq)
    species = alignment[2][0]['species']
    if species not in ALLOWED_SPECIES:
        raise Exception(f"Skipping because species is {species}")

    aligned = "".join([x[1] for x in alignment[1][0][0]])
    if len(aligned) != 149 and chain_id == 'H':
        raise Exception(f"Skipping because length is {len(aligned)}")
    if len(aligned) != 148 and chain_id == 'L':
        raise Exception(f"Skipping because length is {len(aligned)}")

    return aligned

def get_species(seq):
    alignment = align_with_anarci(seq)
    species = alignment[2][0]['species']
    return species

def contiguous_true_subarrs(arr):
    start_indices = np.where(np.diff(np.hstack(([0], arr, [0]))) > 0)[0]
    end_indices = np.where(np.diff(np.hstack(([0], arr, [0]))) < 0)[0] - 1    
    contiguous_subarrays = list(zip(start_indices, end_indices))
    return contiguous_subarrays

def replace_dash_between_x(sequence):
    new_sequence = sequence.copy()
    i = 1
    while i < len(sequence) - 1:
        if sequence[i] == "-" and sequence[i - 1] == "X":
            j = i
            while j < len(sequence) and sequence[j] == "-":
                j += 1

            if j < len(sequence) and sequence[j] == "X":
                for k in range(i, j):
                    new_sequence[k] = "X"

            i = j
        else:
            i += 1
    return new_sequence

def mask_regions_chain(
    seq,
    chain_id,
    region_ids,
    mask_token="X",
    scheme="aho",
    fixed_length=False,
):
    aligned_seq = list(align_chain(seq, chain_id.upper()))

    if scheme == "aho":
        masked = copy.deepcopy(aligned_seq)
        for range in region_ids:
            start, end = AHO_CDR_RANGES[range.upper()]
            masked[start:end] = [mask_token] * (end - start)

        if fixed_length:
            masked = [m if og != "-" else "-" for m, og in zip(masked, aligned_seq)]

    elif scheme == "chothia":    
        chain = abnumber.Chain(seq, scheme='chothia')
        masked = []
        for range in CHOTIA_RANGES:
            if range in region_ids:
                masked += [mask_token] * len(getattr(chain, range + "_seq"))
            else:
                masked += list(getattr(chain, range + "_seq"))

        aho_masked = []
        new_idx = 0
        for x in aligned_seq:
            if x == "-":
                aho_masked.append(x)
            else:
                aho_masked.append(masked[new_idx])
                new_idx += 1

        masked = aho_masked

    masked_arr = np.array([
        int(x == mask_token) for x in replace_dash_between_x(masked)
    ])
    masked_ranges = contiguous_true_subarrs(masked_arr)

    aligned_mask_arr = np.array([
        int(x == mask_token) for x in masked
    ])
    aligned_mask_str = ''.join([str(x) for x in aligned_mask_arr])

    mask_arr = np.array([
        int(x == mask_token) for x in masked if x != '-'
    ])
    mask_str = ''.join([str(x) for x in mask_arr])
    mask_ranges = contiguous_true_subarrs(mask_arr)

    return {
        "seed": seq,
        "masked_seed": masked,
        "masked_arr": masked_arr,
        "mask_arr": mask_arr,
        "mask_str": mask_str,
        "mask_ranges": mask_ranges,
        "masked_ranges": masked_ranges,
        "aligned_mask_str": aligned_mask_str,
        "aligned_mask_arr": aligned_mask_arr,
    }

def mask_regions(
    seqs,
    mask_spec,
    mask_token="X",
    fixed_length=False,
):
    scheme, region_ids = mask_spec.split(":")
    tag = scheme + "_" + region_ids.replace("/", "_")
    region_ids = region_ids.split("/")

    return {
        "vh": mask_regions_chain(
            seqs["vh"], 
            'h',
            [x[1:] for x in region_ids if x[0] == "h"],
            mask_token, scheme, fixed_length
        ),
        "vl": mask_regions_chain(
            seqs["vl"], 
            'l',
            [x[1:] for x in region_ids if x[0] == "l"],
            mask_token, scheme, fixed_length
        ),
        "tag": tag
    }

reformat = lambda x: "".join(x).replace("-", "")
