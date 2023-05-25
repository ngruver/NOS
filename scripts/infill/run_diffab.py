import os
import glob
import yaml
import pprint
import subprocess
import pandas as pd

from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1

import sys
sys.path.append("diffab")
import diffab.utils.protein.constants as constants

DIFFAB_NAME_MAP = {
    "H1": "H_CDR1",
    "H2": "H_CDR2",
    "H3": "H_CDR3",
    "L1": "L_CDR1",
    "L2": "L_CDR2",
    "L3": "L_CDR3",
}

DIFFAB_CDRS = {
    'H1': constants.ChothiaCDRRange.H1,
    'H2': constants.ChothiaCDRRange.H2,
    'H3': constants.ChothiaCDRRange.H3,
    'L1': constants.ChothiaCDRRange.L1,
    'L2': constants.ChothiaCDRRange.L2,
    'L3': constants.ChothiaCDRRange.L3,
}

def create_config(
    cdr_ids,
    results_dir,
    tag="",
    num_samples=10,
):
    config_dir = "./diffab/configs/test"
    if len(cdr_ids) == 1:
        base_config = "codesign_single.yml"
    else:
        base_config = "codesign_multicdrs.yml"
    
    config_fn = os.path.join(config_dir, base_config)
    with open(config_fn, 'r') as fd:
        config = yaml.safe_load(fd)

    config["sampling"]["cdrs"] = [DIFFAB_NAME_MAP[cdr_id] for cdr_id in cdr_ids]
    config["sampling"]["num_samples"] = num_samples

    new_fn = os.path.join(results_dir, f"{tag}.yml")
    with open(new_fn, 'w') as fd:
        yaml.dump(config, fd, default_flow_style=False)

    return new_fn

def run_diffab(
    pdb_dir,
    config_fn,
    results_dir,
):
    pdb_fns = glob.glob(os.path.join(pdb_dir, "*.pdb"))
    pdb_fns = [x for x in pdb_fns if not "chothia" in x]
    for pdb_fn in pdb_fns:
        command = (
            f"python design_pdb.py "
            f"{pdb_fn} "
            f"--config {config_fn} "
            f"--out_root {results_dir}"
        )
        print(command)
        p = subprocess.Popen(command.split(" "), cwd='./diffab')
        p.wait()


def parse_chains(pdb_file):
    structure = PDBParser().get_structure("", pdb_file)
    chains = {
        chain.id:seq1(''.join(residue.resname for residue in chain)) 
            for chain in structure.get_chains()
    } 
    return chains


def parse_diffab_results(
    results_dir,
    cdr_ids,
    tag,
):
    base_dir = os.path.join(results_dir, tag)
    sample_dirs = glob.glob(os.path.join(base_dir, "*pdb*", "*"))
    sample_dirs = [x for x in sample_dirs if os.path.isdir(x)]
    
    out = []
    for d in sample_dirs:
        pdb_files = glob.glob(os.path.join(d, "*.pdb"))
        
        ref_file = [x for x in pdb_files if "REF" in x][0]
        seed_chains = parse_chains(ref_file)

        vh_ranges = [
            DIFFAB_CDRS[cdr_id] for cdr_id in cdr_ids if "H" in cdr_id
        ]
        vl_ranges = [
            DIFFAB_CDRS[cdr_id] for cdr_id in cdr_ids if "L" in cdr_id
        ]

        vh_seed = seed_chains['H']
        vl_seed = seed_chains['L']
        
        vh_mask = len(vh_seed) * ["0"]
        for vh_range in vh_ranges:
            vh_mask[vh_range[0]:vh_range[1]] = (vh_range[1] - vh_range[0]) * ["1"]
        vh_mask = "".join(vh_mask)

        vl_mask = len(vl_seed) * ["0"]
        for vl_range in vl_ranges:
            vl_mask[vl_range[0]:vl_range[1]] = (vl_range[1] - vl_range[0]) * ["1"]
        vl_mask = "".join(vl_mask)

        sample_files = [x for x in pdb_files if not "REF" in x]
        for i, pdb_file in enumerate(sample_files): 
            sample_chains = parse_chains(pdb_file)

            out.append({
                "vh_seed": seed_chains['H'],
                "vl_seed": seed_chains['L'], 
                "sample_num": i,
                "vh_sample": sample_chains['H'],
                "vl_sample": sample_chains['L'],
                "vh_mask": vh_mask,
                "vl_mask": vl_mask,
            })

    df = pd.DataFrame(out)
    return df

def main():
    pdb_dir = "/home/nvg7279/src/seq-struct/poas_seed_pdbs"
    results_dir = "/home/nvg7279/src/seq-struct/diffab_infill"

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    cdr_combos = [
        ["H1"],
        ["H2"],
        ["H3"],
        ["H1", "H2", "H3"],
        ["L1"],
        ["L2"],
        ["L3"],
    ]
    for cdr_ids in cdr_combos:
        tag = "_".join([cdr.lower() for cdr in cdr_ids])
        config_fn = create_config(cdr_ids, results_dir, tag)
        run_diffab(pdb_dir, config_fn, results_dir)
        df = parse_diffab_results(results_dir, cdr_ids, tag)
        df.to_csv(os.path.join(results_dir, tag + ".csv"), index=False)

if __name__ == "__main__":
    main()