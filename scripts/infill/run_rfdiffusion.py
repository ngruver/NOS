import os
import glob
import pandas as pd
import subprocess
from difflib import SequenceMatcher

from Bio import SeqIO
from Bio.PDB import PDBParser, PDBIO, Chain
from Bio.PDB.Polypeptide import PPBuilder

from seq_models.util.numbering import mask_regions
from seq_models.sample import make_tags

def parse_pdb_chains(pdb_file):
    parser = PDBParser()
    structure = parser.get_structure("protein", pdb_file)
    pp_builder = PPBuilder()

    sequences = {}
    for model in structure:
        for chain in model:
            chain_id = chain.get_id()
            sequence = "".join([str(pp.get_sequence()) for pp in pp_builder.build_peptides(chain)])
            sequences[chain_id] = sequence

    return sequences

def parse_fasta(file):
    sequences = {}
    with open(file, "r") as fasta_file:
        for i, record in enumerate(SeqIO.parse(fasta_file, "fasta")):
            sequences[i] = str(record.seq).split("/")
    return sequences

def get_strs(mask_info, chain_id, fixed_length):
    ranges = mask_info["mask_ranges"] 
    chain_len = len(mask_info["seed"]) 
    
    if not fixed_length:
        masked_seed = mask_info['masked_seed']
        masked_ranges = mask_info['masked_ranges']
        
        len_ranges = []
        for r in masked_ranges:
            vals = masked_seed[r[0]:r[1]] 
            max_len = len(vals)
            min_len = max_len - vals.count("-")
            len_ranges += [(min_len, max_len)]

    contigs = []
    inpaint = []
    start_idx = 0
    for i, r in enumerate(ranges):
        contigs += [f"{chain_id}{start_idx+1}-{r[0]}"]
        if fixed_length:
            contigs += [f"{r[1]-r[0]}-{r[1]-r[0]}"]
        else:
            min_len, max_len = len_ranges[i]
            contigs += [f"{min_len}-{max_len}"]
        inpaint += [f"{chain_id}{r[0]}-{r[1]}"]
        start_idx = r[1]

    contigs += [f"{chain_id}{start_idx+1}-{chain_len}"]
    return "/".join(contigs), "/".join(inpaint)

def get_rfdiffusion_range_str(mask_info, fixed_length):
    h_contigs, h_inpaint = get_strs(mask_info["vh"], "H", fixed_length)
    l_contigs, l_inpaint = get_strs(mask_info["vl"], "L", fixed_length)

    contigs = h_contigs + "/0 " + l_contigs + "/0"
    
    if len(h_inpaint) > 0 and len(l_inpaint) > 0:
        inpaint_seq = h_inpaint + "/" + l_inpaint
    elif len(h_inpaint) > 0:
        inpaint_seq = h_inpaint
    elif len(l_inpaint) > 0:
        inpaint_seq = l_inpaint

    return contigs, inpaint_seq

def renumber_pdb(input_pdb, output_pdb):
    parser = PDBParser()
    structure = parser.get_structure("protein", input_pdb)

    for model in structure:
        old_chains = []
        new_chains = []
        for chain in model:
            new_chain_id = chain.id + "_renum"
            new_chain = Chain.Chain(new_chain_id)
            for i, residue in enumerate(chain):
                new_residue = residue.copy()
                new_residue_id = (residue.id[0], i + 1, residue.id[2])
                new_residue.id = new_residue_id
                new_chain.add(new_residue)
            old_chains.append(chain)
            new_chains.append(new_chain)

        for chain, new_chain in zip(old_chains, new_chains):
            model.detach_child(chain.id)
            new_chain.id = chain.id
            model.add(new_chain)

    io = PDBIO()
    io.set_structure(structure)
    io.save(output_pdb)

def run_inference(pdb_file, output_prefix, contigs, inpaint_seq, num_samples=10):
    h_infill = len(contigs.split(" ")[0].split("/")) > 2
    l_infill = len(contigs.split(" ")[1].split("/")) > 2

    if h_infill and l_infill:
        
        command = [
            "RFdiffusion/scripts/run_inference.py",
            f"inference.output_prefix={output_prefix}_vh",
            f"inference.input_pdb={pdb_file}",
            f"contigmap.contigs=[{contigs}]",
            f"inference.num_designs={num_samples}",
            f"contigmap.inpaint_seq=[{inpaint_seq}]",
        ]
        # hydra.output_subdir

        subprocess.run(command, check=True)
    else:
        command = [
            "RFdiffusion/scripts/run_inference.py",
            f"inference.output_prefix={output_prefix}",
            f"inference.input_pdb={pdb_file}",
            f"contigmap.contigs=[{contigs}]",
            f"inference.num_designs={num_samples}",
            f"contigmap.inpaint_seq=[{inpaint_seq}]",
        ]

        subprocess.run(command, check=True)

def run_inverse_folding(
    output_prefix,
    seq_per_sample=5,
):
    pdb_dir = os.path.dirname(output_prefix)

    # Run ProteinMPNN
    output_path = os.path.join(pdb_dir, "parsed_pdbs.jsonl")
    command = [
        'python',
        f'ProteinMPNN/helper_scripts/parse_multiple_chains.py',
        f'--input_path={pdb_dir}',
        f'--output_path={output_path}',
    ]
    subprocess.run(command, check=True)

    command = [
        'python',
        f'ProteinMPNN/protein_mpnn_run.py',
        '--out_folder',
        pdb_dir,
        '--jsonl_path',
        output_path,
        '--num_seq_per_target',
        str(seq_per_sample),
        '--sampling_temp',
        '0.1',
        '--seed',
        '38',
        '--batch_size',
        '1',
    ]
    subprocess.run(command, check=True)

def match_dicts(vh_seed, vl_seed, chains):
    def get_match(seed, chains):
        keys = list(chains.keys())
        sims = [
            SequenceMatcher(None, seed, chains[k]).ratio() for k in keys
        ]
        match = keys[sims.index(max(sims))]
        return chains[match]
    
    return {
        "vh": get_match(vh_seed, chains),
        "vl": get_match(vl_seed, chains),
    }

def parse_to_df(output_prefix, info, model_tag, fixed_length):
    pdb_dir = os.path.dirname(output_prefix)
    fastas = glob.glob(os.path.join(pdb_dir,'seqs','*.fa'))

    samples = []
    for f in fastas:
        pdb_path = os.path.join(
            pdb_dir, 
            os.path.basename(f).split(".")[0] + ".pdb"
        )
        pdb_chains = parse_pdb_chains(pdb_path)
        
        rfdiffusion_samples = match_dicts(
            info["vh"]["seed"],
            info["vl"]["seed"],
            pdb_chains
        )

        mpnn_samples = parse_fasta(f)
        og_seqs = mpnn_samples[0]
        mpnn_samples = [v for k,v in mpnn_samples.items() if k != 0]

        chain_to_idx = {
            "vh": og_seqs.index(rfdiffusion_samples["vh"]),
            "vl": og_seqs.index(rfdiffusion_samples["vl"]),
        }

        og_seqs = {k: og_seqs[v] for k,v in chain_to_idx.items()}
        for s in mpnn_samples:
            sample = {}
            for chain_id in chain_to_idx:
                sampled = s[chain_to_idx[chain_id]]
                og = og_seqs[chain_id]

                # print(len(info[chain_id]["seed"]))
                # print(info[chain_id])
                # print(len(sampled))
                # print(len(og))
                # print(1/0)

                mask = info[chain_id]["mask_arr"]
                sample[chain_id] = "".join([
                    sampled[i] if mask[i] else og[i] for i in range(len(mask))
                ])
            samples.append(sample)

    df = []
    for i, sample in enumerate(samples):
        df.append({
            "vh_seed": info["vh"]["seed"],
            "vl_seed": info["vl"]["seed"], 
            "sample_num": i,
            "vh_sample": sample["vh"],
            "vl_sample": sample["vl"],
            "vh_mask": info["vh"]["mask_str"],
            "vl_mask": info["vl"]["mask_str"],
            "sample_tag": info["tag"],
            "model_tag": model_tag,
            "fixed_length": fixed_length,
        })

    df = pd.DataFrame(df)

    return df

def _tag_to_df(i, pdb_file, results_dir, sample_tag, model_tag, fixed_length, extra_tag=None):
    def rename_chain_id(chain_id):
        return "v" + chain_id.replace("_renum","").lower()
    
    chains = {
        rename_chain_id(k): v for k,v in parse_pdb_chains(pdb_file).items()
    }

    mask_info = mask_regions(
        chains,
        sample_tag,
        fixed_length=fixed_length,
    )

    sub_dir = f"{model_tag}_{sample_tag.replace('/','_')}_seed_{i}"
    if extra_tag is not None:
        sub_dir += f"_{extra_tag}"
    
    output_prefix = os.path.join(
        results_dir, sub_dir, "sample",
    )

    contigs, inpaint_seq = get_rfdiffusion_range_str(
        mask_info, fixed_length,
    )

    print(contigs, inpaint_seq)

    run_inference(
        pdb_file, output_prefix, contigs, inpaint_seq,
    )
    
    run_inverse_folding(
        output_prefix,
    )

    df = parse_to_df(
        output_prefix, mask_info, model_tag, fixed_length
    )

    return df

def tag_to_df(i, pdb_file, results_dir, sample_tag, model_tag, fixed_length):
    scheme, regions = sample_tag.split(":")

    if ("h" in regions) and ("l" in regions):                    
        dfs = {}
        for c in ["h", "l"]:
            rs = [r for r in regions.split("/") if c in r]
            sub_tag = f"{scheme}:{'/'.join(rs)}"
            dfs[c] = _tag_to_df(
                i, pdb_file, results_dir, sub_tag, model_tag, fixed_length, 
                extra_tag=c
            )

        df = dfs["h"]
        df["vl_sample"] = dfs["l"]["vl_sample"]
        df["vl_mask"] = dfs["l"]["vl_mask"] 
        df["sample_tag"] = scheme + "_" + regions.replace("/", "_")

    else:
        df = _tag_to_df(
            i, pdb_file, results_dir, sample_tag, model_tag, fixed_length
        )

    return df

if __name__ == "__main__":

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

    model_tag = 'rfdiffusion'
    tags = make_tags(numbering_schemes, cdr_combos)
    pdb_dir = "/home/nvg7279/src/seq-struct/poas_seed_pdbs"
    results_dir = "/scratch/nvg7279/rfdiffusion_results"

    pdb_files = []
    for i in range(0, 10):
        pdb_file = os.path.join(pdb_dir, f"{i}.pdb")
        new_pdb_file = os.path.join(pdb_dir, f"{i}_renum.pdb")
        # renumber_pdb(pdb_file, new_pdb_file)
        pdb_files.append(new_pdb_file)

    dfs = []
    for fixed_length in [True]:#[True, False]:
        full_tag = f"{model_tag}_{'fixed' if fixed_length else 'variable'}"

        for tag in tags:
            for i, pdb_file in enumerate(pdb_files):
                df = tag_to_df(
                    i, pdb_file, results_dir, tag, full_tag, fixed_length
                )
                print(df)
                dfs.append(df)

    df = pd.concat(dfs)
    df.to_csv(os.path.join(results_dir, f"{model_tag}_results.csv"))
