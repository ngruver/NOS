import os
import re
import wandb
import torch
import tempfile
import numpy as np
import pandas as pd
from collections import defaultdict

from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley
from Bio.PDB.ResidueDepth import get_surface, residue_depth

from Bio.SeqUtils.ProtParam import ProteinAnalysis

import transformers
from transformers import AutoTokenizer
from transformers import GPT2LMHeadModel

from seq_models.data import random_sequences

class BioPythonSeqLabeler():

    def __init__(self, use_lm=True):
        self.use_lm = use_lm
        if use_lm:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.tokenizer = AutoTokenizer.from_pretrained('nferruz/ProtGPT2') # replace with the actual path
            self.model = GPT2LMHeadModel.from_pretrained('nferruz/ProtGPT2').to(device) 

    def lm_wrapper(self, samples):
        ppls = []
        for seq in samples:
            out = self.tokenizer(seq, return_tensors="pt")
            input_ids = out.input_ids.cuda()

            with torch.no_grad():
                outputs = self.model(input_ids, labels=input_ids)

            ppl = (outputs.loss * input_ids.shape[1]).item()
            ppls.append(ppl)
        
        ppls = np.array(ppls)
        return ppls

    def label_seq(self, seq):  
        X = ProteinAnalysis(str(seq))

        label_names = [
            "molecular_weight",
            "aromaticity",
            "instability_index",
            "isoelectric_point",
            "avg_flexibility",
            "gravy"
        ]
        label_funcs = [
            X.molecular_weight,
            X.aromaticity,
            X.instability_index,
            X.isoelectric_point,
            lambda: np.mean(X.flexibility()),
            X.gravy
        ]

        seq_labels = {
            "seq": seq,
            "length": len(seq), 
        }
        for name, f in zip(label_names, label_funcs):
            try:
                seq_labels[name] = f()
            except Exception as e:
                continue

        ss_types = ["helix", "turn", "sheet"]
        ss_frac = X.secondary_structure_fraction()
        seq_labels.update({
            f"ss_perc_{sstype}": perc for sstype, perc in zip(ss_types, ss_frac)
        })

        aa_breakdown = X.get_amino_acids_percent()
        aa_types = aa_breakdown.keys()
        aa_frac = aa_breakdown.values()
        seq_labels.update({
            f"aa_perc_{aatype}": perc for aatype, perc in zip(aa_types, aa_frac)
        })

        if self.use_lm:
            seq_labels.update({
                f"lm_nll": self.lm_wrapper([seq])[0]
            })

        return seq_labels

    def label_seqs(self, seqs):  
        return pd.DataFrame([self.label_seq(s) for s in seqs])


class BioPythonStructLabeler():
    
    def __init__(self):
        self.parser = PDBParser(QUIET=1)  
        self.sr = ShrakeRupley()

    def avg_res_depth(self, struct):
        surface = get_surface(struct)
        rd = 0.0
        for res in struct:
            rd += residue_depth(res, surface)
        return rd / float(len(struct))

    def label_pdb(self, pdb_file):
        struct = self.parser.get_structure("dummy", pdb_file)

        labels = {}
        try:
            self.sr.compute(struct, level="S")
            labels["sasa"] = struct.sasa
            # labels["avg_rd"] = self.avg_res_depth(struct)
        except Exception as e:
            pass

        return labels

class IgFoldWrapper():

    def __init__(self):
        from igfold import IgFoldRunner
        from igfold.refine.pyrosetta_ref import init_pyrosetta

        init_pyrosetta()
        self.igfold = IgFoldRunner()

    def fold_and_label(self, chains, struct_labeler):        
        pdb_fn = tempfile.NamedTemporaryFile(suffix='.pdb').name
        self.igfold.fold(
            pdb_fn, # Output PDB file
            sequences=chains, # Antibody sequences
            do_refine=True, # Refine the antibody structure with PyRosetta
            do_renum=True, # Renumber predicted antibody structure (Chothia)
        )
        
        labels = struct_labeler.label_pdb(pdb_fn)

        return labels

def regression_labels(samples, model, vocab_file):
    tokenizer = transformers.BertTokenizerFast(
        vocab_file=vocab_file, 
        do_lower_case=False,
    )
    
    sample_tokens = [tokenizer.encode(s, add_special_tokens=False) for s in samples]
    sample_tokens = torch.stack([torch.from_numpy(np.array(t)) for t in sample_tokens], dim=0)
    embeds = model.get_embeds(sample_tokens.cuda())
    t = torch.tensor([0] * embeds.shape[0], device=embeds.device)
    attn_mask = torch.ones_like(sample_tokens, device=embeds.device)
    reg_labels = model.get_labels(embeds, t, attn_mask)

    targets = [f"target_{i}" for i in range(reg_labels.shape[1])]
    log = {}
    for labels, name in zip(reg_labels.permute(1,0), targets):
        # print(f"{name}: {labels.mean().item()}")
        log[name] = labels.data.cpu().numpy()

    return log

def make_wandb_log(log, log_prefix):
    wandb_log = {}
    for log_type in log:
        if log_type == "seq_samples":
            flat_k = f"{log_prefix}/{log_type}"
            wandb_log[flat_k] = wandb.Table(
                data=[[x] for x in log[log_type]], 
                columns=["Sampled Sequences"]
            )
        if log_type == 'biopython':
            for data_type in log[log_type]:
                for k in log[log_type][data_type]:
                    flat_k = f"{log_prefix}/{log_type}/{k}_{data_type}"
                    wandb_log[flat_k] = wandb.Histogram(
                        log[log_type][data_type][k]
                    )
        elif log_type == 'reg_label':
            for k in log[log_type]:
                flat_k = f"{log_prefix}/{log_type}/{k}"
                wandb_log[flat_k] = wandb.Histogram(
                    log[log_type][k]
                )

    return wandb_log

def evaluate_samples(
    samples, 
    infill_seed=None, 
    vocab_file=None, 
    gt_file=None,
    log_prefix="",
    reg_labels=False,
    model=None,
):  
    log = defaultdict(lambda: defaultdict(dict))
    log["seq_samples"] = samples

    # TODO: add more sanity checking here
    labeler = BioPythonSeqLabeler()
    s_for_labels = [re.sub("[\(\[].*?[\)\]]", "", s.replace(" ","")) for s in samples]
    s_for_labels = [s.replace("-", "") for s in s_for_labels]
    s_for_labels = [s for s in s_for_labels if len(s) > 0]
    
    if len(s_for_labels) == 0:
        return log

    samp_df = labeler.label_seqs(s_for_labels)
    
    gt_df = None if gt_file is None else pd.read_csv(gt_file)
    
    rand_df = None
    if vocab_file:
        lengths = [len(s) for s in s_for_labels]
        rand_df = labeler.label_seqs(
            random_sequences(len(samples), lengths, vocab_file)
        )
        
    cols = [c for c in samp_df if not "aa_perc" in c and c != "seq"]
    
    tags = ["samp", "gt", "rand"]
    dfs = [samp_df, gt_df, rand_df]
    for df, tag in zip(dfs, tags):
        if df is None:
            continue

        df_cols = [c for c in cols if c in df]
        for c in df_cols:
            log["biopython"][tag].update({c: df[c].dropna().to_numpy()})

    if reg_labels:
        labels = regression_labels(s_for_labels, model, vocab_file)
        log["reg_label"].update(labels)

    wandb_log = make_wandb_log(log, log_prefix)
    log = {log_prefix: log}

    return log, wandb_log

def main():
    pass

if __name__ == "__main__":
    main()
