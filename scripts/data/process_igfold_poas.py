# code to label the predicted structures 
# structure are retrieved from
# https://data.graylab.jhu.edu/OAS_paired.tar.gz

import os
import glob
import argparse
import pandas as pd
from p_tqdm import p_map

import anarci as an
from Bio import SeqIO

from seq_models.metrics import (
    BioPythonSeqLabeler,
    BioPythonStructLabeler,
)

ALLOWED_SPECIES = ["rabbit", "rat", "mouse", "human", "rhesus", "camel"]


def parse_fasta(fasta_file):
    out = {}
    fasta_sequences = SeqIO.parse(open(fasta_file),'fasta')
    for fasta in fasta_sequences:
        chain_id = fasta.id.split(":")[-1]
        out[chain_id] = str(fasta.seq)
    return out

def align_with_anarci(chain_sequence):
    # Species currently restricted to the most common in our affinity set,
    # but can expand this as appropriate
    renumbered_aho = list(
        an.run_anarci(
            seq=chain_sequence,
            scheme="aho",
            assign_germline=True,
            allowed_species=ALLOWED_SPECIES,
            ncpu=1,#os.cpu_count(),
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

def process_protein(data_dir, protein_id, verbose=False):
    pdb_file = os.path.join(data_dir, f"{protein_id}.pdb")
    seq_file = os.path.join(data_dir, f"{protein_id}.fasta")

    try:
        seq_labeler = BioPythonSeqLabeler(use_lm=False)
        seqs = parse_fasta(seq_file)
        seq = seqs['H'] + seqs['L']
        seq_labels = seq_labeler.label_seq(seq)

        struct_labeler = BioPythonStructLabeler()
        struct_labels = struct_labeler.label_pdb(pdb_file)

        labels = {**seq_labels, **struct_labels}
        
        labels['HeavyAA'] = seqs['H']
        labels['LightAA'] = seqs['L']
        labels['igfold_id'] = protein_id

        labels['HeavyAA_aligned'] = align_chain(seqs['H'], 'H')
        labels['LightAA_aligned'] = align_chain(seqs['L'], 'L')

        labels['full_seq'] = " ".join(
            ["[AbHC]"] + list(labels['HeavyAA_aligned']) + \
            ["[AbLC]"] + list(labels['LightAA_aligned']) + ["[Ag]"]
        )

    except Exception as e:
        if verbose:
            print(e)
        return {}

    return labels

def get_ids(data_dir):
    fasta_files = glob.glob(os.path.join(data_dir, '*.fasta'))
    filenames = [os.path.basename(fn) for fn in fasta_files]
    ids = [fn.split(".")[0] for fn in filenames]
    return ids

def get_args_parser():
    parser = argparse.ArgumentParser(description="Labeling IgFold POAS")

    parser.add_argument(
        '--data_dir', default='/igfold_poas', help='directory containing the data'
    )
    parser.add_argument(
        '--output_dir', default='.', help='output directory'
    )
    parser.add_argument(
        '--verbose', type=int, default=0, help='print error messages'
    )
    return parser

def main(args):
    ids = get_ids(args.data_dir)[:100]
    process_fn = lambda x: process_protein(args.data_dir, x, verbose=args.verbose)
    labels = p_map(process_fn, ids)
    df = pd.DataFrame([a for a in labels if len(a) > 0])
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    df.to_csv(os.path.join(args.output_dir, "igfold_labeled.csv"), index=False)
    print(df)

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)