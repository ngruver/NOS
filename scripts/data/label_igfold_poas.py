# code to label the predicted structures 
# structure are retrieved from
# https://data.graylab.jhu.edu/OAS_paired.tar.gz

import os
import glob
import tqdm
import pandas as pd
import multiprocessing 

import anarci as an
from Bio import SeqIO

from seq_models.metrics import (
    BioPythonSeqLabeler,
    BioPythonStructLabeler,
)

# DATA_DIR = "/predictions_flat"
DATA_DIR = "/igfold_poas"
ALLOWED_SPECIES = ["rabbit", "rat", "mouse", "human"]#, "rhesus", "camel"]

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
    print(species)
    if species not in ALLOWED_SPECIES:
        # print(seq)
        raise Exception(f"Skipping because species is {species}")

    aligned = "".join([x[1] for x in alignment[1][0][0]])
    if len(aligned) != 149 and chain_id == 'H':
        raise Exception(f"Skipping because length is {len(aligned)}")
    if len(aligned) != 148 and chain_id == 'L':
        raise Exception(f"Skipping because length is {len(aligned)}")

    return aligned

def process_protein(protein_id):
    pdb_file = os.path.join(DATA_DIR, f"{protein_id}.pdb")
    seq_file = os.path.join(DATA_DIR, f"{protein_id}.fasta")

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

        labels['vh_seq_aho'] = align_chain(seqs['H'], 'H')
        labels['vl_seq_aho'] = align_chain(seqs['L'], 'L')

    except Exception as e:
        print(e)
        return {}

    return labels

def get_ids():
    fasta_files = glob.glob(os.path.join(DATA_DIR, '*.fasta'))
    filenames = [os.path.basename(fn) for fn in fasta_files]
    ids = [fn.split(".")[0] for fn in filenames]
    return ids

def main():
    ids = get_ids()

    pool = multiprocessing.Pool(processes=1)
    labels = list(tqdm.tqdm(
        pool.imap(
            process_protein, ids, chunksize=10
        ), total=len(ids)
    ))
    pool.close()
    pool.join()

    df = pd.DataFrame([a for a in labels if len(a) > 0])

    df.to_csv("igfold_jaffe_processed.csv", index=False)
    print(df)

if __name__ == "__main__":
    main()