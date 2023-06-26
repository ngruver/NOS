import os
import argparse
import pandas as pd
import numpy as np
from p_tqdm import p_map
from sklearn.model_selection import train_test_split

def check_if_in_test_set(protein_seq, test_set):
    for seq in test_set:
        hv, lv = seq.split("[AbLC]")
        hv = hv.replace("[AbHC]", "").strip()
        lv = lv.replace("[Ag]", "").strip()
        if hv in protein_seq or lv in protein_seq:
            return True
    return False

def get_args_parser():
    parser = argparse.ArgumentParser(description="Labeling IgFold POAS")

    parser.add_argument(
        '--data_fn', default='./igfold_labeled.csv', help='data file'
    )
    parser.add_argument(
        '--test_set_fn', default='./poas_seeds.txt', help='test set file'
    )
    parser.add_argument(
        '--output_dir', default='data_t', help='output directory'
    )
    return parser

def main(args):
    df = pd.read_csv(args.data_fn)
    test_set = pd.read_csv(args.test_set_fn, header=None).values[:,0]
    
    check_fn = lambda x: check_if_in_test_set(x, test_set)
    labels = np.array(p_map(check_fn, df['full_seq'].values))
    df = df[~labels]

    train, val = train_test_split(df, test_size=0.1)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for i, chunk in enumerate(np.array_split(train, 10)):
        chunk.to_csv(os.path.join(args.output_dir, f"train_{i}.csv"), index=False)

    val.to_csv(os.path.join(args.output_dir, "val_iid.csv"), index=False)

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)