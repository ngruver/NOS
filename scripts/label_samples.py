import os
import glob
import random
import argparse
from seq_models.compare_samples import label_csv

def main(input_dir, structure_labels=False):
    fns = glob.glob(os.path.join(input_dir, "*samples.csv"))
    random.shuffle(fns)

    for input_fn in fns:
        results_fn = os.path.join(
            os.path.dirname(input_fn), 
            f"{os.path.basename(input_fn).split('.')[0]}_labeled.csv",
        )

        if os.path.exists(results_fn):
            print(f"Skipping {input_fn}")
            continue

        label_csv(input_fn, structure_labels=structure_labels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add input file path")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input directory")
    parser.add_argument("-s", "--structure_labels", type=int, default=0, help="Whether to compute structure labels")

    args = parser.parse_args()

    main(args.input, args.structure_labels > 0)
