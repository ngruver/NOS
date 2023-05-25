import os
import tqdm
import pandas as pd
import multiprocessing

import anarci as an

ALLOWED_SPECIES = ["rabbit", "rat", "mouse", "human"]

def align_with_anarci(chain_sequence: str):
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

def align_chains(vals):
    try:
        aligned = []
        for s in [vals['HeavyAA'], vals['LightAA']]:
            alignment = align_with_anarci(s)
            species = alignment[2][0]['species']
            if species not in ALLOWED_SPECIES:
                print(f"Skipping {vals['igfold_id']} because species is {species}")
                return {}

            aligned.append(
                "".join([x[1] for x in alignment[1][0][0]])
            )

        vals['HeavyAA_aligned'] = aligned[0]
        vals['LightAA_aligned'] = aligned[1]
        return vals
    except Exception as e:
        print(e)
        return {}


def main():
    df = pd.read_csv(
        "/home/nvg7279/src/seq-struct/igfold_labeled.csv"
    )

    vals = df.to_dict('records')

    pool = multiprocessing.Pool(processes=16)
    aligned = list(tqdm.tqdm(
        pool.imap(
            align_chains, vals, chunksize=10
        ), total=len(vals)
    ))
    pool.close()
    pool.join()

    df = pd.DataFrame([a for a in aligned if len(a) > 0])

    df = df[
        (df['HeavyAA_aligned'].str.len() == 149) &
        (df['LightAA_aligned'].str.len() == 148)
    ]

    df.to_csv("igfold_aligned.csv", index=False)

    print(df)

if __name__ == "__main__":
    main()