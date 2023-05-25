# Protein Design with Guided Discrete Diffusion

## Installation
```
pip install -r requirements.txt
```

## Datasets

In order to obtain many SASA labels, we use [IgFold's archive of pre-computed structures on paired OAS (pOAS)](https://github.com/Graylab/IgFold#synthetic-antibody-structures). We extract the sequences and structures and labeled them by running the [labeling script](https://github.com/ngruver/seq-struct/blob/main/scripts/data/label_igfold_poas.py):
```
PYTHONPATH="." python scripts/data/label_igfold_poas.py
```
In order to used diffusion models with fixed dimension, we align the labeled sequences using a [wrapper script](https://github.com/ngruver/seq-struct/blob/main/scripts/data/align_igfold_poas.py) around ANARCI. 

For infilling-based sampling, our scripts expect sequences formatted 
Expects protein sequences with spaces

## Basic Usage

To train a sequence diffusion model, you can run
```
PYTHONPATH="." python scripts/train_seq_model.py \
  model=[MODEL TYPE] \
  model.optimizer.lr=[MODEL LR] \
  data_dir=[DATASET DIRECTORY] \
  train_fn=[TRAINING CSV FILE] \
  val_fn=[VALIDATION CSV FILE] \
  vocab_file=[VOCAB FILE IN THIS REPO'S BASE DIR] \
  log_dir=[LOGGING DIRECTORY]
```

'target_cols=["ss_perc_sheet"]' \
model.noise_schedule.noise_scale=5 \
discr_batch_ratio=4 \


To sample from a model, you can run
```
PYTHONPATH="." python scripts/sample.py
```

## Infilling Experiments



## Datasets

Expects protein sequences with spaces
