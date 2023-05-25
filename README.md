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
In order to used diffusion models with fixed dimension, we align the labeled sequences using a [wrapper script](https://github.com/ngruver/seq-struct/blob/main/scripts/data/align_igfold_poas.py) around ANARCI:
```
PYTHONPATH="." python scripts/data/align_igfold_poas.py
```

For infilling-based sampling, our scripts expect space separated sequences with "\[MASK\]" denoting the infilling locations. An example can be found in the [test infill seed file](https://github.com/ngruver/seq-struct/blob/main/infill_test_seeds.txt). 

## Basic Usage

To train a sequence diffusion model without a discriminative head, you can run
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
For the Gaussian corruption process (i.e. model="gaussian"), the additional argument should be set carefully:
```
  model.noise_schedule.noise_scale=[NOISE SCALE] \
```
This parameter effects the variance of the noise applied to the token embeddings. The noise schedule is unchanged, but the variance at each forward step, and the corresponding prior, is scaled multiplicatively. Reasonable defaults are in the range \[2, 10\]. 

To train a model with a discriminative head, the following additional arguments are necessary:

```
  'target_cols=[[OBJECTIVE NAME 1], ..., [OBJECTIVE NAME K]]' \
  discr_batch_ratio=4 \
```


model.noise_schedule.noise_scale=5 \

To perform basic sampling from a model, you can run
```
PYTHONPATH="." python scripts/sample.py
```

Guidance

## Infilling Experiments



## Datasets

Expects protein sequences with spaces
