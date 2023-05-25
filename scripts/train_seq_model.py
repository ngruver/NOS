import os
import sys
import pprint
from pathlib import Path
import hydra
import torch
import wandb
import warnings
from omegaconf import OmegaConf

from seq_models.trainer import get_trainer
from seq_models.data import get_loaders
from scripts.utils import (
    flatten_config, 
    convert_to_dict,
)

@hydra.main(config_path="../configs", config_name="train_seq_model")
def main(config):
    Path(config.exp_dir).mkdir(parents=True, exist_ok=True)

    log_config = flatten_config(OmegaConf.to_container(config, resolve=True), sep='/')
    log_config = {'/'.join(('config', key)): val for key, val in log_config.items()}
    wandb.init(
         project="guided_protein_seq",
         config=log_config,
    )

    # there must be a way to get hydra to do this for me
    if config.target_cols and len(config.target_cols) > 0:
        config.model.network.target_channels = len(config.target_cols)
    
    pprint.PrettyPrinter(depth=4).pprint(convert_to_dict(config))

    model = hydra.utils.instantiate(config.model, _recursive_=False)
    if config.discr_batch_ratio is not None:
        model.discr_batch_ratio = config.discr_batch_ratio

    train_dl, valid_dl = get_loaders(config)
    trainer = get_trainer(config, len(train_dl))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # catch really annoying BioPython warnings
        
        trainer.fit(
            model=model,
            train_dataloaders=train_dl,
            val_dataloaders=valid_dl,
            ckpt_path=config.resume_ckpt
        )

if __name__ == "__main__":    
    main()
    sys.exit()