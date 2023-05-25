import os
import sys
import pprint
from pathlib import Path
import hydra
import torch
import wandb
import warnings
from omegaconf import OmegaConf

import seq_models.metrics as metrics
from scripts.utils import flatten_config

from seq_models.trainer import sample_model

@hydra.main(config_path="../configs", config_name="sample")
def main(config):
    Path(config.exp_dir).mkdir(parents=True, exist_ok=True)

    log_config = flatten_config(OmegaConf.to_container(config, resolve=True), sep='/')
    log_config = {'/'.join(('config', key)): val for key, val in log_config.items()}
    wandb.init(
         project="guided_protein_seq",
         config=log_config,
    )

    pprint.pprint(dict(config))
    
    model = hydra.utils.instantiate(config.model)
    
    if config.ckpt_path is not None:
        state_dict = torch.load(config.ckpt_path)['state_dict']
        model.load_state_dict(state_dict)

    if torch.cuda.is_available():
        model.cuda()

    model.eval()

    results = sample_model(
        model,
        num_samples=config.num_samples,
        infill_seed_file=config.infill_seeds_fn,
        vocab_file=config.vocab_file,
        gt_data_file=os.path.join(config.data_dir, config.val_fn),
        guidance_kwargs=config.guidance_kwargs,
    )

    pprint.pprint(results)

    wandb.log(results)

if __name__ == "__main__":    
    main()
    sys.exit()