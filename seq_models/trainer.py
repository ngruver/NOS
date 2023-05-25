import os
import time
import torch
import wandb

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
# from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import Callback

from seq_models.sample import sample_model

class BaseModel(pl.LightningModule):

    def __init__(self):
        super().__init__()

        self.discr_batch_ratio = None
        self.train_epoch_counter = 0
        self.train_epoch_last_time = time.time()

    def training_step(self, batch, batch_idx):        
        out = self.forward(
            batch["seq"],
            batch["corrupt_mask"],
            batch["attn_mask"],
            labels=batch["labels"] if "labels" in batch else None,
            return_by_timestep=True,
        )        

        log_dict = {f"train_{k}" : v for k, v in out.items()}
        self.log_dict(log_dict)  # Don't seem to need rank zero or sync dist

        if "labels" in batch:
            if self.discr_batch_ratio is None:
                out["loss"] = out["loss"] + out["regression_mse"]
            elif batch_idx % self.discr_batch_ratio == 0:
                out["loss"] = out["regression_mse"]
        
        return out["loss"]

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            out = self.forward(
                batch["seq"],
                batch["corrupt_mask"],
                batch["attn_mask"],
                labels=batch["labels"] if "labels" in batch else None,
                return_by_timestep=True,
            )
                
        log_dict = {f"val_{k}" : v for k, v in out.items()}
        self.log_dict(log_dict, rank_zero_only=True)

        return {"val_loss": out['loss']}

    def configure_optimizers(self):
        config = {
            "optimizer": self.opt
        }

        if self.lr_scheduler is not None:
            self.lr_scheduler.step() #avoid lr=0 at start for warmup

            config["lr_scheduler"] = {
                "scheduler": self.lr_scheduler,
                "frequency": 1,
                "interval": "epoch",    # Call after 1 epoch
            }

        return config
    

class SampleEvaluationCallback(Callback):

    def __init__(
        self,
        config,
    ):
        super().__init__()

        self.vocab_file=config.vocab_file
        self.gt_data_file=os.path.join(config.data_dir, config.val_fn)
        self.infill_seed_file=config.infill_seeds_fn
        self.num_samples=config.num_samples
        self.sample_frequency=config.val_sample_frequency
        self.guidance_kwargs=config.guidance_kwargs
        self.use_alignment_tokens=config.use_alignment_tokens
        self.autoregressive_sample=config.autoregressive_sample

    def on_validation_epoch_end(self, trainer, pl_module):
        if pl_module.current_epoch == 0:
            return
        
        if pl_module.current_epoch % self.sample_frequency != 0:
            return
        
        pl_module.eval()

        _, log = sample_model(
            pl_module,
            self.num_samples,
            self.infill_seed_file,
            self.vocab_file,
            self.gt_data_file,
            self.guidance_kwargs,
            self.use_alignment_tokens,
            self.autoregressive_sample,
        )

        wandb.log(log)

def get_trainer(config, num_train_batches):
    os.makedirs(os.path.join(config.exp_dir, "models/best_by_valid"), exist_ok=True)
    os.makedirs(os.path.join(config.exp_dir, "models/best_by_train"), exist_ok=True)
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            monitor="val_loss",
            dirpath=os.path.join(config.exp_dir, "models/best_by_valid"),
            save_top_k=5,
            # save_weights_only=True,
            mode="min",
        ),
        pl.callbacks.ModelCheckpoint(
            monitor="train_loss",
            dirpath=os.path.join(config.exp_dir, "models/best_by_train"),
            save_top_k=5,
            # save_weights_only=True,
            mode="min",
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
        SampleEvaluationCallback(
            config,
        )
    ]

    wandb_logger = WandbLogger(project="guided_protein_seq", dir=config.exp_dir)

    accelerator, strategy = "cpu", None
    if torch.cuda.is_available():
        accelerator = "gpu"
        strategy = "ddp"
        # if torch.cuda.device_count() > 1:
        #     strategy = DDPStrategy(find_unused_parameters=False)

    trainer = pl.Trainer(
        default_root_dir=config.exp_dir,
        gradient_clip_val=config.gradient_clip,
        min_epochs=config.min_epochs,
        max_epochs=config.max_epochs,
        check_val_every_n_epoch=1,
        callbacks=callbacks,
        logger=wandb_logger,
        log_every_n_steps=min(200, num_train_batches),  # Log >= once per epoch
        accelerator=accelerator,
        strategy=strategy,
        devices=config.ngpu,
        enable_progress_bar=True,#False,
        # move_metrics_to_cpu=False,  # Saves memory
    )

    return trainer