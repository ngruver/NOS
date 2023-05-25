import os
import time
import tqdm
import wandb
import hydra
import numpy as np

from scipy.stats import spearmanr

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

import pytorch_lightning as pl

from transformers import AutoConfig
from transformers.models.bert.modeling_bert import (
    BertPreTrainedModel,
    BertPooler,
    BertEncoder,
    BertEmbeddings,
    BertOnlyMLMHead,
)

from seq_models.trainer import BaseModel
from seq_models.nets.regression import (
    RegressionHead,
    RegressionModel,
)
from seq_models.nets.util import timestep_embedding

class MLMDiffusionTransformer(nn.Module):

    def __init__(
        self,
        vocab_size,
        dropout=0,
        bert_config_name='bert-base-uncased',
        target_channels=2,
        discr_stop_grad=True,
    ):
        super().__init__()

        config = AutoConfig.from_pretrained(bert_config_name)
        config.hidden_dropout_prob = dropout
        config.vocab_size = vocab_size
        # config.hidden_size = 512

        self.target_channels = target_channels
        self.dropout = dropout
        self.vocab_size = vocab_size

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.cls = BertOnlyMLMHead(config)

        # self.cls.predictions.decoder.weight = self.embeddings.word_embeddings.weight

        self.time_embed_dim = config.hidden_size
        self.time_embed = nn.Sequential(
            nn.Linear(config.hidden_size, 4 * config.hidden_size),
            nn.SiLU(),
            nn.Linear(4 * config.hidden_size, config.hidden_size),
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if target_channels > 0:
            self.regression_head = RegressionHead(
                config, 
                target_channels, 
                stop_grad=discr_stop_grad
            )

    def forward(
        self, 
        corrupted_ids,
        timesteps, 
        attn_mask=None,
        token_embed=None,
    ):
        if token_embed is None:
            token_embed = self.embeddings(input_ids=corrupted_ids)

        time_embed = self.time_embed(timestep_embedding(timesteps, self.time_embed_dim))
        time_embed = time_embed.unsqueeze(1).expand(-1, token_embed.size(1), -1)
        embed = self.dropout(self.LayerNorm(token_embed + time_embed))

        sequence_output = self.encoder(embed, encoder_attention_mask=attn_mask)[0]
        prediction_scores = self.cls(sequence_output)

        out = {
            "logits": prediction_scores,
            "sequence_output": sequence_output,
            "embeds": token_embed,
        }

        return out

    def get_labels(
        self, 
        input_ids, 
        timesteps, 
        attn_mask=None, 
        sequence_output=None
    ):
        if sequence_output is None:
            sequence_output = self.forward(input_ids, timesteps, attn_mask)['sequence_output']
        return self.regression_head(sequence_output)

    def guidance_score(
        self, 
        input_ids, 
        timesteps, 
        attn_mask=None, 
        sequence_output=None
    ):
        labels = self.get_labels(input_ids, timesteps, attn_mask, sequence_output)
        return labels.sum(-1) 


class MLMDiffusion(BaseModel):

    def __init__(
        self,
        network,
        noise_schedule,
        optimizer,
        lr_scheduler,
    ):
        super().__init__()

        self.network = hydra.utils.instantiate(network)
        self.noise_schedule = hydra.utils.instantiate(noise_schedule)
        self.opt = hydra.utils.instantiate(optimizer, params=self.parameters())
        self.lr_scheduler = None
        if lr_scheduler:
            self.lr_scheduler = hydra.utils.instantiate(lr_scheduler, self.opt)

    def freeze_for_discriminative(self):
        for _, p in enumerate(self.network.parameters()):
            p.requires_grad_(False)

        for _, p in enumerate(self.network.regression_head.parameters()):
            p.requires_grad_(True)

    def forward(
        self,
        input_ids,
        corrupt_mask,
        attn_mask,
        labels=None,
        return_by_timestep=False,
    ):
        timesteps = self.noise_schedule.timesteps
        t = torch.randint(
            timesteps, 
            size=(input_ids.shape[0],),
            device=input_ids.device,
            dtype=torch.int64,
        )

        corrupt_ids, corrupt_mask = (
            self.noise_schedule.corrupt(input_ids, t, corrupt_mask)
        )

        model_output = self.network(
            corrupt_ids,
            t, 
            attn_mask,
        )
        logits = model_output['logits']
        hiddens = model_output['sequence_output']
        
        loss_fct = nn.CrossEntropyLoss(reduction='none')  # -100 index = padding token
        nll = loss_fct(logits.view(-1, logits.shape[-1]), input_ids.view(-1))
        nll = nll.view(*input_ids.shape[:2])

        loss_mask = attn_mask * corrupt_mask

        denom = loss_mask.sum(dim=-1)
        denom[denom == 0] = 1

        nll = (nll * loss_mask).sum(dim=-1) / denom
        accuracy = ((logits.argmax(-1) == input_ids) * loss_mask).sum(dim=-1) / denom
        loss = nll.mean()

        out = {}
        out["loss"] = loss.mean()
        out["nll"] = nll.mean()
        out["accuracy"] = accuracy.mean()

        if labels is not None:
            pred_labels = self.network.regression_head(hiddens.detach())
            regression_loss = (pred_labels - labels).pow(2)
            out["regression_mse"] = regression_loss.mean()
            out["regression_spearman"] = spearmanr(
                pred_labels[:,0].detach().cpu().numpy(),  
                labels[:,0].detach().cpu().numpy(),
            ).correlation

        if not return_by_timestep:
            return out
        
        num_buckets = 4
        step_size = timesteps // num_buckets
        for t_lower in np.arange(0, timesteps, step_size):
            t_upper = t_lower + step_size
            t_mask = (t > t_lower) * (t < t_upper)
            
            tag = f"accuracy_{t_lower}-{t_upper}"
            out[tag] = accuracy[t_mask].mean()
            
            if labels is not None:
                tag = f"regression_mse_{t_lower}-{t_upper}"
                out[tag] = regression_loss[t_mask].mean()

        return out


    def guidance_steps(
        self,
        model_output,
        t,
        attn_mask,
        infill_mask,
        guidance_layer="first",
        step_size=0.1,
        stability_coef=1e-2,
        num_steps=5,
    ):
        kl_loss = torch.nn.KLDivLoss(log_target=True)

        logits = model_output['logits']
        if guidance_layer == "last":
            h = model_output['sequence_output']
        elif guidance_layer == "first":
            h = model_output['embeds']
        else:
            raise NotImplementedError()
        
        delta = torch.nn.Parameter(torch.zeros_like(h), requires_grad=True)
        optimizer = torch.optim.Adagrad([delta], lr=step_size)
        
        with torch.enable_grad():
            for _ in range(num_steps):
                h_current = h + infill_mask.unsqueeze(-1) * delta

                if guidance_layer == "last":
                    target_loss = self.network.guidance_score(
                        None, t, attn_mask, sequence_output=h_current
                    ).sum()
                    new_logits = self.network.cls(h_current)
                elif guidance_layer == "first":
                    out = self.network.forward(
                        None, t, attn_mask, token_embed=h_current
                    )
                    target_loss = self.network.guidance_score(
                        None, t, attn_mask, sequence_output=out['sequence_output']
                    ).sum()
                    new_logits = out['logits']

                kl = kl_loss(new_logits, logits)
                loss = -target_loss + stability_coef * kl
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # delta_grad_norm = delta.grad.norm(dim=(2), keepdim=True)
                # delta.grad /= delta_grad_norm.clamp_min(1e-7)
                # if i < 100:
                #     print(target_loss)
                #     print(sample_loss)
                #     print(delta.pow(2).mean())
                #     print("")

                # eps = torch.randn_like(feature_grad) / grad_norm.clamp_min(1e-7)
                # new_features.add_(
                #     eps, alpha=math.sqrt(2 * guidance_step_size * langevin_noise_scale)
                # )
                
                # print(target_loss.item(), kl.item())

                # store best seqs?
            
            # print("\n")

        logits = self.network.cls(h + delta.data)
        return logits


    def sample(
        self,
        infill_seed,
        infill_mask,
        corrupt_mask,
        num_samples,
        guidance_kwargs=None,
        bad_word_ids=None,
    ):
        device = next(self.parameters()).device
        
        infill_mask = infill_mask[None,:]
        corrupt_mask = corrupt_mask[None,:]
        gt_vals = infill_seed[None]

        indices = list(range(self.noise_schedule.timesteps))[::-1]

        t = torch.tensor([indices[0]], device=device)
        noisy_gt = self.noise_schedule.corrupt(gt_vals, t)[0]
        noisy_gt = torch.where(corrupt_mask, noisy_gt, gt_vals)
        
        shape = (num_samples, infill_seed.shape[0])
        x = self.noise_schedule.sample_prior(shape, device)
        x = torch.where(infill_mask, x, noisy_gt)
        attn_mask = torch.ones_like(infill_mask, dtype=torch.bool)

        return_best = guidance_kwargs.pop("return_best", False) \
            if guidance_kwargs is not None else False

        traj = []
        for i in tqdm.tqdm(indices):
            t = torch.tensor([i] * shape[0], device=device)

            with torch.no_grad():
                model_output = self.network(x, t, attn_mask)
            
            logits = model_output["logits"]
            
            if guidance_kwargs is not None:
                logits = self.guidance_steps(
                    model_output, t, attn_mask, infill_mask, 
                    **guidance_kwargs
                )

            if bad_word_ids is not None:
                logits[:, :, bad_word_ids] = -1e9
                
            if logits.shape[2] == 30:    
                logits[:,:,29] = -1e9

            x = Categorical(logits=logits).sample()
            clean_x = x.clone()

            if i != indices[-1]:
                x = self.noise_schedule.corrupt(x, t, infill_mask)[0]

                noise_t = torch.tensor([i-1] * shape[0], device=device)
                noisy_gt = self.noise_schedule.corrupt(gt_vals, noise_t[:1])[0]
                noisy_gt = torch.where(corrupt_mask.bool(), noisy_gt, gt_vals) #TODO: clean up
                x = torch.where(infill_mask, x, noisy_gt)

            pred_ids = torch.where(infill_mask.squeeze(-1), clean_x, infill_seed[None])

            if guidance_kwargs is not None:
                labels = self.network.guidance_score(pred_ids, t, attn_mask).cpu().numpy()
                pred_ids = (pred_ids.cpu().numpy(), labels)
            else:
                pred_ids = pred_ids.cpu().numpy()

            traj.append(pred_ids)

        if return_best:
            best_idx = np.argmax(np.stack([t[1] for t in traj], axis=1), axis=1)
            samples = np.stack([
                traj[idx][0][i] for i, idx in enumerate(best_idx)
            ], axis=0)
        else:
            samples = traj[-1][0]

        return samples
    
    def sample_autoregressive(
        self,
        infill_seed,
        infill_mask,
        corrupt_mask,
        num_samples,
        guidance_kwargs=None,
        bad_word_ids=None,
    ):
        device = next(self.parameters()).device
        
        infill_mask = infill_mask[None,:]
        corrupt_mask = corrupt_mask[None,:]
        gt_vals = infill_seed[None]
 
        unmasked_indices = torch.nonzero(infill_mask[0])[:,0].cpu().numpy()
        num_autoregessive_steps = len(unmasked_indices)

        shape = (num_samples, infill_seed.shape[0])

        T = self.noise_schedule.timesteps - 1
        T = torch.tensor([T] * shape[0], device=device)

        noisy_gt = self.noise_schedule.corrupt(gt_vals, T[:1])[0]
        noisy_gt = torch.where(corrupt_mask, noisy_gt, gt_vals)
        
        x = self.noise_schedule.sample_prior(shape, device)
        x = torch.where(infill_mask, x, noisy_gt)
        attn_mask = torch.ones_like(infill_mask, dtype=torch.bool)

        # tokenizer = transformers.BertTokenizerFast(
        #     vocab_file="/home/nvg7279/src/seq-struct/vocab.txt", 
        #     do_lower_case=False,
        # )

        return_best = guidance_kwargs.pop("return_best", False) \
            if guidance_kwargs is not None else False

        traj = []
        for i, idx in tqdm.tqdm(enumerate(unmasked_indices), total=len(unmasked_indices)):

            perc = float(i) / num_autoregessive_steps
            timestep = np.argmin(np.abs(self.noise_schedule.mask_rates - perc))
            t = torch.tensor([timestep] * shape[0], device=device)

            with torch.no_grad():
                model_output = self.network(x, t, attn_mask)
            
            logits = model_output["logits"]
            
            if guidance_kwargs is not None:
                logits = self.guidance_steps(
                    model_output, t, attn_mask, infill_mask, 
                    **guidance_kwargs
                )
    
            if bad_word_ids is not None:
                logits[:, :, bad_word_ids] = -1e9

            if logits.shape[2] == 30:    
                logits[:,:,29] = -1e9

            autoregressive_mask = torch.clone(infill_mask)
            autoregressive_mask[:,:idx] = 0

            sample_x = Categorical(logits=logits).sample()
            x = torch.where(autoregressive_mask, sample_x, x) #don't resample AR
            clean_x = x.clone()

            if i != len(unmasked_indices) - 1:
                autoregressive_mask[:,:idx+1] = 0
                x = self.noise_schedule.corrupt(x, T, autoregressive_mask)[0]

                noise_t = torch.tensor([timestep-1] * shape[0], device=device)
                noise_t = torch.maximum(noise_t, torch.zeros_like(noise_t))

                noisy_gt = self.noise_schedule.corrupt(gt_vals, noise_t[:1])[0]
                noisy_gt = torch.where(corrupt_mask.bool(), noisy_gt, gt_vals) #TODO: clean up
                x = torch.where(infill_mask, x, noisy_gt)
                
            pred_ids = torch.where(infill_mask.squeeze(-1), clean_x, infill_seed[None])

            if guidance_kwargs is not None:
                labels = self.network.guidance_score(pred_ids, t, attn_mask).cpu().numpy()
                pred_ids = (pred_ids.cpu().numpy(), labels)
            else:
                pred_ids = pred_ids.cpu().numpy()

            traj.append(pred_ids)

        if return_best:
            best_idx = np.argmax(np.stack([t[1] for t in traj], axis=1), axis=1)
            samples = np.stack([
                traj[idx][0][i] for i, idx in enumerate(best_idx)
            ], axis=0)
        else:
            samples = traj[-1][0]

        return samples
