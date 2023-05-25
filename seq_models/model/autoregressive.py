import os
import multiprocessing
import pickle
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple, Union
import pytorch_lightning as pl

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.distributions.categorical import Categorical
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

import transformers
from transformers import GPT2LMHeadModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

import iglm
from iglm import IgLM
from iglm.model.tokens import *
from iglm.model.utils import *
from iglm.utils.general import exists

from seq_models.util.numbering import (
    get_species,
)

def pplm_step(
    hidden_states, 
    # last_hiddens,
    # last_logits,
    lm_head,
    guidance_model,
    guidance_step_size,
    guidance_num_steps,
    guidance_stability_coef,
):
    kl_loss = torch.nn.KLDivLoss(log_target=True, reduction='batchmean')
    old_logits = lm_head(hidden_states[..., -1:, :])
    
    delta = torch.nn.Parameter(
        torch.zeros_like(hidden_states[..., -1:, :])
    )
    optimizer = torch.optim.Adagrad([delta], lr=guidance_step_size)
    
    with torch.enable_grad():
        for _ in range(guidance_num_steps):
            last_h = hidden_states[..., -1:, :] + delta
            
            all_h = torch.cat([
                hidden_states[..., :-1, :], last_h
            ], dim=-2)
                                
            new_logits = lm_head(last_h)
            
            kl = kl_loss(new_logits, old_logits)
            guide_loss = -guidance_model(all_h.mean(1)).sum()
            loss = guide_loss + guidance_stability_coef * kl
            
            # print(kl)
            # print(guide_loss)
            # print(loss)
            # print("********")

            optimizer.zero_grad()
            loss.backward()
            # delta_grad_norm = delta.grad.norm(keepdim=True)
            # delta.grad /= delta_grad_norm.clamp_min(1e-7)
            # delta.grad *= math.sqrt(h.size(-2))
            optimizer.step()
            
    # print("\n")

    hidden_states[..., -1:, :] += delta.data
    
    return hidden_states

class GuidedModel(GPT2LMHeadModel):
    
    def set_guidance_model(
        self, 
        guidance_model, 
        step_size=0.1,
        num_steps=5,
        stability_coef=0,
    ):
        # self.saved_guide_scores = None
        self.guidance_model = guidance_model
        self.guidance_step_size = step_size
        self.guidance_num_steps = num_steps
        self.guidance_stability_coef = stability_coef
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)
    
        # if self.saved_guide_scores is None:
        #     self.saved_guide_scores = []
            
        if self.guidance_model is not None:
            hidden_states = pplm_step(
                hidden_states,
                # self.last_hiddens.clone(),
                # self.last_logits,
                self.lm_head,
                self.guidance_model,
                self.guidance_step_size,
                self.guidance_num_steps,
                self.guidance_stability_coef,
            )
            
        lm_logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )
    
class GuidedIgLM():

    def __init__(self, model_name: str = "IgLM"):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        project_path = os.path.dirname(os.path.realpath(iglm.__file__))
        trained_models_dir = os.path.join(project_path, 'trained_models')
        ckpt = os.path.join(trained_models_dir, 'IgLM')
        
        self.model = GuidedModel.from_pretrained(ckpt).to(self.device)
        self.model.eval()

        vocab_file = os.path.join(trained_models_dir, 'vocab.txt')
        self.tokenizer = transformers.BertTokenizerFast(vocab_file=vocab_file,
                                                        do_lower_case=False)

    def _generate(
        self, 
        starting_tokens, 
        num_to_generate, 
        max_length, 
        min_new_tokens,
        max_new_tokens,
        top_p, 
        temperature
    ):
        decoded_seqs = set()  # Set to remove duplicates
        # pbar = tqdm(total=num_to_generate)
        for _ in range(10 * num_to_generate):
            if len(decoded_seqs) >= num_to_generate:
                break

            if max_new_tokens is not None:
                out = self.model.generate(
                    starting_tokens,
                    min_new_tokens=min_new_tokens,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.cls_token_id,
                    forced_eos_token_id=self.tokenizer.cls_token_id,
                    bad_words_ids=BAD_WORD_IDS,
                    do_sample=True,
                    use_cache=False,
                    top_p=top_p,
                    temperature=temperature,
                    output_scores=True,
                    output_hidden_states=True,
                    return_dict_in_generate=True
                )
            else:
                out = self.model.generate(
                    starting_tokens,
                    max_length=max_length,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.cls_token_id,
                    forced_eos_token_id=self.tokenizer.cls_token_id,
                    bad_words_ids=BAD_WORD_IDS,
                    do_sample=True,
                    use_cache=False,
                    top_p=top_p,
                    temperature=temperature,
                    output_scores=True,
                    output_hidden_states=True,
                    return_dict_in_generate=True
                )
            
            seq = out.sequences[0].detach().cpu().numpy()
            
            if not validate_generated_seq(seq, self.tokenizer):
                continue
                
            # print(self.tokenizer.decode(seq)) 
                
            decoded_tokens = self.tokenizer.decode(
                iglm_to_infilled(seq, self.tokenizer))
            decoded_seq = ''.join(decoded_tokens).replace(' ', '')
            if decoded_seq not in decoded_seqs:
                decoded_seqs.add(decoded_seq)
                # pbar.update(1)

        # pbar.close()
        return list(decoded_seqs)

    def generate(self,
                 chain_token,
                 species_token,
                 prompt_sequence=None,
                 num_to_generate=1000,
                 top_p=1,
                 temperature=1):
        start_tokens = [chain_token, species_token]

        if exists(prompt_sequence):
            prompt_tokens = list(prompt_sequence)
            start_tokens += prompt_tokens

        start_tokens = torch.Tensor([
            self.tokenizer.convert_tokens_to_ids(start_tokens)
        ]).int().to(self.device)

        assert (start_tokens != self.tokenizer.unk_token_id
                ).all(), "Unrecognized token supplied in starting tokens"

        generated_seqs = self._generate(
            start_tokens,
            num_to_generate=num_to_generate,
            top_p=top_p,
            temperature=temperature,
        )

        return generated_seqs
    
    def infill(
        self,
        sequence,
        chain_token,
        species_token,
        infill_range,
        num_to_generate=1000,
        max_length=150,
        min_new_tokens=None,
        max_new_tokens=None,
        top_p=1,
        temperature=1,
    ):
        sequence = list(sequence)
        masked_seq = mask_span(
            sequence,
            infill_range[0],
            infill_range[1],
        )  # mask using provided indices
        start_tokens = [chain_token, species_token] + masked_seq
        start_tokens = torch.Tensor([
            self.tokenizer.convert_tokens_to_ids(start_tokens)
        ]).int().to(self.device)

        assert (start_tokens != self.tokenizer.unk_token_id
                ).all(), "Unrecognized token supplied in starting tokens"

        generated_seqs = self._generate(
            start_tokens,
            num_to_generate=num_to_generate,
            max_length=max_length,
            min_new_tokens=min_new_tokens,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            temperature=temperature,
        )

        return generated_seqs
    




SPECIES_TO_TOKEN = {
    "camel": "[CAMEL]",
    "alpaca": "[CAMEL]",
    "human": "[HUMAN]",
    "mouse": "[MOUSE]",
    "rabbit": "[RABBIT]",
    "rat": "[RAT]",
    "rhesus": "[RHESUS]",
}

CHAIN_TOKENS = {
    "vh": "[HEAVY]",
    "vl": "[LIGHT]",
}

class Dataset(Dataset):
    def __init__(self, X, y):
        """Reads source and target sequences from txt files."""
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        data = {}
        data["X"] = self.X[index]
        data["y"] = self.y[index]
        return data
    
class ClassificationHead(pl.LightningModule):
    """Classification Head for transformer encoders"""

    def __init__(self, embed_size=512, target_size=1):
        super(ClassificationHead, self).__init__()
        self.embed_size = embed_size
        self.target_size = target_size
        # self.lin = torch.nn.Linear(embed_size, target_size)

        self.regression_head = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.Tanh(),
            nn.Linear(4 * embed_size, target_size)
        )

    def forward(self, hidden_state):
        return self.regression_head(hidden_state)
        # return self.lin(hidden_state)
    
    def training_step(self, batch, batch_idx):
        batch = {k: v.cuda() for k,v in batch.items()}
        pred = self.forward(batch["X"]).flatten()
        loss = (pred - batch["y"]).pow(2).mean()
        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx):
        batch = {k: v.cuda() for k,v in batch.items()}
        with torch.no_grad():
            pred = self.forward(batch["X"]).flatten()
        loss = (pred - batch["y"]).pow(2).mean()
        return {"loss": loss}
    
    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(),
            lr=1e-3,
            # weight_decay=self.l2_lambda,
        )
        return {"optimizer": optim}

def get_species_token(sequence):
    species = get_species(sequence)
    if species not in SPECIES_TO_TOKEN:
        return None
    species_token = SPECIES_TO_TOKEN[species]
    return (sequence, species_token)

def get_iglm_embeddings(
    seqs, 
    iglm_model, 
    chain_token, 
    cache_path=None
):
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            lookup_table = pickle.load(f)

    else:
        pool = multiprocessing.Pool(processes=16)
        pairs = list(tqdm(
            pool.imap(
                get_species_token,
                seqs, 
                chunksize=10
            ), total=len(seqs)
        ))
        pool.close()
        pool.join()

        pairs = [p for p in pairs if p is not None]

        lookup_table = {}
        for sequence, species_token in tqdm(pairs):
            if species_token is None:
                continue
            
            token_seq = [chain_token, species_token] + list(sequence)
            token_seq += [iglm_model.tokenizer.sep_token]

            token_seq = torch.Tensor([
                iglm_model.tokenizer.convert_tokens_to_ids(token_seq)
            ]).int().to(iglm_model.device)

            with torch.no_grad():
                out = iglm_model.model(
                    token_seq,
                    output_hidden_states=True
                )
            h = out.hidden_states[-1]
            h = h.mean((0,1)).cpu().numpy()
            
            lookup_table[sequence] = (species_token, h)

        with open(cache_path, "wb") as f:
            pickle.dump(lookup_table, f)

    return lookup_table

def path_to_dataloader(
    data_path, 
    scaler,
    model_save_path,
    labels, 
    iglm_model, 
    chain_token,
    batch_size=512,
    is_train=True,
):
    df = pd.read_csv(data_path)
    df["target"] = scaler.transform(df[labels])

    df_seq_name = {
        "[HEAVY]": "HeavyAA",
        "[LIGHT]": "LightAA",
    }
    seq_name = df_seq_name[chain_token]

    lookup_table = get_iglm_embeddings(
        df[seq_name].values, 
        iglm_model, 
        chain_token,
        cache_path=os.path.join(
            model_save_path, 
            f"embed_cache_{'train' if is_train else 'val'}.pkl"
        )
    )

    seq_arrays = df[[seq_name, "target"]].values

    filtered_labels, hiddens = [], []
    for sequence, label in tqdm(seq_arrays):
        if sequence not in lookup_table:
            continue
        
        filtered_labels.append(label)
        hiddens.append(lookup_table[sequence][1])

    labels = np.array(filtered_labels)
    hiddens = np.array(hiddens)

    loader = DataLoader(
        dataset=Dataset(hiddens, labels),
        batch_size=batch_size,
        shuffle=is_train,  # Shuffle only train loader
        num_workers=1,#multiprocessing.cpu_count() if multithread else 1,
        pin_memory=True,
    )

    return loader

def train_guidance_model(
    iglm_model,
    train_data_path, 
    val_data_path,
    labels,
    model_save_dir,
    chain_token,
):
    scaler = StandardScaler()
    train_df = pd.read_csv(train_data_path)
    val_df = pd.read_csv(val_data_path)

    targets = labels
    train_targets = train_df[targets]
    test_targets = val_df[targets]

    scaler = StandardScaler().fit(pd.concat([train_targets, test_targets]))

    paths = [train_data_path, val_data_path]
    train_loader, val_loader = [
        path_to_dataloader(
            data_path,
            scaler,
            model_save_dir,
            labels,
            iglm_model,
            chain_token,
            is_train=i == 0,
        ) for i, data_path in enumerate(paths)
    ]

    guidance_model = ClassificationHead().cuda()
    
    trainer = pl.Trainer(
        default_root_dir='.',
        max_epochs=200,
        check_val_every_n_epoch=1,
        accelerator="gpu",
        gpus=1,
        enable_progress_bar=True,
    )
    trainer.fit(
        model=guidance_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    model_save_path = os.path.join(model_save_dir, f"guidance_model.pt")
    torch.save(guidance_model.state_dict(), model_save_path)

def load_guidance_models(args):
    labels = args.labels.split(",")

    if not os.path.exists(args.guidance_model_dir):
        os.makedirs(args.guidance_model_dir)

    guidance_models = {}
    for k,v in CHAIN_TOKENS.items():
        guidance_model_dir = os.path.join(args.guidance_model_dir, k)
        if not os.path.exists(guidance_model_dir):
             os.makedirs(guidance_model_dir)

        model_path = os.path.join(guidance_model_dir, f"guidance_model.pt")

        if not os.path.exists(model_path):
            print(f"Training guidance model for {k}")

            train_guidance_model(
                IgLM(),
                args.train_data_path, 
                args.val_data_path,
                labels,
                guidance_model_dir,
                v,
            )
    
        guidance_model = ClassificationHead()
        guidance_model.load_state_dict(torch.load(model_path))
        guidance_model.eval()

        guidance_models[k] = guidance_model

    return guidance_models