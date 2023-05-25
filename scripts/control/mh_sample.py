import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt

from iglm import IgLM

def get_hidden(
    iglm_model,
    x,
    chain_token,
    species_token,
):
    h = []
    for sequence in x:
        sequence = list(sequence)
        token_seq = [chain_token, species_token] + sequence
        token_seq += [iglm_model.tokenizer.sep_token]

        token_seq = torch.Tensor([
            iglm_model.tokenizer.convert_tokens_to_ids(token_seq)
        ]).int().to(iglm_model.device)
    
        with torch.no_grad():
            out = iglm_model.model(
                token_seq,
                output_hidden_states=True
            )
        h.append(out.hidden_states[-1].mean(1))
        
    h = torch.cat(h, dim=0)
    return h

def energy_function(
    iglm_model, 
    guidance_model,
    x, 
    chain_token, 
    species_token,
    fitness_weight=0.1
):
    likelihoods = []
    for seq in x:
        likelihoods.append(
            iglm_model.log_likelihood(
                seq,
                chain_token=chain_token,
                species_token=species_token,
            )
        )
    likelihoods = np.array(likelihoods)
    
    fitness = guidance_model(
        get_hidden(
            iglm_model,
            x,
            chain_token,
            species_token,
        )
    ).squeeze(-1)
    fitness = fitness.detach().cpu().numpy()
        
    return likelihoods + fitness_weight * fitness

def mutate(x, vocab):
    for _ in range(1):
        pos = [np.random.randint(len(seq)) for seq in x]
        vals = [vocab[np.random.randint(len(vocab))] for _ in x]
        x_mut = []
        for seq, i, val in zip(x, pos, vals):
            x_mut.append(
                seq[:i] + val + seq[i+1:]
            )
        x = x_mut
    return x_mut

def mcmc_step(
    iglm_model, 
    guidance_model, 
    x, 
    vocab, 
    chain_token,
    species_token,
    T=1,
    fitness_weight=0.1
):
    x_mut = mutate(x, vocab)
    
    E = energy_function(
        iglm_model,
        guidance_model,
        x, 
        chain_token, 
        species_token,
        fitness_weight=fitness_weight
    )

    E_mut = energy_function(
        iglm_model,
        guidance_model,
        x_mut, 
        chain_token, 
        species_token,
        fitness_weight=fitness_weight
    )
    
    accept_p = np.minimum(1, np.exp(-(E - E_mut) / T))
    accept = np.random.rand(len(x)) < accept_p
    # print(np.mean(np.minimum(1-accept_p,accept_p)))
    
    x_new = [s_new if a else s for a, s_new, s in zip(accept, x_mut, x)]
    E_new = [e_new if a else e for a, e_new, e in zip(accept, E_mut, E)]    
        
    return x_new, E_new
    
def mcmc_sample(
    iglm_model,
    guidance_model,
    vocab, 
    chain_token,
    species_token,
    num_steps=2000, 
    fitness_weight=0.1
):
    x = iglm_model.generate(
        chain_token=chain_token,
        species_token=species_token,
        num_to_generate=10,
    )
    x = [''.join(np.random.choice(vocab, size=len(s))) for s in x]

    E_traj, x_traj = [], []
    for i in tqdm(list(range(num_steps))):
        new_x, E_new = mcmc_step(
            iglm_model, 
            guidance_model,
            x, 
            vocab,
            chain_token,
            species_token,
            T=1e-2 * (0.5) ** (i // 2000),#8-(i // 200), 
            fitness_weight=fitness_weight,
        )

        # for s, s_mut in zip(x, new_x):
        #     print(s == s_mut)

        x = new_x

        E_traj.append(E_new)
        x_traj.append(new_x)

    E_traj = np.array(E_traj)
    return x_traj, E_traj
    
def main():
    iglm_model = IgLM()
    vocab = [k for k in iglm_model.tokenizer.vocab.keys() if '[' not in k]
    guidance_model = None

    iglm_model.model.cuda()
    guidance_model.cuda()
        
    fitness_weight=0.1
    num_steps = 20000

    lv_traj, lv_E = mcmc_sample(
        iglm_model,
        vocab,
        "[LIGHT]", 
        "[HUMAN]",
        num_steps=num_steps, 
        fitness_weight=fitness_weight
    )
    plt.plot(-lv_E)
    plt.show()

    hv_traj, hv_E = mcmc_sample(
        iglm_model,
        vocab,
        "[HEAVY]", 
        "[HUMAN]",
        num_steps=num_steps, 
        fitness_weight=fitness_weight
    )
    plt.plot(-hv_E)
    plt.show()

    for i in range(10):
        lvs = [x[i] for x in lv_traj]
        e1s = [x[i] for x in lv_E]
        hvs = [x[i] for x in hv_traj]
        e2s = [x[i] for x in hv_E]
        dicts = []
        for lv, e1, hv, e2 in zip(lvs, e1s, hvs, e2s):
            dicts.append({
                'lv': lv,
                'e_lv': e1,
                'hv': hv,
                'e_hv': e2,
            })
        df = pd.DataFrame(dicts)
        df.to_csv(f"mcmc_i={i}_fitness={fitness_weight}.csv", index=False)


if __name__ == "__main__":
    main()