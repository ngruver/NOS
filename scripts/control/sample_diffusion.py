import os
import pandas as pd
import itertools
import torch
import hydra

from seq_models.sample import sample_outer_loop

@hydra.main(config_path="../../configs", config_name="sample")
def main(config):

    if config.ckpt_path is None:
        raise ValueError("Must specify a checkpoint path")
    
    if config.seeds_fn is None:
        raise ValueError("Must specify a seeds file")
    
    if config.results_dir is None:
        raise ValueError("Must specify a results directory")

    if config.guidance_kwargs is None:
        raise ValueError("Must specify guidance kwargs")

    if not os.path.exists(config.results_dir):
        os.makedirs(config.results_dir)

    model = hydra.utils.instantiate(config.model, _recursive_=False)
        
    if config.ckpt_path is not None:
        state_dict = torch.load(config.ckpt_path)['state_dict']
        result = model.load_state_dict(state_dict, strict=False)
        if len(result.missing_keys) > 0:
            raise ValueError(f"Missing keys: {result.missing_keys}")
        elif len(result.unexpected_keys) > 0:
            print(f"Unexpected keys: {result.unexpected_keys}")

    if torch.cuda.is_available():
        model.cuda()

    model.eval()

    model_tag = ''
    if 'mlm' in config.model['_target_']:
        model_tag += 'mlm'
    elif 'gaussian' in config.model['_target_']:
        model_tag += 'gaussian'

    numbering_schemes = ["aho"]
    cdr_combos = [
        ["hcdr1","hcdr2","hcdr3"],
    ]

    # guidance_options = {
    #     "step_size": [2.0, 1.0, 0.5, 0.1],
    #     "stability_coef": [0.1, 0.01, 0.001],
    #     "num_steps": [5, 10, 20, 40],
    #     "guidance_layer": ["last", "first"],
    #     "return_best": [True, False]
    # }
    guidance_options = {
        "step_size": [1.0, 0.5, 0.1],
        "stability_coef": [10.0, 1.0, 0.1, 0.01, 0.001],
        "num_steps": [5, 10],
        "guidance_layer": ["last", "first"],
        "return_best": [True, False]
    }
    combos = pd.DataFrame(
        list(itertools.product(*guidance_options.values())), 
        columns=guidance_options.keys()
    )
    for k in config.guidance_kwargs.keys():
        if k in guidance_options:
            continue
        combos[k] = config.guidance_kwargs[k]

    base_sampling_kwargs = {
        "fixed_length": True,
        "autoregressive_sample": False,
    }
    sampling_kwargs_list = []
    for guidance_options in combos.to_dict('records'):
        kwargs = base_sampling_kwargs.copy()
        kwargs.update({"guidance_kwargs": guidance_options})
        sampling_kwargs_list.append(kwargs)
        if 'mlm' in model_tag and not guidance_options['return_best']:
            kwargs = base_sampling_kwargs.copy()
            kwargs.update({
                "guidance_kwargs": guidance_options, 
                "autoregressive_sample": True
            })
            sampling_kwargs_list.append(kwargs)

    sample_outer_loop(
        model,
        model_tag,
        config.results_dir,
        config.seeds_fn,
        config.vocab_file,
        numbering_schemes,
        cdr_combos,
        sampling_kwargs_list,
    )

if __name__ == "__main__":
    main()