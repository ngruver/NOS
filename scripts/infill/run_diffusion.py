import os
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

    numbering_schemes = ["chothia", "aho"]
    cdr_combos = [
        ["hcdr1"],
        ["hcdr2"],
        ["hcdr3"],
        ["hcdr1", "hcdr2", "hcdr3"],
        ["lcdr1"],
        ["lcdr2"],
        ["lcdr3"],
    ]

    sampling_kwargs_list = [
        {"fixed_length": True},
        {"fixed_length": False},
    ]

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