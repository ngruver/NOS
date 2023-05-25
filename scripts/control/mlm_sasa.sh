PYTHONPATH="." ./python-greene scripts/control/sample_diffusion.py \
    model=mlm \
    model.network.target_channels=1 \
    ckpt_path="/scratch/nvg7279/logs/guided_protein_seq/mlm_sasa_discr_joint/models/best_by_valid/epoch\=53-step\=38394.ckpt" \
    +guidance_kwargs.step_size=1.0 \
    +guidance_kwargs.stability_coef=0.01 \
    +guidance_kwargs.num_steps=10 \
    +seeds_fn=/home/nvg7279/src/seq-struct/poas_seeds_small.csv \
    +results_dir=/home/nvg7279/src/seq-struct/mlm_control_sasa \
