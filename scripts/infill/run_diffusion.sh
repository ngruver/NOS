# PYTHONPATH="." python scripts/infill/run_diffusion.py \
#     model=mlm \
#     model.network.bert_config_name="prajjwal1/bert-small" \
#     ckpt_path="/scratch/nvg7279/logs/guided_protein_seq/cleaned_mlm/models/best_by_valid/epoch\=3-step\=2844.ckpt" \
#     +seeds_fn=/home/nvg7279/src/seq-struct/poas_seeds.csv \
#     +results_dir=/home/nvg7279/src/seq-struct/infill_new \

# PYTHONPATH="." python scripts/infill/run_diffusion.py \
#     model=gaussian \
#     ckpt_path="/scratch/nvg7279/logs/guided_protein_seq/ar_mlm_test/models/best_by_valid/epoch\=74-step\=53700.ckpt" \
#     +seeds_fn=/home/nvg7279/src/seq-struct/poas_seeds.csv \
#     +results_dir=/home/nvg7279/src/seq-struct/infill_new \

# PYTHONPATH="." python scripts/infill/run_diffusion.py \
#     model=mlm \
#     ckpt_path="/scratch/nvg7279/logs/guided_protein_seq/mlm_sheet_discr_joint/models/best_by_valid/epoch\=49-step\=35550.ckpt" \
#     +seeds_fn=/home/nvg7279/src/seq-struct/poas_seeds.csv \
#     +results_dir=/home/nvg7279/src/seq-struct/infill_fix_mask \

PYTHONPATH="." python scripts/infill/run_diffusion.py \
    model=gaussian \
    ckpt_path="/scratch/nvg7279/logs/guided_protein_seq/gaussian_sasa_discr_joint_2/models/best_by_valid/epoch\=47-step\=34127.ckpt" \
    +seeds_fn=/home/nvg7279/src/seq-struct/poas_seeds.csv \
    +results_dir=/home/nvg7279/src/seq-struct/infill_fix_mask \

