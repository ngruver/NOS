PYTHONPATH="." python scripts/control/pplm.py \
    --train_data_path="/scratch/nvg7279/datasets/igfold_labeled_cleaned/train.csv" \
    --val_data_path="/scratch/nvg7279/datasets/igfold_labeled_cleaned/val_iid.csv" \
    --labels="ss_perc_sheet" \
    --guidance_model_dir="/scratch/nvg7279/iglm_results/guidance_models/test_sheet" \
    --seeds_fn="/home/nvg7279/src/seq-struct/poas_seeds_small.csv" \
    --results_dir="/home/nvg7279/src/seq-struct/iglm_control_sheet"
