#ss_perc_sheet
# model.noise_schedule.noise_scale=2 \
PYTHONPATH="." python scripts/train_seq_model.py \
    model=gaussian \
    'target_cols=["ss_perc_sheet"]' \
    model.noise_schedule.noise_scale=5 \
    model.optimizer.lr=0.0002 \
    model.network.discr_stop_grad=False \
    discr_batch_ratio=4 \
    exp_name=gaussian_5_sheet_discr_joint \