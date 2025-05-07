"""Varied and fixed hyperparameters used in run_hpsearch.py"""

hp_varied = {
    "batch_size": [10, 25, 100],
    "learning_rate": [1e-2, 1e-4, 1e-6],
    "nheads": [8, 16, 64],
    "dropout": [0.1, 0.3, 0.5]
}

hp_fixed = {
    "nepochs": 10,
    "emb_size": 128,
    "n_layers_encoder":3,
    "n_layers_decoder":3,
    "activation": "gelu",
    "ff_multiplier": 4,
    "lr_end_factor": 0.05,
    "lr_warmup": True
}
