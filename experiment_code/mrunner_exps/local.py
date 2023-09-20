from pathlib import Path

from mrunner.helpers.specification_helper import create_experiments_helper


name = globals()["script"][:-3]

# params for all exps
config = {
    "exp_tags": [name],
    "connect": "0.0.0.0:4431",
    "exp_set": "2G",
    "exp_point": "monk-AA-LSTM-KLAA",
    "num_actor_cpus": 20,
    "total_steps": 2_000_000_000,
    "group": "monk-AA-LSTM-KLAA",
    "character": "mon-hum-neu-mal",
    "use_checkpoint_actor": True,
    "ttyrec_batch_size": 256,
    "kickstarting_loss_bc": 0.2,
    "use_kickstarting_bc": True,
    "kickstarting_path": "/home/bartek/Workspace/CW/forecast-model-and-code/model_115.tar",
    "model_checkpoint_path": "/home/bartek/Workspace/CW/forecast-model-and-code/model_115.tar",
    "dataset": "autoascend",
    "use_resnet": True,
    "model": "NetHackNetTtyrec",
    "unfreeze_actor_steps": 50_000_000,
    "actor_batch_size": 64,
    "batch_size": 32,
    "virtual_batch_size": 32,
    "ttyrec_batch_size": 128,
    "unroll_length": 80,
    "ttyrec_unroll_length": 80,
}


# params different between exps
params_grid = {
    "actor_batch_size": [16],
    "batch_size": [8],
    "virtual_batch_size": [8],
    "ttyrec_batch_size": [16],
    "dbfilename": ["/home/bartek/Workspace/data/nethack/AA-taster/ttyrecs.db"],
    "wandb": [True],
    "unfreeze_actor_steps": [5_000],
}

experiments_list = create_experiments_helper(
    experiment_name=name,
    project_name="nle",
    with_neptune=False,
    script="python3 mrunner_run.py",
    python_path=".",
    tags=[name],
    exclude=["checkpoint"],
    base_config=config,
    params_grid=params_grid,
)
