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
    "actor_batch_size": 256,
    "batch_size": 128,
    "ttyrec_batch_size": 512,
    "supervised_loss": 1,
    "adam_learning_rate": 0.001,
    "behavioural_clone": True,
    "group": "monk-AA-LSTM-KLAA",
    "character": "mon-hum-neu-mal",
    "use_checkpoint_actor": True,
    # "ttyrec_batch_size": 256,
    # "kickstarting_loss_bc": 0.2,
    # "use_kickstarting_bc": True,
    # "kickstarting_path": "/home/bartek/Workspace/data/nethack_checkpoints/monk-AA-BC/checkpoint.tar",
    "model_checkpoint_path": "/home/bartek/Workspace/data/sf_checkpoints/dungeons/monk-AA-BC/checkpoint_v2000000000",
    # "kickstarting_path": "/home/bartek/Workspace/data/nethack_checkpoints/AMZN/checkpoint_v0",
    # "model_checkpoint_path": "/home/bartek/Workspace/data/nethack_checkpoints/AMZN/checkpoint_v0",
    "dataset": "autoascend",
    "dataset_shuffle": False,
    "use_prev_action": False,
    "ttyrec_envpool_size": 1,
    "adam_learning_rate": 0.0,
    # "use_resnet": True,
    # "model": "NetHackNetTtyrec",
    # "unfreeze_actor_steps": 50_000_000,
    # "actor_batch_size": 64,
    # "batch_size": 32,
    # "virtual_batch_size": 32,
    # "ttyrec_batch_size": 128,
    # "unroll_length": 80,
    # "ttyrec_unroll_length": 80,
    # "supervised_loss": 0.5,
    # "use_prev_action": False,
}


# params different between exps
params_grid = {
    "actor_batch_size": [8],
    "batch_size": [4],
    "virtual_batch_size": [4],
    "ttyrec_batch_size": [2],
    "dbfilename": ["/home/bartek/Workspace/data/nethack/AA-taster/ttyrecs.db"],
    "wandb": [False],
    # "unfreeze_actor_steps": [5_000],
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
