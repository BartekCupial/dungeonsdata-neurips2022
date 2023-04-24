from pathlib import Path
from itertools import product
from mrunner.helpers.specification_helper import create_experiments_helper


name = globals()["script"][:-3]

# params for all exps
config = {
    "exp_tags": [name],
    "connect": "0.0.0.0:4431",
    "exp_set": "2G",
    "exp_point": "@-NAO-BC",
    "num_actor_cpus": 20,
    "total_steps": 2_000_000_000,
    "actor_batch_size": 128,
    "batch_size": 64,
    "ttyrec_batch_size": 256,
    "supervised_loss": 1,
    "adam_learning_rate": 0.001,
    "behavioural_clone": True,
    "group": name,
    "dataset": "altorg",
    "dataset_bootstrap_actions": True,
    "bootstrap_pred_max": True,
    "dataset_bootstrap_path": "/checkpoint/hackrl/nle/@-AA-IV/checkpoint.tar",
}


# params different between exps
params_grid = [
    {
        "dataset_demigod": [demigod],
        "dataset_highscore": [highscore],
        "dataset_midscore": [midscore],
        "group": [f"{name}_{i}"],
    }
    for i, (demigod, highscore, midscore) in enumerate(
        zip(
            [True, False, False, False],
            [False, True, False, False],
            [False, False, True, False],
        )
    )
]

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
