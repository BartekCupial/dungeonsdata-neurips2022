from random_words import RandomWords

from mrunner.helpers.specification_helper import create_experiments_helper, get_combinations


name = globals()["script"][:-3]

# params for all exps
config = {
    "exp_tags": [name],
    "connect":"0.0.0.0:4431",
    "exp_set": "2G",
    "exp_point": "monk-APPO-BC1-EWC",
    "num_actor_cpus": 20,
    "total_steps": 2_000_000_000,
    "group": "monk-APPO-BC1-EWC",
    "character": "mon-hum-neu-mal",
    "ttyrec_batch_size": 256,
    "use_ewc": True,
    "dataset": "bc1",
    "freeze_from_the_beginning": False,
    "use_checkpoint_actor": True,
    "model_checkpoint_path": "/checkpoint/checkpoint.tar",
    "unfreeze_actor_steps": 0,
}

# params different between exps
params_grid = [
    {
        "seed":  list(range(5)),
        "ewc_penalty_scaler": [1, 400, 2000],
        "ewc_n_batches": [10, 100]
    },
]

params_configurations = get_combinations(params_grid)

final_grid = []
for e, cfg in enumerate(params_configurations):
    cfg = {key: [value] for key, value in cfg.items()}
    r = RandomWords().random_word()
    cfg["group"] = [f"{name}_{e}_{r}"]
    final_grid.append(dict(cfg))


experiments_list = create_experiments_helper(
    experiment_name=name,
    project_name="nle",
    with_neptune=False,
    script="python3 mrunner_run.py",
    python_path=".",
    tags=[name],
    exclude=["checkpoint"],
    base_config=config,
    params_grid=final_grid,
    exclude_git_files=False,
)
