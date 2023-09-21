from random_word import RandomWords

from mrunner.helpers.specification_helper import (
    create_experiments_helper,
    get_combinations,
)

name = globals()["script"][:-3]

# params for all exps
config = {
    "exp_tags": [name],
    "connect": "0.0.0.0:4431",
    "exp_set": "2G",
    "exp_point": "monk-APPO-AMZN",
    "num_actor_cpus": 20,
    "total_steps": 2_000_000_000,
    "group": "monk-APPO-AMZN",
    "character": "mon-hum-neu-mal",
    "use_checkpoint_actor": True,
    "model_checkpoint_path": "/net/pr2/projects/plgrid/plgggmum_crl/bcupial/AMZN/checkpoint_v0",
    "use_resnet": True,
    "model": "NetHackNetTtyrec",
    "unfreeze_actor_steps": 50_000_000,
    "actor_batch_size": 64,
    "batch_size": 32,
    "virtual_batch_size": 32,
    "unroll_length": 80,
}

# params different between exps
params_grid = [
    {
        "seed": list(range(5)),
        "entropy_cost": [0.001],
        "baseline_cost": [0.5],
        "reward_clip": [False],
        "adam_learning_rate": [0.0002],
    },
    {
        "seed": list(range(5)),
        "adam_learning_rate": [0.001],
    },
]

params_configurations = get_combinations(params_grid)

final_grid = []
for e, cfg in enumerate(params_configurations):
    cfg = {key: [value] for key, value in cfg.items()}
    r = RandomWords().get_random_word()
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
)
