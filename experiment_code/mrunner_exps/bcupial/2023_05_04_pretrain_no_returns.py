from pathlib import Path

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
    "exp_point": "monk-AA-DT",
    "num_actor_cpus": 20,
    "total_steps": 2_000_000_000,
    "actor_batch_size": 256,
    "batch_size": 128,
    "ttyrec_batch_size": 512,
    "supervised_loss": 1,
    "adam_learning_rate": 0.001,
    "behavioural_clone": True,
    "group": name,
    "character": "mon-hum-neu-mal",
    "model": "DecisionTransformer",
    "return_to_go": True,
    "use_timesteps": True,
    "use_returns": True,
    "use_timesteps": True,
    "score_target_value": 10000,
    "score_scale": 10000,
    "n_layer": 3,
    "n_head": 4,
    "grad_norm_clipping": 4,
    "hidden_dim": 512,
    "warmup_steps": 10000,
    "weight_decay": 0.04,
}


# params different between exps
params_grid = [
    {
        "seed": [0],
        "unroll_length": [64],
        "ttyrec_unroll_length": [64],
        "batch_size": [64],
        "actor_batch_size": [128],
        "ttyrec_batch_size": [256],
        "use_returns": [False],
    },
    {
        "adam_learning_rate": [0.0001, 0.0002, 0.0006],
        "seed": [0],
        "unroll_length": [128],
        "ttyrec_unroll_length": [128],
        "batch_size": [32],
        "actor_batch_size": [64],
        "ttyrec_batch_size": [128],
        "use_returns": [False],
    },
]

params_configurations = get_combinations(params_grid)

final_grid = []
for e, cfg in enumerate(params_configurations):
    cfg = {key: [value] for key, value in cfg.items()}
    cfg["group"] = [f"{name}_{e}"]
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
