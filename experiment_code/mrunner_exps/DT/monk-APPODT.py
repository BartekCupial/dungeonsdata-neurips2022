from pathlib import Path

from mrunner.helpers.specification_helper import create_experiments_helper


name = globals()["script"][:-3]

# params for all exps
config = {
    "exp_tags": [name],
    "connect":"0.0.0.0:4431",
    "exp_set": "2G",
    "exp_point": "monk-APPODT",
    "num_actor_cpus": 20,
    "total_steps": 2_000_000_000,
    'group': "monk-APPODT",
    "character": "mon-hum-neu-mal",
    "model": "DecisionTransformer",
    "return_to_go": True,
    "use_timesteps": True,
    "use_returns": True,
    "use_timesteps": True,
    "score_target_value": 10000,
    "score_scale": 10000,
    "n_layer": 6,
    "n_head": 8,
    "grad_norm_clipping": 4,
    "hidden_dim": 512,
    "warmup_steps": 10000,
    "weight_decay": 0.01,
}


# params different between exps
params_grid = [
    {
        "seed": [seed],
        "group": [f"{name}_{i}"],
    } for i, seed in enumerate([0, 1, 2]) 
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
