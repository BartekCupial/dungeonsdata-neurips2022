from pathlib import Path

from mrunner.helpers.specification_helper import create_experiments_helper

# take configuration name without .py extension
name = globals()["script"][:-3]

# params for all exps
config = {
    "exp_tags": [name],
    "connect":"0.0.0.0:4431",
    "exp_set": "2G",
    "exp_point": "monk-APPO",
    "num_actor_cpus": 20,
    "total_steps": 2_000_000_000,
    'group': "monk-APPO",
    "character": "mon-hum-neu-mal",
    "use_checkpoint_actor": False
}

# params different between exps
params_grid = {}
params_grid = [
    {
    "seed":[i],
    "group": [f"{name}_s{i}"],
    }   
for i in range(3)
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
    exclude_git_files=False,
)
