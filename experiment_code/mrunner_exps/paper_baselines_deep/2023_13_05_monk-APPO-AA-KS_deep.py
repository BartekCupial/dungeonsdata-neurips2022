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
    "exp_point": "monk-APPO-AA-KS",
    "num_actor_cpus": 20,
    "total_steps": 2_000_000_000,
    "group": "monk-APPO-AA-KS",
    "character": "mon-hum-neu-mal",
    "use_checkpoint_actor": False,
    "kickstarting_loss": 0.1,
    "use_kickstarting": True,
    "kickstarting_path": "/net/tscratch/people/plgbartekcupial/mrunner_scratch/nle/10_05-09_22-awesome_heisenberg/monk-aa-bc-deep_hp0i_0/checkpoint/hackrl/nle/monk-AA-BC_deep_0/checkpoint.tar",
}


# params different between exps
params_grid = [
    {
        "seed": list(range(5)),
        "kickstarting_loss": [0.5],
        # log forgetting
        "log_forgetting": [True],
        "forgetting_dataset": ["bc_deep"],
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
