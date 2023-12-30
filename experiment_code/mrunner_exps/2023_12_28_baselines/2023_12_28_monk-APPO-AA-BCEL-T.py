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
    "exp_point": "monk-APPO-AA-BCEL-T",
    "num_actor_cpus": 20,
    "total_steps": 2_000_000_000,
    "group": "monk-APPO-AA-BCEL-T",
    "character": "mon-hum-neu-mal",
    "use_checkpoint_actor": False,
    "ttyrec_batch_size": 256,
    "supervised_loss": 0.1,
    "model": "NetHackNetTtyrec",
    "sampling_type": "softmax",
    "h_dim": 512,
}


# params different between exps
params_grid = [
    {
        "seed": list(range(5)),
        "supervised_loss": [0.5],
        # load from checkpoint
        "unfreeze_actor_steps": [50_000_000],
        "use_checkpoint_actor": [True],
        "model_checkpoint_path": [
            "/net/pr2/projects/plgrid/plgggmum_crl/bcupial/mrunner_scratch/nle/28_12-09_48-zealous_mestorf/2023-12-18-monk-aa-bc-amzn_cf4e_0/checkpoint/hackrl/nle/2023_12_18_monk-AA-BC_amzn_0/checkpoint.tar"
        ],
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
